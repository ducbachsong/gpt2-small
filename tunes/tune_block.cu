// tune_block.cu — the timings of tests/test_block.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size.
//
// Built as tests/test_block.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/block_<date>_<time>.txt.
#include <torch/types.h>              // torch::layer_norm, torch::linear, scaled_dot_product_attention; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/block.cuh"

// In bfloat16, against PyTorch in float on the same values: each kernel rounds its outputs once (up to 2^-8,
// 0.4%), and the next kernel reads those rounded values, ten times in a row.
static const double BFLOAT16_TOLERANCE = 3e-2;

#define BLOCK_PARAMS 12               // a block's parameter tensors: 4 matrices, then 8 vectors
#define GPT2_EMBED_DIM 768            // its row width
#define GPT2_NUM_HEADS 12             // its heads

// ── helpers ───────────────────────────────────────────────────────────────────

/// Fills my tensor (float or bfloat16) with random values, std `scale_val` around `shift_val`, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's sizes.
static void fill_random(const Tensor &tensor, float scale_val, float shift_val = 0.0f)
{
    torch::Tensor values_tensor =
        torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * scale_val + shift_val;
    Tensor floats_tensor = Tensor::empty(tensor.shape());
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    tensor.copy_(floats_tensor);                // rounded, if tensor is bfloat16
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values; bfloat16 is turned into float
/// first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);  // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Gives every value of my tensor as a float list on the CPU, whatever its dtype.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Everything one block needs, for one size and dtype: its parameters and their gradients in a TensorBuffer
/// each, as the optimizer keeps them, its activations and scratch, and input, output and their gradients.
/// All 0 to start.
struct BlockTensors
{
    size_t batch_size_val;
    size_t seq_len_val;
    size_t embed_dim_val;
    size_t num_heads_val;
    TensorBuffer param_buffer_val;
    TensorBuffer grad_buffer_val;
    BlockParams params_val;
    BlockParams grads_val;
    BlockActivations activations_val;
    BlockScratch scratch_val;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor output_grad_tensor;
    Tensor input_grad_tensor;

    BlockTensors(size_t block_batch_size_val, size_t block_seq_len_val, size_t block_embed_dim_val,
                 size_t block_num_heads_val, DType dtype_val = DType::Float32)
        : batch_size_val(block_batch_size_val), seq_len_val(block_seq_len_val), embed_dim_val(block_embed_dim_val),
          num_heads_val(block_num_heads_val),
          activations_val(block_batch_size_val, block_seq_len_val, block_embed_dim_val, block_num_heads_val, dtype_val),
          scratch_val(block_batch_size_val, block_seq_len_val, block_embed_dim_val, block_num_heads_val, dtype_val)
    {
        // 1. the parameters and their gradients, in a TensorBuffer each
        int group_val = param_buffer_val.add_many(BlockParams::shapes(embed_dim_val));
        param_buffer_val.allocate(dtype_val);
        grad_buffer_val = TensorBuffer::zeros_like(param_buffer_val);
        std::vector<Tensor> param_tensors_list;
        std::vector<Tensor> grad_tensors_list;
        for (int param_index_val = 0; param_index_val < BLOCK_PARAMS; param_index_val++)
        {
            param_tensors_list.push_back(param_buffer_val.tensor(group_val, param_index_val));
            grad_tensors_list.push_back(grad_buffer_val.tensor(group_val, param_index_val));
        }
        params_val = BlockParams(param_tensors_list);
        grads_val = BlockParams(grad_tensors_list);
        // 2. input, output and their gradients
        Shape rows_list = {batch_size_val, seq_len_val, embed_dim_val};
        input_tensor = Tensor::zeros(rows_list, dtype_val), output_tensor = Tensor::zeros(rows_list, dtype_val);
        output_grad_tensor = Tensor::zeros(rows_list, dtype_val), input_grad_tensor = Tensor::zeros(rows_list, dtype_val);
    }

    /// Fills the parameters, input and output_grad with random values. Not GPT-2's starting values: there
    /// the biases are 0 and the projections tiny, which would hide a mistake in them. Each matrix gets std
    /// 1 / sqrt(in_features), so every linear's output has std about 1; the biases std 0.1; the LayerNorm
    /// weights 1 plus std 0.1.
    void fill_random_values()
    {
        fill_random(params_val.qkv_weight, 1.0f / sqrtf((float)embed_dim_val));
        fill_random(params_val.attention_out_weight, 1.0f / sqrtf((float)embed_dim_val));
        fill_random(params_val.mlp_up_weight, 1.0f / sqrtf((float)embed_dim_val));
        fill_random(params_val.mlp_down_weight, 1.0f / sqrtf(4.0f * embed_dim_val));
        for (const Tensor &bias_tensor : {params_val.ln1_bias, params_val.qkv_bias, params_val.attention_out_bias,
                                          params_val.ln2_bias, params_val.mlp_up_bias, params_val.mlp_down_bias})
            fill_random(bias_tensor, 0.1f);
        fill_random(params_val.ln1_weight, 0.1f, 1.0f);
        fill_random(params_val.ln2_weight, 0.1f, 1.0f);
        fill_random(input_tensor, 1.0f);
        fill_random(output_grad_tensor, 1.0f);
    }

    /// Runs block_forward on these tensors.
    void forward() { block_forward(output_tensor, input_tensor, params_val, activations_val, scratch_val); }

    /// Runs block_backward on these tensors.
    void backward()
    {
        block_backward(input_grad_tensor, grads_val, output_grad_tensor, input_tensor, params_val, activations_val,
                       scratch_val);
    }
};

/// Runs the same block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and
/// the 12 parameters in BlockParams::shapes()' order, gives the output.
static torch::Tensor pytorch_block_forward(const torch::Tensor &input_tensor, const std::vector<torch::Tensor> &p_list,
                                           int64_t num_heads_val)
{
    // 1. the sizes
    int64_t batch_size_val = input_tensor.size(0);
    int64_t seq_len_val = input_tensor.size(1);
    int64_t embed_dim_val = input_tensor.size(2);
    int64_t head_size_val = embed_dim_val / num_heads_val;
    std::vector<int64_t> normalized_shape_list = {embed_dim_val};
    // p: 0 qkv_weight, 1 attention_out_weight, 2 mlp_up_weight, 3 mlp_down_weight, 4 ln1_weight, 5 ln1_bias,
    //    6 qkv_bias, 7 attention_out_bias, 8 ln2_weight, 9 ln2_bias, 10 mlp_up_bias, 11 mlp_down_bias
    // 2. ln1, qkv, the causal attention, its projection, the residual add
    torch::Tensor qkv_tensor =
        torch::linear(torch::layer_norm(input_tensor, normalized_shape_list, p_list[4], p_list[5], 1e-5), p_list[0], p_list[6]);
    std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
    auto heads_val = [&](const torch::Tensor &part_tensor) {
        return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, head_size_val}).transpose(1, 2);
    };
    torch::Tensor y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]),
                                                                 heads_val(parts_list[2]), {}, 0.0, true);
    y_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
    torch::Tensor residual2_tensor = input_tensor + torch::linear(y_tensor, p_list[1], p_list[7]);
    // 3. ln2, the MLP, the residual add
    torch::Tensor ln2_output_tensor = torch::layer_norm(residual2_tensor, normalized_shape_list, p_list[8], p_list[9], 1e-5);
    torch::Tensor mlp_up_tensor = torch::linear(ln2_output_tensor, p_list[2], p_list[10]);
    return residual2_tensor + torch::linear(torch::gelu(mlp_up_tensor, "tanh"), p_list[3], p_list[11]);
}

/// PyTorch's answers for my block's input, parameters and output_grad: output, input.grad() and each
/// parameter's .grad(), with autograd.
struct PytorchBlock
{
    torch::Tensor input_tensor;
    torch::Tensor output_tensor;
    std::vector<torch::Tensor> params_list;

    PytorchBlock(const BlockTensors &mine_val)
    {
        input_tensor = pytorch_copy_of(mine_val.input_tensor).requires_grad_(true);
        for (const Tensor &param_tensor : mine_val.params_val.tensors())
            params_list.push_back(pytorch_copy_of(param_tensor).requires_grad_(true));
        output_tensor = pytorch_block_forward(input_tensor, params_list, (int64_t)mine_val.num_heads_val);
        output_tensor.backward(pytorch_copy_of(mine_val.output_grad_tensor));
    }
};

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of one block on GPT-2 small's size, {4, 1024, 768}
    // with 12 heads, in bfloat16, the dtype training runs in: mine against PyTorch's block with autograd,
    // also in bfloat16, on the same values. Steps 1 and 2 do first-time work (loading kernels, cuBLAS
    // picking its own), so the averages are of 3..20. Each side starts every backward from gradients of 0,
    // as a training step does. A T4 has no bfloat16 tensor cores (they came with the A100), so neither
    // side's products get them.
    //
    // Expect: at the end, my output and input_grad still match PyTorch's block in float on the same
    // bfloat16 values within BFLOAT16_TOLERANCE, as in forward_and_backward_in_bfloat16 at this size.

    // 1. random values, mine and PyTorch's in bfloat16
    torch::manual_seed(0);
    BlockTensors mine_val(4, 1024, GPT2_EMBED_DIM, GPT2_NUM_HEADS, DType::BFloat16);
    mine_val.fill_random_values();
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(mine_val.input_tensor).to(torch::kBFloat16).requires_grad_(true);
    std::vector<torch::Tensor> pytorch_params_list;
    for (const Tensor &param_tensor : mine_val.params_val.tensors())
        pytorch_params_list.push_back(pytorch_copy_of(param_tensor).to(torch::kBFloat16).requires_grad_(true));
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(mine_val.output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps, each backward from gradients of 0
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { mine_val.forward(); });
        mine_val.grad_buffer_val.flat().zero_();
        times_list[1] = gpu_milliseconds([&] { mine_val.backward(); });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_output_tensor = pytorch_block_forward(pytorch_input_tensor, pytorch_params_list, GPT2_NUM_HEADS);
        });
        pytorch_input_tensor.mutable_grad() = torch::Tensor();
        for (torch::Tensor &param_tensor : pytorch_params_list)
            param_tensor.mutable_grad() = torch::Tensor();
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. still PyTorch's in float, within BFLOAT16_TOLERANCE
    PytorchBlock pytorch_in_float_val(mine_val);         // once, not timed: the float answer to check against
    ASSERT_NEAR(list_of(mine_val.output_tensor), list_of(pytorch_in_float_val.output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(list_of(mine_val.input_grad_tensor), list_of(pytorch_in_float_val.input_tensor.grad()), BFLOAT16_TOLERANCE);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();

    printf("\nall tunes done\n");
    return 0;
}
