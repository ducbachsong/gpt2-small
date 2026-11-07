// test_block.cu — tests for llmc/block.cuh (BlockParams, block_init_params, block_forward, block_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks a block against the same block written in PyTorch, with autograd for the backward, in float
// and in bfloat16; part 2 checks what can be worked out by hand. The timing is in tunes/tune_block.cu.
#include <torch/types.h>              // torch::layer_norm, torch::linear, scaled_dot_product_attention; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/block.cuh"

// A block is ten kernels in a row, each within 1e-5 of PyTorch's; the differences add up through them.
static const double TOLERANCE = 1e-4;
// In bfloat16, against PyTorch in float on the same values: each kernel rounds its outputs once (up to 2^-8,
// 0.4%), and the next kernel reads those rounded values, ten times in a row.
static const double BFLOAT16_TOLERANCE = 3e-2;

#define BLOCK_PARAMS 12               // a block's parameter tensors: 4 matrices, then 8 vectors
#define GPT2_LAYERS 12                // GPT-2 small's blocks
#define GPT2_EMBED_DIM 768            // its row width
#define GPT2_NUM_HEADS 12             // its heads

static const char *PARAM_NAMES[BLOCK_PARAMS] = {
    "qkv_weight", "attention_out_weight", "mlp_up_weight", "mlp_down_weight",
    "ln1_weight", "ln1_bias",             "qkv_bias",      "attention_out_bias",
    "ln2_weight", "ln2_bias",             "mlp_up_bias",   "mlp_down_bias"};

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

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward of a block of these sizes and dtype on random values, mine and PyTorch's,
/// and checks output, input_grad and the 12 parameter gradients within `tolerance_val`.
static void forward_and_backward_match_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val,
                                               size_t embed_dim_val, size_t num_heads_val, DType dtype_val,
                                               double tolerance_val)
{
    // 1. random values; mine, then PyTorch's
    torch::manual_seed(0);
    BlockTensors mine_val(batch_size_val, seq_len_val, embed_dim_val, num_heads_val, dtype_val);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    PytorchBlock pytorch_val(mine_val);

    // 2. print the sizes, then check output, input_grad and every parameter's gradient
    print_test_name(test_ptr);
    printf("    {%zu, %zu, %zu}, %zu heads, %s\n", batch_size_val, seq_len_val, embed_dim_val, num_heads_val,
           dtype_val == DType::BFloat16 ? "bfloat16" : "float");
    ASSERT_NEAR(list_of(mine_val.output_tensor), list_of(pytorch_val.output_tensor), tolerance_val);
    ASSERT_NEAR(list_of(mine_val.input_grad_tensor), list_of(pytorch_val.input_tensor.grad()), tolerance_val);
    std::vector<Tensor> grads_list = mine_val.grads_val.tensors();
    for (int param_index_val = 0; param_index_val < BLOCK_PARAMS; param_index_val++)
    {
        printf("    %s's gradient:\n", PARAM_NAMES[param_index_val]);
        ASSERT_NEAR(list_of(grads_list[param_index_val]), list_of(pytorch_val.params_list[param_index_val].grad()),
                    tolerance_val);
    }
}

static void matches_pytorch_on_gpt2_widths(void)
{
    // Expect: 2 rows of 64 tokens, GPT-2 small's widths (768, 12 heads of 64, MLP 3072), float: the output,
    // input_grad and all 12 parameter gradients equal PyTorch's within 1e-4.
    forward_and_backward_match_pytorch(__func__, 2, 64, GPT2_EMBED_DIM, GPT2_NUM_HEADS, DType::Float32, TOLERANCE);
}

static void matches_pytorch_with_small_odd_sizes(void)
{
    // Expect: {3, 37, 64} with 4 heads of 16: a seq_len no warp or square divides, and rows of 64, so
    // every kernel's part-full edges are crossed. The same as PyTorch within 1e-4.
    forward_and_backward_match_pytorch(__func__, 3, 37, 64, 4, DType::Float32, TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_widths's block with every tensor bfloat16 (mean and rstd float).
    // PyTorch is given the same bfloat16 values as floats and works in float: within BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, 2, 64, GPT2_EMBED_DIM, GPT2_NUM_HEADS, DType::BFloat16,
                                       BFLOAT16_TOLERANCE);
}

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void shapes_are_gpt2s_with_the_matrices_first(void)
{
    // Expect: for embed_dim 768, the 4 matrices {2304, 768}, {768, 768}, {3072, 768}, {768, 3072}, then the 8
    // vectors {768}, {768}, {2304}, {768}, {768}, {768}, {3072}, {768}. The matrices hold 3 + 1 + 4 + 4 = 12
    // times 768² = 7,077,888 values and the vectors 1 + 1 + 3 + 1 + 1 + 1 + 4 + 1 = 13 times 768 = 9,984, so
    // a block has 7,087,872, and GPT-2 small's 12 blocks 85,054,464.
    // 1. the shapes
    std::vector<Shape> shapes_list = BlockParams::shapes(GPT2_EMBED_DIM);
    std::vector<Shape> expected_list = {{2304, 768}, {768, 768}, {3072, 768}, {768, 3072}, {768}, {768},
                                        {2304},      {768},      {768},       {768},       {3072}, {768}};
    ASSERT_EQ(shapes_list.size(), (size_t)BLOCK_PARAMS);
    // 2. each shape, and the values the matrices and the vectors hold
    size_t matrix_values_val = 0;
    size_t vector_values_val = 0;
    for (size_t param_index_val = 0; param_index_val < BLOCK_PARAMS; param_index_val++)
    {
        ASSERT_EQ(shapes_list[param_index_val], expected_list[param_index_val]);
        size_t numel_val = 1;
        for (size_t size_val : shapes_list[param_index_val])
            numel_val *= size_val;
        (shapes_list[param_index_val].size() == 2 ? matrix_values_val : vector_values_val) += numel_val;
    }
    ASSERT_EQ(matrix_values_val, (size_t)7077888);
    ASSERT_EQ(vector_values_val, (size_t)9984);
}

static void init_gives_gpt2s_starting_values(void)
{
    // Expect: block_init_params for a model of 12 layers: the LayerNorm weights all 1, the 6 biases all 0;
    // qkv_weight and mlp_up_weight mean 0 and std 0.02; the two projections std 0.02 / sqrt(24) = 0.0040825. Each
    // std within 2% (a matrix holds 589,824 or more values, so by chance about 0.1%), each mean within 0.0005.
    // 1. init
    Tensor::manual_seed(0);
    BlockTensors mine_val(1, 4, GPT2_EMBED_DIM, GPT2_NUM_HEADS);
    block_init_params(mine_val.params_val, GPT2_LAYERS);
    // 2. the LayerNorm weights 1, the biases 0
    ASSERT_EQ(list_of(mine_val.params_val.ln1_weight), std::vector<float>(GPT2_EMBED_DIM, 1.0f));
    ASSERT_EQ(list_of(mine_val.params_val.ln2_weight), std::vector<float>(GPT2_EMBED_DIM, 1.0f));
    for (const Tensor &bias_tensor : {mine_val.params_val.ln1_bias, mine_val.params_val.qkv_bias,
                                      mine_val.params_val.attention_out_bias, mine_val.params_val.ln2_bias,
                                      mine_val.params_val.mlp_up_bias, mine_val.params_val.mlp_down_bias})
        ASSERT_EQ(list_of(bias_tensor), std::vector<float>(bias_tensor.numel(), 0.0f));
    // 3. each matrix's mean and std
    float projection_std_val = 0.02f / sqrtf(2.0f * GPT2_LAYERS);
    const Tensor matrices_list[] = {mine_val.params_val.qkv_weight, mine_val.params_val.attention_out_weight,
                                    mine_val.params_val.mlp_up_weight, mine_val.params_val.mlp_down_weight};
    float stds_list[] = {0.02f, projection_std_val, 0.02f, projection_std_val};
    for (int matrix_index_val = 0; matrix_index_val < 4; matrix_index_val++)
    {
        std::vector<float> values_list = list_of(matrices_list[matrix_index_val]);
        double sum_val = 0.0;
        double sum_squares_val = 0.0;
        for (float value_val : values_list)
            sum_val += value_val;
        double mean_val = sum_val / values_list.size();
        for (float value_val : values_list)
            sum_squares_val += (value_val - mean_val) * (value_val - mean_val);
        ASSERT_NEAR(mean_val, 0.0, 0.0005);
        ASSERT_NEAR(sqrt(sum_squares_val / values_list.size()), stds_list[matrix_index_val], 0.02 * stds_list[matrix_index_val]);
    }
}

static void zero_projections_pass_the_input_through(void)
{
    // Expect: random parameters, but the two projections' weights and biases 0. Both branches then give 0,
    // so output = input + 0 + 0 = input, to the last bit. Backward: the branches' input gradients are
    // output_grad times a 0 weight, 0, so nothing reaches the LayerNorms, the attention, the qkv or MLP up linear:
    // input_grad = output_grad to the last bit, and the gradients of ln1, qkv, ln2 and mlp_up are all exactly 0.
    // Only the projections learn: each bias's gradient is output_grad summed over the 2 x 64 rows, the same
    // for both, as both see output_grad.
    // 1. random values, the projections 0; a forward and a backward
    torch::manual_seed(0);
    BlockTensors mine_val(2, 64, GPT2_EMBED_DIM, GPT2_NUM_HEADS);
    mine_val.fill_random_values();
    for (const Tensor &projection_tensor : {mine_val.params_val.attention_out_weight, mine_val.params_val.attention_out_bias,
                                            mine_val.params_val.mlp_down_weight, mine_val.params_val.mlp_down_bias})
        projection_tensor.zero_();
    mine_val.forward();
    mine_val.backward();
    // 2. output is input, input_grad is output_grad
    ASSERT_EQ(list_of(mine_val.output_tensor), list_of(mine_val.input_tensor));
    ASSERT_EQ(list_of(mine_val.input_grad_tensor), list_of(mine_val.output_grad_tensor));
    // 3. nothing before the projections learns
    for (const Tensor &grad_tensor : {mine_val.grads_val.ln1_weight, mine_val.grads_val.ln1_bias, mine_val.grads_val.qkv_weight,
                                      mine_val.grads_val.qkv_bias, mine_val.grads_val.ln2_weight, mine_val.grads_val.ln2_bias,
                                      mine_val.grads_val.mlp_up_weight, mine_val.grads_val.mlp_up_bias})
        ASSERT_EQ(list_of(grad_tensor), std::vector<float>(grad_tensor.numel(), 0.0f));
    // 4. both projection biases' gradients are output_grad's column sums
    std::vector<float> column_sums_list = list_of(pytorch_copy_of(mine_val.output_grad_tensor).sum(torch::IntArrayRef{0, 1}));
    ASSERT_NEAR(list_of(mine_val.grads_val.mlp_down_bias), column_sums_list, 1e-5);
    ASSERT_EQ(list_of(mine_val.grads_val.attention_out_bias), list_of(mine_val.grads_val.mlp_down_bias));
}

static void backward_adds_onto_the_param_grads_and_writes_input_grad(void)
{
    // Expect: two backwards in a row, the gradients not set back to 0 in between: each of the 12 parameter
    // gradients is twice one backward's, within 1e-6 (LayerNorm adds its weight_grad and bias_grad with
    // atomics, in whatever order); input_grad is written over, so the same both times, to the last bit.
    // 1. a forward and one backward; its gradients kept
    torch::manual_seed(0);
    BlockTensors mine_val(2, 64, GPT2_EMBED_DIM, GPT2_NUM_HEADS);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    std::vector<std::vector<float>> once_list;
    for (const Tensor &grad_tensor : mine_val.grads_val.tensors())
        once_list.push_back(list_of(grad_tensor));
    std::vector<float> input_grad_once_list = list_of(mine_val.input_grad_tensor);
    // 2. a second backward: twice the parameter gradients, the same input_grad
    mine_val.backward();
    std::vector<Tensor> grads_list = mine_val.grads_val.tensors();
    for (int param_index_val = 0; param_index_val < BLOCK_PARAMS; param_index_val++)
    {
        std::vector<float> twice_once_list = once_list[param_index_val];
        for (float &value_val : twice_once_list)
            value_val *= 2.0f;
        printf("    %s's gradient:\n", PARAM_NAMES[param_index_val]);
        ASSERT_NEAR(list_of(grads_list[param_index_val]), twice_once_list, 1e-6);
    }
    ASSERT_EQ(list_of(mine_val.input_grad_tensor), input_grad_once_list);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_widths();
    matches_pytorch_with_small_odd_sizes();
    forward_and_backward_in_bfloat16();

    shapes_are_gpt2s_with_the_matrices_first();
    init_gives_gpt2s_starting_values();
    zero_projections_pass_the_input_through();
    backward_adds_onto_the_param_grads_and_writes_input_grad();

    printf("\nall tests passed\n");
    return 0;
}
