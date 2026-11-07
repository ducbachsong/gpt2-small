// tune_gpt2.cu — the timings of tests/test_gpt2.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size; and find_the_micro_batch_with_the_best_mfu, the --batch-size to train with.
//
// Built as tests/test_gpt2.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/gpt2_<date>_<time>.txt.
#include <algorithm>                  // std::min
#include <string>
#include <torch/types.h>              // torch::embedding, torch::linear, cross_entropy_loss, ...; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed, torch::NoGradGuard
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/gpt2.cuh"

// In bfloat16, against PyTorch in float on the same values: every kernel rounds its outputs once (up to 2^-8,
// 0.4%), and the next one reads them. The block's test allows 3e-2 for ten kernels; here are 25.
static const double BFLOAT16_TOLERANCE = 5e-2;

#define BLOCK_PARAMS 12               // a block's parameter tensors: 4 matrices, then 8 vectors
#define BLOCK_MATRICES 4              // of them matrices
#define SEQ_LEN 1024                  // tokens per row, as train_gpt2.cu trains
#define LARGEST_MICRO_BATCH 512       // rows: the micro-batch sweep tries 1, 2, 4, ... this many, while they fit
#define MOST_ROWS_STEP 8              // then the most rows that fit, rounded down to a multiple of this
#define TOKENS_PER_STEP 524288        // train_gpt2.cu's step, all GPUs together: 2^19
#define MEMORY_HEADROOM 0.9           // the next size is tried only if it needs at most this much of the free memory

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

/// Fills my int tensor with random token ids 0 .. vocab_size - 1, made by PyTorch on the GPU.
static void fill_random_ids(const Tensor &ids_tensor, size_t vocab_size_val)
{
    torch::Tensor values_tensor = torch::randint((int64_t)vocab_size_val, {(int64_t)ids_tensor.numel()},
                                                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    cudaCheck(cudaMemcpy(ids_tensor.data_ptr(), values_tensor.data_ptr<int>(), ids_tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
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

/// Makes a PyTorch tensor of int64 ids on the GPU from my int tensor: embedding and cross_entropy want int64.
static torch::Tensor pytorch_ids_of(const Tensor &ids_tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(ids_tensor.shape().begin(), ids_tensor.shape().end());
    return torch::from_blob(ids_tensor.data_ptr(), sizes_list, on_gpu_val).to(torch::kLong);
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().to(torch::kFloat).contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Gives every value of my tensor as a float list on the CPU, whatever its dtype.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Everything one model needs, for one config, size and dtype: its parameters and their gradients in a
/// TensorBuffer each, as the optimizer keeps them, its activations and scratch, and token_ids and targets.
/// All 0 to start.
struct GPT2Tensors
{
    GPT2Config config_val;
    size_t batch_size_val;
    size_t seq_len_val;
    TensorBuffer param_buffer_val;
    TensorBuffer grad_buffer_val;
    GPT2Params params_val;
    GPT2Params grads_val;
    GPT2Activations activations_val;
    GPT2Scratch scratch_val;
    Tensor token_ids_tensor;
    Tensor targets_tensor;

    GPT2Tensors(const GPT2Config &model_config_val, size_t model_batch_size_val, size_t model_seq_len_val,
                DType dtype_val = DType::Float32)
        : config_val(model_config_val), batch_size_val(model_batch_size_val), seq_len_val(model_seq_len_val),
          activations_val(model_batch_size_val, model_seq_len_val, model_config_val, dtype_val),
          scratch_val(model_batch_size_val, model_seq_len_val, model_config_val, dtype_val)
    {
        // 1. the parameters and their gradients, in a TensorBuffer each
        int group_val = param_buffer_val.add_many(GPT2Params::shapes(config_val));
        param_buffer_val.allocate(dtype_val);
        grad_buffer_val = TensorBuffer::zeros_like(param_buffer_val);
        std::vector<Tensor> param_tensors_list;
        std::vector<Tensor> grad_tensors_list;
        for (int param_index_val = 0; param_index_val < param_buffer_val.group_size(group_val); param_index_val++)
        {
            param_tensors_list.push_back(param_buffer_val.tensor(group_val, param_index_val));
            grad_tensors_list.push_back(grad_buffer_val.tensor(group_val, param_index_val));
        }
        params_val = GPT2Params(param_tensors_list, config_val);
        grads_val = GPT2Params(grad_tensors_list, config_val);
        // 2. token_ids and targets
        token_ids_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32);
        targets_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32);
    }

    /// Fills the parameters, token_ids and targets with random values. Not GPT-2's starting values: there
    /// the biases are 0 and the projections tiny, which would hide a mistake in them. wte and wpe get std
    /// 1 / sqrt(embed_dim), so the head's logits have std about 1; each block's matrices std
    /// 1 / sqrt(in_features); the biases std 0.1; the LayerNorm weights 1 plus std 0.1.
    void fill_random_values()
    {
        float embed_std_val = 1.0f / sqrtf((float)config_val.embed_dim);
        fill_random(params_val.wte, embed_std_val);
        fill_random(params_val.wpe, embed_std_val);
        for (const BlockParams &block_val : params_val.blocks)
        {
            fill_random(block_val.qkv_weight, embed_std_val);
            fill_random(block_val.attention_out_weight, embed_std_val);
            fill_random(block_val.mlp_up_weight, embed_std_val);
            fill_random(block_val.mlp_down_weight, 1.0f / sqrtf(4.0f * config_val.embed_dim));
            for (const Tensor &bias_tensor : {block_val.ln1_bias, block_val.qkv_bias, block_val.attention_out_bias,
                                              block_val.ln2_bias, block_val.mlp_up_bias, block_val.mlp_down_bias})
                fill_random(bias_tensor, 0.1f);
            fill_random(block_val.ln1_weight, 0.1f, 1.0f);
            fill_random(block_val.ln2_weight, 0.1f, 1.0f);
        }
        fill_random(params_val.lnf_weight, 0.1f, 1.0f);
        fill_random(params_val.lnf_bias, 0.1f);
        fill_random_ids(token_ids_tensor, config_val.vocab_size);
        fill_random_ids(targets_tensor, config_val.vocab_size);
    }

    /// Runs gpt2_forward on these tensors.
    void forward() { gpt2_forward(activations_val, token_ids_tensor, targets_tensor, params_val, scratch_val, config_val); }

    /// Runs gpt2_backward on these tensors.
    void backward()
    {
        gpt2_backward(grads_val, token_ids_tensor, targets_tensor, params_val, activations_val, scratch_val, config_val);
    }

    /// Gives every parameter's gradient as a CPU list, in GPT2Params::shapes()' order.
    std::vector<std::vector<float>> grad_lists() const
    {
        std::vector<std::vector<float>> lists_list;
        for (const Tensor &grad_tensor : grads_val.tensors())
            lists_list.push_back(list_of(grad_tensor));
        return lists_list;
    }
};

/// Runs the block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and the 12
/// parameters in BlockParams::shapes()' order, gives the output. The same as test_block.cu's.
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

/// Runs the same model the plain PyTorch way, on PyTorch tensors: from int64 token_ids {B, T} and the parameters
/// in GPT2Params::shapes()' order, gives the logits {B, T, padded_vocab_size}.
static torch::Tensor pytorch_gpt2_logits(const torch::Tensor &token_ids_tensor, const std::vector<torch::Tensor> &p_list,
                                         const GPT2Config &config_val)
{
    // 1. the encoder
    int64_t num_layers_val = (int64_t)config_val.num_layers;
    int64_t seq_len_val = token_ids_tensor.size(1);
    torch::Tensor x_tensor = torch::embedding(p_list[0], token_ids_tensor) + p_list[1].narrow(0, 0, seq_len_val);
    // 2. the blocks, each's 12 gathered as GPT2Params does
    for (int64_t layer_val = 0; layer_val < num_layers_val; layer_val++)
    {
        std::vector<torch::Tensor> block_list;
        for (int64_t param_index_val = 0; param_index_val < BLOCK_MATRICES; param_index_val++)
            block_list.push_back(p_list[2 + BLOCK_MATRICES * layer_val + param_index_val]);
        for (int64_t param_index_val = 0; param_index_val < BLOCK_PARAMS - BLOCK_MATRICES; param_index_val++)
            block_list.push_back(p_list[2 + BLOCK_MATRICES * num_layers_val + (BLOCK_PARAMS - BLOCK_MATRICES) * layer_val +
                                        param_index_val]);
        x_tensor = pytorch_block_forward(x_tensor, block_list, (int64_t)config_val.num_heads);
    }
    // 3. lnf, then the head: wte, no bias
    x_tensor = torch::layer_norm(x_tensor, {(int64_t)config_val.embed_dim}, p_list[2 + BLOCK_PARAMS * num_layers_val],
                                 p_list[3 + BLOCK_PARAMS * num_layers_val], 1e-5);
    return torch::linear(x_tensor, p_list[0]);
}

/// Gives PyTorch's loss for these logits: cross-entropy over the first vocab_size columns (the padding left out),
/// the mean over all B x T rows.
static torch::Tensor pytorch_loss(const torch::Tensor &logits_tensor, const torch::Tensor &targets_tensor,
                                  size_t vocab_size_val)
{
    torch::Tensor words_tensor = logits_tensor.narrow(2, 0, (int64_t)vocab_size_val);
    return torch::cross_entropy_loss(words_tensor.reshape({-1, (int64_t)vocab_size_val}), targets_tensor.reshape({-1}));
}

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of GPT-2 small, 12 blocks on {4, 1024}, in bfloat16,
    // the dtype training runs in: mine against PyTorch's model with autograd, also in bfloat16, on the same
    // values. Steps 1 and 2 do first-time work (loading kernels, cuBLAS picking its own), so the averages
    // are of 3..20. Each side starts every backward from gradients of 0, as a training step does.
    //
    // Expect: my loss still matches PyTorch's model in float on the same bfloat16 values within
    // BFLOAT16_TOLERANCE.

    // 1. GPT-2 small, random values; PyTorch's copy in bfloat16
    GPT2Config config_val;
    torch::manual_seed(0);
    GPT2Tensors mine_val(config_val, 4, 1024, DType::BFloat16);
    mine_val.fill_random_values();
    torch::Tensor token_ids_tensor = pytorch_ids_of(mine_val.token_ids_tensor);
    torch::Tensor targets_tensor = pytorch_ids_of(mine_val.targets_tensor);
    std::vector<torch::Tensor> pytorch_params_list;
    for (const Tensor &param_tensor : mine_val.params_val.tensors())
        pytorch_params_list.push_back(pytorch_copy_of(param_tensor).to(torch::kBFloat16).requires_grad_(true));
    torch::Tensor pytorch_loss_tensor;

    // 2. time TIMED_STEPS steps, each backward from gradients of 0
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { mine_val.forward(); });
        mine_val.grad_buffer_val.flat().zero_();
        times_list[1] = gpu_milliseconds([&] { mine_val.backward(); });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_loss_tensor = pytorch_loss(pytorch_gpt2_logits(token_ids_tensor, pytorch_params_list, config_val),
                                               targets_tensor, config_val.vocab_size);
        });
        for (torch::Tensor &param_tensor : pytorch_params_list)
            param_tensor.mutable_grad() = torch::Tensor();
        times_list[3] = gpu_milliseconds([&] { pytorch_loss_tensor.backward(); });
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

    // 4. my loss against PyTorch's in float, once, not timed
    double loss_in_float_val;
    {
        torch::NoGradGuard no_grad_val;
        std::vector<torch::Tensor> float_params_list;
        for (const Tensor &param_tensor : mine_val.params_val.tensors())
            float_params_list.push_back(pytorch_copy_of(param_tensor));
        loss_in_float_val = pytorch_loss(pytorch_gpt2_logits(token_ids_tensor, float_params_list, config_val), targets_tensor,
                                         config_val.vocab_size).item<double>();
    }
    ASSERT_NEAR((double)gpt2_mean_loss(mine_val.activations_val), loss_in_float_val, BFLOAT16_TOLERANCE);
}

// ══ part 2: the micro-batch size with the best MFU ═════════════════════════════

/// Gives the flops one token of training takes, forward and backward, counted as nanoGPT and llm.c count them
/// (from the PaLM paper's appendix B). Each parameter is one multiply and one add in the forward, 2 flops, and
/// twice that in the backward, one gradient for the input and one for the weight, 4 flops: 6 a parameter. The
/// attention's scores and weighted sums have no parameters, so they come on top: 12 x layers x embed_dim x
/// seq_len. GPT-2 small at T = 1024: 6 x 124,475,904 + 12 x 12 x 768 x 1024 = 746,855,424 + 113,246,208
/// = 860,101,632.
static double training_flops_per_token(size_t num_params_val, const GPT2Config &config_val, size_t seq_len_val)
{
    return 6.0 * num_params_val + 12.0 * config_val.num_layers * config_val.embed_dim * seq_len_val;
}

/// Gives the GPU memory free now, in bytes, to tell whether the next micro-batch size fits.
static size_t gpu_bytes_free(void)
{
    size_t free_bytes_val = 0;
    size_t total_bytes_val = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes_val, &total_bytes_val));
    return free_bytes_val;
}

static void find_the_micro_batch_with_the_best_mfu(void)
{
    // Finds how many rows a micro-batch should have (train_gpt2.cu's --batch-size) for the best MFU on this GPU.
    // MFU, model flops utilization, is the flops training does a second over the most the tensor cores can do:
    //
    //     MFU = tokens/s x flops a token / peak flops a second
    //
    // The H100 PCIe run of 2026-11-04 trained with 4 rows of 1,024 tokens: 186,000 tokens/s x 860.1M flops a
    // token = 160 TFLOPS of the GPU's 820 (peak_bf16_flops), an MFU of 20%, with 6 GB of its 85 GB in use. With
    // so few rows each of the many launches a micro-batch makes has little work: the matrices are too small to
    // keep every SM busy, the GPU waits between launches, and the loss read back after each micro-batch stops
    // the CPU from queueing the next. More rows give every launch more work for the same waits.
    //
    // For 1, 2, 4, ... LARGEST_MICRO_BATCH rows of SEQ_LEN tokens, GPT-2 small in bfloat16 as train_gpt2.cu
    // trains it: a micro-batch is a forward, a backward and the loss read back to the CPU, as the training loop
    // runs it, averaged over TIMED_STEPS after a second of warm-up. The parameters and gradients are made once,
    // with GPT-2's starting values; the activations and scratch again for each size. The activations grow with
    // the rows (0.55 GB a row on an H100 PCIe), so the doubling stops at the first size that would pass
    // MEMORY_HEADROOM of the memory free; then the sweep tries the most rows that fit, a multiple of
    // MOST_ROWS_STEP (136 on an H100 PCIe's 85 GB), as a micro-batch need not be a power of 2: only
    // --tokens-per-step must be a multiple of GPUs x rows x SEQ_LEN. Left out: the optimizer, once a step (about 1.5 ms on an H100 PCIe), the data loading, and
    // the GPUs' gradient exchange when there are several.
    //
    // Prints for each size: its activation memory, the ms of a micro-batch and of a TOKENS_PER_STEP step on one
    // GPU, tokens/s, TFLOPS and MFU; then the best size.

    // 1. GPT-2 small's parameters and gradients in bfloat16, a TensorBuffer each, with GPT-2's starting values
    GPT2Config config_val;
    TensorBuffer param_buffer_val;
    int group_val = param_buffer_val.add_many(GPT2Params::shapes(config_val));
    param_buffer_val.allocate(DType::BFloat16);
    TensorBuffer grad_buffer_val = TensorBuffer::zeros_like(param_buffer_val);
    std::vector<Tensor> param_tensors_list;
    std::vector<Tensor> grad_tensors_list;
    for (int param_index_val = 0; param_index_val < param_buffer_val.group_size(group_val); param_index_val++)
    {
        param_tensors_list.push_back(param_buffer_val.tensor(group_val, param_index_val));
        grad_tensors_list.push_back(grad_buffer_val.tensor(group_val, param_index_val));
    }
    GPT2Params params_val(param_tensors_list, config_val);
    GPT2Params grads_val(grad_tensors_list, config_val);
    Tensor::manual_seed(0);
    gpt2_init_params(params_val, config_val);
    double flops_per_token_val = training_flops_per_token(param_buffer_val.numel(), config_val, SEQ_LEN);
    double peak_flops_val = peak_bf16_flops();                 // 0 if this GPU's generation is not in its table

    // 2. each size, while it fits
    print_test_name(__func__);
    printf("    %zu parameters, %.1fM flops a token at T = %d; this GPU's peak %.0f TFLOPS in bfloat16\n",
           param_buffer_val.numel(), flops_per_token_val / 1e6, SEQ_LEN, peak_flops_val / TERA);
    printf("    rows   activations GB   ms a micro-batch   ms a step of %d   tokens/s   TFLOPS    MFU\n", TOKENS_PER_STEP);
    size_t bytes_per_row_val = 0;                              // activation bytes a row, from the last size run
    size_t best_rows_val = 0;
    double best_tokens_per_second_val = 0.0;

    // 2a. one size: false if it would not fit
    auto try_rows_val = [&](size_t rows_val) -> bool {
        // its memory, from the last size's bytes a row; too much: say so and stop
        double needed_bytes_val = (double)rows_val * bytes_per_row_val;
        if (needed_bytes_val > MEMORY_HEADROOM * gpu_bytes_free())
        {
            printf("    %4zu   would need about %.1f GB, with %.1f GB free: stop\n", rows_val, needed_bytes_val / GIGA,
                   gpu_bytes_free() / GIGA);
            return false;
        }

        // the activations, scratch, token_ids and targets for this many rows; any ids below vocab_size
        size_t free_before_val = gpu_bytes_free();
        GPT2Activations activations_val(rows_val, SEQ_LEN, config_val, DType::BFloat16);
        GPT2Scratch scratch_val(rows_val, SEQ_LEN, config_val, DType::BFloat16);
        Tensor token_ids_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        Tensor targets_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        std::vector<int> ids_list(rows_val * SEQ_LEN);
        for (size_t i_val = 0; i_val < ids_list.size(); i_val++)
            ids_list[i_val] = (int)((i_val * 7919) % config_val.vocab_size);  // 7919, a prime: ids spread over the vocabulary
        token_ids_tensor.copy_from_cpu(ids_list.data());
        std::rotate(ids_list.begin(), ids_list.begin() + 1, ids_list.end()); // each target the next id
        targets_tensor.copy_from_cpu(ids_list.data());
        size_t activation_bytes_val = free_before_val - gpu_bytes_free();
        bytes_per_row_val = activation_bytes_val / rows_val;

        // one micro-batch as the training loop runs it, warmed up, then timed
        auto micro_batch_val = [&] {
            gpt2_forward(activations_val, token_ids_tensor, targets_tensor, params_val, scratch_val, config_val);
            gpt2_backward(grads_val, token_ids_tensor, targets_tensor, params_val, activations_val, scratch_val,
                          config_val);
            gpt2_mean_loss(activations_val);                    // read back to the CPU, as the training loop does
        };
        warm_up_gpu(micro_batch_val);
        double total_ms_val = 0.0;
        for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
            total_ms_val += gpu_milliseconds(micro_batch_val);

        // its numbers
        double ms_val = total_ms_val / TIMED_STEPS;
        double tokens_val = (double)rows_val * SEQ_LEN;
        double tokens_per_second_val = tokens_val / (ms_val / MILLISECONDS_PER_SECOND);
        double flops_per_second_val = tokens_per_second_val * flops_per_token_val;
        double step_ms_val = ms_val * (TOKENS_PER_STEP / tokens_val);
        printf("    %4zu   %14.2f   %16.2f   %18.0f   %8.0f   %6.1f   ", rows_val, activation_bytes_val / GIGA, ms_val,
               step_ms_val, tokens_per_second_val, flops_per_second_val / TERA);
        if (peak_flops_val > 0.0)
            printf("%4.1f%%\n", 100.0 * flops_per_second_val / peak_flops_val);
        else
            printf("   -\n");
        if (tokens_per_second_val > best_tokens_per_second_val)
        {
            best_tokens_per_second_val = tokens_per_second_val;
            best_rows_val = rows_val;
        }
        return true;
    };

    // 2b. 1, 2, 4, ... rows while they fit
    size_t last_rows_val = 0;
    for (size_t rows_val = 1; rows_val <= LARGEST_MICRO_BATCH && try_rows_val(rows_val); rows_val *= 2)
        last_rows_val = rows_val;

    // 2c. then the most rows that fit, a multiple of MOST_ROWS_STEP
    size_t most_rows_val = (size_t)(MEMORY_HEADROOM * gpu_bytes_free() / bytes_per_row_val) / MOST_ROWS_STEP * MOST_ROWS_STEP;
    if (most_rows_val > last_rows_val)
        try_rows_val(most_rows_val);

    // 3. the best size
    double best_flops_val = best_tokens_per_second_val * flops_per_token_val;
    printf("    best: --batch-size %zu, %.0f tokens/s, %.1f TFLOPS", best_rows_val, best_tokens_per_second_val,
           best_flops_val / TERA);
    if (peak_flops_val > 0.0)
        printf(", MFU %.1f%%", 100.0 * best_flops_val / peak_flops_val);
    printf(" (--tokens-per-step must be a multiple of GPUs x rows x %d)\n", SEQ_LEN);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    find_the_micro_batch_with_the_best_mfu();    // first: part 1's PyTorch keeps the memory it used, cached
    time_20_steps_on_gpt2_small_size();

    printf("\nall tunes done\n");
    return 0;
}
