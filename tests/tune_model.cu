// tune_model.cu — the whole model's speed at the size train_bf16.cu and train_8bit.cu train (nanochat's d24: 24 layers x
// 1,536, 12 heads of 128, rows of 2,048 tokens, 32,768 ids, no linear bias, 3 of 4 layers a sliding window of 512), apart from tests/test_model.cu so the tests only check numbers:
//     find_the_micro_batch_with_the_best_mfu   our bfloat16 model, then our 8-bit one: tokens/s and MFU at each
//                                              --batch-size that fits, the best to train with
//     eight_bit_gradients_follow_bfloat16s     the 8-bit model's gradients against the bfloat16 model's (cosine)
//     time_against_pytorch                     PyTorch's model in bfloat16 on the same weights (cuBLAS for every
//                                              product, its own flash attention, which has no window: every layer
//                                              the whole row), at the same rows: tokens/s against ours; and our loss
//                                              against PyTorch's in float with the same windows (masks), once
//
// Built as tests/test_model.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/model_<date>_<time>.txt.
#include <algorithm>                  // std::rotate
#include <map>
#include <string>
#include <torch/types.h>              // torch::embedding, torch::linear, cross_entropy_loss, ...; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed, torch::NoGradGuard
#include <c10/cuda/CUDACachingAllocator.h>   // emptyCache: PyTorch's cached memory back to the GPU after a size that did not fit
#include "test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, peak_bf16_flops
#include "../llmc/common/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/model_bf16.cuh"      // the bfloat16 model: gpt2_bf16_train_step
#include "../llmc/model_8bit.cuh"      // the 8-bit model: gpt2_8bit_train_step

// Against PyTorch in float on the same values: every kernel rounds its outputs once (up to 2^-8, 0.4%), and the next
// one reads them; the loss over a row of 2,048 tokens averages much of it out.
static const double BFLOAT16_TOLERANCE = 5e-2;

// The model the trainers train (train_bf16.cu's and train_8bit.cu's defines)
#define MODEL_NUM_LAYERS 24
#define MODEL_EMBED_DIM 1536
#define MODEL_NUM_HEADS 12
#define MODEL_VOCAB_SIZE 32768        // nanochat's tokenizer: 2^15 ids, no padding
#define MODEL_ATTENTION_WINDOW 512    // the short layers' sliding window (SSSL); 0: none
#define MODEL_LINEAR_BIAS false       // the trainers' default: no linear bias, the LayerNorms with theirs
#define SEQ_LEN 2048                  // tokens per row, as the trainers train
#define LARGEST_MICRO_BATCH 64        // rows: the micro-batch sweep tries 1, 2, 4, ... this many, while they fit
#define MOST_ROWS_STEP 2              // then the most rows that fit, rounded down to a multiple of this
#define TOKENS_PER_STEP 524288        // the trainers' step, all GPUs together: 2^19
#define MEMORY_HEADROOM 0.9           // the next size is tried only if it needs at most this much of the free memory

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives the model the trainers train: GPT2Config with the sizes above.
static GPT2Config model_config_of(void)
{
    GPT2Config config_val;
    config_val.num_layers = MODEL_NUM_LAYERS;
    config_val.embed_dim = MODEL_EMBED_DIM;
    config_val.num_heads = MODEL_NUM_HEADS;
    config_val.max_seq_len = SEQ_LEN;
    config_val.linear_bias = MODEL_LINEAR_BIAS;
    config_val.vocab_size = MODEL_VOCAB_SIZE;
    config_val.padded_vocab_size = MODEL_VOCAB_SIZE;
    config_val.attention_window = MODEL_ATTENTION_WINDOW;
    return config_val;
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

/// Gives the GPU memory free now, in bytes, to tell whether the next micro-batch size fits.
static size_t gpu_bytes_free(void)
{
    size_t free_bytes_val = 0;
    size_t total_bytes_val = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes_val, &total_bytes_val));
    return free_bytes_val;
}

/// The model's parameters and their gradients in bfloat16, a TensorBuffer each as the optimizer keeps them, with
/// GPT-2's starting values: what both our models read and write (the 8-bit model's params are bfloat16 too).
struct ModelBuffers
{
    TensorBuffer param_buffer_val;
    TensorBuffer grad_buffer_val;
    GPT2Params params_val;
    GPT2Params grads_val;

    /// Makes them for config_val, the values GPT-2's start (gpt2_init_params, seed 0), the gradients 0.
    explicit ModelBuffers(const GPT2Config &config_val)
    {
        int group_index_val = param_buffer_val.add_many(GPT2Params::shapes(config_val));
        param_buffer_val.allocate(DType::BFloat16);
        grad_buffer_val = TensorBuffer::zeros_like(param_buffer_val);
        std::vector<Tensor> param_tensors_list;
        std::vector<Tensor> grad_tensors_list;
        for (int param_index_val = 0; param_index_val < param_buffer_val.group_size(group_index_val); param_index_val++)
        {
            param_tensors_list.push_back(param_buffer_val.tensor(group_index_val, param_index_val));
            grad_tensors_list.push_back(grad_buffer_val.tensor(group_index_val, param_index_val));
        }
        params_val = GPT2Params(param_tensors_list, config_val);
        grads_val = GPT2Params(grad_tensors_list, config_val);
        Tensor::manual_seed(0);
        gpt2_init_params(params_val, config_val);
    }
};

/// Fills token_ids {rows, SEQ_LEN} with ids spread over the vocabulary, and targets with each next id.
///
///   token_ids_tensor, targets_tensor   Int32 {rows, SEQ_LEN}: written
///   vocab_size_val                     the ids stay below it
static void fill_ids(const Tensor &token_ids_tensor, const Tensor &targets_tensor, size_t vocab_size_val)
{
    std::vector<int> ids_list(token_ids_tensor.numel());
    for (size_t id_index_val = 0; id_index_val < ids_list.size(); id_index_val++)
        ids_list[id_index_val] = (int)((id_index_val * 7919) % vocab_size_val);  // 7919, a prime: ids spread over the vocabulary
    token_ids_tensor.copy_from_cpu(ids_list.data());
    std::rotate(ids_list.begin(), ids_list.begin() + 1, ids_list.end()); // each target the next id
    targets_tensor.copy_from_cpu(ids_list.data());
}

/// Runs the block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and the 12
/// parameters in BlockParams::shapes()' order (a linear bias undefined, torch::Tensor(), is none), gives the output.
/// window_val 0: causal attention over the whole row (PyTorch's flash attention); else a token sees itself and the
/// window_val tokens before it, by a mask {T, T} (PyTorch's flash attention has no window: its slower masked path).
static torch::Tensor pytorch_block_forward(const torch::Tensor &input_tensor, const std::vector<torch::Tensor> &p_list,
                                           int64_t num_heads_val, int64_t window_val = 0)
{
    // 1. the sizes
    int64_t batch_size_val = input_tensor.size(0);
    int64_t seq_len_val = input_tensor.size(1);
    int64_t embed_dim_val = input_tensor.size(2);
    int64_t head_size_val = embed_dim_val / num_heads_val;
    std::vector<int64_t> normalized_shape_list = {embed_dim_val};
    // p: 0 qkv_weight, 1 attention_out_weight, 2 mlp_up_weight, 3 mlp_down_weight, 4 ln1_weight, 5 ln1_bias,
    //    6 qkv_bias, 7 attention_out_bias, 8 ln2_weight, 9 ln2_bias, 10 mlp_up_bias, 11 mlp_down_bias
    // 2. ln1, qkv, the causal attention (PyTorch's flash attention), its projection, the residual add
    torch::Tensor qkv_tensor =
        torch::linear(torch::layer_norm(input_tensor, normalized_shape_list, p_list[4], p_list[5], 1e-5), p_list[0], p_list[6]);
    std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
    auto heads_val = [&](const torch::Tensor &part_tensor) {
        return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, head_size_val}).transpose(1, 2);
    };
    torch::Tensor y_tensor;
    if (window_val == 0)
        y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]), heads_val(parts_list[2]),
                                                       {}, 0.0, true);
    else
    {
        // true where the query (row) sees the key (column): the key at or before it, at most window_val before
        torch::Tensor seen_tensor = torch::ones({seq_len_val, seq_len_val}, torch::TensorOptions().dtype(torch::kBool).device(input_tensor.device()));
        seen_tensor = seen_tensor.tril(0).triu(-window_val);
        y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]), heads_val(parts_list[2]),
                                                       seen_tensor, 0.0, false);
    }
    y_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
    torch::Tensor residual2_tensor = input_tensor + torch::linear(y_tensor, p_list[1], p_list[7]);
    // 3. ln2, the MLP, the residual add
    torch::Tensor ln2_output_tensor = torch::layer_norm(residual2_tensor, normalized_shape_list, p_list[8], p_list[9], 1e-5);
    torch::Tensor mlp_up_tensor = torch::linear(ln2_output_tensor, p_list[2], p_list[10]);
    return residual2_tensor + torch::linear(torch::gelu(mlp_up_tensor, "tanh"), p_list[3], p_list[11]);
}

/// Runs the same model the plain PyTorch way, on PyTorch tensors: from int64 token_ids {B, T} and the parameters in
/// GPT2Params::shapes()' order (with no linear bias, each block's 4 vectors the LayerNorms'), gives the logits
/// {B, T, padded_vocab_size}. windows_val: the short layers' sliding windows as ours (gpt2_attention_window), by
/// masks; false: every layer the whole row.
static torch::Tensor pytorch_gpt2_logits(const torch::Tensor &token_ids_tensor, const std::vector<torch::Tensor> &p_list,
                                         const GPT2Config &config_val, bool windows_val = false)
{
    // 1. the encoder
    int64_t num_layers_val = (int64_t)config_val.num_layers;
    int64_t seq_len_val = token_ids_tensor.size(1);
    int64_t vectors_val = (int64_t)BlockParams::vector_count(config_val.linear_bias);
    torch::Tensor x_tensor = torch::embedding(p_list[0], token_ids_tensor) + p_list[1].narrow(0, 0, seq_len_val);
    // 2. the blocks, each's tensors gathered as GPT2Params does, made 12 with the linear biases undefined if none
    for (int64_t layer_index_val = 0; layer_index_val < num_layers_val; layer_index_val++)
    {
        std::vector<torch::Tensor> block_list;
        for (int64_t param_index_val = 0; param_index_val < 4; param_index_val++)
            block_list.push_back(p_list[2 + 4 * layer_index_val + param_index_val]);
        const torch::Tensor *vectors_ptr = &p_list[2 + 4 * num_layers_val + vectors_val * layer_index_val];
        if (config_val.linear_bias)
            block_list.insert(block_list.end(), vectors_ptr, vectors_ptr + vectors_val);
        else      // ln1_weight, ln1_bias, qkv_bias, attention_out_bias, ln2_weight, ln2_bias, mlp_up_bias, mlp_down_bias
            block_list.insert(block_list.end(), {vectors_ptr[0], vectors_ptr[1], torch::Tensor(), torch::Tensor(),
                                                 vectors_ptr[2], vectors_ptr[3], torch::Tensor(), torch::Tensor()});
        x_tensor = pytorch_block_forward(x_tensor, block_list, (int64_t)config_val.num_heads,
                                         windows_val ? (int64_t)gpt2_attention_window(config_val, (size_t)layer_index_val) : 0);
    }
    // 3. lnf (the last two tensors), then the head: wte, no bias
    x_tensor = torch::layer_norm(x_tensor, {(int64_t)config_val.embed_dim}, p_list[p_list.size() - 2],
                                 p_list[p_list.size() - 1], 1e-5);
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

/// Gives the flops one token of training takes, forward and backward, counted as nanoGPT and llm.c count them
/// (from the PaLM paper's appendix B). Each parameter is one multiply and one add in the forward, 2 flops, and
/// twice that in the backward, one gradient for the input and one for the weight, 4 flops: 6 a parameter. The
/// attention's scores and weighted sums have no parameters, so they come on top: 12 x embed_dim x the keys a token
/// can see, each layer: seq_len, or a short layer's window (as nanochat's estimate_flops counts it).
static double training_flops_per_token(size_t num_params_val, const GPT2Config &config_val, size_t seq_len_val)
{
    double attention_val = 0.0;
    for (size_t layer_index_val = 0; layer_index_val < config_val.num_layers; layer_index_val++)
    {
        size_t window_val = gpt2_attention_window(config_val, layer_index_val);
        attention_val += 12.0 * config_val.embed_dim * (window_val == 0 ? seq_len_val : std::min(window_val, seq_len_val));
    }
    return 6.0 * num_params_val + attention_val;
}

// ══ part 1: the micro-batch size with the best MFU, each of our models ══════════

/// One model's sweep: tokens/s at each number of rows it ran.
using TokensPerSecond = std::map<size_t, double>;

/// Times one of our models, as its trainer runs a micro-batch (train_step: a forward, a backward and the loss read
/// back), at 1, 2, 4, ... LARGEST_MICRO_BATCH rows of SEQ_LEN tokens while they fit, then the most rows that fit (a
/// multiple of MOST_ROWS_STEP); each size warmed up, then averaged over TIMED_STEPS. Prints a line per size and the
/// best; gives the tokens/s of each size.
///
///   model_name_ptr      for the lines
///   buffers_val         the params and grads, shared by the models
///   config_val          the model
///   train_step_ptr      gpt2_bf16_train_step or gpt2_8bit_train_step
/// ACTIVATIONS: GPT2ActivationsBf16 or GPT2Activations8bit, made again for each size.
template <typename ACTIVATIONS>
static TokensPerSecond sweep_micro_batches(const char *model_name_ptr, const ModelBuffers &buffers_val, const GPT2Config &config_val,
                                           float (*train_step_ptr)(const GPT2Params &, const Tensor &, const Tensor &,
                                                                   const GPT2Params &, const ACTIVATIONS &, const GPT2Config &, size_t))
{
    // 1. the flops a token and the GPU's peak
    double flops_per_token_val = training_flops_per_token(buffers_val.param_buffer_val.numel(), config_val, SEQ_LEN);
    double peak_flops_val = peak_bf16_flops();                 // 0 if this GPU's generation is not in its table
    printf("    %s: %zu parameters, %.2fG flops a token at T = %d; this GPU's peak %.0f TFLOPS in bfloat16\n", model_name_ptr,
           buffers_val.param_buffer_val.numel(), flops_per_token_val / 1e9, SEQ_LEN, peak_flops_val / TERA);
    printf("    rows   activations GB   ms a micro-batch   ms a step of %d   tokens/s   TFLOPS    MFU   loss\n", TOKENS_PER_STEP);
    TokensPerSecond results_val;
    size_t bytes_per_row_val = 0;                              // activation bytes a row, from the last size run
    size_t best_rows_val = 0;
    double best_tokens_per_second_val = 0.0;

    // 2. one size: false if it would not fit
    auto try_rows_val = [&](size_t rows_val) -> bool {
        double needed_bytes_val = (double)rows_val * bytes_per_row_val;
        if (needed_bytes_val > MEMORY_HEADROOM * gpu_bytes_free())
        {
            printf("    %4zu   would need about %.1f GB, with %.1f GB free: stop\n", rows_val, needed_bytes_val / GIGA,
                   gpu_bytes_free() / GIGA);
            return false;
        }
        // the activations, token_ids and targets for this many rows
        size_t free_before_val = gpu_bytes_free();
        ACTIVATIONS activations_val(rows_val, SEQ_LEN, config_val);
        Tensor token_ids_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        Tensor targets_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        fill_ids(token_ids_tensor, targets_tensor, config_val.vocab_size);
        size_t activation_bytes_val = free_before_val - gpu_bytes_free();
        bytes_per_row_val = activation_bytes_val / rows_val;
        // one micro-batch as the training loop runs it, warmed up, then timed
        float loss_val = 0.0f;
        auto micro_batch_val = [&] {
            loss_val = train_step_ptr(buffers_val.grads_val, token_ids_tensor, targets_tensor, buffers_val.params_val,
                                      activations_val, config_val, 1);
        };
        warm_up_gpu(micro_batch_val);
        double total_ms_val = 0.0;
        for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
            total_ms_val += gpu_milliseconds(micro_batch_val);
        // its numbers
        double ms_val = total_ms_val / TIMED_STEPS;
        double tokens_val = (double)rows_val * SEQ_LEN;
        double tokens_per_second_val = tokens_val / (ms_val / MILLISECONDS_PER_SECOND);
        double flops_per_second_val = tokens_per_second_val * flops_per_token_val;
        printf("    %4zu   %14.2f   %16.2f   %18.0f   %8.0f   %6.1f   ", rows_val, activation_bytes_val / GIGA, ms_val,
               ms_val * (TOKENS_PER_STEP / tokens_val), tokens_per_second_val, flops_per_second_val / TERA);
        if (peak_flops_val > 0.0)
            printf("%4.1f%%", 100.0 * flops_per_second_val / peak_flops_val);
        else
            printf("   -");
        printf("   %.3f\n", loss_val);
        results_val[rows_val] = tokens_per_second_val;
        if (tokens_per_second_val > best_tokens_per_second_val)
            best_tokens_per_second_val = tokens_per_second_val, best_rows_val = rows_val;
        return true;
    };

    // 3. 1, 2, 4, ... rows while they fit, then the most rows that fit
    size_t last_rows_val = 0;
    for (size_t rows_val = 1; rows_val <= LARGEST_MICRO_BATCH && try_rows_val(rows_val); rows_val *= 2)
        last_rows_val = rows_val;
    size_t most_rows_val = bytes_per_row_val > 0
                               ? (size_t)(MEMORY_HEADROOM * gpu_bytes_free() / bytes_per_row_val) / MOST_ROWS_STEP * MOST_ROWS_STEP
                               : 0;
    if (most_rows_val > last_rows_val)
        try_rows_val(most_rows_val);

    // 4. the best size
    double best_flops_val = best_tokens_per_second_val * flops_per_token_val;
    printf("    best: --batch-size %zu, %.0f tokens/s, %.1f TFLOPS", best_rows_val, best_tokens_per_second_val,
           best_flops_val / TERA);
    if (peak_flops_val > 0.0)
        printf(", MFU %.1f%%", 100.0 * best_flops_val / peak_flops_val);
    printf(" (--tokens-per-step must be a multiple of GPUs x rows x %d)\n\n", SEQ_LEN);
    return results_val;
}

/// Checks the 8-bit model's backward against the bfloat16 model's: one micro-batch of 1 row from gradients of 0, our
/// bfloat16 model then our 8-bit one on the same weights and ids; each parameter's two gradients compared by their
/// cosine similarity (a · b / |a| |b|, 1 for the same direction), in float by PyTorch. E4M3's rounding moves each
/// value by up to 6%, which leaves a gradient's direction nearly as it was; a wrong copy (turned the wrong way, a
/// scale on the wrong side) gives about 0. Prints the smallest and which tensor; expects it at least
/// EIGHT_BIT_LEAST_COSINE. A gradient of 0 in both counts as the same.
///
///   buffers_val   the params (GPT-2's starting values) and grads (written over)
///   config_val    the model
static void eight_bit_gradients_follow_bfloat16s(const ModelBuffers &buffers_val, const GPT2Config &config_val)
{
    const double EIGHT_BIT_LEAST_COSINE = 0.9;
    // 1. the ids, and both models' activations for 1 row
    Tensor token_ids_tensor = Tensor::zeros({1, (size_t)SEQ_LEN}, DType::Int32);
    Tensor targets_tensor = Tensor::zeros({1, (size_t)SEQ_LEN}, DType::Int32);
    fill_ids(token_ids_tensor, targets_tensor, config_val.vocab_size);
    // 2. the bfloat16 model's gradients, kept; then the 8-bit model's, in the same buffer
    TensorBuffer bf16_grads_val = TensorBuffer::zeros_like(buffers_val.grad_buffer_val);
    {
        GPT2ActivationsBf16 activations_val(1, SEQ_LEN, config_val);
        buffers_val.grad_buffer_val.flat().zero_();
        gpt2_bf16_train_step(buffers_val.grads_val, token_ids_tensor, targets_tensor, buffers_val.params_val, activations_val, config_val, 1);
        bf16_grads_val.flat().copy_(buffers_val.grad_buffer_val.flat());
    }
    {
        GPT2Activations8bit activations_val(1, SEQ_LEN, config_val);
        buffers_val.grad_buffer_val.flat().zero_();
        gpt2_8bit_train_step(buffers_val.grads_val, token_ids_tensor, targets_tensor, buffers_val.params_val, activations_val, config_val, 1);
        // again: with one scale per tensor, this micro-batch's scales come from the one before (the first's are exact)
        buffers_val.grad_buffer_val.flat().zero_();
        gpt2_8bit_train_step(buffers_val.grads_val, token_ids_tensor, targets_tensor, buffers_val.params_val, activations_val, config_val, 1);
    }
    // 3. each tensor's cosine, the smallest kept
    std::vector<Tensor> eight_list = buffers_val.grads_val.tensors();
    double least_cosine_val = 2.0;
    size_t least_index_val = 0;
    for (size_t param_index_val = 0; param_index_val < eight_list.size(); param_index_val++)
    {
        torch::Tensor eight_tensor = pytorch_copy_of(eight_list[param_index_val]).flatten().to(torch::kDouble);
        torch::Tensor bf16_tensor = pytorch_copy_of(bf16_grads_val.tensor(0, (int)param_index_val)).flatten().to(torch::kDouble);
        double norms_val = eight_tensor.norm().item<double>() * bf16_tensor.norm().item<double>();
        double cosine_val = norms_val > 0.0 ? (eight_tensor * bf16_tensor).sum().item<double>() / norms_val
                                            : (eight_tensor.norm().item<double>() == bf16_tensor.norm().item<double>() ? 1.0 : 0.0);
        if (cosine_val < least_cosine_val)
            least_cosine_val = cosine_val, least_index_val = param_index_val;
    }
    print_test_name(__func__);
    printf("    %zu gradients; the least alike: tensor %zu %s, cosine %.4f\n", eight_list.size(), least_index_val,
           eight_list[least_index_val].shape().size() == 2 ? "(a matrix)" : "(a vector)", least_cosine_val);
    ASSERT_EQ(least_cosine_val >= EIGHT_BIT_LEAST_COSINE, true);
}

// ══ part 2: against PyTorch (cuBLAS) ═══════════════════════════════════════════

/// Times PyTorch's model in bfloat16 at rows_val rows on params_list, a micro-batch as a training loop runs it
/// (gradients set to none, the loss, its backward, the loss read back), warmed up, then averaged over TIMED_STEPS.
/// Gives its tokens/s, or 0 if it ran out of memory (its cached memory then given back).
///
///   params_list   PyTorch's bfloat16 copies of the params, requires_grad
///   config_val    the model
///   rows_val      rows of SEQ_LEN tokens
static double pytorch_tokens_per_second(std::vector<torch::Tensor> &params_list, const GPT2Config &config_val, size_t rows_val)
{
    try
    {
        // the ids, as ours: spread over the vocabulary, each target the next
        Tensor token_ids_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        Tensor targets_tensor = Tensor::zeros({rows_val, (size_t)SEQ_LEN}, DType::Int32);
        fill_ids(token_ids_tensor, targets_tensor, config_val.vocab_size);
        torch::Tensor ids_tensor = pytorch_ids_of(token_ids_tensor);
        torch::Tensor next_ids_tensor = pytorch_ids_of(targets_tensor);
        auto micro_batch_val = [&] {
            for (torch::Tensor &param_tensor : params_list)
                param_tensor.mutable_grad() = torch::Tensor();
            torch::Tensor loss_tensor = pytorch_loss(pytorch_gpt2_logits(ids_tensor, params_list, config_val), next_ids_tensor,
                                                     config_val.vocab_size);
            loss_tensor.backward();
            (void)loss_tensor.item<float>();                   // read back, as the loop does
        };
        warm_up_gpu(micro_batch_val);
        double total_ms_val = 0.0;
        for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
            total_ms_val += gpu_milliseconds(micro_batch_val);
        return (double)rows_val * SEQ_LEN / (total_ms_val / TIMED_STEPS / MILLISECONDS_PER_SECOND);
    }
    catch (const c10::Error &)
    {
        for (torch::Tensor &param_tensor : params_list)
            param_tensor.mutable_grad() = torch::Tensor();
        c10::cuda::CUDACachingAllocator::emptyCache();
        return 0.0;
    }
}

/// Prints the table: at each row count our two models ran, PyTorch's tokens/s on the same weights and how many
/// times as fast ours are; PyTorch stops at the first size it has no memory for. Then our bfloat16 loss at 1 row
/// against PyTorch's model in float on the same values, within BFLOAT16_TOLERANCE.
///
///   buffers_val            the params (GPT-2's starting values) and grads
///   config_val             the model
///   bf16_val, eight_val    part 1's tokens/s
static void time_against_pytorch(const ModelBuffers &buffers_val, const GPT2Config &config_val, const TokensPerSecond &bf16_val,
                                 const TokensPerSecond &eight_val)
{
    // 1. PyTorch's bfloat16 copies of the same params
    std::vector<torch::Tensor> params_list;
    for (const Tensor &param_tensor : buffers_val.params_val.tensors())
        params_list.push_back(pytorch_copy_of(param_tensor).to(torch::kBFloat16).requires_grad_(true));

    // 2. each row count, while PyTorch fits
    print_test_name(__func__);
    printf("    tokens/s, a micro-batch as the training loop runs it; PyTorch: cuBLAS for every product, its flash attention\n");
    int short_layers_val = 0;
    for (size_t layer_index_val = 0; layer_index_val < config_val.num_layers; layer_index_val++)
        short_layers_val += gpt2_attention_window(config_val, layer_index_val) != 0;
    printf("    (which has no window: PyTorch's every layer sees the whole row, ours %d of %d layers a window of %d tokens)\n",
           short_layers_val, (int)config_val.num_layers, (int)config_val.attention_window);
    printf("    rows   ours bf16   ours 8-bit   PyTorch (cuBLAS)   bf16 / PyTorch   8-bit / PyTorch\n");
    for (const auto &row_val : bf16_val)
    {
        size_t rows_val = row_val.first;
        double pytorch_val = pytorch_tokens_per_second(params_list, config_val, rows_val);
        auto eight_it = eight_val.find(rows_val);
        double eight_tokens_val = eight_it != eight_val.end() ? eight_it->second : 0.0;
        printf("    %4zu   %9.0f   %10.0f   ", rows_val, row_val.second, eight_tokens_val);
        if (pytorch_val == 0.0)
        {
            printf("%16s\n", "out of memory");
            break;
        }
        printf("%16.0f   %13.2fx   ", pytorch_val, row_val.second / pytorch_val);
        if (eight_tokens_val > 0.0)
            printf("%14.2fx\n", eight_tokens_val / pytorch_val);
        else
            printf("%15s\n", "-");
    }
    params_list.clear();
    c10::cuda::CUDACachingAllocator::emptyCache();

    // 3. our bfloat16 loss at 1 row against PyTorch's model in float, not timed
    GPT2ActivationsBf16 activations_val(1, SEQ_LEN, config_val);
    Tensor token_ids_tensor = Tensor::zeros({1, (size_t)SEQ_LEN}, DType::Int32);
    Tensor targets_tensor = Tensor::zeros({1, (size_t)SEQ_LEN}, DType::Int32);
    fill_ids(token_ids_tensor, targets_tensor, config_val.vocab_size);
    gpt2_bf16_forward(activations_val, token_ids_tensor, targets_tensor, buffers_val.params_val, config_val);
    double loss_in_float_val;
    {
        torch::NoGradGuard no_grad_val;
        std::vector<torch::Tensor> float_params_list;
        for (const Tensor &param_tensor : buffers_val.params_val.tensors())
            float_params_list.push_back(pytorch_copy_of(param_tensor));
        loss_in_float_val = pytorch_loss(pytorch_gpt2_logits(pytorch_ids_of(token_ids_tensor), float_params_list, config_val, true),
                                         pytorch_ids_of(targets_tensor), config_val.vocab_size)
                                .item<double>();
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    ASSERT_NEAR((double)gpt2_mean_loss(activations_val), loss_in_float_val, BFLOAT16_TOLERANCE);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    GPT2Config config_val = model_config_of();
    ModelBuffers buffers_val(config_val);

    // part 1 first: part 2's PyTorch keeps the memory it used, cached
    print_test_name("find_the_micro_batch_with_the_best_mfu");
    TokensPerSecond bf16_val = sweep_micro_batches<GPT2ActivationsBf16>("bfloat16", buffers_val, config_val, gpt2_bf16_train_step);
    TokensPerSecond eight_val = sweep_micro_batches<GPT2Activations8bit>("8-bit", buffers_val, config_val, gpt2_8bit_train_step);
    eight_bit_gradients_follow_bfloat16s(buffers_val, config_val);
    time_against_pytorch(buffers_val, config_val, bf16_val, eight_val);

    printf("\nall tunes done\n");
    return 0;
}
