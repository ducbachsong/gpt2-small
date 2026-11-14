// test_attention_bf16.cu — tests for llmc/kernels/bf16/attention_bf16.cuh and attention_flash_bf16.cuh: the attention steps in bfloat16
// (attention_bf16_forward, attention_bf16_backward) and flash attention (attention_bf16_flash_forward,
// attention_bf16_flash_backward). Every check prints its test case, input, expected and actual value, and ok;
// the first FAIL stops the program.
//
// Part 1 checks the steps (their products on an H100's tensor cores, fed by TMA) against PyTorch's
// scaled_dot_product_attention (causal) in float on the same bfloat16 values; part 2 checks flash attention
// against PyTorch and against the steps; part 3 checks that each kernel's registers fit its block, and other
// block sizes. Both need an H100 (WGMMA and TMA): on another GPU their tests are skipped. Built with
// -arch=sm_90a. The numbers worked out by hand are
// in tests/fp32/test_attention.cu (float: the same math). The timings are in tests/bf16/tune_attention_bf16.cu.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/bf16/attention_bf16.cuh"
#include "../../llmc/kernels/bf16/attention_flash_bf16.cuh"

// Against PyTorch in float on the same values. A bfloat16 keeps 8 significant bits, so each rounding moves a
// value by up to 2^-8, about 0.4% of it, and attention rounds several times in a row: the scores before the
// softmax (a score of 4 by up to 0.008, which moves its attention_prob by 0.8%), the attention_probs, the outputs. The backward
// rounds da, ds and the three gradients on top, and ds = a (da - row_dot) takes two close numbers apart, so its
// errors are larger next to its own size. Each tolerance is the largest difference, as a fraction of the
// largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_HEAD_SIZE 64             // and their columns
#define GPT2_SEQ_LEN 1024             // its tokens per row

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a float tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 9.4M qkv values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
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

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: GPT-2's attention_probs are 50M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_bf16_forward and attention_bf16_backward take, for one size, all bfloat16, made with
/// 0s except qkv and output_grad, which the test fills.
struct AttentionTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor attention_probs_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor attention_probs_grad_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens and head_size columns.
    AttentionTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val, size_t head_size_val)
    {
        size_t embed_dim_val = num_heads_val * head_size_val;
        Shape per_pair_list = {batch_size_val, num_heads_val, seq_len_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        attention_probs_tensor = Tensor::zeros(per_pair_list, DType::BFloat16);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        attention_probs_grad_tensor = Tensor::zeros(per_pair_list, DType::BFloat16);
    }

    /// Runs attention_bf16_forward on these tensors, the softmax at this block size (the default if not given).
    void forward(int softmax_threads_per_block_val = ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK)
    {
        attention_bf16_forward(output_tensor, attention_probs_tensor, qkv_tensor, softmax_threads_per_block_val);
    }

    /// Runs attention_bf16_backward on these tensors, the softmax at this block size (the default if not given).
    void backward(int softmax_threads_per_block_val = ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK)
    {
        attention_bf16_backward(qkv_grad_tensor, output_grad_tensor, qkv_tensor, attention_probs_tensor, attention_probs_grad_tensor,
                                softmax_threads_per_block_val);
    }
};

/// PyTorch's answers in float for one qkv and output_grad, the plain PyTorch way, as GPT-2's model.py does it:
/// qkv split into q, k, v, each cut into heads, then scaled_dot_product_attention with is_causal, and autograd.
/// attention_probs and logsumexp are worked out on the side: softmax and logsumexp of the masked scores.
struct PytorchAttention
{
    torch::Tensor qkv_tensor;
    torch::Tensor output_tensor;
    torch::Tensor attention_probs_tensor;
    torch::Tensor logsumexp_tensor;                                // logsumexp of each row of masked scores

    /// Works out output, qkv.grad(), attention_probs and logsumexp from my qkv and output_grad, as floats.
    PytorchAttention(const AttentionTensors &mine_val)
    {
        // 1. the sizes
        int64_t batch_size_val = mine_val.attention_probs_tensor.size(0);
        int64_t num_heads_val = mine_val.attention_probs_tensor.size(1);
        int64_t seq_len_val = mine_val.attention_probs_tensor.size(2);
        int64_t embed_dim_val = mine_val.output_tensor.size(2);
        int64_t head_size_val = embed_dim_val / num_heads_val;
        // 2. qkv split and cut into heads, then the causal attention and its backward
        qkv_tensor = pytorch_copy_of(mine_val.qkv_tensor).requires_grad_(true);
        std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
        auto heads_val = [&](const torch::Tensor &part_tensor) {
            return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, head_size_val}).transpose(1, 2);
        };
        torch::Tensor q_tensor = heads_val(parts_list[0]);
        torch::Tensor k_tensor = heads_val(parts_list[1]);
        torch::Tensor v_tensor = heads_val(parts_list[2]);
        torch::Tensor y_tensor = torch::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor, {}, 0.0, true);
        output_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
        output_tensor.backward(pytorch_copy_of(mine_val.output_grad_tensor));
        // 3. the attention_probs and logsumexp, on the side
        torch::NoGradGuard no_grad_val;
        torch::Tensor scores_tensor = q_tensor.matmul(k_tensor.transpose(-2, -1)) / std::sqrt((double)head_size_val);
        torch::Tensor after_tensor = torch::ones({seq_len_val, seq_len_val}, scores_tensor.options().dtype(torch::kBool)).triu(1);
        attention_probs_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).softmax(-1);
        logsumexp_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).logsumexp(-1);
    }
};

/// Every tensor attention_bf16_flash_forward and attention_bf16_flash_backward take, for one size, all bfloat16
/// but logsumexp and row_dot (float), made with 0s except qkv and output_grad, which the test fills.
struct FlashTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor logsumexp_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor row_dot_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_bf16_flash_forward on these tensors.
    void forward()
    {
        attention_bf16_flash_forward(output_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_bf16_flash_backward on these tensors, row_dot at this block size (the default if not given).
    void backward(int threads_per_block_val = ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK)
    {
        attention_bf16_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, qkv_tensor, logsumexp_tensor,
                                      row_dot_tensor, threads_per_block_val);
    }
};

/// Tells if this GPU runs flash attention (an H100); if not, prints why the test that asked is skipped.
static bool flash_runs_here(const char *test_ptr)
{
    if (attention_bf16_flash_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: flash attention needs a GPU of compute capability %d.%d (an H100: WGMMA and TMA)\n",
           HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
    return false;
}

/// Tells if this GPU runs the attention steps (an H100); if not, prints why the test that asked is skipped.
static bool steps_run_here(const char *test_ptr)
{
    if (attention_bf16_fits(GPT2_SEQ_LEN, GPT2_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: the attention steps need a GPU of compute capability %d.%d (an H100: WGMMA and TMA)\n",
           HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
    return false;
}

// ══ part 1: the steps against PyTorch ═════════════════════════════════════════

/// Runs the steps' forward and backward on random bfloat16 qkv and output_grad, and PyTorch in float on the same
/// values; checks that output and attention_probs are within BFLOAT16_TOLERANCE of PyTorch's, and qkv_grad
/// within BFLOAT16_GRAD_TOLERANCE.
static void steps_match_pytorch(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val, size_t head_size_val)
{
    // 1. random qkv and output_grad in bfloat16; mine
    torch::manual_seed(0);
    AttentionTensors mine_val(batch_size_val, num_heads_val, seq_len_val, head_size_val);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape()).to(DType::BFloat16);
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape()).to(DType::BFloat16);
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's, in float on the same values
    PytorchAttention pytorch_val(mine_val);
    // 3. check
    ASSERT_NEAR(mine_val.output_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine_val.attention_probs_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.attention_probs_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine_val.qkv_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.qkv_tensor.grad()),
                BFLOAT16_GRAD_TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: batch_size = 2, 12 heads of 64 as GPT-2 small, seq_len = 64: qkv {2, 64, 2304}, attention_probs
    // {2, 12, 64, 64}, every tensor bfloat16: the six products by attention_product_kernel on the tensor cores
    // (WGMMA, float sums), each a single tile of left and of right brought by TMA; the softmax kernels reading
    // and writing bfloat16. PyTorch is given the same bfloat16 values as floats and works in float: output and
    // attention_probs within BFLOAT16_TOLERANCE, qkv_grad within BFLOAT16_GRAD_TOLERANCE.
    if (steps_run_here(__func__))
        steps_match_pytorch(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE);
}

static void steps_need_whole_wgmmas_and_an_h100(void)
{
    // Expect: attention_bf16_fits says no to a seq_len that is not a multiple of 16 (1000) and to a head_size
    // that is not (40); and yes to 1024 and 64, and to 80 and 32 (which flash attention does not take), exactly
    // when the GPU's compute capability is 9.0 (an H100: the only one with WGMMA and TMA).
    // 1. this GPU's compute capability
    int device_index_val = 0;
    int major_val = 0;
    int minor_val = 0;
    cudaCheck(cudaGetDevice(&device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, device_index_val));
    bool h100_val = 10 * major_val + minor_val == HARDWARE_WGMMA_CAPABILITY;
    // 2. check
    ASSERT_EQ(attention_bf16_fits(1000, GPT2_HEAD_SIZE), false);
    ASSERT_EQ(attention_bf16_fits(GPT2_SEQ_LEN, 40), false);
    ASSERT_EQ(attention_bf16_fits(GPT2_SEQ_LEN, GPT2_HEAD_SIZE), h100_val);
    ASSERT_EQ(attention_bf16_fits(80, 32), h100_val);
}

static void steps_match_on_part_tiles(void)
{
    // Expect: {2, 4 heads of 32, seq_len 80}: qkv {2, 80, 384}, attention_probs {2, 4, 80, 80}. Nothing is a
    // whole tile of 64: a head's 32 values are half a tile's row, so a sum over them is 2 WGMMAs of the tile's 4
    // and the other half of the tile holds the next head's values (or TMA's 0s past value's last head), which no
    // WGMMA reads; the 80 tokens are a tile and a quarter, so the second step of a sum over tokens is 1 WGMMA,
    // and the second square of a product has 16 rows to store, the tensor cores' other 48 left out. Within the
    // bfloat16 tolerances of PyTorch.
    if (steps_run_here(__func__))
        steps_match_pytorch(2, 4, 80, 32);
}

static void steps_match_on_heads_wider_than_a_tile(void)
{
    // Expect: {2, 3 heads of 80, seq_len 208}: qkv {2, 208, 720}, attention_probs {2, 3, 208, 208}. A head's 80
    // values are a tile and a quarter: a sum over them takes 2 steps (4 WGMMAs, then 1), TMA filling the second
    // set of tiles during the first step's WGMMAs, and a product {208, 80} is 2 squares across. The 208 tokens
    // are 3 tiles and a quarter: squares below the diagonal (no mask), sums over tokens of up to 4 steps, both
    // sets of tiles filled twice. Within the bfloat16 tolerances of PyTorch.
    if (steps_run_here(__func__))
        steps_match_pytorch(2, 3, 208, 80);
}

static void steps_backward_writes_over_qkv_grad_the_same_every_time(void)
{
    // Expect: qkv_grad filled with 7s first: after a backward, no 7 is left (every value of its three parts is
    // written), and a second backward gives the same values to the last bit (each value is one block's, and a
    // step's tiles are whole before its WGMMAs read them, whichever set they are in).
    if (!steps_run_here(__func__))
        return;
    // 1. a forward, then a backward onto 7s
    torch::manual_seed(0);
    AttentionTensors mine_val(2, 3, 208, 80);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape()).to(DType::BFloat16);
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape()).to(DType::BFloat16);
    mine_val.forward();
    mine_val.qkv_grad_tensor.fill_(7.0);
    mine_val.backward();
    torch::Tensor first_tensor = pytorch_copy_of(mine_val.qkv_grad_tensor);
    ASSERT_EQ((first_tensor == 7.0f).sum().item<int64_t>(), (int64_t)0);
    // 2. a second backward: the same bits
    mine_val.backward();
    ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.qkv_grad_tensor), first_tensor), true);
}

// ══ part 2: flash attention against PyTorch, and against the attention steps ═════

/// Runs flash attention's forward and backward on random bfloat16 qkv and output_grad, and the attention
/// steps on the same values in bfloat16, and PyTorch in float on them too; checks that flash attention's
/// output and logsumexp are within BFLOAT16_TOLERANCE of PyTorch's, and its qkv_grad within
/// BFLOAT16_GRAD_TOLERANCE, and that it is as close to the steps' bfloat16 results.
static void flash_matches_pytorch_and_the_steps(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
{
    // 1. the steps on random bfloat16 values; flash attention on the same
    torch::manual_seed(0);
    AttentionTensors steps_val(batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE);
    steps_val.qkv_tensor = random_tensor_on_gpu(steps_val.qkv_tensor.shape()).to(DType::BFloat16);
    steps_val.output_grad_tensor = random_tensor_on_gpu(steps_val.output_tensor.shape()).to(DType::BFloat16);
    steps_val.forward();
    steps_val.backward();
    FlashTensors flash_val(batch_size_val, num_heads_val, seq_len_val);
    flash_val.qkv_tensor.copy_(steps_val.qkv_tensor);
    flash_val.output_grad_tensor.copy_(steps_val.output_grad_tensor);
    flash_val.forward();
    flash_val.backward();
    // 2. PyTorch's, in float
    PytorchAttention pytorch_val(steps_val);
    // 3. check against PyTorch, then against the steps
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.output_tensor, pytorch_val.output_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.logsumexp_tensor, pytorch_val.logsumexp_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.qkv_grad_tensor, pytorch_val.qkv_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.output_tensor, pytorch_copy_of(steps_val.output_tensor)), 0.0,
                BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.qkv_grad_tensor, pytorch_copy_of(steps_val.qkv_grad_tensor)), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

static void flash_needs_head_size_64_and_whole_tiles(void)
{
    // Expect: attention_bf16_flash_fits says no to a seq_len that is not a multiple of 64 (1000) and to a
    // head_size other than 64 (32); and yes to 1024 and 64 exactly when the GPU's compute capability is 9.0
    // (an H100: the only one with WGMMA and TMA).
    // 1. this GPU's compute capability
    int device_index_val = 0;
    int major_val = 0;
    int minor_val = 0;
    cudaCheck(cudaGetDevice(&device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, device_index_val));
    // 2. check
    ASSERT_EQ(attention_bf16_flash_fits(1000, FLASH_HEAD_SIZE), false);
    ASSERT_EQ(attention_bf16_flash_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE / 2), false);
    ASSERT_EQ(attention_bf16_flash_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE), 10 * major_val + minor_val == HARDWARE_WGMMA_CAPABILITY);
}

static void flash_matches_on_one_tile(void)
{
    // Expect: {1, 2, 64}: one tile of 64 queries and 64 keys, all on the diagonal, so every score goes through
    // the mask. Within the bfloat16 tolerances of PyTorch and of the steps.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(1, 2, FLASH_TILE);
}

static void flash_matches_on_gpt2_heads_of_256_tokens(void)
{
    // Expect: {2, 12, 256}: 4 tiles per head, so tiles below the diagonal (no mask), the online softmax's
    // max moving from tile to tile, the backward's key blocks walking 4, 3, 2, 1 query tiles. The same.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
}

static void flash_matches_on_gpt2_small_size(void)
{
    // Expect: {4, 12, 1024}, as training runs it: 16 tiles per head, 768 blocks. The same.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
}

static void flash_backward_writes_over_qkv_grad_the_same_every_time(void)
{
    // Expect: qkv_grad filled with 7s first: after a backward, no 7 is left (every value of its three parts
    // is written), and a second backward gives the same values to the last bit (no two blocks add onto one
    // value, so nothing depends on which block finishes first).
    if (!flash_runs_here(__func__))
        return;
    // 1. a forward, then a backward onto 7s
    torch::manual_seed(0);
    FlashTensors flash_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    flash_val.qkv_tensor = random_tensor_on_gpu(flash_val.qkv_tensor.shape()).to(DType::BFloat16);
    flash_val.output_grad_tensor = random_tensor_on_gpu(flash_val.output_tensor.shape()).to(DType::BFloat16);
    flash_val.forward();
    flash_val.qkv_grad_tensor.fill_(7.0);
    flash_val.backward();
    torch::Tensor first_tensor = pytorch_copy_of(flash_val.qkv_grad_tensor);
    ASSERT_EQ((first_tensor == 7.0f).sum().item<int64_t>(), (int64_t)0);
    // 2. a second backward: the same bits
    flash_val.backward();
    ASSERT_EQ(torch::equal(pytorch_copy_of(flash_val.qkv_grad_tensor), first_tensor), true);
}

/// One flash attention backward on these tensors with KEY_GRAD_STAGES and QUERY_GRAD_STAGES sets of tiles: what the
/// check below calls for each pair of stage counts.
///
///   flash_val   the tensors: qkv_grad written, the rest read
template <int KEY_GRAD_STAGES, int QUERY_GRAD_STAGES>
static void flash_backward_at(FlashTensors &flash_val)
{
    attention_bf16_flash_backward<KEY_GRAD_STAGES, QUERY_GRAD_STAGES>(flash_val.qkv_grad_tensor, flash_val.output_grad_tensor,
                                                                      flash_val.output_tensor, flash_val.qkv_tensor,
                                                                      flash_val.logsumexp_tensor, flash_val.row_dot_tensor);
}

/// A pair of stage counts, and the backward made for them.
struct StageSetting
{
    int key_grad_stages_val;
    int query_grad_stages_val;
    void (*backward_ptr)(FlashTensors &);
};

static const StageSetting STAGE_SETTINGS_LIST[] = {
    {1, 1, flash_backward_at<1, 1>}, {1, 2, flash_backward_at<1, 2>}, {1, 3, flash_backward_at<1, 3>}, {1, 4, flash_backward_at<1, 4>},
    {2, 1, flash_backward_at<2, 1>}, {2, 2, flash_backward_at<2, 2>}, {2, 3, flash_backward_at<2, 3>}, {2, 4, flash_backward_at<2, 4>},
    {3, 1, flash_backward_at<3, 1>}, {3, 2, flash_backward_at<3, 2>}, {3, 3, flash_backward_at<3, 3>}, {3, 4, flash_backward_at<3, 4>},
    {4, 1, flash_backward_at<4, 1>}, {4, 2, flash_backward_at<4, 2>}, {4, 3, flash_backward_at<4, 3>}, {4, 4, flash_backward_at<4, 4>},
};

static void every_stage_count_gives_the_same_bits(void)
{
    // Expect: random bfloat16 qkv and output_grad, {2, 12, 1024}: 16 tiles a head, so walks of 1 to 16 steps, longer
    // and shorter than the sets. The backward with key_grad's and query_grad's sets of tiles each 1, 2, 3 and 4
    // gives the default's qkv_grad to the last bit, the 7s it started with all written over: the same tiles, the
    // same sums in the same order; only when TMA brings the tiles changes.
    if (!flash_runs_here(__func__))
        return;
    // 1. random qkv and output_grad; the default's backward
    torch::manual_seed(0);
    FlashTensors flash_val(2, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    flash_val.qkv_tensor = random_tensor_on_gpu(flash_val.qkv_tensor.shape()).to(DType::BFloat16);
    flash_val.output_grad_tensor = random_tensor_on_gpu(flash_val.output_tensor.shape()).to(DType::BFloat16);
    flash_val.forward();
    flash_val.backward();
    torch::Tensor expected_tensor = pytorch_copy_of(flash_val.qkv_grad_tensor);
    print_test_name(__func__);
    // 2. every pair of stage counts against it
    for (const StageSetting &setting_val : STAGE_SETTINGS_LIST)
    {
        flash_val.qkv_grad_tensor.fill_(7.0);
        setting_val.backward_ptr(flash_val);
        printf("    key_grad %d sets, query_grad %d sets:\n", setting_val.key_grad_stages_val, setting_val.query_grad_stages_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(flash_val.qkv_grad_tensor), expected_tensor), true);
    }
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (the softmax's and row_dot's 256; the
    // products' and flash attention's 128, one warpgroup), and nothing is spilled. The products' kernel is
    // made three times: neither side turned (F2, B2), right turned (F5, B7), both turned (B3, B8). The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it
    // wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // Flash attention's key_grad kernel holds the most: key_grad and value_grad (32 sums each), the attention_probs
    // and scores_grad (32 each) and both again packed as A (16 registers each), 160 registers of them, under the
    // 255 a thread may have. Its keys, values, queries and output_grad stay in shared memory.
    const int kernel_count_val = 9;
    const void *kernels_list[kernel_count_val] = {(const void *)attention_product_kernel<false, false>,
                                                  (const void *)attention_product_kernel<false, true>,
                                                  (const void *)attention_product_kernel<true, true>,
                                                  (const void *)attention_bf16_softmax_forward_kernel,
                                                  (const void *)attention_bf16_softmax_backward_kernel,
                                                  (const void *)attention_bf16_flash_forward_kernel,
                                                  (const void *)attention_bf16_flash_row_dot_kernel,
                                                  (const void *)attention_bf16_flash_key_grad_kernel<ATTENTION_FLASH_BF16_KEY_GRAD_STAGES>,
                                                  (const void *)attention_bf16_flash_query_grad_kernel<ATTENTION_FLASH_BF16_QUERY_GRAD_STAGES>};
    const char *names_list[kernel_count_val] = {
        "attention_product_kernel<false, false>", "attention_product_kernel<false, true>",
        "attention_product_kernel<true, true>", "attention_bf16_softmax_forward_kernel",
        "attention_bf16_softmax_backward_kernel", "attention_bf16_flash_forward_kernel",
        "attention_bf16_flash_row_dot_kernel", "attention_bf16_flash_key_grad_kernel",
        "attention_bf16_flash_query_grad_kernel"};
    int threads_list[kernel_count_val] = {ATTENTION_PRODUCT_THREADS_PER_BLOCK, ATTENTION_PRODUCT_THREADS_PER_BLOCK,
                                          ATTENTION_PRODUCT_THREADS_PER_BLOCK, ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK,
                                          ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK, FLASH_THREADS,
                                          ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK, FLASH_THREADS, FLASH_THREADS};
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: random bfloat16 qkv and output_grad, {2, 12, 64, 64}. attention_bf16_forward and
    // attention_bf16_backward with the softmax's block size set to 32, 128, 512 and 1024 (one warp up to
    // HARDWARE_MAX_THREADS_PER_BLOCK) give the defaults' output, attention_probs and qkv_grad to the last bit: each softmax row
    // is one warp's whatever the block, and the products' block is always ATTENTION_PRODUCT_THREADS_PER_BLOCK. On an H100, flash attention's backward with row_dot at those block
    // sizes gives the default's qkv_grad to the last bit too: each row_dot is one warp's.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    if (!steps_run_here(__func__))
        return;
    // 1. random qkv and output_grad; the defaults' results
    torch::manual_seed(0);
    AttentionTensors usual_val(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE);
    usual_val.qkv_tensor = random_tensor_on_gpu(usual_val.qkv_tensor.shape()).to(DType::BFloat16);
    usual_val.output_grad_tensor = random_tensor_on_gpu(usual_val.output_tensor.shape()).to(DType::BFloat16);
    usual_val.forward();
    usual_val.backward();

    // 2. each block size, from the same qkv and output_grad, against the defaults'
    for (int threads_per_block_val : block_sizes_list)
    {
        AttentionTensors tensors_val(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE);
        tensors_val.qkv_tensor.copy_(usual_val.qkv_tensor);
        tensors_val.output_grad_tensor.copy_(usual_val.output_grad_tensor);
        tensors_val.forward(threads_per_block_val);
        tensors_val.backward(threads_per_block_val);
        ASSERT_EQ(tensors_val.output_tensor.to(DType::Float32).tolist(), usual_val.output_tensor.to(DType::Float32).tolist());
        ASSERT_EQ(tensors_val.attention_probs_tensor.to(DType::Float32).tolist(), usual_val.attention_probs_tensor.to(DType::Float32).tolist());
        ASSERT_EQ(tensors_val.qkv_grad_tensor.to(DType::Float32).tolist(),
                  usual_val.qkv_grad_tensor.to(DType::Float32).tolist());
    }

    // 3. flash attention's row_dot block size (an H100 too; the same GPU as the steps')
    if (!flash_runs_here(__func__))
        return;
    torch::manual_seed(0);
    FlashTensors flash_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    flash_val.qkv_tensor = random_tensor_on_gpu(flash_val.qkv_tensor.shape()).to(DType::BFloat16);
    flash_val.output_grad_tensor = random_tensor_on_gpu(flash_val.output_tensor.shape()).to(DType::BFloat16);
    flash_val.forward();
    flash_val.backward();
    torch::Tensor usual_qkv_grad_tensor = pytorch_copy_of(flash_val.qkv_grad_tensor);
    for (int threads_per_block_val : block_sizes_list)
    {
        flash_val.qkv_grad_tensor.zero_();
        flash_val.backward(threads_per_block_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(flash_val.qkv_grad_tensor), usual_qkv_grad_tensor), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_and_backward_in_bfloat16();
    steps_need_whole_wgmmas_and_an_h100();
    steps_match_on_part_tiles();
    steps_match_on_heads_wider_than_a_tile();
    steps_backward_writes_over_qkv_grad_the_same_every_time();

    flash_needs_head_size_64_and_whole_tiles();
    flash_matches_on_one_tile();
    flash_matches_on_gpt2_heads_of_256_tokens();
    flash_matches_on_gpt2_small_size();
    flash_backward_writes_over_qkv_grad_the_same_every_time();
    every_stage_count_gives_the_same_bits();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
