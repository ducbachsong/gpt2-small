// test_attention.cu — tests for llmc/kernels/attention.cuh (attention_forward, attention_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's scaled_dot_product_attention (causal), with autograd for the
// backward, and the weights against PyTorch's softmax of the masked scores; part 1b checks flash attention
// (bfloat16, sm_80 or later) against PyTorch and against the attention steps in bfloat16; part 2 checks
// numbers worked out by hand and what the mask must give; part 3 checks that each kernel's registers fit its
// block, and other block sizes. The timings are in tunes/tune_attention.cu.
// (The C# trainer had no attention test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/attention.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values. A bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it, and attention rounds several times in a row:
// the scores before the softmax (a score of 4 by up to 0.008, which moves its weight by 0.8%), the weights,
// the outputs. The backward rounds da, ds and the three gradients on top, and ds = a (da - row_dot) takes
// two close numbers apart, so its errors are larger next to its own size.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_HEAD_SIZE 64             // and their columns
#define GPT2_SEQ_LEN 1024             // its tokens per row

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
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
/// worked out on the GPU: GPT-2's weights are 50M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_forward and attention_backward take, for one size, made with 0s except qkv and
/// output_grad, which the test fills.
struct AttentionTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor weights_tensor;
    Tensor head_outputs_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor head_outputs_grad_tensor;
    Tensor weights_grad_tensor;
    Tensor query_grad_tensor;
    Tensor key_grad_tensor;
    Tensor value_grad_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens and head_size columns, all of dtype.
    AttentionTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val, size_t head_size_val,
                     DType dtype_val = DType::Float32)
    {
        size_t embed_dim_val = num_heads_val * head_size_val;
        Shape per_head_list = {batch_size_val, num_heads_val, seq_len_val, head_size_val};
        Shape per_pair_list = {batch_size_val, num_heads_val, seq_len_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, dtype_val);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, dtype_val);
        query_tensor = Tensor::zeros(per_head_list, dtype_val), key_tensor = Tensor::zeros(per_head_list, dtype_val);
        value_tensor = Tensor::zeros(per_head_list, dtype_val);
        weights_tensor = Tensor::zeros(per_pair_list, dtype_val), head_outputs_tensor = Tensor::zeros(per_head_list, dtype_val);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), dtype_val);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), dtype_val);
        head_outputs_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
        weights_grad_tensor = Tensor::zeros(per_pair_list, dtype_val);
        query_grad_tensor = Tensor::zeros(per_head_list, dtype_val), key_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
        value_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
    }

    /// Runs attention_forward on these tensors, at these block sizes (the defaults if not given).
    void forward(int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                 int softmax_threads_per_block_val = ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK)
    {
        attention_forward(output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor, head_outputs_tensor,
                          qkv_tensor, threads_per_block_val, softmax_threads_per_block_val);
    }

    /// Runs attention_backward on these tensors, at these block sizes (the defaults if not given).
    void backward(int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                  int softmax_threads_per_block_val = ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK)
    {
        attention_backward(qkv_grad_tensor, output_grad_tensor, query_tensor, key_tensor, value_tensor, weights_tensor,
                           head_outputs_grad_tensor, weights_grad_tensor, query_grad_tensor, key_grad_tensor,
                           value_grad_tensor, threads_per_block_val, softmax_threads_per_block_val);
    }
};

/// PyTorch's answers for one qkv and output_grad, the plain PyTorch way, as GPT-2's model.py does it: qkv
/// split into q, k, v, each cut into heads, then scaled_dot_product_attention with is_causal, and autograd.
/// weights is worked out on the side as GPT-2's original code does: softmax of the masked scores.
struct PytorchAttention
{
    torch::Tensor qkv_tensor;
    torch::Tensor output_tensor;
    torch::Tensor weights_tensor;
    torch::Tensor logsumexp_tensor;                                // logsumexp of each row of masked scores

    /// Works out output, qkv.grad() and weights from my qkv and output_grad.
    PytorchAttention(const AttentionTensors &mine_val)
    {
        // 1. the sizes
        int64_t batch_size_val = mine_val.query_tensor.size(0);
        int64_t num_heads_val = mine_val.query_tensor.size(1);
        int64_t seq_len_val = mine_val.query_tensor.size(2);
        int64_t head_size_val = mine_val.query_tensor.size(3);
        int64_t embed_dim_val = num_heads_val * head_size_val;
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
        // 3. the weights and logsumexp, on the side
        torch::NoGradGuard no_grad_val;
        torch::Tensor scores_tensor = q_tensor.matmul(k_tensor.transpose(-2, -1)) / std::sqrt((double)head_size_val);
        torch::Tensor after_tensor = torch::ones({seq_len_val, seq_len_val}, scores_tensor.options().dtype(torch::kBool)).triu(1);
        weights_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).softmax(-1);
        logsumexp_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).logsumexp(-1);
    }
};

/// Every tensor attention_flash_forward and attention_flash_backward take, for one size, all bfloat16 but
/// logsumexp and row_dot (float), made with 0s except qkv and output_grad, which the test fills.
struct FlashTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor logsumexp_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor row_dot_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        Shape per_head_list = {batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        query_tensor = Tensor::zeros(per_head_list, DType::BFloat16), key_tensor = Tensor::zeros(per_head_list, DType::BFloat16);
        value_tensor = Tensor::zeros(per_head_list, DType::BFloat16);
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_flash_forward on these tensors.
    void forward()
    {
        attention_flash_forward(output_tensor, query_tensor, key_tensor, value_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_flash_backward on these tensors, row_dot at this block size (the default if not given).
    void backward(int threads_per_block_val = ATTENTION_ROW_DOT_THREADS_PER_BLOCK)
    {
        attention_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, query_tensor, key_tensor, value_tensor,
                                 logsumexp_tensor, row_dot_tensor, threads_per_block_val);
    }
};

/// Tells if this GPU runs flash attention (sm_80 or later); if not, prints why the test that asked is skipped.
static bool flash_runs_here(const char *test_ptr)
{
    if (attention_flash_fits(DType::BFloat16, GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: flash attention needs a GPU of compute capability 8.0 or more (A100, H100)\n");
    return false;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random qkv and output_grad, mine and PyTorch's, and checks that
/// output, weights and qkv_grad agree within TOLERANCE.
static void forward_and_backward_match_pytorch(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val,
                                               size_t head_size_val)
{
    // 1. random qkv and output_grad; mine
    torch::manual_seed(0);
    AttentionTensors mine_val(batch_size_val, num_heads_val, seq_len_val, head_size_val);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape());
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape());
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's
    PytorchAttention pytorch_val(mine_val);
    // 3. check
    ASSERT_NEAR(mine_val.output_tensor.tolist(), list_of(pytorch_val.output_tensor), TOLERANCE);
    ASSERT_NEAR(mine_val.weights_tensor.tolist(), list_of(pytorch_val.weights_tensor), TOLERANCE);
    ASSERT_NEAR(mine_val.qkv_grad_tensor.tolist(), list_of(pytorch_val.qkv_tensor.grad()), TOLERANCE);
}

static void matches_pytorch_on_gpt2_heads(void)
{
    // Expect: batch_size = 2, 12 heads of 64 as GPT-2 small, seq_len = 64 tokens: qkv {2, 64, 2304}, weights
    // {2, 12, 64, 64}. output, weights and qkv_grad equal PyTorch's within 1e-5.
    forward_and_backward_match_pytorch(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE);
}

static void matches_pytorch_with_a_seq_len_of_37(void)
{
    // Expect: {3, 2, 37, 8}: rows of 37 weights, so in each warp's row, lane 5 has 2 columns and lane 6 one;
    // tokens 0 .. 30 leave some lanes with no column at all, which must add nothing. The same as PyTorch
    // within 1e-5.
    forward_and_backward_match_pytorch(3, 2, 37, 8);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_heads's {2, 12, 64, 64} with every tensor bfloat16: the six products
    // by cublasGemmStridedBatchedEx (CUDA_R_16BF, float sums), the kernels reading and writing bfloat16.
    // PyTorch is given the same bfloat16 values as floats and works in float: output and weights within
    // BFLOAT16_TOLERANCE, qkv_grad within BFLOAT16_GRAD_TOLERANCE.
    // 1. random qkv and output_grad in bfloat16; mine
    torch::manual_seed(0);
    AttentionTensors mine_val(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE, DType::BFloat16);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape()).to(DType::BFloat16);
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape()).to(DType::BFloat16);
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's, in float on the same values
    PytorchAttention pytorch_val(mine_val);
    // 3. check
    ASSERT_NEAR(mine_val.output_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine_val.weights_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.weights_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine_val.qkv_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_val.qkv_tensor.grad()),
                BFLOAT16_GRAD_TOLERANCE);
}

// ══ part 1b: flash attention against PyTorch, and against the attention steps ════

/// Runs flash attention's forward and backward on random bfloat16 qkv and output_grad, and the attention
/// steps on the same values in bfloat16, and PyTorch in float on them too; checks that flash attention's
/// output and logsumexp are within BFLOAT16_TOLERANCE of PyTorch's, and its qkv_grad within
/// BFLOAT16_GRAD_TOLERANCE, and that it is as close to the steps' bfloat16 results.
static void flash_matches_pytorch_and_the_steps(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
{
    // 1. the steps on random bfloat16 values; flash attention on the same
    torch::manual_seed(0);
    AttentionTensors steps_val(batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE, DType::BFloat16);
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

static void flash_needs_bfloat16_head_size_64_and_whole_tiles(void)
{
    // Expect: attention_flash_fits says no to float, to a seq_len that is not a multiple of 64 (1000) and to a
    // head_size other than 64 (32); and yes to bfloat16, 1024 and 64 exactly when the GPU is sm_80 or later.
    // 1. this GPU's compute capability
    int device_val = 0;
    int major_val = 0;
    cudaCheck(cudaGetDevice(&device_val));
    cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_val));
    // 2. check
    ASSERT_EQ(attention_flash_fits(DType::Float32, GPT2_SEQ_LEN, FLASH_HEAD_SIZE), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, 1000, FLASH_HEAD_SIZE), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, GPT2_SEQ_LEN, FLASH_HEAD_SIZE / 2), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, GPT2_SEQ_LEN, FLASH_HEAD_SIZE), major_val >= 8);
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

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example: one head of 3 tokens with head_size 4, so qkv is {1, 3, 12}, each
// row q | k | v. scale = 1 / sqrt(4) = 0.5.
//
//     token   q            k            v            scores (0.5 q · k), columns 0 .. token   weights
//     0       2 0 0 0      1 0 0 0      1 0 0 0      1                                        1
//     1       0 2 0 0      0 1 0 0      0 1 0 0      0  1                                     0.2689414  0.7310586
//     2       2 2 0 0      1 1 0 0      0 0 1 0      1  1  2                                  0.2119416  0.2119416  0.5761169
//
// Token 1: e^0 / (e^0 + e^1) = 1 / 3.7182818 = 0.2689414. Each v has one 1, so output row t is weights row t
// followed by 0s. With output_grad 1 0 0 0 in every row (the loss wants output column 0 up: the weight on
// token 0):
//
//     da = output_grad · vᵀ        1 0 0 in every row
//     ds = a (da - row_dot)        token 1: 0.2689414 x (1 - 0.2689414) = 0.1966119, and -0.1966119
//                                  token 2: 0.1670223  -0.0449192  -0.1221031         (token 0: 0)
//     dq = 0.5 ds · k,  dk = 0.5 dsᵀ · q,  dv = aᵀ · output_grad

static const std::vector<float> NOTE_QKV = {2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                            0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                            2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0};
static const std::vector<float> NOTE_WEIGHTS = {1, 0, 0,
                                                0.2689414f, 0.7310586f, 0,
                                                0.2119416f, 0.2119416f, 0.5761169f};

/// Makes the note's example: one head of 3 tokens, head_size 4, qkv as above, output_grad 1 0 0 0 per row.
static AttentionTensors note_example(void)
{
    AttentionTensors note_val(1, 1, 3, 4);
    note_val.qkv_tensor = tensor_of({1, 3, 12}, NOTE_QKV);
    note_val.output_grad_tensor = tensor_of({1, 3, 4}, {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0});
    return note_val;
}

static void forward_of_the_note_s_example(void)
{
    // Expect: the weights above, and output row t = weights row t with a 0 after it, within 1e-6.
    // 1. the forward
    AttentionTensors note_val = note_example();
    note_val.forward();
    // 2. check
    ASSERT_NEAR(note_val.weights_tensor.tolist(), NOTE_WEIGHTS, 1e-6);
    ASSERT_NEAR(note_val.output_tensor.tolist(), std::vector<float>({1, 0, 0, 0,
                                                                     0.2689414f, 0.7310586f, 0, 0,
                                                                     0.2119416f, 0.2119416f, 0.5761169f, 0}), 1e-6);
}

static void backward_of_the_note_s_example(void)
{
    // Expect: qkv_grad row t = dq | dk | dv of token t, worked out above:
    //
    //     token   dq                          dk                          dv
    //     0       0          0          0 0   0.1670223  0.3636343  0 0   1.4808830  0 0 0
    //     1       0.0983060  -0.0983060 0 0   -0.0449192 -0.2415312 0 0   0.9430001  0 0 0
    //     2       0.0224596  -0.0835112 0 0   -0.1221031 -0.1221031 0 0   0.5761169  0 0 0
    //
    // dv of token 0 is 1 + 0.2689414 + 0.2119416: every token's weight on token 0. And weights_grad ends
    // holding ds. Within 1e-6.
    // 1. the forward, then the backward
    AttentionTensors note_val = note_example();
    note_val.forward();
    note_val.backward();
    // 2. check
    ASSERT_NEAR(note_val.qkv_grad_tensor.tolist(),
                std::vector<float>({0, 0, 0, 0, 0.1670223f, 0.3636343f, 0, 0, 1.4808830f, 0, 0, 0,
                                    0.0983060f, -0.0983060f, 0, 0, -0.0449192f, -0.2415312f, 0, 0, 0.9430001f, 0, 0, 0,
                                    0.0224596f, -0.0835112f, 0, 0, -0.1221031f, -0.1221031f, 0, 0, 0.5761169f, 0, 0, 0}),
                1e-6);
    ASSERT_NEAR(note_val.weights_grad_tensor.tolist(), std::vector<float>({0, 0, 0,
                                                                           0.1966119f, -0.1966119f, 0,
                                                                           0.1670223f, -0.0449192f, -0.1221031f}), 1e-6);
}

static void weights_after_the_token_are_0_and_each_row_adds_up_to_1(void)
{
    // Expect: random qkv, {1, 2, 8, 4}: in every row t of weights, columns t + 1 .. 7 are exactly 0 (F3),
    // and the row adds up to 1 within 1e-6 (F4).
    // 1. the forward on random qkv
    torch::manual_seed(1);
    AttentionTensors tensors_val(1, 2, 8, 4);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.forward();
    // 2. the weights after each token, and each row's sum
    std::vector<float> weights_list = tensors_val.weights_tensor.tolist();
    std::vector<float> after_the_token_list;
    std::vector<float> row_sums_list;
    for (size_t row_index_val = 0; row_index_val < 2 * 8; row_index_val++)
    {
        size_t token_val = row_index_val % 8;
        float row_sum_val = 0.0f;
        for (size_t column_val = 0; column_val < 8; column_val++)
        {
            float weight_val = weights_list[row_index_val * 8 + column_val];
            if (column_val > token_val)
                after_the_token_list.push_back(weight_val);
            row_sum_val += weight_val;
        }
        row_sums_list.push_back(row_sum_val);
    }
    // 3. check
    ASSERT_EQ(after_the_token_list, std::vector<float>(after_the_token_list.size(), 0.0f));
    ASSERT_NEAR(row_sums_list, std::vector<float>(16, 1.0f), 1e-6);
}

static void the_first_token_sees_only_itself(void)
{
    // Expect: token 0 sees column 0 only, so its weight is exp(0) / exp(0) = 1 exactly, and its output is its
    // own v, exactly: 1 x v + 0 x the others. Random qkv, {1, 2, 8, 4}: output row 0 = v of token 0 for both
    // heads, qkv row 0's last 8 values.
    // 1. the forward on random qkv
    torch::manual_seed(2);
    AttentionTensors tensors_val(1, 2, 8, 4);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.forward();
    // 2. check output row 0 against qkv row 0's v
    std::vector<float> qkv_list = tensors_val.qkv_tensor.tolist();
    std::vector<float> output_list = tensors_val.output_tensor.tolist();
    ASSERT_EQ(std::vector<float>(output_list.begin(), output_list.begin() + 8),
              std::vector<float>(qkv_list.begin() + 16, qkv_list.begin() + 24));
}

static void a_later_token_does_not_change_earlier_outputs(void)
{
    // Expect: random qkv, {1, 2, 8, 8}; then token 7's whole qkv row is set to 100 and the forward run again.
    // Tokens 0 .. 6 cannot see token 7, so their output rows are the same to the last bit: token 7's scores
    // are in their rows only after them, where the weights are 0, and 0 x 100 adds 0.
    // 1. the forward on random qkv
    torch::manual_seed(3);
    AttentionTensors tensors_val(1, 2, 8, 8);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.forward();
    std::vector<float> before_list = tensors_val.output_tensor.tolist();
    // 2. token 7 set to 100, the forward again
    tensors_val.qkv_tensor[0][7].fill_(100);
    tensors_val.forward();
    std::vector<float> after_list = tensors_val.output_tensor.tolist();
    // 3. check the first 7 rows
    size_t first_7_rows_val = 7 * 16;
    ASSERT_EQ(std::vector<float>(after_list.begin(), after_list.begin() + first_7_rows_val),
              std::vector<float>(before_list.begin(), before_list.begin() + first_7_rows_val));
}

static void equal_scores_give_the_mean_of_the_values(void)
{
    // Expect: q = 0, so every score is 0 and token t gives each of its t + 1 columns 1 / (t + 1): its output is
    // the mean of v over tokens 0 .. t. With v of token t = t t t t, output row t is the mean of 0 .. t = t / 2:
    // 0, 0.5, 1, 1.5 in all 4 columns, within 1e-6. One head of 4 tokens, head_size 4.
    // 1. qkv: q and k 0, v of token t all t
    AttentionTensors tensors_val(1, 1, 4, 4);
    std::vector<float> qkv_list(4 * 12, 0.0f);
    for (int token_val = 0; token_val < 4; token_val++)
        for (int column_val = 8; column_val < 12; column_val++)
            qkv_list[token_val * 12 + column_val] = (float)token_val;
    tensors_val.qkv_tensor = tensor_of({1, 4, 12}, qkv_list);
    // 2. the forward
    tensors_val.forward();
    // 3. check
    ASSERT_NEAR(tensors_val.output_tensor.tolist(), std::vector<float>({0, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f,
                                                                        1, 1, 1, 1, 1.5f, 1.5f, 1.5f, 1.5f}), 1e-6);
}

static void large_scores_do_not_overflow(void)
{
    // Expect: q = 1000 0 0 0 for every token, k of token j = j 0 0 0, so token 2's scores are 0.5 x 1000 x j =
    // 0, 500, 1000. exp(1000) is infinity in a float, but after taking the row's max, exp(-1000) and exp(-500)
    // are 0 and exp(0) is 1: weights 0 0 1, exactly. Each token puts all its weight on itself, and its output
    // is its own v = (token + 1) 0 0 0: 1, 2, 3 in column 0, exactly.
    // 1. qkv as above
    std::vector<float> qkv_list(3 * 12, 0.0f);
    for (int token_val = 0; token_val < 3; token_val++)
    {
        qkv_list[token_val * 12 + 0] = 1000;
        qkv_list[token_val * 12 + 4] = (float)token_val;
        qkv_list[token_val * 12 + 8] = (float)(token_val + 1);
    }
    // 2. the forward
    AttentionTensors tensors_val(1, 1, 3, 4);
    tensors_val.qkv_tensor = tensor_of({1, 3, 12}, qkv_list);
    tensors_val.forward();
    // 3. check
    ASSERT_EQ(tensors_val.weights_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 1, 0, 0, 0, 1}));
    ASSERT_EQ(tensors_val.output_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}));
}

static void heads_do_not_mix(void)
{
    // Expect: random qkv, {1, 2, 8, 4}; then head 1's columns of q, k and v are set to 100 for every token
    // and the forward run again. Head 0's half of every output row (its first 4 columns) is the same to the
    // last bit: each head's scores, weights and mix use its own columns only.
    // 1. the forward on random qkv
    torch::manual_seed(4);
    AttentionTensors tensors_val(1, 2, 8, 4);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.forward();
    std::vector<float> before_list = tensors_val.output_tensor.tolist();
    // 2. head 1's columns set to 100, the forward again
    std::vector<float> qkv_list = tensors_val.qkv_tensor.tolist();
    for (size_t token_val = 0; token_val < 8; token_val++)
        for (size_t part_val = 0; part_val < 3; part_val++)          // q, k, v: 8 columns each, head 1 the last 4
            for (size_t column_val = 4; column_val < 8; column_val++)
                qkv_list[token_val * 24 + part_val * 8 + column_val] = 100;
    tensors_val.qkv_tensor.copy_from_cpu(qkv_list.data());
    tensors_val.forward();
    std::vector<float> after_list = tensors_val.output_tensor.tolist();
    // 3. head 0's columns of every row, before and after
    std::vector<float> head_0_before_list;
    std::vector<float> head_0_after_list;
    for (size_t token_val = 0; token_val < 8; token_val++)
        for (size_t column_val = 0; column_val < 4; column_val++)
        {
            head_0_before_list.push_back(before_list[token_val * 8 + column_val]);
            head_0_after_list.push_back(after_list[token_val * 8 + column_val]);
        }
    // 4. check
    ASSERT_EQ(head_0_after_list, head_0_before_list);
}

static void a_gradient_on_the_first_token_reaches_only_its_v(void)
{
    // Expect: random qkv, {1, 2, 8, 4}, and output_grad random in row 0, 0 in the others. Output row 0 is
    // 1 x v of token 0 (the_first_token_sees_only_itself), so the only gradient is v's of token 0, which is
    // output_grad row 0 itself; q and k of token 0 get ds = 1 x (da - 1 x da) = 0, and every other token 0,
    // all exactly.
    // 1. random qkv; output_grad random in row 0 only
    torch::manual_seed(5);
    AttentionTensors tensors_val(1, 2, 8, 4);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    std::vector<float> output_grad_list(8 * 8, 0.0f), row_0_list = Tensor::randn({8}).tolist();
    std::copy(row_0_list.begin(), row_0_list.end(), output_grad_list.begin());
    tensors_val.output_grad_tensor = tensor_of({1, 8, 8}, output_grad_list);
    // 2. the forward, then the backward
    tensors_val.forward();
    tensors_val.backward();
    // 3. check: row 0's v gets output_grad row 0, everything else 0
    std::vector<float> expected_list(8 * 24, 0.0f);                // row 0: q 0, k 0, v = output_grad row 0
    std::copy(row_0_list.begin(), row_0_list.end(), expected_list.begin() + 16);
    ASSERT_EQ(tensors_val.qkv_grad_tensor.tolist(), expected_list);
}

static void each_row_of_scores_grad_adds_up_to_0(void)
{
    // Expect: random qkv and output_grad, {1, 2, 8, 4}. After the backward, weights_grad holds ds, and each
    // row adds up to 0 within 1e-6: adding one number to every score of a row changes no weight (note 5).
    // The columns after each token are exactly 0.
    // 1. the forward, then the backward
    torch::manual_seed(6);
    AttentionTensors tensors_val(1, 2, 8, 4);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.output_grad_tensor = random_tensor_on_gpu(tensors_val.output_tensor.shape());
    tensors_val.forward();
    tensors_val.backward();
    // 2. each row's sum, and the values after each token
    std::vector<float> scores_grad_list = tensors_val.weights_grad_tensor.tolist();
    std::vector<float> row_sums_list;
    std::vector<float> after_the_token_list;
    for (size_t row_index_val = 0; row_index_val < 16; row_index_val++)
    {
        float row_sum_val = 0.0f;
        for (size_t column_val = 0; column_val < 8; column_val++)
        {
            row_sum_val += scores_grad_list[row_index_val * 8 + column_val];
            if (column_val > row_index_val % 8)
                after_the_token_list.push_back(scores_grad_list[row_index_val * 8 + column_val]);
        }
        row_sums_list.push_back(row_sum_val);
    }
    // 3. check
    ASSERT_NEAR(row_sums_list, std::vector<float>(16, 0.0f), 1e-6);
    ASSERT_EQ(after_the_token_list, std::vector<float>(after_the_token_list.size(), 0.0f));
}

static void backward_writes_over_qkv_grad(void)
{
    // Expect: qkv_grad starts at 10 (left from somewhere else); the backward writes over it, so after it
    // qkv_grad is the note's, with no 10 added: v of token 0 gets 1.4808830, not 11.4808830.
    // 1. qkv_grad of 10s; the forward, then the backward
    AttentionTensors note_val = note_example();
    note_val.qkv_grad_tensor.fill_(10);
    note_val.forward();
    note_val.backward();
    // 2. check
    ASSERT_NEAR(note_val.qkv_grad_tensor[0][0].tolist()[8], 1.4808830, 1e-6);
    ASSERT_NEAR(note_val.qkv_grad_tensor[0][1].tolist()[0], 0.0983060, 1e-6);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (256; flash attention's 128), and nothing
    // is spilled. The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it
    // wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // Flash attention's key_grad kernel holds the most: its 16 keys and values as tiles, key_grad, value_grad,
    // the weights and scores_grad, 160 registers of them, under the 255 a thread may have. On a GPU older than
    // sm_80 the flash kernels are empty, and fit trivially.
    const void *kernels_list[] = {(const void *)attention_split_qkv_kernel<float>,
                                  (const void *)attention_softmax_forward_kernel<float>,
                                  (const void *)attention_merge_heads_kernel<float>,
                                  (const void *)attention_split_heads_kernel<float>,
                                  (const void *)attention_softmax_backward_kernel<float>,
                                  (const void *)attention_merge_qkv_grad_kernel<float>,
                                  (const void *)attention_flash_forward_kernel,
                                  (const void *)attention_flash_row_dot_kernel,
                                  (const void *)attention_flash_key_grad_kernel,
                                  (const void *)attention_flash_query_grad_kernel};
    const char *names_list[] = {"attention_split_qkv_kernel", "attention_softmax_forward_kernel",
                                "attention_merge_heads_kernel", "attention_split_heads_kernel",
                                "attention_softmax_backward_kernel", "attention_merge_qkv_grad_kernel",
                                "attention_flash_forward_kernel", "attention_flash_row_dot_kernel",
                                "attention_flash_key_grad_kernel", "attention_flash_query_grad_kernel"};
    int threads_list[] = {ATTENTION_THREADS_PER_BLOCK, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK,
                          ATTENTION_THREADS_PER_BLOCK, ATTENTION_THREADS_PER_BLOCK,
                          ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK, ATTENTION_THREADS_PER_BLOCK,
                          FLASH_THREADS, ATTENTION_ROW_DOT_THREADS_PER_BLOCK, FLASH_THREADS, FLASH_THREADS};
    for (int kernel_index_val = 0; kernel_index_val < 10; kernel_index_val++)
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
    // Expect: random qkv and output_grad, {2, 12, 64, 64}, in float and in bfloat16. attention_forward and
    // attention_backward with both block sizes (the copies' and the softmax's) set to 32, 128, 512 and 1024
    // (one warp up to MAX_THREADS_PER_BLOCK) give the defaults' output, weights and qkv_grad to the last bit:
    // the copies move each float4 alone, each softmax row is one warp's whatever the block, and the cuBLAS
    // products are the same calls. On an sm_80 GPU, flash attention's backward with row_dot at those block
    // sizes gives the default's qkv_grad to the last bit too: each row_dot is one warp's.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random qkv and output_grad; the defaults' results
        torch::manual_seed(0);
        AttentionTensors usual_val(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE, dtype_val);
        usual_val.qkv_tensor = random_tensor_on_gpu(usual_val.qkv_tensor.shape()).to(dtype_val);
        usual_val.output_grad_tensor = random_tensor_on_gpu(usual_val.output_tensor.shape()).to(dtype_val);
        usual_val.forward();
        usual_val.backward();

        // 2. each block size, from the same qkv and output_grad, against the defaults'
        for (int threads_per_block_val : block_sizes_list)
        {
            AttentionTensors tensors_val(2, GPT2_NUM_HEADS, 64, GPT2_HEAD_SIZE, dtype_val);
            tensors_val.qkv_tensor.copy_(usual_val.qkv_tensor);
            tensors_val.output_grad_tensor.copy_(usual_val.output_grad_tensor);
            tensors_val.forward(threads_per_block_val, threads_per_block_val);
            tensors_val.backward(threads_per_block_val, threads_per_block_val);
            ASSERT_EQ(tensors_val.output_tensor.to(DType::Float32).tolist(), usual_val.output_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(tensors_val.weights_tensor.to(DType::Float32).tolist(), usual_val.weights_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(tensors_val.qkv_grad_tensor.to(DType::Float32).tolist(),
                      usual_val.qkv_grad_tensor.to(DType::Float32).tolist());
        }
    }

    // 3. flash attention's row_dot block size, on an sm_80 GPU
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
    matches_pytorch_on_gpt2_heads();
    matches_pytorch_with_a_seq_len_of_37();
    forward_and_backward_in_bfloat16();

    flash_needs_bfloat16_head_size_64_and_whole_tiles();
    flash_matches_on_one_tile();
    flash_matches_on_gpt2_heads_of_256_tokens();
    flash_matches_on_gpt2_small_size();
    flash_backward_writes_over_qkv_grad_the_same_every_time();

    forward_of_the_note_s_example();
    backward_of_the_note_s_example();
    weights_after_the_token_are_0_and_each_row_adds_up_to_1();
    the_first_token_sees_only_itself();
    a_later_token_does_not_change_earlier_outputs();
    equal_scores_give_the_mean_of_the_values();
    large_scores_do_not_overflow();
    heads_do_not_mix();
    a_gradient_on_the_first_token_reaches_only_its_v();
    each_row_of_scores_grad_adds_up_to_0();
    backward_writes_over_qkv_grad();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
