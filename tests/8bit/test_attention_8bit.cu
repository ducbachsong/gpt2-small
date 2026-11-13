// test_attention_8bit.cu — tests for llmc/kernels/8bit/attention_8bit.cuh and attention_flash_8bit.cuh (attention_8bit_forward,
// attention_8bit_backward, attention_8bit_turn_back): flash attention with query, key and value kept in E4M3.
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
// Every test needs an H100 (compute capability 9.0: WGMMA and TMA) and is skipped on others. Built with
// -arch=sm_90a.
//
// Part 1 checks the rounding: each head row's scale, and the rounded values against PyTorch's own float8_e4m3fn;
// part 2 checks attention against PyTorch in float, on the rounded values (only bfloat16's roundings apart)
// and on the values before rounding (what 8 bits cost); part 3 checks that the backward writes over qkv_grad,
// that each kernel's registers fit its block, and other block sizes. The timings are in
// tests/8bit/tune_attention_8bit.cu.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/8bit/attention_flash_8bit.cuh"     // and through it attention_8bit.cuh, attention_flash_bf16.cuh

// Against PyTorch in float on the same rounded values: only bfloat16's roundings are left (the attention_probs before
// attention_probs · value, the output, the backward's), so attention_bf16.cuh's tolerances. Each is the largest
// difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;
// Against PyTorch in float on the values before rounding. E4M3 keeps 4 significant bits, so each value moves by
// up to 2^-4, about 6%: the largest output (token 0's, its own value) by up to 6% of itself, and a score, a sum of
// 64 products each off by about 5%, by about 0.05, which moves its attention_prob by 5%. The gradients add the backward's
// bfloat16 roundings on top.
static const double EIGHT_BIT_TOLERANCE = 1e-1;
static const double EIGHT_BIT_GRAD_TOLERANCE = 1.5e-1;

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
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

/// Makes a PyTorch float tensor on the GPU with a copy of my float or bfloat16 tensor's values, GPU to GPU. A
/// bfloat16 tensor's values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);  // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives a tensor's bytes, copied to the CPU, of any dtype: to see that two tensors hold the same bits.
static std::vector<uint8_t> bytes_of(const Tensor &tensor)
{
    std::vector<uint8_t> bytes_list(tensor.numel() * tensor.element_size());
    cudaCheck(cudaMemcpy(bytes_list.data(), tensor.data_ptr(), bytes_list.size(), cudaMemcpyDeviceToHost));
    return bytes_list;
}

/// Every tensor attention_8bit_forward and attention_8bit_backward take, for one size, made with 0s except qkv
/// and output_grad, which the test fills.
struct EightBitTensors
{
    // forward
    Tensor qkv_tensor;                 // bfloat16
    Tensor output_tensor;              // bfloat16
    Tensor query_8bit_tensor;          // Int32, 4 E4M3s in each int
    Tensor key_8bit_tensor;
    Tensor value_8bit_tensor;
    Tensor query_scales_tensor;        // float, one per head row
    Tensor key_scales_tensor;
    Tensor value_scales_tensor;
    Tensor logsumexp_tensor;           // float
    // backward
    Tensor output_grad_tensor;         // bfloat16
    Tensor qkv_grad_tensor;            // bfloat16
    Tensor qkv_rounded_tensor;         // bfloat16 scratch: the 8-bit ones turned back, where qkv has them
    Tensor row_dot_tensor;             // float scratch

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    EightBitTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        Shape per_head_8bit_list = {batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE / ATTENTION_8BIT_VALUES_PER_INT};
        Shape per_row_list = {batch_size_val, num_heads_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        query_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        key_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        value_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        query_scales_tensor = Tensor::zeros(per_row_list), key_scales_tensor = Tensor::zeros(per_row_list);
        value_scales_tensor = Tensor::zeros(per_row_list), logsumexp_tensor = Tensor::zeros(per_row_list);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        qkv_rounded_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros(per_row_list);
    }

    /// Runs attention_8bit_forward on these tensors, the rounding at this block size (the default if not given).
    void forward(int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK)
    {
        attention_8bit_forward(output_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor,
                               key_scales_tensor, value_scales_tensor, logsumexp_tensor, qkv_tensor, threads_per_block_val);
    }

    /// Runs attention_8bit_backward on these tensors, the turn-back and row_dot at these block sizes (the defaults
    /// if not given).
    void backward(int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK,
                  int row_dot_threads_per_block_val = ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK)
    {
        attention_8bit_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, query_8bit_tensor, key_8bit_tensor,
                                value_8bit_tensor, query_scales_tensor, key_scales_tensor, value_scales_tensor,
                                logsumexp_tensor, qkv_rounded_tensor, row_dot_tensor, threads_per_block_val,
                                row_dot_threads_per_block_val);
    }

    /// Fills qkv and output_grad with random values rounded to bfloat16.
    void fill_random(void)
    {
        qkv_tensor = random_tensor_on_gpu(qkv_tensor.shape()).to(DType::BFloat16);
        output_grad_tensor = random_tensor_on_gpu(output_tensor.shape()).to(DType::BFloat16);
    }
};

/// Gives qkv rounded the way attention_8bit_forward rounds it, done by PyTorch: each head row divided by my
/// scale, rounded to float8_e4m3fn (PyTorch's E4M3), back to float and times the scale. {batch_size, seq_len,
/// 3 · embed_dim} float.
///
///   mine_val   my tensors after a forward: its qkv and the three scales
static torch::Tensor pytorch_rounded_qkv(const EightBitTensors &mine_val)
{
    // 1. the sizes
    int64_t batch_size_val = mine_val.query_scales_tensor.size(0);
    int64_t num_heads_val = mine_val.query_scales_tensor.size(1);
    int64_t seq_len_val = mine_val.query_scales_tensor.size(2);
    // 2. the scales as {batch_size, seq_len, 3, num_heads, 1}, next to qkv's head rows
    torch::Tensor scales_tensor = torch::stack({pytorch_copy_of(mine_val.query_scales_tensor),
                                                pytorch_copy_of(mine_val.key_scales_tensor),
                                                pytorch_copy_of(mine_val.value_scales_tensor)})   // {3, B, NH, T}
                                      .permute({1, 3, 0, 2})
                                      .unsqueeze(-1);
    // 3. divide, round to E4M3, back to float, multiply
    torch::Tensor rows_tensor = pytorch_copy_of(mine_val.qkv_tensor).view({batch_size_val, seq_len_val, 3, num_heads_val, FLASH_HEAD_SIZE});
    torch::Tensor rounded_tensor = (rows_tensor / scales_tensor).to(torch::kFloat8_e4m3fn).to(torch::kFloat) * scales_tensor;
    return rounded_tensor.reshape({batch_size_val, seq_len_val, 3 * num_heads_val * FLASH_HEAD_SIZE});
}

/// PyTorch's answers in float for one qkv and output_grad: qkv split into q, k, v, each cut into heads, then
/// scaled_dot_product_attention with is_causal, and autograd; logsumexp of each row of masked scores on the side.
struct PytorchAttention
{
    torch::Tensor qkv_tensor;
    torch::Tensor output_tensor;
    torch::Tensor logsumexp_tensor;

    /// Works out output, qkv.grad() and logsumexp.
    ///
    ///   qkv_floats_tensor           {batch_size, seq_len, 3 · embed_dim} float
    ///   output_grad_floats_tensor   {batch_size, seq_len, embed_dim} float
    ///   num_heads_val               heads of FLASH_HEAD_SIZE
    PytorchAttention(const torch::Tensor &qkv_floats_tensor, const torch::Tensor &output_grad_floats_tensor,
                     int64_t num_heads_val)
    {
        // 1. the sizes
        int64_t batch_size_val = qkv_floats_tensor.size(0);
        int64_t seq_len_val = qkv_floats_tensor.size(1);
        int64_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        // 2. qkv split and cut into heads, then the causal attention and its backward
        qkv_tensor = qkv_floats_tensor.clone().requires_grad_(true);
        std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
        auto heads_val = [&](const torch::Tensor &part_tensor) {
            return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, FLASH_HEAD_SIZE}).transpose(1, 2);
        };
        torch::Tensor q_tensor = heads_val(parts_list[0]);
        torch::Tensor k_tensor = heads_val(parts_list[1]);
        torch::Tensor v_tensor = heads_val(parts_list[2]);
        torch::Tensor y_tensor = torch::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor, {}, 0.0, true);
        output_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
        output_tensor.backward(output_grad_floats_tensor);
        // 3. logsumexp, on the side
        torch::NoGradGuard no_grad_val;
        torch::Tensor scores_tensor = q_tensor.matmul(k_tensor.transpose(-2, -1)) / std::sqrt((double)FLASH_HEAD_SIZE);
        torch::Tensor after_tensor = torch::ones({seq_len_val, seq_len_val}, scores_tensor.options().dtype(torch::kBool)).triu(1);
        logsumexp_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).logsumexp(-1);
    }
};

/// Tells if this GPU runs 8-bit attention (an H100); if not, prints why the test that asked is skipped.
static bool eight_bit_runs_here(const char *test_ptr)
{
    if (attention_8bit_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: 8-bit attention needs a GPU of compute capability %d.%d (an H100: WGMMA and TMA)\n",
           HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
    return false;
}

// ══ part 1: the rounding ══════════════════════════════════════════════════════

static void needs_head_size_64_whole_tiles_and_an_h100(void)
{
    // Expect: attention_8bit_fits says no to a seq_len that is not a multiple of 64 (1000) and to a head_size
    // other than 64 (32); and yes to 1024 and 64 exactly when the GPU's compute capability is 9.0 (an H100: the
    // only one with WGMMA and TMA).
    // 1. this GPU's compute capability
    int device_index_val = 0;
    int major_val = 0;
    int minor_val = 0;
    cudaCheck(cudaGetDevice(&device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_index_val));
    cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, device_index_val));
    // 2. check
    ASSERT_EQ(attention_8bit_fits(1000, FLASH_HEAD_SIZE), false);
    ASSERT_EQ(attention_8bit_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE / 2), false);
    ASSERT_EQ(attention_8bit_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE), 10 * major_val + minor_val == HARDWARE_WGMMA_CAPABILITY);
}

static void each_row_s_scale_is_a_power_of_2_that_fits_448(void)
{
    // Expect: random qkv {2, 12, 256}, with token 0 of batch 0 all 0. Every head row's scale is a power of 2,
    // log2(scale) a whole number, and the smallest one with largest |value| / scale <= 448: so that ratio is in
    // (224, 448]. Its 36 rows of 0s (12 heads x query, key, value) get scale 1.
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. random qkv with one token of 0s; the forward
    torch::manual_seed(0);
    EightBitTensors mine_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    mine_val.fill_random();
    mine_val.qkv_tensor[0][0].fill_(0);
    mine_val.forward();
    // 2. each head row's largest |value|, from PyTorch, as {3, batch_size, num_heads, seq_len} like the scales
    torch::Tensor largest_tensor =
        pytorch_copy_of(mine_val.qkv_tensor).view({2, 4 * FLASH_TILE, 3, GPT2_NUM_HEADS, FLASH_HEAD_SIZE}).abs().amax(-1).permute({2, 0, 3, 1});
    torch::Tensor scales_tensor = torch::stack({pytorch_copy_of(mine_val.query_scales_tensor),
                                                pytorch_copy_of(mine_val.key_scales_tensor),
                                                pytorch_copy_of(mine_val.value_scales_tensor)});
    torch::Tensor ratio_tensor = largest_tensor / scales_tensor;
    torch::Tensor log2_tensor = scales_tensor.log2();
    torch::Tensor nonzero_tensor = largest_tensor > 0;
    // 3. check
    ASSERT_EQ((log2_tensor != log2_tensor.round()).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ((nonzero_tensor & ((ratio_tensor <= 224.0) | (ratio_tensor > HARDWARE_E4M3_LARGEST))).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ((~nonzero_tensor).sum().item<int64_t>(), (int64_t)(3 * GPT2_NUM_HEADS));
    ASSERT_EQ((~nonzero_tensor & (scales_tensor != 1.0)).sum().item<int64_t>(), (int64_t)0);
}

static void rounding_matches_pytorch_s_float8(void)
{
    // Expect: random qkv {2, 12, 256}. The forward's 8-bit query, key and value, turned back to bfloat16 by
    // attention_8bit_turn_back where qkv has them, are exactly PyTorch's: qkv / my scale, to float8_e4m3fn, back
    // to float, times the scale. Both round to the nearest E4M3 with ties to even, and turning back is exact.
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. the forward, then the turn-back
    torch::manual_seed(1);
    EightBitTensors mine_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    mine_val.fill_random();
    mine_val.forward();
    attention_8bit_turn_back(mine_val.qkv_rounded_tensor, mine_val.query_8bit_tensor, mine_val.key_8bit_tensor,
                             mine_val.value_8bit_tensor, mine_val.query_scales_tensor, mine_val.key_scales_tensor,
                             mine_val.value_scales_tensor);
    // 2. check against PyTorch's rounding, {batch_size, seq_len, 3 · embed_dim} as qkv: the same bits
    ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.qkv_rounded_tensor), pytorch_rounded_qkv(mine_val)), true);
}

// ══ part 2: attention against PyTorch ═════════════════════════════════════════

/// Runs 8-bit attention's forward and backward on random bfloat16 qkv and output_grad, and PyTorch in float on
/// the same values rounded the same way; checks output and logsumexp within BFLOAT16_TOLERANCE, qkv_grad within
/// BFLOAT16_GRAD_TOLERANCE.
static void matches_pytorch_on_the_rounded_values(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
{
    // 1. mine
    torch::manual_seed(0);
    EightBitTensors mine_val(batch_size_val, num_heads_val, seq_len_val);
    mine_val.fill_random();
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's, in float on the rounded qkv
    PytorchAttention pytorch_val(pytorch_rounded_qkv(mine_val), pytorch_copy_of(mine_val.output_grad_tensor),
                                 (int64_t)num_heads_val);
    // 3. check
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_val.output_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.logsumexp_tensor, pytorch_val.logsumexp_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_val.qkv_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

static void matches_pytorch_on_one_tile(void)
{
    // Expect: {1, 2, 64}: one tile of 64 queries and 64 keys, all on the diagonal. Within the bfloat16
    // tolerances of PyTorch on the rounded values.
    if (eight_bit_runs_here(__func__))
        matches_pytorch_on_the_rounded_values(1, 2, FLASH_TILE);
}

static void matches_pytorch_on_gpt2_heads_of_256_tokens(void)
{
    // Expect: {2, 12, 256}: 4 tiles per head, the online softmax's max moving from tile to tile, each key
    // tile's scales loaded with it. The same.
    if (eight_bit_runs_here(__func__))
        matches_pytorch_on_the_rounded_values(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
}

static void matches_pytorch_on_gpt2_small_size(void)
{
    // Expect: {4, 12, 1024}, as training runs it: 16 tiles per head, 768 blocks. The same.
    if (eight_bit_runs_here(__func__))
        matches_pytorch_on_the_rounded_values(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
}

static void stays_near_pytorch_on_the_values_before_rounding(void)
{
    // Expect: {4, 12, 1024}, PyTorch in float on the bfloat16 qkv as it was, before rounding to E4M3: output
    // within EIGHT_BIT_TOLERANCE, qkv_grad within EIGHT_BIT_GRAD_TOLERANCE. What 8 bits cost, next to
    // bfloat16's 2e-2 and 5e-2.
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. mine
    torch::manual_seed(0);
    EightBitTensors mine_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    mine_val.fill_random();
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's, in float on qkv before rounding
    PytorchAttention pytorch_val(pytorch_copy_of(mine_val.qkv_tensor), pytorch_copy_of(mine_val.output_grad_tensor),
                                 GPT2_NUM_HEADS);
    // 3. check
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_val.output_tensor), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_val.qkv_tensor.grad()), 0.0,
                EIGHT_BIT_GRAD_TOLERANCE);
}

// ══ part 3: writing over, the kernels' launch, and any block size ═════════════

static void backward_writes_over_qkv_grad_the_same_every_time(void)
{
    // Expect: qkv_grad filled with 7s first: after a backward, no 7 is left (every value of its three parts is
    // written), and a second backward gives the same bits (no two blocks add onto one value).
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. a forward, then a backward onto 7s
    torch::manual_seed(0);
    EightBitTensors mine_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    mine_val.fill_random();
    mine_val.forward();
    mine_val.qkv_grad_tensor.fill_(7.0);
    mine_val.backward();
    torch::Tensor first_tensor = pytorch_copy_of(mine_val.qkv_grad_tensor);
    ASSERT_EQ((first_tensor == 7.0f).sum().item<int64_t>(), (int64_t)0);
    // 2. a second backward: the same bits
    mine_val.backward();
    ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.qkv_grad_tensor), first_tensor), true);
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (256; the tile kernel's 128), and nothing
    // is spilled: cudaFuncGetAttributes' numRegs (registers per thread), maxThreadsPerBlock (the largest block)
    // and localSizeBytes (local memory per thread, 0 if nothing spills). The forward kernel holds the scores and
    // the output (32 sums each) and the attention_probs packed as A (16 registers); its query and key tiles stay
    // in shared memory. The backward's kernels are attention_bf16.cuh's, tested there.
    const int kernel_count_val = 3;
    const void *kernels_list[kernel_count_val] = {(const void *)attention_8bit_round_qkv_kernel,
                                                  (const void *)attention_8bit_forward_kernel,
                                                  (const void *)attention_8bit_turn_back_kernel};
    const char *names_list[kernel_count_val] = {"attention_8bit_round_qkv_kernel", "attention_8bit_forward_kernel",
                                                "attention_8bit_turn_back_kernel"};
    int threads_list[kernel_count_val] = {ATTENTION_8BIT_THREADS_PER_BLOCK, FLASH_THREADS, ATTENTION_8BIT_THREADS_PER_BLOCK};
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
    // Expect: random qkv and output_grad, {2, 12, 256}. The forward with the rounding's block size, and the
    // backward with the turn-back's and row_dot's, set to 32, 128, 512 and 1024 (one warp up to
    // HARDWARE_MAX_THREADS_PER_BLOCK) give the defaults' 8-bit query, key, value, scales, output and qkv_grad to the
    // last bit: each head row is one warp's, each int of the turn-back one thread's, whatever the block.
    if (!eight_bit_runs_here(__func__))
        return;
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random qkv and output_grad; the defaults' results
    torch::manual_seed(0);
    EightBitTensors usual_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
    usual_val.fill_random();
    usual_val.forward();
    usual_val.backward();

    // 2. each block size, from the same qkv and output_grad, against the defaults'
    for (int threads_per_block_val : block_sizes_list)
    {
        EightBitTensors tensors_val(2, GPT2_NUM_HEADS, 4 * FLASH_TILE);
        tensors_val.qkv_tensor.copy_(usual_val.qkv_tensor);
        tensors_val.output_grad_tensor.copy_(usual_val.output_grad_tensor);
        tensors_val.forward(threads_per_block_val);
        tensors_val.backward(threads_per_block_val, threads_per_block_val);
        ASSERT_EQ(bytes_of(tensors_val.query_8bit_tensor) == bytes_of(usual_val.query_8bit_tensor), true);
        ASSERT_EQ(bytes_of(tensors_val.key_8bit_tensor) == bytes_of(usual_val.key_8bit_tensor), true);
        ASSERT_EQ(bytes_of(tensors_val.value_8bit_tensor) == bytes_of(usual_val.value_8bit_tensor), true);
        ASSERT_EQ(bytes_of(tensors_val.query_scales_tensor) == bytes_of(usual_val.query_scales_tensor), true);
        ASSERT_EQ(bytes_of(tensors_val.output_tensor) == bytes_of(usual_val.output_tensor), true);
        ASSERT_EQ(bytes_of(tensors_val.qkv_grad_tensor) == bytes_of(usual_val.qkv_grad_tensor), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    needs_head_size_64_whole_tiles_and_an_h100();
    each_row_s_scale_is_a_power_of_2_that_fits_448();
    rounding_matches_pytorch_s_float8();

    matches_pytorch_on_one_tile();
    matches_pytorch_on_gpt2_heads_of_256_tokens();
    matches_pytorch_on_gpt2_small_size();
    stays_near_pytorch_on_the_values_before_rounding();

    backward_writes_over_qkv_grad_the_same_every_time();
    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
