// test_round_8bit.cu — tests for llmc/kernels/8bit/round_8bit.cuh (round_8bit): a bfloat16 matrix rounded to FP8 E4M3
// with one power-of-2 scale per row, from a matrix as it sits or turned round. Every check prints its test case,
// input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the scales and the E4M3s against PyTorch's float8_e4m3fn; part 2 checks that a turned matrix
// gives the same bits as rounding the matrix turned round first; part 3 checks that each kernel's registers fit
// its block, and other block sizes. Any GPU. The timings are in tunes/tune_round_8bit.cu.
#include <torch/types.h>              // torch::randn, float8_e4m3fn; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ
#include "../../llmc/kernels/8bit/round_8bit.cuh"

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
/// on the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);  // the source_tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: the head's output is 206M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// A matrix and its 8-bit copy: the bfloat16 values as they sit, their E4M3s (4 in each int) and one scale per row.
struct RoundedMatrix
{
    Tensor values_tensor;              // {rows, row_length} bfloat16, or turned {row_length, rows}
    Tensor rounded_tensor;             // {rows, row_length / 4} Int32
    Tensor scales_tensor;              // {rows} float
    size_t rows_val;
    size_t row_length_val;

    /// Makes it from `matrix_values_tensor`, rounded by round_8bit (turned: the values sit turned round) at this
    /// block size (the default if not given).
    RoundedMatrix(const Tensor &matrix_values_tensor, bool turned_val = false, int threads_per_block_val = ROUND_8BIT_THREADS_PER_BLOCK)
        : values_tensor(matrix_values_tensor)
    {
        rows_val = values_tensor.size(turned_val ? 1 : 0);
        row_length_val = values_tensor.size(turned_val ? 0 : 1);
        rounded_tensor = Tensor::zeros({rows_val, row_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
        scales_tensor = Tensor::zeros({rows_val});
        round_8bit(rounded_tensor, scales_tensor, values_tensor, turned_val, threads_per_block_val);
    }

    /// Gives the rounded values turned back, by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times its
    /// row's scale, exact. {rows, row_length} float.
    torch::Tensor turned_back() const
    {
        torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {(int64_t)rows_val, (int64_t)row_length_val},
                                                     torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
    }
};

/// Makes a bfloat16 tensor holding `source_tensor`'s values turned round, GPU to GPU (exact: they are bfloat16's).
static Tensor turned_round_copy(const Tensor &source_tensor)
{
    torch::Tensor turned_tensor = pytorch_copy_of(source_tensor).t().contiguous();
    Tensor floats_tensor = Tensor::empty({(size_t)turned_tensor.size(0), (size_t)turned_tensor.size(1)});
    cudaCheck(cudaMemcpy(floats_tensor.data(), turned_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

// ══ part 1: the rounding ══════════════════════════════════════════════════════

static void each_row_s_scale_is_a_power_of_2_that_fits_448(void)
{
    // Expect: random {300, 768}, with row 0 all 0. Every row's scale is a power of 2, log2(scale) a whole number,
    // and the smallest one with largest |value| / scale <= 448: so that ratio is in (224, 448]. Row 0 gets 1.
    // 1. random values with a row of 0s; the rounding
    torch::manual_seed(0);
    Tensor values_tensor = random_bf16_tensor({300, GPT2_EMBED_DIM});
    values_tensor[0].fill_(0);
    RoundedMatrix matrix_val(values_tensor);
    // 2. each row's largest |value|, from PyTorch
    torch::Tensor largest_tensor = pytorch_copy_of(values_tensor).abs().amax(-1);
    torch::Tensor scales_tensor = pytorch_copy_of(matrix_val.scales_tensor);
    torch::Tensor ratio_tensor = largest_tensor / scales_tensor;
    torch::Tensor log2_tensor = scales_tensor.log2();
    torch::Tensor nonzero_tensor = largest_tensor > 0;
    // 3. check
    ASSERT_EQ((log2_tensor != log2_tensor.round()).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ((nonzero_tensor & ((ratio_tensor <= 224.0) | (ratio_tensor > HARDWARE_E4M3_LARGEST))).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ((~nonzero_tensor).sum().item<int64_t>(), (int64_t)1);
    ASSERT_EQ(scales_tensor[0].item<float>(), 1.0f);
}

static void rounding_matches_pytorch_s_float8(void)
{
    // Expect: random {300, 768}. The E4M3s turned back are exactly PyTorch's: value / my scale, to float8_e4m3fn,
    // back to float, times the scale. Both round to the nearest E4M3 with ties to even, and turning back is exact.
    // 1. the rounding
    torch::manual_seed(1);
    RoundedMatrix matrix_val(random_bf16_tensor({300, GPT2_EMBED_DIM}));
    // 2. PyTorch's rounding with my scales
    torch::Tensor scales_tensor = pytorch_copy_of(matrix_val.scales_tensor).unsqueeze(1);
    torch::Tensor expected_tensor =
        (pytorch_copy_of(matrix_val.values_tensor) / scales_tensor).to(torch::kFloat8_e4m3fn).to(torch::kFloat) * scales_tensor;
    // 3. check: the same bits
    ASSERT_EQ(torch::equal(matrix_val.turned_back(), expected_tensor), true);
}

// ══ part 2: turned round ══════════════════════════════════════════════════════

/// Rounds random `values` {row_length, rows} as a turned matrix, and the same values turned round first as a plain
/// one, and checks that both give the same E4M3s and scales to the last bit.
static void turned_gives_the_bits_of_the_turned_round_matrix(const char *test_ptr, size_t row_length_val, size_t rows_val)
{
    // 1. random values sitting turned; both roundings
    torch::manual_seed(2);
    Tensor values_tensor = random_bf16_tensor({row_length_val, rows_val});
    RoundedMatrix turned_val(values_tensor, true);
    RoundedMatrix plain_val(turned_round_copy(values_tensor));
    // 2. check
    print_test_name(test_ptr);
    printf("    input {%zu, %zu} turned: rows %zu, row_length %zu\n", row_length_val, rows_val, rows_val, row_length_val);
    ASSERT_EQ(torch::equal(pytorch_copy_of(turned_val.scales_tensor), pytorch_copy_of(plain_val.scales_tensor)), true);
    ASSERT_EQ(torch::equal(turned_val.turned_back(), plain_val.turned_back()), true);
}

static void turned_gives_the_bits_of_the_turned_round_matrix_on_gpt2_output_grad(void)
{
    // Expect: the qkv layer's output_grad {4096, 2304} as weight_grad's left: rows 2304, row_length 4096. Each row's
    // scale is the largest of a column of output_grad, read by round_8bit_turned_scales_kernel; the E4M3s come out
    // of 32 x 32 tiles turned round. The same bits as round_8bit on output_gradᵀ.
    turned_gives_the_bits_of_the_turned_round_matrix(__func__, GPT2_ROWS, GPT2_QKV_DIM);
}

static void turned_gives_the_bits_of_the_turned_round_matrix_with_sizes_no_tile_divides(void)
{
    // Expect: input {80, 50}: rows 50 and row_length 80, neither a whole number of 32 x 32 tiles, so the tiles at
    // the edges are part-full. The same bits.
    turned_gives_the_bits_of_the_turned_round_matrix(__func__, 80, 50);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block, and nothing is spilled: cudaFuncGetAttributes'
    // numRegs (registers per thread), maxThreadsPerBlock (the largest block) and localSizeBytes (0 if nothing
    // spills), read without running anything.
    const int kernel_count_val = 3;
    const void *kernels_list[kernel_count_val] = {(const void *)round_8bit_kernel, (const void *)round_8bit_turned_scales_kernel,
                                                  (const void *)round_8bit_turned_kernel};
    const char *names_list[kernel_count_val] = {"round_8bit_kernel", "round_8bit_turned_scales_kernel", "round_8bit_turned_kernel"};
    int threads_list[kernel_count_val] = {ROUND_8BIT_THREADS_PER_BLOCK, ROUND_8BIT_THREADS_PER_BLOCK, ROUND_8BIT_TURN_THREADS};
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory, %zu bytes shared memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes, attributes_val.sharedSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: random {300, 768}, rounded as it sits and as a turned matrix, with block sizes 32, 128, 512 and 1024
    // (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK): the same E4M3s and scales as the default's, to the last bit.
    // As it sits each row is one warp's; turned, the warps share a column, and the largest is the same in any order.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random values; the default's roundings
    torch::manual_seed(5);
    Tensor values_tensor = random_bf16_tensor({300, GPT2_EMBED_DIM});
    torch::Tensor usual_tensor = RoundedMatrix(values_tensor).turned_back();
    torch::Tensor usual_turned_tensor = RoundedMatrix(values_tensor, true).turned_back();
    // 2. each block size against them
    for (int threads_per_block_val : block_sizes_list)
    {
        ASSERT_EQ(torch::equal(RoundedMatrix(values_tensor, false, threads_per_block_val).turned_back(), usual_tensor), true);
        ASSERT_EQ(torch::equal(RoundedMatrix(values_tensor, true, threads_per_block_val).turned_back(), usual_turned_tensor), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    each_row_s_scale_is_a_power_of_2_that_fits_448();
    rounding_matches_pytorch_s_float8();

    turned_gives_the_bits_of_the_turned_round_matrix_on_gpt2_output_grad();
    turned_gives_the_bits_of_the_turned_round_matrix_with_sizes_no_tile_divides();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
