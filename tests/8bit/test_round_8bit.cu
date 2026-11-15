// test_round_8bit.cu — tests for llmc/kernels/8bit/round_8bit.cuh (round_8bit): a bfloat16 matrix rounded to FP8 E4M3
// with one power-of-2 scale per row, from a matrix as it sits or turned round. Every check prints its test case,
// input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the scales and the E4M3s against PyTorch's float8_e4m3fn; part 2 checks that a turned matrix
// gives the same bits as rounding the matrix turned round first, and that round_8bit_rows_and_columns gives both
// roundings' bits from its two reads, and round_8bit_per_tensor's one scale for the whole tensor (exact, and from the
// tensor before with a history); part 3 checks that each kernel's registers fit its block, and other block
// sizes. Any GPU. The timings are in tests/8bit/tune_round_8bit.cu.
#include <torch/types.h>              // torch::randn, float8_e4m3fn; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <cmath>                      // std::exp2, std::ceil, std::log2
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

static void long_rows_match_pytorch_s_float8(void)
{
    // Expect: random {64, 9216} at the block size its width picks (mlp down's input: 1,152 loads a row, 512 threads,
    // 2.25 a thread: round_8bit_rows_kernel<4>), {16, 32768} at 256 threads (4,096 loads, 16 a thread: <16>, the longest
    // row a block of 256 takes), and {16, 50304} at its pick (the head's logits' gradient: 6,288 loads, more than a
    // block's registers hold: the tile kernels). The E4M3s turned back are exactly PyTorch's with my scales, and each
    // scale the smallest power of 2 that fits 448, as for short rows.
    const size_t shapes_list[3][3] = {{64, 9216, 0}, {16, 32768, ROUND_8BIT_LEAST_THREADS}, {16, 50304, 0}};   // rows, row length, threads (0: picked)
    for (const auto &shape_val : shapes_list)
    {
        // 1. the rounding
        torch::manual_seed(4);
        RoundedMatrix matrix_val(random_bf16_tensor({shape_val[0], shape_val[1]}), false, (int)shape_val[2]);
        // 2. PyTorch's rounding with my scales, and the scales' ratios
        torch::Tensor scales_tensor = pytorch_copy_of(matrix_val.scales_tensor).unsqueeze(1);
        torch::Tensor expected_tensor =
            (pytorch_copy_of(matrix_val.values_tensor) / scales_tensor).to(torch::kFloat8_e4m3fn).to(torch::kFloat) * scales_tensor;
        torch::Tensor ratio_tensor = pytorch_copy_of(matrix_val.values_tensor).abs().amax(-1) / scales_tensor.squeeze(1);
        // 3. check
        print_test_name(__func__);
        printf("    {%zu, %zu}, %zu threads a block (0: picked)\n", shape_val[0], shape_val[1], shape_val[2]);
        ASSERT_EQ(torch::equal(matrix_val.turned_back(), expected_tensor), true);
        ASSERT_EQ(((ratio_tensor <= 224.0) | (ratio_tensor > HARDWARE_E4M3_LARGEST)).sum().item<int64_t>(), (int64_t)0);
    }
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
    // scale is the largest of a column of output_grad, an atomic max of each 128 x 128 tile's (round_8bit_largest_kernel);
    // the E4M3s come out of the tiles turned round (round_8bit_tile_kernel). The same bits as round_8bit on output_gradᵀ.
    turned_gives_the_bits_of_the_turned_round_matrix(__func__, GPT2_ROWS, GPT2_QKV_DIM);
}

static void turned_gives_the_bits_of_the_turned_round_matrix_with_sizes_no_tile_divides(void)
{
    // Expect: input {208, 72}: rows 72 and row_length 208, neither a whole number of 128 x 128 tiles, so the tiles at
    // the edges are part-full (the turned pieces of 16 past the input's rows not written). The same bits.
    turned_gives_the_bits_of_the_turned_round_matrix(__func__, 208, 72);
}

/// Rounds random values {rows, columns} by round_8bit_rows_and_columns (both copies, then each alone), and by
/// round_8bit as it sits and turned, and checks that each copy and its scales are the same to the last bit.
static void rows_and_columns_give_the_bits_of_both_roundings(const char *test_ptr, size_t rows_val, size_t columns_val)
{
    // 1. random values, with row 0 all 0 (its scale 1); round_8bit's two copies
    torch::manual_seed(3);
    Tensor values_tensor = random_bf16_tensor({rows_val, columns_val});
    values_tensor[0].fill_(0);
    RoundedMatrix plain_val(values_tensor);
    RoundedMatrix turned_val(values_tensor, true);
    // 2. round_8bit_rows_and_columns: both, then rows alone, then columns alone
    Tensor rows_8bit_tensor = Tensor::zeros({rows_val, columns_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor row_scales_tensor = Tensor::zeros({rows_val});
    Tensor columns_8bit_tensor = Tensor::zeros({columns_val, rows_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor column_scales_tensor = Tensor::zeros({columns_val});
    // the same bits: the Int32 copies compared as ints, on the CPU
    auto same_ints_val = [](const Tensor &first_tensor, const Tensor &second_tensor) {
        std::vector<int> first_list(first_tensor.numel()), second_list(second_tensor.numel());
        cudaCheck(cudaMemcpy(first_list.data(), first_tensor.data_ptr(), first_list.size() * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(second_list.data(), second_tensor.data_ptr(), second_list.size() * sizeof(int), cudaMemcpyDeviceToHost));
        return first_list == second_list;
    };
    print_test_name(test_ptr);
    printf("    input {%zu, %zu}: both copies\n", rows_val, columns_val);
    round_8bit_rows_and_columns(rows_8bit_tensor, row_scales_tensor, columns_8bit_tensor, column_scales_tensor, values_tensor);
    ASSERT_EQ(same_ints_val(rows_8bit_tensor, plain_val.rounded_tensor), true);
    ASSERT_EQ(row_scales_tensor.tolist() == plain_val.scales_tensor.tolist(), true);
    ASSERT_EQ(same_ints_val(columns_8bit_tensor, turned_val.rounded_tensor), true);
    ASSERT_EQ(column_scales_tensor.tolist() == turned_val.scales_tensor.tolist(), true);
    printf("    the row copy alone, then the turned copy alone\n");
    rows_8bit_tensor.zero_(), row_scales_tensor.zero_(), columns_8bit_tensor.zero_(), column_scales_tensor.zero_();
    round_8bit_rows_and_columns(rows_8bit_tensor, row_scales_tensor, Tensor(), Tensor(), values_tensor);
    ASSERT_EQ(same_ints_val(rows_8bit_tensor, plain_val.rounded_tensor), true);
    ASSERT_EQ(row_scales_tensor.tolist() == plain_val.scales_tensor.tolist(), true);
    round_8bit_rows_and_columns(Tensor(), Tensor(), columns_8bit_tensor, column_scales_tensor, values_tensor);
    ASSERT_EQ(same_ints_val(columns_8bit_tensor, turned_val.rounded_tensor), true);
    ASSERT_EQ(column_scales_tensor.tolist() == turned_val.scales_tensor.tolist(), true);
}

static void rows_and_columns_give_the_bits_of_both_roundings_on_qkv_output_grad(void)
{
    // Expect: the qkv layer's output_grad {4096, 2304}, which its input_grad rounds as it sits and its weight_grad
    // turned: both copies from round_8bit_rows_and_columns's two reads, the same bits as round_8bit's two calls, and each
    // copy alone too. Row 0 all 0: scale 1.
    rows_and_columns_give_the_bits_of_both_roundings(__func__, GPT2_ROWS, GPT2_QKV_DIM);
}

static void rows_and_columns_give_the_bits_of_both_roundings_with_sizes_no_tile_divides(void)
{
    // Expect: {208, 272}, no whole number of 128 x 128 tiles either way. The same bits.
    rows_and_columns_give_the_bits_of_both_roundings(__func__, 208, 272);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

/// The copies round_8bit_per_tensor writes of one {rows, columns} matrix, and PyTorch's reading of them.
struct PerTensorCopies
{
    Tensor rows_8bit_tensor;           // {rows, columns / 4} Int32
    Tensor row_scales_tensor;          // {rows}
    Tensor columns_8bit_tensor;        // {columns, rows / 4} Int32
    Tensor column_scales_tensor;       // {columns}
    size_t rows_val;
    size_t columns_val;

    PerTensorCopies(size_t rows_val, size_t columns_val)
        : rows_8bit_tensor(Tensor::zeros({rows_val, columns_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32)),
          row_scales_tensor(Tensor::zeros({rows_val})),
          columns_8bit_tensor(Tensor::zeros({columns_val, rows_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32)),
          column_scales_tensor(Tensor::zeros({columns_val})), rows_val(rows_val), columns_val(columns_val) {}

    /// Rounds values into both copies by round_8bit_per_tensor.
    void round(const Tensor &values_tensor, const Round8bitHistory *history_ptr = nullptr, bool same_input_val = false)
    {
        round_8bit_per_tensor(rows_8bit_tensor, row_scales_tensor, columns_8bit_tensor, column_scales_tensor, values_tensor, history_ptr,
                              same_input_val);
    }

    /// Gives the row copy turned back by PyTorch (each E4M3 as float8_e4m3fn, times its row's scale): {rows, columns}.
    torch::Tensor rows_turned_back() const
    {
        torch::Tensor e4m3_tensor = torch::from_blob(rows_8bit_tensor.data_ptr(), {(int64_t)rows_val, (int64_t)columns_val},
                                                     torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(row_scales_tensor).unsqueeze(1);
    }

    /// Gives the turned copy turned back by PyTorch, and turned round again: {rows, columns}.
    torch::Tensor columns_turned_back() const
    {
        torch::Tensor e4m3_tensor = torch::from_blob(columns_8bit_tensor.data_ptr(), {(int64_t)columns_val, (int64_t)rows_val},
                                                     torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        return (e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(column_scales_tensor).unsqueeze(1)).t();
    }
};

/// Gives the scale of a largest |value|, as round_8bit_scale: the smallest power of 2 with largest / scale <= 448.
static double scale_of(double largest_val)
{
    return std::exp2(std::ceil(std::log2(largest_val / HARDWARE_E4M3_LARGEST)));
}

/// Checks both of copies_val's copies: every row's and column's scale is scale_val, and the E4M3s turned back are
/// PyTorch's rounding of values at that scale, clamped to the largest E4M3 first (what saturating gives).
static void check_per_tensor(const PerTensorCopies &copies_val, const Tensor &values_tensor, double scale_val)
{
    torch::Tensor clamped_tensor = (pytorch_copy_of(values_tensor) / scale_val).clamp(-HARDWARE_E4M3_LARGEST, HARDWARE_E4M3_LARGEST);
    torch::Tensor expected_tensor = clamped_tensor.to(torch::kFloat8_e4m3fn).to(torch::kFloat) * scale_val;
    ASSERT_EQ((pytorch_copy_of(copies_val.row_scales_tensor) != scale_val).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ((pytorch_copy_of(copies_val.column_scales_tensor) != scale_val).sum().item<int64_t>(), (int64_t)0);
    ASSERT_EQ(torch::equal(copies_val.rows_turned_back(), expected_tensor), true);
    ASSERT_EQ(torch::equal(copies_val.columns_turned_back(), expected_tensor), true);
}

static void per_tensor_gives_one_scale_and_pytorch_s_float8(void)
{
    // Expect: random {208, 272} (no tile divides it), row 5 a hundred times larger, row 0 all 0, no history. Every
    // row's and column's scale is the one scale of the whole tensor's largest |value| (a power of 2, largest / scale
    // in (224, 448]), and both copies turned back are exactly PyTorch's float8_e4m3fn at that scale. The same at the
    // qkv output_grad's size.
    torch::manual_seed(4);
    print_test_name(__func__);
    const size_t sizes_list[2][2] = {{208, 272}, {GPT2_ROWS, GPT2_QKV_DIM}};
    for (const auto &size_val : sizes_list)
    {
        torch::Tensor floats_tensor = torch::randn({(int64_t)size_val[0], (int64_t)size_val[1]}, torch::TensorOptions().device(torch::kCUDA));
        floats_tensor[5].mul_(100.0);
        floats_tensor[0].zero_();
        Tensor values_floats_tensor = Tensor::empty({size_val[0], size_val[1]});
        cudaCheck(cudaMemcpy(values_floats_tensor.data(), floats_tensor.data_ptr<float>(), values_floats_tensor.numel() * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        Tensor values_tensor = values_floats_tensor.to(DType::BFloat16);
        PerTensorCopies copies_val(size_val[0], size_val[1]);
        copies_val.round(values_tensor);
        double largest_val = pytorch_copy_of(values_tensor).abs().max().item<double>();
        printf("    input {%zu, %zu}, largest %.3f, scale %g:\n", size_val[0], size_val[1], largest_val, scale_of(largest_val));
        check_per_tensor(copies_val, values_tensor, scale_of(largest_val));
    }
}

static void per_tensor_with_a_history_takes_the_scale_of_the_tensor_before(void)
{
    // Expect: three random {208, 272} tensors rounded in turn with one history, the second 5 times the first's size:
    //     the first    its own largest's scale (exact: a read of its own)
    //     the second   the scale of the first's largest x ROUND_8BIT_DELAYED_MARGIN: too small for it, so its values
    //                  past 448 x scale are clamped to the largest E4M3
    //     the third    the scale of the second's largest x the margin (found while the second was rounded)
    //     the third again, as the same input: its own largest's scale, exact (found while it was rounded)
    torch::manual_seed(5);
    Tensor first_tensor = random_bf16_tensor({208, 272});
    torch::Tensor larger_tensor = pytorch_copy_of(random_bf16_tensor({208, 272})) * 5.0;
    Tensor second_floats_tensor = Tensor::empty({208, 272});
    cudaCheck(cudaMemcpy(second_floats_tensor.data(), larger_tensor.data_ptr<float>(), second_floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    Tensor second_tensor = second_floats_tensor.to(DType::BFloat16);
    Tensor third_tensor = random_bf16_tensor({208, 272});
    double first_largest_val = pytorch_copy_of(first_tensor).abs().max().item<double>();
    double second_largest_val = pytorch_copy_of(second_tensor).abs().max().item<double>();
    double third_largest_val = pytorch_copy_of(third_tensor).abs().max().item<double>();
    Round8bitHistory history_val;
    PerTensorCopies copies_val(208, 272);
    print_test_name(__func__);
    printf("    the first, exact:\n");
    copies_val.round(first_tensor, &history_val);
    check_per_tensor(copies_val, first_tensor, scale_of(first_largest_val));
    printf("    the second, 5 times larger, at the first's scale with its margin:\n");
    copies_val.round(second_tensor, &history_val);
    check_per_tensor(copies_val, second_tensor, scale_of(first_largest_val * ROUND_8BIT_DELAYED_MARGIN));
    printf("    the third, at the second's:\n");
    copies_val.round(third_tensor, &history_val);
    check_per_tensor(copies_val, third_tensor, scale_of(second_largest_val * ROUND_8BIT_DELAYED_MARGIN));
    printf("    the third again as the same input, exact:\n");
    copies_val.round(third_tensor, &history_val, true);
    check_per_tensor(copies_val, third_tensor, scale_of(third_largest_val));
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block, and nothing is spilled: cudaFuncGetAttributes'
    // numRegs (registers per thread), maxThreadsPerBlock (the largest block) and localSizeBytes (0 if nothing
    // spills), read without running anything.
    // The row kernel is made for 1, 2, 4, 8 and 16 loads a thread (8 to 128 floats of a row kept in registers): those
    // up to 8 at the most threads round_8bit_threads_for picks (512), the 16-load one at the least (256; a larger
    // block asked of it goes to the tile kernels).
    const int kernel_count_val = 8;
    const void *kernels_list[kernel_count_val] = {(const void *)round_8bit_rows_kernel<1>, (const void *)round_8bit_rows_kernel<2>,
                                                  (const void *)round_8bit_rows_kernel<4>, (const void *)round_8bit_rows_kernel<8>,
                                                  (const void *)round_8bit_rows_kernel<16>, (const void *)round_8bit_largest_kernel,
                                                  (const void *)round_8bit_tile_kernel, (const void *)round_8bit_tensor_largest_kernel};
    const char *names_list[kernel_count_val] = {"round_8bit_rows_kernel<1>", "round_8bit_rows_kernel<2>", "round_8bit_rows_kernel<4>",
                                                "round_8bit_rows_kernel<8>", "round_8bit_rows_kernel<16>", "round_8bit_largest_kernel",
                                                "round_8bit_tile_kernel", "round_8bit_tensor_largest_kernel"};
    int threads_list[kernel_count_val] = {ROUND_8BIT_MOST_THREADS, ROUND_8BIT_MOST_THREADS, ROUND_8BIT_MOST_THREADS,
                                          ROUND_8BIT_MOST_THREADS, ROUND_8BIT_LEAST_THREADS, ROUND_8BIT_TILE_THREADS,
                                          ROUND_8BIT_TILE_THREADS, ROUND_8BIT_LARGEST_THREADS};
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
    // Expect: random {304, 768}, rounded as it sits and as a turned matrix, with block sizes 32, 128, 512 and 1024
    // (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK): the same E4M3s and scales as the default's, to the last bit.
    // As it sits each row is one block's, its largest the same in any order (32 threads hold 4 loads each of a row's
    // 96, 1024 threads 1 load, most of them none); turned, the block size does not apply (the tile kernels' is fixed),
    // so the same launches. 304 rows: turned, they are the row length, which must be a multiple of
    // ROUND_8BIT_ROW_LENGTH_MULTIPLE (16); 304 is not a whole number of the 128-row tiles, so the last is part-full.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random values; the default's roundings
    torch::manual_seed(5);
    Tensor values_tensor = random_bf16_tensor({304, GPT2_EMBED_DIM});
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
    long_rows_match_pytorch_s_float8();

    turned_gives_the_bits_of_the_turned_round_matrix_on_gpt2_output_grad();
    turned_gives_the_bits_of_the_turned_round_matrix_with_sizes_no_tile_divides();
    rows_and_columns_give_the_bits_of_both_roundings_on_qkv_output_grad();
    rows_and_columns_give_the_bits_of_both_roundings_with_sizes_no_tile_divides();
    per_tensor_gives_one_scale_and_pytorch_s_float8();
    per_tensor_with_a_history_takes_the_scale_of_the_tensor_before();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
