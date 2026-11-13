// test_fma_matmul_8bit.cu — tests for llmc/kernels/8bit/fma_matmul_8bit.cuh (fma_matmul_8bit): left and right rounded to FP8 E4M3 by
// round_8bit, one scale per row, then output = left_scale · right_scale · (left_8bit · right_8bitᵀ) (+ bias) (+
// output) on the CUDA cores. Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks the product against PyTorch's float product of the same rounded values, on GPT-2 small's 5 linear
// layers, a backward product (both copies rounded from turned matrices) and sizes no square divides, and how far 8
// bits move it from the values before rounding; part 2 checks numbers that must come out exactly; part 3 checks
// that the kernel's registers fit its block. The rounding itself is tested in tests/test_round_8bit.cu. Any GPU. The
// timings are in tests/8bit/tune_fma_matmul_8bit.cu.
#include <torch/types.h>              // torch::randn, torch::matmul, float8_e4m3fn; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/8bit/fma_matmul_8bit.cuh"

// Against PyTorch in float on the same rounded values: the sums are float on both sides, then mine are rounded
// to bfloat16, by up to 2^-9 (0.2%) of each value. Against the values before rounding: each E4M3 is off by up to
// 2^-4 (6%) of its value, so the product moves by a few percent. Each tolerance is the largest difference, as a
// fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double EIGHT_BIT_TOLERANCE = 1e-1;

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304

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

/// Makes the identity {size, size} in bfloat16.
static Tensor identity_bf16(size_t size_val)
{
    std::vector<float> identity_list(size_val * size_val, 0.0f);
    for (size_t index_val = 0; index_val < size_val; index_val++)
        identity_list[index_val * size_val + index_val] = 1.0f;
    Tensor identity_tensor = Tensor::empty({size_val, size_val});
    identity_tensor.copy_from_cpu(identity_list.data());
    return identity_tensor.to(DType::BFloat16);
}

// ══ part 1: the product against PyTorch's ══════════════════════════════════════

/// Runs fma_matmul_8bit on two rounded matrices with a random bias (or none), and checks the output against PyTorch's float
/// product of the same rounded values (+ bias) within BFLOAT16_TOLERANCE.
static void rounded_matmul_matches_pytorch(const char *test_ptr, const RoundedMatrix &left_val, const RoundedMatrix &right_val,
                                           bool with_bias_val)
{
    // 1. mine
    Tensor bias_tensor = with_bias_val ? random_bf16_tensor({right_val.rows_val}) : Tensor();
    Tensor output_tensor = Tensor::zeros({left_val.rows_val, right_val.rows_val}, DType::BFloat16);
    fma_matmul_8bit(output_tensor, left_val.rounded_tensor, left_val.scales_tensor, right_val.rounded_tensor, right_val.scales_tensor, bias_tensor);
    // 2. PyTorch's, in float on the rounded values
    torch::Tensor expected_tensor = left_val.turned_back().matmul(right_val.turned_back().t());
    if (with_bias_val)
        expected_tensor = expected_tensor + pytorch_copy_of(bias_tensor);
    // 3. check
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu%s\n", left_val.rows_val, right_val.rows_val, left_val.row_length_val,
           with_bias_val ? ", bias" : "");
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

/// Rounds random bfloat16 left {rows, sum_length} and right {columns, sum_length}, then rounded_matmul_matches_pytorch.
static void matmul_matches_pytorch(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                   bool with_bias_val)
{
    torch::manual_seed(0);
    RoundedMatrix left_val(random_bf16_tensor({rows_val, sum_length_val}));
    RoundedMatrix right_val(random_bf16_tensor({columns_val, sum_length_val}));
    rounded_matmul_matches_pytorch(test_ptr, left_val, right_val, with_bias_val);
}

static void matches_pytorch_on_gpt2_qkv(void)
{
    // Expect: x {4096, 768} · W {2304, 768}ᵀ + b. Within 1e-2.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_QKV_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_attention_output(void)
{
    // Expect: {4096, 768} · {768, 768}ᵀ + b. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_mlp_up(void)
{
    // Expect: {4096, 768} · {3072, 768}ᵀ + b. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_mlp_down(void)
{
    // Expect: {4096, 3072} · {768, 3072}ᵀ + b: the longest sum of the five. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, true);
}

static void matches_pytorch_on_gpt2_head(void)
{
    // Expect: {4096, 768} · wte {50304, 768}ᵀ, no bias: the widest. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, false);
}

static void matches_pytorch_on_gpt2_qkv_weight_grad(void)
{
    // Expect: the qkv layer's weight_grad = output_gradᵀ · x, both sitting turned round (output_grad {4096, 2304},
    // x {4096, 768}): round_8bit turns them while rounding, into {2304, 4096} and {768, 4096}, k next to each
    // other as the product reads them. Within 1e-2 of PyTorch on the same rounded values.
    torch::manual_seed(5);
    RoundedMatrix left_val(random_bf16_tensor({GPT2_ROWS, GPT2_QKV_DIM}), true);
    RoundedMatrix right_val(random_bf16_tensor({GPT2_ROWS, GPT2_EMBED_DIM}), true);
    rounded_matmul_matches_pytorch(__func__, left_val, right_val, false);
}

static void matches_pytorch_with_sizes_no_tile_divides(void)
{
    // Expect: {100, 144} · {200, 144}ᵀ + b: 100 rows, 200 columns, 144 values of k, none a whole number of squares
    // or steps, so every square at an edge is part-full and the last step mostly past the edge: 0s there add
    // nothing. Within 1e-2.
    matmul_matches_pytorch(__func__, 100, 200, 144, true);
}

static void stays_near_pytorch_on_the_values_before_rounding(void)
{
    // Expect: the qkv shape, {4096, 768} · {2304, 768}ᵀ, PyTorch in float on the bfloat16 values as they were,
    // before rounding to E4M3: within EIGHT_BIT_TOLERANCE (1e-1). What 8 bits cost, next to bfloat16's 1e-2.
    // 1. random values, rounded; mine
    torch::manual_seed(4);
    RoundedMatrix left_val(random_bf16_tensor({GPT2_ROWS, GPT2_EMBED_DIM}));
    RoundedMatrix right_val(random_bf16_tensor({GPT2_QKV_DIM, GPT2_EMBED_DIM}));
    Tensor output_tensor = Tensor::zeros({GPT2_ROWS, GPT2_QKV_DIM}, DType::BFloat16);
    fma_matmul_8bit(output_tensor, left_val.rounded_tensor, left_val.scales_tensor, right_val.rounded_tensor, right_val.scales_tensor);
    // 2. PyTorch's, in float on the values before rounding
    torch::Tensor expected_tensor = pytorch_copy_of(left_val.values_tensor).matmul(pytorch_copy_of(right_val.values_tensor).t());
    // 3. check
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, EIGHT_BIT_TOLERANCE);
}

// ══ part 2: numbers that must come out exactly ════════════════════════════════

static void times_the_identity_gives_right_turned_round(void)
{
    // Expect: left = the identity {128, 128}: each row's largest |value| is 1, so its scale is 2^-8 (1 / 448 is
    // between 2^-9 and 2^-8) and its 1 is stored as 256, an E4M3 exactly. output[i][j] = 2^-8 x right_scale[j] x
    // 256 x right_8bit[j][i] = right's rounded value [j][i]: an E4M3 times a power of 2, 4 significant bits, exact
    // in bfloat16. So output = (right rounded)ᵀ to the last bit, and the 7s it started with are all written over.
    // 1. the identity, random right, output full of 7s
    torch::manual_seed(1);
    RoundedMatrix left_val(identity_bf16(128));
    RoundedMatrix right_val(random_bf16_tensor({64, 128}));
    Tensor output_tensor = Tensor::zeros({128, 64}, DType::BFloat16);
    output_tensor.fill_(7.0);
    // 2. the product
    fma_matmul_8bit(output_tensor, left_val.rounded_tensor, left_val.scales_tensor, right_val.rounded_tensor, right_val.scales_tensor);
    // 3. check against right rounded, turned round
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), right_val.turned_back().t().contiguous()), true);
}

static void zero_left_gives_the_bias_in_every_row(void)
{
    // Expect: left all 0 (every row's scale 1, every E4M3 0), random right and bias {96}: every sum is 0, so output
    // row r = bias exactly, in all 70 rows.
    // 1. the tensors
    torch::manual_seed(2);
    RoundedMatrix left_val(Tensor::zeros({70, 128}, DType::BFloat16));
    RoundedMatrix right_val(random_bf16_tensor({96, 128}));
    Tensor bias_tensor = random_bf16_tensor({96});
    Tensor output_tensor = Tensor::zeros({70, 96}, DType::BFloat16);
    // 2. the product
    fma_matmul_8bit(output_tensor, left_val.rounded_tensor, left_val.scales_tensor, right_val.rounded_tensor, right_val.scales_tensor, bias_tensor);
    // 3. check every row against bias
    torch::Tensor expected_tensor = pytorch_copy_of(bias_tensor).unsqueeze(0).expand({70, 96}).contiguous();
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
}

static void add_onto_adds_to_what_output_holds(void)
{
    // Expect: left the identity {128, 128}, right random {128, 128}, output starting at random values: with
    // add_onto the output is old + (right rounded)ᵀ, rounded once to bfloat16, as PyTorch's (old + (right
    // rounded)ᵀ) rounded to bfloat16.
    // 1. the tensors
    torch::manual_seed(3);
    RoundedMatrix left_val(identity_bf16(128));
    RoundedMatrix right_val(random_bf16_tensor({128, 128}));
    Tensor output_tensor = random_bf16_tensor({128, 128});
    torch::Tensor expected_tensor =
        (pytorch_copy_of(output_tensor) + right_val.turned_back().t()).to(torch::kBFloat16).to(torch::kFloat);
    // 2. the product, added onto
    fma_matmul_8bit(output_tensor, left_val.rounded_tensor, left_val.scales_tensor, right_val.rounded_tensor, right_val.scales_tensor,
           Tensor(), true);
    // 3. check
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
}

// ══ part 3: the kernel fits its launch ════════════════════════════════════════

static void the_kernel_fits_its_block(void)
{
    // Expect: fma_matmul_8bit_kernel can be launched with FMA_MATMUL_BF16_THREADS_PER_BLOCK threads, and nothing is spilled:
    // cudaFuncGetAttributes' numRegs (registers per thread), maxThreadsPerBlock (the largest block) and
    // localSizeBytes (0 if nothing spills), read without running anything.
    // 1. what the compiler wrote down
    cudaFuncAttributes attributes_val;
    cudaCheck(cudaFuncGetAttributes(&attributes_val, (const void *)fma_matmul_8bit_kernel));
    print_test_name(__func__);
    printf("    fma_matmul_8bit_kernel: %d registers per thread, largest block %d threads, %zu bytes local memory, %zu bytes shared memory\n",
           attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes, attributes_val.sharedSizeBytes);
    // 2. check
    ASSERT_EQ(attributes_val.maxThreadsPerBlock >= FMA_MATMUL_BF16_THREADS_PER_BLOCK, true);
    ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_qkv();
    matches_pytorch_on_gpt2_attention_output();
    matches_pytorch_on_gpt2_mlp_up();
    matches_pytorch_on_gpt2_mlp_down();
    matches_pytorch_on_gpt2_head();
    matches_pytorch_on_gpt2_qkv_weight_grad();
    matches_pytorch_with_sizes_no_tile_divides();
    stays_near_pytorch_on_the_values_before_rounding();

    times_the_identity_gives_right_turned_round();
    zero_left_gives_the_bias_in_every_row();
    add_onto_adds_to_what_output_holds();

    the_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
