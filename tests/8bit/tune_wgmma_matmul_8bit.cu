// tune_wgmma_matmul_8bit.cu — the timings of llmc/kernels/8bit/wgmma_matmul_8bit.cuh, apart from tests/8bit/test_wgmma_matmul_8bit.cu so the tests only check
// numbers: time_20_products_of_each_shape. Mine (E4M3 tensor cores, by WGMMA and TMA) on the 8-bit copies, with the rounding that makes them
// (round_8bit), against wgmma_matmul_bf16.cuh (bfloat16
// tensor cores, the same steps), mma_matmul_8bit.cuh (E4M3 by mma.sync) and PyTorch's bfloat16 matmul (cuBLAS), on GPT-2 small's 5 linear layers and the worklog's 4096³;
// then time_each_promotion_on_long_sums: every how many steps the tensor cores' sum moves into float (0: never),
// each timed and checked against PyTorch on the two longest sums of a step. The rounding's own
// launches are in tests/8bit/tune_round_8bit.cu.
//
// Built as tests/8bit/test_wgmma_matmul_8bit.cu is, with -arch=sm_90a (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/wgmma_matmul_8bit_<date>_<time>.txt. Needs an H100.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, peak_bf16_flops, print_gpu
#include "../../llmc/kernels/8bit/wgmma_matmul_8bit.cuh"
#include "../../llmc/kernels/bf16/wgmma_matmul_bf16.cuh" // wgmma_matmul_bf16: the same product in bfloat16, to see what 8 bits add
#include "../../llmc/kernels/8bit/mma_matmul_8bit.cuh" // mma_matmul_8bit: the same 8-bit copies by mma.sync

static const double BFLOAT16_TOLERANCE = 1e-2; // mine against PyTorch on the rounded values, as a fraction of the largest

#define GPT2_ROWS 4096                  // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768              // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define WORKLOG_SIZE 4096               // the worklog's M = N = K
#define FLOPS_PER_MULTIPLY_ADD 2        // a product's flops: 2 x rows x columns x sum_length

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

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU, bfloat16 turned into
/// floats exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives a matrix's E4M3s turned back by PyTorch: each (read as float8_e4m3fn) times its row's scale, exact.
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scales_tensor, int64_t rows_val,
                                 int64_t row_length_val)
{
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, row_length_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
}

/// One product's shape: output {rows, columns} = left {rows, sum_length} · right {columns, sum_length}ᵀ.
struct ProductShape
{
    const char *name_ptr;
    size_t rows_val;
    size_t columns_val;
    size_t sum_length_val;
};

// ══ part 1: timed against wgmma_matmul_bf16, mma_matmul_8bit and PyTorch ═══════════════════════════════

/// Prints the averages of 20 products of one shape, steps 3..20: the rounding of left and of right, mine on the
/// 8-bit copies, the bfloat16 sides on the bfloat16 values; TFLOPS and the share of this GPU's bfloat16 peak. Then
/// checks mine against PyTorch's float product of the rounded values.
static void time_20_products_of_one_shape(const ProductShape &shape_val)
{
    // 1. random bfloat16 left and right, their 8-bit copies, the outputs
    torch::manual_seed(0);
    Tensor left_tensor = random_bf16_tensor({shape_val.rows_val, shape_val.sum_length_val});
    Tensor right_tensor = random_bf16_tensor({shape_val.columns_val, shape_val.sum_length_val});
    Tensor left_8bit_tensor = Tensor::zeros({shape_val.rows_val, shape_val.sum_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor right_8bit_tensor = Tensor::zeros({shape_val.columns_val, shape_val.sum_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor left_scales_tensor = Tensor::zeros({shape_val.rows_val});
    Tensor right_scales_tensor = Tensor::zeros({shape_val.columns_val});
    Tensor output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    Tensor other_output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    torch::Tensor pytorch_left_tensor = pytorch_copy_of(left_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_right_tensor = pytorch_copy_of(right_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS of each; the first WARM_UP_STEPS do first-time work, so the averages are of the rest
    const char *sides_list[] = {"rounding left", "rounding right", "mine (E4M3)", "wgmma_matmul_bf16 (bf16)", "mma_matmul_8bit (E4M3, mma)", "PyTorch (cuBLAS, bf16)"};
    const int sides_val = (int)(sizeof(sides_list) / sizeof(sides_list[0]));
    double totals_list[sides_val] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[sides_val];
        int side_index_val = 0;
        times_list[side_index_val++] = gpu_milliseconds([&] { round_8bit(left_8bit_tensor, left_scales_tensor, left_tensor); });
        times_list[side_index_val++] = gpu_milliseconds([&] { round_8bit(right_8bit_tensor, right_scales_tensor, right_tensor); });
        times_list[side_index_val++] = gpu_milliseconds([&] {
            wgmma_matmul_8bit(output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor);
        });
        times_list[side_index_val++] = gpu_milliseconds([&] { wgmma_matmul_bf16(other_output_tensor, left_tensor, right_tensor); });
        times_list[side_index_val++] = gpu_milliseconds([&] { mma_matmul_8bit(other_output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor); });
        times_list[side_index_val++] = gpu_milliseconds([&] { pytorch_output_tensor = pytorch_left_tensor.matmul(pytorch_right_tensor.t()); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (side_index_val = 0; side_index_val < sides_val; side_index_val++)
            totals_list[side_index_val] += times_list[side_index_val];
    }

    // 3. print the averages: ms; for the products TFLOPS and the share of the bfloat16 peak
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double flops_val = (double)FLOPS_PER_MULTIPLY_ADD * shape_val.rows_val * shape_val.columns_val * shape_val.sum_length_val;
    double peak_val = peak_bf16_flops();
    print_test_name(__func__);
    printf("    %s: left {%zu, %zu}, right {%zu, %zu}: %.1f GFLOP\n", shape_val.name_ptr, shape_val.rows_val,
           shape_val.sum_length_val, shape_val.columns_val, shape_val.sum_length_val, flops_val / GIGA);
    printf("    average of steps %d..%d:                ms     TFLOPS   of bf16 peak\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    for (int side_index_val = 0; side_index_val < sides_val; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        if (side_index_val < 2)
        {
            printf("        %-30s %9.3f\n", sides_list[side_index_val], milliseconds_val);
            continue;
        }
        double tflops_val = flops_val / (milliseconds_val / MILLISECONDS_PER_SECOND) / TERA;
        printf("        %-30s %9.3f   %8.1f   %6.1f%%\n", sides_list[side_index_val], milliseconds_val, tflops_val,
               100.0 * tflops_val * TERA / peak_val);
    }
    printf("    PyTorch takes %.2fx as long as mine; with right rounded once and left each time, mine and the rounding of "
           "left take %.3f ms\n",
           totals_list[sides_val - 1] / totals_list[2], (totals_list[0] + totals_list[2]) / averaged_val);

    // 4. mine agrees with PyTorch on the rounded values at the end
    torch::Tensor expected_tensor =
        turned_back(left_8bit_tensor, left_scales_tensor, (int64_t)shape_val.rows_val, (int64_t)shape_val.sum_length_val)
            .matmul(turned_back(right_8bit_tensor, right_scales_tensor, (int64_t)shape_val.columns_val,
                                (int64_t)shape_val.sum_length_val).t());
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

static void time_20_products_of_each_shape(void)
{
    // Prints, for each of GPT-2 small's 5 forward products (a micro-batch of 4 x 1024 rows) and the worklog's
    // 4096³, the rounding's and the products' times (at the default promotion; part 2 sweeps it). An H100's E4M3 tensor cores have twice its bfloat16 ones' flops, so the
    // share of the bfloat16 peak can pass 100%; each 8-bit tile is half the bytes, so TMA brings twice the values a step.
    const ProductShape shapes_list[] = {
        {"qkv", GPT2_ROWS, GPT2_QKV_DIM, GPT2_EMBED_DIM},
        {"attention output", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM},
        {"mlp up", GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM},
        {"mlp down", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM},
        {"head", GPT2_ROWS, GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM},
        {"worklog 4096^3", WORKLOG_SIZE, WORKLOG_SIZE, WORKLOG_SIZE},
    };
    for (const ProductShape &shape_val : shapes_list)
        time_20_products_of_one_shape(shape_val);
}

// ══ part 2: the promotion swept ═══════════════════════════════════════════════

/// Prints, for one long sum, each promotion_steps (0: the whole sum in the tensor cores) with its time, averaged
/// over SWEEP_ROUNDS rounds that take turns, rounds WARM_UP_STEPS + 1 on, and its largest difference from
/// PyTorch's float product of the same 8-bit copies, as a fraction of the largest value; then the fastest
/// within BFLOAT16_TOLERANCE.
static void time_each_promotion_of_one_shape(const ProductShape &shape_val)
{
    // 1. random bfloat16 left and right, rounded once; PyTorch's float product of the copies
    torch::manual_seed(6);
    Tensor left_tensor = random_bf16_tensor({shape_val.rows_val, shape_val.sum_length_val});
    Tensor right_tensor = random_bf16_tensor({shape_val.columns_val, shape_val.sum_length_val});
    Tensor left_8bit_tensor = Tensor::zeros({shape_val.rows_val, shape_val.sum_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor right_8bit_tensor = Tensor::zeros({shape_val.columns_val, shape_val.sum_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor left_scales_tensor = Tensor::zeros({shape_val.rows_val});
    Tensor right_scales_tensor = Tensor::zeros({shape_val.columns_val});
    round_8bit(left_8bit_tensor, left_scales_tensor, left_tensor);
    round_8bit(right_8bit_tensor, right_scales_tensor, right_tensor);
    torch::Tensor expected_tensor =
        turned_back(left_8bit_tensor, left_scales_tensor, (int64_t)shape_val.rows_val, (int64_t)shape_val.sum_length_val)
            .matmul(turned_back(right_8bit_tensor, right_scales_tensor, (int64_t)shape_val.columns_val,
                                (int64_t)shape_val.sum_length_val).t());

    // 2. each promotion's product, kept for its check; then SWEEP_ROUNDS rounds, each timing every promotion once
    const int promotions_list[] = {0, 1, 2, 4, 8, 16};
    const int promotion_count_val = (int)(sizeof(promotions_list) / sizeof(promotions_list[0]));
    std::vector<Tensor> outputs_list;
    for (int promotion_index_val = 0; promotion_index_val < promotion_count_val; promotion_index_val++)
    {
        outputs_list.push_back(Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16));
        wgmma_matmul_8bit(outputs_list.back(), left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor,
                          Tensor(), false, WgmmaMatmul8bitStore(), promotions_list[promotion_index_val]);
    }
    double totals_list[promotion_count_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int promotion_index_val = 0; promotion_index_val < promotion_count_val; promotion_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] {
                wgmma_matmul_8bit(outputs_list[promotion_index_val], left_8bit_tensor, left_scales_tensor, right_8bit_tensor,
                                  right_scales_tensor, Tensor(), false, WgmmaMatmul8bitStore(), promotions_list[promotion_index_val]);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[promotion_index_val] += milliseconds_val;
        }

    // 3. print each: time and difference; then the fastest within BFLOAT16_TOLERANCE
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    %s: left {%zu, %zu}, right {%zu, %zu}: %zu steps of %d values of k\n", shape_val.name_ptr, shape_val.rows_val,
           shape_val.sum_length_val, shape_val.columns_val, shape_val.sum_length_val,
           (size_t)CEIL_DIV(shape_val.sum_length_val, (size_t)WGMMA_MATMUL_8BIT_BLOCK_K), WGMMA_MATMUL_8BIT_BLOCK_K);
    printf("    promotion steps        ms   largest difference from PyTorch on the copies\n");
    int fastest_index_val = -1;
    for (int promotion_index_val = 0; promotion_index_val < promotion_count_val; promotion_index_val++)
    {
        double milliseconds_val = totals_list[promotion_index_val] / averaged_val;
        double difference_val = relative_difference_on_gpu(outputs_list[promotion_index_val], expected_tensor);
        bool within_val = difference_val <= BFLOAT16_TOLERANCE;
        printf("    %15d %9.3f   %.5f%s%s\n", promotions_list[promotion_index_val], milliseconds_val, difference_val,
               within_val ? "" : "   past the tolerance",
               promotions_list[promotion_index_val] == WGMMA_MATMUL_8BIT_PROMOTION_STEPS ? "   <- the default" : "");
        if (within_val && (fastest_index_val < 0 || totals_list[promotion_index_val] < totals_list[fastest_index_val]))
            fastest_index_val = promotion_index_val;
    }
    if (fastest_index_val >= 0)
        printf("    fastest within %.0e: promotion every %d steps, %.3f ms\n", BFLOAT16_TOLERANCE,
               promotions_list[fastest_index_val], totals_list[fastest_index_val] / averaged_val);
    else
        printf("    none within %.0e\n", BFLOAT16_TOLERANCE);
}

static void time_each_promotion_on_long_sums(void)
{
    // Prints the promotion sweep on the two long sums of an 8-bit step: the head's input_grad (logits_grad {rows,
    // 50304} · wte, 50,304 terms a sum) and a weight_grad at 64 rows a micro-batch (mlp up: output_gradᵀ {3072,
    // 65536} · input {65536, 768}, one term per token).
    const ProductShape shapes_list[] = {
        {"head input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_PADDED_VOCAB_SIZE},
        {"mlp up weight_grad, 64 rows of 1024 tokens", GPT2_MLP_DIM, GPT2_EMBED_DIM, 16 * GPT2_ROWS},
    };
    for (const ProductShape &shape_val : shapes_list)
        time_each_promotion_of_one_shape(shape_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_products_of_each_shape();
    time_each_promotion_on_long_sums();

    printf("\nall tunes done\n");
    return 0;
}
