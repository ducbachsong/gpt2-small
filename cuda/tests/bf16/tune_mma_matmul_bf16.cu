// tune_mma_matmul_bf16.cu — the timings of cuda/kernels/bf16/mma_matmul_bf16.cuh, apart from cuda/tests/bf16/test_mma_matmul_bf16.cu so the tests only check
// numbers: time_20_products_of_each_shape. Mine (the tensor cores, by mma.sync) against fma_matmul_bf16.cuh (the CUDA cores) and PyTorch's
// bfloat16 matmul (cuBLAS), on GPT-2 small's 5 linear layers, each layer's three
// products (the forward, and the backward's input_grad and weight_grad, with left or right turned round), and
// the worklog's 4096 x 4096 x 4096.
//
// Built as cuda/tests/bf16/test_mma_matmul_bf16.cu is (see cuda/tests/README.md); what it prints also goes to
// cuda/tests/results/<GPU>/mma_matmul_bf16_<date>_<time>.txt. Needs an sm_80 GPU or later.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, peak_bf16_flops, print_gpu
#include "../../kernels/bf16/mma_matmul_bf16.cuh"
#include "../../kernels/bf16/fma_matmul_bf16.cuh" // fma_matmul_bf16: the CUDA cores, to see what the tensor cores add

static const double BFLOAT16_TOLERANCE = 1e-2; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define WORKLOG_SIZE 4096             // the worklog's M = N = K
#define FLOPS_PER_MULTIPLY_ADD 2      // a product's flops: 2 x rows x columns x sum_length

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    // Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
    // on the GPU, then rounded to bfloat16.
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   bfloat16, on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU, bfloat16 turned into floats exactly.
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

/// One product's shape: output {rows, columns} = L · Rᵀ, each side as it sits or turned round.
struct ProductShape
{
    // One product's shape: output {rows, columns} = L {rows, sum_length} · R {columns, sum_length}ᵀ, left and right
    // holding L and R as they sit, or turned round.
    const char *name_ptr;
    size_t rows_val;
    size_t columns_val;
    size_t sum_length_val;
    bool left_turned_val;
    bool right_turned_val;
};

// ══ part 1: timed against PyTorch and fma_matmul_bf16 ═══════════════════════════════

/// Prints the averages of 20 products of one shape for each side, then checks mine against PyTorch.
///
///   shape_val   the product's shape
static void time_20_products_of_one_shape(const ProductShape &shape_val)
{
    // Prints the averages of 20 products of one shape, steps 3..20, for each side, with TFLOPS and the share of this
    // GPU's bfloat16 peak (peak_bf16_flops); then checks mine against PyTorch's within BFLOAT16_TOLERANCE.
    // 1. random bfloat16 left and right as they sit; the sides' outputs
    torch::manual_seed(0);
    bool left_turned_val = shape_val.left_turned_val;
    bool right_turned_val = shape_val.right_turned_val;
    Tensor left_tensor = random_bf16_tensor(left_turned_val ? Shape{shape_val.sum_length_val, shape_val.rows_val}
                                                            : Shape{shape_val.rows_val, shape_val.sum_length_val});
    Tensor right_tensor = random_bf16_tensor(right_turned_val ? Shape{shape_val.sum_length_val, shape_val.columns_val}
                                                              : Shape{shape_val.columns_val, shape_val.sum_length_val});
    Tensor output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    Tensor other_output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    torch::Tensor pytorch_left_tensor = pytorch_copy_of(left_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_right_tensor = pytorch_copy_of(right_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_left_matrix_tensor = left_turned_val ? pytorch_left_tensor.t() : pytorch_left_tensor;     // views, no copies:
    torch::Tensor pytorch_right_matrix_tensor = right_turned_val ? pytorch_right_tensor.t() : pytorch_right_tensor; // cuBLAS reads them turned
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS products of each side; the first WARM_UP_STEPS do first-time work (loading kernels,
    //    cuBLAS picking its own), so the averages are of the rest
    const char *sides_list[] = {"mine (mma.sync)", "fma_matmul_bf16 (CUDA cores)", "PyTorch (cuBLAS)"};
    const int sides_val = (int)(sizeof(sides_list) / sizeof(sides_list[0]));
    double totals_list[sides_val] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[sides_val];
        times_list[0] = gpu_milliseconds([&] { mma_matmul_bf16(output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val, right_turned_val); });
        times_list[1] = gpu_milliseconds([&] { fma_matmul_bf16(other_output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val, right_turned_val); });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_output_tensor = pytorch_left_matrix_tensor.matmul(pytorch_right_matrix_tensor.t());
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int side_index_val = 0; side_index_val < sides_val; side_index_val++)
            totals_list[side_index_val] += times_list[side_index_val];
    }

    // 3. print the averages: ms, TFLOPS, and the share of the peak
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double flops_val = (double)FLOPS_PER_MULTIPLY_ADD * shape_val.rows_val * shape_val.columns_val * shape_val.sum_length_val;
    double peak_val = peak_bf16_flops();
    print_test_name(__func__);
    printf("    %s: rows %zu, columns %zu, sum_length %zu%s%s: %.1f GFLOP\n", shape_val.name_ptr, shape_val.rows_val,
           shape_val.columns_val, shape_val.sum_length_val, left_turned_val ? ", left turned" : "",
           right_turned_val ? ", right turned" : "", flops_val / GIGA);
    printf("    average of steps %d..%d:                ms     TFLOPS   of bf16 peak\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    for (int side_index_val = 0; side_index_val < sides_val; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        double tflops_val = flops_val / (milliseconds_val / MILLISECONDS_PER_SECOND) / TERA;
        printf("        %-30s %9.3f   %8.1f   %6.1f%%\n", sides_list[side_index_val], milliseconds_val, tflops_val,
               100.0 * tflops_val * TERA / peak_val);
    }
    printf("    PyTorch takes %.2fx as long as mine\n", totals_list[sides_val - 1] / totals_list[0]);

    // 4. mine agrees with PyTorch at the end
    torch::Tensor expected_tensor = pytorch_left_matrix_tensor.to(torch::kFloat).matmul(pytorch_right_matrix_tensor.to(torch::kFloat).t());
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

static void time_20_products_of_each_shape(void)
{
    // Prints, for each of GPT-2 small's 5 linear layers (a micro-batch of 4 x 1024 rows), its forward product and
    // the backward's two (input_grad = output_grad · W: right turned; weight_grad = output_gradᵀ · x: both turned,
    // the sum over the 4096 rows), then the worklog's 4096³, each side's times.
    const ProductShape shapes_list[] = {
        {"qkv", GPT2_ROWS, GPT2_QKV_DIM, GPT2_EMBED_DIM, false, false},
        {"qkv input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_QKV_DIM, false, true},
        {"qkv weight_grad", GPT2_QKV_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true},
        {"attention output", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, false, false},
        {"attention output input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, false, true},
        {"attention output weight_grad", GPT2_EMBED_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true},
        {"mlp up", GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, false, false},
        {"mlp up input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, false, true},
        {"mlp up weight_grad", GPT2_MLP_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true},
        {"mlp down", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, false, false},
        {"mlp down input_grad", GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, false, true},
        {"mlp down weight_grad", GPT2_EMBED_DIM, GPT2_MLP_DIM, GPT2_ROWS, true, true},
        {"head", GPT2_ROWS, GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, false, false},
        {"head input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_PADDED_VOCAB_SIZE, false, true},
        {"head weight_grad", GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, GPT2_ROWS, true, true},
        {"worklog 4096^3", WORKLOG_SIZE, WORKLOG_SIZE, WORKLOG_SIZE, false, false},
    };
    for (const ProductShape &shape_val : shapes_list)
        time_20_products_of_one_shape(shape_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and cuda/tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_products_of_each_shape();

    printf("\nall tunes done\n");
    return 0;
}
