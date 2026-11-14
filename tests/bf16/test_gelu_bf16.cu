// test_gelu_bf16.cu — tests for llmc/kernels/bf16/gelu_bf16.cuh (gelu_bf16_forward, gelu_bf16_backward): GPT-2's
// MLP GELU done inside the linear layers around it, as each pair of their products is stored (wgmma_matmul_bf16's
// epilogue). Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks against the unfused path (linear_bf16 then gelu.cuh: the products the same bits, GELU's results
// within one bfloat16 rounding) and against PyTorch's float linear and tanh GELU with autograd; part 2 checks
// numbers worked out by hand; part 3 checks that the products' kernels made with GELU's epilogues fit their block.
// Run it on an H100 (compute capability 9.0), built with -arch=sm_90a in place of -arch=native. GELU alone is
// tested in tests/test_gelu.cu. The timings are in tests/bf16/tune_gelu_bf16.cu.
#include <torch/types.h>              // torch::linear, torch::gelu, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/bf16/gelu_bf16.cuh"

// Against the unfused path: the same math on the same bfloat16 values, so the same bits but where the compiler
// orders GELU's float math another way inside the product's kernel: then by one bfloat16 rounding, 2^-8 of a
// value. Against PyTorch in float on the same bfloat16 values: each result rounded once or twice as it is stored.
// Each tolerance is the largest difference, as a fraction of the largest value.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;
static const float WEIGHT_STD = 0.02f;   // GPT-2's starting std: GELU's inputs of about 1, where it bends

#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation `std_val`, made by
/// PyTorch on the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list, float std_val = 1.0f)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * std_val;
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded.
static Tensor bf16_tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Gives a bfloat16 tensor's values as a CPU list of floats, exactly.
static std::vector<float> values_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
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
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// The tensors of GPT-2's MLP around GELU, in bfloat16, for one size: input {batch_size, seq_len, embed_dim}, the
/// first linear to mlp_dim (gelu_input, gelu_output), the second back to embed_dim (output_grad); the weights,
/// biases and output_grad random, the rest 0.
struct MlpTensors
{
    Tensor input_tensor;
    Tensor up_weight_tensor;
    Tensor up_bias_tensor;
    Tensor gelu_input_tensor;
    Tensor gelu_output_tensor;
    Tensor down_weight_tensor;
    Tensor output_grad_tensor;
    Tensor gelu_input_grad_tensor;
    Tensor down_weight_grad_tensor;
    Tensor down_bias_grad_tensor;

    /// Makes them, random from torch::manual_seed(0).
    MlpTensors(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val, size_t mlp_dim_val)
    {
        torch::manual_seed(0);
        input_tensor = random_bf16_tensor({batch_size_val, seq_len_val, embed_dim_val});
        up_weight_tensor = random_bf16_tensor({mlp_dim_val, embed_dim_val}, WEIGHT_STD);
        up_bias_tensor = random_bf16_tensor({mlp_dim_val}, WEIGHT_STD);
        gelu_input_tensor = Tensor::zeros({batch_size_val, seq_len_val, mlp_dim_val}, DType::BFloat16);
        gelu_output_tensor = Tensor::zeros(gelu_input_tensor.shape(), DType::BFloat16);
        down_weight_tensor = random_bf16_tensor({embed_dim_val, mlp_dim_val}, WEIGHT_STD);
        output_grad_tensor = random_bf16_tensor({batch_size_val, seq_len_val, embed_dim_val});
        gelu_input_grad_tensor = Tensor::zeros(gelu_input_tensor.shape(), DType::BFloat16);
        down_weight_grad_tensor = Tensor::zeros(down_weight_tensor.shape(), DType::BFloat16);
        down_bias_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    }
};

// ══ part 1: mine against the unfused path, and against PyTorch's ════════════════

/// Runs the fused forward and backward, and the unfused path on the same values (linear_bf16_forward then
/// gelu_forward; linear_bf16_backward into gelu_output_grad then gelu_backward). Checks: gelu_input,
/// down_weight_grad and down_bias_grad the same bits (the same products); gelu_output and gelu_input_grad within
/// ONE_ROUNDING_TOLERANCE. Then against PyTorch's float linear and tanh GELU with autograd on the same values: the
/// forward's two within BFLOAT16_TOLERANCE, the backward's three within BFLOAT16_GRAD_TOLERANCE. Prints the sizes.
static void fused_matches_unfused_and_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val,
                                              size_t embed_dim_val, size_t mlp_dim_val)
{
    // 1. the fused forward and backward
    MlpTensors fused_val(batch_size_val, seq_len_val, embed_dim_val, mlp_dim_val);
    gelu_bf16_forward(fused_val.gelu_input_tensor, fused_val.gelu_output_tensor, fused_val.input_tensor,
                      fused_val.up_weight_tensor, fused_val.up_bias_tensor);
    gelu_bf16_backward(fused_val.gelu_input_grad_tensor, fused_val.down_weight_grad_tensor, fused_val.down_bias_grad_tensor,
                       fused_val.output_grad_tensor, fused_val.gelu_output_tensor, fused_val.down_weight_tensor,
                       fused_val.gelu_input_tensor);

    // 2. the unfused path on the same values
    MlpTensors unfused_val(batch_size_val, seq_len_val, embed_dim_val, mlp_dim_val);
    Tensor gelu_output_grad_tensor = Tensor::zeros(unfused_val.gelu_input_tensor.shape(), DType::BFloat16);
    linear_bf16_forward(unfused_val.gelu_input_tensor, unfused_val.input_tensor, unfused_val.up_weight_tensor, unfused_val.up_bias_tensor);
    gelu_forward(unfused_val.gelu_output_tensor, unfused_val.gelu_input_tensor);
    linear_bf16_backward(gelu_output_grad_tensor, unfused_val.down_weight_grad_tensor, unfused_val.down_bias_grad_tensor,
                         unfused_val.output_grad_tensor, unfused_val.gelu_output_tensor, unfused_val.down_weight_tensor);
    gelu_backward(unfused_val.gelu_input_grad_tensor, gelu_output_grad_tensor, unfused_val.gelu_input_tensor);

    // 3. PyTorch's, in float on the same values: the forward from input, the backward from my gelu_input
    torch::Tensor pytorch_gelu_input_tensor = torch::linear(pytorch_copy_of(fused_val.input_tensor), pytorch_copy_of(fused_val.up_weight_tensor),
                                                            pytorch_copy_of(fused_val.up_bias_tensor));
    torch::Tensor pytorch_gelu_output_tensor = torch::gelu(pytorch_gelu_input_tensor, "tanh");
    torch::Tensor backward_gelu_input_tensor = pytorch_copy_of(fused_val.gelu_input_tensor).requires_grad_(true);
    torch::Tensor pytorch_down_weight_tensor = pytorch_copy_of(fused_val.down_weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_down_bias_tensor = torch::zeros({(int64_t)embed_dim_val}, torch::TensorOptions().device(torch::kCUDA)).requires_grad_(true);
    torch::Tensor pytorch_output_tensor = torch::linear(torch::gelu(backward_gelu_input_tensor, "tanh"), pytorch_down_weight_tensor,
                                                        pytorch_down_bias_tensor);
    pytorch_output_tensor.backward(pytorch_copy_of(fused_val.output_grad_tensor));

    // 4. print the sizes, then check: against the unfused path, then against PyTorch
    print_test_name(test_ptr);
    printf("    %zu rows, embed_dim %zu, mlp_dim %zu\n", batch_size_val * seq_len_val, embed_dim_val, mlp_dim_val);
    ASSERT_EQ(values_of(fused_val.gelu_input_tensor), values_of(unfused_val.gelu_input_tensor));
    ASSERT_EQ(values_of(fused_val.down_weight_grad_tensor), values_of(unfused_val.down_weight_grad_tensor));
    ASSERT_EQ(values_of(fused_val.down_bias_grad_tensor), values_of(unfused_val.down_bias_grad_tensor));
    ASSERT_NEAR(values_of(fused_val.gelu_output_tensor), values_of(unfused_val.gelu_output_tensor), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(values_of(fused_val.gelu_input_grad_tensor), values_of(unfused_val.gelu_input_grad_tensor), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(fused_val.gelu_input_tensor, pytorch_gelu_input_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(fused_val.gelu_output_tensor, pytorch_gelu_output_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(fused_val.gelu_input_grad_tensor, backward_gelu_input_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(fused_val.down_weight_grad_tensor, pytorch_down_weight_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(fused_val.down_bias_grad_tensor, pytorch_down_bias_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

static void matches_on_gpt2_mlp_sizes(void)
{
    // Expect: 4 rows of 64 tokens (256 rows), embed_dim 768, mlp_dim 3072 as GPT-2 small's MLP: the checks above.
    fused_matches_unfused_and_pytorch(__func__, 4, 64, GPT2_EMBED_DIM, GPT2_MLP_DIM);
}

static void matches_on_sizes_that_do_not_fill_the_squares(void)
{
    // Expect: 2 x 37 = 74 rows (64 + 10), embed_dim 96, mlp_dim 136 (2 x 64 + 8): part-full squares at the edges,
    // where the epilogue must neither read nor write past them. The same checks.
    fused_matches_unfused_and_pytorch(__func__, 2, 37, 96, 136);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// GELU (the tanh form) and its slope at the 8 values used below, worked out in double:
//
//     x          0       1          -1          2          -2          0.5        -0.5        3
//     gelu(x)    0       0.8411920  -0.1588080  1.9545977  -0.0454023  0.3457140  -0.1542860  2.9963626
//     gelu'(x)   0.5     1.0829641  -0.0829641  1.0860993  -0.0860993  0.8673699  0.1326301   1.0115842
//
// All 8 x hold exactly in bfloat16.

static const std::vector<float> X_LIST = {0, 1, -1, 2, -2, 0.5f, -0.5f, 3};
static const std::vector<float> GELU_LIST = {0, 0.8411920f, -0.1588080f, 1.9545977f, -0.0454023f, 0.3457140f, -0.1542860f, 2.9963626f};
static const std::vector<float> SLOPE_LIST = {0.5f, 1.0829641f, -0.0829641f, 1.0860993f, -0.0860993f, 0.8673699f, 0.1326301f, 1.0115842f};

/// Gives `row_list` again `rows_val` times, one after the other: the same row for every token.
static std::vector<float> rows_of(const std::vector<float> &row_list, size_t rows_val)
{
    std::vector<float> values_list;
    for (size_t row_index_val = 0; row_index_val < rows_val; row_index_val++)
        values_list.insert(values_list.end(), row_list.begin(), row_list.end());
    return values_list;
}

/// Makes the identity matrix {size_val, size_val} in bfloat16: 1 where the row is the column, 0 elsewhere.
static Tensor identity(size_t size_val)
{
    std::vector<float> values_list(size_val * size_val, 0.0f);
    for (size_t diagonal_index_val = 0; diagonal_index_val < size_val; diagonal_index_val++)
        values_list[diagonal_index_val * size_val + diagonal_index_val] = 1.0f;
    return bf16_tensor_of({size_val, size_val}, values_list);
}

static void gelu_and_its_slope_at_whole_and_half_numbers(void)
{
    // Expect: 16 rows, the first linear 8 -> 8 with input all 0 and bias the 8 x above, so every row of gelu_input
    // is x exactly and gelu_output is gelu(x), rounded to bfloat16 (within 1e-2 of the largest, 2.996). The
    // second linear 8 -> 8 with weight the identity and output_grad all 1: gelu_output's gradient is 1 everywhere,
    // so gelu_input_grad is gelu'(x), rounded (within 1e-2 of the largest, 1.086).
    // 1. the forward
    Tensor input_tensor = Tensor::zeros({1, 16, 8}, DType::BFloat16);
    Tensor gelu_input_tensor = Tensor::zeros({1, 16, 8}, DType::BFloat16);
    Tensor gelu_output_tensor = Tensor::zeros({1, 16, 8}, DType::BFloat16);
    gelu_bf16_forward(gelu_input_tensor, gelu_output_tensor, input_tensor, identity(8), bf16_tensor_of({8}, X_LIST));
    // 2. the backward
    Tensor gelu_input_grad_tensor = Tensor::zeros({1, 16, 8}, DType::BFloat16);
    Tensor down_weight_grad_tensor = Tensor::zeros({8, 8}, DType::BFloat16);
    gelu_bf16_backward(gelu_input_grad_tensor, down_weight_grad_tensor, Tensor(), Tensor::ones({1, 16, 8}).to(DType::BFloat16),
                       gelu_output_tensor, identity(8), gelu_input_tensor);
    // 3. check
    ASSERT_EQ(values_of(gelu_input_tensor), rows_of(X_LIST, 16));
    ASSERT_NEAR(values_of(gelu_output_tensor), rows_of(GELU_LIST, 16), 1e-2);
    ASSERT_NEAR(values_of(gelu_input_grad_tensor), rows_of(SLOPE_LIST, 16), 1e-2);
}

static void backward_writes_gelu_input_grad_and_adds_onto_weight_grad_and_bias_grad(void)
{
    // Expect: 16 rows, the second linear 16 -> 8: output_grad and weight all 1, gelu_output all 1, gelu_input all 0
    // (gelu'(0) = 0.5); every gradient starts at 10 (as if an earlier micro-batch had left it there).
    // gelu_input_grad is written over: (sum over 8 outputs of 1 x 1) x 0.5 = 4. down_weight_grad and down_bias_grad
    // are added onto: each a sum over 16 rows of 1, so 10 + 16 = 26. A second backward writes gelu_input_grad again
    // (4) and adds again (42). All exact in bfloat16.
    // 1. gradients that start at 10
    Tensor output_grad_tensor = Tensor::ones({1, 16, 8}).to(DType::BFloat16);
    Tensor gelu_output_tensor = Tensor::ones({1, 16, 16}).to(DType::BFloat16);
    Tensor down_weight_tensor = Tensor::ones({8, 16}).to(DType::BFloat16);
    Tensor gelu_input_tensor = Tensor::zeros({1, 16, 16}, DType::BFloat16);
    Tensor gelu_input_grad_tensor = Tensor::zeros({1, 16, 16}, DType::BFloat16).fill_(10);
    Tensor down_weight_grad_tensor = Tensor::zeros({8, 16}, DType::BFloat16).fill_(10);
    Tensor down_bias_grad_tensor = Tensor::zeros({8}, DType::BFloat16).fill_(10);
    // 2. one backward, then a second
    gelu_bf16_backward(gelu_input_grad_tensor, down_weight_grad_tensor, down_bias_grad_tensor, output_grad_tensor,
                       gelu_output_tensor, down_weight_tensor, gelu_input_tensor);
    ASSERT_EQ(values_of(gelu_input_grad_tensor), std::vector<float>(16 * 16, 4.0f));
    ASSERT_EQ(values_of(down_weight_grad_tensor), std::vector<float>(8 * 16, 26.0f));
    ASSERT_EQ(values_of(down_bias_grad_tensor), std::vector<float>(8, 26.0f));
    gelu_bf16_backward(gelu_input_grad_tensor, down_weight_grad_tensor, down_bias_grad_tensor, output_grad_tensor,
                       gelu_output_tensor, down_weight_tensor, gelu_input_tensor);
    ASSERT_EQ(values_of(gelu_input_grad_tensor), std::vector<float>(16 * 16, 4.0f));
    ASSERT_EQ(values_of(down_weight_grad_tensor), std::vector<float>(8 * 16, 42.0f));
    ASSERT_EQ(values_of(down_bias_grad_tensor), std::vector<float>(8, 42.0f));
}

static void gelu_input_grad_can_be_gelu_input_itself(void)
{
    // Expect: the backward with gelu_input_grad = gelu_input (no tensor of its own): each pair of gelu_input is read
    // by the thread that then writes the same pair, so the result is the same bits as with gelu_input_grad apart.
    // GPT-2's MLP sizes, 256 rows.
    // 1. a forward, then a backward into gelu_input_grad apart
    MlpTensors mlp_val(4, 64, GPT2_EMBED_DIM, GPT2_MLP_DIM);
    gelu_bf16_forward(mlp_val.gelu_input_tensor, mlp_val.gelu_output_tensor, mlp_val.input_tensor, mlp_val.up_weight_tensor,
                      mlp_val.up_bias_tensor);
    gelu_bf16_backward(mlp_val.gelu_input_grad_tensor, mlp_val.down_weight_grad_tensor, mlp_val.down_bias_grad_tensor,
                       mlp_val.output_grad_tensor, mlp_val.gelu_output_tensor, mlp_val.down_weight_tensor, mlp_val.gelu_input_tensor);
    // 2. the same backward into gelu_input itself
    Tensor down_weight_grad_tensor = Tensor::zeros(mlp_val.down_weight_tensor.shape(), DType::BFloat16);
    gelu_bf16_backward(mlp_val.gelu_input_tensor, down_weight_grad_tensor, Tensor(), mlp_val.output_grad_tensor,
                       mlp_val.gelu_output_tensor, mlp_val.down_weight_tensor, mlp_val.gelu_input_tensor);
    // 3. check: the same bits
    ASSERT_EQ(values_of(mlp_val.gelu_input_tensor), values_of(mlp_val.gelu_input_grad_tensor));
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: wgmma_matmul_bf16_kernel made with GELU's epilogues, the forward's (neither side turned: mlp_up's
    // output) and the backward's (right turned: mlp_down's input_grad), can be launched with the product's block (its producer and
    // consumer warpgroups), and nothing is spilled: the epilogue adds a few registers to the 32 sums. cudaFuncGetAttributes reads
    // what the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const int kernel_count_val = 2;
    const void *kernels_list[kernel_count_val] = {(const void *)wgmma_matmul_bf16_kernel<false, false, GeluBf16ForwardEpilogue>,
                                                  (const void *)wgmma_matmul_bf16_kernel<false, true, GeluBf16BackwardEpilogue>};
    const char *names_list[kernel_count_val] = {"wgmma_matmul_bf16_kernel<false, false, GeluBf16ForwardEpilogue>",
                                                "wgmma_matmul_bf16_kernel<false, true, GeluBf16BackwardEpilogue>"};
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
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >=
                      (WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS + WGMMA_MATMUL_BF16_ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS,
                  true);   // the product's block: its producer and consumer warpgroups
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_on_gpt2_mlp_sizes();
    matches_on_sizes_that_do_not_fill_the_squares();

    gelu_and_its_slope_at_whole_and_half_numbers();
    backward_writes_gelu_input_grad_and_adds_onto_weight_grad_and_bias_grad();
    gelu_input_grad_can_be_gelu_input_itself();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
