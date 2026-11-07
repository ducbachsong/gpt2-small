// test_gelu.cu — tests for llmc/kernels/gelu.cuh (gelu_forward, gelu_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's gelu (tanh), with autograd for the backward; part 2 checks
// numbers worked out by hand; part 3 checks that each kernel's registers fit its block, and that any block
// size gives the same bits. The timings are in tunes/tune_gelu.cu.
// (The C# trainer had no GELU test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::gelu, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/gelu.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so one
// rounding moves a value by up to 2^-8, about 0.4% of it. Twice that.
static const double BFLOAT16_TOLERANCE = 8e-3;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for the MLP's 12.6M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values, mine and PyTorch's gelu (tanh) with autograd,
/// and checks that output and input_grad agree within TOLERANCE.
static void forward_and_backward_match_pytorch(const Shape &shape_list)
{
    // 1. random inputs; mine
    torch::manual_seed(0);
    Tensor input_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_tensor = Tensor::zeros(shape_list);
    Tensor input_grad_tensor = Tensor::zeros(shape_list);
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, output_grad_tensor, input_tensor);

    // 2. PyTorch's, with autograd
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_tensor = torch::gelu(pytorch_input_tensor, "tanh");
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor));

    // 3. check
    ASSERT_NEAR(output_tensor.tolist(), list_of(pytorch_output_tensor), TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), TOLERANCE);
}

static void matches_pytorch_on_mlp_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens of 3072, the MLP's middle in GPT-2 small:
    // 49,152 values, 12,288 float4s, 48 full blocks. output and input_grad equal PyTorch's gelu with
    // approximate = "tanh" and its autograd backward, within 1e-5.
    forward_and_backward_match_pytorch({2, 8, 3072});
}

static void matches_pytorch_with_a_part_full_last_block(void)
{
    // Expect: {3, 7, 100}: 2100 values, 525 float4s, 3 blocks of 256 threads; the last has 13 float4s
    // for 256 threads, and its other 243 threads must write nothing. The same as PyTorch within 1e-5.
    forward_and_backward_match_pytorch({3, 7, 100});
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: {2, 8, 3072} random values rounded to bfloat16, in bfloat16 tensors. Each value is read as a
    // float, its gelu and gelu' worked out in float, and the result rounded once as it is stored. So output
    // and input_grad are PyTorch's float gelu of the same values, within one bfloat16 rounding:
    // BFLOAT16_TOLERANCE.

    // 1. random inputs in bfloat16; mine
    torch::manual_seed(0);
    Shape shape_list = {2, 8, 3072};
    Tensor input_tensor = random_tensor_on_gpu(shape_list).to(DType::BFloat16);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list).to(DType::BFloat16);
    Tensor output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, output_grad_tensor, input_tensor);

    // 2. PyTorch's, in float on the same values
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_output_tensor = torch::gelu(pytorch_input_tensor, "tanh");
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor.to(DType::Float32)));

    // 3. check, within one bfloat16 rounding
    ASSERT_NEAR(output_tensor.to(DType::Float32).tolist(), list_of(pytorch_output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_input_tensor.grad()), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the 4 values 1, -1, 2, -2 (one float4: thread 0 has it all). x = 1 works out as:
//
//     inner    = 0.7978846 x (1 + 0.044715)                     = 0.8335620
//     tanh     = tanh(0.8335620)                                = 0.6823840
//     gelu     = 0.5 x 1 x (1 + 0.6823840)                      = 0.8411920
//     gelu'    = 0.5 x (1 + 0.6823840)                          = 0.8411920
//              + 0.5 x 1 x (1 - 0.6823840²) x 0.7978846 x (1 + 3 x 0.044715)
//              = 0.8411920 + 0.5 x 0.5343521 x 0.9049168        = 1.0829641

static const std::vector<float> GELU_OF_1_MINUS_1_2_MINUS_2 = {0.8411920f, -0.1588080f, 1.9545977f, -0.0454023f};
static const std::vector<float> SLOPE_AT_1_MINUS_1_2_MINUS_2 = {1.0829641f, -0.0829641f, 1.0860993f, -0.0860993f};

static void gelu_of_0_is_0_and_its_slope_is_one_half(void)
{
    // Expect: x = 0: inner = 0, tanh = 0, so gelu = 0.5 x 0 x 1 = 0 and gelu' = 0.5 x 1 + 0 = 0.5, both
    // exactly. With output_grad 1, input_grad is the slope: 0.5.
    // 1. the forward and the backward at 0
    Tensor input_tensor = Tensor::zeros({4});
    Tensor output_tensor = Tensor::ones({4});
    Tensor input_grad_tensor = Tensor::zeros({4});
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, Tensor::ones({4}), input_tensor);
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>(4, 0.5f));
}

static void gelu_and_its_slope_at_1_minus_1_2_minus_2(void)
{
    // Expect: the values worked out above for x = 1, and the same way for -1, 2, -2:
    //
    //     x        1           -1           2           -2
    //     gelu     0.8411920   -0.1588080   1.9545977   -0.0454023
    //     gelu'    1.0829641   -0.0829641   1.0860993   -0.0860993
    //
    // With output_grad 1, input_grad is gelu'. The slope goes above 1 just past 0 and below 0 just
    // under it: GELU dips a little under 0 for negative x (lowest about -0.17 at x = -0.75) before
    // rising to 0 at -infinity.
    // 1. the forward and the backward
    Tensor input_tensor = tensor_of({4}, {1, -1, 2, -2});
    Tensor output_tensor = Tensor::zeros({4});
    Tensor input_grad_tensor = Tensor::zeros({4});
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, Tensor::ones({4}), input_tensor);
    // 2. check
    ASSERT_NEAR(output_tensor.tolist(), GELU_OF_1_MINUS_1_2_MINUS_2, 1e-6);
    ASSERT_NEAR(input_grad_tensor.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
}

static void large_values_pass_through_and_large_negatives_give_0(void)
{
    // Expect: x = 10, -10, 100, -100. inner is about +-43.7 at 10, and tanh of it is 1 or -1 in a
    // float, so gelu = 0.5 x (1 + 1) = x, or 0.5 x (1 - 1) = 0, exactly: 10 0 100 0 (the 0s print as
    // -0: 0.5 x -10 x 0, which equals 0). The slopes are then 0.5 x (1 +- 1) + 0 = 1 0 1 0, exactly.
    // 1. the forward and the backward
    Tensor input_tensor = tensor_of({4}, {10, -10, 100, -100});
    Tensor output_tensor = Tensor::zeros({4});
    Tensor input_grad_tensor = Tensor::zeros({4});
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, Tensor::ones({4}), input_tensor);
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({10, 0, 100, 0}));
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>({1, 0, 1, 0}));
}

static void gelu_of_x_minus_gelu_of_minus_x_is_x(void)
{
    // Expect: tanh(-inner) = -tanh(inner), so
    //
    //     gelu(x) - gelu(-x) = 0.5 x (1 + tanh) + 0.5 x (1 - tanh) = x
    //     gelu'(x) + gelu'(-x) = 1                                    (the slope of the line above)
    //
    // Row 0 holds 0.5 1 2 3, row 1 the same with minus signs: row 0 - row 1 of output is 0.5 1 2 3, and
    // row 0 + row 1 of input_grad (output_grad 1) is 1 1 1 1, within 1e-6.
    // 1. the forward and the backward on x and -x
    Tensor input_tensor = tensor_of({2, 4}, {0.5f, 1, 2, 3, -0.5f, -1, -2, -3});
    Tensor output_tensor = Tensor::zeros({2, 4});
    Tensor input_grad_tensor = Tensor::zeros({2, 4});
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, Tensor::ones({2, 4}), input_tensor);
    // 2. the differences of the outputs and the sums of the slopes
    std::vector<float> output_of_x_list = output_tensor[0].tolist();
    std::vector<float> output_of_minus_x_list = output_tensor[1].tolist();
    std::vector<float> slope_at_x_list = input_grad_tensor[0].tolist();
    std::vector<float> slope_at_minus_x_list = input_grad_tensor[1].tolist();
    std::vector<float> output_differences_list(4), slope_sums_list(4);
    for (int i_val = 0; i_val < 4; i_val++)
    {
        output_differences_list[i_val] = output_of_x_list[i_val] - output_of_minus_x_list[i_val];
        slope_sums_list[i_val] = slope_at_x_list[i_val] + slope_at_minus_x_list[i_val];
    }
    // 3. check
    ASSERT_NEAR(output_differences_list, std::vector<float>({0.5f, 1, 2, 3}), 1e-6);
    ASSERT_NEAR(slope_sums_list, std::vector<float>(4, 1.0f), 1e-6);
}

static void backward_writes_over_input_grad(void)
{
    // Expect: input_grad starts at 10 (left from somewhere else); the backward writes over it, so after
    // it input_grad is just the slopes at 1 -1 2 -2, with no 10 added. A second backward gives the same.
    // 1. input_grad that starts at 10
    Tensor input_tensor = tensor_of({4}, {1, -1, 2, -2});
    Tensor input_grad_tensor = Tensor::zeros({4}).fill_(10);
    // 2. one backward: written over
    gelu_backward(input_grad_tensor, Tensor::ones({4}), input_tensor);
    ASSERT_NEAR(input_grad_tensor.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
    // 3. a second backward: the same
    gelu_backward(input_grad_tensor, Tensor::ones({4}), input_tensor);
    ASSERT_NEAR(input_grad_tensor.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
}

static void backward_can_write_into_output_grad_itself(void)
{
    // Expect: output_grad 2 2 2 2, and the same tensor passed as input_grad: each thread reads its
    // output_grad before writing its input_grad, so the result is 2 x the slopes at 1 -1 2 -2:
    // 2.1659282 -0.1659282 2.1721986 -0.1721986.
    // 1. the backward, in place
    Tensor input_tensor = tensor_of({4}, {1, -1, 2, -2});
    Tensor grad_tensor = Tensor::zeros({4}).fill_(2);
    gelu_backward(grad_tensor, grad_tensor, input_tensor);
    // 2. check
    ASSERT_NEAR(grad_tensor.tolist(), std::vector<float>({2.1659282f, -0.1659282f, 2.1721986f, -0.1721986f}), 1e-6);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with GELU_THREADS_PER_BLOCK (256), and nothing is spilled.
    // The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote
    // down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)gelu_forward_kernel<float>, (const void *)gelu_backward_kernel<float>};
    const char *names_list[] = {"gelu_forward_kernel", "gelu_backward_kernel"};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= GELU_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: gelu_forward and gelu_backward called with threads_per_block 32, 128, 512 and 1024 (one warp up
    // to MAX_THREADS_PER_BLOCK) give the same output and input_grad as with the default,
    // GELU_THREADS_PER_BLOCK, to the last bit, in float and in bfloat16: each value is worked out alone, the
    // same whichever thread does it. {3, 7, 100}: 525 float4s, so every size has a part-full last block.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random inputs; the default launch's results
        torch::manual_seed(0);
        Shape shape_list = {3, 7, 100};
        Tensor input_tensor = random_tensor_on_gpu(shape_list).to(dtype_val);
        Tensor output_grad_tensor = random_tensor_on_gpu(shape_list).to(dtype_val);
        Tensor usual_output_tensor = Tensor::zeros(shape_list, dtype_val);
        Tensor usual_input_grad_tensor = Tensor::zeros(shape_list, dtype_val);
        gelu_forward(usual_output_tensor, input_tensor);
        gelu_backward(usual_input_grad_tensor, output_grad_tensor, input_tensor);

        // 2. each block size, against the default's
        for (int threads_per_block_val : block_sizes_list)
        {
            Tensor output_tensor = Tensor::zeros(shape_list, dtype_val);
            Tensor input_grad_tensor = Tensor::zeros(shape_list, dtype_val);
            gelu_forward(output_tensor, input_tensor, threads_per_block_val);
            gelu_backward(input_grad_tensor, output_grad_tensor, input_tensor, threads_per_block_val);
            ASSERT_EQ(output_tensor.to(DType::Float32).tolist(), usual_output_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(input_grad_tensor.to(DType::Float32).tolist(), usual_input_grad_tensor.to(DType::Float32).tolist());
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_mlp_rows();
    matches_pytorch_with_a_part_full_last_block();
    forward_and_backward_in_bfloat16();

    gelu_of_0_is_0_and_its_slope_is_one_half();
    gelu_and_its_slope_at_1_minus_1_2_minus_2();
    large_values_pass_through_and_large_negatives_give_0();
    gelu_of_x_minus_gelu_of_minus_x_is_x();
    backward_writes_over_input_grad();
    backward_can_write_into_output_grad_itself();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
