// test_gelu.cu — tests for llmc/kernels/gelu.cuh (gelu_forward, gelu_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's gelu (tanh), with autograd for the backward, and times them
// on GPT-2 small's MLP size; part 2 checks numbers worked out by hand; part 3 checks that each
// kernel's registers fit its block; part 4 times each kernel with other block and thread counts.
// (The C# trainer had no GELU test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::gelu, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/gelu.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape` on the GPU holding `values`, given in row order.
static Tensor tensor_of(const Shape &shape, const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty(shape);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes a tensor of `shape` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for the MLP's 12.6M values.
static Tensor random_tensor_on_gpu(const Shape &shape)
{
    Tensor tensor = Tensor::empty(shape);
    torch::Tensor values = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes, on_gpu).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu.data_ptr<float>(), on_cpu.data_ptr<float>() + on_cpu.numel());
}

/// Measures how long `run` takes on the GPU, in milliseconds: a mark on the GPU's queue
/// before it and one after; the GPU writes down the time as it reaches each.
template <typename Run>
static float gpu_milliseconds(Run run)
{
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    run();
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end)); // wait until the GPU has passed the second mark
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    return milliseconds;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values, mine and PyTorch's gelu (tanh) with autograd,
/// and checks that output and input_grad agree within TOLERANCE.
static void forward_and_backward_match_pytorch(const Shape &shape)
{
    torch::manual_seed(0);
    Tensor input = random_tensor_on_gpu(shape);
    Tensor output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape), input_grad = Tensor::zeros(shape);
    gelu_forward(output, input);
    gelu_backward(input_grad, output_grad, input);

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_output = torch::gelu(pytorch_input, "tanh");
    pytorch_output.backward(pytorch_copy_of(output_grad));
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), TOLERANCE);
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), TOLERANCE);
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

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's MLP middle, {4, 1024, 3072}:
    // 12,582,912 values, 50.3 MB a tensor. Mine against PyTorch's gelu (tanh) with autograd. A forward
    // reads input and writes output, 2 x 50.3 MB: at a T4's 320 GB/s at least 0.315 ms. A backward reads
    // input and output_grad and writes input_grad, 3 x 50.3 MB: at least 0.472 ms. Both sides write
    // input_grad over each step; at the end they must agree.
    torch::manual_seed(0);
    Shape shape = {4, 1024, 3072};
    Tensor input = random_tensor_on_gpu(shape), output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape), input_grad = Tensor::zeros(shape);

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_output_grad = pytorch_copy_of(output_grad);
    torch::Tensor pytorch_output;

    // Steps 1 and 2 do first-time work, so the averages are of 3..20: in step 1 CUDA loads each side's
    // kernels.
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { gelu_forward(output, input); });
        times[1] = gpu_milliseconds([&] { gelu_backward(input_grad, output_grad, input); });
        times[2] = gpu_milliseconds([&] { pytorch_output = torch::gelu(pytorch_input, "tanh"); });
        pytorch_input.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times[3] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    double tensor_gb = (double)input.numel() * sizeof(float) / 1e9;          // 0.0503 GB
    double moved_gb[4] = {2 * tensor_gb, 3 * tensor_gb, 2 * tensor_gb, 3 * tensor_gb};
    print_test_name(__func__);
    printf("    average of steps 3..20:   my forward   my backward   PyTorch forward   PyTorch backward\n");
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("        GB/s                  %10.1f   %11.1f   %15.1f   %16.1f\n", moved_gb[0] / (totals[0] / 18 / 1e3),
           moved_gb[1] / (totals[1] / 18 / 1e3), moved_gb[2] / (totals[2] / 18 / 1e3),
           moved_gb[3] / (totals[3] / 18 / 1e3));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), TOLERANCE);
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), TOLERANCE);
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
    Tensor input = Tensor::zeros({4});
    Tensor output = Tensor::ones({4}), input_grad = Tensor::zeros({4});
    gelu_forward(output, input);
    gelu_backward(input_grad, Tensor::ones({4}), input);
    ASSERT_EQ(output.tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(input_grad.tolist(), std::vector<float>(4, 0.5f));
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
    Tensor input = tensor_of({4}, {1, -1, 2, -2});
    Tensor output = Tensor::zeros({4}), input_grad = Tensor::zeros({4});
    gelu_forward(output, input);
    gelu_backward(input_grad, Tensor::ones({4}), input);
    ASSERT_NEAR(output.tolist(), GELU_OF_1_MINUS_1_2_MINUS_2, 1e-6);
    ASSERT_NEAR(input_grad.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
}

static void large_values_pass_through_and_large_negatives_give_0(void)
{
    // Expect: x = 10, -10, 100, -100. inner is about +-43.7 at 10, and tanh of it is 1 or -1 in a
    // float, so gelu = 0.5 x (1 + 1) = x, or 0.5 x (1 - 1) = 0, exactly: 10 0 100 0 (the 0s print as
    // -0: 0.5 x -10 x 0, which equals 0). The slopes are then 0.5 x (1 +- 1) + 0 = 1 0 1 0, exactly.
    Tensor input = tensor_of({4}, {10, -10, 100, -100});
    Tensor output = Tensor::zeros({4}), input_grad = Tensor::zeros({4});
    gelu_forward(output, input);
    gelu_backward(input_grad, Tensor::ones({4}), input);
    ASSERT_EQ(output.tolist(), std::vector<float>({10, 0, 100, 0}));
    ASSERT_EQ(input_grad.tolist(), std::vector<float>({1, 0, 1, 0}));
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
    Tensor input = tensor_of({2, 4}, {0.5f, 1, 2, 3, -0.5f, -1, -2, -3});
    Tensor output = Tensor::zeros({2, 4}), input_grad = Tensor::zeros({2, 4});
    gelu_forward(output, input);
    gelu_backward(input_grad, Tensor::ones({2, 4}), input);
    std::vector<float> output_of_x = output[0].tolist(), output_of_minus_x = output[1].tolist();
    std::vector<float> slope_at_x = input_grad[0].tolist(), slope_at_minus_x = input_grad[1].tolist();
    std::vector<float> output_differences(4), slope_sums(4);
    for (int i = 0; i < 4; i++)
    {
        output_differences[i] = output_of_x[i] - output_of_minus_x[i];
        slope_sums[i] = slope_at_x[i] + slope_at_minus_x[i];
    }
    ASSERT_NEAR(output_differences, std::vector<float>({0.5f, 1, 2, 3}), 1e-6);
    ASSERT_NEAR(slope_sums, std::vector<float>(4, 1.0f), 1e-6);
}

static void backward_writes_over_input_grad(void)
{
    // Expect: input_grad starts at 10 (left from somewhere else); the backward writes over it, so after
    // it input_grad is just the slopes at 1 -1 2 -2, with no 10 added. A second backward gives the same.
    Tensor input = tensor_of({4}, {1, -1, 2, -2});
    Tensor input_grad = Tensor::zeros({4}).fill_(10);
    gelu_backward(input_grad, Tensor::ones({4}), input);
    ASSERT_NEAR(input_grad.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
    gelu_backward(input_grad, Tensor::ones({4}), input);
    ASSERT_NEAR(input_grad.tolist(), SLOPE_AT_1_MINUS_1_2_MINUS_2, 1e-6);
}

static void backward_can_write_into_output_grad_itself(void)
{
    // Expect: output_grad 2 2 2 2, and the same tensor passed as input_grad: each thread reads its
    // output_grad before writing its input_grad, so the result is 2 x the slopes at 1 -1 2 -2:
    // 2.1659282 -0.1659282 2.1721986 -0.1721986.
    Tensor input = tensor_of({4}, {1, -1, 2, -2});
    Tensor grad = Tensor::zeros({4}).fill_(2);
    gelu_backward(grad, grad, input);
    ASSERT_NEAR(grad.tolist(), std::vector<float>({2.1659282f, -0.1659282f, 2.1721986f, -0.1721986f}), 1e-6);
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with GELU_THREADS_PER_BLOCK (256), and nothing is spilled.
    // The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote
    // down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)gelu_forward_kernel, (const void *)gelu_backward_kernel};
    const char *names[] = {"gelu_forward_kernel", "gelu_backward_kernel"};
    for (int k = 0; k < 2; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= GELU_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes.localSizeBytes, (size_t)0);
    }
}

// ══ part 4: other block and thread counts ═════════════════════════════════════

/// Runs `run` 20 times on the GPU and gives the average time of runs 2..20, in milliseconds;
/// run 1 does first-time work, so it is left out.
template <typename Run>
static float average_milliseconds(Run run)
{
    float total = 0.0f;
    for (int t = 1; t <= 20; t++)
    {
        float milliseconds = gpu_milliseconds(run);
        if (t > 1)
            total += milliseconds;
    }
    return total / 19;
}

/// Keeps the GPU busy with `run` for about a second before timing, so the first launches timed do not
/// run while its clock is still rising from idle.
template <typename Run>
static void warm_up_gpu(Run run)
{
    float total = 0.0f;
    while (total < 1000.0f)
        total += gpu_milliseconds(run);
}

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's MLP middle, {4, 1024, 3072}
    // (3,145,728 float4s), for 8 block counts x 4 thread counts, each the average of runs 2..20. The
    // kernels are launched here directly, <<<blocks, threads>>>, instead of through gelu_forward and
    // gelu_backward, which always launch 12,288 x 256: one float4 per thread. With fewer, each thread
    // walks more float4s.
    //
    // Expect: every launch gives the same output and input_grad as the usual ones, to the last bit:
    // each value is worked out alone, the same way whichever thread does it. Only the time changes.
    // A T4 runs at most 40 SMs x 1024 threads = 40,960 threads at once (160 blocks of 256): far fewer
    // than that likely cannot keep enough reads in flight to fill the memory bus.
    torch::manual_seed(0);
    Shape shape = {4, 1024, 3072};
    size_t total_float4s = 4 * 1024 * 3072 / 4;
    Tensor input = random_tensor_on_gpu(shape), output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape), input_grad = Tensor::zeros(shape);

    // The usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with.
    gelu_forward(output, input);
    gelu_backward(input_grad, output_grad, input);
    torch::Tensor usual_output = pytorch_copy_of(output), usual_input_grad = pytorch_copy_of(input_grad);
    int usual_blocks = gelu_blocks(total_float4s);           // 12,288

    int blocks = 0, threads = 0;                             // the launch the 2 below use, set by the loop
    auto forward = [&] {
        gelu_forward_kernel<<<blocks, threads>>>((float4 *)output.data(), (const float4 *)input.data(),
                                                 total_float4s);
        cudaCheck(cudaGetLastError());
    };
    auto backward = [&] {
        gelu_backward_kernel<<<blocks, threads>>>((float4 *)input_grad.data(), (const float4 *)output_grad.data(),
                                                  (const float4 *)input.data(), total_float4s);
        cudaCheck(cudaGetLastError());
    };
    blocks = usual_blocks;
    threads = GELU_THREADS_PER_BLOCK;
    warm_up_gpu(forward);

    int block_counts[] = {40, 80, 160, 320, 640, 1280, 3072, 12288};
    int thread_counts[] = {128, 256, 512, 1024};
    const char *kernel_names[] = {"forward", "backward"};
    float fastest_ms[2] = {1e9f, 1e9f};
    int fastest_blocks[2] = {0, 0}, fastest_threads[2] = {0, 0};
    int launches_with_the_same_result = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   float4s per thread   forward   backward   same result\n");
    for (int block_count : block_counts)
    {
        for (int thread_count : thread_counts)
        {
            blocks = block_count;
            threads = thread_count;
            float times[2];
            times[0] = average_milliseconds(forward);
            times[1] = average_milliseconds(backward);
            for (int k = 0; k < 2; k++)
            {
                if (times[k] < fastest_ms[k])
                {
                    fastest_ms[k] = times[k];
                    fastest_blocks[k] = blocks;
                    fastest_threads[k] = threads;
                }
            }

            // One forward and one backward from 0 with this launch, against the usual launches'.
            output.zero_();
            input_grad.zero_();
            forward();
            backward();
            bool same = torch::equal(pytorch_copy_of(output), usual_output) &&
                        torch::equal(pytorch_copy_of(input_grad), usual_input_grad);
            if (same)
                launches_with_the_same_result++;

            size_t threads_launched = (size_t)blocks * threads;
            int float4s_per_thread = (int)((total_float4s + threads_launched - 1) / threads_launched); // the most
            const char *usual = blocks == usual_blocks && threads == GELU_THREADS_PER_BLOCK
                                    ? "   <- what gelu_forward and gelu_backward launch"
                                    : "";
            printf("                  %6d   %7d   %18d   %7.3f   %8.3f   %-11s%s\n", blocks, threads,
                   float4s_per_thread, times[0], times[1], same ? "yes" : "NO", usual);
        }
    }
    for (int k = 0; k < 2; k++)
        printf("    fastest %-8s  %5d blocks x %4d threads: %.3f ms\n", kernel_names[k], fastest_blocks[k],
               fastest_threads[k], fastest_ms[k]);
    ASSERT_EQ(launches_with_the_same_result, 8 * 4);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_on_mlp_rows();
    matches_pytorch_with_a_part_full_last_block();
    time_20_steps_on_gpt2_small_size();

    gelu_of_0_is_0_and_its_slope_is_one_half();
    gelu_and_its_slope_at_1_minus_1_2_minus_2();
    large_values_pass_through_and_large_negatives_give_0();
    gelu_of_x_minus_gelu_of_minus_x_is_x();
    backward_writes_over_input_grad();
    backward_can_write_into_output_grad_itself();

    every_kernel_fits_its_block();

    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tests passed\n");
    return 0;
}
