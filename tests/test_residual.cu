// test_residual.cu — tests for llmc/kernels/residual.cuh (residual_forward, residual_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's + with autograd for the backward, and times them on GPT-2
// small's size; part 2 checks numbers worked out by hand; part 3 checks that each kernel's registers
// fit its block; part 4 times each kernel with other block and thread counts.
// (The C# trainer had no residual test: TorchSharp's + was trusted.)
#include <torch/types.h>              // torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ: every check here is exact
#include "../llmc/kernels/residual.cuh"

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape` on the GPU holding `values`, given in row order.
static Tensor tensor_of(const Shape &shape, const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty(shape);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes a tensor of `shape` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 3.1M values.
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

/// Runs one forward and one backward on random values, mine and PyTorch's residual + branch_output with
/// autograd, and checks that output, residual_grad and branch_output_grad are the same to the last bit:
/// both sides do the same one float add per value, and the backward only copies.
static void forward_and_backward_match_pytorch(const Shape &shape)
{
    torch::manual_seed(0);
    Tensor residual = random_tensor_on_gpu(shape), branch_output = random_tensor_on_gpu(shape);
    Tensor output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape);
    Tensor residual_grad = Tensor::zeros(shape), branch_output_grad = Tensor::zeros(shape);
    residual_forward(output, residual, branch_output);
    residual_backward(residual_grad, branch_output_grad, output_grad);

    torch::Tensor pytorch_residual = pytorch_copy_of(residual).requires_grad_(true);
    torch::Tensor pytorch_branch_output = pytorch_copy_of(branch_output).requires_grad_(true);
    torch::Tensor pytorch_output = pytorch_residual + pytorch_branch_output;
    pytorch_output.backward(pytorch_copy_of(output_grad));
    ASSERT_EQ(output.tolist(), list_of(pytorch_output));
    ASSERT_EQ(residual_grad.tolist(), list_of(pytorch_residual.grad()));
    ASSERT_EQ(branch_output_grad.tolist(), list_of(pytorch_branch_output.grad()));
}

static void matches_pytorch_on_gpt2_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens of 768, the residual stream of GPT-2 small:
    // 12,288 values, 3072 float4s, 12 full blocks. output equals PyTorch's residual + branch_output, and
    // both gradients its autograd's, exactly.
    forward_and_backward_match_pytorch({2, 8, 768});
}

static void matches_pytorch_with_a_part_full_last_block(void)
{
    // Expect: {3, 7, 100}: 2100 values, 525 float4s, 3 blocks of 256 threads; the last has 13 float4s
    // for 256 threads, and its other 243 threads must write nothing. The same as PyTorch, exactly.
    forward_and_backward_match_pytorch({3, 7, 100});
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's residual stream, {4, 1024, 768}:
    // 3,145,728 values, 12.6 MB a tensor. Mine against PyTorch's + with autograd. A forward reads residual
    // and branch_output and writes output, 3 x 12.6 MB: at a T4's 320 GB/s at least 0.118 ms. A backward
    // reads output_grad and writes the two gradients, 3 x 12.6 MB: also at least 0.118 ms. Both sides
    // write the gradients over each step; at the end they must agree.
    torch::manual_seed(0);
    Shape shape = {4, 1024, 768};
    Tensor residual = random_tensor_on_gpu(shape), branch_output = random_tensor_on_gpu(shape);
    Tensor output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape);
    Tensor residual_grad = Tensor::zeros(shape), branch_output_grad = Tensor::zeros(shape);

    torch::Tensor pytorch_residual = pytorch_copy_of(residual).requires_grad_(true);
    torch::Tensor pytorch_branch_output = pytorch_copy_of(branch_output).requires_grad_(true);
    torch::Tensor pytorch_output_grad = pytorch_copy_of(output_grad);
    torch::Tensor pytorch_output;

    // Steps 1 and 2 do first-time work, so the averages are of 3..20: in step 1 CUDA loads each side's
    // kernels.
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { residual_forward(output, residual, branch_output); });
        times[1] = gpu_milliseconds([&] { residual_backward(residual_grad, branch_output_grad, output_grad); });
        times[2] = gpu_milliseconds([&] { pytorch_output = pytorch_residual + pytorch_branch_output; });
        pytorch_residual.mutable_grad() = torch::Tensor();      // gradients written, as mine, not added onto
        pytorch_branch_output.mutable_grad() = torch::Tensor();
        times[3] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    double moved_gb = 3 * (double)residual.numel() * sizeof(float) / 1e9;     // 0.0377 GB, for all four
    print_test_name(__func__);
    printf("    average of steps 3..20:   my forward   my backward   PyTorch forward   PyTorch backward\n");
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("        GB/s                  %10.1f   %11.1f   %15.1f   %16.1f\n", moved_gb / (totals[0] / 18 / 1e3),
           moved_gb / (totals[1] / 18 / 1e3), moved_gb / (totals[2] / 18 / 1e3), moved_gb / (totals[3] / 18 / 1e3));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_EQ(output.tolist(), list_of(pytorch_output));
    ASSERT_EQ(residual_grad.tolist(), list_of(pytorch_residual.grad()));
    ASSERT_EQ(branch_output_grad.tolist(), list_of(pytorch_branch_output.grad()));
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example, 4 values (one float4: thread 0 has it all):
//
//     residual             1      2      3      4
//     branch_output        0.5   -2      0      2
//     output               1.5    0      3      6        F1: y = r + b
//     output_grad          1     -2      0.5    3
//     residual_grad        1     -2      0.5    3        B2: dr(1) = dy
//     branch_output_grad   1     -2      0.5    3        B1: db = dy

static const std::vector<float> NOTE_RESIDUAL = {1, 2, 3, 4};
static const std::vector<float> NOTE_BRANCH_OUTPUT = {0.5f, -2, 0, 2};
static const std::vector<float> NOTE_OUTPUT = {1.5f, 0, 3, 6};
static const std::vector<float> NOTE_OUTPUT_GRAD = {1, -2, 0.5f, 3};

static void forward_of_the_note_s_example(void)
{
    // Expect: 1 + 0.5, 2 - 2, 3 + 0, 4 + 2 = 1.5 0 3 6, exactly: every value is a float.
    Tensor output = Tensor::zeros({4});
    residual_forward(output, tensor_of({4}, NOTE_RESIDUAL), tensor_of({4}, NOTE_BRANCH_OUTPUT));
    ASSERT_EQ(output.tolist(), NOTE_OUTPUT);
}

static void backward_gives_output_grad_to_both(void)
{
    // Expect: the slope of r + b is 1 for r and 1 for b, so residual_grad and branch_output_grad are both
    // output_grad, 1 -2 0.5 3, exactly. Not halved: the add copies the gradient, it does not split it.
    Tensor residual_grad = Tensor::zeros({4}), branch_output_grad = Tensor::zeros({4});
    residual_backward(residual_grad, branch_output_grad, tensor_of({4}, NOTE_OUTPUT_GRAD));
    ASSERT_EQ(residual_grad.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad.tolist(), NOTE_OUTPUT_GRAD);
}

static void a_branch_of_0_leaves_the_row_as_it_is(void)
{
    // Expect: branch_output all 0, as a block that has learned nothing gives: output is residual,
    // 1 2 3 4, exactly. The block is then the identity, and the row passes through it unchanged.
    Tensor output = Tensor::zeros({4});
    residual_forward(output, tensor_of({4}, NOTE_RESIDUAL), Tensor::zeros({4}));
    ASSERT_EQ(output.tolist(), NOTE_RESIDUAL);
}

static void adding_minus_itself_gives_0(void)
{
    // Expect: x + (-x) = 0 exactly for any float, also 1000000 and -3.25: 0 0 0 0.
    Tensor output = Tensor::ones({4});
    residual_forward(output, tensor_of({4}, {1.5f, -2, 1000000, -3.25f}), tensor_of({4}, {-1.5f, 2, -1000000, 3.25f}));
    ASSERT_EQ(output.tolist(), std::vector<float>(4, 0.0f));
}

static void adds_in_float_as_pytorch_does(void)
{
    // Expect: a float has 24 bits of digits, so above 2^24 = 16,777,216 the floats are 2 apart, and
    // above 10^8 they are 8 apart. A value between two floats is rounded to the nearer one, to the even
    // one on a tie:
    //
    //     16777216 + 1      = 16777217, halfway between 16777216 and 16777218: 16777216
    //     1 + 0.00000001    = 1.00000001, much nearer 1 than the next float, 1.0000001: 1
    //     16777216 + 2      = 16777218, a float: 16777218
    //     100000000 + 1     = 100000001, nearest float 100000000
    //
    // PyTorch's + rounds the same way, so part 1 can ask for the same bits.
    Tensor output = Tensor::zeros({4});
    residual_forward(output, tensor_of({4}, {16777216, 1, 16777216, 100000000}), tensor_of({4}, {1, 1e-8f, 2, 1}));
    ASSERT_EQ(output.tolist(), std::vector<float>({16777216, 1, 16777218, 100000000}));
}

static void forward_can_write_into_residual_itself(void)
{
    // Expect: the residual tensor passed as output too, as a trainer that keeps one residual stream
    // would: each thread reads its residual before writing its output, so the result is 1.5 0 3 6.
    Tensor residual = tensor_of({4}, NOTE_RESIDUAL);
    residual_forward(residual, residual, tensor_of({4}, NOTE_BRANCH_OUTPUT));
    ASSERT_EQ(residual.tolist(), NOTE_OUTPUT);
}

static void backward_writes_over_both_grads(void)
{
    // Expect: residual_grad and branch_output_grad start at 10 (left from somewhere else); the backward
    // writes over them, so after it both are output_grad, with no 10 added. A second backward gives the
    // same: the layer after (LayerNorm) adds its part onto residual_grad, not the residual add.
    Tensor output_grad = tensor_of({4}, NOTE_OUTPUT_GRAD);
    Tensor residual_grad = Tensor::zeros({4}).fill_(10), branch_output_grad = Tensor::zeros({4}).fill_(10);
    residual_backward(residual_grad, branch_output_grad, output_grad);
    ASSERT_EQ(residual_grad.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad.tolist(), NOTE_OUTPUT_GRAD);
    residual_backward(residual_grad, branch_output_grad, output_grad);
    ASSERT_EQ(residual_grad.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad.tolist(), NOTE_OUTPUT_GRAD);
}

static void backward_can_write_into_output_grad_itself(void)
{
    // Expect: the output_grad tensor passed as residual_grad too, as llm.c's trainer does: each thread
    // reads its output_grad before writing, so residual_grad stays 1 -2 0.5 3 and branch_output_grad
    // gets the same.
    Tensor grad = tensor_of({4}, NOTE_OUTPUT_GRAD), branch_output_grad = Tensor::zeros({4});
    residual_backward(grad, branch_output_grad, grad);
    ASSERT_EQ(grad.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad.tolist(), NOTE_OUTPUT_GRAD);
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with RESIDUAL_THREADS_PER_BLOCK (256), and nothing is spilled.
    // The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote
    // down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)residual_forward_kernel, (const void *)residual_backward_kernel};
    const char *names[] = {"residual_forward_kernel", "residual_backward_kernel"};
    for (int k = 0; k < 2; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= RESIDUAL_THREADS_PER_BLOCK, true);
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
    // Prints the time of the forward and the backward on GPT-2 small's residual stream, {4, 1024, 768}
    // (786,432 float4s), for 8 block counts x 4 thread counts, each the average of runs 2..20. The
    // kernels are launched here directly, <<<blocks, threads>>>, instead of through residual_forward and
    // residual_backward, which always launch 3072 x 256: one float4 per thread. With fewer, each thread
    // walks more float4s.
    //
    // Expect: every launch gives the same output and gradients as the usual ones, to the last bit: each
    // value is worked out alone, the same way whichever thread does it. Only the time changes. A T4 runs
    // at most 40 SMs x 1024 threads = 40,960 threads at once (160 blocks of 256): far fewer than that
    // likely cannot keep enough reads in flight to fill the memory bus.
    torch::manual_seed(0);
    Shape shape = {4, 1024, 768};
    size_t total_float4s = 4 * 1024 * 768 / 4;
    Tensor residual = random_tensor_on_gpu(shape), branch_output = random_tensor_on_gpu(shape);
    Tensor output_grad = random_tensor_on_gpu(shape);
    Tensor output = Tensor::zeros(shape);
    Tensor residual_grad = Tensor::zeros(shape), branch_output_grad = Tensor::zeros(shape);

    // The usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with.
    residual_forward(output, residual, branch_output);
    residual_backward(residual_grad, branch_output_grad, output_grad);
    torch::Tensor usual_output = pytorch_copy_of(output), usual_residual_grad = pytorch_copy_of(residual_grad);
    torch::Tensor usual_branch_output_grad = pytorch_copy_of(branch_output_grad);
    int usual_blocks = residual_blocks(total_float4s);       // 3072

    int blocks = 0, threads = 0;                             // the launch the 2 below use, set by the loop
    auto forward = [&] {
        residual_forward_kernel<<<blocks, threads>>>((float4 *)output.data(), (const float4 *)residual.data(),
                                                     (const float4 *)branch_output.data(), total_float4s);
        cudaCheck(cudaGetLastError());
    };
    auto backward = [&] {
        residual_backward_kernel<<<blocks, threads>>>((float4 *)residual_grad.data(),
                                                      (float4 *)branch_output_grad.data(),
                                                      (const float4 *)output_grad.data(), total_float4s);
        cudaCheck(cudaGetLastError());
    };
    blocks = usual_blocks;
    threads = RESIDUAL_THREADS_PER_BLOCK;
    warm_up_gpu(forward);

    int block_counts[] = {40, 80, 160, 320, 640, 1280, 1536, 3072};
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
            residual_grad.zero_();
            branch_output_grad.zero_();
            forward();
            backward();
            bool same = torch::equal(pytorch_copy_of(output), usual_output) &&
                        torch::equal(pytorch_copy_of(residual_grad), usual_residual_grad) &&
                        torch::equal(pytorch_copy_of(branch_output_grad), usual_branch_output_grad);
            if (same)
                launches_with_the_same_result++;

            size_t threads_launched = (size_t)blocks * threads;
            int float4s_per_thread = (int)((total_float4s + threads_launched - 1) / threads_launched); // the most
            const char *usual = blocks == usual_blocks && threads == RESIDUAL_THREADS_PER_BLOCK
                                    ? "   <- what residual_forward and residual_backward launch"
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
    matches_pytorch_on_gpt2_rows();
    matches_pytorch_with_a_part_full_last_block();
    time_20_steps_on_gpt2_small_size();

    forward_of_the_note_s_example();
    backward_gives_output_grad_to_both();
    a_branch_of_0_leaves_the_row_as_it_is();
    adding_minus_itself_gives_0();
    adds_in_float_as_pytorch_does();
    forward_can_write_into_residual_itself();
    backward_writes_over_both_grads();
    backward_can_write_into_output_grad_itself();

    every_kernel_fits_its_block();

    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tests passed\n");
    return 0;
}
