// tune_gelu.cu — the timings of tests/test_gelu.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size, time_each_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_gelu.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/gelu_<date>_<time>.txt.
#include <torch/types.h>              // torch::gelu, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/kernels/gelu.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers ───────────────────────────────────────────────────────────────────

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

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's MLP middle, {4, 1024, 3072}:
    // 12,582,912 values, 50.3 MB a tensor. Mine against PyTorch's gelu (tanh) with autograd. A forward
    // reads input and writes output, 2 x 50.3 MB: at a T4's 320 GB/s at least 0.315 ms. A backward reads
    // input and output_grad and writes input_grad, 3 x 50.3 MB: at least 0.472 ms. Both sides write
    // input_grad over each step; at the end they must agree.

    // 1. random inputs, mine and PyTorch's
    torch::manual_seed(0);
    Shape shape_list = {4, 1024, 3072};
    Tensor input_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_tensor = Tensor::zeros(shape_list);
    Tensor input_grad_tensor = Tensor::zeros(shape_list);
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(output_grad_tensor);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work (in step 1 CUDA loads each side's
    //    kernels), so the averages are of the rest
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { gelu_forward(output_tensor, input_tensor); });
        times_list[1] = gpu_milliseconds([&] { gelu_backward(input_grad_tensor, output_grad_tensor, input_tensor); });
        times_list[2] = gpu_milliseconds([&] { pytorch_output_tensor = torch::gelu(pytorch_input_tensor, "tanh"); });
        pytorch_input_tensor.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double tensor_gb_val = (double)input_tensor.numel() * sizeof(float) / 1e9;          // 0.0503 GB
    double moved_gb_list[4] = {2 * tensor_gb_val, 3 * tensor_gb_val, 2 * tensor_gb_val, 3 * tensor_gb_val};
    print_test_name(__func__);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("        GB/s                  %10.1f   %11.1f   %15.1f   %16.1f\n",
           moved_gb_list[0] / (totals_list[0] / averaged_val / 1e3), moved_gb_list[1] / (totals_list[1] / averaged_val / 1e3),
           moved_gb_list[2] / (totals_list[2] / averaged_val / 1e3), moved_gb_list[3] / (totals_list[3] / averaged_val / 1e3));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_NEAR(output_tensor.tolist(), list_of(pytorch_output_tensor), TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's MLP middle, {4, 1024, 3072}
    // (3,145,728 float4s), in float and in bfloat16, for 10 block counts x 4 thread counts, each the average
    // of 20 runs: the 8 picked on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe,
    // 132 and 264 on an SXM; on a T4 they repeat 40 and 80). The 40 launches take turns: 20 rounds, each timing every launch once (the forward and the
    // backward in float, then both in bfloat16). One launch's 20 runs in a row would all land in the same
    // stretch of the test, and a T4's clock can drift as it heats up (the first launch timed would get the
    // fastest clock); taking turns gives every launch runs from early, middle and late alike. Run 1 is
    // counted too: warm_up_gpu has already done the first-time work. The kernels are launched here directly,
    // <<<blocks, threads>>>, instead of through gelu_forward and gelu_backward, which always launch 12,288 x
    // 256: one float4 per thread. With fewer, each thread walks more float4s. A bfloat16 float4 is one 8-byte
    // read instead of 16, so each thread keeps half the bytes in flight: its best launch may differ.
    //
    // Expect: every launch gives the same output and input_grad as the usual ones of its dtype, to the last
    // bit: each value is worked out alone, the same way whichever thread does it. Only the time changes.
    // A T4 runs at most 40 SMs x 1024 threads = 40,960 threads at once (160 blocks of 256): far fewer
    // than that likely cannot keep enough reads in flight to fill the memory bus.

    // 1. random inputs in float, and the same values in bfloat16, with outputs of their own
    torch::manual_seed(0);
    Shape shape_list = {4, 1024, 3072};
    size_t total_float4s_val = 4 * 1024 * 3072 / VALUES_PER_FLOAT4;
    Tensor input_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_tensor = Tensor::zeros(shape_list);
    Tensor input_grad_tensor = Tensor::zeros(shape_list);
    Tensor input_bf16_tensor = input_tensor.to(DType::BFloat16);
    Tensor output_grad_bf16_tensor = output_grad_tensor.to(DType::BFloat16);
    Tensor output_bf16_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor input_grad_bf16_tensor = Tensor::zeros(shape_list, DType::BFloat16);

    // 2. the usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with
    gelu_forward(output_tensor, input_tensor);
    gelu_backward(input_grad_tensor, output_grad_tensor, input_tensor);
    gelu_forward(output_bf16_tensor, input_bf16_tensor);
    gelu_backward(input_grad_bf16_tensor, output_grad_bf16_tensor, input_bf16_tensor);
    torch::Tensor usual_output_tensor = pytorch_copy_of(output_tensor);
    torch::Tensor usual_input_grad_tensor = pytorch_copy_of(input_grad_tensor);
    torch::Tensor usual_output_bf16_tensor = pytorch_copy_of(output_bf16_tensor.to(DType::Float32));
    torch::Tensor usual_input_grad_bf16_tensor = pytorch_copy_of(input_grad_bf16_tensor.to(DType::Float32));
    int usual_blocks_val = gelu_blocks(total_float4s_val);   // 12,288

    // 3. the four launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        gelu_forward_kernel<float><<<blocks_val, threads_val>>>(output_tensor.data(), input_tensor.data(), total_float4s_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_val = [&] {
        gelu_backward_kernel<float><<<blocks_val, threads_val>>>(input_grad_tensor.data(), output_grad_tensor.data(),
                                                                 input_tensor.data(), total_float4s_val);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16_val = [&] {
        gelu_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(output_bf16_tensor.data_ptr<nv_bfloat16>(),
                                                                      input_bf16_tensor.data_ptr<nv_bfloat16>(),
                                                                      total_float4s_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_bf16_val = [&] {
        gelu_backward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(input_grad_bf16_tensor.data_ptr<nv_bfloat16>(),
                                                                       output_grad_bf16_tensor.data_ptr<nv_bfloat16>(),
                                                                       input_bf16_tensor.data_ptr<nv_bfloat16>(),
                                                                       total_float4s_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = usual_blocks_val;
    threads_val = GELU_THREADS_PER_BLOCK;
    warm_up_gpu(forward_val);

    // 4. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 80, 160, 320, 640, 1280, 3072, 12288, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 10 * thread_count_choices_val;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]; one time per kernel, in kernel_names' order
    float total_ms_list[launches_val][4] = {};
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(forward_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(backward_val);
            total_ms_list[launch_val][2] += gpu_milliseconds(forward_bf16_val);
            total_ms_list[launch_val][3] += gpu_milliseconds(backward_bf16_val);
        }
    }

    // 5. print each launch's averages, and check its results against the usual ones
    const char *kernel_names_list[] = {"forward", "backward", "bf16 forward", "bf16 backward"};
    float fastest_ms_list[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    int fastest_blocks_list[4] = {};
    int fastest_threads_list[4] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   float4s per thread   forward   backward   bf16 forward   "
           "bf16 backward   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[4];
        for (int kernel_index_val = 0; kernel_index_val < 4; kernel_index_val++)
        {
            times_list[kernel_index_val] = total_ms_list[launch_val][kernel_index_val] / SWEEP_ROUNDS;
            if (times_list[kernel_index_val] < fastest_ms_list[kernel_index_val])
            {
                fastest_ms_list[kernel_index_val] = times_list[kernel_index_val];
                fastest_blocks_list[kernel_index_val] = blocks_val;
                fastest_threads_list[kernel_index_val] = threads_val;
            }
        }

        // one forward and one backward from 0 with this launch in each dtype, against the usual launches'
        output_tensor.zero_();
        input_grad_tensor.zero_();
        output_bf16_tensor.zero_();
        input_grad_bf16_tensor.zero_();
        forward_val();
        backward_val();
        forward_bf16_val();
        backward_bf16_val();
        bool same_val = torch::equal(pytorch_copy_of(output_tensor), usual_output_tensor) &&
                        torch::equal(pytorch_copy_of(input_grad_tensor), usual_input_grad_tensor) &&
                        torch::equal(pytorch_copy_of(output_bf16_tensor.to(DType::Float32)), usual_output_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(input_grad_bf16_tensor.to(DType::Float32)), usual_input_grad_bf16_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(total_float4s_val, threads_launched_val); // the most
        const char *usual_ptr = blocks_val == usual_blocks_val && threads_val == GELU_THREADS_PER_BLOCK
                                    ? "   <- what gelu_forward and gelu_backward launch"
                                    : "";
        printf("                  %6d   %7d   %18d   %7.3f   %8.3f   %12.3f   %13.3f   %-11s%s\n", blocks_val,
               threads_val, float4s_per_thread_val, times_list[0], times_list[1], times_list[2], times_list[3],
               same_val ? "yes" : "NO", usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < 4; kernel_index_val++)
        printf("    fastest %-13s  %5d blocks x %4d threads: %.3f ms\n", kernel_names_list[kernel_index_val],
               fastest_blocks_list[kernel_index_val], fastest_threads_list[kernel_index_val],
               fastest_ms_list[kernel_index_val]);

    // 6. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();
    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
