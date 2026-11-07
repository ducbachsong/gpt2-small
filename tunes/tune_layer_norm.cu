// tune_layer_norm.cu — the timings of tests/test_layer_norm.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size, time_each_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_layer_norm.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/layer_norm_<date>_<time>.txt.
#include <torch/types.h>              // torch::layer_norm, backward(); libtorch before any CUDA header
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/kernels/layer_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers ───────────────────────────────────────────────────────────────────

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
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's sizes, batch_size = 4 rows
    // of seq_len = 1024 tokens, embed_dim = 768, mine against PyTorch's layer_norm with autograd. A
    // forward reads input and writes output, 2 x 12.6 MB: at a T4's 320 GB/s at least 0.079 ms. A
    // backward reads input, output_grad and input_grad and writes input_grad, 4 x 12.6 MB: at least
    // 0.157 ms. Both sides add 20 backwards onto their gradients; at the end they must still agree.

    // 1. GPT-2 small's sizes, mine and PyTorch's
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t embed_dim_val = 768;
    Tensor input_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});      // made on the CPU: a few seconds
    Tensor weight_tensor = Tensor::randn({embed_dim_val});
    Tensor bias_tensor = Tensor::randn({embed_dim_val});
    Tensor output_grad_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor weight_grad_tensor = Tensor::zeros({embed_dim_val});
    Tensor bias_grad_tensor = Tensor::zeros({embed_dim_val});
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = pytorch_copy_of(bias_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(output_grad_tensor);
    std::vector<int64_t> normalized_shape_list = {(int64_t)embed_dim_val};
    torch::Tensor pytorch_output_tensor;

    // 2. time each step; the first WARM_UP_STEPS do first-time work, so the average is of the rest: in step 1
    //    CUDA loads each side's kernels; in step 2 PyTorch's backward first adds onto the .grad that step 1
    //    made (26.6 ms on a T4, against 0.54 ms in every later step), most likely loading the kernel that adds
    print_test_name(__func__);
    printf("    time in ms:   step   my forward   my backward   PyTorch forward   PyTorch backward\n");
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] {
            layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor);
        });
        times_list[1] = gpu_milliseconds([&] {
            layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                                weight_tensor, mean_tensor, rstd_tensor);
        });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_output_tensor = torch::layer_norm(pytorch_input_tensor, normalized_shape_list, pytorch_weight_tensor,
                                                      pytorch_bias_tensor, 1e-5);
        });
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        printf("                  %4d   %10.3f   %11.3f   %15.3f   %16.3f\n", step_val, times_list[0], times_list[1],
               times_list[2], times_list[3]);
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    printf("    average of steps %d..%d:  %10.3f   %11.3f   %15.3f   %16.3f\n", WARM_UP_STEPS + 1, TIMED_STEPS,
           totals_list[0] / averaged_val, totals_list[1] / averaged_val, totals_list[2] / averaged_val,
           totals_list[3] / averaged_val);
    printf("    PyTorch takes, against mine:                         %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_NEAR(output_tensor.tolist(), list_of(pytorch_output_tensor), TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(weight_grad_tensor.tolist(), list_of(pytorch_weight_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(bias_grad_tensor.tolist(), list_of(pytorch_bias_tensor.grad()), TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's sizes (batch_size = 4,
    // seq_len = 1024: 4096 rows of embed_dim = 768), in float and in bfloat16, for 10 block counts x 4 thread
    // counts, each the average of 20 runs: the 8 picked on a T4, then this GPU's SM count x 1 and x 2 (114
    // and 228 on an H100 PCIe, 132 and 264 on an SXM; on a T4 they repeat 40 and 80). The 40 launches take
    // turns: 20 rounds, each timing every launch
    // once (the forward and the backward in float, then both in bfloat16). One launch's 20 runs in a row
    // would all land in the same stretch of the test, and a T4's clock can drift as it heats up (the first
    // launch timed would get the fastest clock); taking turns gives every launch runs from early, middle and
    // late alike. Run 1 is counted too: warm_up_gpu has already done the first-time work. The kernels are
    // launched here directly, <<<blocks, threads>>>, instead of through layer_norm_forward and
    // layer_norm_backward, which always launch 256 x 512 and 80 x 512 (a T4's sweep picked 256 x 512 and
    // 40 x 512, before them 512 x 256 and 128 x 512; an H100 PCIe's, 80 x 512 for the backward). Not 1024
    // threads: since the kernels
    // became templates for bfloat16, each takes 66 registers per thread, and a block of 1024 may have
    // 65,536 / 1024 = 64 (cudaFuncGetAttributes: largest block 896). Held to 64, the backward spilled 8
    // bytes; and blocks of 1024 were never the fastest here. A bfloat16 row is 1.5 KB instead of 3 KB, so
    // twice as many rows fit in an SM's L1: its best launch may differ.
    //
    // Expect: every launch gives the same result as the usual ones of its dtype. Whatever the counts, each
    // row is one warp's and its threads add in the same order, so output, mean, rstd and input_grad are the
    // same to the last bit; weight_grad and bias_grad within TOLERANCE in float (atomicAdd's order). In
    // bfloat16 they are not checked, only printed: each block's add onto them is rounded to 8 bits, in
    // whatever order the blocks finish, so even two runs of one launch differ, by about 2.5% of the largest
    // value with 128 blocks (each column about ±64, a bfloat16 step there 0.25 to 0.5, 128 rounded adds).
    // The column "bf16 grad error" is the largest difference of the bfloat16 weight_grad and bias_grad from
    // the float ones, as a fraction of their largest value: more blocks, more rounded adds, a larger error.
    // Only the time changes. On a Colab T4 (at most 40 SMs x 32 warps = 1,280 warps at once), in float, both were fastest
    // with far fewer warps than that: the forward with 160 to 640 (0.117 ms, against 0.167 for its usual
    // 4096), the backward with about 512 (0.275 ms, against 0.333 for its usual 2048). Most likely the
    // cache: the forward reads each row 3 times and the backward 2, and the re-reads come from an SM's L1
    // (at most 64 KB) only while its warps' rows fit: 16 x 3 KB = 48 KB does, 26 x 3 KB = 77 KB does not.
    // Not measured directly yet.

    // 1. GPT-2 small's sizes in float, and the same values in bfloat16 with outputs of their own (mean and
    //    rstd are float in both)
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t embed_dim_val = 768;
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor input_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});      // made on the CPU: a few seconds
    Tensor weight_tensor = Tensor::randn({embed_dim_val});
    Tensor bias_tensor = Tensor::randn({embed_dim_val});
    Tensor output_grad_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor weight_grad_tensor = Tensor::zeros({embed_dim_val});
    Tensor bias_grad_tensor = Tensor::zeros({embed_dim_val});
    Tensor input_bf16_tensor = input_tensor.to(DType::BFloat16);
    Tensor output_grad_bf16_tensor = output_grad_tensor.to(DType::BFloat16);
    Tensor weight_bf16_tensor = weight_tensor.to(DType::BFloat16);
    Tensor bias_bf16_tensor = bias_tensor.to(DType::BFloat16);
    Tensor output_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
    Tensor mean_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor rstd_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor input_grad_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
    Tensor weight_grad_bf16_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    Tensor bias_grad_bf16_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);

    // 2. the usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor);
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                        weight_tensor, mean_tensor, rstd_tensor);
    layer_norm_forward(output_bf16_tensor, mean_bf16_tensor, rstd_bf16_tensor, input_bf16_tensor, weight_bf16_tensor,
                       bias_bf16_tensor);
    layer_norm_backward(input_grad_bf16_tensor, weight_grad_bf16_tensor, bias_grad_bf16_tensor, output_grad_bf16_tensor,
                        input_bf16_tensor, weight_bf16_tensor, mean_bf16_tensor, rstd_bf16_tensor);
    torch::Tensor usual_output_tensor = pytorch_copy_of(output_tensor);
    torch::Tensor usual_mean_tensor = pytorch_copy_of(mean_tensor);
    torch::Tensor usual_rstd_tensor = pytorch_copy_of(rstd_tensor);
    torch::Tensor usual_input_grad_tensor = pytorch_copy_of(input_grad_tensor);
    torch::Tensor usual_weight_grad_tensor = pytorch_copy_of(weight_grad_tensor);
    torch::Tensor usual_bias_grad_tensor = pytorch_copy_of(bias_grad_tensor);
    torch::Tensor usual_output_bf16_tensor = pytorch_copy_of(output_bf16_tensor.to(DType::Float32));
    torch::Tensor usual_mean_bf16_tensor = pytorch_copy_of(mean_bf16_tensor);
    torch::Tensor usual_rstd_bf16_tensor = pytorch_copy_of(rstd_bf16_tensor);
    torch::Tensor usual_input_grad_bf16_tensor = pytorch_copy_of(input_grad_bf16_tensor.to(DType::Float32));

    // what layer_norm_forward and layer_norm_backward launch, worked out as they do: 256 and 80 blocks
    int forward_warps_per_block_val = LAYER_NORM_THREADS_PER_BLOCK / THREADS_PER_WARP;
    int usual_forward_blocks_val = (int)CEIL_DIV(rows_val, (size_t)forward_warps_per_block_val);
    int usual_backward_blocks_val = LAYER_NORM_BACKWARD_BLOCKS;

    // 3. the four launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    size_t shared_bytes_val = 2 * embed_dim_val * sizeof(float);     // the backward's block copy of bias_grad, weight_grad
    auto forward_val = [&] {
        layer_norm_forward_kernel<float><<<blocks_val, threads_val>>>(output_tensor.data(), mean_tensor.data(),
                                                                      rstd_tensor.data(), input_tensor.data(),
                                                                      weight_tensor.data(), bias_tensor.data(), rows_val,
                                                                      embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_val = [&] {
        layer_norm_backward_kernel<float><<<blocks_val, threads_val, shared_bytes_val>>>(
            input_grad_tensor.data(), weight_grad_tensor.data(), bias_grad_tensor.data(), output_grad_tensor.data(),
            input_tensor.data(), weight_tensor.data(), mean_tensor.data(), rstd_tensor.data(), rows_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16_val = [&] {
        layer_norm_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            output_bf16_tensor.data_ptr<nv_bfloat16>(), mean_bf16_tensor.data(), rstd_bf16_tensor.data(),
            input_bf16_tensor.data_ptr<nv_bfloat16>(), weight_bf16_tensor.data_ptr<nv_bfloat16>(),
            bias_bf16_tensor.data_ptr<nv_bfloat16>(), rows_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_bf16_val = [&] {
        layer_norm_backward_kernel<nv_bfloat16><<<blocks_val, threads_val, shared_bytes_val>>>(
            input_grad_bf16_tensor.data_ptr<nv_bfloat16>(), weight_grad_bf16_tensor.data_ptr<nv_bfloat16>(),
            bias_grad_bf16_tensor.data_ptr<nv_bfloat16>(), output_grad_bf16_tensor.data_ptr<nv_bfloat16>(),
            input_bf16_tensor.data_ptr<nv_bfloat16>(), weight_bf16_tensor.data_ptr<nv_bfloat16>(), mean_bf16_tensor.data(),
            rstd_bf16_tensor.data(), rows_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = usual_forward_blocks_val;
    threads_val = LAYER_NORM_THREADS_PER_BLOCK;
    warm_up_gpu(forward_val);

    // 4. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 80, 128, 256, 512, 1024, 2048, 4096, sms_val, 2 * sms_val};
    int thread_counts_list[] = {2 * THREADS_PER_WARP, 4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP};
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
    // the largest difference between a launch's gradient and the usual one, as a fraction of the usual one's
    // largest value
    auto difference_val = [](const torch::Tensor &actual_tensor, const torch::Tensor &usual_tensor) {
        return ((actual_tensor - usual_tensor).abs().max() / usual_tensor.abs().max()).item<float>();
    };
    const char *kernel_names_list[] = {"forward", "backward", "bf16 forward", "bf16 backward"};
    float fastest_ms_list[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    int fastest_blocks_list[4] = {};
    int fastest_threads_list[4] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads    warps   rows per warp   forward   backward   bf16 forward   "
           "bf16 backward   bf16 grad error   same result\n");
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

        // one forward and one backward from 0 with this launch in each dtype, against the usual launches':
        // output, mean, rstd and input_grad to the last bit, weight_grad and bias_grad within TOLERANCE in
        // float; in bfloat16 only their error from the float ones, printed
        for (const Tensor *tensor_ptr : {&output_tensor, &mean_tensor, &rstd_tensor, &input_grad_tensor,
                                         &weight_grad_tensor, &bias_grad_tensor, &output_bf16_tensor, &mean_bf16_tensor,
                                         &rstd_bf16_tensor, &input_grad_bf16_tensor, &weight_grad_bf16_tensor,
                                         &bias_grad_bf16_tensor})
            tensor_ptr->zero_();
        forward_val();
        backward_val();
        forward_bf16_val();
        backward_bf16_val();
        bool same_val = torch::equal(pytorch_copy_of(output_tensor), usual_output_tensor) &&
                        torch::equal(pytorch_copy_of(mean_tensor), usual_mean_tensor) &&
                        torch::equal(pytorch_copy_of(rstd_tensor), usual_rstd_tensor) &&
                        torch::equal(pytorch_copy_of(input_grad_tensor), usual_input_grad_tensor) &&
                        difference_val(pytorch_copy_of(weight_grad_tensor), usual_weight_grad_tensor) <= TOLERANCE &&
                        difference_val(pytorch_copy_of(bias_grad_tensor), usual_bias_grad_tensor) <= TOLERANCE &&
                        torch::equal(pytorch_copy_of(output_bf16_tensor.to(DType::Float32)), usual_output_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(mean_bf16_tensor), usual_mean_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(rstd_bf16_tensor), usual_rstd_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(input_grad_bf16_tensor.to(DType::Float32)), usual_input_grad_bf16_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        float bf16_grad_error_val =
            std::max(difference_val(pytorch_copy_of(weight_grad_bf16_tensor.to(DType::Float32)), usual_weight_grad_tensor),
                     difference_val(pytorch_copy_of(bias_grad_bf16_tensor.to(DType::Float32)), usual_bias_grad_tensor));

        int warps_val = blocks_val * threads_val / THREADS_PER_WARP;
        int rows_per_warp_val = (int)CEIL_DIV(rows_val, (size_t)warps_val);    // the most any warp takes
        const char *usual_ptr = "";
        if (blocks_val == usual_forward_blocks_val && threads_val == LAYER_NORM_THREADS_PER_BLOCK)
            usual_ptr = "   <- what layer_norm_forward launches";
        if (blocks_val == usual_backward_blocks_val && threads_val == LAYER_NORM_BACKWARD_THREADS_PER_BLOCK)
            usual_ptr = "   <- what layer_norm_backward launches";
        printf("                  %6d   %7d   %6d   %13d   %7.3f   %8.3f   %12.3f   %13.3f   %14.2f%%   %-11s%s\n",
               blocks_val, threads_val, warps_val, rows_per_warp_val, times_list[0], times_list[1], times_list[2],
               times_list[3], 100 * bf16_grad_error_val, same_val ? "yes" : "NO", usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < 4; kernel_index_val++)
        printf("    fastest %-13s  %4d blocks x %4d threads: %.3f ms\n", kernel_names_list[kernel_index_val],
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
