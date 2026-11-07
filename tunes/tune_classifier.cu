// tune_classifier.cu — the timings of tests/test_classifier.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size, time_each_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_classifier.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/classifier_<date>_<time>.txt.
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/kernels/classifier.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_VOCAB_SIZE 50257         // GPT-2's real words
#define GPT2_PADDED_VOCAB_SIZE 50304  // the row length: the words, then padding up to a multiple of 128

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes minutes for the logits' 206M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes an int tensor of `shape_list` with random targets 0 .. vocab_size - 1, made by PyTorch on the GPU.
static Tensor random_targets_on_gpu(const Shape &shape_list, size_t vocab_size_val)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    torch::Tensor values_tensor = torch::randint((int64_t)vocab_size_val, {(int64_t)tensor.numel()}, ints_on_gpu_val);
    cudaCheck(cudaMemcpy(tensor.data_ptr<int>(), values_tensor.data_ptr<int>(), tensor.numel() * sizeof(int),
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

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets_tensor)
{
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(targets_tensor.shape().begin(), targets_tensor.shape().end());
    return torch::from_blob(targets_tensor.data_ptr<int>(), sizes_list, ints_on_gpu_val).to(torch::kLong);
}

/// Gives the first vocab_size columns of pytorch_logits {batch_size, seq_len, padded_vocab_size} as
/// {rows, vocab_size}: the words, without the padding, the way cross_entropy takes them.
static torch::Tensor word_logits_of(const torch::Tensor &pytorch_logits_tensor, size_t vocab_size_val)
{
    int64_t rows_val = pytorch_logits_tensor.size(0) * pytorch_logits_tensor.size(1);
    return pytorch_logits_tensor.narrow(2, 0, (int64_t)vocab_size_val).reshape({rows_val, (int64_t)vocab_size_val});
}

/// Gives the largest difference between my tensor and a PyTorch tensor of the same values, as a
/// fraction of the PyTorch tensor's largest value, worked out on the GPU: for tensors too big to copy
/// to the CPU and compare there.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's sizes, batch_size = 4 rows of
    // seq_len = 1024 tokens, padded_vocab_size = 50304: logits of 206M floats, 824 MB. Mine against
    // PyTorch's cross_entropy (mean) and autograd. My forward reads the logits once: at a T4's 320 GB/s
    // at least 2.58 ms. My backward reads them once and writes logits_grad: 2 x 824 MB, at least
    // 5.15 ms. PyTorch's forward includes copying the 50257 word columns out of the padded rows (narrow,
    // then reshape), as a PyTorch user would write it. Both sides write the gradient over each step; at
    // the end they must agree.

    // 1. random logits and targets, mine and PyTorch's
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t vocab_size_val = GPT2_VOCAB_SIZE;
    size_t padded_vocab_size_val = GPT2_PADDED_VOCAB_SIZE;
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor logits_tensor = random_tensor_on_gpu({batch_size_val, seq_len_val, padded_vocab_size_val});
    Tensor targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    Tensor losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor max_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor sum_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor losses_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val}).fill_(1.0 / rows_val);
    Tensor logits_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, padded_vocab_size_val});
    torch::Tensor pytorch_logits_tensor = pytorch_copy_of(logits_tensor).requires_grad_(true);
    torch::Tensor pytorch_targets_tensor = pytorch_targets_of(targets_tensor).reshape({-1});
    torch::Tensor pytorch_losses_tensor;
    torch::Tensor pytorch_loss_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work (in step 1 CUDA loads each side's
    //    kernels), so the averages are of the rest
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] {
            classifier_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor,
                               targets_tensor, vocab_size_val);
        });
        times_list[1] = gpu_milliseconds([&] {
            classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor,
                                max_of_each_row_tensor, sum_of_each_row_tensor, vocab_size_val);
        });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_losses_tensor = torch::cross_entropy_loss(word_logits_of(pytorch_logits_tensor, vocab_size_val),
                                                              pytorch_targets_tensor, {}, at::Reduction::None);
            pytorch_loss_tensor = pytorch_losses_tensor.mean();
        });
        pytorch_logits_tensor.mutable_grad() = torch::Tensor(); // the gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_loss_tensor.backward(); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double logits_gb_val = (double)logits_tensor.numel() * sizeof(float) / 1e9;     // 0.824 GB
    double moved_gb_list[2] = {logits_gb_val, 2 * logits_gb_val};
    print_test_name(__func__);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("        GB/s                  %10.1f   %11.1f\n", moved_gb_list[0] / (totals_list[0] / averaged_val / 1e3),
           moved_gb_list[1] / (totals_list[1] / averaged_val / 1e3));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_NEAR(relative_difference_on_gpu(losses_tensor, pytorch_losses_tensor), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(logits_grad_tensor, pytorch_logits_tensor.grad()), 0.0, TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's sizes (4096 rows of 50304), in float
    // and in bfloat16, each for 9 block counts x 4 thread counts, each the average of 20 runs: the 7 picked on
    // a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264 on an SXM; on a T4
    // they repeat 40 and 80). Each kernel's 36 launches take turns: 20 rounds, each timing every launch once (in float, then in bfloat16). One
    // launch's 20 runs in a row would all land in the same stretch of the test, and a T4's clock can drift
    // as it heats up (the first launch timed would get the fastest clock); taking turns gives every launch
    // runs from early, middle and late alike. Run 1 is counted too: warm_up_gpu has already done the
    // first-time work. The kernels are launched here directly, <<<blocks, threads>>>, instead of through
    // classifier_forward and classifier_backward, which always launch 4096 x 256 (one row per block; 4096 x
    // 1024 before this sweep picked 256) and CLASSIFIER_BACKWARD_BLOCKS x CLASSIFIER_BACKWARD_THREADS_PER_BLOCK
    // (40,960 x 128, 10 float4s per thread: an H100 PCIe's pick; a T4's was 201,216 x 256, one float4 per
    // thread). A bfloat16 row is 98 KB instead of 196 KB, and a bfloat16 float4
    // one 8-byte read instead of 16: the best launch may differ.
    //
    // Expect: every forward launch gives the usual losses, max_of_each_row and sum_of_each_row of its dtype
    // within TOLERANCE, not to the last bit: another thread count adds each row's exps in another order (in
    // float either way). Every backward launch gives the usual logits_grad of its dtype to the last bit: each
    // value is worked out alone, from the same numbers. A T4 runs at most 1024 threads per SM, 40,960 in all.

    // 1. GPT-2 small's logits and targets in float, and the logits in bfloat16 with results of their own
    //    (losses, the row maxes and sums are float in both)
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t vocab_size_val = GPT2_VOCAB_SIZE;
    size_t padded_vocab_size_val = GPT2_PADDED_VOCAB_SIZE;
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor logits_tensor = random_tensor_on_gpu({batch_size_val, seq_len_val, padded_vocab_size_val});
    Tensor targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    Tensor losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor max_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor sum_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor losses_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val}).fill_(1.0 / rows_val);
    Tensor logits_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, padded_vocab_size_val});
    Tensor logits_bf16_tensor = logits_tensor.to(DType::BFloat16);
    Tensor losses_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor max_of_each_row_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor sum_of_each_row_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor logits_grad_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val, padded_vocab_size_val}, DType::BFloat16);

    // 2. the usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with
    classifier_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor, targets_tensor,
                       vocab_size_val);
    classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor, max_of_each_row_tensor,
                        sum_of_each_row_tensor, vocab_size_val);
    classifier_forward(losses_bf16_tensor, max_of_each_row_bf16_tensor, sum_of_each_row_bf16_tensor, logits_bf16_tensor,
                       targets_tensor, vocab_size_val);
    classifier_backward(logits_grad_bf16_tensor, losses_grad_tensor, logits_bf16_tensor, targets_tensor,
                        max_of_each_row_bf16_tensor, sum_of_each_row_bf16_tensor, vocab_size_val);
    torch::Tensor usual_losses_tensor = pytorch_copy_of(losses_tensor);
    torch::Tensor usual_max_of_each_row_tensor = pytorch_copy_of(max_of_each_row_tensor);
    torch::Tensor usual_sum_of_each_row_tensor = pytorch_copy_of(sum_of_each_row_tensor);
    torch::Tensor usual_logits_grad_tensor = pytorch_copy_of(logits_grad_tensor);
    torch::Tensor usual_losses_bf16_tensor = pytorch_copy_of(losses_bf16_tensor);
    torch::Tensor usual_max_of_each_row_bf16_tensor = pytorch_copy_of(max_of_each_row_bf16_tensor);
    torch::Tensor usual_sum_of_each_row_bf16_tensor = pytorch_copy_of(sum_of_each_row_bf16_tensor);
    torch::Tensor usual_logits_grad_bf16_tensor = pytorch_copy_of(logits_grad_bf16_tensor.to(DType::Float32));
    const int t4_backward_threads_val = 8 * THREADS_PER_WARP;  // 256: a T4's pick, one float4 per thread
    int one_float4_per_thread_blocks_val =
        (int)CEIL_DIV(logits_tensor.numel() / VALUES_PER_FLOAT4, (size_t)t4_backward_threads_val);   // 201,216

    // 3. the four launches, at the launch the loops set
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        classifier_forward_kernel<float><<<blocks_val, threads_val>>>(
            losses_tensor.data(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), logits_tensor.data(),
            targets_tensor.data_ptr<int>(), rows_val, vocab_size_val, padded_vocab_size_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_val = [&] {
        classifier_backward_kernel<float><<<blocks_val, threads_val>>>(
            logits_grad_tensor.data(), losses_grad_tensor.data(), logits_tensor.data(), targets_tensor.data_ptr<int>(),
            max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), rows_val, vocab_size_val, padded_vocab_size_val);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16_val = [&] {
        classifier_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            losses_bf16_tensor.data(), max_of_each_row_bf16_tensor.data(), sum_of_each_row_bf16_tensor.data(),
            logits_bf16_tensor.data_ptr<nv_bfloat16>(), targets_tensor.data_ptr<int>(), rows_val, vocab_size_val,
            padded_vocab_size_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_bf16_val = [&] {
        classifier_backward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            logits_grad_bf16_tensor.data_ptr<nv_bfloat16>(), losses_grad_tensor.data(),
            logits_bf16_tensor.data_ptr<nv_bfloat16>(), targets_tensor.data_ptr<int>(), max_of_each_row_bf16_tensor.data(),
            sum_of_each_row_bf16_tensor.data(), rows_val, vocab_size_val, padded_vocab_size_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = (int)rows_val;
    threads_val = CLASSIFIER_FORWARD_THREADS_PER_BLOCK;
    warm_up_gpu(forward_val);

    // 4. the forward: time every launch over SWEEP_ROUNDS rounds; losses, max_of_each_row and sum_of_each_row
    //    within TOLERANCE of the usual launch's
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int forward_block_counts_list[] = {40, 80, 160, 320, 640, 1280, 4096, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 9 * thread_count_choices_val;
    float total_ms_list[launches_val][2] = {}; // launch k is block count number k / 4 x thread_counts[k % 4]; [0] float, [1] bf16
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = forward_block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(forward_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(forward_bf16_val);
        }
    }
    float fastest_ms_list[2] = {1e9f, 1e9f};
    int fastest_blocks_list[2] = {};
    int fastest_threads_list[2] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    forward, time in ms:    blocks   threads   rows per block    float     bf16   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = forward_block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[2];
        for (int dtype_index_val = 0; dtype_index_val < 2; dtype_index_val++)
        {
            times_list[dtype_index_val] = total_ms_list[launch_val][dtype_index_val] / SWEEP_ROUNDS;
            if (times_list[dtype_index_val] < fastest_ms_list[dtype_index_val])
            {
                fastest_ms_list[dtype_index_val] = times_list[dtype_index_val];
                fastest_blocks_list[dtype_index_val] = blocks_val;
                fastest_threads_list[dtype_index_val] = threads_val;
            }
        }
        for (const Tensor *tensor_ptr : {&losses_tensor, &max_of_each_row_tensor, &sum_of_each_row_tensor,
                                         &losses_bf16_tensor, &max_of_each_row_bf16_tensor, &sum_of_each_row_bf16_tensor})
            tensor_ptr->zero_();
        forward_val();
        forward_bf16_val();
        bool same_val = relative_difference_on_gpu(losses_tensor, usual_losses_tensor) <= TOLERANCE &&
                        relative_difference_on_gpu(max_of_each_row_tensor, usual_max_of_each_row_tensor) <= TOLERANCE &&
                        relative_difference_on_gpu(sum_of_each_row_tensor, usual_sum_of_each_row_tensor) <= TOLERANCE &&
                        relative_difference_on_gpu(losses_bf16_tensor, usual_losses_bf16_tensor) <= TOLERANCE &&
                        relative_difference_on_gpu(max_of_each_row_bf16_tensor, usual_max_of_each_row_bf16_tensor) <= TOLERANCE &&
                        relative_difference_on_gpu(sum_of_each_row_bf16_tensor, usual_sum_of_each_row_bf16_tensor) <= TOLERANCE;
        if (same_val)
            launches_with_the_same_result_val++;
        int rows_per_block_val = (int)CEIL_DIV(rows_val, (size_t)blocks_val); // the most any block takes
        const char *usual_ptr = blocks_val == (int)rows_val && threads_val == CLASSIFIER_FORWARD_THREADS_PER_BLOCK
                                    ? "   <- what classifier_forward launches"
                                    : "";
        printf("                            %6d   %7d   %14d   %6.3f   %6.3f   %-11s%s\n", blocks_val, threads_val,
               rows_per_block_val, times_list[0], times_list[1], same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest forward        %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[0], fastest_threads_list[0],
           fastest_ms_list[0]);
    printf("    fastest bf16 forward   %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[1], fastest_threads_list[1],
           fastest_ms_list[1]);
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);

    // 5. the backward: logits_grad equal to the usual launch's, to the last bit. It reads the usual forward's
    //    max_of_each_row and sum_of_each_row, which the last forwards above wrote again within TOLERANCE, so
    //    put those back first, in both dtypes
    cudaCheck(cudaMemcpy(max_of_each_row_tensor.data(), usual_max_of_each_row_tensor.data_ptr<float>(),
                         rows_val * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(sum_of_each_row_tensor.data(), usual_sum_of_each_row_tensor.data_ptr<float>(),
                         rows_val * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(max_of_each_row_bf16_tensor.data(), usual_max_of_each_row_bf16_tensor.data_ptr<float>(),
                         rows_val * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(sum_of_each_row_bf16_tensor.data(), usual_sum_of_each_row_bf16_tensor.data_ptr<float>(),
                         rows_val * sizeof(float), cudaMemcpyDeviceToDevice));
    int backward_block_counts_list[] = {40, 160, 640, 2560, 10240, CLASSIFIER_BACKWARD_BLOCKS,
                                        one_float4_per_thread_blocks_val, sms_val, 2 * sms_val};
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
        total_ms_list[launch_val][0] = total_ms_list[launch_val][1] = 0.0f;
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = backward_block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(backward_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(backward_bf16_val);
        }
    }
    fastest_ms_list[0] = fastest_ms_list[1] = 1e9f;
    launches_with_the_same_result_val = 0;
    printf("    backward, time in ms:   blocks   threads   float4s per thread    float     bf16   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = backward_block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[2];
        for (int dtype_index_val = 0; dtype_index_val < 2; dtype_index_val++)
        {
            times_list[dtype_index_val] = total_ms_list[launch_val][dtype_index_val] / SWEEP_ROUNDS;
            if (times_list[dtype_index_val] < fastest_ms_list[dtype_index_val])
            {
                fastest_ms_list[dtype_index_val] = times_list[dtype_index_val];
                fastest_blocks_list[dtype_index_val] = blocks_val;
                fastest_threads_list[dtype_index_val] = threads_val;
            }
        }
        logits_grad_tensor.zero_();
        logits_grad_bf16_tensor.zero_();
        backward_val();
        backward_bf16_val();
        bool same_val = torch::equal(pytorch_copy_of(logits_grad_tensor), usual_logits_grad_tensor) &&
                        torch::equal(pytorch_copy_of(logits_grad_bf16_tensor.to(DType::Float32)), usual_logits_grad_bf16_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(logits_tensor.numel() / VALUES_PER_FLOAT4, threads_launched_val); // the most
        const char *usual_ptr = blocks_val == CLASSIFIER_BACKWARD_BLOCKS && threads_val == CLASSIFIER_BACKWARD_THREADS_PER_BLOCK
                                    ? "   <- what classifier_backward launches"
                                : blocks_val == one_float4_per_thread_blocks_val && threads_val == t4_backward_threads_val
                                    ? "   <- one float4 per thread: a T4's pick, what it launched before"
                                    : "";
        printf("                            %6d   %7d   %18d   %6.3f   %6.3f   %-11s%s\n", blocks_val, threads_val,
               float4s_per_thread_val, times_list[0], times_list[1], same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest backward       %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[0], fastest_threads_list[0],
           fastest_ms_list[0]);
    printf("    fastest bf16 backward  %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[1], fastest_threads_list[1],
           fastest_ms_list[1]);
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
