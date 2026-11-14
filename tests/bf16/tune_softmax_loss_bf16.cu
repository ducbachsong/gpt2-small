// tune_softmax_loss_bf16.cu — the timings of llmc/kernels/bf16/softmax_loss_bf16.cuh, apart from
// tests/bf16/test_softmax_loss_bf16.cu so the tests only check numbers: on a training micro-batch's logits, fp32/classifier.cuh
// (the forward and the backward as the model ran them before) against softmax_loss_bf16's forward then backward, and its
// one fused launch; then the fused launch at each block size.
//
// Built as tests/bf16/test_softmax_loss_bf16.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/softmax_loss_bf16_<date>_<time>.txt.
#include <torch/types.h>              // torch::randn; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"             // gpu_milliseconds, warm_up_gpu, gigabytes_per_second, peak_gigabytes_per_second
#include "../../llmc/kernels/bf16/softmax_loss_bf16.cuh"
#include "../../llmc/kernels/fp32/classifier.cuh"   // the model's classifier before this one

#define GPT2_VOCAB_SIZE 50257         // GPT-2's real words
#define GPT2_PADDED_VOCAB_SIZE 50304  // the row length: the words, then padding up to a multiple of 128
#define TRAINING_BATCH_SIZE 16        // a training micro-batch: 16 rows of 1024 tokens (64 in the model: 4x the bytes, the same speed)
#define GPT2_SEQ_LEN 1024

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a training micro-batch's logits, bfloat16, random with std 2, made by PyTorch on the GPU.
static Tensor random_logits(void)
{
    Tensor floats_tensor = Tensor::empty({TRAINING_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_PADDED_VOCAB_SIZE});
    torch::Tensor values_tensor = 2.0f * torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes random targets under GPT2_VOCAB_SIZE for a training micro-batch, made by PyTorch on the GPU.
static Tensor random_targets(void)
{
    Tensor targets_tensor = Tensor::empty({TRAINING_BATCH_SIZE, GPT2_SEQ_LEN}, DType::Int32);
    torch::Tensor values_tensor = torch::randint(GPT2_VOCAB_SIZE, {(int64_t)targets_tensor.numel()},
                                                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    cudaCheck(cudaMemcpy(targets_tensor.data_ptr<int>(), values_tensor.data_ptr<int>(), targets_tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    return targets_tensor;
}

// ══ part 1: against the classifier the model ran before ═══════════════════════

static void time_the_three_ways_on_a_training_micro_batch(void)
{
    // Prints, on a training micro-batch's logits ({16, 1024} rows of 50304 bfloat16s, 1.65 GB), the time of the loss
    // and its gradient three ways, averages of steps WARM_UP_STEPS + 1 .. TIMED_STEPS: fp32/classifier.cuh's forward and
    // backward (the model's before); softmax_loss_bf16_forward then _backward; softmax_loss_bf16_forward_backward. Each
    // with the bytes the way has to move at least (the forward reads the logits; the backward reads them and writes
    // the gradient; fused, once each: the second read is L2's) and the share of the GPU's peak bandwidth that is.
    // 1. the tensors
    torch::manual_seed(0);
    Tensor logits_tensor = random_logits();
    Tensor targets_tensor = random_targets();
    Tensor logits_grad_tensor = Tensor::zeros(logits_tensor.shape(), DType::BFloat16);
    Tensor losses_tensor = Tensor::zeros(targets_tensor.shape());
    Tensor max_tensor = Tensor::zeros(targets_tensor.shape());
    Tensor sum_tensor = Tensor::zeros(targets_tensor.shape());
    Tensor losses_grad_tensor = Tensor::zeros(targets_tensor.shape());
    float loss_grad_val = 1.0f / (float)losses_tensor.numel();
    losses_grad_tensor.fill_(loss_grad_val);
    // 2. TIMED_STEPS of each way, the first WARM_UP_STEPS left out
    double totals_list[3] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[3];
        times_list[0] = gpu_milliseconds([&] {
            classifier_forward(losses_tensor, max_tensor, sum_tensor, logits_tensor, targets_tensor, GPT2_VOCAB_SIZE);
            classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor, max_tensor, sum_tensor,
                                GPT2_VOCAB_SIZE);
        });
        times_list[1] = gpu_milliseconds([&] {
            softmax_loss_bf16_forward(losses_tensor, max_tensor, sum_tensor, logits_tensor, targets_tensor, GPT2_VOCAB_SIZE);
            softmax_loss_bf16_backward(logits_grad_tensor, logits_tensor, targets_tensor, max_tensor, sum_tensor, GPT2_VOCAB_SIZE,
                                       loss_grad_val);
        });
        times_list[2] = gpu_milliseconds([&] {
            softmax_loss_bf16_forward_backward(losses_tensor, max_tensor, sum_tensor, logits_grad_tensor, logits_tensor,
                                               targets_tensor, GPT2_VOCAB_SIZE, loss_grad_val);
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int way_index_val = 0; way_index_val < 3; way_index_val++)
            totals_list[way_index_val] += times_list[way_index_val];
    }
    // 3. print
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double logits_bytes_val = (double)logits_tensor.numel() * sizeof(nv_bfloat16);
    double bytes_list[3] = {3 * logits_bytes_val, 3 * logits_bytes_val, 2 * logits_bytes_val};
    const char *ways_list[3] = {"fp32/classifier.cuh", "forward then backward", "fused"};
    print_test_name(__func__);
    printf("    {%d, %d} rows, %.2f GB of logits; average of steps %d..%d\n", TRAINING_BATCH_SIZE, GPT2_SEQ_LEN,
           logits_bytes_val / GIGA, WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        %-24s %8s %10s %10s %8s\n", "way", "ms", "GB moved", "GB/s", "of peak");
    for (int way_index_val = 0; way_index_val < 3; way_index_val++)
    {
        double milliseconds_val = totals_list[way_index_val] / averaged_val;
        double rate_val = gigabytes_per_second(bytes_list[way_index_val], milliseconds_val);
        printf("        %-24s %8.3f %10.2f %10.0f %7.1f%%\n", ways_list[way_index_val], milliseconds_val,
               bytes_list[way_index_val] / GIGA, rate_val, 100.0 * rate_val / peak_gigabytes_per_second());
    }
    printf("    fp32/classifier.cuh takes %.2fx as long as fused\n", totals_list[0] / totals_list[2]);
}

// ══ part 2: the block size swept ══════════════════════════════════════════════

static void time_each_block_size_of_the_fused_launch(void)
{
    // Prints the fused launch's time on a training micro-batch at 64 to 1024 threads a block, taking turns over
    // SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged; then the fastest against the default
    // (SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK). Fewer threads, more blocks on an SM, more rows waiting for their second read
    // in L2 (50 MB: about 500 rows of 100 KB).
    const int threads_list[] = {2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    const int threads_count_val = (int)(sizeof(threads_list) / sizeof(threads_list[0]));
    // 1. the tensors
    torch::manual_seed(0);
    Tensor logits_tensor = random_logits();
    Tensor targets_tensor = random_targets();
    Tensor logits_grad_tensor = Tensor::zeros(logits_tensor.shape(), DType::BFloat16);
    Tensor losses_tensor = Tensor::zeros(targets_tensor.shape());
    Tensor max_tensor = Tensor::zeros(targets_tensor.shape());
    Tensor sum_tensor = Tensor::zeros(targets_tensor.shape());
    float loss_grad_val = 1.0f / (float)losses_tensor.numel();
    // 2. SWEEP_ROUNDS rounds, each timing every size once
    std::vector<double> totals_list(threads_count_val, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] {
                softmax_loss_bf16_forward_backward(losses_tensor, max_tensor, sum_tensor, logits_grad_tensor, logits_tensor,
                                                   targets_tensor, GPT2_VOCAB_SIZE, loss_grad_val, threads_list[threads_index_val]);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[threads_index_val] += milliseconds_val;
        }
    // 3. print, the fastest and the default
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    int best_index_val = 0;
    int default_index_val = 0;
    print_test_name(__func__);
    printf("    fused, ms, average of rounds %d..%d\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS);
    for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
    {
        printf("        %5d threads   %8.3f\n", threads_list[threads_index_val], totals_list[threads_index_val] / averaged_val);
        if (totals_list[threads_index_val] < totals_list[best_index_val])
            best_index_val = threads_index_val;
        if (threads_list[threads_index_val] == SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK)
            default_index_val = threads_index_val;
    }
    printf("    fastest: %d threads, %.3f ms; the default (%d) %.3f ms, %.2fx as long\n", threads_list[best_index_val],
           totals_list[best_index_val] / averaged_val, SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK,
           totals_list[default_index_val] / averaged_val, totals_list[default_index_val] / totals_list[best_index_val]);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_the_three_ways_on_a_training_micro_batch();
    time_each_block_size_of_the_fused_launch();

    printf("\nall tunes done\n");
    return 0;
}
