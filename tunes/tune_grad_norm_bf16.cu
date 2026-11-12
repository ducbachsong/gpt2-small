// tune_grad_norm_bf16.cu — the timings of llmc/kernels/bf16/grad_norm_bf16.cuh, apart from
// tests/bf16/test_grad_norm_bf16.cu so the tests only check numbers: time_8_values_against_4_values (one 16-byte load
// a thread against fp32/grad_norm.cuh's 8-byte one, on GPT-2 small's gradients and on a slice that starts off a
// 16-byte boundary), time_the_kernel_with_other_block_and_thread_counts. With the bytes each reads, GB/s and MBU.
//
// Built as tests/bf16/test_grad_norm_bf16.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/grad_norm_bf16_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <algorithm>                  // std::max
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../llmc/kernels/fp32/grad_norm.cuh"      // the 4-value path
#include "../llmc/kernels/bf16/grad_norm_bf16.cuh"

static const double TOLERANCE = 1e-5; // between two norms of the same values, as a fraction of the norm

#define GPT2_PARAMETERS 124475904     // GPT-2 small's gradients, with the padded vocabulary
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `numel_val` random values, mean 0 and standard deviation 1, made by PyTorch on the GPU
/// (the CPU would take minutes for GPT-2's 124M), then rounded to bfloat16.
static Tensor random_bf16_tensor(size_t numel_val)
{
    Tensor floats_tensor = Tensor::empty({numel_val});
    torch::Tensor values_tensor = torch::randn({(int64_t)numel_val}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), numel_val * sizeof(float), cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

// ══ part 1: 8 values a thread against 4 ═══════════════════════════════════════

static void time_8_values_against_4_values(void)
{
    // Prints the norm of GPT-2 small's 124,475,904 bfloat16 gradients (249 MB), each the average of SWEEP_ROUNDS runs
    // taking turns: GradNormBf16 (8 values a thread, one 16-byte load), GradNormBf16 on a view starting 2 bytes in
    // (a head of 7 values read one each, as a GPU's slice can start), and fp32/grad_norm.cuh's GradNorm (4 values, one
    // 8-byte load), each at its default launch (512 x 1024), with GB/s and MBU.
    //
    // Expect: the three norms within TOLERANCE of each other; the 8-value one as fast or faster than the 4-value one:
    // the same bytes in half the loads; the view as fast as the whole (7 values more or less).

    // 1. GPT-2 small's gradients; the three norms' tensors
    torch::manual_seed(0);
    Tensor grads_tensor = random_bf16_tensor(GPT2_PARAMETERS + 1);
    Tensor whole_tensor = grads_tensor.narrow(0, 0, GPT2_PARAMETERS);
    Tensor off_boundary_tensor = grads_tensor.narrow(0, 1, GPT2_PARAMETERS);
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BF16_BLOCKS});
    Tensor four_block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    Tensor norms_tensor = Tensor::zeros({3});
    GradNormBf16 grad_norm_val(block_sums_tensor);
    GradNorm four_grad_norm_val(four_block_sums_tensor);
    const int runs_val = 3;
    auto run_val = [&](int run_index_val) {
        if (run_index_val == 0) grad_norm_val.compute(whole_tensor, norms_tensor[0]);
        else if (run_index_val == 1) grad_norm_val.compute(off_boundary_tensor, norms_tensor[1]);
        else four_grad_norm_val.compute(whole_tensor, norms_tensor[2]);
    };
    warm_up_gpu([&] { run_val(0); });

    // 2. time the 3, taking turns over SWEEP_ROUNDS rounds
    float total_ms_list[runs_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
            total_ms_list[run_index_val] += gpu_milliseconds([&] { run_val(run_index_val); });

    // 3. print each with its GB/s and MBU
    std::vector<float> norms_list = norms_tensor.tolist();
    double bytes_val = (double)GPT2_PARAMETERS * sizeof(nv_bfloat16);
    double peak_val = peak_gigabytes_per_second();
    const char *names_list[runs_val] = {"GradNormBf16", "GradNormBf16, 2 bytes in", "GradNorm (4 values)"};
    print_test_name(__func__);
    printf("    %.1f MB read; this GPU's peak %.0f GB/s\n", bytes_val / MEGA, peak_val);
    printf("    run                               ms     GB/s    MBU   norm\n");
    for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
    {
        float ms_val = total_ms_list[run_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_val, ms_val);
        printf("    %-26s   %7.4f   %6.0f   %3.0f%%   %.6f\n", names_list[run_index_val], ms_val, speed_val,
               100.0 * speed_val / peak_val, norms_list[run_index_val]);
    }
    printf("    GradNorm (4 values) takes %.2fx GradNormBf16's time\n", total_ms_list[2] / total_ms_list[0]);

    // 4. the norms agree: the whole with the 4-value one; the view (one value moved along) with the whole, loosely
    ASSERT_NEAR(norms_list[0], norms_list[2], TOLERANCE * norms_list[2]);
    ASSERT_NEAR(norms_list[1], norms_list[0], 1e-3 * norms_list[0]);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_the_kernel_with_other_block_and_thread_counts(void)
{
    // Prints grad_norm_bf16_kernel on GPT-2 small's gradients for 8 block counts x 3 thread counts, each the average
    // of SWEEP_ROUNDS runs taking turns, with GB/s and MBU: 128 .. 2048 blocks and this GPU's SM count x 1, 2 and 4. The
    // kernel is launched directly, <<<blocks, threads>>>, with block sums of its own for up to the largest count.
    //
    // Expect: every launch's norm within TOLERANCE of the default's (512 x 1024): another launch adds the squares in
    // another order. Only the time changes.

    // 1. GPT-2 small's gradients; block sums for the largest launch, then the counter; the default's norm
    torch::manual_seed(0);
    Tensor grads_tensor = random_bf16_tensor(GPT2_PARAMETERS);
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 8;
    const int thread_count_choices_val = 3;
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    int block_counts_list[block_count_choices_val] = {128, 256, 512, 1024, 2048, sms_val, 2 * sms_val, 4 * sms_val};
    int thread_counts_list[thread_count_choices_val] = {8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                                        HARDWARE_MAX_THREADS_PER_BLOCK};
    int largest_blocks_val = 0;
    for (int blocks_val : block_counts_list)
        largest_blocks_val = std::max(largest_blocks_val, blocks_val);
    Tensor block_sums_tensor = Tensor::zeros({(size_t)largest_blocks_val + 1});   // the block sums, then the counter
    Tensor norm_tensor = Tensor::zeros({1});
    auto launch_val = [&](int blocks_val, int threads_val) {
        grad_norm_bf16_kernel<<<blocks_val, threads_val>>>(norm_tensor.data(), block_sums_tensor.data(),
                                                           block_sums_tensor.data() + largest_blocks_val,
                                                           grads_tensor.data_ptr<nv_bfloat16>(), GPT2_PARAMETERS, 0);
        cudaCheck(cudaGetLastError());
    };
    launch_val(GRAD_NORM_BF16_BLOCKS, GRAD_NORM_BF16_THREADS);
    float usual_norm_val = norm_tensor.item();
    warm_up_gpu([&] { launch_val(GRAD_NORM_BF16_BLOCKS, GRAD_NORM_BF16_THREADS); });

    // 2. time every launch, taking turns over SWEEP_ROUNDS rounds; launch k is block_counts[k / 3] x thread_counts[k % 3]
    float total_ms_list[launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            total_ms_list[launch_index_val] += gpu_milliseconds([&] {
                launch_val(block_counts_list[launch_index_val / thread_count_choices_val],
                           thread_counts_list[launch_index_val % thread_count_choices_val]);
            });

    // 3. print each with GB/s and MBU, and check its norm against the default's
    double bytes_val = (double)GPT2_PARAMETERS * sizeof(nv_bfloat16);
    double peak_val = peak_gigabytes_per_second();
    float fastest_ms_val = 1e9f;
    int fastest_launch_val = 0;
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB read; this GPU's peak %.0f GB/s\n", bytes_val / MEGA, peak_val);
    printf("    blocks   threads        ms     GB/s    MBU   same norm\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        int blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        int threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        launch_val(blocks_val, threads_val);
        bool same_val = fabs(norm_tensor.item() - usual_norm_val) <= TOLERANCE * usual_norm_val;
        if (same_val)
            launches_with_the_same_result_val++;
        float ms_val = total_ms_list[launch_index_val] / SWEEP_ROUNDS;
        if (ms_val < fastest_ms_val)
        {
            fastest_ms_val = ms_val;
            fastest_launch_val = launch_index_val;
        }
        double speed_val = gigabytes_per_second(bytes_val, ms_val);
        printf("    %6d   %7d   %7.4f   %6.0f   %3.0f%%   %-9s%s\n", blocks_val, threads_val, ms_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO",
               blocks_val == GRAD_NORM_BF16_BLOCKS && threads_val == GRAD_NORM_BF16_THREADS ? "   <- what GradNormBf16 launches" : "");
    }
    double fastest_speed_val = gigabytes_per_second(bytes_val, fastest_ms_val);
    printf("    fastest  %4d blocks x %4d threads: %.4f ms, %.0f GB/s, MBU %.0f%%\n",
           block_counts_list[fastest_launch_val / thread_count_choices_val],
           thread_counts_list[fastest_launch_val % thread_count_choices_val], fastest_ms_val, fastest_speed_val,
           100.0 * fastest_speed_val / peak_val);

    // 4. every launch gave the same norm
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8_values_against_4_values();
    time_the_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
