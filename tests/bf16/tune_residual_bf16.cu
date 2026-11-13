// tune_residual_bf16.cu — the timings of llmc/kernels/bf16/residual_bf16.cuh, apart from tests/bf16/test_residual_bf16.cu
// so the tests only check numbers: time_8_values_against_4_values (one 16-byte load a thread against
// fp32/residual.cuh's 8-byte one), time_each_kernel_with_other_block_and_thread_counts. With the bytes each kernel
// moves, its GB/s and MBU.
//
// Built as tests/bf16/test_residual_bf16.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/residual_bf16_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ: every check here is exact
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../llmc/kernels/fp32/residual.cuh"       // the 4-value path
#include "../../llmc/kernels/bf16/residual_bf16.cuh"

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens of 768
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_VALUES (GPT2_BATCH_SIZE * GPT2_SEQ_LEN * GPT2_EMBED_DIM)
#define GPT2_RESIDUAL_ADDS 24         // residual adds in a GPT-2 small forward (and backward): 2 a block, 12 blocks
#define KERNELS 2                     // forward, backward
#define SIDES 2                       // 8 values a thread, 4 values a thread
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of GPT-2 small's residual stream with random values, mean 0 and standard deviation 1, made
/// by PyTorch on the GPU, then rounded to bfloat16.
static Tensor random_bf16_stream(void)
{
    Tensor tensor = Tensor::empty({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values (bfloat16 turned into floats exactly).
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// GPT-2 small's residual add in bfloat16 (3,145,728 random values a tensor), and one set of results per side (8
/// values a thread, 4 values a thread).
struct Gpt2Residual
{
    Tensor residual_tensor = random_bf16_stream();
    Tensor branch_output_tensor = random_bf16_stream();
    Tensor output_grad_tensor = random_bf16_stream();
    Tensor results_list[SIDES][3];   // [side][output, residual_grad, branch_output_grad]

    Gpt2Residual()
    {
        for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
            for (int result_index_val = 0; result_index_val < 3; result_index_val++)
                results_list[side_index_val][result_index_val] =
                    Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    }

    /// Launches kernel `kernel_index_val` (forward, backward) of side `side_index_val` (8 values, 4 values) as
    /// <<<blocks_val, threads_val>>>, into that side's results.
    void launch(int side_index_val, int kernel_index_val, int blocks_val, int threads_val) const
    {
        const Tensor *results_ptr = results_list[side_index_val];
        if (side_index_val == 0 && kernel_index_val == 0)
            residual_bf16_forward_kernel<<<blocks_val, threads_val>>>(
                results_ptr[0].data_ptr<nv_bfloat16>(), residual_tensor.data_ptr<nv_bfloat16>(),
                branch_output_tensor.data_ptr<nv_bfloat16>(), GPT2_VALUES / RESIDUAL_BF16_VALUES_PER_LOAD);
        else if (side_index_val == 0)
            residual_bf16_backward_kernel<<<blocks_val, threads_val>>>(
                results_ptr[1].data_ptr<nv_bfloat16>(), results_ptr[2].data_ptr<nv_bfloat16>(),
                output_grad_tensor.data_ptr<nv_bfloat16>(), GPT2_VALUES / RESIDUAL_BF16_VALUES_PER_LOAD);
        else if (kernel_index_val == 0)
            residual_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
                results_ptr[0].data_ptr<nv_bfloat16>(), residual_tensor.data_ptr<nv_bfloat16>(),
                branch_output_tensor.data_ptr<nv_bfloat16>(), GPT2_VALUES / VALUES_PER_FLOAT4);
        else
            residual_backward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
                results_ptr[1].data_ptr<nv_bfloat16>(), results_ptr[2].data_ptr<nv_bfloat16>(),
                output_grad_tensor.data_ptr<nv_bfloat16>(), GPT2_VALUES / VALUES_PER_FLOAT4);
        cudaCheck(cudaGetLastError());
    }

    /// Gives the blocks of side `side_index_val`'s default launch at `threads_val` threads: one load per thread,
    /// rounded up.
    static int default_blocks(int side_index_val, int threads_val)
    {
        return side_index_val == 0 ? residual_bf16_blocks(GPT2_VALUES / RESIDUAL_BF16_VALUES_PER_LOAD, threads_val)
                                   : residual_blocks(GPT2_VALUES / VALUES_PER_FLOAT4, threads_val);
    }

    /// Tells if side `side_index_val`'s results are `usual_list`'s, to the last bit.
    bool same_as(int side_index_val, const Tensor (&usual_list)[3]) const
    {
        for (int result_index_val = 0; result_index_val < 3; result_index_val++)
            if (!torch::equal(pytorch_copy_of(results_list[side_index_val][result_index_val]), pytorch_copy_of(usual_list[result_index_val])))
                return false;
        return true;
    }
};

/// Gives the bytes each kernel moves, in kernel order: the forward reads 2 tensors and writes 1, the backward reads 1
/// and writes 2; 3 x 6.3 MB = 18.9 MB each.
static double kernel_bytes(void)
{
    return 3.0 * GPT2_VALUES * sizeof(nv_bfloat16);
}

// ══ part 1: 8 values a thread against 4 ═══════════════════════════════════════

static void time_8_values_against_4_values(void)
{
    // Prints the forward and the backward on GPT-2 small's residual stream in bfloat16 (3,145,728 values), here (8
    // values a thread, one 16-byte load) and in fp32/residual.cuh (4 values, one 8-byte load), each at its default
    // launch (RESIDUAL_BF16_THREADS_PER_BLOCK threads, one load a thread). The 4 take turns over SWEEP_ROUNDS rounds,
    // each the average of its rounds, with GB/s and MBU, and what the difference is over a step's GPT2_RESIDUAL_ADDS
    // forwards and backwards.
    //
    // Expect: the same results on both sides, to the last bit; the 8-value kernels as fast or faster: the same bytes
    // in half the loads.

    // 1. GPT-2 small's residual add; both sides once, checked
    torch::manual_seed(0);
    Gpt2Residual residual_val;
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
            residual_val.launch(side_index_val, kernel_index_val, Gpt2Residual::default_blocks(side_index_val, RESIDUAL_BF16_THREADS_PER_BLOCK),
                                RESIDUAL_BF16_THREADS_PER_BLOCK);
    bool same_val = residual_val.same_as(0, residual_val.results_list[1]);
    warm_up_gpu([&] { residual_val.launch(0, 0, Gpt2Residual::default_blocks(0, RESIDUAL_BF16_THREADS_PER_BLOCK), RESIDUAL_BF16_THREADS_PER_BLOCK); });

    // 2. time the 4, taking turns over SWEEP_ROUNDS rounds
    float total_ms_list[SIDES][KERNELS] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
            for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
                total_ms_list[side_index_val][kernel_index_val] += gpu_milliseconds([&] {
                    residual_val.launch(side_index_val, kernel_index_val,
                                        Gpt2Residual::default_blocks(side_index_val, RESIDUAL_BF16_THREADS_PER_BLOCK),
                                        RESIDUAL_BF16_THREADS_PER_BLOCK);
                });

    // 3. print each with its GB/s and MBU
    double bytes_val = kernel_bytes();
    double peak_val = peak_gigabytes_per_second();
    const char *kernel_names_list[KERNELS] = {"forward", "backward"};
    float saved_ms_val = 0.0f;
    print_test_name(__func__);
    printf("    %.1f MB moved by each; this GPU's peak %.0f GB/s; the same results on both sides: %s\n", bytes_val / MEGA, peak_val,
           same_val ? "yes" : "NO");
    printf("    kernel      8 values: ms    GB/s   MBU     4 values: ms    GB/s   MBU     4 / 8\n");
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
    {
        float eight_ms_val = total_ms_list[0][kernel_index_val] / SWEEP_ROUNDS;
        float four_ms_val = total_ms_list[1][kernel_index_val] / SWEEP_ROUNDS;
        double eight_speed_val = gigabytes_per_second(bytes_val, eight_ms_val);
        double four_speed_val = gigabytes_per_second(bytes_val, four_ms_val);
        saved_ms_val += four_ms_val - eight_ms_val;
        printf("    %-10s     %7.4f  %6.0f  %3.0f%%         %7.4f  %6.0f  %3.0f%%     %5.2fx\n", kernel_names_list[kernel_index_val],
               eight_ms_val, eight_speed_val, 100.0 * eight_speed_val / peak_val, four_ms_val, four_speed_val,
               100.0 * four_speed_val / peak_val, four_ms_val / eight_ms_val);
    }
    printf("    over a step's %d forwards and backwards: %.3f ms saved\n", GPT2_RESIDUAL_ADDS, GPT2_RESIDUAL_ADDS * saved_ms_val);

    // 4. both sides gave the same results
    ASSERT_EQ(same_val, true);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of both kernels here (8 values a thread) on GPT-2 small's residual stream, for 6 block counts x
    // 4 thread counts, each the average of SWEEP_ROUNDS runs taking turns, with GB/s and MBU: as many as one load a
    // thread needs, 512, 1024, 2048, and this GPU's SM count x 2 and x 4 (the grid-stride loop takes the rest).
    //
    // Expect: every launch gives the default's results to the last bit (each value one add or one copy). Only the time
    // changes.

    // 1. GPT-2 small's residual add; the default launch's results
    torch::manual_seed(0);
    Gpt2Residual residual_val;
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
        residual_val.launch(0, kernel_index_val, Gpt2Residual::default_blocks(0, RESIDUAL_BF16_THREADS_PER_BLOCK),
                            RESIDUAL_BF16_THREADS_PER_BLOCK);
    Tensor usual_list[3];
    for (int result_index_val = 0; result_index_val < 3; result_index_val++)
        usual_list[result_index_val] = residual_val.results_list[0][result_index_val].clone();
    warm_up_gpu([&] { residual_val.launch(0, 0, Gpt2Residual::default_blocks(0, RESIDUAL_BF16_THREADS_PER_BLOCK), RESIDUAL_BF16_THREADS_PER_BLOCK); });

    // 2. time every launch, taking turns over SWEEP_ROUNDS rounds; block count choice 0 is "as many as the work needs"
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 6;
    const int thread_count_choices_val = 4;
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    int block_counts_list[block_count_choices_val] = {0, 512, 1024, 2048, 2 * sms_val, 4 * sms_val};
    int thread_counts_list[thread_count_choices_val] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                        16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    auto blocks_of_val = [&](int launch_index_val) {
        int blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        return blocks_val > 0 ? blocks_val : Gpt2Residual::default_blocks(0, thread_counts_list[launch_index_val % thread_count_choices_val]);
    };
    float total_ms_list[launches_val][KERNELS] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
                total_ms_list[launch_index_val][kernel_index_val] += gpu_milliseconds([&] {
                    residual_val.launch(0, kernel_index_val, blocks_of_val(launch_index_val),
                                        thread_counts_list[launch_index_val % thread_count_choices_val]);
                });

    // 3. print each launch's averages with GB/s and MBU, and check its results against the default's
    double bytes_val = kernel_bytes();
    double peak_val = peak_gigabytes_per_second();
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB moved by each; this GPU's peak %.0f GB/s\n", bytes_val / MEGA, peak_val);
    printf("    blocks   threads   forward: ms    GB/s    MBU   backward: ms    GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        int blocks_val = blocks_of_val(launch_index_val);
        int threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        for (int result_index_val = 0; result_index_val < 3; result_index_val++)
            residual_val.results_list[0][result_index_val].zero_();
        residual_val.launch(0, 0, blocks_val, threads_val);
        residual_val.launch(0, 1, blocks_val, threads_val);
        bool same_val = residual_val.same_as(0, usual_list);
        if (same_val)
            launches_with_the_same_result_val++;
        float forward_ms_val = total_ms_list[launch_index_val][0] / SWEEP_ROUNDS;
        float backward_ms_val = total_ms_list[launch_index_val][1] / SWEEP_ROUNDS;
        double forward_speed_val = gigabytes_per_second(bytes_val, forward_ms_val);
        double backward_speed_val = gigabytes_per_second(bytes_val, backward_ms_val);
        bool usual_val = threads_val == RESIDUAL_BF16_THREADS_PER_BLOCK && blocks_val == Gpt2Residual::default_blocks(0, threads_val);
        printf("    %6d   %7d   %11.4f  %6.0f   %3.0f%%   %12.4f  %6.0f   %3.0f%%   %-11s%s\n", blocks_val, threads_val,
               forward_ms_val, forward_speed_val, 100.0 * forward_speed_val / peak_val, backward_ms_val, backward_speed_val,
               100.0 * backward_speed_val / peak_val, same_val ? "yes" : "NO", usual_val ? "   <- the default" : "");
    }

    // 4. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8_values_against_4_values();
    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
