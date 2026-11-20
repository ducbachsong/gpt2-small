// tune_encoder_bf16.cu — the timings of cuda/kernels/bf16/encoder_bf16.cuh, apart from cuda/tests/bf16/test_encoder_bf16.cu
// so the tests only check numbers: time_8_values_against_4_values (one 16-byte load a thread against
// fp32/encoder.cuh's 8-byte one), time_each_kernel_with_other_block_and_thread_counts. With the bytes each kernel
// moves, its GB/s and MBU.
//
// Built as cuda/tests/bf16/test_encoder_bf16.cu is (see cuda/tests/README.md); what it prints also goes to
// cuda/tests/results/<GPU>/encoder_bf16_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include "../common/test_helpers.h"    // ASSERT_EQ
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../kernels/fp32/encoder.cuh"       // the 4-value path
#include "../../kernels/bf16/encoder_bf16.cuh"

// wte_grad between two launches: a repeated token's adds land in any order, each rounded to bfloat16. The largest
// difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;

#define GPT2_HIGHEST_TOKEN_ID 50256   // GPT-2's real ids: 0 .. 50256
#define GPT2_PADDED_VOCAB_SIZE 50304  // GPT-2 small's sizes
#define GPT2_MAX_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_BATCH_SIZE 4
#define KERNELS 3                     // forward, wte backward, wpe backward
#define SIDES 2                       // 8 values a thread, 4 values a thread
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes an int tensor {GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN} of random token ids on the GPU.
static Tensor random_token_ids(void)
{
    // Makes an int tensor {GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN} of random token ids from 0 .. GPT2_HIGHEST_TOKEN_ID on the
    // GPU. The same seed gives the same ids every run.
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, GPT2_HIGHEST_TOKEN_ID);
    std::vector<int> ids_list(GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    Tensor tensor = Tensor::empty({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN}, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my bfloat16 tensor's values, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between two of my tensors, as a fraction of the second one's largest value.
static double relative_difference_on_gpu(const Tensor &actual_tensor, const Tensor &usual_tensor)
{
    torch::Tensor usual_values_tensor = pytorch_copy_of(usual_tensor);
    return ((pytorch_copy_of(actual_tensor) - usual_values_tensor).abs().max() / usual_values_tensor.abs().max()).item<double>();
}

/// Gives the bytes each kernel asks for, in kernel order, on GPT-2 small's sizes.
///
///   bytes_list   written
static void kernel_bytes(double (&bytes_list)[KERNELS])
{
    // Gives the bytes each kernel asks for, in kernel order (forward, wte backward, wpe backward), on GPT-2 small's
    // sizes. wpe's rows are asked for once per row of the batch (4 times), mostly from the L2 cache.
    //
    //     forward        reads 4096 wte rows, 4096 wpe rows, the ids; writes encoded
    //     wte backward   reads encoded_grad, the ids, 4096 wte_grad rows; writes those rows
    //     wpe backward   reads encoded_grad, wpe_grad; writes wpe_grad
    double rows_val = (double)GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN;
    double row_bytes_val = (double)GPT2_EMBED_DIM * sizeof(nv_bfloat16);
    double ids_bytes_val = rows_val * sizeof(int);
    bytes_list[0] = 3 * rows_val * row_bytes_val + ids_bytes_val;
    bytes_list[1] = 3 * rows_val * row_bytes_val + ids_bytes_val;
    bytes_list[2] = rows_val * row_bytes_val + 2 * (double)GPT2_MAX_SEQ_LEN * row_bytes_val;
}

/// GPT-2 small's encoder in bfloat16, and one set of results per side (8 values a thread, 4 values a thread).
struct Gpt2Encoder
{
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor token_ids_tensor = random_token_ids();
    Tensor encoded_grad_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor results_list[SIDES][KERNELS];   // [side][encoded, wte_grad, wpe_grad]

    Gpt2Encoder()
    {
        for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        {
            results_list[side_index_val][0] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
            results_list[side_index_val][1] = Tensor::zeros({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM}, DType::BFloat16);
            results_list[side_index_val][2] = Tensor::zeros({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
        }
    }

    /// Launches one kernel of one side into that side's results.
    ///
    ///   side_index_val     the side: 8 values or 4 values
    ///   kernel_index_val   forward, wte backward or wpe backward
    ///   blocks_val         the launch's blocks
    ///   threads_val        its block size
    void launch(int side_index_val, int kernel_index_val, int blocks_val, int threads_val) const
    {
        const Tensor *results_ptr = results_list[side_index_val];
        nv_bfloat16 *result_ptr = results_ptr[kernel_index_val].data_ptr<nv_bfloat16>();
        const int *ids_ptr = token_ids_tensor.data_ptr<int>();
        const nv_bfloat16 *grad_ptr = encoded_grad_tensor.data_ptr<nv_bfloat16>();
        if (side_index_val == 0 && kernel_index_val == 0)
            encoder_bf16_forward_kernel<<<blocks_val, threads_val>>>(result_ptr, ids_ptr, wte_tensor.data_ptr<nv_bfloat16>(),
                                                                     wpe_tensor.data_ptr<nv_bfloat16>(), GPT2_BATCH_SIZE,
                                                                     GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        else if (side_index_val == 0 && kernel_index_val == 1)
            encoder_bf16_backward_wte_kernel<<<blocks_val, threads_val>>>(result_ptr, grad_ptr, ids_ptr, GPT2_BATCH_SIZE,
                                                                          GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        else if (side_index_val == 0)
            encoder_bf16_backward_wpe_kernel<<<blocks_val, threads_val>>>(result_ptr, grad_ptr, GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN,
                                                                          GPT2_EMBED_DIM);
        else if (kernel_index_val == 0)
            encoder_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(result_ptr, ids_ptr, wte_tensor.data_ptr<nv_bfloat16>(),
                                                                             wpe_tensor.data_ptr<nv_bfloat16>(), GPT2_BATCH_SIZE,
                                                                             GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        else if (kernel_index_val == 1)
            encoder_backward_wte_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(result_ptr, grad_ptr, ids_ptr, GPT2_BATCH_SIZE,
                                                                                  GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        else
            encoder_backward_wpe_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(result_ptr, grad_ptr, GPT2_BATCH_SIZE,
                                                                                  GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    }

    /// Zeroes side `side_index_val`'s results and runs its 3 kernels once at <<<blocks_val, threads_val>>>.
    void run_once(int side_index_val, int blocks_val, int threads_val) const
    {
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
        {
            results_list[side_index_val][kernel_index_val].zero_();
            launch(side_index_val, kernel_index_val, blocks_val, threads_val);
        }
    }

    /// Tells if one side's results equal `usual_list`'s.
    ///
    ///   side_index_val   the side
    ///   usual_list       the default launch's results
    bool same_as(int side_index_val, const Tensor (&usual_list)[KERNELS]) const
    {
        // Tells if side `side_index_val`'s results equal `usual_list`'s: encoded and wpe_grad to the last bit, wte_grad
        // within BFLOAT16_TOLERANCE.
        const Tensor *results_ptr = results_list[side_index_val];
        return torch::equal(pytorch_copy_of(results_ptr[0]), pytorch_copy_of(usual_list[0])) &&
               relative_difference_on_gpu(results_ptr[1], usual_list[1]) <= BFLOAT16_TOLERANCE &&
               torch::equal(pytorch_copy_of(results_ptr[2]), pytorch_copy_of(usual_list[2]));
    }
};

// ══ part 1: 8 values a thread against 4 ═══════════════════════════════════════

static void time_8_values_against_4_values(void)
{
    // Prints each of the 3 kernels on GPT-2 small's sizes in bfloat16 (batch_size = 4, seq_len = 1024), here (8
    // values a thread, one 16-byte load) and in fp32/encoder.cuh (4 values, one 8-byte load), each at its own default
    // launch: ENCODER_BF16_BLOCKS x ENCODER_BF16_FORWARD_THREADS for the forward, x ENCODER_BF16_THREADS for the
    // backwards (fp32/encoder.cuh's are the same numbers). The 6 take turns over SWEEP_ROUNDS rounds, each the average
    // of its rounds, with the bytes it asks for, GB/s and MBU.
    //
    // Expect: the same results on both sides (encoded and wpe_grad to the last bit, wte_grad within
    // BFLOAT16_TOLERANCE); the 8-value kernels as fast or faster: the same bytes in half the loads. The wte backward
    // gains least: its time is mostly its atomic compare-and-swaps, 2 per load here, 1 per load there, the same per
    // byte.

    // 1. GPT-2 small's encoder; each side's results, checked against each other
    torch::manual_seed(0);
    Gpt2Encoder encoder_val;
    const int threads_list[KERNELS] = {ENCODER_BF16_FORWARD_THREADS, ENCODER_BF16_THREADS, ENCODER_BF16_THREADS};
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
        {
            encoder_val.results_list[side_index_val][kernel_index_val].zero_();
            encoder_val.launch(side_index_val, kernel_index_val, ENCODER_BF16_BLOCKS, threads_list[kernel_index_val]);
        }
    bool same_val = encoder_val.same_as(0, encoder_val.results_list[1]);
    warm_up_gpu([&] { encoder_val.launch(0, 0, ENCODER_BF16_BLOCKS, ENCODER_BF16_FORWARD_THREADS); });

    // 2. time the 6, taking turns over SWEEP_ROUNDS rounds
    float total_ms_list[SIDES][KERNELS] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
            for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
                total_ms_list[side_index_val][kernel_index_val] += gpu_milliseconds([&] {
                    encoder_val.launch(side_index_val, kernel_index_val, ENCODER_BF16_BLOCKS, threads_list[kernel_index_val]);
                });

    // 3. print each with its GB/s and MBU
    double bytes_list[KERNELS];
    kernel_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    const char *kernel_names_list[KERNELS] = {"forward", "wte backward", "wpe backward"};
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s; same results on both sides: %s\n", peak_val, same_val ? "yes" : "NO");
    printf("    kernel          MB      8 values: ms    GB/s   MBU     4 values: ms    GB/s   MBU     4 / 8\n");
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
    {
        float eight_ms_val = total_ms_list[0][kernel_index_val] / SWEEP_ROUNDS;
        float four_ms_val = total_ms_list[1][kernel_index_val] / SWEEP_ROUNDS;
        double eight_speed_val = gigabytes_per_second(bytes_list[kernel_index_val], eight_ms_val);
        double four_speed_val = gigabytes_per_second(bytes_list[kernel_index_val], four_ms_val);
        printf("    %-12s  %5.1f       %7.4f  %6.0f  %3.0f%%         %7.4f  %6.0f  %3.0f%%     %5.2fx\n",
               kernel_names_list[kernel_index_val], bytes_list[kernel_index_val] / MEGA, eight_ms_val, eight_speed_val,
               100.0 * eight_speed_val / peak_val, four_ms_val, four_speed_val, 100.0 * four_speed_val / peak_val,
               four_ms_val / eight_ms_val);
    }

    // 4. both sides gave the same results
    ASSERT_EQ(same_val, true);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of each of the 3 kernels here (8 values a thread) on GPT-2 small's sizes, for 10 block counts x
    // 4 thread counts, each the average of SWEEP_ROUNDS runs, with GB/s and MBU: fp32/encoder.cuh's 8 block counts,
    // then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe). The 40 launches take turns: SWEEP_ROUNDS
    // rounds, each timing every launch once, so every launch gets runs from early, middle and late alike. The
    // defaults (ENCODER_BF16_BLOCKS = 512 and the block sizes) are fp32/encoder.cuh's float4 sweep's; with 8 values a
    // thread there is half the work to share out, so the best launch may be another.
    //
    // Expect: every launch gives the same results as the default one (the grid-stride loop covers every value
    // whatever the count): encoded and wpe_grad to the last bit, wte_grad within BFLOAT16_TOLERANCE. Only the time
    // changes.

    // 1. GPT-2 small's encoder; the default launches' results
    torch::manual_seed(0);
    Gpt2Encoder encoder_val;
    encoder_val.launch(0, 0, ENCODER_BF16_BLOCKS, ENCODER_BF16_FORWARD_THREADS);
    encoder_val.launch(0, 1, ENCODER_BF16_BLOCKS, ENCODER_BF16_THREADS);
    encoder_val.launch(0, 2, ENCODER_BF16_BLOCKS, ENCODER_BF16_THREADS);
    Tensor usual_list[KERNELS];
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
        usual_list[kernel_index_val] = encoder_val.results_list[0][kernel_index_val].clone();
    warm_up_gpu([&] { encoder_val.launch(0, 0, ENCODER_BF16_BLOCKS, ENCODER_BF16_FORWARD_THREADS); });

    // 2. time every launch, taking turns over SWEEP_ROUNDS rounds; launch k is block_counts[k / 4] x thread_counts[k % 4]
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 10;
    const int thread_count_choices_val = 4;
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    int block_counts_list[block_count_choices_val] = {40, 80, 160, 320, 512, 1024, 2048, 4096, sms_val, 2 * sms_val};
    int thread_counts_list[thread_count_choices_val] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                        16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    float total_ms_list[launches_val][KERNELS] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
                total_ms_list[launch_index_val][kernel_index_val] += gpu_milliseconds([&] {
                    encoder_val.launch(0, kernel_index_val, block_counts_list[launch_index_val / thread_count_choices_val],
                                       thread_counts_list[launch_index_val % thread_count_choices_val]);
                });

    // 3. print each launch's averages with GB/s, and check its results against the default one's
    double bytes_list[KERNELS];
    kernel_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    const char *kernel_names_list[KERNELS] = {"forward", "wte backward", "wpe backward"};
    float fastest_ms_list[KERNELS] = {1e9f, 1e9f, 1e9f};
    int fastest_launch_list[KERNELS] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s\n", peak_val);
    printf("    blocks   threads   forward: ms   GB/s   wte backward: ms   GB/s   wpe backward: ms   GB/s   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        int blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        int threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        float times_list[KERNELS];
        for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
        {
            times_list[kernel_index_val] = total_ms_list[launch_index_val][kernel_index_val] / SWEEP_ROUNDS;
            if (times_list[kernel_index_val] < fastest_ms_list[kernel_index_val])
            {
                fastest_ms_list[kernel_index_val] = times_list[kernel_index_val];
                fastest_launch_list[kernel_index_val] = launch_index_val;
            }
        }
        encoder_val.run_once(0, blocks_val, threads_val);
        bool same_val = encoder_val.same_as(0, usual_list);
        if (same_val)
            launches_with_the_same_result_val++;
        const char *usual_ptr = blocks_val == ENCODER_BF16_BLOCKS && threads_val == ENCODER_BF16_FORWARD_THREADS
                                    ? "   <- the forward's default"
                                : blocks_val == ENCODER_BF16_BLOCKS && threads_val == ENCODER_BF16_THREADS
                                    ? "   <- the backwards' default"
                                    : "";
        printf("    %6d   %7d   %11.4f   %4.0f   %16.4f   %4.0f   %16.4f   %4.0f   %-11s%s\n", blocks_val, threads_val,
               times_list[0], gigabytes_per_second(bytes_list[0], times_list[0]), times_list[1],
               gigabytes_per_second(bytes_list[1], times_list[1]), times_list[2],
               gigabytes_per_second(bytes_list[2], times_list[2]), same_val ? "yes" : "NO", usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
    {
        int launch_index_val = fastest_launch_list[kernel_index_val];
        double speed_val = gigabytes_per_second(bytes_list[kernel_index_val], fastest_ms_list[kernel_index_val]);
        printf("    fastest %-13s %4d blocks x %4d threads: %.4f ms, %.0f GB/s, MBU %.0f%%\n", kernel_names_list[kernel_index_val],
               block_counts_list[launch_index_val / thread_count_choices_val],
               thread_counts_list[launch_index_val % thread_count_choices_val], fastest_ms_list[kernel_index_val], speed_val,
               100.0 * speed_val / peak_val);
    }

    // 4. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and cuda/tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8_values_against_4_values();
    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
