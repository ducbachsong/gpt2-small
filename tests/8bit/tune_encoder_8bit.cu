// tune_encoder_8bit.cu — the timings of llmc/kernels/8bit/encoder_8bit.cuh, apart from tests/8bit/test_encoder_8bit.cu
// so the tests only check numbers: time_8bit_forward_against_bfloat16 (the rounding and the gather apart, and what
// the step pays with the head's rounding counted), time_gather_with_other_block_and_thread_counts. With the bytes
// each moves, its GB/s and MBU. The backward is encoder_bf16's: tests/bf16/tune_encoder_bf16.cu.
//
// Built as tests/8bit/test_encoder_8bit.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/encoder_8bit_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include "../test_helpers.h"    // ASSERT_EQ
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../llmc/kernels/8bit/encoder_8bit.cuh"

// encoded against the bfloat16 encoder's: each E4M3 is off by up to 2^-4 (6%) of its value. The largest difference,
// as a fraction of the largest value.
static const double EIGHT_BIT_TOLERANCE = 1e-1;

#define GPT2_HIGHEST_TOKEN_ID 50256   // GPT-2's real ids: 0 .. 50256
#define GPT2_PADDED_VOCAB_SIZE 50304  // GPT-2 small's sizes
#define GPT2_MAX_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_BATCH_SIZE 4
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes an int tensor {GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN} of random token ids from 0 .. GPT2_HIGHEST_TOKEN_ID on the
/// GPU. The same seed gives the same ids every run.
static Tensor random_token_ids(void)
{
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

/// GPT-2 small's encoder in bfloat16, wte's 8-bit copy, and the results of the 8-bit and the bfloat16 forward.
struct Gpt2Encoder
{
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor token_ids_tensor = random_token_ids();
    Tensor wte_8bit_tensor = Tensor::zeros({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM}, DType::E4M3);
    Tensor wte_scales_tensor = Tensor::zeros({GPT2_PADDED_VOCAB_SIZE});
    Tensor encoded_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor bf16_encoded_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);

    /// Launches the gather alone, encoder_8bit_forward_kernel <<<blocks_val, threads_val>>>, on the copy as it is.
    void gather(int blocks_val, int threads_val) const
    {
        encoder_8bit_forward_kernel<<<blocks_val, threads_val>>>(
            encoded_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(),
            reinterpret_cast<const __nv_fp8_e4m3 *>(wte_8bit_tensor.data_ptr()), wte_scales_tensor.data(),
            wpe_tensor.data_ptr<nv_bfloat16>(), GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    }

    /// Gives the largest difference between encoded and the bfloat16 encoder's, as a fraction of its largest value.
    double difference_from_bfloat16(void) const
    {
        torch::Tensor bf16_tensor = pytorch_copy_of(bf16_encoded_tensor);
        return ((pytorch_copy_of(encoded_tensor) - bf16_tensor).abs().max() / bf16_tensor.abs().max()).item<double>();
    }
};

/// Gives the bytes the rounding of all of wte moves: reads wte (2 bytes a value), writes the copy (1 byte a value)
/// and a scale a row. GPT-2 small: 77.3 + 38.6 + 0.2 = 116.1 MB.
static double rounding_bytes(void)
{
    double values_val = (double)GPT2_PADDED_VOCAB_SIZE * GPT2_EMBED_DIM;
    return values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + (double)GPT2_PADDED_VOCAB_SIZE * sizeof(float);
}

/// Gives the bytes the gather asks for: the ids, 4096 rows of the copy (1 byte a value) and their scales, 4096 rows of
/// wpe (asked for once per row of the batch, mostly from the L2 cache), and encoded written. GPT-2 small: 15.8 MB.
static double gather_bytes(void)
{
    double rows_val = (double)GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN;
    return rows_val * (sizeof(int) + sizeof(float)) +
           rows_val * GPT2_EMBED_DIM * (sizeof(__nv_fp8_e4m3) + 2 * sizeof(nv_bfloat16));
}

/// Gives the bytes the bfloat16 forward asks for: the ids, 4096 rows of wte and of wpe, and encoded written. GPT-2
/// small: 18.9 MB.
static double bfloat16_forward_bytes(void)
{
    double rows_val = (double)GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN;
    return rows_val * sizeof(int) + rows_val * GPT2_EMBED_DIM * 3 * sizeof(nv_bfloat16);
}

// ══ part 1: the 8-bit forward against the bfloat16 one ════════════════════════

static void time_8bit_forward_against_bfloat16(void)
{
    // Prints, on GPT-2 small's sizes (batch_size = 4, seq_len = 1024, wte {50304, 768}), each the average of
    // SWEEP_ROUNDS runs taking turns: encoder_8bit_forward (the rounding, then the gather), the rounding alone
    // (round_8bit on all of wte), the gather alone (on a copy already made), and encoder_bf16_forward, with the bytes
    // each moves, GB/s and MBU. Then the two ways a step can go, with the head's products in 8 bits either way:
    //
    //     bfloat16 encoder   encoder_bf16_forward, and the head rounds wte itself       = bfloat16 forward + rounding
    //     8-bit encoder      encoder_8bit_forward, and the head takes its copy           = rounding + gather
    //
    // Expect: the 8-bit forward far slower than the bfloat16 one alone (the rounding moves about 116 MB, against
    // 19 MB), the gather a little faster than the bfloat16 forward (16 MB against 19), so the 8-bit way of the step a
    // little ahead. encoded within EIGHT_BIT_TOLERANCE of the bfloat16 encoder's.

    // 1. GPT-2 small's encoder; one forward each way, checked
    torch::manual_seed(0);
    Gpt2Encoder encoder_val;
    encoder_8bit_forward(encoder_val.encoded_tensor, encoder_val.wte_8bit_tensor, encoder_val.wte_scales_tensor,
                         encoder_val.token_ids_tensor, encoder_val.wte_tensor, encoder_val.wpe_tensor);
    encoder_bf16_forward(encoder_val.bf16_encoded_tensor, encoder_val.token_ids_tensor, encoder_val.wte_tensor, encoder_val.wpe_tensor);
    double difference_val = encoder_val.difference_from_bfloat16();

    // 2. the 4 runs, taking turns over SWEEP_ROUNDS rounds
    const int runs_val = 4;
    auto run_val = [&](int run_index_val) {
        if (run_index_val == 0)
            encoder_8bit_forward(encoder_val.encoded_tensor, encoder_val.wte_8bit_tensor, encoder_val.wte_scales_tensor,
                                 encoder_val.token_ids_tensor, encoder_val.wte_tensor, encoder_val.wpe_tensor);
        else if (run_index_val == 1)
            round_8bit(encoder_val.wte_8bit_tensor, encoder_val.wte_scales_tensor, encoder_val.wte_tensor);
        else if (run_index_val == 2)
            encoder_val.gather(ENCODER_8BIT_BLOCKS, ENCODER_8BIT_FORWARD_THREADS);
        else
            encoder_bf16_forward(encoder_val.bf16_encoded_tensor, encoder_val.token_ids_tensor, encoder_val.wte_tensor,
                                 encoder_val.wpe_tensor);
    };
    warm_up_gpu([&] { run_val(0); });
    float total_ms_list[runs_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
            total_ms_list[run_index_val] += gpu_milliseconds([&] { run_val(run_index_val); });

    // 3. print each with its GB/s and MBU, then the two ways of the step
    double bytes_list[runs_val] = {rounding_bytes() + gather_bytes(), rounding_bytes(), gather_bytes(), bfloat16_forward_bytes()};
    const char *names_list[runs_val] = {"encoder_8bit_forward", "  the rounding", "  the gather", "encoder_bf16_forward"};
    float ms_list[runs_val];
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s; encoded's largest difference from the bfloat16 encoder: %.2e of its largest value\n",
           peak_val, difference_val);
    printf("    run                        MB        ms     GB/s    MBU\n");
    for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
    {
        ms_list[run_index_val] = total_ms_list[run_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_list[run_index_val], ms_list[run_index_val]);
        printf("    %-22s   %6.1f   %7.4f   %6.0f   %3.0f%%\n", names_list[run_index_val], bytes_list[run_index_val] / MEGA,
               ms_list[run_index_val], speed_val, 100.0 * speed_val / peak_val);
    }
    printf("    the step's encoder and the head's rounding of wte:   bfloat16 encoder %.4f ms   8-bit encoder %.4f ms\n",
           ms_list[3] + ms_list[1], ms_list[0]);

    // 4. check
    ASSERT_EQ(difference_val <= EIGHT_BIT_TOLERANCE, true);
}

// ══ part 2: the gather with other block and thread counts ═════════════════════

static void time_gather_with_other_block_and_thread_counts(void)
{
    // Prints the time of the gather (encoder_8bit_forward_kernel) on GPT-2 small's sizes, for 10 block counts x 4
    // thread counts, each the average of SWEEP_ROUNDS runs taking turns, with GB/s and MBU: fp32/encoder.cuh's 8 block
    // counts, then this GPU's SM count x 1 and x 2. The rounding is round_8bit's, swept in tests/8bit/tune_round_8bit.cu.
    //
    // Expect: every launch gives the same encoded as the default one, to the last bit (each value worked out alone).
    // Only the time changes.

    // 1. GPT-2 small's encoder, its copy made once; the default launch's result
    torch::manual_seed(0);
    Gpt2Encoder encoder_val;
    round_8bit(encoder_val.wte_8bit_tensor, encoder_val.wte_scales_tensor, encoder_val.wte_tensor);
    encoder_val.gather(ENCODER_8BIT_BLOCKS, ENCODER_8BIT_FORWARD_THREADS);
    torch::Tensor usual_tensor = pytorch_copy_of(encoder_val.encoded_tensor);
    warm_up_gpu([&] { encoder_val.gather(ENCODER_8BIT_BLOCKS, ENCODER_8BIT_FORWARD_THREADS); });

    // 2. time every launch, taking turns over SWEEP_ROUNDS rounds; launch k is block_counts[k / 4] x thread_counts[k % 4]
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 10;
    const int thread_count_choices_val = 4;
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    int block_counts_list[block_count_choices_val] = {40, 80, 160, 320, 512, 1024, 2048, 4096, sms_val, 2 * sms_val};
    int thread_counts_list[thread_count_choices_val] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                        16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    float total_ms_list[launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            total_ms_list[launch_index_val] += gpu_milliseconds([&] {
                encoder_val.gather(block_counts_list[launch_index_val / thread_count_choices_val],
                                   thread_counts_list[launch_index_val % thread_count_choices_val]);
            });

    // 3. print each launch's average with GB/s and MBU, and check its result against the default one's
    double bytes_val = gather_bytes();
    double peak_val = peak_gigabytes_per_second();
    float fastest_ms_val = 1e9f;
    int fastest_launch_val = 0;
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB asked for; this GPU's peak %.0f GB/s\n", bytes_val / MEGA, peak_val);
    printf("    blocks   threads        ms     GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        int blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        int threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        float ms_val = total_ms_list[launch_index_val] / SWEEP_ROUNDS;
        if (ms_val < fastest_ms_val)
        {
            fastest_ms_val = ms_val;
            fastest_launch_val = launch_index_val;
        }
        encoder_val.encoded_tensor.zero_();
        encoder_val.gather(blocks_val, threads_val);
        bool same_val = torch::equal(pytorch_copy_of(encoder_val.encoded_tensor), usual_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        double speed_val = gigabytes_per_second(bytes_val, ms_val);
        printf("    %6d   %7d   %7.4f   %6.0f   %3.0f%%   %-11s%s\n", blocks_val, threads_val, ms_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO",
               blocks_val == ENCODER_8BIT_BLOCKS && threads_val == ENCODER_8BIT_FORWARD_THREADS ? "   <- the default" : "");
    }
    double fastest_speed_val = gigabytes_per_second(bytes_val, fastest_ms_val);
    printf("    fastest gather  %4d blocks x %4d threads: %.4f ms, %.0f GB/s, MBU %.0f%%\n",
           block_counts_list[fastest_launch_val / thread_count_choices_val],
           thread_counts_list[fastest_launch_val % thread_count_choices_val], fastest_ms_val, fastest_speed_val,
           100.0 * fastest_speed_val / peak_val);

    // 4. every launch gave the same result
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8bit_forward_against_bfloat16();
    time_gather_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
