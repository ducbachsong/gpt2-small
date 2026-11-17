// tune_layer_norm_8bit.cu — the timings of llmc/kernels/8bit/layer_norm_8bit.cuh, apart from
// tests/8bit/test_layer_norm_8bit.cu so the tests only check numbers: time_8bit_forward_against_bf16_and_round_8bit
// (the copy made inside LayerNorm against LayerNorm then round_8bit), time_forward_with_other_thread_counts. With the
// bytes each moves, its GB/s and MBU. The backward is layer_norm_bf16's: tests/bf16/tune_layer_norm_bf16.cu.
//
// Built as tests/8bit/test_layer_norm_8bit.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/layer_norm_8bit_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../llmc/kernels/8bit/layer_norm_8bit.cuh"

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_ROWS (GPT2_BATCH_SIZE * GPT2_SEQ_LEN)
#define GPT2_EMBED_DIM 768
#define GPT2_LAYER_NORMS 25           // LayerNorms in a GPT-2 small forward: 2 a block, 12 blocks, and lnf
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

/// GPT-2 small's LayerNorm (4096 random rows of 768) with the 8-bit copy's tensors, and a second copy for
/// round_8bit's to land in.
struct Gpt2LayerNorm
{
    Tensor input_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor weight_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor bias_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor output_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
    Tensor rstd_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
    Tensor output_8bit_tensor = Tensor::zeros({GPT2_ROWS, GPT2_EMBED_DIM}, DType::E4M3);
    Tensor output_scales_tensor = Tensor::zeros({GPT2_ROWS});
    Tensor rounded_8bit_tensor = Tensor::zeros({GPT2_ROWS, GPT2_EMBED_DIM}, DType::E4M3);
    Tensor rounded_scales_tensor = Tensor::zeros({GPT2_ROWS});

    /// Launches layer_norm_8bit_forward_kernel with `threads_val` threads a block, one warp a row, the blocks rounded up.
    void forward_8bit(int threads_val) const
    {
        int blocks_val = (int)CEIL_DIV((size_t)GPT2_ROWS, (size_t)(threads_val / HARDWARE_THREADS_PER_WARP));
        layer_norm_8bit_forward_kernel<<<blocks_val, threads_val>>>(
            output_tensor.data_ptr<nv_bfloat16>(), reinterpret_cast<__nv_fp8_e4m3 *>(output_8bit_tensor.data_ptr()),
            output_scales_tensor.data(), mean_tensor.data(), rstd_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(),
            weight_tensor.data_ptr<nv_bfloat16>(), bias_tensor.data_ptr<nv_bfloat16>(), GPT2_ROWS, GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    }
};

/// Gives the bytes asked for, in run order: the 8-bit forward, layer_norm_bf16_forward, round_8bit of the output.
/// A row's values read again in a later pass of the same kernel (from the cache) and weight and bias are left out.
///
///     8-bit forward      reads input; writes output, its E4M3 copy, scales, mean, rstd       15.8 MB
///     bfloat16 forward   reads input; writes output, mean, rstd                              12.6 MB
///     round_8bit         reads output; writes the E4M3 copy, scales                           9.5 MB
static void run_bytes(double (&bytes_list)[3])
{
    double values_val = (double)GPT2_ROWS * GPT2_EMBED_DIM;
    double row_floats_bytes_val = (double)GPT2_ROWS * sizeof(float);
    bytes_list[0] = values_val * (2 * sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + 3 * row_floats_bytes_val;
    bytes_list[1] = values_val * 2 * sizeof(nv_bfloat16) + 2 * row_floats_bytes_val;
    bytes_list[2] = values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + row_floats_bytes_val;
}

// ══ part 1: the copy made inside LayerNorm against LayerNorm then round_8bit ═══

static void time_8bit_forward_against_bf16_and_round_8bit(void)
{
    // Prints, on GPT-2 small's sizes (4096 rows of 768), each the average of SWEEP_ROUNDS runs taking turns:
    // layer_norm_8bit_forward, layer_norm_bf16_forward, and round_8bit of its output, with the bytes each asks for,
    // GB/s and MBU; then the two ways to the same bfloat16 output and E4M3 copy, and what the difference is over a
    // forward's GPT2_LAYER_NORMS LayerNorms.
    //
    // Expect: the 8-bit forward a little slower than the bfloat16 one alone (3.1 MB more written, and the output read
    // back, most likely from the L2 cache), and faster than the bfloat16 one plus round_8bit (which reads the 6.3 MB
    // output again from GPU memory, with a launch of its own). The two copies the same bits.

    // 1. GPT-2 small's LayerNorm; both ways once, checked
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    layer_norm_val.forward_8bit(LAYER_NORM_8BIT_THREADS_PER_BLOCK);
    round_8bit(layer_norm_val.rounded_8bit_tensor, layer_norm_val.rounded_scales_tensor, layer_norm_val.output_tensor);
    bool same_val = layer_norm_val.output_8bit_tensor.equal(layer_norm_val.rounded_8bit_tensor) &&
                    layer_norm_val.output_scales_tensor.tolist() == layer_norm_val.rounded_scales_tensor.tolist();

    // 2. the 3 runs, taking turns over SWEEP_ROUNDS rounds
    const int runs_val = 3;
    auto run_val = [&](int run_index_val) {
        const Gpt2LayerNorm &l = layer_norm_val;
        if (run_index_val == 0)
            l.forward_8bit(LAYER_NORM_8BIT_THREADS_PER_BLOCK);
        else if (run_index_val == 1)
            layer_norm_bf16_forward(l.output_tensor, l.mean_tensor, l.rstd_tensor, l.input_tensor, l.weight_tensor, l.bias_tensor);
        else
            round_8bit(l.rounded_8bit_tensor, l.rounded_scales_tensor, l.output_tensor);
    };
    warm_up_gpu([&] { run_val(0); });
    float total_ms_list[runs_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
            total_ms_list[run_index_val] += gpu_milliseconds([&] { run_val(run_index_val); });

    // 3. print each with its GB/s and MBU, then the two ways
    double bytes_list[runs_val];
    run_bytes(bytes_list);
    const char *names_list[runs_val] = {"layer_norm_8bit_forward", "layer_norm_bf16_forward", "round_8bit of the output"};
    float ms_list[runs_val];
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s; the two copies the same bits: %s\n", peak_val, same_val ? "yes" : "NO");
    printf("    run                          MB        ms     GB/s    MBU\n");
    for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
    {
        ms_list[run_index_val] = total_ms_list[run_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_list[run_index_val], ms_list[run_index_val]);
        printf("    %-24s   %6.1f   %7.4f   %6.0f   %3.0f%%\n", names_list[run_index_val], bytes_list[run_index_val] / MEGA,
               ms_list[run_index_val], speed_val, 100.0 * speed_val / peak_val);
    }
    float apart_ms_val = ms_list[1] + ms_list[2];
    printf("    output and copy:   in one kernel %.4f ms   LayerNorm then round_8bit %.4f ms   over %d LayerNorms: %.3f ms saved\n",
           ms_list[0], apart_ms_val, GPT2_LAYER_NORMS, GPT2_LAYER_NORMS * (apart_ms_val - ms_list[0]));

    // 4. check
    ASSERT_EQ(same_val, true);
}

// ══ part 2: other thread counts ═══════════════════════════════════════════════

static void time_forward_with_other_thread_counts(void)
{
    // Prints layer_norm_8bit_forward_kernel on GPT-2 small's sizes with 1, 2, 4, 8 and 16 warps a block (one warp a
    // row, the blocks rounded up: 4096 .. 256), each the average of SWEEP_ROUNDS runs taking turns, with GB/s and MBU.
    // Not 1024 threads: fp32/layer_norm.cuh's kernels allow at most 896, and this one keeps more.
    //
    // Expect: every launch gives the default's output and copy to the last bit (each row one warp's). Only the time
    // changes.

    // 1. GPT-2 small's LayerNorm; the default's results
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    layer_norm_val.forward_8bit(LAYER_NORM_8BIT_THREADS_PER_BLOCK);
    Tensor usual_copy_tensor = layer_norm_val.output_8bit_tensor.clone();
    std::vector<float> usual_scales_list = layer_norm_val.output_scales_tensor.tolist();
    warm_up_gpu([&] { layer_norm_val.forward_8bit(LAYER_NORM_8BIT_THREADS_PER_BLOCK); });

    // 2. time every launch, taking turns over SWEEP_ROUNDS rounds
    const int launches_val = 5;
    int threads_list[launches_val] = {HARDWARE_THREADS_PER_WARP, 2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP,
                                      8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP};
    float total_ms_list[launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            total_ms_list[launch_index_val] += gpu_milliseconds([&] { layer_norm_val.forward_8bit(threads_list[launch_index_val]); });

    // 3. print each with GB/s and MBU, and check its results against the default's
    double bytes_list[3];
    run_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB asked for; this GPU's peak %.0f GB/s\n", bytes_list[0] / MEGA, peak_val);
    printf("    threads   blocks        ms     GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        int threads_val = threads_list[launch_index_val];
        layer_norm_val.output_8bit_tensor.zero_();
        layer_norm_val.output_scales_tensor.zero_();
        layer_norm_val.forward_8bit(threads_val);
        bool same_val = layer_norm_val.output_8bit_tensor.equal(usual_copy_tensor) &&
                        layer_norm_val.output_scales_tensor.tolist() == usual_scales_list;
        if (same_val)
            launches_with_the_same_result_val++;
        float ms_val = total_ms_list[launch_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_list[0], ms_val);
        printf("    %7d   %6d   %7.4f   %6.0f   %3.0f%%   %-11s%s\n", threads_val,
               (int)CEIL_DIV((size_t)GPT2_ROWS, (size_t)(threads_val / HARDWARE_THREADS_PER_WARP)), ms_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO",
               threads_val == LAYER_NORM_8BIT_THREADS_PER_BLOCK ? "   <- the default" : "");
    }

    // 4. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8bit_forward_against_bf16_and_round_8bit();
    time_forward_with_other_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
