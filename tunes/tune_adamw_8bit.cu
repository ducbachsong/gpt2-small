// tune_adamw_8bit.cu — the timings of tests/test_adamw_8bit.cu, moved here so the tests only check numbers:
// time_the_three_adamws_on_gpt2_small_size, time_the_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_adamw_8bit.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/adamw_8bit_<date>_<time>.txt.
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/types.h>              // torch::randn, torch::from_blob
#include <torch/utils.h>              // torch::manual_seed, torch::NoGradGuard
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/optimizer.cuh"            // AdamWConfig; adamw_update and adamw_update_bf16, timed against
#include "../llmc/kernels/adamw_8bit.cuh"   // adamw_update_8bit

#define GPT2_PARAMS 124439808         // GPT-2 small's values
#define GPT2_DECAYED_PARAMS 124439040 // of them in matrices

// Bytes each AdamW moves per value, for its memory bandwidth (GB/s) and MBU (GB/s over the GPU's peak). The
// 8-bit scales add 2 floats read and written per ADAMW_VALUES_PER_SCALE values, 1/8 byte a value: left out.
#define ADAMW_FLOAT_BYTES_PER_VALUE 28 // adamw_update:      reads 4+4+4+4 (p, g, m, v), writes 4+4+4 (p, m, v)
#define ADAMW_BF16_BYTES_PER_VALUE 20  // adamw_update_bf16: reads 4+2+2+2 (master, g, m, v), writes 4+2+2+2 (master, p, m, v)
#define ADAMW_8BIT_BYTES_PER_VALUE 16  // adamw_update_8bit: reads 4+2+1+1 (master, g, m, sqrt v), writes 4+2+1+1

// ── helpers: 8-bit m and v ────────────────────────────────────────────────────

/// 8-bit m and sqrt(v) for `count` values, all 0 (m = v = 0): 1 byte per value, and one float scale per
/// ADAMW_VALUES_PER_SCALE values. Tensor has no 1-byte dtype, so an Int32 tensor of count / FP8_PER_INT holds the
/// bytes, FP8_PER_INT to an int.
struct EightBitMoments
{
    Tensor m_tensor;
    Tensor m_scales_tensor;
    Tensor sqrt_v_tensor;
    Tensor sqrt_v_scales_tensor;
    explicit EightBitMoments(size_t count_val)
        : m_tensor(Tensor::zeros({count_val / FP8_PER_INT}, DType::Int32)),
          m_scales_tensor(Tensor::zeros({CEIL_DIV(count_val, (size_t)ADAMW_VALUES_PER_SCALE)})),
          sqrt_v_tensor(Tensor::zeros({count_val / FP8_PER_INT}, DType::Int32)),
          sqrt_v_scales_tensor(Tensor::zeros({CEIL_DIV(count_val, (size_t)ADAMW_VALUES_PER_SCALE)}))
    {
    }
    __nv_fp8_e4m3 *m_bytes() const { return (__nv_fp8_e4m3 *)m_tensor.data_ptr(); }
    __nv_fp8_e4m3 *sqrt_v_bytes() const { return (__nv_fp8_e4m3 *)sqrt_v_tensor.data_ptr(); }
};

// ── helpers: my side ──────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation `scale_val`, made by PyTorch
/// on the GPU: Tensor::randn makes them on the CPU, which takes minutes for GPT-2 small's 124M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list, float scale_val)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * scale_val;
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

// ── helpers: PyTorch's side ───────────────────────────────────────────────────

/// Makes a PyTorch copy of a float tensor of mine, on the GPU: to keep a result and compare a later one
/// with it there, without moving 124M values to the CPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

// ══ part 1: the three AdamWs on GPT-2 small's size ════════════════════════════

static void time_the_three_adamws_on_gpt2_small_size(void)
{
    // Prints the time of one update on as many values as GPT-2 small has, 124,439,808 (the matrices' 124,439,040
    // decayed, then 768), taking turns over 20 rounds after a second of warm-up. Bytes per value, and the least
    // time at an H100 PCIe's 2.0 TB/s:
    //
    //     adamw_update        all float                                   reads 4+4+4+4, writes 4+4+4   28   1.74 ms
    //     adamw_update_bf16   bfloat16, float master, bfloat16 m and v   reads 4+2+2+2, writes 4+2+2+2 20   1.24 ms
    //     adamw_update_8bit   bfloat16, float master, 8-bit m and v      reads 4+2+1+1, writes 4+2+1+1 16   1.00 ms

    // 1. GPT-2 small's number of values in float; bfloat16 params and grads, each with a float master of its own
    torch::manual_seed(0);
    AdamWConfig config_val;
    config_val.max_norm = 1.0;
    size_t count_val = GPT2_PARAMS;
    size_t decayed_count_val = GPT2_DECAYED_PARAMS;
    Tensor params_float_tensor = random_tensor_on_gpu({count_val}, 0.02f);
    Tensor grads_float_tensor = random_tensor_on_gpu({count_val}, 1.0f);
    Tensor m_tensor = Tensor::zeros({count_val});
    Tensor v_tensor = Tensor::zeros({count_val});
    Tensor grad_norm_tensor = Tensor::empty({1});
    float norm_val = pytorch_copy_of(grads_float_tensor).norm().item<float>(); // about sqrt(124M) = 11,155: clipped
    grad_norm_tensor.copy_from_cpu(&norm_val);
    Tensor grads_tensor = grads_float_tensor.to(DType::BFloat16);
    Tensor master_bf16_tensor = params_float_tensor.clone();
    Tensor master_8bit_tensor = params_float_tensor.clone();
    Tensor params_bf16_tensor = params_float_tensor.to(DType::BFloat16);
    Tensor params_8bit_tensor = params_bf16_tensor.clone();
    Tensor m_bf16_tensor = Tensor::zeros({count_val}, DType::BFloat16);
    Tensor v_bf16_tensor = Tensor::zeros({count_val}, DType::BFloat16);
    EightBitMoments moments_val(count_val);

    // 2. the three updates, at step t
    int t_val = 1;
    auto update_float_val = [&] {
        adamw_update(params_float_tensor.data(), grads_float_tensor.data(), m_tensor.data(), v_tensor.data(), count_val,
                     decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate,
                     config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay, t_val);
    };
    auto update_bf16_val = [&] {
        adamw_update_bf16(params_bf16_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                          m_bf16_tensor.data_ptr<nv_bfloat16>(), v_bf16_tensor.data_ptr<nv_bfloat16>(), count_val,
                          decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate,
                          config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay, t_val,
                          master_bf16_tensor.data());
    };
    auto update_8bit_val = [&] {
        adamw_update_8bit(params_8bit_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                          moments_val.m_bytes(), moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(),
                          moments_val.sqrt_v_scales_tensor.data(), count_val, decayed_count_val, grad_norm_tensor.data(),
                          config_val.max_norm, config_val.learning_rate, config_val.beta1, config_val.beta2, config_val.eps,
                          config_val.weight_decay, t_val, master_8bit_tensor.data());
    };
    warm_up_gpu(update_float_val);

    // 3. time them, taking turns
    double float_total_val = 0.0;
    double bf16_total_val = 0.0;
    double eight_bit_total_val = 0.0;
    for (t_val = 1; t_val <= TIMED_STEPS; t_val++)
    {
        float_total_val += gpu_milliseconds(update_float_val);
        bf16_total_val += gpu_milliseconds(update_bf16_val);
        eight_bit_total_val += gpu_milliseconds(update_8bit_val);
    }

    // 4. print each one's average time, its GB/s and its MBU (GB/s over this GPU's peak)
    double peak_val = peak_gigabytes_per_second();
    const char *names_list[] = {"float", "bfloat16", "8-bit m and v"};
    double totals_list[] = {float_total_val, bf16_total_val, eight_bit_total_val};
    int bytes_per_value_list[] = {ADAMW_FLOAT_BYTES_PER_VALUE, ADAMW_BF16_BYTES_PER_VALUE, ADAMW_8BIT_BYTES_PER_VALUE};
    print_test_name(__func__);
    printf("    average of %d updates; this GPU's peak memory bandwidth %.0f GB/s\n", TIMED_STEPS, peak_val);
    printf("    AdamW            bytes a value   time ms     GB/s    MBU\n");
    for (int k_val = 0; k_val < 3; k_val++)
    {
        double milliseconds_val = totals_list[k_val] / TIMED_STEPS;
        double speed_val = gigabytes_per_second((double)count_val * bytes_per_value_list[k_val], milliseconds_val);
        printf("    %-16s %13d   %7.3f   %6.0f   %3.0f%%\n", names_list[k_val], bytes_per_value_list[k_val],
               milliseconds_val, speed_val, 100.0 * speed_val / peak_val);
    }
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

/// Makes a PyTorch copy of an int tensor of mine (8-bit m or sqrt(v), FP8_PER_INT bytes to an int), on the GPU,
/// to compare its bytes with a later result there.
static torch::Tensor pytorch_bytes_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    return torch::from_blob(tensor.data_ptr(), {(int64_t)tensor.numel()}, on_gpu_val).clone();
}

static void time_the_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of one update on as many values as GPT-2 small has, 124,439,808 (the matrices'
    // 124,439,040 decayed, then 768), as training would run it: bfloat16 params and grads, a float master, m
    // and sqrt(v) in 8 bits. 9 block counts x 4 thread counts (all multiples of THREADS_PER_WARP, as the kernel needs), each
    // the average of 20 runs: the 7 picked on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an
    // H100 PCIe). The 36 launches take turns over 20 rounds, so each gets runs from early, middle and late
    // alike. The kernel is launched directly, <<<blocks, threads>>>; adamw_update_8bit launches one float4 per
    // thread, 121,524 x 256. Each update reads master, g, m, sqrt(v) and writes master, p, m, sqrt(v):
    // 4 + 2 + 1 + 1 and 4 + 2 + 1 + 1 = 16 bytes per value, 2.0 GB: at a T4's 320 GB/s at least 6.2 ms, at an
    // H100 PCIe's 2.0 TB/s 1.00 ms.
    //
    // Each run changes the master, p, m, sqrt(v) and the scales, so before each check they are put back: the
    // master and p from kept copies, the rest to 0. Expect: every launch gives the same master, p, bytes and
    // scales as the usual one, to the last bit: a warp always takes THREADS_PER_WARP float4s in a row, one scale, whatever
    // the launch.

    // 1. GPT-2 small's number of values: a random float master and gradients, the model's in bfloat16
    torch::manual_seed(0);
    AdamWConfig config_val;
    config_val.max_norm = 1.0;
    size_t params_in_slice_val = GPT2_PARAMS;
    size_t decayed_count_val = GPT2_DECAYED_PARAMS;
    Tensor master_tensor = random_tensor_on_gpu({params_in_slice_val}, 0.02f);
    Tensor grads_float_tensor = random_tensor_on_gpu({params_in_slice_val}, 1.0f);
    Tensor start_master_tensor = master_tensor.clone();
    Tensor params_tensor = master_tensor.to(DType::BFloat16);
    Tensor grads_tensor = grads_float_tensor.to(DType::BFloat16);
    Tensor start_params_tensor = params_tensor.clone();
    EightBitMoments moments_val(params_in_slice_val);
    Tensor grad_norm_tensor = Tensor::empty({1});
    float norm_val = pytorch_copy_of(grads_float_tensor).norm().item<float>(); // about sqrt(124M) = 11,155: clipped
    grad_norm_tensor.copy_from_cpu(&norm_val);

    // 2. step t = 1's numbers, worked out as adamw_update_8bit does
    int t_val = 1;
    float step_size_val = config_val.learning_rate / (1.0f - powf(config_val.beta1, (float)t_val));
    float inverse_sqrt_bias_correction2_val = 1.0f / sqrtf(1.0f - powf(config_val.beta2, (float)t_val));
    float weight_decay_factor_val = 1.0f - config_val.learning_rate * config_val.weight_decay;

    // 3. the usual launch's results, kept on the GPU as PyTorch tensors to compare each launch with
    adamw_update_8bit(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), moments_val.m_bytes(),
                      moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(), moments_val.sqrt_v_scales_tensor.data(),
                      params_in_slice_val, decayed_count_val, grad_norm_tensor.data(), config_val.max_norm,
                      config_val.learning_rate, config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay,
                      t_val, master_tensor.data());
    torch::Tensor usual_master_tensor = pytorch_copy_of(master_tensor);
    torch::Tensor usual_params_tensor = pytorch_copy_of(params_tensor.to(DType::Float32));
    torch::Tensor usual_m_tensor = pytorch_bytes_copy_of(moments_val.m_tensor);
    torch::Tensor usual_sqrt_v_tensor = pytorch_bytes_copy_of(moments_val.sqrt_v_tensor);
    torch::Tensor usual_m_scales_tensor = pytorch_copy_of(moments_val.m_scales_tensor);
    torch::Tensor usual_sqrt_v_scales_tensor = pytorch_copy_of(moments_val.sqrt_v_scales_tensor);
    int one_float4_per_thread_blocks_val =
        (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, (size_t)ADAMW_8BIT_THREADS_PER_BLOCK); // 121,524

    // 4. the update, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto update_val = [&] {
        adamw_8bit_kernel<<<blocks_val, threads_val>>>(
            params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), moments_val.m_bytes(),
            moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(), moments_val.sqrt_v_scales_tensor.data(),
            params_in_slice_val, decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, weight_decay_factor_val,
            config_val.beta1, config_val.beta2, config_val.eps, step_size_val, inverse_sqrt_bias_correction2_val,
            master_tensor.data());
        cudaCheck(cudaGetLastError());
    };
    blocks_val = one_float4_per_thread_blocks_val;
    threads_val = ADAMW_8BIT_THREADS_PER_BLOCK;
    warm_up_gpu(update_val);

    // 5. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 160, 640, 2560, 10240, 40960, one_float4_per_thread_blocks_val, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 9 * thread_count_choices_val;
    float total_ms_list[launches_val] = {}; // launch k is block_counts[k / 4] x thread_counts[k % 4]
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val] += gpu_milliseconds(update_val);
        }
    }

    // 6. print each launch's average, and check its results against the usual ones
    float fastest_ms_val = 1e9f;
    int fastest_blocks_val = 0;
    int fastest_threads_val = 0;
    int launches_with_the_same_result_val = 0;
    double peak_val = peak_gigabytes_per_second();
    double bytes_val = (double)params_in_slice_val * ADAMW_8BIT_BYTES_PER_VALUE;
    print_test_name(__func__);
    printf("    this GPU's peak memory bandwidth %.0f GB/s\n", peak_val);
    printf("    time in ms:   blocks   threads   float4s per thread     time     GB/s    MBU   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float time_val = total_ms_list[launch_val] / SWEEP_ROUNDS;
        if (time_val < fastest_ms_val)
        {
            fastest_ms_val = time_val;
            fastest_blocks_val = blocks_val;
            fastest_threads_val = threads_val;
        }

        // one update from the start with this launch, against the usual launch's
        master_tensor.copy_(start_master_tensor);
        params_tensor.copy_(start_params_tensor);
        for (const Tensor *tensor_ptr : {&moments_val.m_tensor, &moments_val.m_scales_tensor, &moments_val.sqrt_v_tensor,
                                         &moments_val.sqrt_v_scales_tensor})
            tensor_ptr->zero_();
        update_val();
        bool same_val = torch::equal(pytorch_copy_of(master_tensor), usual_master_tensor) &&
                        torch::equal(pytorch_copy_of(params_tensor.to(DType::Float32)), usual_params_tensor) &&
                        torch::equal(pytorch_bytes_copy_of(moments_val.m_tensor), usual_m_tensor) &&
                        torch::equal(pytorch_bytes_copy_of(moments_val.sqrt_v_tensor), usual_sqrt_v_tensor) &&
                        torch::equal(pytorch_copy_of(moments_val.m_scales_tensor), usual_m_scales_tensor) &&
                        torch::equal(pytorch_copy_of(moments_val.sqrt_v_scales_tensor), usual_sqrt_v_scales_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, threads_launched_val); // the most
        const char *usual_ptr = blocks_val == one_float4_per_thread_blocks_val && threads_val == ADAMW_8BIT_THREADS_PER_BLOCK
                                    ? "   <- what adamw_update_8bit launches: one float4 per thread"
                                    : "";
        double speed_val = gigabytes_per_second(bytes_val, time_val);
        printf("                  %6d   %7d   %18d   %6.3f   %6.0f   %3.0f%%   %-11s%s\n", blocks_val, threads_val,
               float4s_per_thread_val, time_val, speed_val, 100.0 * speed_val / peak_val, same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest       %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_val, fastest_threads_val, fastest_ms_val);

    // 7. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_the_three_adamws_on_gpt2_small_size();
    time_the_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
