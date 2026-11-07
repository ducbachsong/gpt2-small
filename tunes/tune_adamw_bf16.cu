// tune_adamw_bf16.cu — the timings of tests/test_adamw_bf16.cu, moved here so the tests only check numbers:
// time_the_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_adamw_bf16.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/adamw_bf16_<date>_<time>.txt.
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/utils.h>              // torch::NoGradGuard
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/optimizer.cuh"      // AdamW, AdamWConfig; and adamw_update_bf16 (kernels/adamw_bf16.cuh)

#define GPT2_PARAMS 124439808         // GPT-2 small's values
#define GPT2_DECAYED_PARAMS 124439040 // of them in matrices

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

// ══ part 1: other block and thread counts ═════════════════════════════════════

static void time_the_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of one update on as many values as GPT-2 small has, 124,439,808 (the matrices'
    // 124,439,040 decayed, then 768), as training runs it: bfloat16 params, grads, m and v, and a float master.
    // 9 block counts x 4 thread counts, each the average of 20 runs: the 7 picked on a T4, then this GPU's SM
    // count x 1 and x 2 (114 and 228 on an H100 PCIe). The 36 launches take turns over 20 rounds, so each gets
    // runs from early, middle and late alike. The kernel is launched directly, <<<blocks, threads>>>;
    // adamw_update_bf16 launches one float4 per thread, 121,524 x 256. Each update reads master, g, m, v and
    // writes master, p, m, v: 4 + 2 + 2 + 2 and 4 + 2 + 2 + 2 = 20 bytes per value, 2.5 GB: at a T4's 320 GB/s at
    // least 7.8 ms, at an H100 PCIe's 2.0 TB/s 1.24 ms.
    //
    // Each run changes the master, p, m and v, so before each check they are put back: the master and p from
    // kept copies, m and v to 0. Expect: every launch gives the same master, p, m and v as the usual one, to the
    // last bit.

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
    Tensor m_tensor = Tensor::zeros({params_in_slice_val}, DType::BFloat16);
    Tensor v_tensor = Tensor::zeros({params_in_slice_val}, DType::BFloat16);
    Tensor grad_norm_tensor = Tensor::empty({1});
    float norm_val = pytorch_copy_of(grads_float_tensor).norm().item<float>(); // about sqrt(124M) = 11,155: clipped
    grad_norm_tensor.copy_from_cpu(&norm_val);

    // 2. step t = 1's numbers, worked out as adamw_update_bf16 does
    int t_val = 1;
    float step_size_val = config_val.learning_rate / (1.0f - powf(config_val.beta1, (float)t_val));
    float inverse_sqrt_bias_correction2_val = 1.0f / sqrtf(1.0f - powf(config_val.beta2, (float)t_val));
    float weight_decay_factor_val = 1.0f - config_val.learning_rate * config_val.weight_decay;

    // 3. the usual launch's results, kept on the GPU as float PyTorch tensors to compare each launch with
    adamw_update_bf16(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                      m_tensor.data_ptr<nv_bfloat16>(), v_tensor.data_ptr<nv_bfloat16>(), params_in_slice_val,
                      decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate,
                      config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay, t_val,
                      master_tensor.data());
    torch::Tensor usual_master_tensor = pytorch_copy_of(master_tensor);
    torch::Tensor usual_params_tensor = pytorch_copy_of(params_tensor.to(DType::Float32));
    torch::Tensor usual_m_tensor = pytorch_copy_of(m_tensor.to(DType::Float32));
    torch::Tensor usual_v_tensor = pytorch_copy_of(v_tensor.to(DType::Float32));
    int one_float4_per_thread_blocks_val =
        (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, (size_t)ADAMW_BF16_THREADS_PER_BLOCK); // 121,524

    // 4. the update, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto update_val = [&] {
        adamw_bf16_kernel<<<blocks_val, threads_val>>>(
            params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), m_tensor.data_ptr<nv_bfloat16>(),
            v_tensor.data_ptr<nv_bfloat16>(), params_in_slice_val, decayed_count_val, grad_norm_tensor.data(),
            config_val.max_norm, weight_decay_factor_val, config_val.beta1, config_val.beta2, config_val.eps, step_size_val,
            inverse_sqrt_bias_correction2_val, master_tensor.data());
        cudaCheck(cudaGetLastError());
    };
    blocks_val = one_float4_per_thread_blocks_val;
    threads_val = ADAMW_BF16_THREADS_PER_BLOCK;
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
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   float4s per thread     time   same result\n");
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
        m_tensor.zero_();
        v_tensor.zero_();
        update_val();
        bool same_val = torch::equal(pytorch_copy_of(master_tensor), usual_master_tensor) &&
                        torch::equal(pytorch_copy_of(params_tensor.to(DType::Float32)), usual_params_tensor) &&
                        torch::equal(pytorch_copy_of(m_tensor.to(DType::Float32)), usual_m_tensor) &&
                        torch::equal(pytorch_copy_of(v_tensor.to(DType::Float32)), usual_v_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, threads_launched_val); // the most
        const char *usual_ptr = blocks_val == one_float4_per_thread_blocks_val && threads_val == ADAMW_BF16_THREADS_PER_BLOCK
                                    ? "   <- what adamw_update_bf16 launches: one float4 per thread"
                                    : "";
        printf("                  %6d   %7d   %18d   %6.3f   %-11s%s\n", blocks_val, threads_val, float4s_per_thread_val,
               time_val, same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest       %6d blocks x %4d threads: %.3f ms\n", fastest_blocks_val, fastest_threads_val, fastest_ms_val);

    // 7. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_the_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
