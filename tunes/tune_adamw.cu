// tune_adamw.cu — the timings of tests/test_adamw.cu, moved here so the tests only check numbers:
// time_20_updates_on_gpt2_small_size, time_the_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_adamw.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/adamw_<date>_<time>.txt.
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/utils.h>              // torch::NoGradGuard
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/optimizer.cuh"      // AdamW, AdamWConfig; and adamw_update, which part 2 calls directly

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_PARAMS 124439808         // GPT-2 small's values
#define GPT2_DECAYED_PARAMS 124439040 // of them in matrices

// ── helpers: my side ──────────────────────────────────────────────────────────

/// Works out the norm of all the gradients on the CPU, for clipping: every value of every
/// tensor as one long list.
static float cpu_grad_norm(const std::vector<std::vector<float>> &param_grads_list)
{
    double sum_squares_val = 0.0;
    for (const std::vector<float> &tensor_grads_list : param_grads_list)
        for (float g_val : tensor_grads_list)
            sum_squares_val += (double)g_val * g_val;
    return (float)sqrt(sum_squares_val);
}

/// Makes random values from Tensor::randn, times `scale_val`, as a CPU list: the same numbers
/// for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list, float scale_val = 1.0f)
{
    std::vector<float> numbers_list = Tensor::randn(shape_list).tolist();
    for (float &number_val : numbers_list)
        number_val *= scale_val;
    return numbers_list;
}

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

/// Gives every value of my AdamW's parameters, one tensor after the other, to compare
/// with PyTorch's.
static std::vector<float> all_my_adamw_values(const AdamW &my_adamw_val)
{
    std::vector<float> values_list;
    for (int param_index_val = 0; param_index_val < my_adamw_val.num_params(); param_index_val++)
    {
        std::vector<float> tensor_values_list = my_adamw_val.param(param_index_val).tolist();
        values_list.insert(values_list.end(), tensor_values_list.begin(), tensor_values_list.end());
    }
    return values_list;
}

// ── helpers: PyTorch's side ───────────────────────────────────────────────────

/// Makes a new GPU tensor of this shape from values in row order: the tests' CPU lists.
static torch::Tensor to_gpu(const std::vector<float> &row_values_list, torch::IntArrayRef shape_list)
{
    return torch::tensor(row_values_list).reshape(shape_list).to(torch::kCUDA);
}

/// Makes one PyTorch parameter on the GPU from a test's values and shape. requires_grad:
/// it gets a .grad, as an optimizer's parameters do.
static torch::Tensor pytorch_param(const std::vector<float> &values_list, const Shape &shape_list)
{
    std::vector<int64_t> sizes_list(shape_list.begin(), shape_list.end());
    return to_gpu(values_list, sizes_list).requires_grad_(true);
}

/// Makes AdamW's m (or v) for each of these parameters: zeros of the same shape, as
/// PyTorch's AdamW starts them.
static std::vector<torch::Tensor> zeros_like_each(const std::vector<torch::Tensor> &params_list)
{
    std::vector<torch::Tensor> zeros_list;
    for (const torch::Tensor &param_tensor : params_list)
        zeros_list.push_back(torch::zeros_like(param_tensor));
    return zeros_list;
}

/// Makes a step count of 0 for each of `count_val` parameters: one float on the GPU each, as
/// fused AdamW keeps them.
static std::vector<torch::Tensor> zero_step_counts(size_t count_val)
{
    std::vector<torch::Tensor> steps_list;
    for (size_t index_val = 0; index_val < count_val; index_val++)
        steps_list.push_back(torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)));
    return steps_list;
}

/// Does one step of PyTorch's fused AdamW on one param group, from the gradient each
/// parameter holds.
static void pytorch_adamw_step(std::vector<torch::Tensor> &params_list, std::vector<torch::Tensor> &m_list,
                               std::vector<torch::Tensor> &v_list, std::vector<torch::Tensor> &steps_list,
                               double weight_decay_val, const AdamWConfig &config_val)
{
    torch::NoGradGuard no_grad_val; // the update is not part of the graph, as in Python
    std::vector<torch::Tensor> grads_list;
    for (const torch::Tensor &param_tensor : params_list)
        grads_list.push_back(param_tensor.grad());
    at::_foreach_add_(steps_list, at::Scalar(1.0)); // step t = t + 1
    at::_fused_adamw_(params_list, grads_list, m_list, v_list, at::TensorList(), // no amsgrad state
                      steps_list, config_val.learning_rate, config_val.beta1, config_val.beta2, weight_decay_val,
                      config_val.eps, /*amsgrad=*/false, /*maximize=*/false);
}

/// Gives a PyTorch tensor's values in row order on the CPU, to compare with mine.
static std::vector<float> values_of(const torch::Tensor &tensor)
{
    torch::Tensor flat_tensor = tensor.detach().to(torch::kCPU).reshape(-1).contiguous();
    return std::vector<float>(flat_tensor.data_ptr<float>(), flat_tensor.data_ptr<float>() + flat_tensor.numel());
}

/// Makes a PyTorch copy of a float tensor of mine, on the GPU: to keep a result and compare a later one
/// with it there, without moving 124M values to the CPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of these PyTorch tensors, one tensor after the other.
static std::vector<float> all_values_of(const std::vector<torch::Tensor> &tensors_list)
{
    std::vector<float> values_list;
    for (const torch::Tensor &tensor : tensors_list)
    {
        std::vector<float> tensor_values_list = values_of(tensor);
        values_list.insert(values_list.end(), tensor_values_list.begin(), tensor_values_list.end());
    }
    return values_list;
}

/// One PyTorch param group to step alongside the kernel: the parameters, and fused AdamW's m, v and step counts.
struct PytorchGroup
{
    std::vector<torch::Tensor> params_list;
    std::vector<torch::Tensor> m_list;
    std::vector<torch::Tensor> v_list;
    std::vector<torch::Tensor> steps_list;
    explicit PytorchGroup(const std::vector<torch::Tensor> &group_params_list)
        : params_list(group_params_list), m_list(zeros_like_each(group_params_list)),
          v_list(zeros_like_each(group_params_list)), steps_list(zero_step_counts(group_params_list.size()))
    {
    }
    /// Takes one fused AdamW step, with this weight decay.
    void step(double weight_decay_val, const AdamWConfig &config_val)
    {
        pytorch_adamw_step(params_list, m_list, v_list, steps_list, weight_decay_val, config_val);
    }
};

// ══ part 1: my AdamW timed against PyTorch's fused AdamW ══════════════════════

/// Times TIMED_STEPS AdamW updates on a matrix and a bias of these shapes, both on the GPU, and
/// prints a table under `test_ptr`'s name:
///
///     my AdamW   AdamW::step: one launch of adamw_update
///     PyTorch    fused: per param group (2 here), +1 on the step counts, then _fused_adamw_
///
/// The update does not care about layers, only about how many values there are and the 2
/// groups (matrices, 1-D), so a matrix and a bias are enough. Only the update is timed:
/// the gradients are set once, before (PyTorch's clipped by clip_grad_norm_), and all the
/// updates use them. Each update reads p, g, m, v and writes p, m, v: 7 x 4 bytes per
/// value. Checks at the end that my AdamW still equals PyTorch's: both did the same work.
static void time_20_updates(const char *test_ptr, const std::vector<Shape> &param_shapes_list)
{
    // 1. random values and gradients
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.max_norm = 1.0;
    std::vector<std::vector<float>> param_values_list;
    std::vector<std::vector<float>> param_grads_list;
    for (const Shape &shape_list : param_shapes_list)
    {
        param_values_list.push_back(random_values(shape_list, 0.02f));
        param_grads_list.push_back(random_values(shape_list));
    }
    // 2. my AdamW, its gradients and their norm
    AdamW my_adamw_val(param_shapes_list, config_val);
    for (int param_index_val = 0; param_index_val < 2; param_index_val++)
    {
        my_adamw_val.param(param_index_val).copy_from_cpu(param_values_list[param_index_val].data());
        my_adamw_val.grad(param_index_val).copy_from_cpu(param_grads_list[param_index_val].data());
    }
    float grad_norm_val = cpu_grad_norm(param_grads_list);
    my_adamw_val.grad_norm().copy_from_cpu(&grad_norm_val);
    // 3. PyTorch's: the matrix and the bias, each its own param group, their gradients clipped
    std::vector<torch::Tensor> pytorch_params_list;
    for (int param_index_val = 0; param_index_val < 2; param_index_val++)
    {
        pytorch_params_list.push_back(pytorch_param(param_values_list[param_index_val], param_shapes_list[param_index_val]));
        pytorch_params_list[param_index_val].mutable_grad() =
            to_gpu(param_grads_list[param_index_val], pytorch_params_list[param_index_val].sizes());
    }
    torch::nn::utils::clip_grad_norm_(pytorch_params_list, config_val.max_norm);
    PytorchGroup matrices_val({pytorch_params_list[0]});
    PytorchGroup rest_val({pytorch_params_list[1]});

    // 4. time each update; step 1 does first-time work (CUDA loads each side's kernels), so the average is of
    //    the rest
    print_test_name(test_ptr);
    printf("    time of one update, in ms:   step   my AdamW   PyTorch fused\n");
    double my_total_val = 0.0;
    double pytorch_total_val = 0.0;
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float my_ms_val = gpu_milliseconds([&] { my_adamw_val.step(); });
        float pytorch_ms_val = gpu_milliseconds([&] {
            matrices_val.step(config_val.weight_decay, config_val);
            rest_val.step(0.0, config_val);
        });
        printf("                                 %4d   %8.3f   %13.3f\n", step_val, my_ms_val, pytorch_ms_val);
        if (step_val == 1)
            continue;
        my_total_val += my_ms_val;
        pytorch_total_val += pytorch_ms_val;
    }
    printf("    average of steps 2..%d:             %8.3f   %13.3f   PyTorch takes %.2fx as long\n", TIMED_STEPS,
           my_total_val / (TIMED_STEPS - 1), pytorch_total_val / (TIMED_STEPS - 1), pytorch_total_val / my_total_val);

    // 5. both did the same work
    ASSERT_NEAR(all_my_adamw_values(my_adamw_val), all_values_of(pytorch_params_list), TOLERANCE);
}

static void time_20_updates_on_gpt2_small_size(void)
{
    // Prints the times on as many values as GPT-2 small has: 162,030 x 768 + 768 =
    // 124,439,040 + 768 = 124,439,808. Each update moves 7 x 124,439,808 x 4 bytes = 3.5 GB:
    // at a T4's 320 GB/s, at least 10.9 ms.
    time_20_updates(__func__, {{162030, 768}, {768}});
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_the_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of one update on as many values as GPT-2 small has, 124,439,808 (the matrices'
    // 124,439,040 decayed, then 768), all in float, for 9 block counts x 4 thread counts, each the average of
    // 20 runs: the 7 picked on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe). The
    // 36 launches take turns over 20 rounds, so each gets runs from early, middle and late alike (a T4's clock
    // drifts as it heats up). The kernel is launched directly, <<<blocks, threads>>>; adamw_update launches one
    // float4 per thread, 121,524 x 256. Each update reads p, g, m, v and writes p, m, v: 7 x 4 = 28 bytes per
    // value, 3.5 GB: at a T4's 320 GB/s at least 10.9 ms, at an H100 PCIe's 2.0 TB/s 1.74 ms.
    //
    // Each run changes p, m and v, so before each check they are put back: p from a kept copy, m and v to 0.
    // Expect: every launch gives the same p, m and v as the usual one, to the last bit.

    // 1. GPT-2 small's number of values, random, made on the GPU
    torch::manual_seed(0);
    AdamWConfig config_val;
    config_val.max_norm = 1.0;
    size_t params_in_slice_val = GPT2_PARAMS;
    size_t decayed_count_val = GPT2_DECAYED_PARAMS;
    Tensor params_tensor = random_tensor_on_gpu({params_in_slice_val}, 0.02f);
    Tensor grads_tensor = random_tensor_on_gpu({params_in_slice_val}, 1.0f);
    Tensor start_params_tensor = params_tensor.clone();
    Tensor m_tensor = Tensor::zeros({params_in_slice_val});
    Tensor v_tensor = Tensor::zeros({params_in_slice_val});
    Tensor grad_norm_tensor = Tensor::empty({1});
    float norm_val = pytorch_copy_of(grads_tensor).norm().item<float>(); // about sqrt(124M) = 11,155: clipped
    grad_norm_tensor.copy_from_cpu(&norm_val);

    // 2. step t = 1's numbers, worked out as adamw_update does
    int t_val = 1;
    float step_size_val = config_val.learning_rate / (1.0f - powf(config_val.beta1, (float)t_val));
    float inverse_sqrt_bias_correction2_val = 1.0f / sqrtf(1.0f - powf(config_val.beta2, (float)t_val));
    float weight_decay_factor_val = 1.0f - config_val.learning_rate * config_val.weight_decay;

    // 3. the usual launch's results, kept on the GPU as PyTorch tensors to compare each launch with
    adamw_update(params_tensor.data(), grads_tensor.data(), m_tensor.data(), v_tensor.data(), params_in_slice_val,
                 decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate,
                 config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay, t_val);
    torch::Tensor usual_params_tensor = pytorch_copy_of(params_tensor);
    torch::Tensor usual_m_tensor = pytorch_copy_of(m_tensor);
    torch::Tensor usual_v_tensor = pytorch_copy_of(v_tensor);
    int one_float4_per_thread_blocks_val =
        (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, (size_t)ADAMW_THREADS_PER_BLOCK); // 121,524

    // 4. the update, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto update_val = [&] {
        adamw_kernel<<<blocks_val, threads_val>>>(params_tensor.data(), grads_tensor.data(), m_tensor.data(),
                                                  v_tensor.data(), params_in_slice_val, decayed_count_val,
                                                  grad_norm_tensor.data(), config_val.max_norm, weight_decay_factor_val,
                                                  config_val.beta1, config_val.beta2, config_val.eps, step_size_val,
                                                  inverse_sqrt_bias_correction2_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = one_float4_per_thread_blocks_val;
    threads_val = ADAMW_THREADS_PER_BLOCK;
    warm_up_gpu(update_val);

    // 5. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int t4_blocks_val = 40;                            // a T4's pick: 40 x 128
    const int t4_threads_val = 4 * THREADS_PER_WARP;
    int block_counts_list[] = {t4_blocks_val, 160, 640, 2560, 10240, 40960, one_float4_per_thread_blocks_val, sms_val,
                               2 * sms_val};
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
        params_tensor.copy_(start_params_tensor);
        m_tensor.zero_();
        v_tensor.zero_();
        update_val();
        bool same_val = torch::equal(pytorch_copy_of(params_tensor), usual_params_tensor) &&
                        torch::equal(pytorch_copy_of(m_tensor), usual_m_tensor) &&
                        torch::equal(pytorch_copy_of(v_tensor), usual_v_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, threads_launched_val); // the most
        const char *usual_ptr = blocks_val == one_float4_per_thread_blocks_val && threads_val == ADAMW_THREADS_PER_BLOCK
                                    ? "   <- what adamw_update launches: one float4 per thread"
                                : blocks_val == t4_blocks_val && threads_val == t4_threads_val
                                    ? "   <- a T4's pick, what it launched before"
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
    time_20_updates_on_gpt2_small_size();
    time_the_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
