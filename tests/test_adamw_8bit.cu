// test_adamw_8bit.cu — tests for llmc/kernels/adamw_8bit.cuh (bfloat16 params and grads, a float master, m and
// v in 8 bits). Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops
// the program.
// The timings are in tunes/tune_adamw_8bit.cu.
//
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/types.h>              // torch::randn, torch::from_blob
#include <torch/utils.h>              // torch::manual_seed, torch::NoGradGuard
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/optimizer.cuh"            // AdamWConfig; adamw_update, which 8-bit AdamW is compared with
#include "../llmc/kernels/adamw_8bit.cuh"   // adamw_update_8bit

// The common tests' largest difference from PyTorch after several steps, as a fraction of the largest value:
// m and v in 8 bits move the params by up to 2.8e-4 of float's (a Python copy).
static const double STEPS_TOLERANCE = 1e-3;


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

/// Gives the first `count_val` 8-bit values of `bytes_tensor` as they are stored, as floats: -448 .. 448.
static std::vector<float> stored_values(const Tensor &bytes_tensor, size_t count_val)
{
    std::vector<__nv_fp8_e4m3> stored_list(count_val);
    cudaCheck(cudaMemcpy(stored_list.data(), bytes_tensor.data_ptr(), count_val, cudaMemcpyDeviceToHost));
    std::vector<float> values_list;
    for (const __nv_fp8_e4m3 &value_val : stored_list)
        values_list.push_back((float)value_val);
    return values_list;
}

/// Gives the first `count_val` values that 8-bit `bytes_tensor` and their `scales_tensor` stand for:
/// stored x scale.
static std::vector<float> decoded_values(const Tensor &bytes_tensor, const Tensor &scales_tensor, size_t count_val)
{
    std::vector<float> values_list = stored_values(bytes_tensor, count_val);
    std::vector<float> scale_values_list = scales_tensor.tolist();
    for (size_t i_val = 0; i_val < count_val; i_val++)
        values_list[i_val] *= scale_values_list[i_val / ADAMW_VALUES_PER_SCALE];
    return values_list;
}

/// Gives the first `count_val` values that 8-bit sqrt(v) and its scales stand for, squared: v.
static std::vector<float> decoded_v_values(const Tensor &bytes_tensor, const Tensor &scales_tensor, size_t count_val)
{
    std::vector<float> values_list = decoded_values(bytes_tensor, scales_tensor, count_val);
    for (float &value_val : values_list)
        value_val *= value_val; // sqrt(v) squared
    return values_list;
}

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

/// The 8-bit kernel's tensors for `count` values, all 0: what the common tests step, with adamw_update_8bit.
/// Params and grads bfloat16, the master float, m and sqrt(v) 8-bit; param_values() gives the master, m_values()
/// and v_values() the values the bytes stand for.
struct KernelState
{
    size_t count_val;
    Tensor params_tensor;
    Tensor grads_tensor;
    Tensor master_tensor;
    Tensor grad_norm_tensor;
    EightBitMoments moments_val;
    explicit KernelState(size_t state_count_val)
        : count_val(state_count_val), params_tensor(Tensor::zeros({state_count_val}, DType::BFloat16)),
          grads_tensor(Tensor::zeros({state_count_val}, DType::BFloat16)), master_tensor(Tensor::zeros({state_count_val})),
          grad_norm_tensor(Tensor::zeros({1})), moments_val(state_count_val)
    {
    }
    void set_params(const std::vector<float> &values_list) const // rounded to bfloat16; the master the same
    {
        master_tensor.copy_from_cpu(values_list.data());
        params_tensor.copy_(master_tensor);
        master_tensor.copy_(params_tensor);
    }
    void set_grads(const std::vector<float> &values_list) const // rounded to bfloat16
    {
        Tensor floats_tensor = Tensor::empty({values_list.size()});
        floats_tensor.copy_from_cpu(values_list.data());
        grads_tensor.copy_(floats_tensor);
    }
    void set_grad_norm(float norm_val) const { grad_norm_tensor.copy_from_cpu(&norm_val); }
    std::vector<float> param_values() const { return master_tensor.tolist(); }
    std::vector<float> grad_values() const { return grads_tensor.to(DType::Float32).tolist(); }
    std::vector<float> m_values() const
    {
        return decoded_values(moments_val.m_tensor, moments_val.m_scales_tensor, count_val);
    }
    std::vector<float> v_values() const
    {
        return decoded_v_values(moments_val.sqrt_v_tensor, moments_val.sqrt_v_scales_tensor, count_val);
    }
    /// Takes one step t on the first `step_count_val` values, the first `decayed_count_val` of them decayed, at
    /// this block size (the default if not given).
    void step(const AdamWConfig &config_val, size_t step_count_val, size_t decayed_count_val, int t_val,
              int threads_per_block_val = ADAMW_8BIT_THREADS_PER_BLOCK) const
    {
        adamw_update_8bit(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                          moments_val.m_bytes(), moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(),
                          moments_val.sqrt_v_scales_tensor.data(), step_count_val, decayed_count_val,
                          grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate, config_val.beta1,
                          config_val.beta2, config_val.eps, config_val.weight_decay, t_val, master_tensor.data(),
                          threads_per_block_val);
    }
};

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

// ══ common: the same tests in test_adamw.cu, test_adamw_bf16.cu and test_adamw_8bit.cu ═══════════
// Each steps this file's kernel through KernelState. PyTorch gets the start values and gradients as the kernel
// holds them (rounded to bfloat16 in a bfloat16 kernel), so the expected numbers are the same in all three.

static void one_step_matches_pytorch(void)
{
    // Expect: one step on a 2x2 weight {0.5, -1, 2, 0} with gradient {0.1, -0.2, 0.3, 0.05}, lr 0.01, weight
    // decay 0.1: each value equals PyTorch's (about 0.4895, -0.989, 1.988, -0.01), within 1e-6 of the largest.
    // Step 1 works in float before anything is stored, so every kernel matches this closely.
    // 1. the kernel and PyTorch, from the same start
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    KernelState state_val(4);
    state_val.set_params({0.5f, -1.0f, 2.0f, 0.0f});
    state_val.set_grads({0.1f, -0.2f, 0.3f, 0.05f});
    std::vector<float> start_values_list = state_val.param_values();
    PytorchGroup pytorch_val({pytorch_param(start_values_list, {2, 2})});
    // 2. one step each
    state_val.step(config_val, 4, 4, 1);
    pytorch_val.params_list[0].mutable_grad() = to_gpu(state_val.grad_values(), pytorch_val.params_list[0].sizes());
    pytorch_val.step(config_val.weight_decay, config_val);
    // 3. check; PyTorch really trained: its weight moved far more than the difference allowed
    ASSERT_NEAR(state_val.param_values(), values_of(pytorch_val.params_list[0]), 1e-6);
    ASSERT_FAR(values_of(pytorch_val.params_list[0]), start_values_list, 1e-3);
}

static void many_steps_match_pytorch(void)
{
    // Expect: 10 steps on a 2x2 weight, a new gradient each step (some values 0), lr 0.003: the kernel equals
    // PyTorch's after every step, within STEPS_TOLERANCE of the largest value, so m and v carry over the same way.
    static float grads_per_step_list[10][4] = {
        {0.3f, -0.2f, 1.0f, 0.01f}, // step 1
        {-0.1f, 0.4f, 0.9f, -0.02f},
        {0.25f, -0.15f, 0.8f, 0.03f},
        {0.0f, 0.3f, 0.7f, -0.04f},
        {-0.4f, 0.0f, 0.6f, 0.05f},
        {0.2f, -0.5f, 0.5f, 0.0f},
        {0.1f, 0.2f, 0.4f, -0.06f},
        {-0.05f, 0.35f, 0.3f, 0.07f},
        {0.5f, -0.1f, 0.2f, -0.08f},
        {-0.3f, 0.05f, 0.1f, 0.09f}, // step 10
    };
    // 1. the kernel and PyTorch, from the same start
    AdamWConfig config_val;
    config_val.learning_rate = 3e-3f;
    KernelState state_val(4);
    state_val.set_params({0.8f, -0.5f, 1.2f, 0.3f});
    std::vector<float> start_values_list = state_val.param_values();
    PytorchGroup pytorch_val({pytorch_param(start_values_list, {2, 2})});
    // 2. ten steps, compared after each
    for (int t_val = 1; t_val <= 10; t_val++)
    {
        state_val.set_grads(std::vector<float>(grads_per_step_list[t_val - 1], grads_per_step_list[t_val - 1] + 4));
        state_val.step(config_val, 4, 4, t_val);
        pytorch_val.params_list[0].mutable_grad() = to_gpu(state_val.grad_values(), pytorch_val.params_list[0].sizes());
        pytorch_val.step(config_val.weight_decay, config_val);
        ASSERT_NEAR(state_val.param_values(), values_of(pytorch_val.params_list[0]), STEPS_TOLERANCE);
    }
    // 3. PyTorch really trained
    ASSERT_FAR(values_of(pytorch_val.params_list[0]), start_values_list, 1e-3);
}

/// Steps the kernel and PyTorch twice on a 2x2 weight {0.5, -1, 2, 0.25} with these two gradients, lr 0.01 and
/// this max_norm (PyTorch clips with clip_grad_norm_ when `pytorch_clips_val`), and checks they agree after each.
static void two_clipped_steps_match_pytorch(const std::vector<float> (&grads_per_step_list)[2], float max_norm_val,
                                            bool pytorch_clips_val)
{
    // 1. the kernel and PyTorch, from the same start
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    config_val.max_norm = max_norm_val;
    KernelState state_val(4);
    state_val.set_params({0.5f, -1.0f, 2.0f, 0.25f});
    std::vector<float> start_values_list = state_val.param_values();
    PytorchGroup pytorch_val({pytorch_param(start_values_list, {2, 2})});
    // 2. two steps, compared after each
    for (int t_val = 1; t_val <= 2; t_val++)
    {
        // the kernel: the gradient, and its norm for the clipping
        state_val.set_grads(grads_per_step_list[t_val - 1]);
        state_val.set_grad_norm(cpu_grad_norm({state_val.grad_values()}));
        state_val.step(config_val, 4, 4, t_val);
        // PyTorch, the same gradient, clipped by clip_grad_norm_
        pytorch_val.params_list[0].mutable_grad() = to_gpu(state_val.grad_values(), pytorch_val.params_list[0].sizes());
        if (pytorch_clips_val)
            torch::nn::utils::clip_grad_norm_(pytorch_val.params_list, config_val.max_norm);
        pytorch_val.step(config_val.weight_decay, config_val);
        ASSERT_NEAR(state_val.param_values(), values_of(pytorch_val.params_list[0]), STEPS_TOLERANCE);
    }
    // 3. PyTorch really trained
    ASSERT_FAR(values_of(pytorch_val.params_list[0]), start_values_list, 1e-3);
}

static void clip_scales_large_gradients_down_to_max_norm(void)
{
    // Expect: two steps on a 2x2 weight with max_norm 1. Step 1's gradient (norm 0.38) is not clipped; step 2's
    // {1, 2, 3, 4} has norm sqrt(30) = 5.48 > 1, so both sides divide it by 5.48 first. The kernel equals
    // PyTorch's after each step, within STEPS_TOLERANCE.
    const std::vector<float> grads_per_step_list[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {1.0f, 2.0f, 3.0f, 4.0f}};
    two_clipped_steps_match_pytorch(grads_per_step_list, 1.0f, true);
}

static void clip_leaves_small_gradients_alone(void)
{
    // Expect: the same two steps, but step 2's gradient {0.3, 0, 0.4, 0} has norm 0.5 < 1: not clipped. The
    // kernel equals PyTorch's after each step, within STEPS_TOLERANCE.
    const std::vector<float> grads_per_step_list[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {0.3f, 0.0f, 0.4f, 0.0f}};
    two_clipped_steps_match_pytorch(grads_per_step_list, 1.0f, true);
}

static void max_norm_0_turns_clipping_off(void)
{
    // Expect: step 2's gradient has norm sqrt(30), but max_norm is 0: PyTorch does not clip, and neither may the
    // kernel, though grad_norm holds 5.48. The kernel equals PyTorch's after each step, within STEPS_TOLERANCE.
    const std::vector<float> grads_per_step_list[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {1.0f, 2.0f, 3.0f, 4.0f}};
    two_clipped_steps_match_pytorch(grads_per_step_list, 0.0f, false);
}

static void first_step_moves_each_value_by_about_the_learning_rate(void)
{
    // Expect: at t = 1, m̂ = g and v̂ = g², so the step is lr g / |g| = ±lr, whatever the size of g. 4 values of 1,
    // gradients {1000, 0.001, -5, -0.3}, lr 0.05, no weight decay: 1 - 0.05 where g > 0, 1 + 0.05 where g < 0.
    // 1. one step
    AdamWConfig config_val;
    config_val.learning_rate = 0.05f;
    config_val.weight_decay = 0.0f;
    KernelState state_val(4);
    state_val.set_params({1.0f, 1.0f, 1.0f, 1.0f});
    state_val.set_grads({1000.0f, 0.001f, -5.0f, -0.3f});
    state_val.step(config_val, 4, 4, 1);
    // 2. check
    std::vector<float> values_list = state_val.param_values();
    std::vector<float> expected_list = {0.95f, 0.95f, 1.05f, 1.05f};
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(values_list[i_val], expected_list[i_val], 1e-4);
}

static void only_the_values_before_decayed_count_decay(void)
{
    // Expect: the kernel's border. 8 values {2, 4, 6, 8, 2, 4, 6, 8}, decayed_count 4, no gradients, lr 0.5,
    // weight decay 0.1: the first 4 x 0.95 = {1.9, 3.8, 5.7, 7.6}, the last 4 not moved.
    // 1. one step, gradients all 0
    AdamWConfig config_val;
    config_val.learning_rate = 0.5f;
    config_val.weight_decay = 0.1f;
    KernelState state_val(8);
    state_val.set_params({2, 4, 6, 8, 2, 4, 6, 8});
    state_val.step(config_val, 8, 4, 1);
    // 2. check
    std::vector<float> values_list = state_val.param_values();
    std::vector<float> expected_list = {1.9f, 3.8f, 5.7f, 7.6f, 2, 4, 6, 8};
    for (size_t i_val = 0; i_val < 8; i_val++)
        ASSERT_NEAR(values_list[i_val], expected_list[i_val], 1e-6);
}

static void nothing_past_a_tensors_end_changes(void)
{
    // Expect: a launch covers at least one block of 256 threads x 4 values, far more than 8. A step on the
    // first 8 of 16 values leaves the last 8's params, m and v as they were: params {10 .. 17}, m and v 0.
    // 1. 16 values; a step on the first 8 only
    AdamWConfig config_val;
    config_val.learning_rate = 0.1f;
    config_val.weight_decay = 0.0f;
    KernelState state_val(16);
    std::vector<float> values_list = sample_data(8);
    std::vector<float> last_values_list = {10, 11, 12, 13, 14, 15, 16, 17};
    values_list.insert(values_list.end(), last_values_list.begin(), last_values_list.end());
    state_val.set_params(values_list);
    state_val.set_grads(std::vector<float>(16, 1.0f));
    state_val.step(config_val, 8, 0, 1);
    // 2. check the last 8
    std::vector<float> params_list = state_val.param_values();
    std::vector<float> m_list = state_val.m_values();
    std::vector<float> v_list = state_val.v_values();
    ASSERT_EQ(std::vector<float>(params_list.begin() + 8, params_list.end()), last_values_list);
    ASSERT_EQ(std::vector<float>(m_list.begin() + 8, m_list.end()), std::vector<float>(8, 0.0f));
    ASSERT_EQ(std::vector<float>(v_list.begin() + 8, v_list.end()), std::vector<float>(8, 0.0f));
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: (new) 10,000 random values, the first 8,000 decayed, 3 steps with clipping; the update launched
    // with threads_per_block 32, 128, 512 and 1024 (one warp up to MAX_THREADS_PER_BLOCK) gives the same params,
    // m and v as the default block size, to the last bit after every step: each value is worked out alone.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    // 1. random start values and gradients for 3 steps
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    config_val.max_norm = 1.0f;
    size_t count_val = 10000;
    size_t decayed_count_val = 8000;
    std::vector<float> start_values_list = random_values({count_val}, 0.02f);
    std::vector<std::vector<float>> grads_per_step_list;
    for (int t_val = 1; t_val <= 3; t_val++)
        grads_per_step_list.push_back(random_values({count_val}));
    // 2. the default block size: 3 steps, each step's params, m and v kept
    std::vector<std::vector<float>> usual_params_list, usual_m_list, usual_v_list;
    KernelState usual_val(count_val);
    usual_val.set_params(start_values_list);
    for (int t_val = 1; t_val <= 3; t_val++)
    {
        usual_val.set_grads(grads_per_step_list[t_val - 1]);
        usual_val.set_grad_norm(cpu_grad_norm({grads_per_step_list[t_val - 1]}));
        usual_val.step(config_val, count_val, decayed_count_val, t_val);
        usual_params_list.push_back(usual_val.param_values());
        usual_m_list.push_back(usual_val.m_values());
        usual_v_list.push_back(usual_val.v_values());
    }
    // 3. each other block size, from the same start, against the default's after every step
    for (int threads_per_block_val : block_sizes_list)
    {
        KernelState state_val(count_val);
        state_val.set_params(start_values_list);
        for (int t_val = 1; t_val <= 3; t_val++)
        {
            state_val.set_grads(grads_per_step_list[t_val - 1]);
            state_val.set_grad_norm(cpu_grad_norm({grads_per_step_list[t_val - 1]}));
            state_val.step(config_val, count_val, decayed_count_val, t_val, threads_per_block_val);
            ASSERT_EQ(state_val.param_values(), usual_params_list[t_val - 1]);
            ASSERT_EQ(state_val.m_values(), usual_m_list[t_val - 1]);
            ASSERT_EQ(state_val.v_values(), usual_v_list[t_val - 1]);
        }
    }
}

// ══ m and v in 8 bits ═════════════════════════════════════════════════════════

static void eight_bit_m_and_v_keep_the_nearest_e4m3(void)
{
    // Expect: (new) one step on 4 bfloat16 params of 1 with a float master, gradients {1, 0.5, -0.25, 3} (exact
    // in bfloat16), lr 0.01, no weight decay. Step 1 does its math in float before anything is stored, so it
    // moves each master value by exactly lr, as the float AdamW does: {0.99, 0.99, 1.01, 0.99}. Then
    // m = 0.1 g = {0.1, 0.05, -0.025, 0.3} is stored as
    // m x 448 / 0.3 = {149.3, 74.7, -37.3, 448}, each to its nearest E4M3 (16 apart from 128 to 256, 8 from 64,
    // 4 from 32): {144, 72, -36, 448}, which read back as {0.0964, 0.0482, -0.0241, 0.3}. sqrt(v) = sqrt(0.05)
    // |g| has the same ratios, so also {144, 72, 36, 448}; v = 0.05 g² = {0.05, 0.0125, 0.003125, 0.45} reads
    // back as {0.0465, 0.0116, 0.0029, 0.45}.

    // 1. one step
    Tensor master_tensor = Tensor::ones({4});
    Tensor params_tensor = Tensor::ones({4}, DType::BFloat16);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    Tensor grads_tensor = Tensor::empty({4});
    std::vector<float> grad_values_list = {1.0f, 0.5f, -0.25f, 3.0f};
    grads_tensor.copy_from_cpu(grad_values_list.data());
    grads_tensor = grads_tensor.to(DType::BFloat16);
    EightBitMoments moments_val(4);
    adamw_update_8bit(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), moments_val.m_bytes(),
                      moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(), moments_val.sqrt_v_scales_tensor.data(),
                      4, 4, grad_norm_tensor.data(), 0.0f, 0.01f, 0.9f, 0.95f, 1e-8f, 0.0f, 1, master_tensor.data());

    // 2. check the master, the bytes, and what they stand for
    ASSERT_NEAR(master_tensor.tolist(), std::vector<float>({0.99f, 0.99f, 1.01f, 0.99f}), 1e-6);
    ASSERT_EQ(stored_values(moments_val.m_tensor, 4), std::vector<float>({144.0f, 72.0f, -36.0f, 448.0f}));
    ASSERT_EQ(stored_values(moments_val.sqrt_v_tensor, 4), std::vector<float>({144.0f, 72.0f, 36.0f, 448.0f}));
    ASSERT_NEAR(decoded_values(moments_val.m_tensor, moments_val.m_scales_tensor, 4),
                std::vector<float>({0.0964286f, 0.0482143f, -0.0241071f, 0.3f}), 1e-5);
    ASSERT_NEAR(decoded_v_values(moments_val.sqrt_v_tensor, moments_val.sqrt_v_scales_tensor, 4),
                std::vector<float>({0.0464924f, 0.0116231f, 0.00290577f, 0.45f}), 1e-5);
}

static void a_small_value_next_to_a_big_one_keeps_moving(void)
{
    // Expect: (new) 8 bfloat16 params of 1 with a float master (one scale), the same bfloat16 gradients every
    // step: {1000, 0.01 (0.0100098 in bfloat16), 1, 1, 1, 1, 1, 1}, lr 1e-3, no weight decay, 10 steps. With the
    // same gradient every step m̂ = g and v̂ = g², so the float AdamW moves every value by exactly lr a step:
    // 0.99 after 10, in the master. In 8 bits the 0.01 one is stored 1e5 times
    // smaller than the 1000 one, 0.0045 of 448, between E4M3's smallest values: its m and sqrt(v) are both a
    // few % off, about the same way, and m / sqrt(v) stays right: 0.99 too, within 1e-4. (Storing v itself
    // instead, 5e-6 next to 5e4, would store 0 and add only 0.05 g² each step: the 0.01 one would have moved by
    // 0.00105 in all, not 0.01.)

    // 1. ten steps of the same gradients
    Tensor master_tensor = Tensor::ones({8});
    Tensor params_tensor = Tensor::ones({8}, DType::BFloat16);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    Tensor grads_tensor = Tensor::empty({8});
    std::vector<float> grad_values_list = {1000.0f, 0.01f, 1, 1, 1, 1, 1, 1};
    grads_tensor.copy_from_cpu(grad_values_list.data());
    grads_tensor = grads_tensor.to(DType::BFloat16);
    EightBitMoments moments_val(8);
    for (int t_val = 1; t_val <= 10; t_val++)
        adamw_update_8bit(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(), moments_val.m_bytes(),
                          moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(),
                          moments_val.sqrt_v_scales_tensor.data(), 8, 8, grad_norm_tensor.data(), 0.0f, 1e-3f, 0.9f, 0.95f,
                          1e-8f, 0.0f, t_val, master_tensor.data());

    // 2. check: every value moved by 10 x lr
    std::vector<float> master_values_list = master_tensor.tolist();
    for (size_t i_val = 0; i_val < 8; i_val++)
        ASSERT_NEAR(master_values_list[i_val], 0.99f, 1e-4);
}

static void eight_bit_adamw_stays_near_float_adamw(void)
{
    // Expect: (new) a {100, 100} weight (values std 0.02, as GPT-2's), 100 steps of random bfloat16 gradients
    // (std 1), lr 1e-3, weight decay 0.1; once all in float (adamw_update, the gradients as floats), once in
    // bfloat16 with a float master and m and v in 8 bits. 10,000 values: 78 scales of 128 and one of 16, so a
    // scale with a warp only half used. The float params and the master do the same math but for m and v, so
    // their moves from the start differ by a few %: about 7% on average, at most about 11%, from a Python
    // copy of this kernel with 10 seeds (E4M3 keeps 3 bits after the point: up to 6% off per value stored).
    // Allowed: 20% of the largest move.

    // 1. the same random start, all in float and with 8-bit m and v
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-3f;
    size_t count_val = 10000;
    std::vector<float> start_values_list = random_values({count_val}, 0.02f);
    Tensor master_float_tensor = Tensor::empty({count_val});
    Tensor master_8bit_tensor = Tensor::empty({count_val});
    master_float_tensor.copy_from_cpu(start_values_list.data());
    master_8bit_tensor.copy_from_cpu(start_values_list.data());
    Tensor params_8bit_tensor = master_8bit_tensor.to(DType::BFloat16);
    Tensor m_tensor = Tensor::zeros({count_val});
    Tensor v_tensor = Tensor::zeros({count_val});
    Tensor grad_norm_tensor = Tensor::zeros({1});
    EightBitMoments moments_val(count_val);

    // 2. 100 steps of the same gradients
    for (int t_val = 1; t_val <= 100; t_val++)
    {
        Tensor grads_tensor = Tensor::empty({count_val});
        grads_tensor.copy_from_cpu(random_values({count_val}, 1.0f).data());
        grads_tensor = grads_tensor.to(DType::BFloat16);
        Tensor grads_as_floats_tensor = grads_tensor.to(DType::Float32);   // the same gradients
        adamw_update(master_float_tensor.data(), grads_as_floats_tensor.data(), m_tensor.data(), v_tensor.data(),
                     count_val, count_val, grad_norm_tensor.data(), 0.0f, config_val.learning_rate, config_val.beta1,
                     config_val.beta2, config_val.eps, config_val.weight_decay, t_val);
        adamw_update_8bit(params_8bit_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                          moments_val.m_bytes(), moments_val.m_scales_tensor.data(), moments_val.sqrt_v_bytes(),
                          moments_val.sqrt_v_scales_tensor.data(), count_val, count_val, grad_norm_tensor.data(), 0.0f,
                          config_val.learning_rate, config_val.beta1, config_val.beta2, config_val.eps,
                          config_val.weight_decay, t_val, master_8bit_tensor.data());
    }

    // 3. each side's moves from the start, compared
    std::vector<float> moves_float_list = master_float_tensor.tolist();
    std::vector<float> moves_8bit_list = master_8bit_tensor.tolist();
    for (size_t i_val = 0; i_val < count_val; i_val++)
    {
        moves_float_list[i_val] -= start_values_list[i_val];
        moves_8bit_list[i_val] -= start_values_list[i_val];
    }
    ASSERT_NEAR(moves_8bit_list, moves_float_list, 0.2);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    one_step_matches_pytorch();
    many_steps_match_pytorch();
    clip_scales_large_gradients_down_to_max_norm();
    clip_leaves_small_gradients_alone();
    max_norm_0_turns_clipping_off();
    first_step_moves_each_value_by_about_the_learning_rate();
    only_the_values_before_decayed_count_decay();
    nothing_past_a_tensors_end_changes();
    any_block_size_gives_the_same_bits();

    eight_bit_m_and_v_keep_the_nearest_e4m3();
    a_small_value_next_to_a_big_one_keeps_moving();
    eight_bit_adamw_stays_near_float_adamw();

    printf("\nall tests passed\n");
    return 0;
}
