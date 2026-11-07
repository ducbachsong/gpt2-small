// test_adamw_bf16.cu — tests for llmc/kernels/adamw_bf16.cuh (bfloat16 params, grads, m and v, a float master)
// and llmc/optimizer.cuh's AdamW on a bfloat16 model. Every check prints its test case, input, expected and
// actual value, and ok; the first FAIL stops the program.
// The timings are in tunes/tune_adamw_bf16.cu.
//
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/utils.h>              // torch::NoGradGuard
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/optimizer.cuh"      // AdamW, AdamWConfig; and adamw_update_bf16 (kernels/adamw_bf16.cuh)

// The common tests' largest difference from PyTorch after several steps, as a fraction of the largest value:
// m and v rounded to bfloat16 each step move the params by up to 5.8e-5 of float's (a Python copy).
static const double STEPS_TOLERANCE = 1e-4;

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

/// The bfloat16 kernel's tensors for `count` values, all 0: what the common tests step, with adamw_update_bf16.
/// Params, grads, m and v bfloat16, the master float; param_values() gives the master.
struct KernelState
{
    Tensor params_tensor;
    Tensor grads_tensor;
    Tensor m_tensor;
    Tensor v_tensor;
    Tensor master_tensor;
    Tensor grad_norm_tensor;
    explicit KernelState(size_t count_val)
        : params_tensor(Tensor::zeros({count_val}, DType::BFloat16)), grads_tensor(Tensor::zeros({count_val}, DType::BFloat16)),
          m_tensor(Tensor::zeros({count_val}, DType::BFloat16)), v_tensor(Tensor::zeros({count_val}, DType::BFloat16)),
          master_tensor(Tensor::zeros({count_val})), grad_norm_tensor(Tensor::zeros({1}))
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
    std::vector<float> m_values() const { return m_tensor.to(DType::Float32).tolist(); }
    std::vector<float> v_values() const { return v_tensor.to(DType::Float32).tolist(); }
    /// Takes one step t on the first `count_val` values, the first `decayed_count_val` of them decayed, at this
    /// block size (the default if not given).
    void step(const AdamWConfig &config_val, size_t count_val, size_t decayed_count_val, int t_val,
              int threads_per_block_val = ADAMW_BF16_THREADS_PER_BLOCK) const
    {
        adamw_update_bf16(params_tensor.data_ptr<nv_bfloat16>(), grads_tensor.data_ptr<nv_bfloat16>(),
                          m_tensor.data_ptr<nv_bfloat16>(), v_tensor.data_ptr<nv_bfloat16>(), count_val, decayed_count_val,
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

// ══ part 1: a bfloat16 model ══════════════════════════════════════════════════

static void bfloat16_params_and_grads_match_pytorch(void)
{
    // Expect: (new) one step of adamw_update_bf16 (no master) on a {32, 32} weight in bfloat16, values about
    // GPT-2's (normal, std 0.02), with bfloat16 gradients (std 1); m and v bfloat16. PyTorch's fused AdamW, in
    // float, on the same values: m and v are PyTorch's rounded to bfloat16, up to 2^-9 of each value: within
    // 4e-3 of the largest (2.3e-3 in a Python copy). The step's math is in float before anything is rounded, so
    // the weight is PyTorch's rounded to bfloat16, which moves a value by up to 2^-8 (0.4%) of it: within twice
    // that of the largest value, 8e-3. And the step is not lost to the rounding: each value moves by about
    // lr = 1e-3, more than the 2^-8 x 0.08 = 3.1e-4 that a bfloat16 near the largest value rounds by, so the
    // weight is far from where it started: 1e-3 / 0.08 = 1.25e-2, more than 8e-3.

    // 1. a random bfloat16 weight and gradient; one step of the kernel, no master
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-3f;
    Shape shape_list = {32, 32};
    Tensor param_tensor = Tensor::empty(shape_list);
    Tensor grad_tensor = Tensor::empty(shape_list);
    param_tensor.copy_from_cpu(random_values(shape_list, 0.02f).data());
    grad_tensor.copy_from_cpu(random_values(shape_list, 1.0f).data());
    param_tensor = param_tensor.to(DType::BFloat16);
    grad_tensor = grad_tensor.to(DType::BFloat16);
    Tensor m_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor v_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    std::vector<float> start_values_list = param_tensor.to(DType::Float32).tolist();
    std::vector<float> grad_values_list = grad_tensor.to(DType::Float32).tolist();
    adamw_update_bf16(param_tensor.data_ptr<nv_bfloat16>(), grad_tensor.data_ptr<nv_bfloat16>(),
                      m_tensor.data_ptr<nv_bfloat16>(), v_tensor.data_ptr<nv_bfloat16>(), param_tensor.numel(),
                      param_tensor.numel(), grad_norm_tensor.data(), 0.0f, config_val.learning_rate, config_val.beta1,
                      config_val.beta2, config_val.eps, config_val.weight_decay, 1, nullptr);

    // 2. PyTorch's fused AdamW, in float, on the same values
    PytorchGroup pytorch_val({pytorch_param(start_values_list, shape_list)});
    pytorch_val.params_list[0].mutable_grad() = to_gpu(grad_values_list, pytorch_val.params_list[0].sizes());
    pytorch_val.step(config_val.weight_decay, config_val);

    // 3. check: m, v and the weight within one rounding; the weight moved
    ASSERT_NEAR(m_tensor.to(DType::Float32).tolist(), values_of(pytorch_val.m_list[0]), 4e-3);
    ASSERT_NEAR(v_tensor.to(DType::Float32).tolist(), values_of(pytorch_val.v_list[0]), 4e-3);
    ASSERT_NEAR(param_tensor.to(DType::Float32).tolist(), values_of(pytorch_val.params_list[0]), 8e-3);
    ASSERT_FAR(param_tensor.to(DType::Float32).tolist(), start_values_list, 8e-3);
}

static void bfloat16_adamw_keeps_a_float_master(void)
{
    // Expect: (new) adamw_counts_the_matrices_and_the_steps's params, for a bfloat16 model. The model's params,
    // grads, m and v are bfloat16, in a buffer of their own: grad 0 starts right after the 24 params (48 bytes),
    // m and v at 0. The float buffer has 2 groups, the master params and grad_norm: 24 + 1 = 25 values; the
    // master on 16 bytes, as the kernel's float4 loads need. decayed_count is still 12. After filling param 0
    // with 0 .. 7 (each exact in bfloat16), copy_params_to_master gives master_param 0 the same values.

    // 1. the dtypes, shapes and layout
    AdamWConfig config_val;
    AdamW my_adamw_val({{2, 4}, {1, 4}, {8}, {4}}, config_val, DType::BFloat16);
    ASSERT_EQ(my_adamw_val.dtype() == DType::BFloat16, true);
    ASSERT_EQ(my_adamw_val.param(0).dtype() == DType::BFloat16, true);
    ASSERT_EQ(my_adamw_val.grad(0).dtype() == DType::BFloat16, true);
    ASSERT_EQ(my_adamw_val.master_param(0).dtype() == DType::Float32, true);
    ASSERT_EQ(my_adamw_val.m(0).dtype() == DType::BFloat16, true);
    ASSERT_EQ(my_adamw_val.v(0).dtype() == DType::BFloat16, true);
    ASSERT_EQ(my_adamw_val.m(0).to(DType::Float32).tolist(), std::vector<float>(8, 0.0f));
    ASSERT_EQ(my_adamw_val.param(0).shape(), Shape({2, 4}));
    ASSERT_EQ(my_adamw_val.master_param(0).shape(), Shape({2, 4}));
    ASSERT_EQ((const void *)my_adamw_val.grad(0).data_ptr<nv_bfloat16>(),
              (const void *)(my_adamw_val.param(0).data_ptr<nv_bfloat16>() + 24));
    ASSERT_EQ(my_adamw_val.all_params().numel(), (size_t)24);
    ASSERT_EQ(my_adamw_val.buffer().num_groups(), 2);          // master params, grad_norm
    ASSERT_EQ(my_adamw_val.buffer().numel(), (size_t)25);      // 24 + 1
    ASSERT_EQ((size_t)my_adamw_val.master_param(0).data() % sizeof(float4), (size_t)0);
    ASSERT_EQ(my_adamw_val.decayed_count(), (size_t)12);

    // 2. copy_params_to_master
    Tensor values_tensor = Tensor::empty({2, 4});
    values_tensor.copy_from_cpu(sample_data(8).data());
    my_adamw_val.param(0).copy_(values_tensor);
    my_adamw_val.copy_params_to_master();
    ASSERT_EQ(my_adamw_val.master_param(0).tolist(), sample_data(8));
}

static void bfloat16_master_steps_match_float_steps(void)
{
    // Expect: (new) a {4, 8} weight and an {8} bias, the same values in a float AdamW and a bfloat16 one (each
    // value rounded to bfloat16 first, so both start equal), 10 steps of the same random gradients (bfloat16,
    // given to the float one as floats), lr 0.01, clipping at 1. The bfloat16 one's master does the same
    // float math, but from m and v rounded to bfloat16 after each step: near the float one's params, within
    // 1e-3 of the largest value (at most 3.7e-4 in a Python copy, 20 seeds). Its model params are that master
    // rounded to bfloat16, to the last bit.

    // 1. a float AdamW and a bfloat16 one, from the same rounded values
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    config_val.max_norm = 1.0f;
    std::vector<Shape> shapes_list = {{4, 8}, {8}};
    AdamW float_adamw_val(shapes_list, config_val), bfloat16_adamw_val(shapes_list, config_val, DType::BFloat16);
    for (int param_index_val = 0; param_index_val < 2; param_index_val++)
    {
        Tensor values_tensor = Tensor::empty(shapes_list[param_index_val]);
        values_tensor.copy_from_cpu(random_values(shapes_list[param_index_val], 0.5f).data());
        bfloat16_adamw_val.param(param_index_val).copy_(values_tensor);                   // rounded
        float_adamw_val.param(param_index_val).copy_(bfloat16_adamw_val.param(param_index_val)); // the rounded values, as floats
    }
    bfloat16_adamw_val.copy_params_to_master();

    // 2. ten steps of the same gradients, compared after each
    for (int t_val = 1; t_val <= 10; t_val++)
    {
        std::vector<std::vector<float>> grads_list;
        for (int param_index_val = 0; param_index_val < 2; param_index_val++)
        {
            Tensor values_tensor = Tensor::empty(shapes_list[param_index_val]);
            values_tensor.copy_from_cpu(random_values(shapes_list[param_index_val], 1.0f).data());
            bfloat16_adamw_val.grad(param_index_val).copy_(values_tensor);
            float_adamw_val.grad(param_index_val).copy_(bfloat16_adamw_val.grad(param_index_val));
            grads_list.push_back(float_adamw_val.grad(param_index_val).tolist());
        }
        float grad_norm_val = cpu_grad_norm(grads_list);         // about sqrt(40) = 6.3 > 1: clipped
        float_adamw_val.grad_norm().copy_from_cpu(&grad_norm_val);
        bfloat16_adamw_val.grad_norm().copy_from_cpu(&grad_norm_val);
        float_adamw_val.step();
        bfloat16_adamw_val.step();
        std::vector<float> master_list = bfloat16_adamw_val.buffer().group(0).tolist();   // group 0: the master params
        ASSERT_NEAR(master_list, float_adamw_val.all_params().tolist(), 1e-3);
        ASSERT_EQ(bfloat16_adamw_val.all_params().to(DType::Float32).tolist(),
                  bfloat16_adamw_val.buffer().group(0).to(DType::BFloat16).to(DType::Float32).tolist());
    }
}

static void a_small_update_adds_up_in_the_master(void)
{
    // Expect: (new) a LayerNorm weight of 1, gradient 1 every step, lr 6e-4, no weight decay: each step moves
    // it by exactly 6e-4. The bfloat16 numbers next to 1 are 0.99609375 below and 1.0078125 above, so
    // 1 - 6e-4 = 0.9994 rounds back to 1.
    //
    //     after 10 steps    float master                 bfloat16 param               without a master
    //                       1 - 10 x 6e-4 = 0.994        0.9921875 (nearest 0.994)    1: every step lost
    //
    // 0.994 is 0.0018 from 0.9921875 and 0.0021 from 0.99609375. m and v are bfloat16, a little off: the master
    // is 0.9939988 (a Python copy), within 1e-5. "Without a master" is adamw_update_bf16 straight on bfloat16
    // params, as AdamW did before it kept one.

    // 1. an AdamW with a master, and the kernel alone without one
    AdamWConfig config_val;
    config_val.learning_rate = 6e-4f;
    config_val.weight_decay = 0.0f;
    AdamW my_adamw_val({{4}}, config_val, DType::BFloat16);
    my_adamw_val.param(0).fill_(1);
    my_adamw_val.copy_params_to_master();
    Tensor alone_tensor = Tensor::ones({4}, DType::BFloat16);
    Tensor alone_grad_tensor = Tensor::ones({4}, DType::BFloat16);
    Tensor m_tensor = Tensor::zeros({4}, DType::BFloat16);
    Tensor v_tensor = Tensor::zeros({4}, DType::BFloat16);
    Tensor grad_norm_tensor = Tensor::zeros({1});

    // 2. ten steps of each
    for (int t_val = 1; t_val <= 10; t_val++)
    {
        my_adamw_val.grad(0).fill_(1);
        my_adamw_val.step();
        adamw_update_bf16(alone_tensor.data_ptr<nv_bfloat16>(), alone_grad_tensor.data_ptr<nv_bfloat16>(),
                          m_tensor.data_ptr<nv_bfloat16>(), v_tensor.data_ptr<nv_bfloat16>(), 4, 0, grad_norm_tensor.data(),
                          0.0f, config_val.learning_rate, config_val.beta1, config_val.beta2, config_val.eps, 0.0f, t_val,
                          nullptr);
    }

    // 3. check
    ASSERT_NEAR(my_adamw_val.master_param(0).tolist(), std::vector<float>(4, 0.994f), 1e-5);
    ASSERT_EQ(my_adamw_val.param(0).to(DType::Float32).tolist(), std::vector<float>(4, 0.9921875f));
    ASSERT_EQ(alone_tensor.to(DType::Float32).tolist(), std::vector<float>(4, 1.0f));
}

static void ranks_each_update_their_own_slice(void)
{
    // Expect: (new) a {3, 4} weight and a {4} bias: 16 values, decayed_count 12. On 2 GPUs each updates 8: rank 0
    // values 0 .. 7 (all decayed), rank 1 values 8 .. 15 (8 .. 11 decayed: the border inside its slice). Rank 1's
    // float buffer holds its master for 8 values, and grad_norm: 8 + 1 = 9 (m and v are bfloat16). One bfloat16 AdamW and
    // ranks 0 and 1 of 2 get the same params and the same (already averaged) grads, 3 steps. Right after a step,
    // rank 0's values 8 .. 15 are still the old ones: rank 1 updated them. Put together as the all-gather does
    // (each rank's slice copied to the other), both equal the one AdamW's params, to the last bit: each value
    // had the same float math.

    // 1. one AdamW, and ranks 0 and 1 of 2; rank 1's slice
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    std::vector<Shape> shapes_list = {{3, 4}, {4}};
    AdamW one_val(shapes_list, config_val, DType::BFloat16);
    AdamW rank0_val(shapes_list, config_val, DType::BFloat16, 0, 2), rank1_val(shapes_list, config_val, DType::BFloat16, 1, 2);
    ASSERT_EQ(rank1_val.slice_start(), (size_t)8);
    ASSERT_EQ(rank1_val.slice_size(), (size_t)8);
    ASSERT_EQ(rank1_val.buffer().numel(), (size_t)9);
    ASSERT_EQ(rank1_val.m_slice().numel(), (size_t)8);
    ASSERT_EQ(rank1_val.decayed_count(), (size_t)12);

    // 2. the same start values for all three
    Tensor::manual_seed(0);
    Tensor values_tensor = Tensor::empty({16});
    values_tensor.copy_from_cpu(random_values({16}, 0.5f).data());
    for (AdamW *adamw_ptr : {&one_val, &rank0_val, &rank1_val})
    {
        adamw_ptr->all_params().copy_(values_tensor);             // rounded to bfloat16
        adamw_ptr->copy_params_to_master();
    }

    // 3. three steps: rank 0 leaves rank 1's half alone; gathered, both equal the one AdamW
    for (int t_val = 1; t_val <= 3; t_val++)
    {
        Tensor grads_tensor = Tensor::empty({16});
        grads_tensor.copy_from_cpu(random_values({16}, 1.0f).data());
        std::vector<float> rank0_before_list = rank0_val.all_params().to(DType::Float32).tolist();
        for (AdamW *adamw_ptr : {&one_val, &rank0_val, &rank1_val})
        {
            adamw_ptr->all_grads().copy_(grads_tensor);
            adamw_ptr->step();
        }
        std::vector<float> rank0_after_list = rank0_val.all_params().to(DType::Float32).tolist();
        ASSERT_EQ(std::vector<float>(rank0_after_list.begin() + 8, rank0_after_list.end()),
                  std::vector<float>(rank0_before_list.begin() + 8, rank0_before_list.end()));
        rank0_val.all_params().narrow(0, 8, 8).copy_(rank1_val.all_params().narrow(0, 8, 8));   // the all-gather, by hand
        rank1_val.all_params().narrow(0, 0, 8).copy_(rank0_val.all_params().narrow(0, 0, 8));
        ASSERT_EQ(rank0_val.all_params().to(DType::Float32).tolist(), one_val.all_params().to(DType::Float32).tolist());
        ASSERT_EQ(rank1_val.all_params().to(DType::Float32).tolist(), one_val.all_params().to(DType::Float32).tolist());
    }
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

    bfloat16_params_and_grads_match_pytorch();
    bfloat16_adamw_keeps_a_float_master();
    bfloat16_master_steps_match_float_steps();
    a_small_update_adds_up_in_the_master();
    ranks_each_update_their_own_slice();

    printf("\nall tests passed\n");
    return 0;
}
