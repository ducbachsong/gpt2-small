// test_adamw.cu — tests for llmc/kernels/adamw.cuh (AdamW all in float) and llmc/optimizer.cuh's AdamW on a
// float model. A bfloat16 model: test_adamw_bf16.cu; 8-bit m and v: test_adamw_8bit.cu. Every check prints its
// test case, input, expected and actual value, and ok; the first FAIL stops the program.
// The timings are in tunes/tune_adamw.cu.
//
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/utils.h>              // torch::NoGradGuard
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/optimizer.cuh"      // AdamW, AdamWConfig; and adamw_update, which the tests call directly

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
static const double STEPS_TOLERANCE = TOLERANCE;   // the common tests' after several steps

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

/// The float kernel's tensors for `count` values, all 0: what the common tests step, with adamw_update.
struct KernelState
{
    Tensor params_tensor;
    Tensor grads_tensor;
    Tensor m_tensor;
    Tensor v_tensor;
    Tensor grad_norm_tensor;
    explicit KernelState(size_t count_val)
        : params_tensor(Tensor::zeros({count_val})), grads_tensor(Tensor::zeros({count_val})),
          m_tensor(Tensor::zeros({count_val})), v_tensor(Tensor::zeros({count_val})), grad_norm_tensor(Tensor::zeros({1}))
    {
    }
    void set_params(const std::vector<float> &values_list) const { params_tensor.copy_from_cpu(values_list.data()); }
    void set_grads(const std::vector<float> &values_list) const { grads_tensor.copy_from_cpu(values_list.data()); }
    void set_grad_norm(float norm_val) const { grad_norm_tensor.copy_from_cpu(&norm_val); }
    std::vector<float> param_values() const { return params_tensor.tolist(); }
    std::vector<float> grad_values() const { return grads_tensor.tolist(); }
    std::vector<float> m_values() const { return m_tensor.tolist(); }
    std::vector<float> v_values() const { return v_tensor.tolist(); }
    /// Takes one step t on the first `count_val` values, the first `decayed_count_val` of them decayed, at this
    /// block size (the default if not given).
    void step(const AdamWConfig &config_val, size_t count_val, size_t decayed_count_val, int t_val,
              int threads_per_block_val = ADAMW_THREADS_PER_BLOCK) const
    {
        adamw_update(params_tensor.data(), grads_tensor.data(), m_tensor.data(), v_tensor.data(), count_val,
                     decayed_count_val, grad_norm_tensor.data(), config_val.max_norm, config_val.learning_rate,
                     config_val.beta1, config_val.beta2, config_val.eps, config_val.weight_decay, t_val,
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

// ══ part 1: my AdamW against PyTorch's fused AdamW ════════════════════════════

static void large_random_weight_matches_pytorch(void)
{
    // Expect: a 256x256 random weight, 100 steps of random gradients, every 7th of them tiny
    // (x 1e-6, where eps matters): my AdamW equals PyTorch's after every step.
    // 1. my AdamW and PyTorch's, from the same random weight
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    std::vector<float> weight_values_list = random_values({256, 256});
    AdamW my_adamw_val({{256, 256}}, config_val);
    my_adamw_val.param(0).copy_from_cpu(weight_values_list.data());
    PytorchGroup pytorch_val({pytorch_param(weight_values_list, {256, 256})});
    // 2. 100 steps, compared after each
    for (int t_val = 1; t_val <= 100; t_val++)
    {
        std::vector<float> step_grad_list = random_values({256, 256}, t_val % 7 == 1 ? 1e-6f : 1.0f);
        my_adamw_val.grad(0).copy_from_cpu(step_grad_list.data());
        my_adamw_val.step();
        pytorch_val.params_list[0].mutable_grad() = to_gpu(step_grad_list, pytorch_val.params_list[0].sizes());
        pytorch_val.step(config_val.weight_decay, config_val);
        ASSERT_NEAR(my_adamw_val.param(0).tolist(), values_of(pytorch_val.params_list[0]), TOLERANCE);
    }
    // 3. PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides
    ASSERT_FAR(values_of(pytorch_val.params_list[0]), weight_values_list, 1e-3);
}

static void bias_matches_pytorch_without_decay(void)
{
    // Expect: weight decay 0.1, but a bias (1-D) is in PyTorch's group without weight decay,
    // and my AdamW does not decay it either: one step, each of its 4 values equals PyTorch's.
    // 1. my AdamW and PyTorch's, from the same bias
    AdamWConfig config_val;
    config_val.learning_rate = 5e-2;
    std::vector<float> bias_values_list = {0.7f, -0.7f, 1.4f, 0.0f};
    std::vector<float> bias_grad_list = {0.2f, -0.2f, 0.1f, 0.3f};
    AdamW my_adamw_val({{4}}, config_val);
    my_adamw_val.param(0).copy_from_cpu(bias_values_list.data());
    PytorchGroup pytorch_val({pytorch_param(bias_values_list, {4})});
    // 2. one step each; a bias is in PyTorch's group with weight decay 0
    my_adamw_val.grad(0).copy_from_cpu(bias_grad_list.data());
    my_adamw_val.step();
    pytorch_val.params_list[0].mutable_grad() = to_gpu(bias_grad_list, pytorch_val.params_list[0].sizes());
    pytorch_val.step(0.0, config_val);
    // 3. compare; PyTorch really trained
    std::vector<float> my_bias_list = my_adamw_val.param(0).tolist();
    std::vector<float> pytorch_bias_list = values_of(pytorch_val.params_list[0]);
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(my_bias_list[i_val], pytorch_bias_list[i_val], 1e-6);
    ASSERT_FAR(pytorch_bias_list, bias_values_list, 1e-3);
}

static void mixed_parameters_match_pytorch(void)
{
    // Expect: 2-D, 3-D and 1-D tensors, {3, 4}, {2, 2, 3}, {4}, {8} (matrices first, as
    // AdamW needs), 20 steps with weight decay: every tensor equals PyTorch's after every
    // step. (C#'s sizes 5 and 7 became 4 and 8: the kernel needs a multiple of 4.)
    // 1. random tensors; my AdamW, and PyTorch's two param groups of the same tensors
    Tensor::manual_seed(1);
    AdamWConfig config_val;
    std::vector<Shape> param_shapes_list = {{3, 4}, {2, 2, 3}, {4}, {8}};
    std::vector<std::vector<float>> param_values_list;
    for (const Shape &shape_list : param_shapes_list)
        param_values_list.push_back(random_values(shape_list));
    AdamW my_adamw_val(param_shapes_list, config_val);
    for (int param_index_val = 0; param_index_val < 4; param_index_val++)
        my_adamw_val.param(param_index_val).copy_from_cpu(param_values_list[param_index_val].data());
    std::vector<torch::Tensor> pytorch_params_list;
    for (int param_index_val = 0; param_index_val < 4; param_index_val++)
        pytorch_params_list.push_back(pytorch_param(param_values_list[param_index_val], param_shapes_list[param_index_val]));
    PytorchGroup matrices_val({pytorch_params_list[0], pytorch_params_list[1]});
    PytorchGroup rest_val({pytorch_params_list[2], pytorch_params_list[3]});
    // 2. 20 steps, compared tensor by tensor after each
    for (int t_val = 1; t_val <= 20; t_val++)
    {
        std::vector<std::vector<float>> param_grads_list;
        for (const Shape &shape_list : param_shapes_list)
            param_grads_list.push_back(random_values(shape_list));
        for (int param_index_val = 0; param_index_val < 4; param_index_val++)
            my_adamw_val.grad(param_index_val).copy_from_cpu(param_grads_list[param_index_val].data());
        my_adamw_val.step();
        for (int param_index_val = 0; param_index_val < 4; param_index_val++)
            pytorch_params_list[param_index_val].mutable_grad() =
                to_gpu(param_grads_list[param_index_val], pytorch_params_list[param_index_val].sizes());
        matrices_val.step(config_val.weight_decay, config_val);
        rest_val.step(0.0, config_val);
        for (int param_index_val = 0; param_index_val < 4; param_index_val++)
            ASSERT_NEAR(my_adamw_val.param(param_index_val).tolist(), values_of(pytorch_params_list[param_index_val]),
                        TOLERANCE);
    }
    // 3. PyTorch really trained each tensor
    for (int param_index_val = 0; param_index_val < 4; param_index_val++)
        ASSERT_FAR(values_of(pytorch_params_list[param_index_val]), param_values_list[param_index_val], 1e-3);
}

static void whole_model_matches_pytorch_with_clipping(void)
{
    // Expect: the parameters of a tiny GPT-2 of 1 layer, 16 wide (16 tensors, 4464 values),
    // 20 steps with clipping at 1: odd steps get big gradients (norm about 67, clipped), even
    // steps small ones (about 0.07, left alone). After every step, all 4464 of my AdamW's
    // values (every tensor, one after the other, as PyTorch's tensors are) equal PyTorch's.
    // 1. the shapes, random values; my AdamW, and PyTorch's two param groups (the 6 matrices, the other 10)
    Tensor::manual_seed(0);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2;
    config_val.max_norm = 1.0;
    std::vector<Shape> param_shapes_list = {
        // the 6 matrices: 4224 values
        {64, 16}, // wte: vocab 64
        {8, 16},  // wpe: sequence 8
        {48, 16}, // qkvw
        {16, 16}, // attprojw
        {64, 16}, // fcw
        {16, 64}, // fcprojw
        // the 10 biases and LayerNorm tensors: 240 values
        {16}, {16}, {48}, {16}, {16}, {16}, {64}, {16}, // ln1w, ln1b, qkvb, attprojb, ln2w, ln2b, fcb, fcprojb
        {16}, {16},                                     // lnfw, lnfb
    };
    int num_params_val = (int)param_shapes_list.size();
    int num_matrices_val = 6;
    std::vector<std::vector<float>> param_values_list;
    for (const Shape &shape_list : param_shapes_list)
        param_values_list.push_back(random_values(shape_list, 0.02f));
    AdamW my_adamw_val(param_shapes_list, config_val);
    for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
        my_adamw_val.param(param_index_val).copy_from_cpu(param_values_list[param_index_val].data());
    std::vector<torch::Tensor> pytorch_params_list;
    for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
        pytorch_params_list.push_back(pytorch_param(param_values_list[param_index_val], param_shapes_list[param_index_val]));
    PytorchGroup matrices_val(std::vector<torch::Tensor>(pytorch_params_list.begin(), pytorch_params_list.begin() + num_matrices_val));
    PytorchGroup rest_val(std::vector<torch::Tensor>(pytorch_params_list.begin() + num_matrices_val, pytorch_params_list.end()));
    // 2. 20 steps, compared after each
    for (int t_val = 1; t_val <= 20; t_val++)
    {
        std::vector<std::vector<float>> param_grads_list;
        for (const Shape &shape_list : param_shapes_list)
            param_grads_list.push_back(random_values(shape_list, t_val % 2 == 1 ? 1.0f : 1e-3f));
        // my AdamW: the gradients, and their norm for the clipping
        for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
            my_adamw_val.grad(param_index_val).copy_from_cpu(param_grads_list[param_index_val].data());
        float grad_norm_val = cpu_grad_norm(param_grads_list);
        my_adamw_val.grad_norm().copy_from_cpu(&grad_norm_val);
        my_adamw_val.step();
        // PyTorch: the same gradients, clipped by clip_grad_norm_ (all 16 together); one step for each param group
        for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
            pytorch_params_list[param_index_val].mutable_grad() =
                to_gpu(param_grads_list[param_index_val], pytorch_params_list[param_index_val].sizes());
        torch::nn::utils::clip_grad_norm_(pytorch_params_list, config_val.max_norm);
        matrices_val.step(config_val.weight_decay, config_val);
        rest_val.step(0.0, config_val);
        // compare every value at once
        ASSERT_NEAR(all_my_adamw_values(my_adamw_val), all_values_of(pytorch_params_list), TOLERANCE);
    }
    // 3. PyTorch really trained the model
    std::vector<float> all_start_values_list;
    for (const std::vector<float> &tensor_values_list : param_values_list)
        all_start_values_list.insert(all_start_values_list.end(), tensor_values_list.begin(), tensor_values_list.end());
    ASSERT_FAR(all_values_of(pytorch_params_list), all_start_values_list, 1e-3);
}

// ══ part 2: rules worked out by hand ══════════════════════════════════════════
// My AdamW starts with every value 0, gradients too: a test that wants 0 gradients leaves
// them as they are.

static void weight_decay_shrinks_matrices_but_not_biases(void)
{
    // Expect: every gradient 0, so the only change is weight decay: the 2x2 weight
    // {2, -4, 6, 8} x (1 - 0.5 x 0.1) = x 0.95, the bias {2, -4, 6, 8} not at all.
    // 1. one step, gradients all 0
    AdamWConfig config_val;
    config_val.learning_rate = 0.5;
    config_val.weight_decay = 0.1;
    AdamW my_adamw_val({{2, 2}, {4}}, config_val);
    std::vector<float> start_values_list = {2.0f, -4.0f, 6.0f, 8.0f};
    my_adamw_val.param(0).copy_from_cpu(start_values_list.data()); // weight
    my_adamw_val.param(1).copy_from_cpu(start_values_list.data()); // bias
    my_adamw_val.step();
    // 2. check
    std::vector<float> weight_list = my_adamw_val.param(0).tolist();
    std::vector<float> expected_list = {1.9f, -3.8f, 5.7f, 7.6f};
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(weight_list[i_val], expected_list[i_val], 1e-6);
    ASSERT_EQ(my_adamw_val.param(1).tolist(), start_values_list);
}

static void a_parameter_the_loss_did_not_use_only_decays(void)
{
    // Expect: a 0 gradient keeps m at 0, so the Adam part is 0 and only weight decay is left:
    // x (1 - 0.1 x 0.1) = x 0.99. unused {5, 6, 7, 8} -> {4.95, 5.94, 6.93, 7.92}; used
    // {1, 2, 3, 4} with gradient 1 also moves by -lr = -0.1: {0.89, 1.88, 2.87, 3.86}.
    // 1. one step; the unused one's gradient stays 0
    AdamWConfig config_val;
    config_val.learning_rate = 0.1;
    config_val.weight_decay = 0.1;
    AdamW my_adamw_val({{2, 2}, {2, 2}}, config_val);
    std::vector<float> used_start_list = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> unused_start_list = {5.0f, 6.0f, 7.0f, 8.0f};
    my_adamw_val.param(0).copy_from_cpu(used_start_list.data());
    my_adamw_val.param(1).copy_from_cpu(unused_start_list.data());
    my_adamw_val.grad(0).fill_(1);
    my_adamw_val.step();
    // 2. check
    std::vector<float> used_list = my_adamw_val.param(0).tolist();
    std::vector<float> unused_list = my_adamw_val.param(1).tolist();
    std::vector<float> expected_used_list = {0.89f, 1.88f, 2.87f, 3.86f};
    std::vector<float> expected_unused_list = {4.95f, 5.94f, 6.93f, 7.92f};
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(unused_list[i_val], expected_unused_list[i_val], 1e-5);
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(used_list[i_val], expected_used_list[i_val], 1e-5);
}

static void weight_decay_follows_the_shape_not_the_order(void)
{
    // Expect: weight {1, 4}, weight {2, 2}, bias {4}, bias {8}, all holding 2, 4, 6, ...;
    // no gradients, lr 0.5, weight decay 0.1: the weights x 0.95, the biases not moved. The
    // shape decides: {1, 4} has one row but is 2-D, a matrix; {8} has as many values as a
    // 2x4 but is 1-D. (C# gave them mixed; AdamW needs the matrices first, so here they are.)
    // 1. one step, gradients all 0
    AdamWConfig config_val;
    config_val.learning_rate = 0.5;
    config_val.weight_decay = 0.1;
    std::vector<float> four_values_list = {2, 4, 6, 8};
    std::vector<float> eight_values_list = {2, 4, 6, 8, 10, 12, 14, 16};
    AdamW my_adamw_val({{1, 4}, {2, 2}, {4}, {8}}, config_val);
    my_adamw_val.param(0).copy_from_cpu(four_values_list.data());
    my_adamw_val.param(1).copy_from_cpu(four_values_list.data());
    my_adamw_val.param(2).copy_from_cpu(four_values_list.data());
    my_adamw_val.param(3).copy_from_cpu(eight_values_list.data());
    my_adamw_val.step();
    // 2. check: the weights shrunk, the biases not
    std::vector<float> weight1_list = my_adamw_val.param(0).tolist();
    std::vector<float> weight2_list = my_adamw_val.param(1).tolist();
    std::vector<float> shrunk_list = {1.9f, 3.8f, 5.7f, 7.6f};
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(weight1_list[i_val], shrunk_list[i_val], 1e-6);
    for (size_t i_val = 0; i_val < 4; i_val++)
        ASSERT_NEAR(weight2_list[i_val], shrunk_list[i_val], 1e-6);
    ASSERT_EQ(my_adamw_val.param(2).tolist(), four_values_list);
    ASSERT_EQ(my_adamw_val.param(3).tolist(), eight_values_list);
}

static void step_leaves_the_gradients_as_they_are(void)
{
    // Expect: C#'s Step zeroed the gradients (StepLeavesTheGradientsAtZero); our kernel only
    // reads them (const float*), so after a step with clipping they are still {1, 2, 3, 4}
    // and {5, 6, 7, 8}. The trainer zeroes them itself before the next backward.
    // 1. one step with clipping
    AdamWConfig config_val;
    config_val.learning_rate = 0.1;
    config_val.max_norm = 1.0;
    AdamW my_adamw_val({{2, 2}, {4}}, config_val);
    std::vector<std::vector<float>> param_grads_list = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    my_adamw_val.grad(0).copy_from_cpu(param_grads_list[0].data());
    my_adamw_val.grad(1).copy_from_cpu(param_grads_list[1].data());
    float grad_norm_val = cpu_grad_norm(param_grads_list); // sqrt(204) = 14.3 > 1: clipped
    my_adamw_val.grad_norm().copy_from_cpu(&grad_norm_val);
    my_adamw_val.step();
    // 2. check
    ASSERT_EQ(my_adamw_val.grad(0).tolist(), param_grads_list[0]);
    ASSERT_EQ(my_adamw_val.grad(1).tolist(), param_grads_list[1]);
}

static void adamw_counts_the_matrices_and_the_steps(void)
{
    // Expect: (new) params {2, 4}, {1, 4}, {8}, {4}, matrices first: decayed_count is the
    // matrices' 8 + 4 = 12 values. One buffer, 5 groups: params and grads (Optimizer's),
    // m and v (AdamW's), then grad_norm last; params, grads, m and v have 4 tensors and 24
    // values each: 17 tensors, 97 values. Tensor 0 is the {2, 4}, and its gradient starts
    // right after the 24 parameter values. m and v start on 16 bytes, as the kernel's
    // float4 loads need (grad_norm, 1 value, before them would put them 4 bytes off), and
    // at 0. steps_taken is 0 before the first step and 2 after two.
    // 1. the counts and the layout
    AdamWConfig config_val;
    AdamW my_adamw_val({{2, 4}, {1, 4}, {8}, {4}}, config_val);
    ASSERT_EQ(my_adamw_val.decayed_count(), (size_t)12);       // 2 x 4 + 1 x 4
    ASSERT_EQ(my_adamw_val.buffer().num_groups(), 5);          // params, grads, m, v, grad_norm
    ASSERT_EQ(my_adamw_val.buffer().num_tensors(), 17);        // 4 x 4 + 1
    ASSERT_EQ(my_adamw_val.buffer().numel(), (size_t)97);      // 4 x 24 + 1
    ASSERT_EQ(my_adamw_val.param(0).shape(), Shape({2, 4}));
    ASSERT_EQ(my_adamw_val.grad(0).data() == my_adamw_val.param(0).data() + 24, true);
    ASSERT_EQ((size_t)my_adamw_val.m(0).data() % sizeof(float4), (size_t)0);
    ASSERT_EQ((size_t)my_adamw_val.v(0).data() % sizeof(float4), (size_t)0);
    for (int param_index_val = 0; param_index_val < 4; param_index_val++)
    {
        ASSERT_EQ(my_adamw_val.m(param_index_val).tolist(), std::vector<float>(my_adamw_val.m(param_index_val).numel(), 0.0f));
        ASSERT_EQ(my_adamw_val.v(param_index_val).tolist(), std::vector<float>(my_adamw_val.v(param_index_val).numel(), 0.0f));
    }
    // 2. the steps counted
    ASSERT_EQ(my_adamw_val.steps_taken(), 0);
    my_adamw_val.step(); // gradients: all 0
    my_adamw_val.step();
    ASSERT_EQ(my_adamw_val.steps_taken(), 2);
}

static void set_learning_rate_is_used_from_the_next_step(void)
{
    // Expect: (new) 4 values of 1, gradient 1 every step, no weight decay. With the same gradient every step
    // m̂ = 1 and v̂ = 1, so each step moves each value by exactly its learning rate: 1 - 0.05 = 0.95 after step
    // 1 at lr 0.05; then set_learning_rate(0.01), and step 2 moves it by 0.01: 0.94.
    // 1. step 1 at lr 0.05
    AdamWConfig config_val;
    config_val.learning_rate = 0.05f;
    config_val.weight_decay = 0.0f;
    AdamW my_adamw_val({{4}}, config_val);
    my_adamw_val.param(0).fill_(1);
    my_adamw_val.grad(0).fill_(1);
    ASSERT_EQ(my_adamw_val.learning_rate(), 0.05f);
    my_adamw_val.step();
    ASSERT_NEAR(my_adamw_val.param(0).tolist(), std::vector<float>(4, 0.95f), 1e-6);
    // 2. step 2 at lr 0.01
    my_adamw_val.set_learning_rate(0.01f);
    ASSERT_EQ(my_adamw_val.learning_rate(), 0.01f);
    my_adamw_val.step();
    ASSERT_NEAR(my_adamw_val.param(0).tolist(), std::vector<float>(4, 0.94f), 1e-6);
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

    large_random_weight_matches_pytorch();
    bias_matches_pytorch_without_decay();
    mixed_parameters_match_pytorch();
    whole_model_matches_pytorch_with_clipping();

    weight_decay_shrinks_matrices_but_not_biases();
    a_parameter_the_loss_did_not_use_only_decays();
    weight_decay_follows_the_shape_not_the_order();
    step_leaves_the_gradients_as_they_are();
    adamw_counts_the_matrices_and_the_steps();
    set_learning_rate_is_used_from_the_next_step();

    printf("\nall tests passed\n");
    return 0;
}
