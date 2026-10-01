// test_adamw.cu — tests for llmc/optimizer.cuh (AdamW) and llmc/kernels/adamw.cuh. Every
// check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
#include <ATen/ops/_foreach_add.h>    // at::_foreach_add_: +1 on every step count; libtorch before any CUDA header
#include <ATen/ops/_fused_adamw.h>    // at::_fused_adamw_: PyTorch's fused AdamW, the op fused=True calls
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; only the headers needed, not <torch/torch.h>
#include <torch/utils.h>              // torch::NoGradGuard
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/optimizer.cuh"      // AdamW, AdamWConfig; and adamw_update, which part 2 calls directly

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers: my side ──────────────────────────────────────────────────────────

/// Works out the norm of all the gradients on the CPU, for clipping: every value of every
static float cpu_grad_norm(const std::vector<std::vector<float>> &param_grads)
{
    double sum_squares = 0.0;
    for (const std::vector<float> &tensor_grads : param_grads)
        for (float g : tensor_grads)
            sum_squares += (double)g * g;
    return (float)sqrt(sum_squares);
}

/// Makes random values from Tensor::randn, times `scale`, as a CPU list: the same numbers
static std::vector<float> random_values(const Shape &shape, float scale = 1.0f)
{
    std::vector<float> numbers = Tensor::randn(shape).tolist();
    for (float &number : numbers)
        number *= scale;
    return numbers;
}

/// Gives every value of my AdamW's parameters, one tensor after the other, to compare
static std::vector<float> all_my_adamw_values(const AdamW &my_adamw)
{
    std::vector<float> values;
    for (int k = 0; k < my_adamw.num_params(); k++)
    {
        std::vector<float> tensor_values = my_adamw.param(k).tolist();
        values.insert(values.end(), tensor_values.begin(), tensor_values.end());
    }
    return values;
}

// ── helpers: PyTorch's side ───────────────────────────────────────────────────

/// Makes a new GPU tensor of this shape from values in row order: the tests' CPU lists
static torch::Tensor to_gpu(const std::vector<float> &row_values, torch::IntArrayRef shape)
{
    return torch::tensor(row_values).reshape(shape).to(torch::kCUDA);
}

/// Makes one PyTorch parameter on the GPU from a test's values and shape. requires_grad:
static torch::Tensor pytorch_param(const std::vector<float> &values, const Shape &shape)
{
    std::vector<int64_t> sizes(shape.begin(), shape.end());
    return to_gpu(values, sizes).requires_grad_(true);
}

/// Makes AdamW's m (or v) for each of these parameters: zeros of the same shape, as
static std::vector<torch::Tensor> zeros_like_each(const std::vector<torch::Tensor> &params)
{
    std::vector<torch::Tensor> zeros;
    for (const torch::Tensor &param : params)
        zeros.push_back(torch::zeros_like(param));
    return zeros;
}

/// Makes a step count of 0 for each of `count` parameters: one float on the GPU each, as
static std::vector<torch::Tensor> zero_step_counts(size_t count)
{
    std::vector<torch::Tensor> steps;
    for (size_t k = 0; k < count; k++)
        steps.push_back(torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)));
    return steps;
}

/// Does one step of PyTorch's fused AdamW on one param group, from the gradient each
static void pytorch_adamw_step(std::vector<torch::Tensor> &params, std::vector<torch::Tensor> &m,
                               std::vector<torch::Tensor> &v, std::vector<torch::Tensor> &steps,
                               double weight_decay, const AdamWConfig &config)
{
    torch::NoGradGuard no_grad; // the update is not part of the graph, as in Python
    std::vector<torch::Tensor> grads;
    for (const torch::Tensor &param : params)
        grads.push_back(param.grad());
    at::_foreach_add_(steps, at::Scalar(1.0)); // step t = t + 1
    at::_fused_adamw_(params, grads, m, v, at::TensorList(), // no amsgrad state
                      steps, config.learning_rate, config.beta1, config.beta2, weight_decay, config.eps,
                      /*amsgrad=*/false, /*maximize=*/false);
}

/// Gives a PyTorch tensor's values in row order on the CPU, to compare with mine.
static std::vector<float> values_of(const torch::Tensor &tensor)
{
    torch::Tensor flat = tensor.detach().to(torch::kCPU).reshape(-1).contiguous();
    return std::vector<float>(flat.data_ptr<float>(), flat.data_ptr<float>() + flat.numel());
}

/// Gives every value of these PyTorch tensors, one tensor after the other.
static std::vector<float> all_values_of(const std::vector<torch::Tensor> &tensors)
{
    std::vector<float> values;
    for (const torch::Tensor &tensor : tensors)
    {
        std::vector<float> tensor_values = values_of(tensor);
        values.insert(values.end(), tensor_values.begin(), tensor_values.end());
    }
    return values;
}

// ══ part 1: my AdamW against PyTorch's fused AdamW ════════════════════════════

static void one_step_matches_pytorch(void)
{
    // Expect: one step on a 2x2 weight {0.5, -1, 2, 0} with gradient {0.1, -0.2, 0.3, 0.05},
    // lr 0.01, weight decay 0.1: each of my AdamW's 4 values equals PyTorch's (about 0.4895,
    // -0.989, 1.988, -0.01).
    AdamWConfig config;
    config.learning_rate = 1e-2;
    std::vector<float> weight_values = {0.5f, -1.0f, 2.0f, 0.0f};
    std::vector<float> weight_grad = {0.1f, -0.2f, 0.3f, 0.05f};
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    // PyTorch's: the weight, and its m, v and step count, all kept here between steps
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {2, 2})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    // 1. my AdamW, step 1
    my_adamw.grad(0).copy_from_cpu(weight_grad.data());
    my_adamw.step();
    // 2. PyTorch, the same gradient
    pytorch_params[0].mutable_grad() = to_gpu(weight_grad, pytorch_params[0].sizes());
    pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
    // 3. compare
    std::vector<float> my_weight = my_adamw.param(0).tolist();
    std::vector<float> pytorch_weight = values_of(pytorch_params[0]);
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(my_weight[i], pytorch_weight[i], 1e-6);
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(pytorch_weight, weight_values, 1e-3);
}

static void many_steps_match_pytorch(void)
{
    // Expect: 10 steps on a 2x2 weight, a new gradient each step (some values 0), lr 0.003:
    // my AdamW equals PyTorch's after every step, so m and v carry over the same way.
    static float grads_per_step[10][4] = {
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
    AdamWConfig config;
    config.learning_rate = 3e-3;
    std::vector<float> weight_values = {0.8f, -0.5f, 1.2f, 0.3f};
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {2, 2})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    for (int t = 1; t <= 10; t++)
    {
        std::vector<float> step_grad(grads_per_step[t - 1], grads_per_step[t - 1] + 4);
        // 1. my AdamW, step t
        my_adamw.grad(0).copy_from_cpu(step_grad.data());
        my_adamw.step();
        // 2. PyTorch, the same gradient
        pytorch_params[0].mutable_grad() = to_gpu(step_grad, pytorch_params[0].sizes());
        pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
        // 3. compare
        ASSERT_NEAR(my_adamw.param(0).tolist(), values_of(pytorch_params[0]), TOLERANCE);
    }
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(values_of(pytorch_params[0]), weight_values, 1e-3);
}

static void large_random_weight_matches_pytorch(void)
{
    // Expect: a 256x256 random weight, 100 steps of random gradients, every 7th of them tiny
    // (x 1e-6, where eps matters): my AdamW equals PyTorch's after every step.
    Tensor::manual_seed(0);
    AdamWConfig config;
    std::vector<float> weight_values = random_values({256, 256});
    AdamW my_adamw({{256, 256}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {256, 256})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    for (int t = 1; t <= 100; t++)
    {
        std::vector<float> step_grad = random_values({256, 256}, t % 7 == 1 ? 1e-6f : 1.0f);
        // 1. my AdamW, step t
        my_adamw.grad(0).copy_from_cpu(step_grad.data());
        my_adamw.step();
        // 2. PyTorch, the same gradient
        pytorch_params[0].mutable_grad() = to_gpu(step_grad, pytorch_params[0].sizes());
        pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
        // 3. compare
        ASSERT_NEAR(my_adamw.param(0).tolist(), values_of(pytorch_params[0]), TOLERANCE);
    }
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(values_of(pytorch_params[0]), weight_values, 1e-3);
}

static void bias_matches_pytorch_without_decay(void)
{
    // Expect: weight decay 0.1, but a bias (1-D) is in PyTorch's group without weight decay,
    // and my AdamW does not decay it either: one step, each of its 4 values equals PyTorch's.
    AdamWConfig config;
    config.learning_rate = 5e-2;
    std::vector<float> bias_values = {0.7f, -0.7f, 1.4f, 0.0f};
    std::vector<float> bias_grad = {0.2f, -0.2f, 0.1f, 0.3f};
    AdamW my_adamw({{4}}, config);
    my_adamw.param(0).copy_from_cpu(bias_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(bias_values, {4})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    // 1. my AdamW, step 1
    my_adamw.grad(0).copy_from_cpu(bias_grad.data());
    my_adamw.step();
    // 2. PyTorch, the same gradient; a bias is in the group with weight decay 0
    pytorch_params[0].mutable_grad() = to_gpu(bias_grad, pytorch_params[0].sizes());
    pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, 0.0, config);
    // 3. compare
    std::vector<float> my_bias = my_adamw.param(0).tolist();
    std::vector<float> pytorch_bias = values_of(pytorch_params[0]);
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(my_bias[i], pytorch_bias[i], 1e-6);
    // PyTorch really trained: its bias moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(pytorch_bias, bias_values, 1e-3);
}

static void mixed_parameters_match_pytorch(void)
{
    // Expect: 2-D, 3-D and 1-D tensors, {3, 4}, {2, 2, 3}, {4}, {8} (matrices first, as
    // AdamW needs), 20 steps with weight decay: every tensor equals PyTorch's after every
    // step. (C#'s sizes 5 and 7 became 4 and 8: the kernel needs a multiple of 4.)
    Tensor::manual_seed(1);
    AdamWConfig config;
    std::vector<Shape> param_shapes = {{3, 4}, {2, 2, 3}, {4}, {8}};
    std::vector<std::vector<float>> param_values;
    for (const Shape &shape : param_shapes)
        param_values.push_back(random_values(shape));
    AdamW my_adamw(param_shapes, config);
    for (int k = 0; k < 4; k++)
        my_adamw.param(k).copy_from_cpu(param_values[k].data());
    // PyTorch's: the 4 tensors, then its two param groups of the same tensors, each with
    // its own m, v and step counts
    std::vector<torch::Tensor> pytorch_params;
    for (int k = 0; k < 4; k++)
        pytorch_params.push_back(pytorch_param(param_values[k], param_shapes[k]));
    std::vector<torch::Tensor> pytorch_matrices = {pytorch_params[0], pytorch_params[1]};
    std::vector<torch::Tensor> matrices_m = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_v = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_steps = zero_step_counts(pytorch_matrices.size());
    std::vector<torch::Tensor> pytorch_rest = {pytorch_params[2], pytorch_params[3]};
    std::vector<torch::Tensor> rest_m = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_v = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_steps = zero_step_counts(pytorch_rest.size());

    for (int t = 1; t <= 20; t++)
    {
        std::vector<std::vector<float>> param_grads;
        for (const Shape &shape : param_shapes)
            param_grads.push_back(random_values(shape));
        // 1. my AdamW, step t
        for (int k = 0; k < 4; k++)
            my_adamw.grad(k).copy_from_cpu(param_grads[k].data());
        my_adamw.step();
        // 2. PyTorch, the same gradients; one step for each param group
        for (int k = 0; k < 4; k++)
            pytorch_params[k].mutable_grad() = to_gpu(param_grads[k], pytorch_params[k].sizes());
        pytorch_adamw_step(pytorch_matrices, matrices_m, matrices_v, matrices_steps, config.weight_decay, config);
        pytorch_adamw_step(pytorch_rest, rest_m, rest_v, rest_steps, 0.0, config);
        // 3. compare, tensor by tensor
        for (int k = 0; k < 4; k++)
            ASSERT_NEAR(my_adamw.param(k).tolist(), values_of(pytorch_params[k]), TOLERANCE);
    }
    // PyTorch really trained each tensor: each moved far more than the 1e-5 we allow between the sides.
    for (int k = 0; k < 4; k++)
        ASSERT_FAR(values_of(pytorch_params[k]), param_values[k], 1e-3);
}

static void clip_scales_large_gradients_down_to_max_norm(void)
{
    // Expect: two steps on a 2x2 weight with max_norm 1. Step 1's gradient (norm 0.38) is not
    // clipped; step 2's {1, 2, 3, 4} has norm sqrt(30) = 5.48 > 1, so both sides divide it by
    // 5.48 first. Step 2 shows it: at step 1 Adam ignores the gradient's size. My AdamW equals
    // PyTorch's after each step.
    AdamWConfig config;
    config.learning_rate = 1e-2;
    config.max_norm = 1.0;
    std::vector<float> weight_values = {0.5f, -1.0f, 2.0f, 0.25f};
    std::vector<float> grads_per_step[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {1.0f, 2.0f, 3.0f, 4.0f}};
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {2, 2})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    for (int t = 1; t <= 2; t++)
    {
        const std::vector<float> &step_grad = grads_per_step[t - 1];
        // 1. my AdamW, step t: the gradient, and its norm for the clipping
        my_adamw.grad(0).copy_from_cpu(step_grad.data());
        float grad_norm = cpu_grad_norm({step_grad});
        my_adamw.grad_norm().copy_from_cpu(&grad_norm);
        my_adamw.step();
        // 2. PyTorch, the same gradient, clipped by PyTorch's clip_grad_norm_
        pytorch_params[0].mutable_grad() = to_gpu(step_grad, pytorch_params[0].sizes());
        torch::nn::utils::clip_grad_norm_(pytorch_params, config.max_norm);
        pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
        // 3. compare
        ASSERT_NEAR(my_adamw.param(0).tolist(), values_of(pytorch_params[0]), TOLERANCE);
    }
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(values_of(pytorch_params[0]), weight_values, 1e-3);
}

static void clip_leaves_small_gradients_alone(void)
{
    // Expect: the same two steps, but step 2's gradient {0.3, 0, 0.4, 0} has norm 0.5 < 1:
    // not clipped. My AdamW equals PyTorch's after each step.
    AdamWConfig config;
    config.learning_rate = 1e-2;
    config.max_norm = 1.0;
    std::vector<float> weight_values = {0.5f, -1.0f, 2.0f, 0.25f};
    std::vector<float> grads_per_step[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {0.3f, 0.0f, 0.4f, 0.0f}};
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {2, 2})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    for (int t = 1; t <= 2; t++)
    {
        const std::vector<float> &step_grad = grads_per_step[t - 1];
        // 1. my AdamW, step t: the gradient, and its norm for the clipping
        my_adamw.grad(0).copy_from_cpu(step_grad.data());
        float grad_norm = cpu_grad_norm({step_grad});
        my_adamw.grad_norm().copy_from_cpu(&grad_norm);
        my_adamw.step();
        // 2. PyTorch, the same gradient, clipped by PyTorch's clip_grad_norm_
        pytorch_params[0].mutable_grad() = to_gpu(step_grad, pytorch_params[0].sizes());
        torch::nn::utils::clip_grad_norm_(pytorch_params, config.max_norm);
        pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
        // 3. compare
        ASSERT_NEAR(my_adamw.param(0).tolist(), values_of(pytorch_params[0]), TOLERANCE);
    }
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(values_of(pytorch_params[0]), weight_values, 1e-3);
}

static void max_norm_0_turns_clipping_off(void)
{
    // Expect: (new) step 2's gradient has norm sqrt(30), but max_norm is 0: PyTorch does not
    // clip, and neither may our kernel, though grad_norm holds 5.48. My AdamW equals PyTorch's
    // after each step.
    AdamWConfig config;
    config.learning_rate = 1e-2;
    config.max_norm = 0.0;
    std::vector<float> weight_values = {0.5f, -1.0f, 2.0f, 0.25f};
    std::vector<float> grads_per_step[2] = {{0.1f, -0.2f, 0.3f, 0.05f}, {1.0f, 2.0f, 3.0f, 4.0f}};
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).copy_from_cpu(weight_values.data());
    std::vector<torch::Tensor> pytorch_params = {pytorch_param(weight_values, {2, 2})};
    std::vector<torch::Tensor> pytorch_m = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_v = zeros_like_each(pytorch_params);
    std::vector<torch::Tensor> pytorch_steps = zero_step_counts(pytorch_params.size());

    for (int t = 1; t <= 2; t++)
    {
        const std::vector<float> &step_grad = grads_per_step[t - 1];
        // 1. my AdamW, step t: the gradient, and its norm, which max_norm 0 must ignore
        my_adamw.grad(0).copy_from_cpu(step_grad.data());
        float grad_norm = cpu_grad_norm({step_grad});
        my_adamw.grad_norm().copy_from_cpu(&grad_norm);
        my_adamw.step();
        // 2. PyTorch, the same gradient, not clipped: max_norm 0 means no clip_grad_norm_
        pytorch_params[0].mutable_grad() = to_gpu(step_grad, pytorch_params[0].sizes());
        pytorch_adamw_step(pytorch_params, pytorch_m, pytorch_v, pytorch_steps, config.weight_decay, config);
        // 3. compare
        ASSERT_NEAR(my_adamw.param(0).tolist(), values_of(pytorch_params[0]), TOLERANCE);
    }
    // PyTorch really trained: its weight moved far more than the 1e-5 we allow between the sides.
    ASSERT_FAR(values_of(pytorch_params[0]), weight_values, 1e-3);
}

static void whole_model_matches_pytorch_with_clipping(void)
{
    // Expect: the parameters of a tiny GPT-2 of 1 layer, 16 wide (16 tensors, 4464 values),
    // 20 steps with clipping at 1: odd steps get big gradients (norm about 67, clipped), even
    // steps small ones (about 0.07, left alone). After every step, all 4464 of my AdamW's
    // values (every tensor, one after the other, as PyTorch's tensors are) equal PyTorch's.
    Tensor::manual_seed(0);
    AdamWConfig config;
    config.learning_rate = 1e-2;
    config.max_norm = 1.0;
    std::vector<Shape> param_shapes = {
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
    int num_params = (int)param_shapes.size();
    int num_matrices = 6;
    std::vector<std::vector<float>> param_values;
    for (const Shape &shape : param_shapes)
        param_values.push_back(random_values(shape, 0.02f));
    AdamW my_adamw(param_shapes, config);
    for (int k = 0; k < num_params; k++)
        my_adamw.param(k).copy_from_cpu(param_values[k].data());
    // PyTorch's: the 16 tensors, then its two param groups of the same tensors (the 6
    // matrices, the other 10), each with its own m, v and step counts
    std::vector<torch::Tensor> pytorch_params;
    for (int k = 0; k < num_params; k++)
        pytorch_params.push_back(pytorch_param(param_values[k], param_shapes[k]));
    std::vector<torch::Tensor> pytorch_matrices(pytorch_params.begin(), pytorch_params.begin() + num_matrices);
    std::vector<torch::Tensor> matrices_m = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_v = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_steps = zero_step_counts(pytorch_matrices.size());
    std::vector<torch::Tensor> pytorch_rest(pytorch_params.begin() + num_matrices, pytorch_params.end());
    std::vector<torch::Tensor> rest_m = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_v = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_steps = zero_step_counts(pytorch_rest.size());

    for (int t = 1; t <= 20; t++)
    {
        std::vector<std::vector<float>> param_grads;
        for (const Shape &shape : param_shapes)
            param_grads.push_back(random_values(shape, t % 2 == 1 ? 1.0f : 1e-3f));
        // 1. my AdamW, step t: the gradients, and their norm for the clipping
        for (int k = 0; k < num_params; k++)
            my_adamw.grad(k).copy_from_cpu(param_grads[k].data());
        float grad_norm = cpu_grad_norm(param_grads);
        my_adamw.grad_norm().copy_from_cpu(&grad_norm);
        my_adamw.step();
        // 2. PyTorch, the same gradients, clipped by PyTorch's clip_grad_norm_ (all 16
        //    together); one step for each param group
        for (int k = 0; k < num_params; k++)
            pytorch_params[k].mutable_grad() = to_gpu(param_grads[k], pytorch_params[k].sizes());
        torch::nn::utils::clip_grad_norm_(pytorch_params, config.max_norm);
        pytorch_adamw_step(pytorch_matrices, matrices_m, matrices_v, matrices_steps, config.weight_decay, config);
        pytorch_adamw_step(pytorch_rest, rest_m, rest_v, rest_steps, 0.0, config);
        // 3. compare every value at once
        ASSERT_NEAR(all_my_adamw_values(my_adamw), all_values_of(pytorch_params), TOLERANCE);
    }
    // PyTorch really trained the model: its values moved far more than the 1e-5 we allow.
    std::vector<float> all_start_values;
    for (const std::vector<float> &tensor_values : param_values)
        all_start_values.insert(all_start_values.end(), tensor_values.begin(), tensor_values.end());
    ASSERT_FAR(all_values_of(pytorch_params), all_start_values, 1e-3);
}

// ══ time: my AdamW against PyTorch's fused AdamW ══════════════════════════════

/// Measures how long `run` takes on the GPU, in milliseconds: a mark on the GPU's queue
/// before it and one after; the GPU writes down the time as it reaches each.
template <typename Run>
static float gpu_milliseconds(Run run)
{
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    run();
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end)); // wait until the GPU has passed the second mark
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    return milliseconds;
}

/// Times 20 AdamW updates on a matrix and a bias of these shapes, both on the GPU, and
/// prints a table under `test`'s name:
///
///     my AdamW   AdamW::step: one launch of adamw_update
///     PyTorch    fused: per param group (2 here), +1 on the step counts, then _fused_adamw_
///
/// The update does not care about layers, only about how many values there are and the 2
/// groups (matrices, 1-D), so a matrix and a bias are enough. Only the update is timed:
/// the gradients are set once, before (PyTorch's clipped by clip_grad_norm_), and all 20
/// updates use them. Each update reads p, g, m, v and writes p, m, v: 7 x 4 bytes per
/// value. Checks at the end that my AdamW still equals PyTorch's: both did the same work.
static void time_20_updates(const char *test, const std::vector<Shape> &param_shapes)
{
    Tensor::manual_seed(0);
    AdamWConfig config;
    config.max_norm = 1.0;
    std::vector<std::vector<float>> param_values, param_grads;
    for (const Shape &shape : param_shapes)
    {
        param_values.push_back(random_values(shape, 0.02f));
        param_grads.push_back(random_values(shape));
    }
    // my AdamW, its gradients and their norm
    AdamW my_adamw(param_shapes, config);
    for (int k = 0; k < 2; k++)
    {
        my_adamw.param(k).copy_from_cpu(param_values[k].data());
        my_adamw.grad(k).copy_from_cpu(param_grads[k].data());
    }
    float grad_norm = cpu_grad_norm(param_grads);
    my_adamw.grad_norm().copy_from_cpu(&grad_norm);
    // PyTorch's: the matrix and the bias, each its own param group, their gradients clipped
    std::vector<torch::Tensor> pytorch_params;
    for (int k = 0; k < 2; k++)
    {
        pytorch_params.push_back(pytorch_param(param_values[k], param_shapes[k]));
        pytorch_params[k].mutable_grad() = to_gpu(param_grads[k], pytorch_params[k].sizes());
    }
    torch::nn::utils::clip_grad_norm_(pytorch_params, config.max_norm);
    std::vector<torch::Tensor> pytorch_matrices = {pytorch_params[0]};
    std::vector<torch::Tensor> matrices_m = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_v = zeros_like_each(pytorch_matrices);
    std::vector<torch::Tensor> matrices_steps = zero_step_counts(1);
    std::vector<torch::Tensor> pytorch_rest = {pytorch_params[1]};
    std::vector<torch::Tensor> rest_m = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_v = zeros_like_each(pytorch_rest);
    std::vector<torch::Tensor> rest_steps = zero_step_counts(1);

    // Step 1 does first-time work (CUDA loads each side's kernels), so the average is of 2..20.
    print_test_name(test);
    printf("    time of one update, in ms:   step   my AdamW   PyTorch fused\n");
    double my_total = 0.0, pytorch_total = 0.0;
    for (int t = 1; t <= 20; t++)
    {
        float my_ms = gpu_milliseconds([&] { my_adamw.step(); });
        float pytorch_ms = gpu_milliseconds([&] {
            pytorch_adamw_step(pytorch_matrices, matrices_m, matrices_v, matrices_steps, config.weight_decay, config);
            pytorch_adamw_step(pytorch_rest, rest_m, rest_v, rest_steps, 0.0, config);
        });
        printf("                                 %4d   %8.3f   %13.3f\n", t, my_ms, pytorch_ms);
        if (t == 1)
            continue;
        my_total += my_ms;
        pytorch_total += pytorch_ms;
    }
    printf("    average of steps 2..20:             %8.3f   %13.3f   PyTorch takes %.2fx as long\n",
           my_total / 19, pytorch_total / 19, pytorch_total / my_total);
    ASSERT_NEAR(all_my_adamw_values(my_adamw), all_values_of(pytorch_params), TOLERANCE);
}

static void time_20_updates_on_gpt2_small_size(void)
{
    // Prints the times on as many values as GPT-2 small has: 162,030 x 768 + 768 =
    // 124,439,040 + 768 = 124,439,808. Each update moves 7 x 124,439,808 x 4 bytes = 3.5 GB:
    // at a T4's 320 GB/s, at least 10.9 ms.
    time_20_updates(__func__, {{162030, 768}, {768}});
}

static void time_20_updates_on_csharp_benchmark_size(void)
{
    // Prints the times on as many values as the C# branch's AdamW benchmark had (GPT-2's
    // 148 tensors, 128 wide): 69,917 x 128 + 128 = 8,949,376 + 128 = 8,949,504. Each update
    // moves 7 x 8,949,504 x 4 bytes = 251 MB: at a T4's 320 GB/s, at least 0.78 ms. The C#
    // numbers to compare, on a T4: PyTorch fused 1.95 ms, the C# flat-buffer AdamW 3.64 ms
    // (timed on the CPU's clock, with 148 tensors instead of 2).
    time_20_updates(__func__, {{69917, 128}, {128}});
}

// ══ part 2: rules worked out by hand ══════════════════════════════════════════
// My AdamW starts with every value 0, gradients too: a test that wants 0 gradients leaves
// them as they are.

static void first_step_moves_each_value_by_about_the_learning_rate(void)
{
    // Expect: at t = 1, m̂ = g and v̂ = g², so the step is lr g / |g| = ±lr, whatever the size
    // of g. A 2x2 of 1s, gradients {1000, 0.001, -5, -0.3}, lr 0.05, no weight decay:
    // 1 - 0.05 where g > 0, 1 + 0.05 where g < 0.
    AdamWConfig config;
    config.learning_rate = 0.05;
    config.weight_decay = 0.0;
    AdamW my_adamw({{2, 2}}, config);
    my_adamw.param(0).fill_(1);
    std::vector<float> weight_grad = {1000.0f, 0.001f, -5.0f, -0.3f};
    my_adamw.grad(0).copy_from_cpu(weight_grad.data());
    my_adamw.step();
    std::vector<float> weight = my_adamw.param(0).tolist();
    std::vector<float> expected = {0.95f, 0.95f, 1.05f, 1.05f};
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(weight[i], expected[i], 1e-4);
}

static void weight_decay_shrinks_matrices_but_not_biases(void)
{
    // Expect: every gradient 0, so the only change is weight decay: the 2x2 weight
    // {2, -4, 6, 8} x (1 - 0.5 x 0.1) = x 0.95, the bias {2, -4, 6, 8} not at all.
    AdamWConfig config;
    config.learning_rate = 0.5;
    config.weight_decay = 0.1;
    AdamW my_adamw({{2, 2}, {4}}, config);
    std::vector<float> start_values = {2.0f, -4.0f, 6.0f, 8.0f};
    my_adamw.param(0).copy_from_cpu(start_values.data()); // weight
    my_adamw.param(1).copy_from_cpu(start_values.data()); // bias
    my_adamw.step();                                      // gradients: all 0
    std::vector<float> weight = my_adamw.param(0).tolist();
    std::vector<float> expected = {1.9f, -3.8f, 5.7f, 7.6f};
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(weight[i], expected[i], 1e-6);
    ASSERT_EQ(my_adamw.param(1).tolist(), start_values);
}

static void a_parameter_the_loss_did_not_use_only_decays(void)
{
    // Expect: a 0 gradient keeps m at 0, so the Adam part is 0 and only weight decay is left:
    // x (1 - 0.1 x 0.1) = x 0.99. unused {5, 6, 7, 8} -> {4.95, 5.94, 6.93, 7.92}; used
    // {1, 2, 3, 4} with gradient 1 also moves by -lr = -0.1: {0.89, 1.88, 2.87, 3.86}.
    AdamWConfig config;
    config.learning_rate = 0.1;
    config.weight_decay = 0.1;
    AdamW my_adamw({{2, 2}, {2, 2}}, config);
    std::vector<float> used_start = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> unused_start = {5.0f, 6.0f, 7.0f, 8.0f};
    my_adamw.param(0).copy_from_cpu(used_start.data());
    my_adamw.param(1).copy_from_cpu(unused_start.data());
    my_adamw.grad(0).fill_(1); // the unused one's gradient stays 0
    my_adamw.step();
    std::vector<float> used = my_adamw.param(0).tolist();
    std::vector<float> unused = my_adamw.param(1).tolist();
    std::vector<float> expected_used = {0.89f, 1.88f, 2.87f, 3.86f};
    std::vector<float> expected_unused = {4.95f, 5.94f, 6.93f, 7.92f};
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(unused[i], expected_unused[i], 1e-5);
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(used[i], expected_used[i], 1e-5);
}

static void weight_decay_follows_the_shape_not_the_order(void)
{
    // Expect: weight {1, 4}, weight {2, 2}, bias {4}, bias {8}, all holding 2, 4, 6, ...;
    // no gradients, lr 0.5, weight decay 0.1: the weights x 0.95, the biases not moved. The
    // shape decides: {1, 4} has one row but is 2-D, a matrix; {8} has as many values as a
    // 2x4 but is 1-D. (C# gave them mixed; AdamW needs the matrices first, so here they are.)
    AdamWConfig config;
    config.learning_rate = 0.5;
    config.weight_decay = 0.1;
    std::vector<float> four_values = {2, 4, 6, 8};
    std::vector<float> eight_values = {2, 4, 6, 8, 10, 12, 14, 16};
    AdamW my_adamw({{1, 4}, {2, 2}, {4}, {8}}, config);
    my_adamw.param(0).copy_from_cpu(four_values.data());
    my_adamw.param(1).copy_from_cpu(four_values.data());
    my_adamw.param(2).copy_from_cpu(four_values.data());
    my_adamw.param(3).copy_from_cpu(eight_values.data());
    my_adamw.step(); // gradients: all 0
    std::vector<float> weight1 = my_adamw.param(0).tolist();
    std::vector<float> weight2 = my_adamw.param(1).tolist();
    std::vector<float> shrunk = {1.9f, 3.8f, 5.7f, 7.6f};
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(weight1[i], shrunk[i], 1e-6);
    for (size_t i = 0; i < 4; i++)
        ASSERT_NEAR(weight2[i], shrunk[i], 1e-6);
    ASSERT_EQ(my_adamw.param(2).tolist(), four_values);
    ASSERT_EQ(my_adamw.param(3).tolist(), eight_values);
}

static void only_the_values_before_decayed_count_decay(void)
{
    // Expect: (new) the kernel's border. 8 values {2, 4, 6, 8, 2, 4, 6, 8}, decayed_count 4,
    // no gradients, lr 0.5, weight decay 0.1: the first 4 x 0.95 = {1.9, 3.8, 5.7, 7.6}, the
    // last 4 not moved.
    AdamWConfig config;
    AdamW my_adamw({{8}}, config);
    std::vector<float> start_values = {2, 4, 6, 8, 2, 4, 6, 8};
    my_adamw.param(0).copy_from_cpu(start_values.data());
    const Tensor &param = my_adamw.param(0);
    const Tensor &grad = my_adamw.grad(0);
    const Tensor &m = my_adamw.m(0);
    const Tensor &v = my_adamw.v(0);
    adamw_update(param.data(), grad.data(), m.data(), v.data(), 8, 4, my_adamw.grad_norm().data(), 0.0f, 0.5f, 0.9f,
                 0.95f, 1e-8f, 0.1f, 1);
    std::vector<float> param_values = param.tolist();
    std::vector<float> expected = {1.9f, 3.8f, 5.7f, 7.6f, 2, 4, 6, 8};
    for (size_t i = 0; i < 8; i++)
        ASSERT_NEAR(param_values[i], expected[i], 1e-6);
}

static void nothing_past_a_tensors_end_changes(void)
{
    // Expect: (new) a launch covers 256 threads x 4 values, far more than a tensor of 8.
    // adamw_update on tensor 0 of two 8-value tensors side by side leaves tensor 1's p, m
    // and v as they were: p {10..17}, m and v all 0.
    AdamWConfig config;
    AdamW my_adamw({{8}, {8}}, config);
    std::vector<float> tensor1_values = {10, 11, 12, 13, 14, 15, 16, 17};
    my_adamw.param(0).copy_from_cpu(sample_data(8).data());
    my_adamw.param(1).copy_from_cpu(tensor1_values.data());
    my_adamw.grad(0).fill_(1);
    my_adamw.grad(1).fill_(1);
    const Tensor &param = my_adamw.param(0); // tensor 0 only
    const Tensor &grad = my_adamw.grad(0);
    const Tensor &m = my_adamw.m(0);
    const Tensor &v = my_adamw.v(0);
    adamw_update(param.data(), grad.data(), m.data(), v.data(), param.numel(), 0, my_adamw.grad_norm().data(), 0.0f,
                 0.1f, config.beta1, config.beta2, config.eps, 0.0f, 1);
    ASSERT_EQ(my_adamw.param(1).tolist(), tensor1_values); // tensor 1: untouched
    ASSERT_EQ(my_adamw.m(1).tolist(), std::vector<float>(8, 0.0f));
    ASSERT_EQ(my_adamw.v(1).tolist(), std::vector<float>(8, 0.0f));
}

static void step_leaves_the_gradients_as_they_are(void)
{
    // Expect: C#'s Step zeroed the gradients (StepLeavesTheGradientsAtZero); our kernel only
    // reads them (const float*), so after a step with clipping they are still {1, 2, 3, 4}
    // and {5, 6, 7, 8}. The trainer zeroes them itself before the next backward.
    AdamWConfig config;
    config.learning_rate = 0.1;
    config.max_norm = 1.0;
    AdamW my_adamw({{2, 2}, {4}}, config);
    std::vector<std::vector<float>> param_grads = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    my_adamw.grad(0).copy_from_cpu(param_grads[0].data());
    my_adamw.grad(1).copy_from_cpu(param_grads[1].data());
    float grad_norm = cpu_grad_norm(param_grads); // sqrt(204) = 14.3 > 1: clipped
    my_adamw.grad_norm().copy_from_cpu(&grad_norm);
    my_adamw.step();
    ASSERT_EQ(my_adamw.grad(0).tolist(), param_grads[0]);
    ASSERT_EQ(my_adamw.grad(1).tolist(), param_grads[1]);
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
    AdamWConfig config;
    AdamW my_adamw({{2, 4}, {1, 4}, {8}, {4}}, config);
    ASSERT_EQ(my_adamw.decayed_count(), (size_t)12);       // 2 x 4 + 1 x 4
    ASSERT_EQ(my_adamw.buffer().num_groups(), 5);          // params, grads, m, v, grad_norm
    ASSERT_EQ(my_adamw.buffer().num_tensors(), 17);        // 4 x 4 + 1
    ASSERT_EQ(my_adamw.buffer().numel(), (size_t)97);      // 4 x 24 + 1
    ASSERT_EQ(my_adamw.param(0).shape(), Shape({2, 4}));
    ASSERT_EQ(my_adamw.grad(0).data() == my_adamw.param(0).data() + 24, true);
    ASSERT_EQ((size_t)my_adamw.m(0).data() % 16, (size_t)0);
    ASSERT_EQ((size_t)my_adamw.v(0).data() % 16, (size_t)0);
    for (int k = 0; k < 4; k++)
    {
        ASSERT_EQ(my_adamw.m(k).tolist(), std::vector<float>(my_adamw.m(k).numel(), 0.0f));
        ASSERT_EQ(my_adamw.v(k).tolist(), std::vector<float>(my_adamw.v(k).numel(), 0.0f));
    }
    ASSERT_EQ(my_adamw.steps_taken(), 0);
    my_adamw.step(); // gradients: all 0
    my_adamw.step();
    ASSERT_EQ(my_adamw.steps_taken(), 2);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    one_step_matches_pytorch();
    many_steps_match_pytorch();
    large_random_weight_matches_pytorch();
    bias_matches_pytorch_without_decay();
    mixed_parameters_match_pytorch();
    clip_scales_large_gradients_down_to_max_norm();
    clip_leaves_small_gradients_alone();
    max_norm_0_turns_clipping_off();
    whole_model_matches_pytorch_with_clipping();
    time_20_updates_on_gpt2_small_size();
    time_20_updates_on_csharp_benchmark_size();

    first_step_moves_each_value_by_about_the_learning_rate();
    weight_decay_shrinks_matrices_but_not_biases();
    a_parameter_the_loss_did_not_use_only_decays();
    weight_decay_follows_the_shape_not_the_order();
    only_the_values_before_decayed_count_decay();
    nothing_past_a_tensors_end_changes();
    step_leaves_the_gradients_as_they_are();
    adamw_counts_the_matrices_and_the_steps();

    printf("\nall tests passed\n");
    return 0;
}
