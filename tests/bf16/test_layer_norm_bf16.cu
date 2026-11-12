// test_layer_norm_bf16.cu — tests for llmc/kernels/bf16/layer_norm_bf16.cuh (layer_norm_bf16_forward,
// layer_norm_bf16_backward): LayerNorm in bfloat16, 8 values a lane (one 16-byte load). Every check prints its test
// case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks both against PyTorch's layer_norm (with autograd for the backward) and against fp32/layer_norm.cuh's
// bfloat16 path (4 values a lane); part 2 checks numbers worked out by hand, the cases of tests/fp32/test_layer_norm.cu
// on rows of 8; part 3 checks that each kernel's registers fit its block, and other block sizes. Any GPU. The timings
// are in tunes/tune_layer_norm_bf16.cu.
#include <torch/types.h>              // torch::layer_norm, backward(); libtorch before any CUDA header
#include "../test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/fp32/layer_norm.cuh"      // the 4-value path, to compare with
#include "../../llmc/kernels/bf16/layer_norm_bf16.cuh"

static const double TOLERANCE = 1e-5; // mean and rstd (float) against PyTorch's, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so each rounding
// moves a value by up to 2^-8, about 0.4% of it. weight_grad and bias_grad are rounded once per block that adds onto
// them, 8 times here.
static const double BFLOAT16_TOLERANCE = 2e-2;
// Against the 4-value path: the same math, a row's sums added in another order, so a value can land one bfloat16
// rounding (2^-8 of it) away.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
// Between block sizes: the backward always launches LAYER_NORM_BF16_BACKWARD_BLOCKS (80) blocks, and each block with
// rows rounds its sum into weight_grad and bias_grad once; more blocks with rows, more roundings (as in
// tests/fp32/test_layer_norm.cu).
static const double BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_EMBED_DIM 768            // GPT-2 small's width
#define SMALL_EMBED_DIM LAYER_NORM_BF16_VALUES_PER_LOAD   // part 2's rows: one load, 8 values

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded to
/// bfloat16.
static Tensor bf16_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values (bfloat16 turned into floats exactly).
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);         // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Gives every value of my tensor as a CPU list of floats (bfloat16 turned into floats, exactly).
static std::vector<float> floats_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Gives the 8 values of a part-2 row: `low_val` four times, then `high_val` four times.
static std::vector<float> row_of(float low_val, float high_val)
{
    return {low_val, low_val, low_val, low_val, high_val, high_val, high_val, high_val};
}

/// One LayerNorm's tensors: the inputs, the forward's results and the gradients, bfloat16 (mean and rstd float),
/// the results and gradients all 0 at first.
struct LayerNormTensors
{
    Tensor input_tensor;
    Tensor weight_tensor;
    Tensor bias_tensor;
    Tensor output_grad_tensor;
    Tensor output_tensor;
    Tensor mean_tensor;
    Tensor rstd_tensor;
    Tensor input_grad_tensor;
    Tensor weight_grad_tensor;
    Tensor bias_grad_tensor;

    /// Gives a copy with the same inputs and results and gradients of its own, all 0.
    LayerNormTensors with_fresh_results(void) const
    {
        LayerNormTensors tensors_val = *this;
        tensors_val.output_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        tensors_val.mean_tensor = Tensor::zeros(mean_tensor.shape());
        tensors_val.rstd_tensor = Tensor::zeros(rstd_tensor.shape());
        tensors_val.input_grad_tensor = Tensor::zeros(input_grad_tensor.shape(), DType::BFloat16);
        tensors_val.weight_grad_tensor = Tensor::zeros(weight_grad_tensor.shape(), DType::BFloat16);
        tensors_val.bias_grad_tensor = Tensor::zeros(bias_grad_tensor.shape(), DType::BFloat16);
        return tensors_val;
    }
};

/// Makes a LayerNorm's tensors of {batch_size, seq_len, embed_dim} from random values, in bfloat16.
static LayerNormTensors random_layer_norm(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val)
{
    Shape shape_list = {batch_size_val, seq_len_val, embed_dim_val};
    LayerNormTensors tensors_val;
    tensors_val.input_tensor = bf16_of(shape_list, random_values(shape_list));
    tensors_val.weight_tensor = bf16_of({embed_dim_val}, random_values({embed_dim_val}));
    tensors_val.bias_tensor = bf16_of({embed_dim_val}, random_values({embed_dim_val}));
    tensors_val.output_grad_tensor = bf16_of(shape_list, random_values(shape_list));
    tensors_val.output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    tensors_val.mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.input_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    tensors_val.weight_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    tensors_val.bias_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    return tensors_val;
}

/// Runs layer_norm_bf16_forward then layer_norm_bf16_backward on `tensors_val`, at this block size for both (the
/// defaults if not given).
static void run_bf16(const LayerNormTensors &tensors_val, int forward_threads_val = LAYER_NORM_BF16_THREADS_PER_BLOCK,
                     int backward_threads_val = LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK)
{
    const LayerNormTensors &t = tensors_val;
    layer_norm_bf16_forward(t.output_tensor, t.mean_tensor, t.rstd_tensor, t.input_tensor, t.weight_tensor, t.bias_tensor,
                            forward_threads_val);
    layer_norm_bf16_backward(t.input_grad_tensor, t.weight_grad_tensor, t.bias_grad_tensor, t.output_grad_tensor, t.input_tensor,
                             t.weight_tensor, t.mean_tensor, t.rstd_tensor, backward_threads_val);
}

// ══ part 1: mine against PyTorch's and against the 4-value path ═══════════════

static void forward_matches_pytorch(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens, embed_dim = 768 as in GPT-2 small (each lane of a warp takes
    // 3 of a row's 96 loads), random input, weight and bias in bfloat16: mean and rstd equal PyTorch's
    // native_layer_norm in float on the same values within TOLERANCE (every sum is in float); output within
    // BFLOAT16_TOLERANCE (rounded once as it is stored).
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(2, 8, GPT2_EMBED_DIM);
    layer_norm_bf16_forward(tensors_val.output_tensor, tensors_val.mean_tensor, tensors_val.rstd_tensor, tensors_val.input_tensor,
                            tensors_val.weight_tensor, tensors_val.bias_tensor);
    // 2. PyTorch's, in float on the same values
    std::vector<int64_t> normalized_shape_list = {GPT2_EMBED_DIM};          // normalize over the last dim
    auto [pytorch_output_tensor, pytorch_mean_tensor, pytorch_rstd_tensor] =
        torch::native_layer_norm(pytorch_copy_of(tensors_val.input_tensor), normalized_shape_list,
                                 pytorch_copy_of(tensors_val.weight_tensor), pytorch_copy_of(tensors_val.bias_tensor), 1e-5);
    // 3. check
    ASSERT_NEAR(tensors_val.mean_tensor.tolist(), list_of(pytorch_mean_tensor), TOLERANCE);
    ASSERT_NEAR(tensors_val.rstd_tensor.tolist(), list_of(pytorch_rstd_tensor), TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.output_tensor), list_of(pytorch_output_tensor), BFLOAT16_TOLERANCE);
}

static void backward_matches_pytorch(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens, embed_dim = 768: 256 rows, 16 blocks of the backward with
    // rows, so weight_grad and bias_grad get adds from 16 blocks. Random input, weight, bias and output_grad in
    // bfloat16, backward from 0: input_grad (rounded once as it is stored), weight_grad and bias_grad (once per block)
    // within BFLOAT16_TOLERANCE of PyTorch's float .grad after autograd's backward on the same values.
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(4, 64, GPT2_EMBED_DIM);
    run_bf16(tensors_val);
    // 2. PyTorch's, in float on the same values, with autograd
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(tensors_val.input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(tensors_val.weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = pytorch_copy_of(tensors_val.bias_tensor).requires_grad_(true);
    std::vector<int64_t> normalized_shape_list = {GPT2_EMBED_DIM};
    torch::Tensor pytorch_output_tensor =
        torch::layer_norm(pytorch_input_tensor, normalized_shape_list, pytorch_weight_tensor, pytorch_bias_tensor, 1e-5);
    pytorch_output_tensor.backward(pytorch_copy_of(tensors_val.output_grad_tensor));
    // 3. check
    ASSERT_NEAR(floats_of(tensors_val.input_grad_tensor), list_of(pytorch_input_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.weight_grad_tensor), list_of(pytorch_weight_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.bias_grad_tensor), list_of(pytorch_bias_tensor.grad()), BFLOAT16_TOLERANCE);
}

static void within_a_rounding_of_the_4_value_path(void)
{
    // Expect: backward_matches_pytorch's 256 random rows of 768, forward and backward here (8 values a lane) and in
    // fp32/layer_norm.cuh's bfloat16 path (4 values a lane), each from 0. The same math on the same values, but a
    // lane holds other values, so each row's sums are added in another order: mean and rstd within TOLERANCE, output
    // and input_grad within ONE_ROUNDING_TOLERANCE, weight_grad and bias_grad within BFLOAT16_TOLERANCE (rounded once
    // per block on both sides, in whatever order the blocks finish).
    // 1. random tensors; both sides
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(4, 64, GPT2_EMBED_DIM);
    LayerNormTensors four_val = tensors_val.with_fresh_results();
    run_bf16(tensors_val);
    layer_norm_forward(four_val.output_tensor, four_val.mean_tensor, four_val.rstd_tensor, four_val.input_tensor,
                       four_val.weight_tensor, four_val.bias_tensor);
    layer_norm_backward(four_val.input_grad_tensor, four_val.weight_grad_tensor, four_val.bias_grad_tensor,
                        four_val.output_grad_tensor, four_val.input_tensor, four_val.weight_tensor, four_val.mean_tensor,
                        four_val.rstd_tensor);
    // 2. check
    ASSERT_NEAR(tensors_val.mean_tensor.tolist(), four_val.mean_tensor.tolist(), TOLERANCE);
    ASSERT_NEAR(tensors_val.rstd_tensor.tolist(), four_val.rstd_tensor.tolist(), TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.output_tensor), floats_of(four_val.output_tensor), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.input_grad_tensor), floats_of(four_val.input_grad_tensor), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.weight_grad_tensor), floats_of(four_val.weight_grad_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.bias_grad_tensor), floats_of(four_val.bias_grad_tensor), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Rows of 8 here (one load: lane 0 of the warp has it all, lanes 1 .. 31 have nothing to read). Most use the row
// 1 1 1 1 3 3 3 3, chosen so its norm rounds to whole numbers in bfloat16:
//
//     mean = 16 / 8 = 2
//     var  = 8 x 1² / 8 = 1
//     rstd = 1 / sqrt(1 + 0.00001) = 0.999995
//     norm = -1 x 4, then +1 x 4, times rstd = -0.999995, 0.999995; in bfloat16 (steps of 2^-8 below 1): -1, 1

static const std::vector<float> NORM_OF_1_3_IN_BFLOAT16 = row_of(-1.0f, 1.0f);

static void forward_moves_a_row_to_mean_0_and_standard_deviation_1(void)
{
    // Expect: the row 1 1 1 1 3 3 3 3, weight 1, bias 0: output is norm, -1 x 4 then 1 x 4 in bfloat16 (mean 0,
    // standard deviation 1); mean is 2 exactly, rstd 0.999995.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(1, 3)),
                            Tensor::ones({SMALL_EMBED_DIM}, DType::BFloat16), Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), NORM_OF_1_3_IN_BFLOAT16);
    ASSERT_EQ(mean_tensor.tolist(), std::vector<float>({2.0f}));
    ASSERT_NEAR(rstd_tensor.tolist()[0], 0.999995, 1e-6);
}

static void weight_and_bias_scale_and_shift_each_column(void)
{
    // Expect: the row 1 1 1 1 3 3 3 3, weight 2 for every column, bias 10 20 ... 80: output = 2 x norm + bias =
    // 10 - 1.99999, 20 - 1.99999, ..., 80 + 1.99999; in bfloat16 (steps of 1/16 up to 1/2 here) 8 18 28 38 52 62 72 82.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(1, 3)),
                            bf16_of({SMALL_EMBED_DIM}, row_of(2, 2)), bf16_of({SMALL_EMBED_DIM}, {10, 20, 30, 40, 50, 60, 70, 80}));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), std::vector<float>({8, 18, 28, 38, 52, 62, 72, 82}));
}

static void a_flat_row_gives_the_bias(void)
{
    // Expect: the row 5 x 8: mean 5, every x - mean is exactly 0, so output is the bias, 10 20 ... 80, exactly. The
    // variance is 0 and rstd = 1 / sqrt(0 + 0.00001) = 316.22777: without the 1e-5 it would be 1 / 0 = infinity, and
    // 0 x infinity = NaN.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(5, 5)),
                            bf16_of({SMALL_EMBED_DIM}, row_of(2, 2)), bf16_of({SMALL_EMBED_DIM}, {10, 20, 30, 40, 50, 60, 70, 80}));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), std::vector<float>({10, 20, 30, 40, 50, 60, 70, 80}));
    ASSERT_NEAR(rstd_tensor.tolist()[0], 316.22777, 1e-3);
}

static void each_row_is_normalized_on_its_own(void)
{
    // Expect: two rows, 1 1 1 1 3 3 3 3 and 101 101 101 101 107 107 107 107 (= 98 + 3 x the first). Each row gets its
    // own mean, 2 and 104, and its own rstd (0.999995 and 0.33333315), so both come out as the same norm in bfloat16:
    // shifting a row or scaling it does not change its output.
    // 1. the forward on two rows
    std::vector<float> input_list = row_of(1, 3);
    for (float value_val : row_of(101, 107))
        input_list.push_back(value_val);
    Tensor output_tensor = Tensor::zeros({1, 2, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 2});
    Tensor rstd_tensor = Tensor::zeros({1, 2});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, bf16_of({1, 2, SMALL_EMBED_DIM}, input_list),
                            Tensor::ones({SMALL_EMBED_DIM}, DType::BFloat16), Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. check
    ASSERT_EQ(mean_tensor.tolist(), std::vector<float>({2.0f, 104.0f}));
    ASSERT_EQ(floats_of(output_tensor[0][0]), NORM_OF_1_3_IN_BFLOAT16);
    ASSERT_EQ(floats_of(output_tensor[0][1]), NORM_OF_1_3_IN_BFLOAT16);
}

static void backward_adds_up_bias_grad_and_weight_grad_over_the_rows(void)
{
    // Expect: two rows, 1 1 1 1 3 3 3 3 and 3 3 3 3 1 1 1 1 (its norm is the first's backwards: -norm), weight 1,
    // output_grad rows of 1s and of 2s.
    //
    //     bias_grad   = 1 + 2                           = 3 x 8, exactly
    //     weight_grad = 1 x norm + 2 x (-norm) = -norm  = 0.999995 x 4, then -0.999995 x 4: 1 x 4, -1 x 4 in bfloat16
    // 1. the forward
    std::vector<float> input_list = row_of(1, 3);
    for (float value_val : row_of(3, 1))
        input_list.push_back(value_val);
    Tensor input_tensor = bf16_of({1, 2, SMALL_EMBED_DIM}, input_list);
    Tensor weight_tensor = Tensor::ones({SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor output_tensor = Tensor::zeros({1, 2, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 2});
    Tensor rstd_tensor = Tensor::zeros({1, 2});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                            Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. the backward
    std::vector<float> output_grad_list = row_of(1, 1);
    for (float value_val : row_of(2, 2))
        output_grad_list.push_back(value_val);
    Tensor input_grad_tensor = Tensor::zeros({1, 2, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16);
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, bf16_of({1, 2, SMALL_EMBED_DIM}, output_grad_list),
                             input_tensor, weight_tensor, mean_tensor, rstd_tensor);
    // 3. check
    ASSERT_EQ(floats_of(bias_grad_tensor), row_of(3, 3));
    ASSERT_EQ(floats_of(weight_grad_tensor), row_of(1, -1));
}

static void the_same_output_grad_on_a_whole_row_gives_no_input_grad(void)
{
    // Expect: the row 1 1 1 1 3 3 3 3, weight 1, output_grad 5 x 8: input_grad is 0 (up to rounding). Raising every
    // output of a row by the same amount is something no change of the input can do, since the forward subtracts the
    // row's mean: norm_grad - mean(norm_grad) = 5 - 5 = 0, and mean(norm_grad x norm) = 5 x mean(norm) = 0.
    // 1. the forward
    Tensor input_tensor = bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(1, 3));
    Tensor weight_tensor = Tensor::ones({SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor output_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                            Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. the backward
    Tensor input_grad_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16);
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(5, 5)),
                             input_tensor, weight_tensor, mean_tensor, rstd_tensor);
    // 3. check
    ASSERT_NEAR(floats_of(input_grad_tensor), std::vector<float>(SMALL_EMBED_DIM, 0.0f), 1e-5);
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch, or for input_grad the residual stream, had
    // left it there). The row 1 1 1 1 3 3 3 3, weight 1, output_grad 1 x 8: bias_grad becomes 11 exactly, weight_grad
    // 10 + norm = 9.000005 and 10.999995: 9 x 4, 11 x 4 in bfloat16; input_grad 10 + 0 (as in the test above). A second
    // backward adds again: bias_grad 12.
    // 1. the forward
    Tensor input_tensor = bf16_of({1, 1, SMALL_EMBED_DIM}, row_of(1, 3));
    Tensor weight_tensor = Tensor::ones({SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor output_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                            Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. one backward onto gradients of 10: added onto
    Tensor output_grad_tensor = Tensor::ones({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16).fill_(10);
    Tensor weight_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16).fill_(10);
    Tensor bias_grad_tensor = Tensor::zeros({SMALL_EMBED_DIM}, DType::BFloat16).fill_(10);
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                             weight_tensor, mean_tensor, rstd_tensor);
    ASSERT_EQ(floats_of(bias_grad_tensor), row_of(11, 11));
    ASSERT_EQ(floats_of(weight_grad_tensor), row_of(9, 11));
    ASSERT_EQ(floats_of(input_grad_tensor), row_of(10, 10));
    // 3. a second backward: added again
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                             weight_tensor, mean_tensor, rstd_tensor);
    ASSERT_EQ(floats_of(bias_grad_tensor), row_of(12, 12));
}

static void many_rows_adding_onto_bias_grad_lose_nothing(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens, embed_dim = 768, random input, weight 1, output_grad all 1.
    // All 256 rows add onto the same 768 values of bias_grad: 16 rows into each of the 16 blocks' shared copies (one
    // row a warp, 16 warps a block), then 16 blocks into GPU memory, many at the same moment. With the
    // compare-and-swap of atomic_add4_from_float no add is lost: 16 x 16 = 256 in every column, exactly (every
    // partial sum, 16, 32, ..., 256, is a whole number bfloat16 holds). Not GPT-2's 4096 rows: past 256, bfloat16
    // steps by 2 and more, so the blocks' sums would be rounded on the way. input_grad is 0 in every row, up to
    // rounding.
    // 1. the forward
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 64;
    Shape shape_list = {batch_size_val, seq_len_val, GPT2_EMBED_DIM};
    Tensor input_tensor = bf16_of(shape_list, random_values(shape_list));
    Tensor weight_tensor = Tensor::ones({GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    layer_norm_bf16_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                            Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16));
    // 2. the backward with output_grad all 1
    Tensor input_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16);
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, Tensor::ones(shape_list, DType::BFloat16),
                             input_tensor, weight_tensor, mean_tensor, rstd_tensor);
    // 3. check
    ASSERT_EQ(floats_of(bias_grad_tensor), std::vector<float>(GPT2_EMBED_DIM, 256.0f));
    ASSERT_NEAR(floats_of(input_grad_tensor), std::vector<float>(batch_size_val * seq_len_val * GPT2_EMBED_DIM, 0.0f), 1e-5);
}

// ══ part 3: the kernels fit their launch, and other block sizes work ═════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the forward can be launched with LAYER_NORM_BF16_THREADS_PER_BLOCK (256) and the backward with
    // LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK (512), and nothing is spilled: a lane's 8 values (and the 16-byte
    // loads they are read into) stay in registers. cudaFuncGetAttributes reads what the compiler wrote down, without
    // running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)layer_norm_bf16_forward_kernel, (const void *)layer_norm_bf16_backward_kernel};
    const char *names_list[] = {"layer_norm_bf16_forward_kernel", "layer_norm_bf16_backward_kernel"};
    int threads_list[] = {LAYER_NORM_BF16_THREADS_PER_BLOCK, LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void other_block_sizes_give_the_same_result(void)
{
    // Expect: 256 random rows of 768, both launchers called with threads_per_block 32, 128, 256 and 512 (one warp up;
    // as in tests/fp32/test_layer_norm.cu). Each row is one warp's whatever the block size, so output, mean, rstd and
    // input_grad are the defaults' to the last bit. weight_grad and bias_grad are each block's shared sums added onto
    // GPU memory in whatever order the blocks finish: within BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                    16 * HARDWARE_THREADS_PER_WARP};
    // 1. random tensors; the defaults' results
    Tensor::manual_seed(0);
    LayerNormTensors usual_val = random_layer_norm(4, 64, GPT2_EMBED_DIM);
    run_bf16(usual_val);

    // 2. each block size, from the same inputs, against the defaults'
    for (int threads_per_block_val : block_sizes_list)
    {
        LayerNormTensors tensors_val = usual_val.with_fresh_results();
        run_bf16(tensors_val, threads_per_block_val, threads_per_block_val);
        ASSERT_EQ(floats_of(tensors_val.output_tensor), floats_of(usual_val.output_tensor));
        ASSERT_EQ(tensors_val.mean_tensor.tolist(), usual_val.mean_tensor.tolist());
        ASSERT_EQ(tensors_val.rstd_tensor.tolist(), usual_val.rstd_tensor.tolist());
        ASSERT_EQ(floats_of(tensors_val.input_grad_tensor), floats_of(usual_val.input_grad_tensor));
        ASSERT_NEAR(floats_of(tensors_val.weight_grad_tensor), floats_of(usual_val.weight_grad_tensor),
                    BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE);
        ASSERT_NEAR(floats_of(tensors_val.bias_grad_tensor), floats_of(usual_val.bias_grad_tensor),
                    BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_matches_pytorch();
    backward_matches_pytorch();
    within_a_rounding_of_the_4_value_path();

    forward_moves_a_row_to_mean_0_and_standard_deviation_1();
    weight_and_bias_scale_and_shift_each_column();
    a_flat_row_gives_the_bias();
    each_row_is_normalized_on_its_own();
    backward_adds_up_bias_grad_and_weight_grad_over_the_rows();
    the_same_output_grad_on_a_whole_row_gives_no_input_grad();
    backward_adds_onto_the_gradients_already_there();
    many_rows_adding_onto_bias_grad_lose_nothing();

    every_kernel_fits_its_block();
    other_block_sizes_give_the_same_result();

    printf("\nall tests passed\n");
    return 0;
}
