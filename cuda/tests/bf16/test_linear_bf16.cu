// test_linear_bf16.cu — tests for cuda/kernels/bf16/linear_bf16.cuh (linear_bf16_forward, linear_bf16_backward): the
// linear layer in bfloat16, its three products by wgmma_matmul_bf16 on an H100's tensor cores. Every check prints
// its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks both against PyTorch's linear in float on the same bfloat16 values, with autograd for the
// backward; part 2 checks numbers worked out by hand; part 3 checks that bias_grad's kernel fits its block. The
// products are tested in cuda/tests/test_wgmma_matmul_bf16.cu. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a in place of -arch=native. The float layer is tested in cuda/tests/test_linear.cu. The timings on GPT-2
// small's 5 linear layers are in cuda/tests/bf16/tune_linear_bf16.cu.
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../kernels/bf16/linear_bf16.cuh"

// Against PyTorch in float on the same bfloat16 values: every sum is in float, and each result is rounded once
// as it is stored. A bfloat16 keeps 8 significant bits, so that rounding moves a value by at most 2^-8, about
// 0.4% of it. Twice that.
static const double BFLOAT16_TOLERANCE = 8e-3;

// The hand-worked tests' tensors are 4 wide, as cuda/tests/fp32/test_linear.cu's, padded with 0s to
// WGMMA_MATMUL_BF16_VALUE_MULTIPLE (8): TMA reads rows of whole 16-byte pieces. The 0s add nothing to any sum.
#define PADDED_WIDTH WGMMA_MATMUL_BF16_VALUE_MULTIPLE

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a bfloat16 tensor on the GPU holding `values_list`, each rounded to bfloat16.
///
///   shape_list    the tensor's shape
///   values_list   in row order
static Tensor bf16_tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Gives a bfloat16 tensor's values as a CPU list of floats, exactly.
static std::vector<float> values_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Gives each value rounded to bfloat16 (to the nearest), back as floats.
///
///   values_list   the values
static std::vector<float> rounded_to_bf16(const std::vector<float> &values_list)
{
    // Gives each value of `values_list` rounded to bfloat16 (to the nearest), back as floats: what a sum worked out
    // in float becomes when it is stored.
    return values_of(bf16_tensor_of({values_list.size()}, values_list));
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU. A bfloat16 tensor's values are turned into floats first, exactly.
    Tensor floats_tensor = source_tensor.to(DType::Float32);  // the source_tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values and checks them against PyTorch's float linear.
///
///   test_ptr           the test's name, printed with the sizes
///   input_shape_list   {batch_size, seq_len, in_features}
///   out_features_val   the layer's output width
///   with_bias_val      false: no bias
static void forward_and_backward_match_pytorch(const char *test_ptr, const Shape &input_shape_list,
                                               size_t out_features_val, bool with_bias_val)
{
    // Runs one forward and one backward on random bfloat16 values, mine and PyTorch's float linear with autograd on
    // the same values, and checks that output and the gradients agree within BFLOAT16_TOLERANCE. Prints the sizes
    // under `test_ptr`'s name first. input_shape is {batch_size, seq_len, in_features}; with_bias false: Tensor() for
    // mine, no bias for PyTorch's. The gradients start at 0.
    // 1. random tensors in bfloat16; mine
    Tensor::manual_seed(0);
    size_t in_features_val = input_shape_list.back();
    size_t rows_val = input_shape_list[0] * input_shape_list[1];
    Shape output_shape_list = {input_shape_list[0], input_shape_list[1], out_features_val};
    Tensor input_tensor = bf16_tensor_of(input_shape_list, random_values(input_shape_list));
    Tensor weight_tensor = bf16_tensor_of({out_features_val, in_features_val}, random_values({out_features_val, in_features_val}));
    Tensor bias_tensor = with_bias_val ? bf16_tensor_of({out_features_val}, random_values({out_features_val})) : Tensor();
    Tensor output_grad_tensor = bf16_tensor_of(output_shape_list, random_values(output_shape_list));
    Tensor output_tensor = Tensor::zeros(output_shape_list, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros(input_shape_list, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}, DType::BFloat16) : Tensor();
    linear_bf16_forward(output_tensor, input_tensor, weight_tensor, bias_tensor);
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);

    // 2. PyTorch's, in float on the same values, with autograd
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = with_bias_val ? pytorch_copy_of(bias_tensor).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_tensor = with_bias_val
                                              ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                              : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor));

    // 3. print the sizes, then check, within one rounding
    print_test_name(test_ptr);
    printf("    %zu rows, in_features %zu, out_features %zu, %s\n", rows_val, in_features_val, out_features_val,
           with_bias_val ? "with bias" : "no bias");
    ASSERT_NEAR(values_of(output_tensor), list_of(pytorch_output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(values_of(input_grad_tensor), list_of(pytorch_input_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(values_of(weight_grad_tensor), list_of(pytorch_weight_tensor.grad()), BFLOAT16_TOLERANCE);
    if (with_bias_val)
        ASSERT_NEAR(values_of(bias_grad_tensor), list_of(pytorch_bias_tensor.grad()), BFLOAT16_TOLERANCE);
}

static void matches_pytorch_on_qkv_sizes(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens (256 rows: 4 squares of 64 down), in_features = 768,
    // out_features = 2304 as GPT-2 small's qkv (36 squares across), random input, weight, bias and output_grad
    // in bfloat16: output, input_grad, weight_grad and bias_grad equal what PyTorch's float linear and its
    // autograd give on the same values, within BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 2304, true);
}

static void matches_pytorch_on_sizes_that_do_not_fill_the_squares(void)
{
    // Expect: input {2, 37, 96}, weight {136, 96} and bias: 74 rows (64 + 10) and 136 columns (2 x 64 + 8), so
    // wgmma_matmul_bf16's squares at the edges are part-full; in_features 96 is a step of 64 then 32, TMA giving
    // 0s past the edge; weight_grad's sum is the 74 rows, a step of 64 then 10. Within BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, {2, 37, 96}, 136, true);
}

static void matches_pytorch_without_bias(void)
{
    // Expect: 256 rows, in_features = 768, out_features = 1000, no bias, like the head with a small vocabulary
    // (1000 is not a multiple of 64 either): within BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 1000, false);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use cuda/tests/fp32/test_linear.cu's 2 rows of input, weight and bias, 4 wide, padded with 0s to
// PADDED_WIDTH (8): input {1, 2, 8}, weight {8, 8} (its rows 4..7 all 0), bias 8 wide. Every value is a whole
// number below 256, which bfloat16 holds exactly:
//
//     input   row 0: 1 2 3 4 | 0 0 0 0        weight   out 0:  1  0  0  0 | 0 0 0 0        bias   10 20 30 40 | 0 0 0 0
//             row 1: 5 6 7 8 | 0 0 0 0                 out 1:  0  1  0  0 | 0 0 0 0
//                                                      out 2:  1  1  1  1 | 0 0 0 0
//                                                      out 3:  1 -1  1 -1 | 0 0 0 0
//                                                      out 4..7: all 0
//
// Each output is a row of input dotted with a row of weight, plus that column's bias:
//
//     output  row 0:  1 + 10,  2 + 20,  1 + 2 + 3 + 4 + 30,  1 - 2 + 3 - 4 + 40 | 0 0 0 0   =  11 22 40 38 | 0 0 0 0
//             row 1:  5 + 10,  6 + 20,  5 + 6 + 7 + 8 + 30,  5 - 6 + 7 - 8 + 40 | 0 0 0 0   =  15 26 56 38 | 0 0 0 0

/// Makes the input of part 2: {1, 2, 8}, rows 1 2 3 4 and 5 6 7 8, each padded with 0s.
static Tensor small_input(void)
{
    return bf16_tensor_of({1, 2, PADDED_WIDTH}, {1, 2, 3, 4, 0, 0, 0, 0,
                                                 5, 6, 7, 8, 0, 0, 0, 0});
}

/// Makes the weight of part 2: {8, 8}, one row per output column, as in the table above.
static Tensor small_weight(void)
{
    return bf16_tensor_of({PADDED_WIDTH, PADDED_WIDTH}, {1, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 1, 0, 0, 0, 0, 0, 0,
                                                         1, 1, 1, 1, 0, 0, 0, 0,
                                                         1, -1, 1, -1, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0});
}

/// Makes the bias of part 2: 10 20 30 40, padded with 0s.
static Tensor small_bias(void)
{
    return bf16_tensor_of({PADDED_WIDTH}, {10, 20, 30, 40, 0, 0, 0, 0});
}

/// Makes the output_grad of part 2's backward tests: 1 only at row 0's output 0 and row 1's output 2.
static Tensor small_output_grad(void)
{
    return bf16_tensor_of({1, 2, PADDED_WIDTH}, {1, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 1, 0, 0, 0, 0, 0});
}

static void forward_dots_each_input_row_with_each_weight_row_and_adds_the_bias(void)
{
    // Expect: output 11 22 40 38 0 0 0 0 and 15 26 56 38 0 0 0 0, exactly (whole numbers this small add exactly
    // in float, and bfloat16 holds them).
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 2, PADDED_WIDTH}, DType::BFloat16);
    linear_bf16_forward(output_tensor, small_input(), small_weight(), small_bias());
    // 2. check
    ASSERT_EQ(values_of(output_tensor), std::vector<float>({11, 22, 40, 38, 0, 0, 0, 0, 15, 26, 56, 38, 0, 0, 0, 0}));
}

static void forward_without_bias_gives_the_dot_products_alone(void)
{
    // Expect: bias Tensor() (none, as the head): output 1 2 10 -2 0 0 0 0 and 5 6 26 -2 0 0 0 0, the table
    // above without the 10 20 30 40.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 2, PADDED_WIDTH}, DType::BFloat16);
    linear_bf16_forward(output_tensor, small_input(), small_weight(), Tensor());
    // 2. check
    ASSERT_EQ(values_of(output_tensor), std::vector<float>({1, 2, 10, -2, 0, 0, 0, 0, 5, 6, 26, -2, 0, 0, 0, 0}));
}

static void backward_sends_each_output_grad_back_through_its_weight_row(void)
{
    // Expect: output_grad 1 only at row 0's output 0 and row 1's output 2, 0 everywhere else:
    //
    //     input_grad[row]  = sum over j of output_grad[row][j] x weight[j]    row 0: weight out 0 = 1 0 0 0 | 0 0 0 0
    //                                                                         row 1: weight out 2 = 1 1 1 1 | 0 0 0 0
    //     weight_grad[j]   = sum over rows of output_grad[row][j] x input[row] out 0: input row 0 = 1 2 3 4 | 0 0 0 0
    //                                                                         out 2: input row 1 = 5 6 7 8 | 0 0 0 0
    //     bias_grad[j]     = sum over rows of output_grad[row][j]             1 0 1 0 | 0 0 0 0
    //
    // The other outputs got no gradient, so their rows of weight_grad stay 0. All exact.
    // 1. the backward
    Tensor input_grad_tensor = Tensor::zeros({1, 2, PADDED_WIDTH}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({PADDED_WIDTH, PADDED_WIDTH}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({PADDED_WIDTH}, DType::BFloat16);
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, small_output_grad(), small_input(),
                         small_weight());
    // 2. check
    std::vector<float> weight_grad_list(PADDED_WIDTH * PADDED_WIDTH, 0.0f);
    for (int column_index_val = 0; column_index_val < 4; column_index_val++)
    {
        weight_grad_list[0 * PADDED_WIDTH + column_index_val] = (float)(1 + column_index_val);   // out 0: 1 2 3 4
        weight_grad_list[2 * PADDED_WIDTH + column_index_val] = (float)(5 + column_index_val);   // out 2: 5 6 7 8
    }
    ASSERT_EQ(values_of(input_grad_tensor), std::vector<float>({1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}));
    ASSERT_EQ(values_of(weight_grad_tensor), weight_grad_list);
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>({1, 0, 1, 0, 0, 0, 0, 0}));
}

static void backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there), output_grad as in the
    // test above. input_grad is written over: 1 0 0 0 0 0 0 0 and 1 1 1 1 0 0 0 0, the 10s gone. weight_grad and
    // bias_grad are added onto: weight_grad out 0 = 10 + 1 2 3 4 = 11 12 13 14 then 10s, out 2 = 15 16 17 18 then
    // 10s, the rest 10; bias_grad 11 10 11 10 10 10 10 10. A second backward adds again (bias_grad
    // 12 10 12 10 10 10 10 10) and writes input_grad again, the same. All below 256: exact in bfloat16.
    // 1. gradients that start at 10
    Tensor input_tensor = small_input();
    Tensor weight_tensor = small_weight();
    Tensor output_grad_tensor = small_output_grad();
    Tensor input_grad_tensor = Tensor::zeros({1, 2, PADDED_WIDTH}, DType::BFloat16).fill_(10);
    Tensor weight_grad_tensor = Tensor::zeros({PADDED_WIDTH, PADDED_WIDTH}, DType::BFloat16).fill_(10);
    Tensor bias_grad_tensor = Tensor::zeros({PADDED_WIDTH}, DType::BFloat16).fill_(10);
    std::vector<float> input_grad_list = {1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
    std::vector<float> weight_grad_list(PADDED_WIDTH * PADDED_WIDTH, 10.0f);
    for (int column_index_val = 0; column_index_val < 4; column_index_val++)
    {
        weight_grad_list[0 * PADDED_WIDTH + column_index_val] = (float)(11 + column_index_val);  // out 0: 11 12 13 14
        weight_grad_list[2 * PADDED_WIDTH + column_index_val] = (float)(15 + column_index_val);  // out 2: 15 16 17 18
    }
    // 2. one backward: input_grad written over, the others added onto
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(values_of(input_grad_tensor), input_grad_list);
    ASSERT_EQ(values_of(weight_grad_tensor), weight_grad_list);
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>({11, 10, 11, 10, 10, 10, 10, 10}));
    // 3. a second backward: the same again
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(values_of(input_grad_tensor), input_grad_list);
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>({12, 10, 12, 10, 10, 10, 10, 10}));
}

/// Makes the identity matrix {size_val, size_val} in bfloat16: 1 where the row is the column, 0 elsewhere.
static Tensor identity(size_t size_val)
{
    std::vector<float> values_list(size_val * size_val, 0.0f);
    for (size_t diagonal_index_val = 0; diagonal_index_val < size_val; diagonal_index_val++)
        values_list[diagonal_index_val * size_val + diagonal_index_val] = 1.0f;
    return bf16_tensor_of({size_val, size_val}, values_list);
}

/// Makes `count_val` values that bfloat16 holds exactly and that tell positions apart nearby: value i is i % 256.
static std::vector<float> whole_numbers_below_256(size_t count_val)
{
    std::vector<float> values_list(count_val);
    for (size_t value_index_val = 0; value_index_val < count_val; value_index_val++)
        values_list[value_index_val] = (float)(value_index_val % 256);
    return values_list;
}

static void identity_weight_gives_back_the_input_in_every_square(void)
{
    // Expect: input {2, 100, 256} holding 0, 1, ..., 255, 0, 1, ... (value i at position i, % 256: whole numbers
    // bfloat16 holds exactly), weight the identity {256, 256}, no bias: output[row][j] = input[row][j] x 1 + 0s,
    // so output is input, exactly. 200 rows and 256 columns are 4 x 4 squares of 64, the bottom ones part-full (8
    // rows): a value written to the wrong place, or not written, shows (a row's 256 values are all different). The
    // backward the same way: output_grad the same values gives input_grad = output_grad · identity = output_grad,
    // exactly.
    // 1. the forward: output is input
    Tensor input_tensor = bf16_tensor_of({2, 100, 256}, whole_numbers_below_256(2 * 100 * 256));
    Tensor weight_tensor = identity(256);
    Tensor output_tensor = Tensor::zeros({2, 100, 256}, DType::BFloat16);
    linear_bf16_forward(output_tensor, input_tensor, weight_tensor, Tensor());
    ASSERT_EQ(values_of(output_tensor), values_of(input_tensor));
    // 2. the backward: input_grad is output_grad
    Tensor output_grad_tensor = bf16_tensor_of({2, 100, 256}, whole_numbers_below_256(2 * 100 * 256));
    Tensor input_grad_tensor = Tensor::zeros({2, 100, 256}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({256, 256}, DType::BFloat16);
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, Tensor(), output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(values_of(input_grad_tensor), values_of(output_grad_tensor));
}

static void weight_grad_is_output_grad_turned_round_when_input_is_the_identity(void)
{
    // Expect: input the identity {1, 256, 256} (256 rows, in_features 256), out_features 384, output_grad
    // {1, 256, 384} holding (128 row + out) % 256 (whole numbers bfloat16 holds, as 384 row % 256 = 128 row % 256):
    // weight_grad[out][in] = sum over rows of output_grad[row][out] x input[row][in] = output_grad[in][out] =
    // (128 in + out) % 256, exactly. This is the product that reads output_grad turned round; 384 x 256 is 6 x 4
    // squares. bias_grad[out] = sum over the 256 rows of output_grad[row][out]: the even rows give out % 256, the
    // odd ones (out + 128) % 256, so 128 (out % 256) + 128 ((out + 128) % 256). The sum is exact in float (whole
    // numbers below 2^24), then rounded to bfloat16 once as it is stored: expected is that rounding.
    // 1. the backward
    Tensor input_tensor = identity(256).view({1, 256, 256});
    Tensor weight_tensor = Tensor::zeros({384, 256}, DType::BFloat16);
    std::vector<float> output_grad_list(256 * 384);
    for (size_t row_index_val = 0; row_index_val < 256; row_index_val++)
        for (size_t out_feature_index_val = 0; out_feature_index_val < 384; out_feature_index_val++)
            output_grad_list[row_index_val * 384 + out_feature_index_val] = (float)((128 * row_index_val + out_feature_index_val) % 256);
    Tensor output_grad_tensor = bf16_tensor_of({1, 256, 384}, output_grad_list);
    Tensor input_grad_tensor = Tensor::zeros({1, 256, 256}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({384, 256}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({384}, DType::BFloat16);
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    // 2. the expected values, worked out
    std::vector<float> turned_round_list(384 * 256), column_sums_list(384);
    for (size_t out_feature_index_val = 0; out_feature_index_val < 384; out_feature_index_val++)
    {
        for (size_t in_feature_index_val = 0; in_feature_index_val < 256; in_feature_index_val++)
            turned_round_list[out_feature_index_val * 256 + in_feature_index_val] = output_grad_list[in_feature_index_val * 384 + out_feature_index_val];
        column_sums_list[out_feature_index_val] =
            (float)(128 * (out_feature_index_val % 256) + 128 * ((out_feature_index_val + 128) % 256));
    }
    // 3. check
    ASSERT_EQ(values_of(weight_grad_tensor), turned_round_list);
    ASSERT_EQ(values_of(bias_grad_tensor), rounded_to_bf16(column_sums_list));
}

static void whole_numbers_come_out_exact_on_mlp_up_sizes(void)
{
    // Expect: GPT-2 small's MLP up, batch_size = 4 rows of seq_len = 1024 tokens, 768 in, 3072 out, with every
    // value 1: input, weight, bias and output_grad. Each output is 768 x 1 x 1 + 1 = 769 in float; bfloat16 holds
    // only every 4th whole number between 512 and 1024, so it is stored as the nearest, 768. Each input_grad is a
    // sum over 3072 outputs, 3072; each weight_grad and bias_grad a sum over 4096 rows, 4096: both held exactly.
    // Every square of every product: no value is missed or added twice.
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t in_features_val = 768;
    size_t out_features_val = 3072;
    size_t rows_val = batch_size_val * seq_len_val;
    // 1. the forward, all 1s
    Tensor input_tensor = Tensor::ones({batch_size_val, seq_len_val, in_features_val}).to(DType::BFloat16);
    Tensor weight_tensor = Tensor::ones({out_features_val, in_features_val}).to(DType::BFloat16);
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, out_features_val}, DType::BFloat16);
    linear_bf16_forward(output_tensor, input_tensor, weight_tensor, Tensor::ones({out_features_val}).to(DType::BFloat16));
    ASSERT_EQ(values_of(output_tensor), std::vector<float>(rows_val * out_features_val, rounded_to_bf16({769.0f})[0]));
    // 2. the backward, all 1s
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, in_features_val}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({out_features_val}, DType::BFloat16);
    linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                         Tensor::ones({batch_size_val, seq_len_val, out_features_val}).to(DType::BFloat16), input_tensor,
                         weight_tensor);
    ASSERT_EQ(values_of(input_grad_tensor), std::vector<float>(rows_val * in_features_val, 3072.0f));
    ASSERT_EQ(values_of(weight_grad_tensor), std::vector<float>(out_features_val * in_features_val, 4096.0f));
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>(out_features_val, 4096.0f));
}

static void every_row_chunk_count_gives_the_column_sums(void)
{
    // Expect: random output_grad {1000, 776} (rows no chunk count divides, 776 columns: 12 blocks of 64 and one of 8),
    // bias_grad starting at 0.5: with the rows cut into 1, 3, 64 and 1000 chunks (1000: a row each) and 2000 (more
    // chunks than rows: the last ones empty), bias_grad = 0.5 + each column's sum, within BFLOAT16_TOLERANCE of PyTorch's
    // float column sums. Every count adds the same values in another order.
    // 1. random values; PyTorch's column sums
    std::vector<float> output_grad_list = random_values({1000, 776});
    Tensor output_grad_tensor = bf16_tensor_of({1000, 776}, output_grad_list);
    torch::Tensor expected_tensor = pytorch_copy_of(output_grad_tensor).sum(0) + 0.5;
    print_test_name(__func__);
    // 2. each chunk count
    const int row_chunks_list[] = {1, 3, 64, 1000, 2000};
    for (int row_chunks_val : row_chunks_list)
    {
        Tensor bias_grad_tensor = Tensor::zeros({776}, DType::BFloat16).fill_(0.5);
        linear_bf16_bias_grad(bias_grad_tensor, output_grad_tensor, row_chunks_val);
        torch::Tensor mine_tensor = pytorch_copy_of(bias_grad_tensor);
        double difference_val = (mine_tensor - expected_tensor).abs().max().item<double>() / expected_tensor.abs().max().item<double>();
        printf("    %d chunks:\n", row_chunks_val);
        ASSERT_NEAR(difference_val, 0.0, BFLOAT16_TOLERANCE);
    }
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: linear_bf16_bias_grad_kernel can be launched with LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK (256), and
    // nothing is spilled. cudaFuncGetAttributes reads what the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // The products' kernels are checked in cuda/tests/test_wgmma_matmul_bf16.cu.
    const void *kernels_list[] = {(const void *)linear_bf16_bias_grad_kernel, (const void *)linear_bf16_bias_grad_sums_kernel};
    const char *names_list[] = {"linear_bf16_bias_grad_kernel", "linear_bf16_bias_grad_sums_kernel"};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val],
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_qkv_sizes();
    matches_pytorch_on_sizes_that_do_not_fill_the_squares();
    matches_pytorch_without_bias();

    forward_dots_each_input_row_with_each_weight_row_and_adds_the_bias();
    forward_without_bias_gives_the_dot_products_alone();
    backward_sends_each_output_grad_back_through_its_weight_row();
    backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad();
    identity_weight_gives_back_the_input_in_every_square();
    weight_grad_is_output_grad_turned_round_when_input_is_the_identity();
    whole_numbers_come_out_exact_on_mlp_up_sizes();
    every_row_chunk_count_gives_the_column_sums();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
