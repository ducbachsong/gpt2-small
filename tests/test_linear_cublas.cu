// test_linear_cublas.cu — tests for llmc/kernels/linear_cublas.cuh (linear_forward, linear_backward made by
// cuBLAS). The same tests as test_linear.cu, with the same names, so the two files' output lines up. Every
// check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks both against PyTorch's linear, with autograd for the backward; part 2 checks numbers worked
// out by hand; part 3 checks tensors that start off 256 bytes, which cuBLASLt must be told about
// (test_linear.cu's part 3 checks its kernels' registers instead: here there are no kernels of our own, and
// cuBLAS picks its own launches). The timings are in tunes/tune_linear_cublas.cu.
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/kernels/linear_cublas.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: every sum is in float, and each result is rounded
// once as it is stored. A bfloat16 keeps 8 significant bits, so that rounding moves a value by at most 2^-8,
// about 0.4% of it. Twice that.
static const double BFLOAT16_TOLERANCE = 8e-3;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values, mine and PyTorch's linear with autograd, and
/// checks that output and the gradients agree within TOLERANCE. Prints the sizes under `test_ptr`'s name
/// first. input_shape is {batch_size, seq_len, in_features}; with_bias false: Tensor() for mine, no bias
/// for PyTorch's. The gradients start at 0.
static void forward_and_backward_match_pytorch(const char *test_ptr, const Shape &input_shape_list,
                                               size_t out_features_val, bool with_bias_val)
{
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    size_t in_features_val = input_shape_list.back();
    size_t rows_val = input_shape_list[0] * input_shape_list[1];
    Shape output_shape_list = {input_shape_list[0], input_shape_list[1], out_features_val};
    Tensor input_tensor = tensor_of(input_shape_list, random_values(input_shape_list));
    Tensor weight_tensor = tensor_of({out_features_val, in_features_val}, random_values({out_features_val, in_features_val}));
    Tensor bias_tensor = with_bias_val ? tensor_of({out_features_val}, random_values({out_features_val})) : Tensor();
    Tensor output_grad_tensor = tensor_of(output_shape_list, random_values(output_shape_list));
    Tensor output_tensor = Tensor::zeros(output_shape_list);
    Tensor input_grad_tensor = Tensor::zeros(input_shape_list);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val});
    Tensor bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}) : Tensor();
    linear_forward(output_tensor, input_tensor, weight_tensor, bias_tensor);
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);

    // 2. PyTorch's, with autograd
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = with_bias_val ? pytorch_copy_of(bias_tensor).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_tensor = with_bias_val
                                              ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                              : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor));

    // 3. print the sizes, then check
    print_test_name(test_ptr);
    printf("    %zu rows, in_features %zu, out_features %zu, %s\n", rows_val, in_features_val, out_features_val,
           with_bias_val ? "with bias" : "no bias");
    ASSERT_NEAR(output_tensor.tolist(), list_of(pytorch_output_tensor), TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(weight_grad_tensor.tolist(), list_of(pytorch_weight_tensor.grad()), TOLERANCE);
    if (with_bias_val)
        ASSERT_NEAR(bias_grad_tensor.tolist(), list_of(pytorch_bias_tensor.grad()), TOLERANCE);
}

static void matches_pytorch_on_qkv_sizes(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens (256 rows), in_features = 768,
    // out_features = 2304 as GPT-2 small's qkv, random input, weight, bias and
    // output_grad: output, input_grad, weight_grad and bias_grad equal what PyTorch's linear and its
    // autograd give, within 1e-5. Not to the last bit: the two add up each dot product in another order.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 2304, true);
}

static void matches_pytorch_on_sizes_that_do_not_fill_the_squares(void)
{
    // Expect: 3 x 50 = 150 rows, in_features = 100, out_features = 300, none a multiple of cuBLAS's
    // squares (128 x 64 for the qkv on a T4): the same as PyTorch within 1e-5. cuBLAS picks kernels
    // that check their edges for these.
    forward_and_backward_match_pytorch(__func__, {3, 50, 100}, 300, true);
}

static void matches_pytorch_without_bias(void)
{
    // Expect: 256 rows, in_features = 768, out_features = 1000, no bias, like the head with a small
    // vocabulary (1000 is not a multiple of 64 either): the same as PyTorch within 1e-5.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 1000, false);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: input {2, 37, 96}, weight {132, 96} and bias, all in bfloat16: cuBLAS told CUDA_R_16BF for every
    // matrix, the bias epilogue and cublasGemmEx's three backward products included. Every sum is in float
    // (CUBLAS_COMPUTE_32F), so output, input_grad, weight_grad and bias_grad are PyTorch's float linear of the
    // same values, each rounded once as it is stored: within BFLOAT16_TOLERANCE.

    // 1. random tensors in bfloat16; mine
    Tensor::manual_seed(0);
    size_t in_features_val = 96;
    size_t out_features_val = 132;
    Shape input_shape_list = {2, 37, in_features_val};
    Shape output_shape_list = {2, 37, out_features_val};
    Tensor input_tensor = tensor_of(input_shape_list, random_values(input_shape_list)).to(DType::BFloat16);
    Tensor weight_tensor =
        tensor_of({out_features_val, in_features_val}, random_values({out_features_val, in_features_val})).to(DType::BFloat16);
    Tensor bias_tensor = tensor_of({out_features_val}, random_values({out_features_val})).to(DType::BFloat16);
    Tensor output_grad_tensor = tensor_of(output_shape_list, random_values(output_shape_list)).to(DType::BFloat16);
    Tensor output_tensor = Tensor::zeros(output_shape_list, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros(input_shape_list, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({out_features_val}, DType::BFloat16);
    linear_forward(output_tensor, input_tensor, weight_tensor, bias_tensor);
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);

    // 2. PyTorch's, in float on the same values
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = pytorch_copy_of(bias_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_output_tensor = torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor);
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor.to(DType::Float32)));

    // 3. check, within one rounding
    ASSERT_NEAR(output_tensor.to(DType::Float32).tolist(), list_of(pytorch_output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(input_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_input_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(weight_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_weight_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(bias_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_bias_tensor.grad()), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the same 2 rows of input, weight and bias, 4 wide:
//
//     input   row 0: 1 2 3 4        weight   out 0:  1  0  0  0        bias   10 20 30 40
//             row 1: 5 6 7 8                 out 1:  0  1  0  0
//                                            out 2:  1  1  1  1
//                                            out 3:  1 -1  1 -1
//
// Each output is a row of input dotted with a row of weight, plus that column's bias:
//
//     output  row 0:  1 + 10,  2 + 20,  1 + 2 + 3 + 4 + 30,  1 - 2 + 3 - 4 + 40   =  11 22 40 38
//             row 1:  5 + 10,  6 + 20,  5 + 6 + 7 + 8 + 30,  5 - 6 + 7 - 8 + 40   =  15 26 56 38

/// Makes the input of part 2: {1, 2, 4}, rows 1 2 3 4 and 5 6 7 8.
static Tensor small_input(void)
{
    return tensor_of({1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
}

/// Makes the weight of part 2: {4, 4}, one row per output column, as in the table above.
static Tensor small_weight(void)
{
    return tensor_of({4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, -1, 1, -1});
}

/// Makes the bias of part 2: 10 20 30 40.
static Tensor small_bias(void)
{
    return tensor_of({4}, {10, 20, 30, 40});
}

static void forward_dots_each_input_row_with_each_weight_row_and_adds_the_bias(void)
{
    // Expect: output 11 22 40 38 and 15 26 56 38, exactly (whole numbers this small add exactly in floats).
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 2, 4});
    linear_forward(output_tensor, small_input(), small_weight(), small_bias());
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({11, 22, 40, 38, 15, 26, 56, 38}));
}

static void forward_without_bias_gives_the_dot_products_alone(void)
{
    // Expect: bias Tensor() (none, as the head): output 1 2 10 -2 and 5 6 26 -2, the table above
    // without the 10 20 30 40.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 2, 4});
    linear_forward(output_tensor, small_input(), small_weight(), Tensor());
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({1, 2, 10, -2, 5, 6, 26, -2}));
}

static void backward_sends_each_output_grad_back_through_its_weight_row(void)
{
    // Expect: output_grad 1 only at row 0's output 0 and row 1's output 2, 0 everywhere else:
    //
    //     input_grad[row]  = sum over j of output_grad[row][j] x weight[j]    row 0: weight out 0 = 1 0 0 0
    //                                                                         row 1: weight out 2 = 1 1 1 1
    //     weight_grad[j]   = sum over rows of output_grad[row][j] x input[row] out 0: input row 0 = 1 2 3 4
    //                                                                         out 2: input row 1 = 5 6 7 8
    //     bias_grad[j]     = sum over rows of output_grad[row][j]             1 0 1 0
    //
    // Outputs 1 and 3 got no gradient, so their rows of weight_grad stay 0. All exact.
    // 1. the backward
    Tensor input_grad_tensor = Tensor::zeros({1, 2, 4});
    Tensor weight_grad_tensor = Tensor::zeros({4, 4});
    Tensor bias_grad_tensor = Tensor::zeros({4});
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, tensor_of({1, 2, 4}, {1, 0, 0, 0, 0, 0, 1, 0}),
                    small_input(), small_weight());
    // 2. check
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(weight_grad_tensor.tolist(), std::vector<float>({1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0}));
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({1, 0, 1, 0}));
}

static void backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there), output_grad
    // as in the test above. input_grad is written over: 1 0 0 0 1 1 1 1, the 10s gone. weight_grad and
    // bias_grad are added onto: weight_grad out 0 = 10 + 1 2 3 4 = 11 12 13 14, out 2 = 15 16 17 18, the
    // rest 10; bias_grad 11 10 11 10. A second backward adds again (bias_grad 12 10 12 10) and writes
    // input_grad again, the same 1 0 0 0 1 1 1 1.
    // 1. gradients that start at 10
    Tensor input_tensor = small_input();
    Tensor weight_tensor = small_weight();
    Tensor output_grad_tensor = tensor_of({1, 2, 4}, {1, 0, 0, 0, 0, 0, 1, 0});
    Tensor input_grad_tensor = Tensor::zeros({1, 2, 4}).fill_(10);
    Tensor weight_grad_tensor = Tensor::zeros({4, 4}).fill_(10);
    Tensor bias_grad_tensor = Tensor::zeros({4}).fill_(10);
    // 2. one backward: input_grad written over, the others added onto
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(weight_grad_tensor.tolist(),
              std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 15, 16, 17, 18, 10, 10, 10, 10}));
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({11, 10, 11, 10}));
    // 3. a second backward: the same again
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({12, 10, 12, 10}));
}

/// Makes the identity matrix {size_val, size_val}: 1 where the row is the column, 0 elsewhere.
static Tensor identity(size_t size_val)
{
    std::vector<float> values_list(size_val * size_val, 0.0f);
    for (size_t i_val = 0; i_val < size_val; i_val++)
        values_list[i_val * size_val + i_val] = 1.0f;
    return tensor_of({size_val, size_val}, values_list);
}

static void identity_weight_gives_back_the_input_in_every_square(void)
{
    // Expect: input {2, 100, 256} holding 0, 1, 2, ..., 51199 (each value its own position), weight the
    // identity {256, 256}, no bias: output[row][j] = input[row][j] x 1 + 0s, so output is input, exactly.
    // 200 rows do not fill cuBLAS's squares either: a value written to the wrong place, or not written,
    // shows; so would a matrix handed to cuBLAS turned the wrong way. The backward the same way:
    // output_grad 0 .. 51199 gives input_grad = output_grad · identity = output_grad, exactly.
    // 1. the forward: output is input
    Tensor input_tensor = tensor_of({2, 100, 256}, sample_data(2 * 100 * 256));
    Tensor weight_tensor = identity(256);
    Tensor output_tensor = Tensor::zeros({2, 100, 256});
    linear_forward(output_tensor, input_tensor, weight_tensor, Tensor());
    ASSERT_EQ(output_tensor.tolist(), input_tensor.tolist());
    // 2. the backward: input_grad is output_grad
    Tensor output_grad_tensor = tensor_of({2, 100, 256}, sample_data(2 * 100 * 256));
    Tensor input_grad_tensor = Tensor::zeros({2, 100, 256});
    Tensor weight_grad_tensor = Tensor::zeros({256, 256});
    linear_backward(input_grad_tensor, weight_grad_tensor, Tensor(), output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(input_grad_tensor.tolist(), output_grad_tensor.tolist());
}

static void weight_grad_is_output_grad_turned_round_when_input_is_the_identity(void)
{
    // Expect: input the identity {1, 256, 256} (256 rows, in_features 256), out_features 384, output_grad
    // {1, 256, 384} holding 0, 1, ..., 98303: weight_grad[out][in] = sum over rows of
    // output_grad[row][out] x input[row][in] = output_grad[in][out] = 384 in + out, exactly. This is the
    // product that reads output_grad turned round (CUBLAS_OP_T in B2). bias_grad[out] = sum over
    // the 256 rows of 384 row + out = 384 x 32640 + 256 out = 12,533,760 + 256 out, also exact: every sum
    // on the way is a whole number below 2^24 = 16,777,216, which floats hold exactly.
    // 1. the backward
    Tensor input_tensor = identity(256).view({1, 256, 256});
    Tensor weight_tensor = Tensor::zeros({384, 256});
    Tensor output_grad_tensor = tensor_of({1, 256, 384}, sample_data(256 * 384));
    Tensor input_grad_tensor = Tensor::zeros({1, 256, 256});
    Tensor weight_grad_tensor = Tensor::zeros({384, 256});
    Tensor bias_grad_tensor = Tensor::zeros({384});
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    // 2. the expected values, worked out
    std::vector<float> turned_round_list(384 * 256), column_sums_list(384);
    for (size_t out_val = 0; out_val < 384; out_val++)
    {
        for (size_t in_val = 0; in_val < 256; in_val++)
            turned_round_list[out_val * 256 + in_val] = (float)(384 * in_val + out_val);
        column_sums_list[out_val] = (float)(12533760 + 256 * out_val);
    }
    // 3. check
    ASSERT_EQ(weight_grad_tensor.tolist(), turned_round_list);
    ASSERT_EQ(bias_grad_tensor.tolist(), column_sums_list);
}

static void whole_numbers_come_out_exact_on_mlp_up_sizes(void)
{
    // Expect: GPT-2 small's MLP up, batch_size = 4 rows of seq_len = 1024 tokens, 768 in, 3072 out, with
    // every value 1: input, weight, bias and output_grad. Each output is 768 x 1 x 1 + 1 = 769; each
    // input_grad a sum over 3072 outputs, 3072; each weight_grad and bias_grad a sum over 4096 rows, 4096.
    // All exact: no value is missed or added twice, even where cuBLAS splits weight_grad's sum over the
    // 4096 rows into parts and adds the parts up.
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t in_features_val = 768;
    size_t out_features_val = 3072;
    size_t rows_val = batch_size_val * seq_len_val;
    // 1. the forward, all 1s
    Tensor input_tensor = Tensor::ones({batch_size_val, seq_len_val, in_features_val});
    Tensor weight_tensor = Tensor::ones({out_features_val, in_features_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, out_features_val});
    linear_forward(output_tensor, input_tensor, weight_tensor, Tensor::ones({out_features_val}));
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>(rows_val * out_features_val, 769.0f));
    // 2. the backward, all 1s
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, in_features_val});
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val});
    Tensor bias_grad_tensor = Tensor::zeros({out_features_val});
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                    Tensor::ones({batch_size_val, seq_len_val, out_features_val}), input_tensor, weight_tensor);
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>(rows_val * in_features_val, 3072.0f));
    ASSERT_EQ(weight_grad_tensor.tolist(), std::vector<float>(out_features_val * in_features_val, 4096.0f));
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>(out_features_val, 4096.0f));
}

// ══ part 3: tensors that start off 256 bytes ══════════════════════════════════

/// Makes a tensor of `shape_list` holding `values_list` that starts one float into its memory: a view of a
/// tensor one value longer, from its second value on. As a weight or bias inside one big buffer of all the
/// parameters can.
static Tensor tensor_one_float_in(const Shape &shape_list, const std::vector<float> &values_list)
{
    std::vector<float> longer_list(1, 0.0f);
    longer_list.insert(longer_list.end(), values_list.begin(), values_list.end());
    return tensor_of({longer_list.size()}, longer_list).narrow(0, 1, values_list.size()).view(shape_list);
}

static void works_on_tensors_that_do_not_start_on_256_bytes(void)
{
    // Expect: part 2's numbers, with input, weight, bias, output and every gradient starting 4 bytes past
    // cudaMalloc's 256: output 11 22 40 38 15 26 56 38; input_grad, weight_grad and bias_grad as in
    // backward_sends_each_output_grad_back_through_its_weight_row. cuBLASLt assumes 256 bytes unless told
    // otherwise; told wrongly, it could pick a kernel that reads 16 bytes at a time and stop with
    // "misaligned address", or read the wrong values.
    // 1. the forward, every tensor one float in
    Tensor input_tensor = tensor_one_float_in({1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor weight_tensor = tensor_one_float_in({4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, -1, 1, -1});
    Tensor bias_tensor = tensor_one_float_in({4}, {10, 20, 30, 40});
    Tensor output_tensor = tensor_one_float_in({1, 2, 4}, std::vector<float>(8, 0.0f));
    print_test_name(__func__);
    printf("    input, weight, bias, output start %zu, %zu, %zu, %zu bytes past %d\n",
           (size_t)input_tensor.data() % LINEAR_CUBLAS_LARGEST_ALIGNMENT,
           (size_t)weight_tensor.data() % LINEAR_CUBLAS_LARGEST_ALIGNMENT,
           (size_t)bias_tensor.data() % LINEAR_CUBLAS_LARGEST_ALIGNMENT,
           (size_t)output_tensor.data() % LINEAR_CUBLAS_LARGEST_ALIGNMENT, LINEAR_CUBLAS_LARGEST_ALIGNMENT);
    linear_forward(output_tensor, input_tensor, weight_tensor, bias_tensor);
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({11, 22, 40, 38, 15, 26, 56, 38}));
    // 2. the backward, every tensor one float in
    Tensor output_grad_tensor = tensor_one_float_in({1, 2, 4}, {1, 0, 0, 0, 0, 0, 1, 0});
    Tensor input_grad_tensor = tensor_one_float_in({1, 2, 4}, std::vector<float>(8, 0.0f));
    Tensor weight_grad_tensor = tensor_one_float_in({4, 4}, std::vector<float>(16, 0.0f));
    Tensor bias_grad_tensor = tensor_one_float_in({4}, std::vector<float>(4, 0.0f));
    linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor, weight_tensor);
    ASSERT_EQ(input_grad_tensor.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(weight_grad_tensor.tolist(), std::vector<float>({1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0}));
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({1, 0, 1, 0}));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_qkv_sizes();
    matches_pytorch_on_sizes_that_do_not_fill_the_squares();
    matches_pytorch_without_bias();
    forward_and_backward_in_bfloat16();

    forward_dots_each_input_row_with_each_weight_row_and_adds_the_bias();
    forward_without_bias_gives_the_dot_products_alone();
    backward_sends_each_output_grad_back_through_its_weight_row();
    backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad();
    identity_weight_gives_back_the_input_in_every_square();
    weight_grad_is_output_grad_turned_round_when_input_is_the_identity();
    whole_numbers_come_out_exact_on_mlp_up_sizes();

    works_on_tensors_that_do_not_start_on_256_bytes();

    printf("\nall tests passed\n");
    return 0;
}
