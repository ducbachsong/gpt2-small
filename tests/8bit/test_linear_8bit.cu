// test_linear_8bit.cu — tests for llmc/kernels/8bit/linear_8bit.cuh (linear_8bit_forward, linear_8bit_backward): the
// linear layer of a bfloat16 model with its three products in 8 bits, each side rounded to E4M3 by round_8bit
// (turned while rounding where it sits turned round), then wgmma_matmul_8bit on an H100's tensor cores. Every
// check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks each result against PyTorch's float product of the very E4M3 copies the layer made (read back
// from its scratch: that the right sides were rounded, turned the right way and multiplied), then against
// PyTorch's float linear on the values before rounding (how far 8 bits move it); part 2 checks numbers that must
// come out exactly. linear_8bit has no kernel of its own: round_8bit is tested in tests/8bit/test_round_8bit.cu, the
// products in tests/8bit/test_wgmma_matmul_8bit.cu, bias_grad's kernel in tests/test_linear_bf16.cu. Run it on an H100
// (compute capability 9.0), built with -arch=sm_90a in place of -arch=native. The timings on GPT-2 small's 5
// linear layers are in tunes/tune_linear_8bit.cu.
#include <torch/types.h>              // torch::linear, torch::randn, float8_e4m3fn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/8bit/linear_8bit.cuh"

// Against PyTorch in float on the same E4M3 copies: the sums are float on both sides, then mine are rounded to
// bfloat16, by up to 2^-9 (0.2%) of each value. Against the values before rounding: each E4M3 is off by up to
// 2^-4 (6%) of its value, so a product moves by a few percent. bias_grad is not rounded to 8 bits: a float sum of
// the bfloat16 output_grad, rounded once. Each tolerance is the largest difference, as a fraction of the largest
// value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double EIGHT_BIT_TOLERANCE = 1e-1;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
/// on the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded to
/// bfloat16.
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

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);  // the source_tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives an 8-bit copy turned back, by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times its row's scale,
/// exact. {rows, k} float.
///
///   rounded_tensor   {rows, k / ROUND_8BIT_VALUES_PER_INT} Int32, as round_8bit wrote it
///   scales_tensor    {rows} float
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scales_tensor)
{
    int64_t rows_val = (int64_t)rounded_tensor.size(0);
    int64_t k_val = (int64_t)(rounded_tensor.size(1) * ROUND_8BIT_VALUES_PER_INT);
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, k_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
}

/// Gives PyTorch's float product left · rightᵀ of one product's E4M3 copies, turned back: what wgmma_matmul_8bit
/// should give before rounding to bfloat16.
///
///   copies_val   one of the scratch's three products
static torch::Tensor product_of_copies(const Linear8bitProductCopies &copies_val)
{
    return turned_back(copies_val.left_8bit_tensor, copies_val.left_scales_tensor)
        .matmul(turned_back(copies_val.right_8bit_tensor, copies_val.right_scales_tensor).t());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random bfloat16 values, then checks: each product against PyTorch's float
/// product of the E4M3 copies in the scratch within BFLOAT16_TOLERANCE (output with the bias added), bias_grad
/// against PyTorch's within BFLOAT16_TOLERANCE, and output and the gradients against PyTorch's float linear and its
/// autograd on the values before rounding within EIGHT_BIT_TOLERANCE. Prints the sizes and the differences under
/// `test_ptr`'s name. input_shape is {batch_size, seq_len, in_features}; with_bias false: Tensor() for mine, no
/// bias for PyTorch's. The gradients start at 0.
static void forward_and_backward_match_pytorch(const char *test_ptr, const Shape &input_shape_list,
                                               size_t out_features_val, bool with_bias_val)
{
    // 1. random tensors in bfloat16; mine
    torch::manual_seed(0);
    size_t in_features_val = input_shape_list.back();
    size_t rows_val = input_shape_list[0] * input_shape_list[1];
    Shape output_shape_list = {input_shape_list[0], input_shape_list[1], out_features_val};
    Tensor input_tensor = random_bf16_tensor(input_shape_list);
    Tensor weight_tensor = random_bf16_tensor({out_features_val, in_features_val});
    Tensor bias_tensor = with_bias_val ? random_bf16_tensor({out_features_val}) : Tensor();
    Tensor output_grad_tensor = random_bf16_tensor(output_shape_list);
    Tensor output_tensor = Tensor::zeros(output_shape_list, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros(input_shape_list, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}, DType::BFloat16) : Tensor();
    Linear8bitScratch scratch_val(rows_val, in_features_val, out_features_val);
    linear_8bit_forward(output_tensor, input_tensor, weight_tensor, bias_tensor, scratch_val);
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                         weight_tensor, scratch_val);

    // 2. PyTorch's float products of the E4M3 copies the layer made
    torch::Tensor copies_output_tensor = product_of_copies(scratch_val.output_copies);
    if (with_bias_val)
        copies_output_tensor = copies_output_tensor + pytorch_copy_of(bias_tensor);
    torch::Tensor copies_input_grad_tensor = product_of_copies(scratch_val.input_grad_copies);
    torch::Tensor copies_weight_grad_tensor = product_of_copies(scratch_val.weight_grad_copies);

    // 3. PyTorch's float linear with autograd, on the values before rounding
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = with_bias_val ? pytorch_copy_of(bias_tensor).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_tensor = with_bias_val
                                              ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                              : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor));

    // 4. print the sizes, then check: against the copies within one rounding, then against the values before
    //    rounding within 8 bits' error
    print_test_name(test_ptr);
    printf("    %zu rows, in_features %zu, out_features %zu, %s\n", rows_val, in_features_val, out_features_val,
           with_bias_val ? "with bias" : "no bias");
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, copies_output_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(input_grad_tensor, copies_input_grad_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(weight_grad_tensor, copies_weight_grad_tensor), 0.0, BFLOAT16_TOLERANCE);
    if (with_bias_val)
        ASSERT_NEAR(relative_difference_on_gpu(bias_grad_tensor, pytorch_bias_tensor.grad()), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, pytorch_output_tensor), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(input_grad_tensor, pytorch_input_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(weight_grad_tensor, pytorch_weight_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
}

static void matches_pytorch_on_qkv_sizes(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens (256 rows), in_features = 768, out_features = 2304 as
    // GPT-2 small's qkv, random input, weight, bias and output_grad in bfloat16: output, input_grad and weight_grad
    // within BFLOAT16_TOLERANCE of PyTorch's products of the E4M3 copies, bias_grad within it of PyTorch's, and all
    // within EIGHT_BIT_TOLERANCE of PyTorch's float linear on the values before rounding.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 2304, true);
}

static void matches_pytorch_on_sizes_that_do_not_fill_the_squares(void)
{
    // Expect: input {2, 40, 96}, weight {144, 96} and bias: 80 rows (64 + 16) and 144 columns (2 x 64 + 16), so
    // wgmma_matmul_8bit's squares at the edges are part-full; each k is less than or not a whole number of its
    // steps of 128: in_features 96 (output), out_features 144 (input_grad: 128 then 16), rows 80 (weight_grad), TMA
    // giving 0s past the edge. The turned roundings' 32 x 32 tiles are part-full too. The same checks.
    forward_and_backward_match_pytorch(__func__, {2, 40, 96}, 144, true);
}

static void matches_pytorch_without_bias(void)
{
    // Expect: 256 rows, in_features = 768, out_features = 1008, no bias, like the head with a small vocabulary
    // (1008: a multiple of 16, as every size here must be, but not of 64): the same checks, without bias_grad.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 1008, false);
}

// ══ part 2: numbers that must come out exactly ═════════════════════════════════
// A whole number from 0 to 15 has at most 4 significant bits, which E4M3 holds (3 stored and 1 implied); a row of
// them whose largest is 15 has scale 2^-4 (15 x 16 = 240 <= 448, 15 x 32 = 480 is not), and 16 x a 4-bit number
// still has 4 significant bits: every value is rounded exactly. A row of 0s and 1s has scale 2^-8 (1 x 256 =
// 256): exact too. So the products below add up exact values in float, and their whole-number results are what
// bfloat16 stores.

/// Makes the identity matrix {size_val, size_val} in bfloat16: 1 where the row is the column, 0 elsewhere.
static Tensor identity(size_t size_val)
{
    std::vector<float> values_list(size_val * size_val, 0.0f);
    for (size_t diagonal_index_val = 0; diagonal_index_val < size_val; diagonal_index_val++)
        values_list[diagonal_index_val * size_val + diagonal_index_val] = 1.0f;
    return bf16_tensor_of({size_val, size_val}, values_list);
}

/// Makes `count_val` values that E4M3 holds exactly in a row of 16 or more: value i is i % 16.
static std::vector<float> whole_numbers_below_16(size_t count_val)
{
    std::vector<float> values_list(count_val);
    for (size_t value_index_val = 0; value_index_val < count_val; value_index_val++)
        values_list[value_index_val] = (float)(value_index_val % 16);
    return values_list;
}

static void identity_weight_gives_back_the_input_in_every_square(void)
{
    // Expect: input {2, 104, 256} holding 0, 1, ..., 15, 0, 1, ... (value i at position i, % 16), weight the
    // identity {256, 256}, no bias: output[row][j] = input[row][j] x 1 + 0s, so output is input, exactly. 208 rows
    // (13 x 16) and 256 columns are 4 x 4 squares of 64, the bottom ones part-full (16 rows): a value written to
    // the wrong place, or not written, shows. The backward the same way: output_grad the same values gives
    // input_grad = output_grad · identity = output_grad, exactly (the identity turned round is the identity).
    // 1. the forward: output is input
    Linear8bitScratch scratch_val(2 * 104, 256, 256);
    Tensor input_tensor = bf16_tensor_of({2, 104, 256}, whole_numbers_below_16(2 * 104 * 256));
    Tensor weight_tensor = identity(256);
    Tensor output_tensor = Tensor::zeros({2, 104, 256}, DType::BFloat16);
    linear_8bit_forward(output_tensor, input_tensor, weight_tensor, Tensor(), scratch_val);
    ASSERT_EQ(values_of(output_tensor), values_of(input_tensor));
    // 2. the backward: input_grad is output_grad
    Tensor output_grad_tensor = bf16_tensor_of({2, 104, 256}, whole_numbers_below_16(2 * 104 * 256));
    Tensor input_grad_tensor = Tensor::zeros({2, 104, 256}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({256, 256}, DType::BFloat16);
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, Tensor(), output_grad_tensor, input_tensor, weight_tensor,
                         scratch_val);
    ASSERT_EQ(values_of(input_grad_tensor), values_of(output_grad_tensor));
}

static void weight_grad_is_output_grad_turned_round_when_input_is_the_identity(void)
{
    // Expect: input the identity {1, 256, 256} (256 rows, in_features 256), out_features 384, output_grad
    // {1, 256, 384} holding (row + out) % 16: weight_grad[out][in] = sum over rows of output_grad[row][out] x
    // input[row][in] = output_grad[in][out] = (in + out) % 16, exactly. This product reads output_grad and input
    // both turned round while rounding (each turned row, a column of output_grad, holds all of 0..15, so its scale
    // is 2^-4). bias_grad[out] = sum over the 256 rows of (row + out) % 16 = 16 x (0 + 1 + ... + 15) = 1920 for
    // every out, which bfloat16 holds (15 x 128).
    // 1. the backward
    Linear8bitScratch scratch_val(256, 256, 384);
    Tensor input_tensor = identity(256).view({1, 256, 256});
    Tensor weight_tensor = Tensor::zeros({384, 256}, DType::BFloat16);
    std::vector<float> output_grad_list(256 * 384);
    for (size_t row_index_val = 0; row_index_val < 256; row_index_val++)
        for (size_t out_feature_index_val = 0; out_feature_index_val < 384; out_feature_index_val++)
            output_grad_list[row_index_val * 384 + out_feature_index_val] = (float)((row_index_val + out_feature_index_val) % 16);
    Tensor output_grad_tensor = bf16_tensor_of({1, 256, 384}, output_grad_list);
    Tensor input_grad_tensor = Tensor::zeros({1, 256, 256}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({384, 256}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({384}, DType::BFloat16);
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                         weight_tensor, scratch_val);
    // 2. the expected values, worked out
    std::vector<float> turned_round_list(384 * 256);
    for (size_t out_feature_index_val = 0; out_feature_index_val < 384; out_feature_index_val++)
        for (size_t in_feature_index_val = 0; in_feature_index_val < 256; in_feature_index_val++)
            turned_round_list[out_feature_index_val * 256 + in_feature_index_val] = (float)((in_feature_index_val + out_feature_index_val) % 16);
    // 3. check
    ASSERT_EQ(values_of(weight_grad_tensor), turned_round_list);
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>(384, 1920.0f));
}

static void backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad(void)
{
    // Expect: input {1, 16, 16}, weight {16, 16} and output_grad {1, 16, 16} all 1s; every gradient starts at 10
    // (as if an earlier micro-batch had left it there). input_grad is written over: each a sum over 16 outputs of
    // 1 x 1 = 16, the 10s gone. weight_grad and bias_grad are added onto: each a sum over 16 rows, 10 + 16 = 26. A
    // second backward writes input_grad again (16) and adds again: 42. All exact in bfloat16.
    // 1. gradients that start at 10
    Linear8bitScratch scratch_val(16, 16, 16);
    Tensor input_tensor = Tensor::ones({1, 16, 16}).to(DType::BFloat16);
    Tensor weight_tensor = Tensor::ones({16, 16}).to(DType::BFloat16);
    Tensor output_grad_tensor = Tensor::ones({1, 16, 16}).to(DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros({1, 16, 16}, DType::BFloat16).fill_(10);
    Tensor weight_grad_tensor = Tensor::zeros({16, 16}, DType::BFloat16).fill_(10);
    Tensor bias_grad_tensor = Tensor::zeros({16}, DType::BFloat16).fill_(10);
    // 2. one backward: input_grad written over, the others added onto
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                         weight_tensor, scratch_val);
    ASSERT_EQ(values_of(input_grad_tensor), std::vector<float>(16 * 16, 16.0f));
    ASSERT_EQ(values_of(weight_grad_tensor), std::vector<float>(16 * 16, 26.0f));
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>(16, 26.0f));
    // 3. a second backward: input_grad the same, the others added onto again
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                         weight_tensor, scratch_val);
    ASSERT_EQ(values_of(input_grad_tensor), std::vector<float>(16 * 16, 16.0f));
    ASSERT_EQ(values_of(weight_grad_tensor), std::vector<float>(16 * 16, 42.0f));
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>(16, 42.0f));
}

static void whole_numbers_come_out_exact_on_mlp_up_sizes(void)
{
    // Expect: GPT-2 small's MLP up, batch_size = 4 rows of seq_len = 1024 tokens, 768 in, 3072 out, with every
    // value 1: input, weight, bias and output_grad, each 1 rounded to E4M3 exactly. Each output is 768 x 1 x 1 + 1
    // = 769 in float; bfloat16 holds only every 4th whole number between 512 and 1024, so it is stored as the
    // nearest, 768. Each input_grad is a sum over 3072 outputs, 3072; each weight_grad and bias_grad a sum over
    // 4096 rows, 4096: both held exactly. Every square of every product: no value is missed or added twice.
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t in_features_val = 768;
    size_t out_features_val = 3072;
    size_t rows_val = batch_size_val * seq_len_val;
    Linear8bitScratch scratch_val(rows_val, in_features_val, out_features_val);
    // 1. the forward, all 1s
    Tensor input_tensor = Tensor::ones({batch_size_val, seq_len_val, in_features_val}).to(DType::BFloat16);
    Tensor weight_tensor = Tensor::ones({out_features_val, in_features_val}).to(DType::BFloat16);
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, out_features_val}, DType::BFloat16);
    linear_8bit_forward(output_tensor, input_tensor, weight_tensor, Tensor::ones({out_features_val}).to(DType::BFloat16),
                        scratch_val);
    ASSERT_EQ(values_of(output_tensor), std::vector<float>(rows_val * out_features_val, 768.0f));
    // 2. the backward, all 1s
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, in_features_val}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = Tensor::zeros({out_features_val}, DType::BFloat16);
    linear_8bit_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                         Tensor::ones({batch_size_val, seq_len_val, out_features_val}).to(DType::BFloat16), input_tensor,
                         weight_tensor, scratch_val);
    ASSERT_EQ(values_of(input_grad_tensor), std::vector<float>(rows_val * in_features_val, 3072.0f));
    ASSERT_EQ(values_of(weight_grad_tensor), std::vector<float>(out_features_val * in_features_val, 4096.0f));
    ASSERT_EQ(values_of(bias_grad_tensor), std::vector<float>(out_features_val, 4096.0f));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_qkv_sizes();
    matches_pytorch_on_sizes_that_do_not_fill_the_squares();
    matches_pytorch_without_bias();

    identity_weight_gives_back_the_input_in_every_square();
    weight_grad_is_output_grad_turned_round_when_input_is_the_identity();
    backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad();
    whole_numbers_come_out_exact_on_mlp_up_sizes();

    printf("\nall tests passed\n");
    return 0;
}
