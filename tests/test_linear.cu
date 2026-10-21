// test_linear.cu — tests for llmc/kernels/linear.cuh (linear_forward, linear_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's linear, with autograd for the backward, and times them on
// GPT-2 small's 5 linear layers; part 2 checks numbers worked out by hand; part 3 checks that each
// kernel's registers fit its block. (The C# trainer had no linear test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/kernels/linear.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape)
{
    return Tensor::randn(shape).tolist();
}

/// Makes a tensor of `shape` on the GPU holding `values`, given in row order.
static Tensor tensor_of(const Shape &shape, const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty(shape);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes a tensor of `shape` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes minutes for the head's 206M values.
static Tensor random_tensor_on_gpu(const Shape &shape)
{
    Tensor tensor = Tensor::empty(shape);
    torch::Tensor values = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes, on_gpu).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu.data_ptr<float>(), on_cpu.data_ptr<float>() + on_cpu.numel());
}

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

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values, mine and PyTorch's linear with autograd, and
/// checks that output and the gradients agree within TOLERANCE. Prints the sizes under `test`'s name
/// first. input_shape is {batch_size, seq_len, in_features}; with_bias false: Tensor() for mine, no bias
/// for PyTorch's. The gradients start at 0.
static void forward_and_backward_match_pytorch(const char *test, const Shape &input_shape, size_t out_features,
                                               bool with_bias)
{
    Tensor::manual_seed(0);
    size_t in_features = input_shape.back(), rows = input_shape[0] * input_shape[1];
    Shape output_shape = {input_shape[0], input_shape[1], out_features};
    Tensor input = tensor_of(input_shape, random_values(input_shape));
    Tensor weight = tensor_of({out_features, in_features}, random_values({out_features, in_features}));
    Tensor bias = with_bias ? tensor_of({out_features}, random_values({out_features})) : Tensor();
    Tensor output_grad = tensor_of(output_shape, random_values(output_shape));
    Tensor output = Tensor::zeros(output_shape), input_grad = Tensor::zeros(input_shape);
    Tensor weight_grad = Tensor::zeros({out_features, in_features});
    Tensor bias_grad = with_bias ? Tensor::zeros({out_features}) : Tensor();
    linear_forward(output, input, weight, bias);
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_weight = pytorch_copy_of(weight).requires_grad_(true);
    torch::Tensor pytorch_bias = with_bias ? pytorch_copy_of(bias).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output = with_bias ? torch::linear(pytorch_input, pytorch_weight, pytorch_bias)
                                             : torch::linear(pytorch_input, pytorch_weight);
    pytorch_output.backward(pytorch_copy_of(output_grad));

    print_test_name(test);
    printf("    %zu rows, in_features %zu, out_features %zu, %s\n", rows, in_features, out_features,
           with_bias ? "with bias" : "no bias");
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), TOLERANCE);
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), TOLERANCE);
    ASSERT_NEAR(weight_grad.tolist(), list_of(pytorch_weight.grad()), TOLERANCE);
    if (with_bias)
        ASSERT_NEAR(bias_grad.tolist(), list_of(pytorch_bias.grad()), TOLERANCE);
}

static void matches_pytorch_on_qkv_sizes(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens (256 rows: 2 blocks down), in_features = 768,
    // out_features = 2304 as GPT-2 small's qkv (18 blocks across), random input, weight, bias and
    // output_grad: output, input_grad, weight_grad and bias_grad equal what PyTorch's linear and its
    // autograd give, within 1e-5. Not to the last bit: the two add up each dot product in another order.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 2304, true);
}

static void matches_pytorch_on_sizes_that_do_not_fill_the_squares(void)
{
    // Expect: 3 x 50 = 150 rows (a block of 128 down, then one of 22), in_features = 100 (12 steps of 8,
    // then one of 4), out_features = 300 (two blocks of 128 across, then one of 44): the same as PyTorch
    // within 1e-5. Past the edges the kernel reads 0s and writes nothing. The backward's products have
    // part-full squares and steps too: weight_grad's inner dim is the 150 rows, 18 steps of 8, then 6.
    forward_and_backward_match_pytorch(__func__, {3, 50, 100}, 300, true);
}

static void matches_pytorch_without_bias(void)
{
    // Expect: 256 rows, in_features = 768, out_features = 1000, no bias, like the head with a small
    // vocabulary (1000 is not a multiple of 128 either): the same as PyTorch within 1e-5.
    forward_and_backward_match_pytorch(__func__, {4, 64, 768}, 1000, false);
}

/// Times 20 forwards and 20 backwards of one linear layer, batch_size = 4 rows of seq_len = 1024
/// tokens, mine against PyTorch's linear with autograd, and prints the averages and TFLOPS under
/// `test`'s name. Both sides write input_grad and add 20 backwards onto weight_grad and bias_grad; at
/// the end they must still agree within `tolerance`.
static void time_20_steps(const char *test, size_t in_features, size_t out_features, bool with_bias,
                          double tolerance)
{
    torch::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, rows = batch_size * seq_len;
    Tensor input = random_tensor_on_gpu({batch_size, seq_len, in_features});
    Tensor weight = random_tensor_on_gpu({out_features, in_features});
    Tensor bias = with_bias ? random_tensor_on_gpu({out_features}) : Tensor();
    Tensor output_grad = random_tensor_on_gpu({batch_size, seq_len, out_features});
    Tensor output = Tensor::zeros({batch_size, seq_len, out_features});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, in_features});
    Tensor weight_grad = Tensor::zeros({out_features, in_features});
    Tensor bias_grad = with_bias ? Tensor::zeros({out_features}) : Tensor();

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_weight = pytorch_copy_of(weight).requires_grad_(true);
    torch::Tensor pytorch_bias = with_bias ? pytorch_copy_of(bias).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_grad = pytorch_copy_of(output_grad);
    torch::Tensor pytorch_output;

    // Steps 1 and 2 do first-time work, so the averages are of 3..20: in step 1 CUDA loads each side's
    // kernels; in step 2 PyTorch's backward first adds onto the .grad that step 1 made.
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { linear_forward(output, input, weight, bias); });
        times[1] = gpu_milliseconds(
            [&] { linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight); });
        times[2] = gpu_milliseconds([&] {
            pytorch_output = with_bias ? torch::linear(pytorch_input, pytorch_weight, pytorch_bias)
                                       : torch::linear(pytorch_input, pytorch_weight);
        });
        pytorch_input.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times[3] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    double forward_gflop = 2.0 * rows * in_features * out_features / 1e9; // a multiply and an add per pair
    double work[4] = {forward_gflop, 2 * forward_gflop, forward_gflop, 2 * forward_gflop};
    print_test_name(test);
    printf("    input {4, 1024, %zu}, weight {%zu, %zu}: %.1f GFLOP forward, %.1f GFLOP backward\n", in_features,
           out_features, in_features, forward_gflop, 2 * forward_gflop);
    printf("    average of steps 3..20:   my forward   my backward   PyTorch forward   PyTorch backward\n");
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("        TFLOPS                %10.2f   %11.2f   %15.2f   %16.2f\n", work[0] / (totals[0] / 18),
           work[1] / (totals[1] / 18), work[2] / (totals[2] / 18), work[3] / (totals[3] / 18));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), tolerance);
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), tolerance);
    ASSERT_NEAR(weight_grad.tolist(), list_of(pytorch_weight.grad()), tolerance);
    if (with_bias)
        ASSERT_NEAR(bias_grad.tolist(), list_of(pytorch_bias.grad()), tolerance);
}

static void time_20_steps_on_qkv(void)
{
    // Prints the times of GPT-2 small's qkv: in_features = 768, out_features = 2304, with bias. A forward
    // is 2 x 4096 x 768 x 2304 = 14.5 GFLOP: at a T4's 8.1 TFLOPS (fp32) at least 1.79 ms; a backward is
    // twice the work, at least 3.58 ms.
    time_20_steps(__func__, 768, 2304, true, TOLERANCE);
}

static void time_20_steps_on_attention_output(void)
{
    // Prints the times of the attention's output: 768 in, 768 out, with bias. 4.8 GFLOP: at least
    // 0.60 ms forward, 1.19 ms backward. weight_grad's product is only 768 x 768: 6 x 6 = 36 blocks, each
    // walking 4096 inner values, on a GPU of 40 SMs.
    time_20_steps(__func__, 768, 768, true, TOLERANCE);
}

static void time_20_steps_on_mlp_up(void)
{
    // Prints the times of the MLP's first linear: 768 in, 3072 out, with bias. 19.3 GFLOP: at least
    // 2.39 ms forward, 4.77 ms backward.
    time_20_steps(__func__, 768, 3072, true, TOLERANCE);
}

static void time_20_steps_on_mlp_down(void)
{
    // Prints the times of the MLP's second linear: 3072 in, 768 out, with bias. The same 19.3 GFLOP as
    // the MLP's first, in another shape: the forward has 6 x 32 = 192 blocks of 384 steps, against
    // 24 x 32 = 768 blocks of 96 steps.
    time_20_steps(__func__, 3072, 768, true, TOLERANCE);
}

static void time_20_steps_on_head(void)
{
    // Prints the times of the head: 768 in, 50304 out (wte's rows), no bias. 316.5 GFLOP: at least
    // 39.1 ms forward, 78.1 ms backward. output and output_grad are {4, 1024, 50304}, 824 MB each.
    //
    // Checked within 1e-4, not 1e-5: each input_grad value here adds up 50,304 products, and rounding
    // drifts with the length of the sum. Worked out for random values, the two sides should differ by
    // about 5e-6 of the largest value: under 1e-5, but too close for a check that stops the program.
    time_20_steps(__func__, 768, 50304, false, 1e-4);
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
    Tensor output = Tensor::zeros({1, 2, 4});
    linear_forward(output, small_input(), small_weight(), small_bias());
    ASSERT_EQ(output.tolist(), std::vector<float>({11, 22, 40, 38, 15, 26, 56, 38}));
}

static void forward_without_bias_gives_the_dot_products_alone(void)
{
    // Expect: bias Tensor() (none, as the head): output 1 2 10 -2 and 5 6 26 -2, the table above
    // without the 10 20 30 40.
    Tensor output = Tensor::zeros({1, 2, 4});
    linear_forward(output, small_input(), small_weight(), Tensor());
    ASSERT_EQ(output.tolist(), std::vector<float>({1, 2, 10, -2, 5, 6, 26, -2}));
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
    Tensor input = small_input(), weight = small_weight();
    Tensor output_grad = tensor_of({1, 2, 4}, {1, 0, 0, 0, 0, 0, 1, 0});
    Tensor input_grad = Tensor::zeros({1, 2, 4}), weight_grad = Tensor::zeros({4, 4}), bias_grad = Tensor::zeros({4});
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);
    ASSERT_EQ(input_grad.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(weight_grad.tolist(), std::vector<float>({1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0}));
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({1, 0, 1, 0}));
}

static void backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there), output_grad
    // as in the test above. input_grad is written over: 1 0 0 0 1 1 1 1, the 10s gone. weight_grad and
    // bias_grad are added onto: weight_grad out 0 = 10 + 1 2 3 4 = 11 12 13 14, out 2 = 15 16 17 18, the
    // rest 10; bias_grad 11 10 11 10. A second backward adds again (bias_grad 12 10 12 10) and writes
    // input_grad again, the same 1 0 0 0 1 1 1 1.
    Tensor input = small_input(), weight = small_weight();
    Tensor output_grad = tensor_of({1, 2, 4}, {1, 0, 0, 0, 0, 0, 1, 0});
    Tensor input_grad = Tensor::zeros({1, 2, 4}).fill_(10);
    Tensor weight_grad = Tensor::zeros({4, 4}).fill_(10), bias_grad = Tensor::zeros({4}).fill_(10);
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);
    ASSERT_EQ(input_grad.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(weight_grad.tolist(),
              std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 15, 16, 17, 18, 10, 10, 10, 10}));
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({11, 10, 11, 10}));
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);
    ASSERT_EQ(input_grad.tolist(), std::vector<float>({1, 0, 0, 0, 1, 1, 1, 1}));
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({12, 10, 12, 10}));
}

/// Makes the identity matrix {size, size}: 1 where the row is the column, 0 elsewhere.
static Tensor identity(size_t size)
{
    std::vector<float> values(size * size, 0.0f);
    for (size_t i = 0; i < size; i++)
        values[i * size + i] = 1.0f;
    return tensor_of({size, size}, values);
}

static void identity_weight_gives_back_the_input_in_every_square(void)
{
    // Expect: input {2, 100, 256} holding 0, 1, 2, ..., 51199 (each value its own position), weight the
    // identity {256, 256}, no bias: output[row][j] = input[row][j] x 1 + 0s, so output is input, exactly.
    // 200 rows and 256 columns are 2 x 2 blocks, the bottom ones part-full (72 rows): a value written to
    // the wrong place, or not written, shows. The backward the same way: output_grad 0 .. 51199 gives
    // input_grad = output_grad · identity = output_grad, exactly.
    Tensor input = tensor_of({2, 100, 256}, sample_data(2 * 100 * 256));
    Tensor weight = identity(256);
    Tensor output = Tensor::zeros({2, 100, 256});
    linear_forward(output, input, weight, Tensor());
    ASSERT_EQ(output.tolist(), input.tolist());

    Tensor output_grad = tensor_of({2, 100, 256}, sample_data(2 * 100 * 256));
    Tensor input_grad = Tensor::zeros({2, 100, 256}), weight_grad = Tensor::zeros({256, 256});
    linear_backward(input_grad, weight_grad, Tensor(), output_grad, input, weight);
    ASSERT_EQ(input_grad.tolist(), output_grad.tolist());
}

static void weight_grad_is_output_grad_turned_round_when_input_is_the_identity(void)
{
    // Expect: input the identity {1, 256, 256} (256 rows, in_features 256), out_features 384, output_grad
    // {1, 256, 384} holding 0, 1, ..., 98303: weight_grad[out][in] = sum over rows of
    // output_grad[row][out] x input[row][in] = output_grad[in][out] = 384 in + out, exactly. This is the
    // product that reads output_grad turned round; 384 x 256 is 3 x 2 blocks. bias_grad[out] = sum over
    // the 256 rows of 384 row + out = 384 x 32640 + 256 out = 12,533,760 + 256 out, also exact: every sum
    // on the way is a whole number below 2^24 = 16,777,216, which floats hold exactly.
    Tensor input = identity(256).view({1, 256, 256});
    Tensor weight = Tensor::zeros({384, 256});
    Tensor output_grad = tensor_of({1, 256, 384}, sample_data(256 * 384));
    Tensor input_grad = Tensor::zeros({1, 256, 256});
    Tensor weight_grad = Tensor::zeros({384, 256}), bias_grad = Tensor::zeros({384});
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);

    std::vector<float> turned_round(384 * 256), column_sums(384);
    for (size_t out = 0; out < 384; out++)
    {
        for (size_t in = 0; in < 256; in++)
            turned_round[out * 256 + in] = (float)(384 * in + out);
        column_sums[out] = (float)(12533760 + 256 * out);
    }
    ASSERT_EQ(weight_grad.tolist(), turned_round);
    ASSERT_EQ(bias_grad.tolist(), column_sums);
}

static void whole_numbers_come_out_exact_on_mlp_up_sizes(void)
{
    // Expect: GPT-2 small's MLP up, batch_size = 4 rows of seq_len = 1024 tokens, 768 in, 3072 out, with
    // every value 1: input, weight, bias and output_grad. Each output is 768 x 1 x 1 + 1 = 769; each
    // input_grad a sum over 3072 outputs, 3072; each weight_grad and bias_grad a sum over 4096 rows, 4096.
    // All exact, every block's square: no value is missed or added twice. The two launches are 24 x 32 = 768
    // blocks and 6 x (24 + 32) = 336: weight_grad's squares, then input_grad's.
    size_t batch_size = 4, seq_len = 1024, in_features = 768, out_features = 3072;
    size_t rows = batch_size * seq_len;
    Tensor input = Tensor::ones({batch_size, seq_len, in_features});
    Tensor weight = Tensor::ones({out_features, in_features}), bias = Tensor::ones({out_features});
    Tensor output = Tensor::zeros({batch_size, seq_len, out_features});
    linear_forward(output, input, weight, bias);
    ASSERT_EQ(output.tolist(), std::vector<float>(rows * out_features, 769.0f));

    Tensor output_grad = Tensor::ones({batch_size, seq_len, out_features});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, in_features});
    Tensor weight_grad = Tensor::zeros({out_features, in_features}), bias_grad = Tensor::zeros({out_features});
    linear_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight);
    ASSERT_EQ(input_grad.tolist(), std::vector<float>(rows * in_features, 3072.0f));
    ASSERT_EQ(weight_grad.tolist(), std::vector<float>(out_features * in_features, 4096.0f));
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>(out_features, 4096.0f));
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: linear_forward_kernel, linear_weight_grad_kernel and linear_input_grad_kernel can each be
    // launched with LINEAR_THREADS_PER_BLOCK (256), and nothing is spilled. A thread keeps its 64 sums and the
    // 16 values it multiplies in registers (weight_grad's also its 8 sums of bias_grad); a spill would move some to
    // local memory, which is GPU memory, and the kernel would wait on it. cudaFuncGetAttributes reads
    // what the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)linear_forward_kernel, (const void *)linear_weight_grad_kernel,
                             (const void *)linear_input_grad_kernel};
    const char *names[] = {"linear_forward_kernel", "linear_weight_grad_kernel", "linear_input_grad_kernel"};
    for (int k = 0; k < 3; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= LINEAR_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_on_qkv_sizes();
    matches_pytorch_on_sizes_that_do_not_fill_the_squares();
    matches_pytorch_without_bias();
    time_20_steps_on_qkv();
    time_20_steps_on_attention_output();
    time_20_steps_on_mlp_up();
    time_20_steps_on_mlp_down();
    time_20_steps_on_head();

    forward_dots_each_input_row_with_each_weight_row_and_adds_the_bias();
    forward_without_bias_gives_the_dot_products_alone();
    backward_sends_each_output_grad_back_through_its_weight_row();
    backward_writes_input_grad_and_adds_onto_weight_grad_and_bias_grad();
    identity_weight_gives_back_the_input_in_every_square();
    weight_grad_is_output_grad_turned_round_when_input_is_the_identity();
    whole_numbers_come_out_exact_on_mlp_up_sizes();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
