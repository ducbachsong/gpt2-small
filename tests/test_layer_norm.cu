// test_layer_norm.cu — tests for llmc/kernels/layer_norm.cuh (layer_norm_forward, layer_norm_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's layer_norm, with autograd for the backward, and times them
// on GPT-2 small's sizes; part 2 checks numbers worked out by hand; part 3 checks that each kernel's
// registers fit its block; part 4 times each kernel with other block and thread counts. (The C#
// trainer had no LayerNorm test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::layer_norm, backward(); libtorch before any CUDA header
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/layer_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it. weight_grad and bias_grad are rounded once per
// block that adds onto them, 8 times here.
static const double BFLOAT16_TOLERANCE = 2e-2;

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

static void forward_matches_pytorch(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens, embed_dim = 768 as in GPT-2 small (each
    // thread of a warp takes 6 of a row's 192 float4s), random input, weight and bias: output, mean
    // and rstd equal what PyTorch's native_layer_norm gives, within 1e-5. Not to the last bit: the
    // two add up a row's 768 values in a different order.
    Tensor::manual_seed(0);
    size_t batch_size = 2, seq_len = 8, embed_dim = 768;
    Tensor input = tensor_of({batch_size, seq_len, embed_dim}, random_values({batch_size, seq_len, embed_dim}));
    Tensor weight = tensor_of({embed_dim}, random_values({embed_dim}));
    Tensor bias = tensor_of({embed_dim}, random_values({embed_dim}));
    Tensor output = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    layer_norm_forward(output, mean, rstd, input, weight, bias);

    std::vector<int64_t> normalized_shape = {(int64_t)embed_dim};    // normalize over the last dim
    auto [pytorch_output, pytorch_mean, pytorch_rstd] = torch::native_layer_norm(
        pytorch_copy_of(input), normalized_shape, pytorch_copy_of(weight), pytorch_copy_of(bias), 1e-5);
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), TOLERANCE);
    ASSERT_NEAR(mean.tolist(), list_of(pytorch_mean), TOLERANCE);
    ASSERT_NEAR(rstd.tolist(), list_of(pytorch_rstd), TOLERANCE);
}

static void backward_matches_pytorch(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens, embed_dim = 768: 256 rows, 8 blocks of the
    // backward, so weight_grad and bias_grad get adds from 8 blocks. Random input, weight, bias and
    // output_grad, backward from 0: input_grad, weight_grad and bias_grad equal PyTorch's .grad after
    // autograd's backward, within 1e-5.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 64, embed_dim = 768;
    Tensor input = tensor_of({batch_size, seq_len, embed_dim}, random_values({batch_size, seq_len, embed_dim}));
    Tensor weight = tensor_of({embed_dim}, random_values({embed_dim}));
    Tensor bias = tensor_of({embed_dim}, random_values({embed_dim}));
    Tensor output_grad =
        tensor_of({batch_size, seq_len, embed_dim}, random_values({batch_size, seq_len, embed_dim}));
    Tensor output = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor weight_grad = Tensor::zeros({embed_dim}), bias_grad = Tensor::zeros({embed_dim});
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_weight = pytorch_copy_of(weight).requires_grad_(true);
    torch::Tensor pytorch_bias = pytorch_copy_of(bias).requires_grad_(true);
    std::vector<int64_t> normalized_shape = {(int64_t)embed_dim};
    torch::Tensor pytorch_output =
        torch::layer_norm(pytorch_input, normalized_shape, pytorch_weight, pytorch_bias, 1e-5);
    pytorch_output.backward(pytorch_copy_of(output_grad));
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), TOLERANCE);
    ASSERT_NEAR(weight_grad.tolist(), list_of(pytorch_weight.grad()), TOLERANCE);
    ASSERT_NEAR(bias_grad.tolist(), list_of(pytorch_bias.grad()), TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: backward_matches_pytorch's 256 rows of 768 (8 blocks of the backward), with input, weight,
    // bias, output_grad and the gradients in bfloat16; mean and rstd stay float. Every sum is in float from
    // the same bfloat16 values PyTorch is given as floats, so mean and rstd are PyTorch's within 1e-5; output
    // and input_grad are rounded once as they are stored, weight_grad and bias_grad once per block: within
    // BFLOAT16_TOLERANCE.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 64, embed_dim = 768;
    Shape shape = {batch_size, seq_len, embed_dim};
    Tensor input = tensor_of(shape, random_values(shape)).to(DType::BFloat16);
    Tensor weight = tensor_of({embed_dim}, random_values({embed_dim})).to(DType::BFloat16);
    Tensor bias = tensor_of({embed_dim}, random_values({embed_dim})).to(DType::BFloat16);
    Tensor output_grad = tensor_of(shape, random_values(shape)).to(DType::BFloat16);
    Tensor output = Tensor::zeros(shape, DType::BFloat16);
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    Tensor input_grad = Tensor::zeros(shape, DType::BFloat16);
    Tensor weight_grad = Tensor::zeros({embed_dim}, DType::BFloat16);
    Tensor bias_grad = Tensor::zeros({embed_dim}, DType::BFloat16);
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);

    torch::Tensor pytorch_input = pytorch_copy_of(input.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_weight = pytorch_copy_of(weight.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_bias = pytorch_copy_of(bias.to(DType::Float32)).requires_grad_(true);
    std::vector<int64_t> normalized_shape = {(int64_t)embed_dim};
    auto [pytorch_output, pytorch_mean, pytorch_rstd] =
        torch::native_layer_norm(pytorch_input, normalized_shape, pytorch_weight, pytorch_bias, 1e-5);
    pytorch_output.backward(pytorch_copy_of(output_grad.to(DType::Float32)));
    ASSERT_NEAR(mean.tolist(), list_of(pytorch_mean), TOLERANCE);
    ASSERT_NEAR(rstd.tolist(), list_of(pytorch_rstd), TOLERANCE);
    ASSERT_NEAR(output.to(DType::Float32).tolist(), list_of(pytorch_output), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(input_grad.to(DType::Float32).tolist(), list_of(pytorch_input.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(weight_grad.to(DType::Float32).tolist(), list_of(pytorch_weight.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(bias_grad.to(DType::Float32).tolist(), list_of(pytorch_bias.grad()), BFLOAT16_TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's sizes, batch_size = 4 rows
    // of seq_len = 1024 tokens, embed_dim = 768, mine against PyTorch's layer_norm with autograd. A
    // forward reads input and writes output, 2 x 12.6 MB: at a T4's 320 GB/s at least 0.079 ms. A
    // backward reads input, output_grad and input_grad and writes input_grad, 4 x 12.6 MB: at least
    // 0.157 ms. Both sides add 20 backwards onto their gradients; at the end they must still agree.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, embed_dim = 768;
    Tensor input = Tensor::randn({batch_size, seq_len, embed_dim});      // made on the CPU: a few seconds
    Tensor weight = Tensor::randn({embed_dim}), bias = Tensor::randn({embed_dim});
    Tensor output_grad = Tensor::randn({batch_size, seq_len, embed_dim});
    Tensor output = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor weight_grad = Tensor::zeros({embed_dim}), bias_grad = Tensor::zeros({embed_dim});

    torch::Tensor pytorch_input = pytorch_copy_of(input).requires_grad_(true);
    torch::Tensor pytorch_weight = pytorch_copy_of(weight).requires_grad_(true);
    torch::Tensor pytorch_bias = pytorch_copy_of(bias).requires_grad_(true);
    torch::Tensor pytorch_output_grad = pytorch_copy_of(output_grad);
    std::vector<int64_t> normalized_shape = {(int64_t)embed_dim};
    torch::Tensor pytorch_output;

    // Steps 1 and 2 do first-time work, so the average is of 3..20: in step 1 CUDA loads each side's
    // kernels; in step 2 PyTorch's backward first adds onto the .grad that step 1 made (26.6 ms on a
    // T4, against 0.54 ms in every later step), most likely loading the kernel that adds.
    print_test_name(__func__);
    printf("    time in ms:   step   my forward   my backward   PyTorch forward   PyTorch backward\n");
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { layer_norm_forward(output, mean, rstd, input, weight, bias); });
        times[1] = gpu_milliseconds([&] {
            layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
        });
        times[2] = gpu_milliseconds([&] {
            pytorch_output = torch::layer_norm(pytorch_input, normalized_shape, pytorch_weight, pytorch_bias, 1e-5);
        });
        times[3] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        printf("                  %4d   %10.3f   %11.3f   %15.3f   %16.3f\n", t, times[0], times[1], times[2],
               times[3]);
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    printf("    average of steps 3..20:  %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("    PyTorch takes, against mine:                         %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_NEAR(output.tolist(), list_of(pytorch_output), TOLERANCE);
    ASSERT_NEAR(input_grad.tolist(), list_of(pytorch_input.grad()), TOLERANCE);
    ASSERT_NEAR(weight_grad.tolist(), list_of(pytorch_weight.grad()), TOLERANCE);
    ASSERT_NEAR(bias_grad.tolist(), list_of(pytorch_bias.grad()), TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use rows of 4 (one float4: thread 0 of the warp has it all, threads 1 .. 31 have
// nothing to read). The row 1 2 3 4 works out as:
//
//     mean = 10 / 4 = 2.5
//     var  = (1.5² + 0.5² + 0.5² + 1.5²) / 4 = 5 / 4 = 1.25
//     rstd = 1 / sqrt(1.25 + 0.00001) = 0.8944236
//     norm = -1.5, -0.5, 0.5, 1.5 times rstd = -1.3416354, -0.4472118, 0.4472118, 1.3416354

static const std::vector<float> NORM_OF_1_2_3_4 = {-1.3416354f, -0.4472118f, 0.4472118f, 1.3416354f};

static void forward_moves_a_row_to_mean_0_and_standard_deviation_1(void)
{
    // Expect: the row 1 2 3 4, weight 1, bias 0: output is norm, -1.3416354 -0.4472118 0.4472118
    // 1.3416354 (mean 0, standard deviation 1); mean is 2.5 exactly, rstd 0.8944236.
    Tensor input = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor output = Tensor::zeros({1, 1, 4}), mean = Tensor::zeros({1, 1}), rstd = Tensor::zeros({1, 1});
    Tensor weight = Tensor::ones({4});
    Tensor bias = Tensor::zeros({4});
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    ASSERT_NEAR(output.tolist(), NORM_OF_1_2_3_4, 1e-6);
    ASSERT_EQ(mean.tolist(), std::vector<float>({2.5f}));
    ASSERT_NEAR(rstd.tolist()[0], 0.8944236, 1e-6);
}

static void weight_and_bias_scale_and_shift_each_column(void)
{
    // Expect: the row 1 2 3 4, weight 2 for every column, bias 10 20 30 40: output = 2 x norm + bias
    // = 10 - 2.6832708, 20 - 0.8944236, 30 + 0.8944236, 40 + 2.6832708.
    Tensor input = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor weight = tensor_of({4}, {2, 2, 2, 2}), bias = tensor_of({4}, {10, 20, 30, 40});
    Tensor output = Tensor::zeros({1, 1, 4}), mean = Tensor::zeros({1, 1}), rstd = Tensor::zeros({1, 1});
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    ASSERT_NEAR(output.tolist(), std::vector<float>({7.3167292f, 19.1055764f, 30.8944236f, 42.6832708f}), 1e-6);
}

static void a_flat_row_gives_the_bias(void)
{
    // Expect: the row 5 5 5 5: mean 5, every x - mean is exactly 0, so output is the bias, 10 20 30
    // 40, exactly. The variance is 0 and rstd = 1 / sqrt(0 + 0.00001) = 316.22777: without the 1e-5
    // it would be 1 / 0 = infinity, and 0 x infinity = NaN.
    Tensor input = tensor_of({1, 1, 4}, {5, 5, 5, 5});
    Tensor weight = tensor_of({4}, {2, 2, 2, 2}), bias = tensor_of({4}, {10, 20, 30, 40});
    Tensor output = Tensor::zeros({1, 1, 4}), mean = Tensor::zeros({1, 1}), rstd = Tensor::zeros({1, 1});
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    ASSERT_EQ(output.tolist(), std::vector<float>({10, 20, 30, 40}));
    ASSERT_NEAR(rstd.tolist()[0], 316.22777, 1e-3);
}

static void each_row_is_normalized_on_its_own(void)
{
    // Expect: two rows, 1 2 3 4 and 103 106 109 112 (= 100 + 3 x the first). Each row gets its own
    // mean, 2.5 and 107.5, and its own rstd, so both come out as the same norm: shifting a row or
    // scaling it does not change its output. Within 1e-5, not exactly: the 1e-5 added to the
    // variance counts for less next to 11.25 than next to 1.25.
    Tensor input = tensor_of({1, 2, 4}, {1, 2, 3, 4, 103, 106, 109, 112});
    Tensor output = Tensor::zeros({1, 2, 4}), mean = Tensor::zeros({1, 2}), rstd = Tensor::zeros({1, 2});
    layer_norm_forward(output, mean, rstd, input, Tensor::ones({4}), Tensor::zeros({4}));
    ASSERT_EQ(mean.tolist(), std::vector<float>({2.5f, 107.5f}));
    ASSERT_NEAR(output[0][0].tolist(), NORM_OF_1_2_3_4, 1e-5);
    ASSERT_NEAR(output[0][1].tolist(), NORM_OF_1_2_3_4, 1e-5);
}

static void backward_adds_up_bias_grad_and_weight_grad_over_the_rows(void)
{
    // Expect: two rows, 1 2 3 4 and 4 3 2 1 (its norm is the first's backwards: -norm), weight 1,
    // output_grad rows of 1s and of 2s.
    //
    //     bias_grad   = 1 + 2                       = 3 3 3 3, exactly
    //     weight_grad = 1 x norm + 2 x (-norm) = -norm  = 1.3416354 0.4472118 -0.4472118 -1.3416354
    Tensor input = tensor_of({1, 2, 4}, {1, 2, 3, 4, 4, 3, 2, 1});
    Tensor weight = Tensor::ones({4});
    Tensor output = Tensor::zeros({1, 2, 4}), mean = Tensor::zeros({1, 2}), rstd = Tensor::zeros({1, 2});
    layer_norm_forward(output, mean, rstd, input, weight, Tensor::zeros({4}));
    Tensor output_grad = tensor_of({1, 2, 4}, {1, 1, 1, 1, 2, 2, 2, 2});
    Tensor input_grad = Tensor::zeros({1, 2, 4}), weight_grad = Tensor::zeros({4}), bias_grad = Tensor::zeros({4});
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({3, 3, 3, 3}));
    ASSERT_NEAR(weight_grad.tolist(), std::vector<float>({1.3416354f, 0.4472118f, -0.4472118f, -1.3416354f}), 1e-6);
}

static void the_same_output_grad_on_a_whole_row_gives_no_input_grad(void)
{
    // Expect: the row 1 2 3 4, weight 1, output_grad 5 5 5 5: input_grad is 0 (up to rounding).
    // Raising every output of a row by the same amount is something no change of the input can do,
    // since the forward subtracts the row's mean: norm_grad - mean(norm_grad) = 5 - 5 = 0, and
    // mean(norm_grad x norm) = 5 x mean(norm) = 0.
    Tensor input = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor weight = Tensor::ones({4});
    Tensor output = Tensor::zeros({1, 1, 4}), mean = Tensor::zeros({1, 1}), rstd = Tensor::zeros({1, 1});
    layer_norm_forward(output, mean, rstd, input, weight, Tensor::zeros({4}));
    Tensor output_grad = tensor_of({1, 1, 4}, {5, 5, 5, 5});
    Tensor input_grad = Tensor::zeros({1, 1, 4}), weight_grad = Tensor::zeros({4}), bias_grad = Tensor::zeros({4});
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    ASSERT_NEAR(input_grad.tolist(), std::vector<float>(4, 0.0f), 1e-5);
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch, or for input_grad the
    // residual stream, had left it there). The row 1 2 3 4, weight 1, output_grad 1 1 1 1:
    // bias_grad becomes 11 exactly, weight_grad 10 + norm = 8.6583646 9.5527882 10.4472118 11.3416354,
    // input_grad 10 + 0 (as in the test above). A second backward adds again: bias_grad 12.
    Tensor input = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor weight = Tensor::ones({4});
    Tensor output = Tensor::zeros({1, 1, 4}), mean = Tensor::zeros({1, 1}), rstd = Tensor::zeros({1, 1});
    layer_norm_forward(output, mean, rstd, input, weight, Tensor::zeros({4}));
    Tensor output_grad = Tensor::ones({1, 1, 4});
    Tensor input_grad = Tensor::zeros({1, 1, 4}).fill_(10);
    Tensor weight_grad = Tensor::zeros({4}).fill_(10), bias_grad = Tensor::zeros({4}).fill_(10);
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({11, 11, 11, 11}));
    ASSERT_NEAR(weight_grad.tolist(), std::vector<float>({8.6583646f, 9.5527882f, 10.4472118f, 11.3416354f}), 1e-6);
    ASSERT_NEAR(input_grad.tolist(), std::vector<float>({10, 10, 10, 10}), 1e-6);
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>({12, 12, 12, 12}));
}

static void many_rows_adding_onto_bias_grad_lose_nothing(void)
{
    // Expect: GPT-2 small's sizes, batch_size = 4 rows of seq_len = 1024 tokens, embed_dim = 768,
    // random input, weight 1, output_grad all 1. All 4096 rows add onto the same 768 values of
    // bias_grad: 32 rows into each block's shared copy, then 128 blocks into GPU memory, many at the
    // same moment. With atomicAdd no add is lost: 4096 x 1 = 4096 in every column, exactly (whole
    // numbers this small add exactly in floats). input_grad is 0 in every row, up to rounding, as
    // in the_same_output_grad_on_a_whole_row_gives_no_input_grad.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, embed_dim = 768;
    Tensor input = Tensor::randn({batch_size, seq_len, embed_dim});      // made on the CPU: a few seconds
    Tensor weight = Tensor::ones({embed_dim});
    Tensor output = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    layer_norm_forward(output, mean, rstd, input, weight, Tensor::zeros({embed_dim}));
    Tensor output_grad = Tensor::ones({batch_size, seq_len, embed_dim});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor weight_grad = Tensor::zeros({embed_dim}), bias_grad = Tensor::zeros({embed_dim});
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    ASSERT_EQ(bias_grad.tolist(), std::vector<float>(embed_dim, 4096.0f));
    ASSERT_NEAR(input_grad.tolist(), std::vector<float>(batch_size * seq_len * embed_dim, 0.0f), 1e-5);
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the forward can be launched with LAYER_NORM_THREADS_PER_BLOCK (256) and the backward
    // with LAYER_NORM_BACKWARD_THREADS_PER_BLOCK (512), and nothing is spilled. The compiler picks
    // each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote down, without
    // running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)layer_norm_forward_kernel<float>,
                             (const void *)layer_norm_backward_kernel<float>};
    const char *names[] = {"layer_norm_forward_kernel", "layer_norm_backward_kernel"};
    int threads[] = {LAYER_NORM_THREADS_PER_BLOCK, LAYER_NORM_BACKWARD_THREADS_PER_BLOCK};
    for (int k = 0; k < 2; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= threads[k], true);
        ASSERT_EQ(attributes.localSizeBytes, (size_t)0);
    }
}

// ══ part 4: other block and thread counts ═════════════════════════════════════

/// Runs `run` 20 times on the GPU and gives the average time of runs 2..20, in milliseconds;
/// run 1 does first-time work, so it is left out.
template <typename Run>
static float average_milliseconds(Run run)
{
    float total = 0.0f;
    for (int t = 1; t <= 20; t++)
    {
        float milliseconds = gpu_milliseconds(run);
        if (t > 1)
            total += milliseconds;
    }
    return total / 19;
}

/// Keeps the GPU busy with `run` for about a second before timing. While the CPU makes random
/// numbers for seconds the GPU idles and its clock drops; without this, the first launches timed
/// run while the clock is still rising.
template <typename Run>
static void warm_up_gpu(Run run)
{
    float total = 0.0f;
    while (total < 1000.0f)
        total += gpu_milliseconds(run);
}

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's sizes (batch_size = 4,
    // seq_len = 1024: 4096 rows of embed_dim = 768) for 8 block counts x 4 thread counts, each the
    // average of runs 2..20. The kernels are launched here directly, <<<blocks, threads>>>, instead
    // of through layer_norm_forward and layer_norm_backward, which always launch 512 x 256 and
    // 128 x 512. Not 1024 threads: since the kernels became templates for bfloat16, each takes 66
    // registers per thread, and a block of 1024 may have 65,536 / 1024 = 64 (cudaFuncGetAttributes:
    // largest block 896). Held to 64, the backward spilled 8 bytes; and blocks of 1024 were never the
    // fastest here.
    //
    // Expect: every launch gives the same result as the usual ones. Whatever the counts, each row
    // is one warp's and its threads add in the same order, so output, mean, rstd and input_grad are
    // the same to the last bit; weight_grad and bias_grad within TOLERANCE (atomicAdd's order). Only
    // the time changes. On a Colab T4 (at most 40 SMs x 32 warps = 1,280 warps at once) both are
    // fastest with far fewer warps than that: the forward with 160 to 640 (0.117 ms, against 0.167
    // for its usual 4096), the backward with about 512 (0.275 ms, against 0.333 for its usual 2048).
    // Most likely the cache: the forward reads each row 3 times and the backward 2, and the re-reads
    // come from an SM's L1 (at most 64 KB) only while its warps' rows fit: 16 x 3 KB = 48 KB does,
    // 26 x 3 KB = 77 KB does not. Not measured directly yet.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, embed_dim = 768;
    size_t rows = batch_size * seq_len;
    Tensor input = Tensor::randn({batch_size, seq_len, embed_dim});      // made on the CPU: a few seconds
    Tensor weight = Tensor::randn({embed_dim}), bias = Tensor::randn({embed_dim});
    Tensor output_grad = Tensor::randn({batch_size, seq_len, embed_dim});
    Tensor output = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor mean = Tensor::zeros({batch_size, seq_len}), rstd = Tensor::zeros({batch_size, seq_len});
    Tensor input_grad = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor weight_grad = Tensor::zeros({embed_dim}), bias_grad = Tensor::zeros({embed_dim});

    // The usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with.
    layer_norm_forward(output, mean, rstd, input, weight, bias);
    layer_norm_backward(input_grad, weight_grad, bias_grad, output_grad, input, weight, mean, rstd);
    torch::Tensor usual_output = pytorch_copy_of(output);
    torch::Tensor usual_mean = pytorch_copy_of(mean), usual_rstd = pytorch_copy_of(rstd);
    torch::Tensor usual_input_grad = pytorch_copy_of(input_grad);
    torch::Tensor usual_weight_grad = pytorch_copy_of(weight_grad), usual_bias_grad = pytorch_copy_of(bias_grad);

    // What layer_norm_forward and layer_norm_backward launch, worked out as they do: 512 and 128 blocks.
    int forward_warps_per_block = LAYER_NORM_THREADS_PER_BLOCK / 32;
    int usual_forward_blocks = (int)((rows + forward_warps_per_block - 1) / forward_warps_per_block);
    int backward_rows_per_block = (LAYER_NORM_BACKWARD_THREADS_PER_BLOCK / 32) * 2;
    int usual_backward_blocks = (int)((rows + backward_rows_per_block - 1) / backward_rows_per_block);

    int blocks = 0, threads = 0;                             // the launch the 2 below use, set by the loop
    size_t shared_bytes = 2 * embed_dim * sizeof(float);     // the backward's block copy of bias_grad, weight_grad
    auto forward = [&] {
        layer_norm_forward_kernel<float><<<blocks, threads>>>(output.data(), mean.data(), rstd.data(), input.data(),
                                                              weight.data(), bias.data(), rows, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    auto backward = [&] {
        layer_norm_backward_kernel<float><<<blocks, threads, shared_bytes>>>(
            input_grad.data(), weight_grad.data(), bias_grad.data(), output_grad.data(), input.data(), weight.data(),
            mean.data(), rstd.data(), rows, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    blocks = usual_forward_blocks;
    threads = LAYER_NORM_THREADS_PER_BLOCK;
    warm_up_gpu(forward);

    int block_counts[] = {40, 80, 128, 256, 512, 1024, 2048, 4096};
    int thread_counts[] = {64, 128, 256, 512};
    const char *kernel_names[] = {"forward", "backward"};
    float fastest_ms[2] = {1e9f, 1e9f};
    int fastest_blocks[2] = {0, 0}, fastest_threads[2] = {0, 0};
    int launches_with_the_same_result = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads    warps   rows per warp   forward   backward   same result\n");
    for (int block_count : block_counts)
    {
        for (int thread_count : thread_counts)
        {
            blocks = block_count;
            threads = thread_count;
            float times[2];
            times[0] = average_milliseconds(forward);
            times[1] = average_milliseconds(backward);
            for (int k = 0; k < 2; k++)
            {
                if (times[k] < fastest_ms[k])
                {
                    fastest_ms[k] = times[k];
                    fastest_blocks[k] = blocks;
                    fastest_threads[k] = threads;
                }
            }

            // One forward and one backward from 0 with this launch, against the usual launches':
            // output, mean, rstd and input_grad to the last bit, weight_grad and bias_grad within
            // TOLERANCE (atomicAdd's order).
            output.zero_();
            mean.zero_();
            rstd.zero_();
            input_grad.zero_();
            weight_grad.zero_();
            bias_grad.zero_();
            forward();
            backward();
            torch::Tensor weight_grad_difference =
                (pytorch_copy_of(weight_grad) - usual_weight_grad).abs().max() / usual_weight_grad.abs().max();
            torch::Tensor bias_grad_difference =
                (pytorch_copy_of(bias_grad) - usual_bias_grad).abs().max() / usual_bias_grad.abs().max();
            bool same = torch::equal(pytorch_copy_of(output), usual_output) &&
                        torch::equal(pytorch_copy_of(mean), usual_mean) &&
                        torch::equal(pytorch_copy_of(rstd), usual_rstd) &&
                        torch::equal(pytorch_copy_of(input_grad), usual_input_grad) &&
                        weight_grad_difference.item<float>() <= TOLERANCE &&
                        bias_grad_difference.item<float>() <= TOLERANCE;
            if (same)
                launches_with_the_same_result++;

            int warps = blocks * threads / 32;
            int rows_per_warp = (int)((rows + warps - 1) / warps);    // the most any warp takes
            const char *usual = "";
            if (blocks == usual_forward_blocks && threads == LAYER_NORM_THREADS_PER_BLOCK)
                usual = "   <- what layer_norm_forward launches";
            if (blocks == usual_backward_blocks && threads == LAYER_NORM_BACKWARD_THREADS_PER_BLOCK)
                usual = "   <- what layer_norm_backward launches";
            printf("                  %6d   %7d   %6d   %13d   %7.3f   %8.3f   %-11s%s\n", blocks, threads, warps,
                   rows_per_warp, times[0], times[1], same ? "yes" : "NO", usual);
        }
    }
    for (int k = 0; k < 2; k++)
        printf("    fastest %-8s  %4d blocks x %4d threads: %.3f ms\n", kernel_names[k], fastest_blocks[k],
               fastest_threads[k], fastest_ms[k]);
    ASSERT_EQ(launches_with_the_same_result, 8 * 4);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    forward_matches_pytorch();
    backward_matches_pytorch();
    forward_and_backward_in_bfloat16();
    time_20_steps_on_gpt2_small_size();

    forward_moves_a_row_to_mean_0_and_standard_deviation_1();
    weight_and_bias_scale_and_shift_each_column();
    a_flat_row_gives_the_bias();
    each_row_is_normalized_on_its_own();
    backward_adds_up_bias_grad_and_weight_grad_over_the_rows();
    the_same_output_grad_on_a_whole_row_gives_no_input_grad();
    backward_adds_onto_the_gradients_already_there();
    many_rows_adding_onto_bias_grad_lose_nothing();

    every_kernel_fits_its_block();

    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tests passed\n");
    return 0;
}
