// tune_linear_8bit.cu — the timings of llmc/kernels/8bit/linear_8bit.cuh, apart from tests/8bit/test_linear_8bit.cu so the
// tests only check numbers: time_20_steps_on_qkv, time_20_steps_on_attention_output, time_20_steps_on_mlp_up,
// time_20_steps_on_mlp_down, time_20_steps_on_head. Each times the 8-bit layer (its roundings included) against
// the bfloat16 layer (linear_bf16.cuh) and PyTorch's bfloat16 linear.
//
// Built as tests/8bit/test_linear_8bit.cu is, on an H100 with -arch=sm_90a (see tunes/README.md); what it prints also
// goes to tunes/results/<GPU>/linear_8bit_<date>_<time>.txt.
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../llmc/kernels/8bit/linear_8bit.cuh"   // and through it linear_bf16.cuh

// Against PyTorch's bfloat16 linear on the same values: each E4M3 is off by up to 2^-4 (6%) of its value, so a
// product moves by a few percent; weight_grad and bias_grad are added onto in bfloat16 every step, 20 times. The
// largest difference, as a fraction of the largest value.
static const double EIGHT_BIT_TOLERANCE = 1e-1;

#define GPT2_PADDED_VOCAB_SIZE 50304  // the head's out_features: wte's rows
#define SIDES 3                       // the 8-bit layer, the bfloat16 layer, PyTorch

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
/// on the GPU: Tensor::randn makes them on the CPU, which takes minutes for the head's 206M values.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
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
/// worked out on the GPU: the head's output is 206M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor floats_tensor = pytorch_tensor.to(torch::kFloat);
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(floats_tensor.sizes()) - floats_tensor).abs().max();
    return difference_tensor.item<double>() / floats_tensor.abs().max().item<double>();
}

/// One side's tensors of a linear layer in bfloat16: what it writes, the gradients 0 to start.
struct LinearOutputs
{
    Tensor output_tensor;
    Tensor input_grad_tensor;
    Tensor weight_grad_tensor;
    Tensor bias_grad_tensor;           // Tensor() for none

    /// Makes them, all 0, for `rows_val` rows of in_features in and out_features out.
    LinearOutputs(size_t rows_val, size_t in_features_val, size_t out_features_val, bool with_bias_val)
    {
        output_tensor = Tensor::zeros({rows_val, out_features_val}, DType::BFloat16);
        input_grad_tensor = Tensor::zeros({rows_val, in_features_val}, DType::BFloat16);
        weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
        bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}, DType::BFloat16) : Tensor();
    }
};

// ══ part 1: timed against the bfloat16 layer and PyTorch's ════════════════════

/// Times TIMED_STEPS forwards and backwards of one linear layer, batch_size = 4 rows of seq_len = 1024 tokens: the
/// 8-bit layer, the bfloat16 layer and PyTorch's bfloat16 linear with autograd, on the same values. Prints the
/// averages and TFLOPS under `test_ptr`'s name, and how much faster 8 bits is. Every side writes input_grad and
/// adds every backward onto weight_grad and bias_grad; at the end the 8-bit layer must agree with PyTorch within
/// EIGHT_BIT_TOLERANCE.
static void time_20_steps(const char *test_ptr, size_t in_features_val, size_t out_features_val, bool with_bias_val)
{
    // 1. random bfloat16 tensors made on the GPU, shared by the two layers of mine; PyTorch's copies
    torch::manual_seed(0);
    size_t rows_val = 4 * 1024;
    Tensor input_tensor = random_bf16_tensor({rows_val, in_features_val});
    Tensor weight_tensor = random_bf16_tensor({out_features_val, in_features_val});
    Tensor bias_tensor = with_bias_val ? random_bf16_tensor({out_features_val}) : Tensor();
    Tensor output_grad_tensor = random_bf16_tensor({rows_val, out_features_val});
    LinearOutputs eight_bit_val(rows_val, in_features_val, out_features_val, with_bias_val);
    LinearOutputs bf16_val(rows_val, in_features_val, out_features_val, with_bias_val);
    Linear8bitScratch scratch_val(rows_val, in_features_val, out_features_val);
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor =
        with_bias_val ? pytorch_copy_of(bias_tensor).to(torch::kBFloat16).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps of each side, taking turns; the first WARM_UP_STEPS left out of the averages
    double totals_list[2 * SIDES] = {};                 // forward, backward of each side
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[2 * SIDES];
        times_list[0] = gpu_milliseconds([&] {
            linear_8bit_forward(eight_bit_val.output_tensor, input_tensor, weight_tensor, bias_tensor, scratch_val);
        });
        times_list[1] = gpu_milliseconds([&] {
            linear_8bit_backward(eight_bit_val.input_grad_tensor, eight_bit_val.weight_grad_tensor, eight_bit_val.bias_grad_tensor,
                                 output_grad_tensor, input_tensor, weight_tensor, scratch_val);
        });
        times_list[2] = gpu_milliseconds([&] { linear_bf16_forward(bf16_val.output_tensor, input_tensor, weight_tensor, bias_tensor); });
        times_list[3] = gpu_milliseconds([&] {
            linear_bf16_backward(bf16_val.input_grad_tensor, bf16_val.weight_grad_tensor, bf16_val.bias_grad_tensor,
                                 output_grad_tensor, input_tensor, weight_tensor);
        });
        times_list[4] = gpu_milliseconds([&] {
            pytorch_output_tensor = with_bias_val
                                        ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                        : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
        });
        pytorch_input_tensor.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times_list[5] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 2 * SIDES; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages and TFLOPS
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double forward_gflop_val = 2.0 * rows_val * in_features_val * out_features_val / 1e9; // a multiply and an add per pair
    print_test_name(test_ptr);
    printf("    input {4, 1024, %zu}, weight {%zu, %zu}: %.1f GFLOP forward, %.1f GFLOP backward\n", in_features_val,
           out_features_val, in_features_val, forward_gflop_val, 2 * forward_gflop_val);
    printf("    average of steps %d..%d:     forward ms   backward ms   forward TFLOPS   backward TFLOPS\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[SIDES] = {"8-bit (with roundings)", "bfloat16", "PyTorch bfloat16"};
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
    {
        double forward_ms_val = totals_list[2 * side_index_val] / averaged_val;
        double backward_ms_val = totals_list[2 * side_index_val + 1] / averaged_val;
        printf("        %-24s %10.3f   %11.3f   %14.2f   %15.2f\n", sides_list[side_index_val], forward_ms_val,
               backward_ms_val, forward_gflop_val / forward_ms_val, 2 * forward_gflop_val / backward_ms_val);
    }
    printf("    bfloat16 takes, against 8-bit:                %8.2fx   %11.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. the 8-bit layer agrees with PyTorch at the end, within 8 bits' error
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.output_tensor, pytorch_output_tensor), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.input_grad_tensor, pytorch_input_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.weight_grad_tensor, pytorch_weight_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
    if (with_bias_val)
        ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.bias_grad_tensor, pytorch_bias_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
}

static void time_20_steps_on_qkv(void)
{
    // Prints the times of GPT-2 small's qkv: in_features = 768, out_features = 2304, with bias. A forward is
    // 2 x 4096 x 768 x 2304 = 14.5 GFLOP; a backward twice the work.
    time_20_steps(__func__, 768, 2304, true);
}

static void time_20_steps_on_attention_output(void)
{
    // Prints the times of the attention's output: 768 in, 768 out, with bias. 4.8 GFLOP forward: the roundings
    // weigh most here, next to the smallest products.
    time_20_steps(__func__, 768, 768, true);
}

static void time_20_steps_on_mlp_up(void)
{
    // Prints the times of the MLP's first linear: 768 in, 3072 out, with bias. 19.3 GFLOP forward.
    time_20_steps(__func__, 768, 3072, true);
}

static void time_20_steps_on_mlp_down(void)
{
    // Prints the times of the MLP's second linear: 3072 in, 768 out, with bias. The same 19.3 GFLOP as the MLP's
    // first, in another shape.
    time_20_steps(__func__, 3072, 768, true);
}

static void time_20_steps_on_head(void)
{
    // Prints the times of the head: 768 in, 50304 out (wte's rows), no bias. 316.5 GFLOP forward. output_grad is
    // rounded twice in the backward (as it sits and turned round): 2 x 206 MB of E4M3s.
    time_20_steps(__func__, 768, GPT2_PADDED_VOCAB_SIZE, false);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_qkv();
    time_20_steps_on_attention_output();
    time_20_steps_on_mlp_up();
    time_20_steps_on_mlp_down();
    time_20_steps_on_head();

    printf("\nall tunes done\n");
    return 0;
}
