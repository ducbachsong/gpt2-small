// tune_linear_bf16.cu — the timings of llmc/kernels/bf16/linear_bf16.cuh, apart from tests/bf16/test_linear_bf16.cu so the
// tests only check numbers: time_20_steps_on_qkv, time_20_steps_on_attention_output, time_20_steps_on_mlp_up,
// time_20_steps_on_mlp_down, time_20_steps_on_head.
//
// Built as tests/bf16/test_linear_bf16.cu is, on an H100 with -arch=sm_90a (see tunes/README.md); what it prints also
// goes to tunes/results/<GPU>/linear_bf16_<date>_<time>.txt.
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../llmc/kernels/bf16/linear_bf16.cuh"

// Against PyTorch's linear in bfloat16 on the same values: both sum in float and round each result to bfloat16,
// by up to 2^-8 of it; weight_grad and bias_grad are added onto in bfloat16 every step, 20 times, each add
// rounded again. Each tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_PADDED_VOCAB_SIZE 50304  // the head's out_features: wte's rows

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

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

/// Times TIMED_STEPS forwards and backwards of one linear layer in bfloat16, batch_size = 4 rows of seq_len = 1024
/// tokens, mine against PyTorch's bfloat16 linear with autograd, and prints the averages and TFLOPS under
/// `test_ptr`'s name. Both sides write input_grad and add every backward onto weight_grad and bias_grad; at the end
/// they must still agree within the bfloat16 tolerances.
static void time_20_steps(const char *test_ptr, size_t in_features_val, size_t out_features_val, bool with_bias_val)
{
    // 1. random bfloat16 tensors made on the GPU, mine and PyTorch's
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor input_tensor = random_bf16_tensor({batch_size_val, seq_len_val, in_features_val});
    Tensor weight_tensor = random_bf16_tensor({out_features_val, in_features_val});
    Tensor bias_tensor = with_bias_val ? random_bf16_tensor({out_features_val}) : Tensor();
    Tensor output_grad_tensor = random_bf16_tensor({batch_size_val, seq_len_val, out_features_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, out_features_val}, DType::BFloat16);
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, in_features_val}, DType::BFloat16);
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val}, DType::BFloat16);
    Tensor bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}, DType::BFloat16) : Tensor();
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor =
        with_bias_val ? pytorch_copy_of(bias_tensor).to(torch::kBFloat16).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work, so the averages are of the rest
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { linear_bf16_forward(output_tensor, input_tensor, weight_tensor, bias_tensor); });
        times_list[1] = gpu_milliseconds([&] {
            linear_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                                 weight_tensor);
        });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_output_tensor = with_bias_val
                                        ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                        : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
        });
        pytorch_input_tensor.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 4; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages and TFLOPS
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double forward_gflop_val = 2.0 * rows_val * in_features_val * out_features_val / 1e9; // a multiply and an add per pair
    double work_list[4] = {forward_gflop_val, 2 * forward_gflop_val, forward_gflop_val, 2 * forward_gflop_val};
    print_test_name(test_ptr);
    printf("    input {4, 1024, %zu}, weight {%zu, %zu}, bfloat16: %.1f GFLOP forward, %.1f GFLOP backward\n",
           in_features_val, out_features_val, in_features_val, forward_gflop_val, 2 * forward_gflop_val);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("        TFLOPS                %10.2f   %11.2f   %15.2f   %16.2f\n", work_list[0] / (totals_list[0] / averaged_val),
           work_list[1] / (totals_list[1] / averaged_val), work_list[2] / (totals_list[2] / averaged_val),
           work_list[3] / (totals_list[3] / averaged_val));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, pytorch_output_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(input_grad_tensor, pytorch_input_tensor.grad()), 0.0, BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(weight_grad_tensor, pytorch_weight_tensor.grad()), 0.0, BFLOAT16_GRAD_TOLERANCE);
    if (with_bias_val)
        ASSERT_NEAR(relative_difference_on_gpu(bias_grad_tensor, pytorch_bias_tensor.grad()), 0.0, BFLOAT16_GRAD_TOLERANCE);
}

static void time_20_steps_on_qkv(void)
{
    // Prints the times of GPT-2 small's qkv: in_features = 768, out_features = 2304, with bias. A forward is
    // 2 x 4096 x 768 x 2304 = 14.5 GFLOP; a backward twice the work.
    time_20_steps(__func__, 768, 2304, true);
}

static void time_20_steps_on_attention_output(void)
{
    // Prints the times of the attention's output: 768 in, 768 out, with bias. 4.8 GFLOP forward. weight_grad's
    // product is only 768 x 768: 12 x 12 = 144 squares of 64, each walking 4096 values of k.
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
    // first, in another shape: the forward has 12 x 64 = 768 squares of 48 steps, against 48 x 64 = 3072 squares
    // of 12 steps.
    time_20_steps(__func__, 3072, 768, true);
}

static void time_20_steps_on_head(void)
{
    // Prints the times of the head: 768 in, 50304 out (wte's rows), no bias. 316.5 GFLOP forward. output and
    // output_grad are {4, 1024, 50304}, 412 MB each in bfloat16.
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
