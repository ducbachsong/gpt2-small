// tune_linear_cublas.cu — the timings of tests/test_linear_cublas.cu, moved here so the tests only check numbers:
// time_20_steps_on_qkv, time_20_steps_on_attention_output, time_20_steps_on_mlp_up, time_20_steps_on_mlp_down,
// time_20_steps_on_head.
//
// Built as tests/test_linear_cublas.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/linear_cublas_<date>_<time>.txt.
#include <torch/types.h>              // torch::linear, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/kernels/linear_cublas.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_PADDED_VOCAB_SIZE 50304  // the head's out_features: wte's rows

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes minutes for the head's 206M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
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

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

/// Times TIMED_STEPS forwards and backwards of one linear layer, batch_size = 4 rows of seq_len = 1024
/// tokens, mine against PyTorch's linear with autograd, and prints the averages and TFLOPS under
/// `test_ptr`'s name. Both sides write input_grad and add every backward onto weight_grad and bias_grad; at
/// the end they must still agree within `tolerance_val`.
static void time_20_steps(const char *test_ptr, size_t in_features_val, size_t out_features_val, bool with_bias_val,
                          double tolerance_val)
{
    // 1. random tensors made on the GPU, mine and PyTorch's
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor input_tensor = random_tensor_on_gpu({batch_size_val, seq_len_val, in_features_val});
    Tensor weight_tensor = random_tensor_on_gpu({out_features_val, in_features_val});
    Tensor bias_tensor = with_bias_val ? random_tensor_on_gpu({out_features_val}) : Tensor();
    Tensor output_grad_tensor = random_tensor_on_gpu({batch_size_val, seq_len_val, out_features_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, out_features_val});
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, in_features_val});
    Tensor weight_grad_tensor = Tensor::zeros({out_features_val, in_features_val});
    Tensor bias_grad_tensor = with_bias_val ? Tensor::zeros({out_features_val}) : Tensor();
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = with_bias_val ? pytorch_copy_of(bias_tensor).requires_grad_(true) : torch::Tensor();
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(output_grad_tensor);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work, so the averages are of the rest:
    //    in step 1 CUDA loads each side's kernels; in step 2 PyTorch's backward first adds onto the .grad that
    //    step 1 made
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { linear_forward(output_tensor, input_tensor, weight_tensor, bias_tensor); });
        times_list[1] = gpu_milliseconds([&] {
            linear_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                            weight_tensor);
        });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_output_tensor = with_bias_val
                                        ? torch::linear(pytorch_input_tensor, pytorch_weight_tensor, pytorch_bias_tensor)
                                        : torch::linear(pytorch_input_tensor, pytorch_weight_tensor);
        });
        pytorch_input_tensor.mutable_grad() = torch::Tensor(); // input's gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages and TFLOPS
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double forward_gflop_val = 2.0 * rows_val * in_features_val * out_features_val / 1e9; // a multiply and an add per pair
    double work_list[4] = {forward_gflop_val, 2 * forward_gflop_val, forward_gflop_val, 2 * forward_gflop_val};
    print_test_name(test_ptr);
    printf("    input {4, 1024, %zu}, weight {%zu, %zu}: %.1f GFLOP forward, %.1f GFLOP backward\n", in_features_val,
           out_features_val, in_features_val, forward_gflop_val, 2 * forward_gflop_val);
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
    ASSERT_NEAR(output_tensor.tolist(), list_of(pytorch_output_tensor), tolerance_val);
    ASSERT_NEAR(input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), tolerance_val);
    ASSERT_NEAR(weight_grad_tensor.tolist(), list_of(pytorch_weight_tensor.grad()), tolerance_val);
    if (with_bias_val)
        ASSERT_NEAR(bias_grad_tensor.tolist(), list_of(pytorch_bias_tensor.grad()), tolerance_val);
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
    // 0.60 ms forward, 1.19 ms backward. weight_grad's product is only 768 x 768, each value a sum over
    // 4096 rows: few squares for a GPU of 40 SMs, which cuBLAS fills by splitting the sum.
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
    // the MLP's first, in another shape: fewer, longer sums.
    time_20_steps(__func__, 3072, 768, true, TOLERANCE);
}

static void time_20_steps_on_head(void)
{
    // Prints the times of the head: 768 in, 50304 out (wte's rows), no bias. 316.5 GFLOP: at least
    // 39.1 ms forward, 78.1 ms backward. output and output_grad are {4, 1024, 50304}, 824 MB each.
    //
    // Checked within 1e-4, not 1e-5, as in test_linear.cu: each input_grad value here adds up 50,304
    // products, and rounding drifts with the length of the sum. Both sides are cuBLAS here, but they may
    // still pick different kernels, which add up in another order.
    time_20_steps(__func__, 768, GPT2_PADDED_VOCAB_SIZE, false, 1e-4);
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
