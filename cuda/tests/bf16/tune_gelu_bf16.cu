// tune_gelu_bf16.cu — the timings of cuda/kernels/bf16/gelu_bf16.cuh, apart from cuda/tests/bf16/test_gelu_bf16.cu so the
// tests only check numbers: time_20_steps_on_gpt2_mlp.
//
// Built as cuda/tests/bf16/test_gelu_bf16.cu is, on an H100 with -arch=sm_90a (see cuda/tests/README.md); what it prints also
// goes to cuda/tests/results/<GPU>/gelu_bf16_<date>_<time>.txt.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../../kernels/bf16/gelu_bf16.cuh"

// The fused and unfused results: the same math on the same values, within one bfloat16 rounding (where the
// compiler orders GELU's float math another way). The largest difference, as a fraction of the largest value.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
static const float WEIGHT_STD = 0.02f;   // GPT-2's starting std

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_NUM_LAYERS 12            // MLPs in GPT-2 small: one per block
#define SIDES 2                       // fused, unfused

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation `std_val`, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
///   std_val      the standard deviation; 1 if not given
static Tensor random_bf16_tensor(const Shape &shape_list, float std_val = 1.0f)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * std_val;
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Gives a bfloat16 tensor's values as a CPU list of floats, exactly.
static std::vector<float> values_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

// ══ part 1: fused against unfused ═════════════════════════════════════════════

static void time_20_steps_on_gpt2_mlp(void)
{
    // Prints the times of 20 forwards and 20 backwards of GPT-2 small's MLP around GELU, {4, 1024} rows, embed_dim
    // 768, mlp_dim 3072, bfloat16: fused (gelu_bf16_forward: mlp_up with GELU as it is stored; gelu_bf16_backward:
    // mlp_down's backward with gelu' as its input_grad is stored) against unfused (linear_bf16_forward then
    // gelu_forward; linear_bf16_backward then gelu_backward). Averages of steps 3..20, and what fusing saves a
    // micro-batch over GPT-2 small's 12 MLPs. Expect fused faster: it leaves out the GELU kernels' passes over the
    // 25 MB tensors (the forward's gelu_input read again; the backward's gelu_output_grad written and read).
    // At the end both sides' results agree within one bfloat16 rounding.
    // 1. random bfloat16 values, shared by both sides; each side's results
    torch::manual_seed(0);
    Shape mlp_shape_list = {GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_MLP_DIM};
    Tensor input_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor up_weight_tensor = random_bf16_tensor({GPT2_MLP_DIM, GPT2_EMBED_DIM}, WEIGHT_STD);
    Tensor up_bias_tensor = random_bf16_tensor({GPT2_MLP_DIM}, WEIGHT_STD);
    Tensor down_weight_tensor = random_bf16_tensor({GPT2_EMBED_DIM, GPT2_MLP_DIM}, WEIGHT_STD);
    Tensor output_grad_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor gelu_input_list[SIDES], gelu_output_list[SIDES], gelu_input_grad_list[SIDES], down_weight_grad_list[SIDES],
        down_bias_grad_list[SIDES];
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
    {
        gelu_input_list[side_index_val] = Tensor::zeros(mlp_shape_list, DType::BFloat16);
        gelu_output_list[side_index_val] = Tensor::zeros(mlp_shape_list, DType::BFloat16);
        gelu_input_grad_list[side_index_val] = Tensor::zeros(mlp_shape_list, DType::BFloat16);
        down_weight_grad_list[side_index_val] = Tensor::zeros(down_weight_tensor.shape(), DType::BFloat16);
        down_bias_grad_list[side_index_val] = Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16);
    }
    Tensor gelu_output_grad_tensor = Tensor::zeros(mlp_shape_list, DType::BFloat16);   // the unfused side's only

    // 2. time TIMED_STEPS steps of each side; the first WARM_UP_STEPS left out
    double totals_list[2 * SIDES] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[2 * SIDES];
        times_list[0] = gpu_milliseconds([&] {
            gelu_bf16_forward(gelu_input_list[0], gelu_output_list[0], input_tensor, up_weight_tensor, up_bias_tensor);
        });
        times_list[1] = gpu_milliseconds([&] {
            gelu_bf16_backward(gelu_input_grad_list[0], down_weight_grad_list[0], down_bias_grad_list[0], output_grad_tensor,
                               gelu_output_list[0], down_weight_tensor, gelu_input_list[0]);
        });
        times_list[2] = gpu_milliseconds([&] {
            linear_bf16_forward(gelu_input_list[1], input_tensor, up_weight_tensor, up_bias_tensor);
            gelu_forward(gelu_output_list[1], gelu_input_list[1]);
        });
        times_list[3] = gpu_milliseconds([&] {
            linear_bf16_backward(gelu_output_grad_tensor, down_weight_grad_list[1], down_bias_grad_list[1], output_grad_tensor,
                                 gelu_output_list[1], down_weight_tensor);
            gelu_backward(gelu_input_grad_list[1], gelu_output_grad_tensor, gelu_input_list[1]);
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 2 * SIDES; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages, and what fusing saves over the 12 MLPs of a micro-batch
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double forward_saved_val = (totals_list[2] - totals_list[0]) / averaged_val;
    double backward_saved_val = (totals_list[3] - totals_list[1]) / averaged_val;
    print_test_name(__func__);
    printf("    average of steps %d..%d:   forward ms   backward ms\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[SIDES] = {"fused (gelu_bf16)", "unfused (linear + gelu)"};
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        printf("        %-24s %10.3f   %11.3f\n", sides_list[side_index_val], totals_list[2 * side_index_val] / averaged_val,
               totals_list[2 * side_index_val + 1] / averaged_val);
    printf("    fusing saves %.3f ms forward, %.3f ms backward a layer: %.3f ms a micro-batch over %d layers\n",
           forward_saved_val, backward_saved_val, GPT2_NUM_LAYERS * (forward_saved_val + backward_saved_val), GPT2_NUM_LAYERS);

    // 4. both sides agree at the end: the products' results the same bits, GELU's within one rounding
    ASSERT_EQ(values_of(gelu_input_list[0]), values_of(gelu_input_list[1]));
    ASSERT_NEAR(values_of(gelu_output_list[0]), values_of(gelu_output_list[1]), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(values_of(gelu_input_grad_list[0]), values_of(gelu_input_grad_list[1]), ONE_ROUNDING_TOLERANCE);
    ASSERT_EQ(values_of(down_weight_grad_list[0]), values_of(down_weight_grad_list[1]));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and cuda/tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_mlp();

    printf("\nall tunes done\n");
    return 0;
}
