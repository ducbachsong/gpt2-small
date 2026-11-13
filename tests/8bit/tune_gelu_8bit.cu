// tune_gelu_8bit.cu — the timings of llmc/kernels/8bit/gelu_8bit.cuh, apart from tests/8bit/test_gelu_8bit.cu so the
// tests only check numbers: time_20_steps_on_gpt2_mlp. It times the fused 8-bit MLP around GELU against the
// unfused 8-bit one and the fused bfloat16 one (bf16/gelu_bf16.cuh).
//
// Built as tests/8bit/test_gelu_8bit.cu is, on an H100 with -arch=sm_90a (see tests/README.md); what it prints also
// goes to tests/results/<GPU>/gelu_8bit_<date>_<time>.txt.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../../llmc/kernels/8bit/gelu_8bit.cuh"   // and through it bf16/gelu_bf16.cuh

// The fused and unfused 8-bit results: the same math on the same values, within one bfloat16 rounding (where the
// compiler orders GELU's float math another way). The largest difference, as a fraction of the largest value.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
static const float WEIGHT_STD = 0.02f;   // GPT-2's starting std

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_NUM_LAYERS 12            // MLPs in GPT-2 small: one per block
#define SIDES 3                       // fused 8-bit, unfused 8-bit, fused bfloat16

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation `std_val`, made by
/// PyTorch on the GPU, then rounded to bfloat16.
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

// ══ part 1: fused against unfused, and against bfloat16 ═══════════════════════

static void time_20_steps_on_gpt2_mlp(void)
{
    // Prints the times of 20 forwards and 20 backwards of GPT-2 small's MLP around GELU, {4, 1024} rows, embed_dim
    // 768, mlp_dim 3072: fused 8-bit (gelu_8bit_forward, gelu_8bit_backward, their roundings included), unfused
    // 8-bit (linear_8bit then gelu_forward; linear_8bit_backward then gelu_backward) and fused bfloat16
    // (gelu_bf16). Averages of steps 3..20, and what fusing saves a micro-batch over GPT-2 small's 12 MLPs. At the
    // end the fused and unfused 8-bit results agree within one bfloat16 rounding.
    // 1. random bfloat16 values, shared by all sides; each side's results and the 8-bit sides' scratch
    torch::manual_seed(0);
    size_t rows_val = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
    Shape mlp_shape_list = {GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_MLP_DIM};
    Tensor input_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor up_weight_tensor = random_bf16_tensor({GPT2_MLP_DIM, GPT2_EMBED_DIM}, WEIGHT_STD);
    Tensor up_bias_tensor = random_bf16_tensor({GPT2_MLP_DIM}, WEIGHT_STD);
    Tensor down_weight_tensor = random_bf16_tensor({GPT2_EMBED_DIM, GPT2_MLP_DIM}, WEIGHT_STD);
    Tensor output_grad_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Linear8bitScratch up_scratch_val(rows_val, GPT2_EMBED_DIM, GPT2_MLP_DIM);
    Linear8bitScratch down_scratch_val(rows_val, GPT2_MLP_DIM, GPT2_EMBED_DIM);
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
            gelu_8bit_forward(gelu_input_list[0], gelu_output_list[0], input_tensor, up_weight_tensor, up_bias_tensor, up_scratch_val);
        });
        times_list[1] = gpu_milliseconds([&] {
            gelu_8bit_backward(gelu_input_grad_list[0], down_weight_grad_list[0], down_bias_grad_list[0], output_grad_tensor,
                               gelu_output_list[0], down_weight_tensor, gelu_input_list[0], down_scratch_val);
        });
        times_list[2] = gpu_milliseconds([&] {
            linear_8bit_forward(gelu_input_list[1], input_tensor, up_weight_tensor, up_bias_tensor, up_scratch_val);
            gelu_forward(gelu_output_list[1], gelu_input_list[1]);
        });
        times_list[3] = gpu_milliseconds([&] {
            linear_8bit_backward(gelu_output_grad_tensor, down_weight_grad_list[1], down_bias_grad_list[1], output_grad_tensor,
                                 gelu_output_list[1], down_weight_tensor, down_scratch_val);
            gelu_backward(gelu_input_grad_list[1], gelu_output_grad_tensor, gelu_input_list[1]);
        });
        times_list[4] = gpu_milliseconds([&] {
            gelu_bf16_forward(gelu_input_list[2], gelu_output_list[2], input_tensor, up_weight_tensor, up_bias_tensor);
        });
        times_list[5] = gpu_milliseconds([&] {
            gelu_bf16_backward(gelu_input_grad_list[2], down_weight_grad_list[2], down_bias_grad_list[2], output_grad_tensor,
                               gelu_output_list[2], down_weight_tensor, gelu_input_list[2]);
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
    printf("    average of steps %d..%d:     forward ms   backward ms\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[SIDES] = {"fused 8-bit (gelu_8bit)", "unfused 8-bit", "fused bfloat16 (gelu_bf16)"};
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        printf("        %-27s %10.3f   %11.3f\n", sides_list[side_index_val], totals_list[2 * side_index_val] / averaged_val,
               totals_list[2 * side_index_val + 1] / averaged_val);
    printf("    fusing saves %.3f ms forward, %.3f ms backward a layer: %.3f ms a micro-batch over %d layers\n",
           forward_saved_val, backward_saved_val, GPT2_NUM_LAYERS * (forward_saved_val + backward_saved_val), GPT2_NUM_LAYERS);

    // 4. the fused and unfused 8-bit sides agree at the end: the products the same bits, GELU's within one rounding
    ASSERT_EQ(values_of(gelu_input_list[0]), values_of(gelu_input_list[1]));
    ASSERT_NEAR(values_of(gelu_output_list[0]), values_of(gelu_output_list[1]), ONE_ROUNDING_TOLERANCE);
    ASSERT_NEAR(values_of(gelu_input_grad_list[0]), values_of(gelu_input_grad_list[1]), ONE_ROUNDING_TOLERANCE);
    ASSERT_EQ(values_of(down_weight_grad_list[0]), values_of(down_weight_grad_list[1]));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_mlp();

    printf("\nall tunes done\n");
    return 0;
}
