// tune_classifier_8bit.cu — the timings of llmc/kernels/8bit/classifier_8bit.cuh, apart from
// tests/8bit/test_classifier_8bit.cu so the tests only check numbers: time_20_steps_on_gpt2_small_size. It times the
// 8-bit classifier (its roundings included) against the bfloat16 one (bf16/classifier_bf16.cuh) and PyTorch's
// bfloat16 linear and cross_entropy.
//
// Built as tests/8bit/test_classifier_8bit.cu is, on an H100 with -arch=sm_90a (see tests/README.md); what it prints
// also goes to tests/results/<GPU>/classifier_8bit_<date>_<time>.txt.
#include <torch/types.h>              // torch::cross_entropy_loss, torch::linear, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../../llmc/kernels/8bit/classifier_8bit.cuh"   // and through it bf16/classifier_bf16.cuh

// Against PyTorch's bfloat16 head and cross_entropy on the same values: each E4M3 is off by up to 2^-4 (6%) of its
// value, so the logits and gradients move by a few percent; wte_grad is added onto every step, 20 times. The
// largest difference, as a fraction of the largest value.
static const double EIGHT_BIT_TOLERANCE = 1e-1;
static const float WTE_STD = 0.02f;   // GPT-2's starting std

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_VOCAB_SIZE 50257
#define GPT2_PADDED_VOCAB_SIZE 50304
#define SIDES 3                       // 8-bit, bfloat16, PyTorch

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

/// Makes an int tensor of `shape_list` with random targets 0 .. vocab_size - 1, made by PyTorch on the GPU.
static Tensor random_targets_on_gpu(const Shape &shape_list, size_t vocab_size_val)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    torch::Tensor values_tensor = torch::randint((int64_t)vocab_size_val, {(int64_t)tensor.numel()}, ints_on_gpu_val);
    cudaCheck(cudaMemcpy(tensor.data_ptr<int>(), values_tensor.data_ptr<int>(), tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    return tensor;
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

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets_tensor)
{
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(targets_tensor.shape().begin(), targets_tensor.shape().end());
    return torch::from_blob(targets_tensor.data_ptr<int>(), sizes_list, ints_on_gpu_val).to(torch::kLong);
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor floats_tensor = pytorch_tensor.to(torch::kFloat);
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(floats_tensor.sizes()) - floats_tensor).abs().max();
    return difference_tensor.item<double>() / floats_tensor.abs().max().item<double>();
}

/// One classifier side's tensors at GPT-2 small's size: what it writes, all 0 to start.
struct ClassifierOutputs
{
    Tensor logits_tensor;
    Tensor losses_tensor;
    Tensor max_of_each_row_tensor;
    Tensor sum_of_each_row_tensor;
    Tensor tile_max_tensor;
    Tensor tile_sum_tensor;
    Tensor lnf_output_grad_tensor;
    Tensor wte_grad_tensor;

    /// Makes them, all 0.
    ClassifierOutputs()
    {
        Shape per_row_list = {GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
        logits_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_PADDED_VOCAB_SIZE}, DType::BFloat16);
        losses_tensor = Tensor::zeros(per_row_list);
        max_of_each_row_tensor = Tensor::zeros(per_row_list);
        sum_of_each_row_tensor = Tensor::zeros(per_row_list);
        tile_max_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, classifier_bf16_tile_columns(GPT2_PADDED_VOCAB_SIZE)});
        tile_sum_tensor = Tensor::zeros(tile_max_tensor.shape());
        lnf_output_grad_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
        wte_grad_tensor = Tensor::zeros({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM}, DType::BFloat16);
    }
};

// ══ part 1: timed against the bfloat16 classifier and PyTorch ═══════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of GPT-2 small's head and loss, {4, 1024} rows, embed_dim
    // 768, vocab_size 50257 padded to 50304: the 8-bit classifier (with its roundings: lnf_output and wte in the
    // forward; logits_grad twice, wte and lnf_output in the backward), the bfloat16 one, and PyTorch's bfloat16
    // linear and cross_entropy with autograd. Averages of steps 3..20. The head's forward is 316.5 GFLOP; the
    // backward twice that. At the end the 8-bit side's mean loss and gradients agree with PyTorch's within
    // EIGHT_BIT_TOLERANCE.
    // 1. random bfloat16 values for all three sides
    torch::manual_seed(0);
    size_t rows_val = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
    Shape per_row_list = {GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
    Tensor lnf_output_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM}, WTE_STD);
    Tensor targets_tensor = random_targets_on_gpu(per_row_list, GPT2_VOCAB_SIZE);
    Tensor losses_grad_tensor = Tensor::zeros(per_row_list).fill_(1.0 / rows_val);
    ClassifierOutputs eight_bit_val;
    ClassifierOutputs bf16_val;
    Linear8bitScratch scratch_val(rows_val, GPT2_EMBED_DIM, GPT2_PADDED_VOCAB_SIZE);
    torch::Tensor pytorch_lnf_output_tensor =
        pytorch_copy_of(lnf_output_tensor).reshape({(int64_t)rows_val, -1}).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_targets_tensor = pytorch_targets_of(targets_tensor).reshape({-1});
    torch::Tensor pytorch_loss_tensor;

    // 2. time TIMED_STEPS steps of each side; each backward writes logits_grad over its logits, so each forward
    //    comes first; the first WARM_UP_STEPS left out
    double totals_list[2 * SIDES] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[2 * SIDES];
        times_list[0] = gpu_milliseconds([&] {
            classifier_8bit_forward(eight_bit_val.losses_tensor, eight_bit_val.max_of_each_row_tensor, eight_bit_val.sum_of_each_row_tensor,
                                    eight_bit_val.tile_max_tensor, eight_bit_val.tile_sum_tensor, eight_bit_val.logits_tensor,
                                    lnf_output_tensor, wte_tensor, targets_tensor, GPT2_VOCAB_SIZE, scratch_val);
        });
        times_list[1] = gpu_milliseconds([&] {
            classifier_8bit_backward(eight_bit_val.lnf_output_grad_tensor, eight_bit_val.wte_grad_tensor, eight_bit_val.logits_tensor,
                                     losses_grad_tensor, eight_bit_val.logits_tensor, lnf_output_tensor, wte_tensor, targets_tensor,
                                     eight_bit_val.max_of_each_row_tensor, eight_bit_val.sum_of_each_row_tensor, GPT2_VOCAB_SIZE,
                                     scratch_val);
        });
        times_list[2] = gpu_milliseconds([&] {
            classifier_bf16_forward(bf16_val.losses_tensor, bf16_val.max_of_each_row_tensor, bf16_val.sum_of_each_row_tensor,
                                    bf16_val.tile_max_tensor, bf16_val.tile_sum_tensor, bf16_val.logits_tensor, lnf_output_tensor,
                                    wte_tensor, targets_tensor, GPT2_VOCAB_SIZE);
        });
        times_list[3] = gpu_milliseconds([&] {
            classifier_bf16_backward(bf16_val.lnf_output_grad_tensor, bf16_val.wte_grad_tensor, bf16_val.logits_tensor,
                                     losses_grad_tensor, bf16_val.logits_tensor, lnf_output_tensor, wte_tensor, targets_tensor,
                                     bf16_val.max_of_each_row_tensor, bf16_val.sum_of_each_row_tensor, GPT2_VOCAB_SIZE);
        });
        times_list[4] = gpu_milliseconds([&] {
            pytorch_loss_tensor = torch::cross_entropy_loss(
                torch::linear(pytorch_lnf_output_tensor, pytorch_wte_tensor).narrow(1, 0, GPT2_VOCAB_SIZE),
                pytorch_targets_tensor);
        });
        pytorch_lnf_output_tensor.mutable_grad() = torch::Tensor();   // lnf_output's gradient written, as mine, not added onto
        times_list[5] = gpu_milliseconds([&] { pytorch_loss_tensor.backward(); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 2 * SIDES; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages and the head's TFLOPS
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double forward_gflop_val = 2.0 * rows_val * GPT2_EMBED_DIM * GPT2_PADDED_VOCAB_SIZE / 1e9;
    print_test_name(__func__);
    printf("    average of steps %d..%d:   forward ms   backward ms   forward TFLOPS   backward TFLOPS\n",
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

    // 4. the 8-bit side agrees with PyTorch at the end, within 8 bits' error
    torch::Tensor mean_loss_tensor = pytorch_copy_of(eight_bit_val.losses_tensor).mean();
    ASSERT_NEAR(mean_loss_tensor.item<double>(), pytorch_loss_tensor.item<double>(),
                EIGHT_BIT_TOLERANCE * pytorch_loss_tensor.item<double>());
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.lnf_output_grad_tensor, pytorch_lnf_output_tensor.grad()), 0.0,
                EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.wte_grad_tensor, pytorch_wte_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();

    printf("\nall tunes done\n");
    return 0;
}
