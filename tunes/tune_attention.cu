// tune_attention.cu — the timings of tests/test_attention.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size, time_flash_against_the_steps_and_pytorch,
// time_each_softmax_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_attention.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/attention_<date>_<time>.txt.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/kernels/attention.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values. A bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it, and attention rounds several times in a row:
// the scores before the softmax (a score of 4 by up to 0.008, which moves its weight by 0.8%), the weights,
// the outputs. The backward rounds da, ds and the three gradients on top, and ds = a (da - row_dot) takes
// two close numbers apart, so its errors are larger next to its own size.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_HEAD_SIZE 64             // and their columns
#define GPT2_SEQ_LEN 1024             // its tokens per row
#define GPT2_EMBED_DIM (GPT2_NUM_HEADS * GPT2_HEAD_SIZE)   // 768

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 9.4M qkv values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);  // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: GPT-2's weights are 50M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_forward and attention_backward take, for one size, made with 0s except qkv and
/// output_grad, which the test fills.
struct AttentionTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor weights_tensor;
    Tensor head_outputs_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor head_outputs_grad_tensor;
    Tensor weights_grad_tensor;
    Tensor query_grad_tensor;
    Tensor key_grad_tensor;
    Tensor value_grad_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens and head_size columns, all of dtype.
    AttentionTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val, size_t head_size_val,
                     DType dtype_val = DType::Float32)
    {
        size_t embed_dim_val = num_heads_val * head_size_val;
        Shape per_head_list = {batch_size_val, num_heads_val, seq_len_val, head_size_val};
        Shape per_pair_list = {batch_size_val, num_heads_val, seq_len_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, dtype_val);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, dtype_val);
        query_tensor = Tensor::zeros(per_head_list, dtype_val), key_tensor = Tensor::zeros(per_head_list, dtype_val);
        value_tensor = Tensor::zeros(per_head_list, dtype_val);
        weights_tensor = Tensor::zeros(per_pair_list, dtype_val), head_outputs_tensor = Tensor::zeros(per_head_list, dtype_val);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), dtype_val);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), dtype_val);
        head_outputs_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
        weights_grad_tensor = Tensor::zeros(per_pair_list, dtype_val);
        query_grad_tensor = Tensor::zeros(per_head_list, dtype_val), key_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
        value_grad_tensor = Tensor::zeros(per_head_list, dtype_val);
    }

    /// Runs attention_forward on these tensors, at these block sizes (the defaults if not given).
    void forward(int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                 int softmax_threads_per_block_val = ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK)
    {
        attention_forward(output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor, head_outputs_tensor,
                          qkv_tensor, threads_per_block_val, softmax_threads_per_block_val);
    }

    /// Runs attention_backward on these tensors, at these block sizes (the defaults if not given).
    void backward(int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                  int softmax_threads_per_block_val = ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK)
    {
        attention_backward(qkv_grad_tensor, output_grad_tensor, query_tensor, key_tensor, value_tensor, weights_tensor,
                           head_outputs_grad_tensor, weights_grad_tensor, query_grad_tensor, key_grad_tensor,
                           value_grad_tensor, threads_per_block_val, softmax_threads_per_block_val);
    }
};

/// Every tensor attention_flash_forward and attention_flash_backward take, for one size, all bfloat16 but
/// logsumexp and row_dot (float), made with 0s except qkv and output_grad, which the test fills.
struct FlashTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor logsumexp_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor row_dot_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        Shape per_head_list = {batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        query_tensor = Tensor::zeros(per_head_list, DType::BFloat16), key_tensor = Tensor::zeros(per_head_list, DType::BFloat16);
        value_tensor = Tensor::zeros(per_head_list, DType::BFloat16);
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_flash_forward on these tensors.
    void forward()
    {
        attention_flash_forward(output_tensor, query_tensor, key_tensor, value_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_flash_backward on these tensors, row_dot at this block size (the default if not given).
    void backward(int threads_per_block_val = ATTENTION_ROW_DOT_THREADS_PER_BLOCK)
    {
        attention_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, query_tensor, key_tensor, value_tensor,
                                 logsumexp_tensor, row_dot_tensor, threads_per_block_val);
    }
};

/// Tells if this GPU runs flash attention (sm_80 or later); if not, prints why the test that asked is skipped.
static bool flash_runs_here(const char *test_ptr)
{
    if (attention_flash_fits(DType::BFloat16, GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: flash attention needs a GPU of compute capability 8.0 or more (A100, H100)\n");
    return false;
}

/// Runs PyTorch's causal attention on GPT-2 small's size into `pytorch_output_tensor`, from `pytorch_qkv_tensor`.
static void pytorch_gpt2_forward(const torch::Tensor &pytorch_qkv_tensor, torch::Tensor &pytorch_output_tensor)
{
    std::vector<torch::Tensor> parts_list = pytorch_qkv_tensor.split(GPT2_EMBED_DIM, 2);
    auto heads_val = [](const torch::Tensor &part_tensor) {
        return part_tensor.view({4, GPT2_SEQ_LEN, GPT2_NUM_HEADS, GPT2_HEAD_SIZE}).transpose(1, 2);
    };
    torch::Tensor y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]),
                                                                 heads_val(parts_list[2]), {}, 0.0, true);
    pytorch_output_tensor = y_tensor.transpose(1, 2).contiguous().view({4, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
}

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size: {4, 1024} tokens, 12 heads of
    // 64. Mine against PyTorch's scaled_dot_product_attention (causal) with autograd, which on a T4 is a
    // fused kernel that never writes the 201 MB of weights. Each side writes its gradient over each step;
    // at the end they must agree.

    // 1. random qkv and output_grad, mine and PyTorch's
    torch::manual_seed(0);
    AttentionTensors mine_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape());
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape());
    torch::Tensor pytorch_qkv_tensor = pytorch_copy_of(mine_val.qkv_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(mine_val.output_grad_tensor);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work, so the averages are of the rest:
    //    in step 1 CUDA loads each side's kernels and cuBLAS picks its own
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { mine_val.forward(); });
        times_list[1] = gpu_milliseconds([&] { mine_val.backward(); });
        times_list[2] = gpu_milliseconds([&] { pytorch_gpt2_forward(pytorch_qkv_tensor, pytorch_output_tensor); });
        pytorch_qkv_tensor.mutable_grad() = torch::Tensor(); // qkv's gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_output_tensor), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_qkv_tensor.grad()), 0.0, TOLERANCE);
}

// ══ part 1b: flash attention timed against PyTorch, and against the attention steps ═══

static void time_flash_against_the_steps_and_pytorch(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size, {4, 1024} tokens, 12 heads of
    // 64, all in bfloat16: flash attention, the attention steps (cuBLAS products, the softmax kernels, the 201
    // MB of weights) and PyTorch's scaled_dot_product_attention (causal), which in bfloat16 on an sm_80 GPU is
    // its own flash attention. Averages of steps 3..20. Expect flash attention faster than the steps, and at
    // the end its output and qkv_grad within the bfloat16 tolerances of PyTorch's.
    if (!flash_runs_here(__func__))
        return;
    // 1. random bfloat16 values for all three sides
    torch::manual_seed(0);
    AttentionTensors steps_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE, DType::BFloat16);
    steps_val.qkv_tensor = random_tensor_on_gpu(steps_val.qkv_tensor.shape()).to(DType::BFloat16);
    steps_val.output_grad_tensor = random_tensor_on_gpu(steps_val.output_tensor.shape()).to(DType::BFloat16);
    FlashTensors flash_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    flash_val.qkv_tensor.copy_(steps_val.qkv_tensor);
    flash_val.output_grad_tensor.copy_(steps_val.output_grad_tensor);
    torch::Tensor pytorch_qkv_tensor = pytorch_copy_of(steps_val.qkv_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(steps_val.output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps of each side; the first WARM_UP_STEPS left out
    double totals_list[6] = {};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[6];
        times_list[0] = gpu_milliseconds([&] { flash_val.forward(); });
        times_list[1] = gpu_milliseconds([&] { flash_val.backward(); });
        times_list[2] = gpu_milliseconds([&] { steps_val.forward(); });
        times_list[3] = gpu_milliseconds([&] { steps_val.backward(); });
        times_list[4] = gpu_milliseconds([&] { pytorch_gpt2_forward(pytorch_qkv_tensor, pytorch_output_tensor); });
        pytorch_qkv_tensor.mutable_grad() = torch::Tensor();
        times_list[5] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_val <= WARM_UP_STEPS)
            continue;
        for (int column_val = 0; column_val < 6; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    average of steps %d..%d, ms:   forward   backward   both\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[] = {"flash attention", "the steps", "PyTorch (flash)"};
    for (int side_val = 0; side_val < 3; side_val++)
        printf("        %-22s %10.3f %10.3f %6.3f\n", sides_list[side_val], totals_list[2 * side_val] / averaged_val,
               totals_list[2 * side_val + 1] / averaged_val,
               (totals_list[2 * side_val] + totals_list[2 * side_val + 1]) / averaged_val);
    printf("    the steps take %.2fx as long as flash attention, PyTorch %.2fx\n",
           (totals_list[2] + totals_list[3]) / (totals_list[0] + totals_list[1]),
           (totals_list[4] + totals_list[5]) / (totals_list[0] + totals_list[1]));

    // 4. flash attention is the faster, and agrees with PyTorch
    ASSERT_EQ(totals_list[0] + totals_list[1] < totals_list[2] + totals_list[3], true);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.output_tensor, pytorch_output_tensor.to(torch::kFloat)), 0.0,
                BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash_val.qkv_grad_tensor, pytorch_qkv_tensor.grad().to(torch::kFloat)), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_softmax_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the two softmax kernels on GPT-2 small's weights, {4, 12, 1024, 1024} (49,152 rows),
    // in float and in bfloat16, for 9 block counts x 4 thread counts, each the average of 20 runs: the 7 picked
    // on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264 on an SXM;
    // on a T4 they repeat 40 and 80). The 36 launches take turns: 20 rounds, each timing every launch once (the forward and the backward in float,
    // then both in bfloat16). One launch's 20 runs in a row would all land in the same stretch of the test,
    // and a T4's clock can drift as it heats up (the first launch timed would get the fastest clock); taking
    // turns gives every launch runs from early, middle and late alike. Run 1 is counted too: warm_up_gpu has
    // already done the first-time work. They are launched here directly, <<<blocks, threads>>>, instead of
    // through attention_forward and attention_backward. Their launches are the fastest in bfloat16, the dtype
    // training runs in, on an H100 PCIe: both one warp per row, 6144 x ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK
    // and 6144 x ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK (256 each). A T4's sweep had picked 80 x 128 for
    // the backward (320 warps, 154 rows each), 5.3x slower on the H100. The two cuBLAS products around them are
    // cuBLAS's to launch, and the four kernels that move values are the same kind as gelu.cuh's. A bfloat16
    // row is 2 KB instead of 4 KB, so twice as many warps' rows fit in the T4's 4 MB L2: its best launch
    // may differ.
    //
    // Both kernels work in place, so the timed runs change the values; before each check of the result,
    // the scores (or da) are copied back from a kept copy. Expect: every launch gives the same weights and
    // ds as the usual ones of its dtype, to the last bit: each row is worked out by one warp, the same way
    // whichever.

    // 1. GPT-2 small's size: a forward and a backward fill weights, and the scratch the steps need
    torch::manual_seed(0);
    AttentionTensors tensors_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.output_grad_tensor = random_tensor_on_gpu(tensors_val.output_tensor.shape());
    tensors_val.forward();
    tensors_val.backward();

    // 2. the scores (F2's product) and da (B2's), kept to start each check from; then the usual launches' results
    size_t rows_val = 4 * GPT2_NUM_HEADS * GPT2_SEQ_LEN;
    size_t seq_len_val = GPT2_SEQ_LEN;
    const int heads_in_batch_val = 4 * GPT2_NUM_HEADS;
    const long long head_values_val = (long long)GPT2_SEQ_LEN * GPT2_HEAD_SIZE;
    const long long head_weights_val = (long long)GPT2_SEQ_LEN * GPT2_SEQ_LEN;
    Tensor scores_tensor = Tensor::empty(tensors_val.weights_tensor.shape());
    Tensor weights_grad_before_tensor = Tensor::empty(tensors_val.weights_tensor.shape());
    float scale_val = 1.0f / sqrtf((float)GPT2_HEAD_SIZE);
    float one_val = 1.0f;
    float zero_val = 0.0f;
    attentionCublasCheck(cublasSgemmStridedBatched(attention_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, GPT2_SEQ_LEN, GPT2_SEQ_LEN,
                                                   GPT2_HEAD_SIZE, &scale_val, tensors_val.key_tensor.data(), GPT2_HEAD_SIZE,
                                                   head_values_val, tensors_val.query_tensor.data(), GPT2_HEAD_SIZE,
                                                   head_values_val, &zero_val, scores_tensor.data(), GPT2_SEQ_LEN,
                                                   head_weights_val, heads_in_batch_val));
    attentionCublasCheck(cublasSgemmStridedBatched(attention_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, GPT2_SEQ_LEN, GPT2_SEQ_LEN,
                                                   GPT2_HEAD_SIZE, &one_val, tensors_val.value_tensor.data(), GPT2_HEAD_SIZE,
                                                   head_values_val, tensors_val.head_outputs_grad_tensor.data(),
                                                   GPT2_HEAD_SIZE, head_values_val, &zero_val,
                                                   weights_grad_before_tensor.data(), GPT2_SEQ_LEN, head_weights_val,
                                                   heads_in_batch_val));
    torch::Tensor usual_weights_tensor = pytorch_copy_of(tensors_val.weights_tensor);
    torch::Tensor usual_scores_grad_tensor = pytorch_copy_of(tensors_val.weights_grad_tensor);
    int one_warp_per_row_blocks_val = attention_blocks(rows_val, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK,
                                                       THREADS_PER_WARP);   // 6144

    // 3. the same scores and da in bfloat16, and the usual launches' bfloat16 weights and ds from them
    Tensor scores_bf16_tensor = scores_tensor.to(DType::BFloat16);
    Tensor weights_grad_before_bf16_tensor = weights_grad_before_tensor.to(DType::BFloat16);
    Tensor weights_bf16_tensor = scores_bf16_tensor.clone();
    Tensor weights_grad_bf16_tensor = weights_grad_before_bf16_tensor.clone();
    attention_softmax_forward_kernel<nv_bfloat16><<<one_warp_per_row_blocks_val, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK>>>(
        weights_bf16_tensor.data_ptr<nv_bfloat16>(), rows_val, seq_len_val);
    attention_softmax_backward_kernel<nv_bfloat16>
        <<<one_warp_per_row_blocks_val, ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK>>>(
            weights_grad_bf16_tensor.data_ptr<nv_bfloat16>(), weights_bf16_tensor.data_ptr<nv_bfloat16>(), rows_val,
            seq_len_val);
    cudaCheck(cudaGetLastError());
    torch::Tensor usual_weights_bf16_tensor = pytorch_copy_of(weights_bf16_tensor);
    torch::Tensor usual_scores_grad_bf16_tensor = pytorch_copy_of(weights_grad_bf16_tensor);

    // 4. the four launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        attention_softmax_forward_kernel<float><<<blocks_val, threads_val>>>(tensors_val.weights_tensor.data(), rows_val,
                                                                             seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_val = [&] {
        attention_softmax_backward_kernel<float><<<blocks_val, threads_val>>>(
            tensors_val.weights_grad_tensor.data(), tensors_val.weights_tensor.data(), rows_val, seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16_val = [&] {
        attention_softmax_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            weights_bf16_tensor.data_ptr<nv_bfloat16>(), rows_val, seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_bf16_val = [&] {
        attention_softmax_backward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            weights_grad_bf16_tensor.data_ptr<nv_bfloat16>(), weights_bf16_tensor.data_ptr<nv_bfloat16>(), rows_val,
            seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = one_warp_per_row_blocks_val;
    threads_val = ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK;
    warm_up_gpu(backward_bf16_val);

    // 5. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int t4_backward_blocks_val = 80;                   // a T4's pick for the backward: 80 x 128
    const int t4_backward_threads_val = 4 * THREADS_PER_WARP;
    int block_counts_list[] = {40, t4_backward_blocks_val, 160, 320, 640, 1280, one_warp_per_row_blocks_val, sms_val,
                               2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 9 * thread_count_choices_val;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]; one time per kernel, in kernel_names' order
    float total_ms_list[launches_val][4] = {};
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(forward_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(backward_val);
            total_ms_list[launch_val][2] += gpu_milliseconds(forward_bf16_val);
            total_ms_list[launch_val][3] += gpu_milliseconds(backward_bf16_val);
        }
    }

    // 6. print each launch's averages, and check its results against the usual ones
    const char *kernel_names_list[] = {"forward", "backward", "bf16 forward", "bf16 backward"};
    float fastest_ms_list[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    int fastest_blocks_list[4] = {};
    int fastest_threads_list[4] = {};
    int launches_with_the_same_result_val = 0;
    size_t weights_bytes_val = tensors_val.weights_tensor.numel() * sizeof(float);
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   rows per warp   forward   backward   bf16 forward   bf16 backward   "
           "same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[4];
        for (int kernel_index_val = 0; kernel_index_val < 4; kernel_index_val++)
        {
            times_list[kernel_index_val] = total_ms_list[launch_val][kernel_index_val] / SWEEP_ROUNDS;
            if (times_list[kernel_index_val] < fastest_ms_list[kernel_index_val])
            {
                fastest_ms_list[kernel_index_val] = times_list[kernel_index_val];
                fastest_blocks_list[kernel_index_val] = blocks_val;
                fastest_threads_list[kernel_index_val] = threads_val;
            }
        }

        // one forward from the scores and one backward from da with this launch in each dtype, against the usual
        cudaCheck(cudaMemcpy(tensors_val.weights_tensor.data(), scores_tensor.data(), weights_bytes_val,
                             cudaMemcpyDeviceToDevice));
        forward_val();
        cudaCheck(cudaMemcpy(tensors_val.weights_grad_tensor.data(), weights_grad_before_tensor.data(), weights_bytes_val,
                             cudaMemcpyDeviceToDevice));
        backward_val();
        weights_bf16_tensor.copy_(scores_bf16_tensor);
        forward_bf16_val();
        weights_grad_bf16_tensor.copy_(weights_grad_before_bf16_tensor);
        backward_bf16_val();
        bool same_val = torch::equal(pytorch_copy_of(tensors_val.weights_tensor), usual_weights_tensor) &&
                        torch::equal(pytorch_copy_of(tensors_val.weights_grad_tensor), usual_scores_grad_tensor) &&
                        torch::equal(pytorch_copy_of(weights_bf16_tensor), usual_weights_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(weights_grad_bf16_tensor), usual_scores_grad_bf16_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t warps_launched_val = (size_t)blocks_val * threads_val / THREADS_PER_WARP;
        int rows_per_warp_val = (int)CEIL_DIV(rows_val, warps_launched_val);   // the most
        const char *usual_ptr =
            blocks_val == one_warp_per_row_blocks_val && threads_val == ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK
                ? "   <- what attention_forward and attention_backward launch: one warp per row"
            : blocks_val == t4_backward_blocks_val && threads_val == t4_backward_threads_val
                ? "   <- a T4's pick for the backward, what it launched before"
                : "";
        printf("                  %6d   %7d   %13d   %7.3f   %8.3f   %12.3f   %13.3f   %-11s%s\n", blocks_val, threads_val,
               rows_per_warp_val, times_list[0], times_list[1], times_list[2], times_list[3], same_val ? "yes" : "NO",
               usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < 4; kernel_index_val++)
        printf("    fastest %-13s  %5d blocks x %4d threads: %.3f ms\n", kernel_names_list[kernel_index_val],
               fastest_blocks_list[kernel_index_val], fastest_threads_list[kernel_index_val],
               fastest_ms_list[kernel_index_val]);

    // 7. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();
    time_flash_against_the_steps_and_pytorch();
    time_each_softmax_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
