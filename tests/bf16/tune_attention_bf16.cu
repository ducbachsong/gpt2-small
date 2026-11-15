// tune_attention_bf16.cu — the timings of llmc/kernels/bf16/attention_bf16.cuh and attention_flash_bf16.cuh, apart from tests/bf16/test_attention_bf16.cu
// so the tests only check numbers: time_20_steps_on_the_model_size, time_flash_against_the_steps_and_pytorch,
// time_each_backward_stage_count, time_each_softmax_kernel_with_other_block_and_thread_counts. The size is the
// model's attention (32 layers x 2,304, train_bf16.cu's): 18 heads of 128, 2,048 tokens a row.
//
// Built as tests/bf16/test_attention_bf16.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/attention_bf16_<date>_<time>.txt.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../../llmc/kernels/bf16/attention_bf16.cuh"
#include "../../llmc/kernels/bf16/attention_flash_bf16.cuh"

// Against PyTorch in bfloat16 on the same values: both round to bfloat16 several times (the scores, the attention_probs,
// the output; the backward's da, ds and gradients on top), each by up to 2^-8 of a value. Each tolerance is the
// largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_NUM_HEADS 18             // the model's heads
#define GPT2_HEAD_SIZE FLASH_HEAD_SIZE   // and their columns: 128
#define GPT2_SEQ_LEN 2048             // its tokens per row
#define GPT2_EMBED_DIM (GPT2_NUM_HEADS * GPT2_HEAD_SIZE)   // 2,304

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a float tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for the model's 57M qkv values (4 rows).
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
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_bf16_forward and attention_bf16_backward take, for one size, all bfloat16, made with
/// 0s except qkv and output_grad, which the tune fills.
struct AttentionTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor attention_probs_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor attention_probs_grad_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens and head_size columns.
    AttentionTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val, size_t head_size_val)
    {
        size_t embed_dim_val = num_heads_val * head_size_val;
        Shape per_pair_list = {batch_size_val, num_heads_val, seq_len_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        attention_probs_tensor = Tensor::zeros(per_pair_list, DType::BFloat16);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        attention_probs_grad_tensor = Tensor::zeros(per_pair_list, DType::BFloat16);
    }

    /// Runs attention_bf16_forward on these tensors, at the default block size.
    void forward()
    {
        attention_bf16_forward(output_tensor, attention_probs_tensor, qkv_tensor);
    }

    /// Runs attention_bf16_backward on these tensors, at the default block size.
    void backward()
    {
        attention_bf16_backward(qkv_grad_tensor, output_grad_tensor, qkv_tensor, attention_probs_tensor, attention_probs_grad_tensor);
    }
};

/// Every tensor attention_bf16_flash_forward and attention_bf16_flash_backward take, for one size, all bfloat16
/// but logsumexp and row_dot (float), made with 0s except qkv and output_grad, which the tune fills.
struct FlashTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor logsumexp_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor row_dot_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_bf16_flash_forward on these tensors.
    void forward()
    {
        attention_bf16_flash_forward(output_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_bf16_flash_backward on these tensors.
    void backward()
    {
        attention_bf16_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, qkv_tensor, logsumexp_tensor,
                                      row_dot_tensor);
    }
};

/// Tells if this GPU runs flash attention (an H100); if not, prints why the tune that asked is skipped.
static bool flash_runs_here(const char *test_ptr)
{
    if (attention_bf16_flash_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: flash attention needs a GPU of compute capability %d.%d (an H100: WGMMA and TMA)\n",
           HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
    return false;
}

/// Runs PyTorch's causal attention on the model's size (4 rows) into `pytorch_output_tensor`, from `pytorch_qkv_tensor`,
/// in the dtype of pytorch_qkv_tensor.
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

static void time_20_steps_on_the_model_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of the attention steps on the model's size, in
    // bfloat16: {4, 2048} tokens, 18 heads of 128. Mine against PyTorch's scaled_dot_product_attention (causal)
    // with autograd, in bfloat16 too, which on an sm_80 GPU or later is its flash attention: it never writes the
    // 604 MB of attention_probs. Each side writes its gradient over each step; at the end they must agree within the
    // bfloat16 tolerances.

    // 1. random bfloat16 qkv and output_grad, mine and PyTorch's
    torch::manual_seed(0);
    AttentionTensors mine_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape()).to(DType::BFloat16);
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape()).to(DType::BFloat16);
    torch::Tensor pytorch_qkv_tensor = pytorch_copy_of(mine_val.qkv_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(mine_val.output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work, so the averages are of the rest:
    //    in step 1 CUDA loads each side's kernels and PyTorch's cuBLAS picks its own
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { mine_val.forward(); });
        times_list[1] = gpu_milliseconds([&] { mine_val.backward(); });
        times_list[2] = gpu_milliseconds([&] { pytorch_gpt2_forward(pytorch_qkv_tensor, pytorch_output_tensor); });
        pytorch_qkv_tensor.mutable_grad() = torch::Tensor(); // qkv's gradient written, as mine, not added onto
        times_list[3] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 4; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
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
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_output_tensor.to(torch::kFloat)), 0.0,
                BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_qkv_tensor.grad().to(torch::kFloat)), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

static void time_flash_against_the_steps_and_pytorch(void)
{
    // Prints the times of 20 forwards and 20 backwards on the model's size, {4, 2048} tokens, 18 heads of
    // 128, all in bfloat16: flash attention, the attention steps (attention_product_kernel's products, the
    // softmax kernels, the 604 MB of attention_probs) and PyTorch's scaled_dot_product_attention (causal), which in bfloat16 on an sm_80 GPU is
    // its own flash attention. Averages of steps 3..20. Expect flash attention faster than the steps, and at
    // the end its output and qkv_grad within the bfloat16 tolerances of PyTorch's.
    if (!flash_runs_here(__func__))
        return;
    // 1. random bfloat16 values for all three sides
    torch::manual_seed(0);
    AttentionTensors steps_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE);
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
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[6];
        times_list[0] = gpu_milliseconds([&] { flash_val.forward(); });
        times_list[1] = gpu_milliseconds([&] { flash_val.backward(); });
        times_list[2] = gpu_milliseconds([&] { steps_val.forward(); });
        times_list[3] = gpu_milliseconds([&] { steps_val.backward(); });
        times_list[4] = gpu_milliseconds([&] { pytorch_gpt2_forward(pytorch_qkv_tensor, pytorch_output_tensor); });
        pytorch_qkv_tensor.mutable_grad() = torch::Tensor();
        times_list[5] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 6; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    average of steps %d..%d, ms:   forward   backward   both\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[] = {"flash attention", "the steps", "PyTorch (flash)"};
    for (int side_index_val = 0; side_index_val < 3; side_index_val++)
        printf("        %-22s %10.3f %10.3f %6.3f\n", sides_list[side_index_val], totals_list[2 * side_index_val] / averaged_val,
               totals_list[2 * side_index_val + 1] / averaged_val,
               (totals_list[2 * side_index_val] + totals_list[2 * side_index_val + 1]) / averaged_val);
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

/// One flash attention backward on these tensors with KEY_GRAD_STAGES and QUERY_GRAD_STAGES sets of tiles: what the
/// sweep below times for each pair of stage counts.
///
///   flash_val   the tensors: qkv_grad written, the rest read
template <int KEY_GRAD_STAGES, int QUERY_GRAD_STAGES>
static void flash_backward_at(FlashTensors &flash_val)
{
    attention_bf16_flash_backward<KEY_GRAD_STAGES, QUERY_GRAD_STAGES>(flash_val.qkv_grad_tensor, flash_val.output_grad_tensor,
                                                                      flash_val.output_tensor, flash_val.qkv_tensor,
                                                                      flash_val.logsumexp_tensor, flash_val.row_dot_tensor);
}

/// A pair of stage counts, and the backward made for them.
struct StageSetting
{
    int key_grad_stages_val;
    int query_grad_stages_val;
    void (*backward_ptr)(FlashTensors &);
};

static const StageSetting STAGE_SETTINGS_LIST[] = {
    {1, 1, flash_backward_at<1, 1>}, {1, 2, flash_backward_at<1, 2>}, {1, 3, flash_backward_at<1, 3>}, {1, 4, flash_backward_at<1, 4>},
    {2, 1, flash_backward_at<2, 1>}, {2, 2, flash_backward_at<2, 2>}, {2, 3, flash_backward_at<2, 3>}, {2, 4, flash_backward_at<2, 4>},
    {3, 1, flash_backward_at<3, 1>}, {3, 2, flash_backward_at<3, 2>}, {3, 3, flash_backward_at<3, 3>}, {3, 4, flash_backward_at<3, 4>},
    {4, 1, flash_backward_at<4, 1>}, {4, 2, flash_backward_at<4, 2>}, {4, 3, flash_backward_at<4, 3>}, {4, 4, flash_backward_at<4, 4>},
};

#define TRAINING_BATCH_SIZE 8         // a training micro-batch: 8 rows of 2048 tokens (train_bf16.cu's default)

static void time_each_backward_stage_count(void)
{
    // Prints flash attention's backward on a training micro-batch of the model, {8, 2048} tokens, 18 heads of 128,
    // with key_grad's and query_grad's sets of tiles each 1 to ATTENTION_FLASH_BF16_MAX_STAGES, taking turns over
    // SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged; then the fastest against the default. 1 and 1:
    // ask, wait, work, as before the sets. row_dot is the same in every one.
    if (!flash_runs_here(__func__))
        return;
    // 1. random bfloat16 qkv and output_grad, and the forward the backward reads
    torch::manual_seed(0);
    FlashTensors flash_val(TRAINING_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    flash_val.qkv_tensor = random_tensor_on_gpu(flash_val.qkv_tensor.shape()).to(DType::BFloat16);
    flash_val.output_grad_tensor = random_tensor_on_gpu(flash_val.output_tensor.shape()).to(DType::BFloat16);
    flash_val.forward();
    // 2. SWEEP_ROUNDS rounds, each timing every pair once
    const int setting_count_val = (int)(sizeof(STAGE_SETTINGS_LIST) / sizeof(STAGE_SETTINGS_LIST[0]));
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    std::vector<double> totals_list(setting_count_val, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] { STAGE_SETTINGS_LIST[setting_index_val].backward_ptr(flash_val); });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[setting_index_val] += milliseconds_val;
        }
    // 3. the table: key_grad's sets down, query_grad's across; the fastest and the default
    int best_index_val = 0;
    int default_index_val = 0;
    print_test_name(__func__);
    printf("    backward ms, average of rounds %d..%d, {%d, %d} tokens; key_grad's sets down, query_grad's across\n    %-8s",
           WARM_UP_STEPS + 1, SWEEP_ROUNDS, TRAINING_BATCH_SIZE, GPT2_SEQ_LEN, "");
    for (int query_stages_val = 1; query_stages_val <= ATTENTION_FLASH_BF16_MAX_STAGES; query_stages_val++)
        printf(" %8d", query_stages_val);
    for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
    {
        const StageSetting &setting_val = STAGE_SETTINGS_LIST[setting_index_val];
        if (setting_val.query_grad_stages_val == 1)
            printf("\n    %-8d", setting_val.key_grad_stages_val);
        printf(" %8.3f", totals_list[setting_index_val] / averaged_val);
        if (totals_list[setting_index_val] < totals_list[best_index_val])
            best_index_val = setting_index_val;
        if (setting_val.key_grad_stages_val == ATTENTION_FLASH_BF16_KEY_GRAD_STAGES &&
            setting_val.query_grad_stages_val == ATTENTION_FLASH_BF16_QUERY_GRAD_STAGES)
            default_index_val = setting_index_val;
    }
    printf("\n    fastest: key_grad %d sets, query_grad %d sets, %.3f ms; the default (%d, %d) %.3f ms, %.2fx as long; 1 and 1 %.3f ms\n",
           STAGE_SETTINGS_LIST[best_index_val].key_grad_stages_val, STAGE_SETTINGS_LIST[best_index_val].query_grad_stages_val,
           totals_list[best_index_val] / averaged_val, ATTENTION_FLASH_BF16_KEY_GRAD_STAGES, ATTENTION_FLASH_BF16_QUERY_GRAD_STAGES,
           totals_list[default_index_val] / averaged_val, totals_list[default_index_val] / totals_list[best_index_val],
           totals_list[0] / averaged_val);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_softmax_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the two bfloat16 softmax kernels on the model's attention_probs, {4, 18, 2048, 2048} (147,456
    // rows), for 9 block counts x 4 thread counts, each the average of SWEEP_ROUNDS runs: the 7 picked on a T4,
    // then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264 on an SXM; on a T4 they
    // repeat 40 and 80). The 36 launches take turns: SWEEP_ROUNDS rounds, each timing every launch once, so a
    // GPU's clock drifting as it heats up falls on all launches alike. Run 1 is counted too: warm_up_gpu has
    // already done the first-time work. They are launched here directly, <<<blocks, threads>>>. Their launches
    // were the fastest on an H100 PCIe at GPT-2 small's size: both one warp per row, 6144 x 256 (18,432 x 256 here). A T4's sweep had picked 80 x 128 for
    // the backward (320 warps, 154 rows each), 5.3x slower on the H100.
    //
    // Both kernels work in place, so the timed runs change the values; before each check of the result, the
    // scores (or da) are copied back from a kept copy. Expect: every launch gives the same attention_probs and ds as the
    // usual ones, to the last bit: each row is worked out by one warp, the same way whichever.

    // 1. the model's size: a forward and a backward fill attention_probs and attention_probs_grad
    torch::manual_seed(0);
    AttentionTensors tensors_val(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN, GPT2_HEAD_SIZE);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape()).to(DType::BFloat16);
    tensors_val.output_grad_tensor = random_tensor_on_gpu(tensors_val.output_tensor.shape()).to(DType::BFloat16);
    tensors_val.forward();
    tensors_val.backward();

    // 2. the scores (F2's product) and da (B2's), kept to start each check from; then the usual launches' results
    //    (the squares after every token stay 0: the softmax kernels never read them)
    size_t rows_val = 4 * GPT2_NUM_HEADS * GPT2_SEQ_LEN;
    size_t seq_len_val = GPT2_SEQ_LEN;
    Tensor scores_tensor = Tensor::zeros(tensors_val.attention_probs_tensor.shape(), DType::BFloat16);
    Tensor attention_probs_grad_before_tensor = Tensor::zeros(tensors_val.attention_probs_tensor.shape(), DType::BFloat16);
    // each head's strides {batch, head, row, column}: query, key or value in qkv, keyᵀ or valueᵀ (the last two
    // swapped), output_grad, attention_probs
    AttentionStrides qkv_strides_val = {GPT2_SEQ_LEN * 3 * GPT2_EMBED_DIM, GPT2_HEAD_SIZE, 3 * GPT2_EMBED_DIM, 1};
    AttentionStrides qkv_turned_strides_val = {GPT2_SEQ_LEN * 3 * GPT2_EMBED_DIM, GPT2_HEAD_SIZE, 1, 3 * GPT2_EMBED_DIM};
    AttentionStrides output_strides_val = {GPT2_SEQ_LEN * GPT2_EMBED_DIM, GPT2_HEAD_SIZE, GPT2_EMBED_DIM, 1};
    AttentionStrides attention_probs_strides_val = {GPT2_NUM_HEADS * GPT2_SEQ_LEN * GPT2_SEQ_LEN, GPT2_SEQ_LEN * GPT2_SEQ_LEN, GPT2_SEQ_LEN, 1};
    const nv_bfloat16 *query_ptr = tensors_val.qkv_tensor.data_ptr<nv_bfloat16>();   // key and value follow, GPT2_EMBED_DIM apart
    attention_product(scores_tensor.data_ptr<nv_bfloat16>(), attention_probs_strides_val, query_ptr, qkv_strides_val,
                      query_ptr + GPT2_EMBED_DIM, qkv_turned_strides_val, GPT2_SEQ_LEN, GPT2_SEQ_LEN,
                      GPT2_HEAD_SIZE, 4, GPT2_NUM_HEADS, 1.0f / sqrtf((float)GPT2_HEAD_SIZE),
                      ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW);
    attention_product(attention_probs_grad_before_tensor.data_ptr<nv_bfloat16>(), attention_probs_strides_val,
                      (const nv_bfloat16 *)tensors_val.output_grad_tensor.data_ptr<nv_bfloat16>(), output_strides_val,
                      query_ptr + 2 * GPT2_EMBED_DIM, qkv_turned_strides_val, GPT2_SEQ_LEN, GPT2_SEQ_LEN,
                      GPT2_HEAD_SIZE, 4, GPT2_NUM_HEADS, 1.0f, ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW);
    int one_warp_per_row_blocks_val = attention_bf16_blocks(rows_val, ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK,
                                                            HARDWARE_THREADS_PER_WARP);   // 18,432
    tensors_val.attention_probs_tensor.copy_(scores_tensor);
    tensors_val.attention_probs_grad_tensor.copy_(attention_probs_grad_before_tensor);
    attention_bf16_softmax_forward_kernel<<<one_warp_per_row_blocks_val, ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK>>>(
        tensors_val.attention_probs_tensor.data_ptr<nv_bfloat16>(), rows_val, seq_len_val);
    attention_bf16_softmax_backward_kernel<<<one_warp_per_row_blocks_val, ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK>>>(
        tensors_val.attention_probs_grad_tensor.data_ptr<nv_bfloat16>(), tensors_val.attention_probs_tensor.data_ptr<nv_bfloat16>(),
        rows_val, seq_len_val);
    cudaCheck(cudaGetLastError());
    torch::Tensor usual_attention_probs_tensor = pytorch_copy_of(tensors_val.attention_probs_tensor);
    torch::Tensor usual_scores_grad_tensor = pytorch_copy_of(tensors_val.attention_probs_grad_tensor);

    // 3. the two launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        attention_bf16_softmax_forward_kernel<<<blocks_val, threads_val>>>(tensors_val.attention_probs_tensor.data_ptr<nv_bfloat16>(),
                                                                           rows_val, seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_val = [&] {
        attention_bf16_softmax_backward_kernel<<<blocks_val, threads_val>>>(
            tensors_val.attention_probs_grad_tensor.data_ptr<nv_bfloat16>(), tensors_val.attention_probs_tensor.data_ptr<nv_bfloat16>(),
            rows_val, seq_len_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = one_warp_per_row_blocks_val;
    threads_val = ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK;
    warm_up_gpu(backward_val);

    // 4. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int t4_backward_blocks_val = 80;                   // a T4's pick for the backward: 80 x 128
    const int t4_backward_threads_val = 4 * HARDWARE_THREADS_PER_WARP;
    int block_counts_list[] = {40, t4_backward_blocks_val, 160, 320, 640, 1280, one_warp_per_row_blocks_val, sms_val,
                               2 * sms_val};
    int thread_counts_list[] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 9 * thread_count_choices_val;
    const int kernel_count_val = 2;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]; one time per kernel, in kernel_names' order
    float total_ms_list[launches_val][kernel_count_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
        {
            blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
            total_ms_list[launch_index_val][0] += gpu_milliseconds(forward_val);
            total_ms_list[launch_index_val][1] += gpu_milliseconds(backward_val);
        }
    }

    // 5. print each launch's averages, and check its results against the usual ones
    const char *kernel_names_list[kernel_count_val] = {"forward", "backward"};
    float fastest_ms_list[kernel_count_val] = {1e9f, 1e9f};
    int fastest_blocks_list[kernel_count_val] = {};
    int fastest_threads_list[kernel_count_val] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   rows per warp   forward   backward   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        float times_list[kernel_count_val];
        for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
        {
            times_list[kernel_index_val] = total_ms_list[launch_index_val][kernel_index_val] / SWEEP_ROUNDS;
            if (times_list[kernel_index_val] < fastest_ms_list[kernel_index_val])
            {
                fastest_ms_list[kernel_index_val] = times_list[kernel_index_val];
                fastest_blocks_list[kernel_index_val] = blocks_val;
                fastest_threads_list[kernel_index_val] = threads_val;
            }
        }

        // one forward from the scores and one backward from da with this launch, against the usual
        tensors_val.attention_probs_tensor.copy_(scores_tensor);
        forward_val();
        tensors_val.attention_probs_grad_tensor.copy_(attention_probs_grad_before_tensor);
        backward_val();
        bool same_val = torch::equal(pytorch_copy_of(tensors_val.attention_probs_tensor), usual_attention_probs_tensor) &&
                        torch::equal(pytorch_copy_of(tensors_val.attention_probs_grad_tensor), usual_scores_grad_tensor);
        if (same_val)
            launches_with_the_same_result_val++;

        size_t warps_launched_val = (size_t)blocks_val * threads_val / HARDWARE_THREADS_PER_WARP;
        int rows_per_warp_val = (int)CEIL_DIV(rows_val, warps_launched_val);   // the most
        const char *usual_ptr =
            blocks_val == one_warp_per_row_blocks_val && threads_val == ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK
                ? "   <- what attention_bf16_forward and _backward launch: one warp per row"
            : blocks_val == t4_backward_blocks_val && threads_val == t4_backward_threads_val
                ? "   <- a T4's pick for the backward, what it launched before"
                : "";
        printf("                  %6d   %7d   %13d   %7.3f   %8.3f   %-11s%s\n", blocks_val, threads_val,
               rows_per_warp_val, times_list[0], times_list[1], same_val ? "yes" : "NO", usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
        printf("    fastest %-9s  %5d blocks x %4d threads: %.3f ms\n", kernel_names_list[kernel_index_val],
               fastest_blocks_list[kernel_index_val], fastest_threads_list[kernel_index_val],
               fastest_ms_list[kernel_index_val]);

    // 6. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_the_model_size();
    time_flash_against_the_steps_and_pytorch();
    time_each_backward_stage_count();
    time_each_softmax_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
