// tune_attention.cu — the timings of llmc/kernels/fp32/attention.cuh (float flash attention), apart from
// tests/fp32/test_attention.cu so the tests only check numbers: time_20_steps_on_gpt2_small_size,
// time_row_dot_kernel_with_other_block_and_thread_counts. bfloat16 (its steps and its flash attention):
// tune_attention_bf16.cu; 8-bit: tune_attention_8bit.cu.
//
// Built as tests/fp32/test_attention.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/attention_<date>_<time>.txt.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu
#include "../../llmc/kernels/fp32/attention.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_BATCH_SIZE 4             // the micro-batch the tunes time
#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_HEAD_SIZE ATTENTION_FLASH_HEAD_SIZE           // and their columns: 64
#define GPT2_SEQ_LEN 1024             // its tokens per row
#define GPT2_EMBED_DIM (GPT2_NUM_HEADS * GPT2_HEAD_SIZE)   // 768
#define MEGA 1e6                      // bytes in an MB

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

/// Makes a PyTorch float tensor on the GPU with a copy of my float tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: GPT-2's qkv_grad is 9.4M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_flash_forward and attention_flash_backward take, for one size, all float, made with 0s
/// except qkv and output_grad, which the tune fills.
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

    /// Makes them for batch_size x num_heads heads of seq_len tokens of ATTENTION_FLASH_HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * ATTENTION_FLASH_HEAD_SIZE;
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val});
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape());
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape());
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_flash_forward on these tensors.
    void forward()
    {
        attention_flash_forward(output_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_flash_backward on these tensors, at the default block size.
    void backward()
    {
        attention_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, qkv_tensor, logsumexp_tensor,
                                 row_dot_tensor);
    }
};

/// Runs PyTorch's causal attention on GPT-2 small's size into `pytorch_output_tensor`, from `pytorch_qkv_tensor`.
static void pytorch_gpt2_forward(const torch::Tensor &pytorch_qkv_tensor, torch::Tensor &pytorch_output_tensor)
{
    std::vector<torch::Tensor> parts_list = pytorch_qkv_tensor.split(GPT2_EMBED_DIM, 2);
    auto heads_val = [](const torch::Tensor &part_tensor) {
        return part_tensor.view({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_NUM_HEADS, GPT2_HEAD_SIZE}).transpose(1, 2);
    };
    torch::Tensor y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]),
                                                                 heads_val(parts_list[2]), {}, 0.0, true);
    pytorch_output_tensor = y_tensor.transpose(1, 2).contiguous().view({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
}

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size, in float: {4, 1024} tokens, 12 heads
    // of 64. Mine against PyTorch's scaled_dot_product_attention (causal) with autograd, which in float is a fused
    // kernel (memory-efficient attention) that never writes the 201 MB of attention_probs either. Both on the CUDA
    // cores. Averages of steps 3..20, with the MB mine keeps for the backward besides qkv and output (logsumexp).
    // Each side writes its gradient over each step; at the end they must agree.

    // 1. random qkv and output_grad, mine and PyTorch's
    torch::manual_seed(0);
    FlashTensors mine_val(GPT2_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    mine_val.qkv_tensor = random_tensor_on_gpu(mine_val.qkv_tensor.shape());
    mine_val.output_grad_tensor = random_tensor_on_gpu(mine_val.output_tensor.shape());
    torch::Tensor pytorch_qkv_tensor = pytorch_copy_of(mine_val.qkv_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(mine_val.output_grad_tensor);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps; the first WARM_UP_STEPS do first-time work, so the averages are of the rest:
    //    in step 1 CUDA loads each side's kernels and PyTorch picks its own
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
    double kept_val = (double)mine_val.logsumexp_tensor.numel() * sizeof(float);
    print_test_name(__func__);
    printf("    average of steps %d..%d:   my forward   my backward   PyTorch forward   PyTorch backward\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);
    printf("    mine keeps %.1f MB for the backward (logsumexp), against the steps' %.1f MB of attention_probs\n",
           kept_val / MEGA, (double)GPT2_BATCH_SIZE * GPT2_NUM_HEADS * GPT2_SEQ_LEN * GPT2_SEQ_LEN * sizeof(float) / MEGA);

    // 4. both sides agree at the end
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_output_tensor), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_qkv_tensor.grad()), 0.0, TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_row_dot_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of attention_flash_row_dot_kernel on GPT-2 small's size, {4, 12, 1024} (49,152 rows), for 9
    // block counts x 4 thread counts, each the average of SWEEP_ROUNDS runs: 40 .. 1280, as many as one warp per
    // row needs, and this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe). The 36 launches take turns:
    // SWEEP_ROUNDS rounds, each timing every launch once, so a GPU's clock drifting as it heats up falls on all
    // launches alike. Run 1 is counted too: warm_up_gpu has already done the first-time work. It is the only
    // flash kernel with a block size to choose; the tile kernels' is fixed by the tiles.
    //
    // Expect: every launch gives the same row_dot as the usual one, to the last bit: each row is one warp's.

    // 1. GPT-2 small's size: a forward fills output; output_grad random
    torch::manual_seed(0);
    FlashTensors tensors_val(GPT2_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.output_grad_tensor = random_tensor_on_gpu(tensors_val.output_tensor.shape());
    tensors_val.forward();
    size_t rows_val = tensors_val.logsumexp_tensor.numel();
    int one_warp_per_row_blocks_val =
        (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK);   // 6144

    // 2. the launch, at the launch the loop sets; the usual one's result
    int blocks_val = one_warp_per_row_blocks_val;
    int threads_val = ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK;
    auto row_dot_val = [&] {
        attention_flash_row_dot_kernel<<<blocks_val, threads_val>>>(
            tensors_val.row_dot_tensor.data(), tensors_val.output_grad_tensor.data(), tensors_val.output_tensor.data(),
            GPT2_NUM_HEADS, GPT2_SEQ_LEN, rows_val);
        cudaCheck(cudaGetLastError());
    };
    warm_up_gpu(row_dot_val);
    torch::Tensor usual_row_dot_tensor = pytorch_copy_of(tensors_val.row_dot_tensor);

    // 3. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 80, 160, 320, 640, 1280, one_warp_per_row_blocks_val, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    const int block_count_choices_val = 9;
    const int thread_count_choices_val = 4;
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]
    float total_ms_list[launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
        {
            blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
            total_ms_list[launch_index_val] += gpu_milliseconds(row_dot_val);
        }
    }

    // 4. print each launch's average, and check its result against the usual one
    float fastest_ms_val = 1e9f;
    int fastest_blocks_val = 0;
    int fastest_threads_val = 0;
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   rows per warp   row_dot   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        blocks_val = block_counts_list[launch_index_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        float time_val = total_ms_list[launch_index_val] / SWEEP_ROUNDS;
        if (time_val < fastest_ms_val)
        {
            fastest_ms_val = time_val;
            fastest_blocks_val = blocks_val;
            fastest_threads_val = threads_val;
        }
        // one run with this launch, against the usual
        tensors_val.row_dot_tensor.zero_();
        row_dot_val();
        bool same_val = torch::equal(pytorch_copy_of(tensors_val.row_dot_tensor), usual_row_dot_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        size_t warps_launched_val = (size_t)blocks_val * threads_val / HARDWARE_THREADS_PER_WARP;
        int rows_per_warp_val = (int)CEIL_DIV(rows_val, warps_launched_val);   // the most
        const char *usual_ptr = blocks_val == one_warp_per_row_blocks_val && threads_val == ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK
                                    ? "   <- what attention_flash_backward launches: one warp per row"
                                    : "";
        printf("                  %6d   %7d   %13d   %7.3f   %-11s%s\n", blocks_val, threads_val, rows_per_warp_val,
               time_val, same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest  %5d blocks x %4d threads: %.3f ms\n", fastest_blocks_val, fastest_threads_val, fastest_ms_val);

    // 5. every launch gave the same result
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();
    time_row_dot_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
