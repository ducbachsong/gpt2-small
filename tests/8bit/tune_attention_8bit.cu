// tune_attention_8bit.cu — the timings of llmc/kernels/8bit/attention_8bit.cuh and attention_flash_8bit.cuh, apart from tests/8bit/test_attention_8bit.cu
// so the tests only check numbers: time_8bit_against_flash_and_pytorch,
// time_rounding_and_turn_back_with_other_block_and_thread_counts.
//
// Built as tests/8bit/test_attention_8bit.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/attention_8bit_<date>_<time>.txt. Needs a GPU of compute capability 8.9 or more.
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, gigabytes_per_second, peak_gigabytes_per_second
#include "../../llmc/kernels/8bit/attention_flash_8bit.cuh"     // and through it attention_8bit.cuh, attention_flash_bf16.cuh

// Against PyTorch in bfloat16 on the values before rounding to E4M3: E4M3 moves each of query, key, value by up
// to 2^-4 (6%), see tests/test_attention_8bit.cu. Each is the largest difference, as a fraction of the largest
// value.
static const double EIGHT_BIT_TOLERANCE = 1e-1;
static const double EIGHT_BIT_GRAD_TOLERANCE = 1.5e-1;

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_SEQ_LEN 1024             // its tokens per row
#define GPT2_EMBED_DIM (GPT2_NUM_HEADS * FLASH_HEAD_SIZE)   // 768
#define GPT2_BATCH_SIZE 4             // rows of the micro-batch timed
#define MEGA 1e6

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a float tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 9.4M qkv values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my float or bfloat16 tensor's values, GPU to GPU.
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

/// Gives a tensor's bytes, copied to the CPU, of any dtype: to see that two tensors hold the same bits.
static std::vector<uint8_t> bytes_of(const Tensor &tensor)
{
    std::vector<uint8_t> bytes_list(tensor.numel() * tensor.element_size());
    cudaCheck(cudaMemcpy(bytes_list.data(), tensor.data_ptr(), bytes_list.size(), cudaMemcpyDeviceToHost));
    return bytes_list;
}

/// Every tensor attention_8bit_forward and attention_8bit_backward take, for one size, made with 0s except qkv
/// and output_grad, which the tune fills.
struct EightBitTensors
{
    // forward
    Tensor qkv_tensor;                 // bfloat16
    Tensor output_tensor;              // bfloat16
    Tensor query_8bit_tensor;          // Int32, 4 E4M3s in each int
    Tensor key_8bit_tensor;
    Tensor value_8bit_tensor;
    Tensor query_scales_tensor;        // float, one per head row
    Tensor key_scales_tensor;
    Tensor value_scales_tensor;
    Tensor logsumexp_tensor;           // float
    // backward
    Tensor output_grad_tensor;         // bfloat16
    Tensor qkv_grad_tensor;            // bfloat16
    Tensor qkv_rounded_tensor;         // bfloat16 scratch: the 8-bit ones turned back, where qkv has them
    Tensor row_dot_tensor;             // float scratch

    /// Makes them for batch_size x num_heads heads of seq_len tokens of FLASH_HEAD_SIZE columns.
    EightBitTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        Shape per_head_8bit_list = {batch_size_val, num_heads_val, seq_len_val, FLASH_HEAD_SIZE / ATTENTION_8BIT_VALUES_PER_INT};
        Shape per_row_list = {batch_size_val, num_heads_val, seq_len_val};
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val}, DType::BFloat16);
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        query_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        key_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        value_8bit_tensor = Tensor::zeros(per_head_8bit_list, DType::Int32);
        query_scales_tensor = Tensor::zeros(per_row_list), key_scales_tensor = Tensor::zeros(per_row_list);
        value_scales_tensor = Tensor::zeros(per_row_list), logsumexp_tensor = Tensor::zeros(per_row_list);
        output_grad_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        qkv_rounded_tensor = Tensor::zeros(qkv_tensor.shape(), DType::BFloat16);
        row_dot_tensor = Tensor::zeros(per_row_list);
    }

    /// Runs attention_8bit_forward on these tensors.
    void forward()
    {
        attention_8bit_forward(output_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor,
                               key_scales_tensor, value_scales_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_8bit_backward on these tensors.
    void backward()
    {
        attention_8bit_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, query_8bit_tensor, key_8bit_tensor,
                                value_8bit_tensor, query_scales_tensor, key_scales_tensor, value_scales_tensor,
                                logsumexp_tensor, qkv_rounded_tensor, row_dot_tensor);
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

/// Tells if this GPU runs 8-bit attention (sm_89 or later); if not, prints why the tune that asked is skipped.
static bool eight_bit_runs_here(const char *test_ptr)
{
    if (attention_8bit_fits(GPT2_SEQ_LEN, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test_ptr);
    printf("    skipped: 8-bit attention needs a GPU of compute capability 8.9 or more (L4, H100)\n");
    return false;
}

/// Runs PyTorch's causal attention on GPT-2 small's size into `pytorch_output_tensor`, from `pytorch_qkv_tensor`,
/// in the dtype of pytorch_qkv_tensor.
static void pytorch_gpt2_forward(const torch::Tensor &pytorch_qkv_tensor, torch::Tensor &pytorch_output_tensor)
{
    std::vector<torch::Tensor> parts_list = pytorch_qkv_tensor.split(GPT2_EMBED_DIM, 2);
    auto heads_val = [](const torch::Tensor &part_tensor) {
        return part_tensor.view({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_NUM_HEADS, FLASH_HEAD_SIZE}).transpose(1, 2);
    };
    torch::Tensor y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]),
                                                                 heads_val(parts_list[2]), {}, 0.0, true);
    pytorch_output_tensor = y_tensor.transpose(1, 2).contiguous().view({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
}

// ══ part 1: timed against bfloat16 flash attention and PyTorch ════════════════

static void time_8bit_against_flash_and_pytorch(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size, {4, 1024} tokens, 12 heads of 64:
    // 8-bit attention, attention_bf16.cuh's flash attention, and PyTorch's scaled_dot_product_attention (causal)
    // in bfloat16, its own flash attention. The three take turns in each step. Averages of steps 3..20. Then the
    // bytes each keeps from the forward for the backward: query, key, value (bfloat16, or E4M3 and a float scale
    // per head row) and logsumexp. Expect 8-bit attention's output and qkv_grad within the 8-bit tolerances of
    // PyTorch's, on the values before rounding.
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. random bfloat16 values for all three sides
    torch::manual_seed(0);
    EightBitTensors eight_bit_val(GPT2_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    eight_bit_val.qkv_tensor = random_tensor_on_gpu(eight_bit_val.qkv_tensor.shape()).to(DType::BFloat16);
    eight_bit_val.output_grad_tensor = random_tensor_on_gpu(eight_bit_val.output_tensor.shape()).to(DType::BFloat16);
    FlashTensors flash_val(GPT2_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    flash_val.qkv_tensor.copy_(eight_bit_val.qkv_tensor);
    flash_val.output_grad_tensor.copy_(eight_bit_val.output_grad_tensor);
    torch::Tensor pytorch_qkv_tensor = pytorch_copy_of(eight_bit_val.qkv_tensor).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_output_grad_tensor = pytorch_copy_of(eight_bit_val.output_grad_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS steps of each side; the first WARM_UP_STEPS left out
    double totals_list[6] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[6];
        times_list[0] = gpu_milliseconds([&] { eight_bit_val.forward(); });
        times_list[1] = gpu_milliseconds([&] { eight_bit_val.backward(); });
        times_list[2] = gpu_milliseconds([&] { flash_val.forward(); });
        times_list[3] = gpu_milliseconds([&] { flash_val.backward(); });
        times_list[4] = gpu_milliseconds([&] { pytorch_gpt2_forward(pytorch_qkv_tensor, pytorch_output_tensor); });
        pytorch_qkv_tensor.mutable_grad() = torch::Tensor(); // qkv's gradient written, as mine, not added onto
        times_list[5] = gpu_milliseconds([&] { pytorch_output_tensor.backward(pytorch_output_grad_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int column_index_val = 0; column_index_val < 6; column_index_val++)
            totals_list[column_index_val] += times_list[column_index_val];
    }

    // 3. print the averages, and the bytes kept for the backward
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double head_values_val = (double)GPT2_BATCH_SIZE * GPT2_NUM_HEADS * GPT2_SEQ_LEN * FLASH_HEAD_SIZE;   // 3.1M per tensor
    double head_rows_val = (double)GPT2_BATCH_SIZE * GPT2_NUM_HEADS * GPT2_SEQ_LEN;                      // 49,152
    double kept_8bit_val = 3 * head_values_val * sizeof(__nv_fp8_e4m3) + 3 * head_rows_val * sizeof(float) +
                           head_rows_val * sizeof(float);
    double kept_bf16_val = 3 * head_values_val * sizeof(nv_bfloat16) + head_rows_val * sizeof(float);
    print_test_name(__func__);
    printf("    average of steps %d..%d, ms:   forward   backward   both     kept for the backward, MB\n",
           WARM_UP_STEPS + 1, TIMED_STEPS);
    const char *sides_list[] = {"8-bit attention", "flash attention", "PyTorch (flash)"};
    const double kept_list[] = {kept_8bit_val, kept_bf16_val, kept_bf16_val};
    for (int side_index_val = 0; side_index_val < 3; side_index_val++)
        printf("        %-22s %10.3f %10.3f %6.3f   %10.1f\n", sides_list[side_index_val], totals_list[2 * side_index_val] / averaged_val,
               totals_list[2 * side_index_val + 1] / averaged_val,
               (totals_list[2 * side_index_val] + totals_list[2 * side_index_val + 1]) / averaged_val, kept_list[side_index_val] / MEGA);
    printf("    flash attention takes %.2fx as long as 8-bit attention in the forward, %.2fx in both; PyTorch %.2fx in both\n",
           totals_list[2] / totals_list[0], (totals_list[2] + totals_list[3]) / (totals_list[0] + totals_list[1]),
           (totals_list[4] + totals_list[5]) / (totals_list[0] + totals_list[1]));

    // 4. 8-bit attention stays near PyTorch on the values before rounding
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.output_tensor, pytorch_output_tensor.to(torch::kFloat)), 0.0,
                EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(eight_bit_val.qkv_grad_tensor, pytorch_qkv_tensor.grad().to(torch::kFloat)), 0.0,
                EIGHT_BIT_GRAD_TOLERANCE);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_rounding_and_turn_back_with_other_block_and_thread_counts(void)
{
    // Prints the time, GB/s and MBU of the two kernels that move values, on GPT-2 small's size (49,152 head rows
    // of query, key and value), for 5 block counts x 5 thread counts, each the average of SWEEP_ROUNDS runs taking
    // turns: the rounding (F1: reads qkv in bfloat16, writes E4M3 and a scale per head row, one warp per row)
    // and the turn-back (reads E4M3 and the scales, writes bfloat16, one int of 4 E4M3s per thread). The block
    // counts: as many as the work needs at that block size (one warp per row, one int per thread), and this GPU's
    // SM count x 1, 2, 4 and 8. Both are memory-bound: their bytes over the time is their speed.
    //
    //     rounding     reads 3 x 2 bytes, writes 3 x 1 byte per value, and 3 scales of 4 bytes per head row
    //     turn-back    reads 3 x 1 byte, writes 3 x 2 bytes per value, and reads the 3 scales
    //
    // Expect: every launch gives the default's bytes, to the last bit.
    if (!eight_bit_runs_here(__func__))
        return;
    // 1. GPT-2 small's size; the defaults' results
    torch::manual_seed(0);
    EightBitTensors tensors_val(GPT2_BATCH_SIZE, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape()).to(DType::BFloat16);
    tensors_val.forward();
    attention_8bit_turn_back(tensors_val.qkv_rounded_tensor, tensors_val.query_8bit_tensor, tensors_val.key_8bit_tensor,
                             tensors_val.value_8bit_tensor, tensors_val.query_scales_tensor, tensors_val.key_scales_tensor,
                             tensors_val.value_scales_tensor);
    std::vector<uint8_t> usual_8bit_list = bytes_of(tensors_val.query_8bit_tensor);
    std::vector<uint8_t> usual_bf16_list = bytes_of(tensors_val.qkv_rounded_tensor);
    size_t rows_val = (size_t)GPT2_BATCH_SIZE * GPT2_NUM_HEADS * GPT2_SEQ_LEN;
    size_t total_ints_val = tensors_val.query_8bit_tensor.numel();
    double values_val = (double)rows_val * FLASH_HEAD_SIZE;
    double moved_bytes_val = 3 * values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + 3 * (double)rows_val * sizeof(float);

    // 2. the two launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto round_qkv_val = [&] {
        attention_8bit_round_qkv_kernel<<<blocks_val, threads_val>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(tensors_val.query_8bit_tensor.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(tensors_val.key_8bit_tensor.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(tensors_val.value_8bit_tensor.data_ptr()), tensors_val.query_scales_tensor.data(),
            tensors_val.key_scales_tensor.data(), tensors_val.value_scales_tensor.data(),
            tensors_val.qkv_tensor.data_ptr<nv_bfloat16>(), GPT2_NUM_HEADS, GPT2_SEQ_LEN, rows_val);
        cudaCheck(cudaGetLastError());
    };
    auto turn_back_val = [&] {
        attention_8bit_turn_back_kernel<<<blocks_val, threads_val>>>(
            tensors_val.qkv_rounded_tensor.data_ptr<nv_bfloat16>(),
            reinterpret_cast<const __nv_fp8_e4m3*>(tensors_val.query_8bit_tensor.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(tensors_val.key_8bit_tensor.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(tensors_val.value_8bit_tensor.data_ptr()),
            tensors_val.query_scales_tensor.data(), tensors_val.key_scales_tensor.data(),
            tensors_val.value_scales_tensor.data(), GPT2_NUM_HEADS, GPT2_SEQ_LEN, total_ints_val);
        cudaCheck(cudaGetLastError());
    };
    threads_val = ATTENTION_8BIT_THREADS_PER_BLOCK;
    blocks_val = (int)CEIL_DIV(total_ints_val, (size_t)threads_val);
    warm_up_gpu(turn_back_val);

    // 3. time every launch, taking turns over SWEEP_ROUNDS rounds; block count choice 0 is "as many as the work needs"
    int sms_val = 0;                                         // this GPU's SMs: 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 5;
    const int thread_count_choices_val = 5;
    int sm_multiples_list[block_count_choices_val] = {0, 1, 2, 4, 8};
    int thread_counts_list[thread_count_choices_val] = {2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                        16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    const int kernel_count_val = 2;
    // the blocks of launch k for each kernel: SMs x the multiple, or as many as the work needs
    auto blocks_of_val = [&](int launch_index_val, int kernel_index_val) {
        int threads_of_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        int multiple_val = sm_multiples_list[launch_index_val / thread_count_choices_val];
        if (multiple_val > 0)
            return multiple_val * sms_val;
        return kernel_index_val == 0 ? (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_of_val)
                                     : (int)CEIL_DIV(total_ints_val, (size_t)threads_of_val);
    };
    float total_ms_list[launches_val][kernel_count_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
        {
            threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
            blocks_val = blocks_of_val(launch_index_val, 0);
            total_ms_list[launch_index_val][0] += gpu_milliseconds(round_qkv_val);
            blocks_val = blocks_of_val(launch_index_val, 1);
            total_ms_list[launch_index_val][1] += gpu_milliseconds(turn_back_val);
        }
    }

    // 4. print each launch's averages with GB/s and MBU, and check its results against the default's
    double peak_val = peak_gigabytes_per_second();
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB moved by each kernel; this GPU's peak %.0f GB/s\n", moved_bytes_val / MEGA, peak_val);
    printf("    threads   rounding: blocks       ms     GB/s    MBU   turn-back: blocks       ms     GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        // one rounding and one turn-back with this launch, against the default's
        tensors_val.query_8bit_tensor.zero_();
        tensors_val.qkv_rounded_tensor.zero_();
        blocks_val = blocks_of_val(launch_index_val, 0);
        round_qkv_val();
        int round_blocks_val = blocks_val;
        blocks_val = blocks_of_val(launch_index_val, 1);
        turn_back_val();
        bool same_val = bytes_of(tensors_val.query_8bit_tensor) == usual_8bit_list &&
                        bytes_of(tensors_val.qkv_rounded_tensor) == usual_bf16_list;
        if (same_val)
            launches_with_the_same_result_val++;
        float round_ms_val = total_ms_list[launch_index_val][0] / SWEEP_ROUNDS;
        float turn_back_ms_val = total_ms_list[launch_index_val][1] / SWEEP_ROUNDS;
        double round_speed_val = gigabytes_per_second(moved_bytes_val, round_ms_val);
        double turn_back_speed_val = gigabytes_per_second(moved_bytes_val, turn_back_ms_val);
        printf("    %7d   %16d   %6.3f   %6.0f   %3.0f%%   %17d   %6.3f   %6.0f   %3.0f%%   %s\n", threads_val,
               round_blocks_val, round_ms_val, round_speed_val, 100.0 * round_speed_val / peak_val, blocks_val,
               turn_back_ms_val, turn_back_speed_val, 100.0 * turn_back_speed_val / peak_val, same_val ? "yes" : "NO");
    }

    // 5. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8bit_against_flash_and_pytorch();
    time_rounding_and_turn_back_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
