// tune_round_8bit.cu — the timings of llmc/kernels/8bit/round_8bit.cuh, apart from tests/8bit/test_round_8bit.cu so the tests
// only check numbers: time_rounding_with_other_block_and_thread_counts (a matrix as it sits),
// time_turned_rounding_of_each_shape (turned matrices, as a linear layer's backward rounds output_grad and input),
// time_both_copies_of_each_output_grad (round_8bit_rows_and_columns against two calls). With the bytes each moves,
// their GB/s and MBU. Sizes: the 32 x 2,304 model's micro-batch of 8 rows of 2,048 tokens.
//
// Built as tests/8bit/test_round_8bit.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/round_8bit_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn, float8_e4m3fn; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"    // ASSERT_EQ
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../llmc/kernels/8bit/round_8bit.cuh"

#define GPT2_ROWS 16384                 // a micro-batch of 8 rows of 2048 tokens (the trainers' default)
#define GPT2_EMBED_DIM 2304             // the model's width
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define MEGA 1e6                        // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
/// on the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU, bfloat16 turned into
/// floats exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives a matrix's E4M3s turned back by PyTorch: each (read as float8_e4m3fn) times its row's scale, exact.
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scales_tensor, int64_t rows_val,
                                 int64_t row_length_val)
{
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, row_length_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
}

// ══ part 1: a matrix as it sits, other block and thread counts ════════════════

static void time_rounding_with_other_block_and_thread_counts(void)
{
    // Prints the time of round_8bit_kernel on x {16384, 2304}, for 5 block counts x 5 thread counts, each the average
    // of SWEEP_ROUNDS runs: as many as one warp per row needs, and this GPU's SM count x 1, 2, 4 and 8 (114 .. 912
    // on an H100 PCIe). The launches take turns: SWEEP_ROUNDS rounds, each timing every launch once. With the bytes
    // it moves (reads 2, writes 1 per value, and a scale per row), its GB/s and MBU.
    //
    // Expect: every launch gives the same E4M3s and scales as the usual one, to the last bit: each row is one warp's.

    // 1. GPT-2's x; the usual rounding
    torch::manual_seed(0);
    Tensor input_tensor = random_bf16_tensor({GPT2_ROWS, GPT2_EMBED_DIM});
    Tensor rounded_tensor = Tensor::zeros({GPT2_ROWS, GPT2_EMBED_DIM / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor scales_tensor = Tensor::zeros({GPT2_ROWS});
    round_8bit(rounded_tensor, scales_tensor, input_tensor);
    torch::Tensor usual_tensor = turned_back(rounded_tensor, scales_tensor, GPT2_ROWS, GPT2_EMBED_DIM);
    size_t rows_val = GPT2_ROWS;
    double moved_bytes_val = (double)GPT2_ROWS * GPT2_EMBED_DIM * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) +
                             (double)GPT2_ROWS * sizeof(float);

    // 2. the launch, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto round_val = [&] {
        round_8bit_kernel<<<blocks_val, threads_val>>>(reinterpret_cast<__nv_fp8_e4m3 *>(rounded_tensor.data_ptr()),
                                                       scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), rows_val,
                                                       GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    };
    threads_val = ROUND_8BIT_THREADS_PER_BLOCK;
    blocks_val = (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_val);
    warm_up_gpu(round_val);

    // 3. time every launch, taking turns over SWEEP_ROUNDS rounds; block count choice 0 is "as many as the work needs"
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int block_count_choices_val = 5;
    const int thread_count_choices_val = 5;
    int sm_multiples_list[block_count_choices_val] = {0, 1, 2, 4, 8};
    int thread_counts_list[thread_count_choices_val] = {2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                        16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    const int launches_val = block_count_choices_val * thread_count_choices_val;
    auto blocks_of_val = [&](int launch_index_val) {
        int multiple_val = sm_multiples_list[launch_index_val / thread_count_choices_val];
        int threads_of_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        return multiple_val > 0 ? multiple_val * sms_val : (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_of_val);
    };
    float total_ms_list[launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
        {
            threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
            blocks_val = blocks_of_val(launch_index_val);
            total_ms_list[launch_index_val] += gpu_milliseconds(round_val);
        }
    }

    // 4. print each launch's average with GB/s and MBU, and check its result against the usual one
    double peak_val = peak_gigabytes_per_second();
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB moved; this GPU's peak %.0f GB/s\n", moved_bytes_val / MEGA, peak_val);
    printf("    threads    blocks       ms     GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
    {
        threads_val = thread_counts_list[launch_index_val % thread_count_choices_val];
        blocks_val = blocks_of_val(launch_index_val);
        rounded_tensor.zero_();
        scales_tensor.zero_();
        round_val();
        bool same_val = torch::equal(turned_back(rounded_tensor, scales_tensor, GPT2_ROWS, GPT2_EMBED_DIM), usual_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        float milliseconds_val = total_ms_list[launch_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(moved_bytes_val, milliseconds_val);
        printf("    %7d   %7d   %6.3f   %6.0f   %3.0f%%   %s\n", threads_val, blocks_val, milliseconds_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO");
    }

    // 5. every launch gave the same result
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ══ part 2: turned matrices ═══════════════════════════════════════════════════

/// Prints the average of 20 turned roundings (steps 3..20) of a random {row_length, rows} matrix, both kernels
/// together, with the bytes moved (the scales' kernel reads the matrix once, the turning reads it again and writes
/// one byte a value), GB/s and MBU; then the same matrix as it sits ({rows, row_length}, round_8bit_kernel) for
/// comparison.
static void time_turned_rounding_of_one_shape(const char *name_ptr, size_t row_length_val, size_t rows_val)
{
    // 1. random values sitting turned, and as they sit
    torch::manual_seed(0);
    Tensor turned_input_tensor = random_bf16_tensor({row_length_val, rows_val});
    Tensor plain_input_tensor = random_bf16_tensor({rows_val, row_length_val});
    Tensor rounded_tensor = Tensor::zeros({rows_val, row_length_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor scales_tensor = Tensor::zeros({rows_val});
    // 2. time both
    double totals_list[2] = {0.0, 0.0};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float turned_ms_val = gpu_milliseconds([&] { round_8bit(rounded_tensor, scales_tensor, turned_input_tensor, true); });
        float plain_ms_val = gpu_milliseconds([&] { round_8bit(rounded_tensor, scales_tensor, plain_input_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        totals_list[0] += turned_ms_val;
        totals_list[1] += plain_ms_val;
    }
    // 3. print
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double values_val = (double)rows_val * row_length_val;
    double moved_list[2] = {values_val * (2 * sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + rows_val * sizeof(float),
                            values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + rows_val * sizeof(float)};
    const char *sides_list[2] = {"turned (tile kernels)", "as it sits"};
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    %s: rows %zu, row_length %zu\n", name_ptr, rows_val, row_length_val);
    printf("        %-20s       ms     MB     GB/s    MBU\n", "");
    for (int side_index_val = 0; side_index_val < 2; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        double speed_val = gigabytes_per_second(moved_list[side_index_val], milliseconds_val);
        printf("        %-20s %8.3f %6.1f %8.0f   %3.0f%%\n", sides_list[side_index_val], milliseconds_val,
               moved_list[side_index_val] / MEGA, speed_val, 100.0 * speed_val / peak_val);
    }
}

static void time_turned_rounding_of_each_shape(void)
{
    // Prints, for each matrix a layer's weight_grad rounds turned (output_grad {16384, out} and input {16384, in}:
    // rows out or in, row_length the 16384 rows), the turned rounding against rounding as it sits.
    time_turned_rounding_of_one_shape("qkv output_grad", GPT2_ROWS, GPT2_QKV_DIM);
    time_turned_rounding_of_one_shape("input x (2304 wide)", GPT2_ROWS, GPT2_EMBED_DIM);
    time_turned_rounding_of_one_shape("mlp up output_grad", GPT2_ROWS, GPT2_MLP_DIM);
}

// ══ part 3: both copies of one matrix ═════════════════════════════════════════

/// Prints the average of 20 roundings (steps 3..20) of a random output_grad {rows, columns} into both copies, the
/// row copy and the turned one: round_8bit_rows_and_columns (reads it twice) against round_8bit as it sits then
/// turned (three reads), with the bytes each moves, GB/s and MBU; then checks both ways give the same bits.
static void time_both_copies_of_one_shape(const char *name_ptr, size_t rows_val, size_t columns_val)
{
    // 1. random values; the copies
    torch::manual_seed(0);
    Tensor input_tensor = random_bf16_tensor({rows_val, columns_val});
    Tensor rows_8bit_tensor = Tensor::zeros({rows_val, columns_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor row_scales_tensor = Tensor::zeros({rows_val});
    Tensor columns_8bit_tensor = Tensor::zeros({columns_val, rows_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    Tensor column_scales_tensor = Tensor::zeros({columns_val});
    // 2. time both ways, taking turns
    double totals_list[2] = {0.0, 0.0};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float together_ms_val = gpu_milliseconds([&] {
            round_8bit_rows_and_columns(rows_8bit_tensor, row_scales_tensor, columns_8bit_tensor, column_scales_tensor, input_tensor);
        });
        float apart_ms_val = gpu_milliseconds([&] {
            round_8bit(rows_8bit_tensor, row_scales_tensor, input_tensor);
            round_8bit(columns_8bit_tensor, column_scales_tensor, input_tensor, true);
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        totals_list[0] += together_ms_val;
        totals_list[1] += apart_ms_val;
    }
    // 3. print: together reads 2 x 2 bytes a value, apart 3 x 2; both write 2 x 1
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double values_val = (double)rows_val * columns_val;
    double scale_bytes_val = (double)(rows_val + columns_val) * sizeof(float);
    double moved_list[2] = {values_val * (2 * sizeof(nv_bfloat16) + 2 * sizeof(__nv_fp8_e4m3)) + scale_bytes_val,
                            values_val * (3 * sizeof(nv_bfloat16) + 2 * sizeof(__nv_fp8_e4m3)) + scale_bytes_val};
    const char *sides_list[2] = {"rows_and_columns", "two round_8bit"};
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    %s {%zu, %zu}: both copies\n", name_ptr, rows_val, columns_val);
    printf("        %-20s       ms     MB     GB/s    MBU\n", "");
    for (int side_index_val = 0; side_index_val < 2; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        double speed_val = gigabytes_per_second(moved_list[side_index_val], milliseconds_val);
        printf("        %-20s %8.3f %6.1f %8.0f   %3.0f%%\n", sides_list[side_index_val], milliseconds_val,
               moved_list[side_index_val] / MEGA, speed_val, 100.0 * speed_val / peak_val);
    }
    // 4. the same bits both ways
    Tensor plain_8bit_tensor = Tensor::zeros(rows_8bit_tensor.shape(), DType::Int32);
    Tensor plain_scales_tensor = Tensor::zeros({rows_val});
    Tensor turned_8bit_tensor = Tensor::zeros(columns_8bit_tensor.shape(), DType::Int32);
    Tensor turned_scales_tensor = Tensor::zeros({columns_val});
    round_8bit(plain_8bit_tensor, plain_scales_tensor, input_tensor);
    round_8bit(turned_8bit_tensor, turned_scales_tensor, input_tensor, true);
    round_8bit_rows_and_columns(rows_8bit_tensor, row_scales_tensor, columns_8bit_tensor, column_scales_tensor, input_tensor);
    ASSERT_EQ(torch::equal(turned_back(rows_8bit_tensor, row_scales_tensor, (int64_t)rows_val, (int64_t)columns_val),
                           turned_back(plain_8bit_tensor, plain_scales_tensor, (int64_t)rows_val, (int64_t)columns_val)), true);
    ASSERT_EQ(torch::equal(turned_back(columns_8bit_tensor, column_scales_tensor, (int64_t)columns_val, (int64_t)rows_val),
                           turned_back(turned_8bit_tensor, turned_scales_tensor, (int64_t)columns_val, (int64_t)rows_val)), true);
}

static void time_both_copies_of_each_output_grad(void)
{
    // Prints, for each output_grad a layer's backward rounds both ways (its input_grad as it sits, its weight_grad
    // turned), round_8bit_rows_and_columns against two round_8bit calls: qkv's {16384, 6912}, mlp up's {16384, 9216},
    // attention output's and mlp down's {16384, 2304}.
    time_both_copies_of_one_shape("qkv output_grad", GPT2_ROWS, GPT2_QKV_DIM);
    time_both_copies_of_one_shape("mlp up output_grad", GPT2_ROWS, GPT2_MLP_DIM);
    time_both_copies_of_one_shape("2304-wide output_grad", GPT2_ROWS, GPT2_EMBED_DIM);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_rounding_with_other_block_and_thread_counts();
    time_turned_rounding_of_each_shape();
    time_both_copies_of_each_output_grad();

    printf("\nall tunes done\n");
    return 0;
}
