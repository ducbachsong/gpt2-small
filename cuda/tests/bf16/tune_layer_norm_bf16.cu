// tune_layer_norm_bf16.cu — the timings of cuda/kernels/bf16/layer_norm_bf16.cuh, apart from
// cuda/tests/bf16/test_layer_norm_bf16.cu so the tests only check numbers: time_8_values_against_4_values (one 16-byte
// load a lane against fp32/layer_norm.cuh's 8-byte one), time_each_kernel_with_other_block_and_thread_counts. With
// the bytes each kernel moves, its GB/s and MBU.
//
// Built as cuda/tests/bf16/test_layer_norm_bf16.cu is (see cuda/tests/README.md); what it prints also goes to
// cuda/tests/results/<GPU>/layer_norm_bf16_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"    // ASSERT_EQ
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../kernels/fp32/layer_norm.cuh"      // the 4-value path
#include "../../kernels/bf16/layer_norm_bf16.cuh"

// Against the 4-value path: a row's sums added in another order, so a value can land one bfloat16 rounding away;
// weight_grad and bias_grad between launches: rounded once per block, in whatever order the blocks finish. The
// largest difference, as a fraction of the largest value.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
static const double BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE = 5e-2;

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_ROWS (GPT2_BATCH_SIZE * GPT2_SEQ_LEN)
#define GPT2_EMBED_DIM 768
#define KERNELS 2                     // forward, backward
#define SIDES 2                       // 8 values a lane, 4 values a lane
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values (bfloat16 turned into floats exactly).
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between two of my tensors, as a fraction of the second one's largest value.
static double relative_difference_on_gpu(const Tensor &actual_tensor, const Tensor &usual_tensor)
{
    torch::Tensor usual_values_tensor = pytorch_copy_of(usual_tensor);
    return ((pytorch_copy_of(actual_tensor) - usual_values_tensor).abs().max() / usual_values_tensor.abs().max()).item<double>();
}

/// Gives the bytes each kernel asks for, in kernel order, on GPT-2 small's sizes.
///
///   bytes_list   written
static void kernel_bytes(double (&bytes_list)[KERNELS])
{
    // Gives the bytes each kernel asks for, in kernel order (forward, backward), on GPT-2 small's sizes; a row's values
    // read again in a later pass (from the cache) and weight and bias (3 KB) are left out.
    //
    //     forward    reads input; writes output, mean, rstd                                  12.6 MB
    //     backward   reads input, output_grad, input_grad, mean, rstd; writes input_grad     25.2 MB
    double matrix_bytes_val = (double)GPT2_ROWS * GPT2_EMBED_DIM * sizeof(nv_bfloat16);
    double row_floats_bytes_val = 2.0 * GPT2_ROWS * sizeof(float);
    bytes_list[0] = 2 * matrix_bytes_val + row_floats_bytes_val;
    bytes_list[1] = 4 * matrix_bytes_val + row_floats_bytes_val;
}

/// GPT-2 small's LayerNorm in bfloat16, and one set of results per side.
struct Gpt2LayerNorm
{
    // GPT-2 small's LayerNorm in bfloat16 (4096 random rows of 768), and one set of results per side (8 values a lane, 4
    // values a lane).
    Tensor input_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor weight_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor bias_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor output_grad_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor output_list[SIDES], mean_list[SIDES], rstd_list[SIDES];
    Tensor input_grad_list[SIDES], weight_grad_list[SIDES], bias_grad_list[SIDES];

    Gpt2LayerNorm()
    {
        for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        {
            output_list[side_index_val] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
            mean_list[side_index_val] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
            rstd_list[side_index_val] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
            input_grad_list[side_index_val] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
            weight_grad_list[side_index_val] = Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16);
            bias_grad_list[side_index_val] = Tensor::zeros({GPT2_EMBED_DIM}, DType::BFloat16);
        }
    }

    /// Launches the forward of one side, one warp a row.
    ///
    ///   side_index_val   the side: 8 values or 4 values a lane
    ///   threads_val      the block size
    void forward(int side_index_val, int threads_val) const
    {
        // Launches the forward of side `side_index_val` (8 values, 4 values) with `threads_val` threads a block, one warp
        // a row, the blocks rounded up.
        int blocks_val = (int)CEIL_DIV((size_t)GPT2_ROWS, (size_t)(threads_val / HARDWARE_THREADS_PER_WARP));
        nv_bfloat16 *output_ptr = output_list[side_index_val].data_ptr<nv_bfloat16>();
        float *mean_ptr = mean_list[side_index_val].data();
        float *rstd_ptr = rstd_list[side_index_val].data();
        if (side_index_val == 0)
            layer_norm_bf16_forward_kernel<<<blocks_val, threads_val>>>(
                output_ptr, mean_ptr, rstd_ptr, input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(),
                bias_tensor.data_ptr<nv_bfloat16>(), GPT2_ROWS, GPT2_EMBED_DIM);
        else
            layer_norm_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
                output_ptr, mean_ptr, rstd_ptr, input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(),
                bias_tensor.data_ptr<nv_bfloat16>(), GPT2_ROWS, GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    }

    /// Launches the backward of side `side_index_val` as <<<blocks_val, threads_val>>>, from that side's mean and rstd.
    void backward(int side_index_val, int blocks_val, int threads_val) const
    {
        size_t shared_bytes_val = 2 * GPT2_EMBED_DIM * sizeof(float);
        if (side_index_val == 0)
            layer_norm_bf16_backward_launch(input_grad_list[0], weight_grad_list[0], bias_grad_list[0], output_grad_tensor,
                                            input_tensor, weight_tensor, mean_list[0], rstd_list[0], blocks_val, threads_val);
        else
            layer_norm_backward_kernel<nv_bfloat16><<<blocks_val, threads_val, shared_bytes_val>>>(
                input_grad_list[1].data_ptr<nv_bfloat16>(), weight_grad_list[1].data_ptr<nv_bfloat16>(),
                bias_grad_list[1].data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
                input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(), mean_list[1].data(),
                rstd_list[1].data(), GPT2_ROWS, GPT2_EMBED_DIM);
        cudaCheck(cudaGetLastError());
    }

    /// Zeroes side `side_index_val`'s gradients and runs its backward once at <<<blocks_val, threads_val>>>.
    void backward_from_0(int side_index_val, int blocks_val, int threads_val) const
    {
        input_grad_list[side_index_val].zero_();
        weight_grad_list[side_index_val].zero_();
        bias_grad_list[side_index_val].zero_();
        backward(side_index_val, blocks_val, threads_val);
    }
};

// ══ part 1: 8 values a lane against 4 ═════════════════════════════════════════

static void time_8_values_against_4_values(void)
{
    // Prints the forward and the backward on GPT-2 small's sizes in bfloat16 (4096 rows of 768), here (8 values a
    // lane, one 16-byte load) and in fp32/layer_norm.cuh (4 values, one 8-byte load), each at its default launch
    // (LAYER_NORM_BF16_THREADS_PER_BLOCK for the forward, LAYER_NORM_BF16_BACKWARD_BLOCKS x
    // LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK for the backward: fp32/layer_norm.cuh's numbers). The 4 take turns
    // over SWEEP_ROUNDS rounds, each the average of its rounds, with the bytes it asks for, GB/s and MBU.
    //
    // Expect: output and input_grad within ONE_ROUNDING_TOLERANCE of each other (a row's sums in another order); the
    // 8-value kernels as fast or faster: the same bytes in half the loads. The backward's adds into shared memory are
    // one a value on both sides, so it may gain less.

    // 1. GPT-2 small's LayerNorm; one forward and backward from 0 each side, checked
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
    {
        layer_norm_val.forward(side_index_val, LAYER_NORM_BF16_THREADS_PER_BLOCK);
        layer_norm_val.backward_from_0(side_index_val, LAYER_NORM_BF16_BACKWARD_BLOCKS, LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK);
    }
    double output_difference_val = relative_difference_on_gpu(layer_norm_val.output_list[0], layer_norm_val.output_list[1]);
    double input_grad_difference_val = relative_difference_on_gpu(layer_norm_val.input_grad_list[0], layer_norm_val.input_grad_list[1]);
    warm_up_gpu([&] { layer_norm_val.forward(0, LAYER_NORM_BF16_THREADS_PER_BLOCK); });

    // 2. time the 4, taking turns over SWEEP_ROUNDS rounds
    float total_ms_list[SIDES][KERNELS] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int side_index_val = 0; side_index_val < SIDES; side_index_val++)
        {
            total_ms_list[side_index_val][0] +=
                gpu_milliseconds([&] { layer_norm_val.forward(side_index_val, LAYER_NORM_BF16_THREADS_PER_BLOCK); });
            total_ms_list[side_index_val][1] += gpu_milliseconds([&] {
                layer_norm_val.backward(side_index_val, LAYER_NORM_BF16_BACKWARD_BLOCKS, LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK);
            });
        }

    // 3. print each with its GB/s and MBU
    double bytes_list[KERNELS];
    kernel_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    const char *kernel_names_list[KERNELS] = {"forward", "backward"};
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s; largest difference from the 4-value path: output %.2e, input_grad %.2e\n", peak_val,
           output_difference_val, input_grad_difference_val);
    printf("    kernel        MB      8 values: ms    GB/s   MBU     4 values: ms    GB/s   MBU     4 / 8\n");
    for (int kernel_index_val = 0; kernel_index_val < KERNELS; kernel_index_val++)
    {
        float eight_ms_val = total_ms_list[0][kernel_index_val] / SWEEP_ROUNDS;
        float four_ms_val = total_ms_list[1][kernel_index_val] / SWEEP_ROUNDS;
        double eight_speed_val = gigabytes_per_second(bytes_list[kernel_index_val], eight_ms_val);
        double four_speed_val = gigabytes_per_second(bytes_list[kernel_index_val], four_ms_val);
        printf("    %-10s  %5.1f       %7.4f  %6.0f  %3.0f%%         %7.4f  %6.0f  %3.0f%%     %5.2fx\n",
               kernel_names_list[kernel_index_val], bytes_list[kernel_index_val] / MEGA, eight_ms_val, eight_speed_val,
               100.0 * eight_speed_val / peak_val, four_ms_val, four_speed_val, 100.0 * four_speed_val / peak_val,
               four_ms_val / eight_ms_val);
    }

    // 4. the two sides agree
    ASSERT_EQ(output_difference_val <= ONE_ROUNDING_TOLERANCE && input_grad_difference_val <= ONE_ROUNDING_TOLERANCE, true);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints, on GPT-2 small's sizes, each the average of SWEEP_ROUNDS runs taking turns, with GB/s and MBU:
    //     the forward with 1, 2, 4, 8 and 16 warps a block (one warp a row, the blocks rounded up: 4096 .. 256)
    //     the backward with 40, 80, 160 blocks and this GPU's SM count x 1 and x 2, x 4, 8 and 16 warps a block
    // The defaults are fp32/layer_norm.cuh's; with 8 values a lane each warp has half the loads, so the best launch may
    // be another.
    //
    // Expect: every forward gives the default's output, mean and rstd to the last bit (each row one warp's), every
    // backward its input_grad to the last bit and weight_grad and bias_grad within BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE.

    // 1. GPT-2 small's LayerNorm; the defaults' results
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    layer_norm_val.forward(0, LAYER_NORM_BF16_THREADS_PER_BLOCK);
    layer_norm_val.backward_from_0(0, LAYER_NORM_BF16_BACKWARD_BLOCKS, LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK);
    Tensor usual_output_tensor = layer_norm_val.output_list[0].clone();
    Tensor usual_input_grad_tensor = layer_norm_val.input_grad_list[0].clone();
    Tensor usual_weight_grad_tensor = layer_norm_val.weight_grad_list[0].clone();
    Tensor usual_bias_grad_tensor = layer_norm_val.bias_grad_list[0].clone();
    warm_up_gpu([&] { layer_norm_val.forward(0, LAYER_NORM_BF16_THREADS_PER_BLOCK); });

    // 2. the launches; time every one, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    const int forward_launches_val = 5;                      // not 1024 threads: fp32/layer_norm.cuh's kernels allow at most 896
    int forward_threads_list[forward_launches_val] = {HARDWARE_THREADS_PER_WARP, 2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP,
                                                      8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP};
    const int block_count_choices_val = 6;
    const int thread_count_choices_val = 3;
    const int backward_launches_val = block_count_choices_val * thread_count_choices_val;
    int backward_blocks_list[block_count_choices_val] = {40, 80, 160, 256, sms_val, 2 * sms_val};
    int backward_threads_list[thread_count_choices_val] = {4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                                           16 * HARDWARE_THREADS_PER_WARP};
    float forward_ms_list[forward_launches_val] = {};
    float backward_ms_list[backward_launches_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        for (int launch_index_val = 0; launch_index_val < forward_launches_val; launch_index_val++)
            forward_ms_list[launch_index_val] +=
                gpu_milliseconds([&] { layer_norm_val.forward(0, forward_threads_list[launch_index_val]); });
        for (int launch_index_val = 0; launch_index_val < backward_launches_val; launch_index_val++)
            backward_ms_list[launch_index_val] += gpu_milliseconds([&] {
                layer_norm_val.backward(0, backward_blocks_list[launch_index_val / thread_count_choices_val],
                                        backward_threads_list[launch_index_val % thread_count_choices_val]);
            });
    }

    // 3. print each launch with GB/s and MBU, and check its results against the defaults'
    double bytes_list[KERNELS];
    kernel_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s\n    forward:   threads   blocks        ms     GB/s    MBU   same result\n", peak_val);
    for (int launch_index_val = 0; launch_index_val < forward_launches_val; launch_index_val++)
    {
        int threads_val = forward_threads_list[launch_index_val];
        layer_norm_val.output_list[0].zero_();
        layer_norm_val.forward(0, threads_val);
        bool same_val = torch::equal(pytorch_copy_of(layer_norm_val.output_list[0]), pytorch_copy_of(usual_output_tensor));
        if (same_val)
            launches_with_the_same_result_val++;
        float ms_val = forward_ms_list[launch_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_list[0], ms_val);
        printf("               %7d   %6d   %7.4f   %6.0f   %3.0f%%   %-11s%s\n", threads_val,
               (int)CEIL_DIV((size_t)GPT2_ROWS, (size_t)(threads_val / HARDWARE_THREADS_PER_WARP)), ms_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO",
               threads_val == LAYER_NORM_BF16_THREADS_PER_BLOCK ? "   <- the default" : "");
    }
    printf("    backward:  blocks   threads        ms     GB/s    MBU   same result\n");
    for (int launch_index_val = 0; launch_index_val < backward_launches_val; launch_index_val++)
    {
        int blocks_val = backward_blocks_list[launch_index_val / thread_count_choices_val];
        int threads_val = backward_threads_list[launch_index_val % thread_count_choices_val];
        layer_norm_val.backward_from_0(0, blocks_val, threads_val);
        bool same_val =
            torch::equal(pytorch_copy_of(layer_norm_val.input_grad_list[0]), pytorch_copy_of(usual_input_grad_tensor)) &&
            relative_difference_on_gpu(layer_norm_val.weight_grad_list[0], usual_weight_grad_tensor) <= BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE &&
            relative_difference_on_gpu(layer_norm_val.bias_grad_list[0], usual_bias_grad_tensor) <= BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE;
        if (same_val)
            launches_with_the_same_result_val++;
        float ms_val = backward_ms_list[launch_index_val] / SWEEP_ROUNDS;
        double speed_val = gigabytes_per_second(bytes_list[1], ms_val);
        printf("               %6d   %7d   %7.4f   %6.0f   %3.0f%%   %-11s%s\n", blocks_val, threads_val, ms_val, speed_val,
               100.0 * speed_val / peak_val, same_val ? "yes" : "NO",
               blocks_val == LAYER_NORM_BF16_BACKWARD_BLOCKS && threads_val == LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK
                   ? "   <- the default" : "");
    }

    // 4. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, forward_launches_val + backward_launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and cuda/tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8_values_against_4_values();
    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
