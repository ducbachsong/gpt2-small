// tune_grad_norm.cu — the timings of tests/test_grad_norm.cu, moved here so the tests only check numbers:
// time_20_norms_on_gpt2_small_size, time_20_norms_on_gpt2_small_148_tensors,
// time_the_kernel_with_other_block_and_thread_counts.
//
// Built as tests/test_grad_norm.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/grad_norm_<date>_<time>.txt.
#include <ATen/ops/_foreach_norm.h>   // at::_foreach_norm: what Python's clip_grad_norm_ uses
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed, for part 2's random values made on the GPU
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the gradients in one allocation, as the optimizer keeps them
#include "../llmc/kernels/grad_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the norm

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives a Shape as the sizes PyTorch takes: {2, 4} as int64_t.
static std::vector<int64_t> sizes_of(const Shape &shape_list)
{
    return std::vector<int64_t>(shape_list.begin(), shape_list.end());
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU: for
/// values too many to pass through a CPU list quickly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    return torch::from_blob(tensor.data(), sizes_of(tensor.shape()), on_gpu_val).clone();
}

// ══ part 1: my GradNorm timed against PyTorch's ═════════════════════════════

/// Times TIMED_STEPS norms of gradients of these shapes, mine against PyTorch's two ways, and prints a
/// table under `test_ptr`'s name. All three read the same values on the GPU:
///
///     my GradNorm          one launch over the flat buffer (every tensor one after the other)
///     PyTorch per tensor   what C++ clip_grad_norm_ does before it clips: grad.norm() of each
///                          tensor, then the norm of those norms (one launch per tensor, + 2)
///     PyTorch foreach      what Python's clip_grad_norm_ does: _foreach_norm over all the
///                          tensors at once (fewer, grouped launches), then the norm of the norms
///
/// The clip itself (multiplying every gradient) is not timed: it is not part of measuring.
/// Each norm reads every value once. Checks at the end that all three measured the same norm.
static void time_20_norms(const char *test_ptr, const std::vector<Shape> &param_shapes_list)
{
    // 1. one flat buffer of random gradients, and PyTorch's copy of each tensor
    Tensor::manual_seed(0);
    int num_params_val = (int)param_shapes_list.size();
    // [ grads | block sums: 2 x GRAD_NORM_BLOCKS | grad_norm: 1 ]
    TensorBuffer buffer_val;
    int grads_group_val = buffer_val.add_many(param_shapes_list);
    int block_sums_index_val = buffer_val.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index_val = buffer_val.add({1});
    buffer_val.allocate();
    const Tensor &grads_tensor = buffer_val.group(grads_group_val); // every gradient, one flat view
    grads_tensor.normal_(0.0f, 1.0f);                                // made on the CPU: a few seconds
    std::vector<torch::Tensor> pytorch_grads_list;
    for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
        pytorch_grads_list.push_back(pytorch_copy_of(buffer_val.tensor(grads_group_val, param_index_val)));
    GradNorm grad_norm_kernel_val(buffer_val.tensor(block_sums_index_val));
    const Tensor &grad_norm_tensor = buffer_val.tensor(grad_norm_index_val);
    torch::Tensor per_tensor_norm_tensor;
    torch::Tensor foreach_norm_tensor;

    // 2. time each step; step 1 does first-time work (CUDA loads each side's kernels), so the average is of
    //    the rest
    print_test_name(test_ptr);
    printf("    %d tensors, %zu values\n", num_params_val, grads_tensor.numel());
    printf("    time of one norm, in ms:   step   my GradNorm   PyTorch per tensor   PyTorch foreach\n");
    double my_total_val = 0.0;
    double per_tensor_total_val = 0.0;
    double foreach_total_val = 0.0;
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float my_ms_val = gpu_milliseconds([&] { grad_norm_kernel_val.compute(grads_tensor, grad_norm_tensor); });
        float per_tensor_ms_val = gpu_milliseconds([&] {
            std::vector<torch::Tensor> norms_list;
            for (const torch::Tensor &grad_tensor : pytorch_grads_list)
                norms_list.push_back(grad_tensor.norm());
            per_tensor_norm_tensor = torch::stack(norms_list).norm();
        });
        float foreach_ms_val = gpu_milliseconds([&] {
            std::vector<torch::Tensor> norms_list = at::_foreach_norm(pytorch_grads_list);
            foreach_norm_tensor = torch::stack(norms_list).norm();
        });
        printf("                               %4d   %11.3f   %18.3f   %15.3f\n", step_val, my_ms_val, per_tensor_ms_val,
               foreach_ms_val);
        if (step_val == 1)
            continue;
        my_total_val += my_ms_val;
        per_tensor_total_val += per_tensor_ms_val;
        foreach_total_val += foreach_ms_val;
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - 1;
    double bytes_val = (double)grads_tensor.numel() * sizeof(float);
    printf("    average of steps 2..%d:           %11.3f   %18.3f   %15.3f\n", TIMED_STEPS, my_total_val / averaged_val,
           per_tensor_total_val / averaged_val, foreach_total_val / averaged_val);
    printf("    memory read, in GB/s:             %11.0f   %18.0f   %15.0f\n", bytes_val / (my_total_val / averaged_val) / 1e6,
           bytes_val / (per_tensor_total_val / averaged_val) / 1e6, bytes_val / (foreach_total_val / averaged_val) / 1e6);
    printf("    PyTorch takes, against mine:                   %16.2fx   %14.2fx\n", per_tensor_total_val / my_total_val,
           foreach_total_val / my_total_val);

    // 4. all three measured the same norm
    double expected_val = per_tensor_norm_tensor.item<double>();
    ASSERT_NEAR(grad_norm_tensor.item(), expected_val, TOLERANCE * expected_val);
    ASSERT_NEAR(foreach_norm_tensor.item<double>(), expected_val, TOLERANCE * expected_val);
}

/// Makes GPT-2 small's 148 parameter shapes, in the model's order: wte (vocab 50,257) and wpe,
/// then the 12 shapes of each of the 12 layers, then the final LayerNorm. 124,439,808 values.
static std::vector<Shape> gpt2_small_param_shapes(void)
{
    std::vector<Shape> shapes_list = {{50257, 768}, {1024, 768}}; // wte, wpe
    for (int layer_val = 0; layer_val < 12; layer_val++)
    {
        std::vector<Shape> layer_shapes_list = {
            {768}, {768},         // ln1w, ln1b
            {2304, 768}, {2304},  // qkvw, qkvb
            {768, 768}, {768},    // attprojw, attprojb
            {768}, {768},         // ln2w, ln2b
            {3072, 768}, {3072},  // fcw, fcb
            {768, 3072}, {768},   // fcprojw, fcprojb
        };
        shapes_list.insert(shapes_list.end(), layer_shapes_list.begin(), layer_shapes_list.end());
    }
    shapes_list.push_back({768}); // lnfw
    shapes_list.push_back({768}); // lnfb
    return shapes_list;
}

static void time_20_norms_on_gpt2_small_size(void)
{
    // Prints the times on as many values as GPT-2 small has, as 2 tensors: 162,030 x 768 + 768 =
    // 124,439,808. Each norm reads every value once: 498 MB, so at a T4's 320 GB/s, at least
    // 1.56 ms. With 2 tensors PyTorch launches only a few kernels: this compares the kernels.
    time_20_norms(__func__, {{162030, 768}, {768}});
}

static void time_20_norms_on_gpt2_small_148_tensors(void)
{
    // Prints the times on GPT-2 small's real 148 tensors: the same 124,439,808 values, but
    // PyTorch now has 148 separate gradients (per tensor: 148 norm launches + 2), while mine is
    // still one launch over one flat buffer. What the flat buffer is worth shows here.
    // 1. the shapes
    std::vector<Shape> param_shapes_list = gpt2_small_param_shapes();
    ASSERT_EQ(param_shapes_list.size(), (size_t)148);
    // 2. time them
    time_20_norms(__func__, param_shapes_list);
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

static void time_the_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of one norm on as many values as GPT-2 small has, 124,439,808 (31,109,952 float4s), in
    // float and in bfloat16, for 10 block counts x 4 thread counts, each the average of 20 runs: the 8 picked
    // on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264 on an SXM; on
    // a T4 they repeat 40 and 80). The 40 launches take turns: 20 rounds, each timing every launch once (in float, then in bfloat16). One
    // launch's 20 runs in a row would all land in the same stretch of the test, and a T4's clock can drift
    // as it heats up (the first launch timed would get the fastest clock); taking turns gives every launch
    // runs from early, middle and late alike. Run 1 is counted too: warm_up_gpu has already done the
    // first-time work. The kernel is launched here directly, <<<blocks, threads>>>, with block sums for up
    // to 4096 blocks, instead of through GradNorm::compute, which always launches GRAD_NORM_BLOCKS x
    // GRAD_NORM_THREADS (512 x 1024). Each norm reads every value once: 498 MB in float, so at a T4's
    // 320 GB/s at least 1.56 ms; 249 MB in bfloat16, at least 0.78 ms. A bfloat16 float4 is one 8-byte read
    // instead of 16, so each thread keeps half the bytes in flight: its best launch may differ.
    //
    // Expect: every launch gives the usual norm of its dtype within TOLERANCE, not to the last bit: another
    // launch gives each thread other values to add up, so the squares are added in another order (in float
    // either way).

    // 1. random gradients made by PyTorch on the GPU, and the same in bfloat16
    torch::manual_seed(0);
    size_t n_val = 124439808;
    Tensor grads_tensor = Tensor::empty({n_val});
    torch::Tensor values_tensor = torch::randn({(int64_t)n_val}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(grads_tensor.data(), values_tensor.data_ptr<float>(), n_val * sizeof(float), cudaMemcpyDeviceToDevice));
    Tensor grads_bf16_tensor = grads_tensor.to(DType::BFloat16);
    // block sums for the largest launch, shared by both dtypes: the launches run one after the other, and
    // each puts the counter back to 0
    const int largest_blocks_val = 4096;
    Tensor block_sums_tensor = Tensor::zeros({(size_t)largest_blocks_val});
    Tensor blocks_done_tensor = Tensor::zeros({1});
    Tensor grad_norm_tensor = Tensor::zeros({1});
    Tensor grad_norm_bf16_tensor = Tensor::zeros({1});

    // 2. the usual launch's norms, to compare each launch with
    Tensor usual_block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: block sums; row 1, value 0: blocks done
    GradNorm(usual_block_sums_tensor).compute(grads_tensor, grad_norm_tensor);
    GradNorm(usual_block_sums_tensor).compute(grads_bf16_tensor, grad_norm_bf16_tensor);
    double usual_norm_val = grad_norm_tensor.item();
    double usual_norm_bf16_val = grad_norm_bf16_tensor.item();

    // 3. the two launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto norm_val = [&] {
        compute_grad_norm_kernel<float><<<blocks_val, threads_val>>>(grad_norm_tensor.data(), block_sums_tensor.data(),
                                                                     blocks_done_tensor.data(), grads_tensor.data(), n_val);
        cudaCheck(cudaGetLastError());
    };
    auto norm_bf16_val = [&] {
        compute_grad_norm_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            grad_norm_bf16_tensor.data(), block_sums_tensor.data(), blocks_done_tensor.data(),
            grads_bf16_tensor.data_ptr<nv_bfloat16>(), n_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = GRAD_NORM_BLOCKS;
    threads_val = GRAD_NORM_THREADS;
    warm_up_gpu(norm_val);

    // 4. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;             // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe; 2 x sms is within block_sums' 4096
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 80, 160, 320, 512, 1024, 2048, largest_blocks_val, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 10 * thread_count_choices_val;
    float total_ms_list[launches_val][2] = {}; // launch k is block_counts[k / 4] x thread_counts[k % 4]; [0] float, [1] bf16
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(norm_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(norm_bf16_val);
        }
    }

    // 5. print each launch's averages, and check its norms against the usual ones
    float fastest_ms_list[2] = {1e9f, 1e9f};
    int fastest_blocks_list[2] = {};
    int fastest_threads_list[2] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   float4s per thread    float    GB/s     bf16    GB/s   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[2];
        for (int dtype_index_val = 0; dtype_index_val < 2; dtype_index_val++)
        {
            times_list[dtype_index_val] = total_ms_list[launch_val][dtype_index_val] / SWEEP_ROUNDS;
            if (times_list[dtype_index_val] < fastest_ms_list[dtype_index_val])
            {
                fastest_ms_list[dtype_index_val] = times_list[dtype_index_val];
                fastest_blocks_list[dtype_index_val] = blocks_val;
                fastest_threads_list[dtype_index_val] = threads_val;
            }
        }

        // one norm with this launch in each dtype, against the usual launch's
        norm_val();
        norm_bf16_val();
        bool same_val = fabs(grad_norm_tensor.item() - usual_norm_val) <= TOLERANCE * usual_norm_val &&
                        fabs(grad_norm_bf16_tensor.item() - usual_norm_bf16_val) <= TOLERANCE * usual_norm_bf16_val;
        if (same_val)
            launches_with_the_same_result_val++;

        size_t threads_launched_val = (size_t)blocks_val * threads_val;
        int float4s_per_thread_val = (int)CEIL_DIV(n_val / VALUES_PER_FLOAT4, threads_launched_val); // the most
        double float_gb_per_second_val = (double)n_val * sizeof(float) / times_list[0] / 1e6;
        double bf16_gb_per_second_val = (double)n_val * sizeof(nv_bfloat16) / times_list[1] / 1e6;
        const char *usual_ptr = blocks_val == GRAD_NORM_BLOCKS && threads_val == GRAD_NORM_THREADS
                                    ? "   <- what GradNorm::compute launches"
                                    : "";
        printf("                  %6d   %7d   %18d   %6.3f   %5.0f   %6.3f   %5.0f   %-11s%s\n", blocks_val, threads_val,
               float4s_per_thread_val, times_list[0], float_gb_per_second_val, times_list[1], bf16_gb_per_second_val,
               same_val ? "yes" : "NO", usual_ptr);
    }
    printf("    fastest       %5d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[0], fastest_threads_list[0],
           fastest_ms_list[0]);
    printf("    fastest bf16  %5d blocks x %4d threads: %.3f ms\n", fastest_blocks_list[1], fastest_threads_list[1],
           fastest_ms_list[1]);

    // 6. every launch gave the same norms
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_norms_on_gpt2_small_size();
    time_20_norms_on_gpt2_small_148_tensors();
    time_the_kernel_with_other_block_and_thread_counts();

    printf("\nall tunes done\n");
    return 0;
}
