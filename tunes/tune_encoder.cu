// tune_encoder.cu — the timings of tests/test_encoder.cu, moved here so the tests only check numbers:
// time_20_steps_on_gpt2_small_size, time_20_steps_on_gpt2_small_size_in_tensor_buffer,
// time_each_kernel_with_other_block_and_thread_counts, time_wte_backward_with_float4_loads,
// time_forward_with_blocks_of_128_around_512.
//
// Built as tests/test_encoder.cu is (see tunes/README.md); what it prints also goes to
// tunes/results/<GPU>/encoder_<date>_<time>.txt.
#include <torch/types.h>              // torch::tensor, torch::embedding, backward(); libtorch before any CUDA header
#include <random>                     // std::mt19937: random token ids
#include "../tests/test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, print_gpu, MBU
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: wte and wpe in one allocation, as the trainer keeps them
#include "../llmc/kernels/encoder.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it. A repeated token's wte_grad row is rounded
// once per add, a few times here.
static const double BFLOAT16_TOLERANCE = 2e-2;

#define GPT2_HIGHEST_TOKEN_ID 50256   // GPT-2's real ids: 0 .. 50256

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes `count_val` random token ids from 0 .. highest_id_val, as a CPU list. The same seed gives
/// the same ids every run.
static std::vector<int> random_token_ids(size_t count_val, int highest_id_val)
{
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, highest_id_val);
    std::vector<int> ids_list(count_val);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    return ids_list;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU. Token ids
/// become int64, the type PyTorch's embedding is usually given.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    bool ints_val = tensor.dtype() == DType::Int32;
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(ints_val ? torch::kInt : torch::kFloat).device(torch::kCUDA);
    void *values_ptr = ints_val ? (void *)tensor.data_ptr<int>() : (void *)tensor.data();
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    torch::Tensor copy_tensor = torch::from_blob(values_ptr, sizes_list, on_gpu_val).clone();
    return ints_val ? copy_tensor.to(torch::kLong) : copy_tensor;
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Makes an int tensor of `shape_list` holding `ids_list` on the GPU: token ids.
static Tensor ids_of(const Shape &shape_list, const std::vector<int> &ids_list)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

// ══ part 1: timed against PyTorch's ══════════════════════════════════════════

/// Times TIMED_STEPS forwards and backwards of the encoder, mine against PyTorch's embedding with
/// autograd, and prints a table under `test_ptr`'s name. wte and wpe hold the weights, wte_grad and
/// wpe_grad start at 0; the batch is 4 rows of seq_len = wpe.size(0) random ids. Both sides add
/// every backward onto their gradients; at the end they must still agree.
static void time_20_steps(const char *test_ptr, const Tensor &wte_tensor, const Tensor &wpe_tensor,
                          const Tensor &wte_grad_tensor, const Tensor &wpe_grad_tensor)
{
    // 1. random ids and encoded_grad; mine and PyTorch's tensors
    size_t batch_size_val = 4;
    size_t seq_len_val = wpe_tensor.size(0);
    size_t embed_dim_val = wte_tensor.size(1);
    Tensor token_ids_tensor =
        ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_grad_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_wpe_tensor = pytorch_copy_of(wpe_tensor).requires_grad_(true);
    torch::Tensor pytorch_ids_tensor = pytorch_copy_of(token_ids_tensor);
    torch::Tensor pytorch_encoded_grad_tensor = pytorch_copy_of(encoded_grad_tensor);
    torch::Tensor pytorch_encoded_tensor;

    // 2. time each step; step 1 does first-time work (CUDA loads each side's kernels), so the average is of
    //    the rest
    print_test_name(test_ptr);
    printf("    time in ms:   step   my forward   my backward   PyTorch forward   PyTorch backward\n");
    double totals_list[4] = {0.0, 0.0, 0.0, 0.0};
    for (int step_val = 1; step_val <= TIMED_STEPS; step_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor); });
        times_list[1] = gpu_milliseconds([&] {
            encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
        });
        times_list[2] = gpu_milliseconds([&] {
            pytorch_encoded_tensor = torch::embedding(pytorch_wte_tensor, pytorch_ids_tensor) +
                                     pytorch_wpe_tensor.narrow(0, 0, seq_len_val);
        });
        times_list[3] = gpu_milliseconds([&] { pytorch_encoded_tensor.backward(pytorch_encoded_grad_tensor); });
        printf("                  %4d   %10.3f   %11.3f   %15.3f   %16.3f\n", step_val, times_list[0], times_list[1],
               times_list[2], times_list[3]);
        if (step_val == 1)
            continue;
        for (int column_val = 0; column_val < 4; column_val++)
            totals_list[column_val] += times_list[column_val];
    }

    // 3. print the averages
    const int averaged_val = TIMED_STEPS - 1;
    printf("    average of steps 2..%d:  %10.3f   %11.3f   %15.3f   %16.3f\n", TIMED_STEPS, totals_list[0] / averaged_val,
           totals_list[1] / averaged_val, totals_list[2] / averaged_val, totals_list[3] / averaged_val);
    printf("    PyTorch takes, against mine:                         %14.2fx   %15.2fx\n", totals_list[2] / totals_list[0],
           totals_list[3] / totals_list[1]);

    // 4. both sides agree at the end
    ASSERT_EQ(encoded_tensor.tolist(), list_of(pytorch_encoded_tensor));
    ASSERT_NEAR(wte_grad_tensor.tolist(), list_of(pytorch_wte_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(wpe_grad_tensor.tolist(), list_of(pytorch_wpe_tensor.grad()), TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times on GPT-2 small's encoder: wte {50304, 768}, wpe {1024, 768}, batch_size =
    // 4 rows of seq_len = 1024 ids, with wte, wpe and their gradients each a Tensor of its own. A
    // forward reads 4096 rows of wte and of wpe and writes 4096 rows of encoded: 3 x 12.6 MB =
    // 37.7 MB (wpe's 3 MB is read again for each of the 4 rows: the T4's 4 MB L2 cache most likely
    // does not keep it while wte's rows stream through), so at a T4's 320 GB/s at least 0.12 ms.
    // 1. GPT-2 small's weights and gradients, each a Tensor
    Tensor::manual_seed(0);
    Tensor wte_tensor = Tensor::randn({50304, 768});                   // made on the CPU: a few seconds
    Tensor wpe_tensor = Tensor::randn({1024, 768});
    Tensor wte_grad_tensor = Tensor::zeros({50304, 768});
    Tensor wpe_grad_tensor = Tensor::zeros({1024, 768});
    // 2. time them
    time_20_steps(__func__, wte_tensor, wpe_tensor, wte_grad_tensor, wpe_grad_tensor);
}

static void time_20_steps_on_gpt2_small_size_in_tensor_buffer(void)
{
    // Prints the times on the same sizes, with wte and wpe one after the other in one
    // TensorBuffer and their gradients in a second one laid out the same, as the trainer keeps
    // them. The kernels get the same pointers either way, so the times should match the test
    // above; a write past wte_grad's end would land in wpe_grad and fail its check.
    // 1. wte and wpe in one TensorBuffer, their gradients in another
    Tensor::manual_seed(0);
    TensorBuffer weights_val;
    int wte_index_val = weights_val.add({50304, 768});
    int wpe_index_val = weights_val.add({1024, 768});
    weights_val.allocate();
    weights_val.flat().normal_(0.0f, 1.0f);                            // both, made on the CPU: a few seconds
    TensorBuffer grads_val = TensorBuffer::zeros_like(weights_val);
    // 2. time them
    time_20_steps(__func__, weights_val.tensor(wte_index_val), weights_val.tensor(wpe_index_val),
                  grads_val.tensor(wte_index_val), grads_val.tensor(wpe_index_val));
}

// ══ part 2: other block and thread counts ═════════════════════════════════════

/// Runs `run_val` TIMED_STEPS times on the GPU and gives the average time of runs 2 on, in milliseconds;
/// run 1 does first-time work, so it is left out.
template <typename Run>
static float average_milliseconds(Run run_val)
{
    float total_val = 0.0f;
    for (int run_index_val = 1; run_index_val <= TIMED_STEPS; run_index_val++)
    {
        float milliseconds_val = gpu_milliseconds(run_val);
        if (run_index_val > 1)
            total_val += milliseconds_val;
    }
    return total_val / (TIMED_STEPS - 1);
}

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of each of the 3 kernels on GPT-2 small's sizes (batch_size = 4, seq_len =
    // 1024), in float and in bfloat16, for 10 block counts x 4 thread counts, each the average of 20 runs:
    // the 8 picked on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264
    // on an SXM; on a T4 they repeat 40 and 80). The 40 launches take turns: 20 rounds, each timing every launch once (the forward and the two
    // backwards in float, then the three in bfloat16). One launch's 20 runs in a row would all land in the
    // same stretch of the test, and a T4's clock can drift as it heats up (the first launch timed would get
    // the fastest clock); taking turns gives every launch runs from early, middle and late alike. Run 1 is
    // counted too: warm_up_gpu has already done the first-time work. The kernels are launched here
    // directly, <<<blocks, threads>>>, instead of through encoder_forward and encoder_backward, which
    // launch ENCODER_BLOCKS x ENCODER_FORWARD_THREADS (512 x 128) and ENCODER_BLOCKS x ENCODER_THREADS
    // (512 x 1024). A bfloat16 float4 is one 8-byte read instead of 16, so each thread keeps half the bytes
    // in flight: its best launch may differ.
    //
    // Expect: every launch gives the same result as the usual ones of its dtype (the grid-stride loop covers
    // every value whatever the count): encoded and wpe_grad to the last bit, wte_grad within TOLERANCE in
    // float and BFLOAT16_TOLERANCE in bfloat16 (atomicAdd's order; in bfloat16 each add onto a repeated
    // token's row is rounded to 8 bits). Only the time changes. A T4 runs at most 40 SMs x 1024 = 40,960
    // threads at once: with fewer, too few warps are left to run while the others wait on memory, so the
    // time should go up; with more, the extra blocks wait their turn and the time should stay about the same.

    // 1. GPT-2 small's sizes in float, and the same values in bfloat16, with results of their own
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t vocab_size_val = 50304;
    size_t embed_dim_val = 768;
    Tensor wte_tensor = Tensor::randn({vocab_size_val, embed_dim_val});    // made on the CPU: a few seconds
    Tensor wpe_tensor = Tensor::randn({seq_len_val, embed_dim_val});
    Tensor token_ids_tensor =
        ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_grad_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor wte_grad_tensor = Tensor::zeros({vocab_size_val, embed_dim_val});
    Tensor wpe_grad_tensor = Tensor::zeros({seq_len_val, embed_dim_val});
    Tensor wte_bf16_tensor = wte_tensor.to(DType::BFloat16);
    Tensor wpe_bf16_tensor = wpe_tensor.to(DType::BFloat16);
    Tensor encoded_grad_bf16_tensor = encoded_grad_tensor.to(DType::BFloat16);
    Tensor encoded_bf16_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
    Tensor wte_grad_bf16_tensor = Tensor::zeros({vocab_size_val, embed_dim_val}, DType::BFloat16);
    Tensor wpe_grad_bf16_tensor = Tensor::zeros({seq_len_val, embed_dim_val}, DType::BFloat16);

    // 2. the usual launch's results, kept on the GPU as PyTorch tensors to compare each launch with
    encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    encoder_forward(encoded_bf16_tensor, token_ids_tensor, wte_bf16_tensor, wpe_bf16_tensor);
    encoder_backward(wte_grad_bf16_tensor, wpe_grad_bf16_tensor, encoded_grad_bf16_tensor, token_ids_tensor);
    torch::Tensor usual_encoded_tensor = pytorch_copy_of(encoded_tensor);
    torch::Tensor usual_wte_grad_tensor = pytorch_copy_of(wte_grad_tensor);
    torch::Tensor usual_wpe_grad_tensor = pytorch_copy_of(wpe_grad_tensor);
    torch::Tensor usual_encoded_bf16_tensor = pytorch_copy_of(encoded_bf16_tensor.to(DType::Float32));
    torch::Tensor usual_wte_grad_bf16_tensor = pytorch_copy_of(wte_grad_bf16_tensor.to(DType::Float32));
    torch::Tensor usual_wpe_grad_bf16_tensor = pytorch_copy_of(wpe_grad_bf16_tensor.to(DType::Float32));

    // 3. the six launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        encoder_forward_kernel<float><<<blocks_val, threads_val>>>(encoded_tensor.data(), token_ids_tensor.data_ptr<int>(),
                                                                   wte_tensor.data(), wpe_tensor.data(), batch_size_val,
                                                                   seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wte_val = [&] {
        encoder_backward_wte_kernel<float><<<blocks_val, threads_val>>>(wte_grad_tensor.data(), encoded_grad_tensor.data(),
                                                                        token_ids_tensor.data_ptr<int>(), batch_size_val,
                                                                        seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wpe_val = [&] {
        encoder_backward_wpe_kernel<float><<<blocks_val, threads_val>>>(wpe_grad_tensor.data(), encoded_grad_tensor.data(),
                                                                        batch_size_val, seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16_val = [&] {
        encoder_forward_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            encoded_bf16_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(),
            wte_bf16_tensor.data_ptr<nv_bfloat16>(), wpe_bf16_tensor.data_ptr<nv_bfloat16>(), batch_size_val, seq_len_val,
            embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wte_bf16_val = [&] {
        encoder_backward_wte_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            wte_grad_bf16_tensor.data_ptr<nv_bfloat16>(), encoded_grad_bf16_tensor.data_ptr<nv_bfloat16>(),
            token_ids_tensor.data_ptr<int>(), batch_size_val, seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wpe_bf16_val = [&] {
        encoder_backward_wpe_kernel<nv_bfloat16><<<blocks_val, threads_val>>>(
            wpe_grad_bf16_tensor.data_ptr<nv_bfloat16>(), encoded_grad_bf16_tensor.data_ptr<nv_bfloat16>(), batch_size_val,
            seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = ENCODER_BLOCKS;
    threads_val = ENCODER_THREADS;
    warm_up_gpu(forward_val);

    // 4. time every launch, taking turns over SWEEP_ROUNDS rounds
    int sms_val = 0;                                         // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    int block_counts_list[] = {40, 80, 160, 320, 512, 1024, 2048, 4096, sms_val, 2 * sms_val};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    const int thread_count_choices_val = 4;
    const int launches_val = 10 * thread_count_choices_val;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]; one time per kernel, in kernel_names' order
    float total_ms_list[launches_val][6] = {};
    for (int round_val = 1; round_val <= SWEEP_ROUNDS; round_val++)
    {
        for (int launch_val = 0; launch_val < launches_val; launch_val++)
        {
            blocks_val = block_counts_list[launch_val / thread_count_choices_val];
            threads_val = thread_counts_list[launch_val % thread_count_choices_val];
            total_ms_list[launch_val][0] += gpu_milliseconds(forward_val);
            total_ms_list[launch_val][1] += gpu_milliseconds(backward_wte_val);
            total_ms_list[launch_val][2] += gpu_milliseconds(backward_wpe_val);
            total_ms_list[launch_val][3] += gpu_milliseconds(forward_bf16_val);
            total_ms_list[launch_val][4] += gpu_milliseconds(backward_wte_bf16_val);
            total_ms_list[launch_val][5] += gpu_milliseconds(backward_wpe_bf16_val);
        }
    }

    // 5. print each launch's averages, and check its results against the usual ones
    // the largest difference between a launch's wte_grad and the usual one, as a fraction of the usual one's
    // largest value
    auto difference_val = [](const torch::Tensor &actual_tensor, const torch::Tensor &usual_tensor) {
        return ((actual_tensor - usual_tensor).abs().max() / usual_tensor.abs().max()).item<float>();
    };
    const char *kernel_names_list[] = {"forward", "wte backward", "wpe backward", "bf16 forward", "bf16 wte backward",
                                       "bf16 wpe backward"};
    float fastest_ms_list[6] = {1e9f, 1e9f, 1e9f, 1e9f, 1e9f, 1e9f};
    int fastest_blocks_list[6] = {};
    int fastest_threads_list[6] = {};
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   all threads   forward   wte backward   wpe backward   "
           "bf16 forward   bf16 wte backward   bf16 wpe backward   same result\n");
    for (int launch_val = 0; launch_val < launches_val; launch_val++)
    {
        blocks_val = block_counts_list[launch_val / thread_count_choices_val];
        threads_val = thread_counts_list[launch_val % thread_count_choices_val];
        float times_list[6];
        for (int kernel_index_val = 0; kernel_index_val < 6; kernel_index_val++)
        {
            times_list[kernel_index_val] = total_ms_list[launch_val][kernel_index_val] / SWEEP_ROUNDS;
            if (times_list[kernel_index_val] < fastest_ms_list[kernel_index_val])
            {
                fastest_ms_list[kernel_index_val] = times_list[kernel_index_val];
                fastest_blocks_list[kernel_index_val] = blocks_val;
                fastest_threads_list[kernel_index_val] = threads_val;
            }
        }

        // one forward and one backward from 0 with this launch in each dtype, against the usual launch's:
        // encoded and wpe_grad to the last bit, wte_grad within TOLERANCE (BFLOAT16_TOLERANCE in bfloat16)
        for (const Tensor *tensor_ptr : {&encoded_tensor, &wte_grad_tensor, &wpe_grad_tensor, &encoded_bf16_tensor,
                                         &wte_grad_bf16_tensor, &wpe_grad_bf16_tensor})
            tensor_ptr->zero_();
        forward_val();
        backward_wte_val();
        backward_wpe_val();
        forward_bf16_val();
        backward_wte_bf16_val();
        backward_wpe_bf16_val();
        bool same_val = torch::equal(pytorch_copy_of(encoded_tensor), usual_encoded_tensor) &&
                        torch::equal(pytorch_copy_of(wpe_grad_tensor), usual_wpe_grad_tensor) &&
                        difference_val(pytorch_copy_of(wte_grad_tensor), usual_wte_grad_tensor) <= TOLERANCE &&
                        torch::equal(pytorch_copy_of(encoded_bf16_tensor.to(DType::Float32)), usual_encoded_bf16_tensor) &&
                        torch::equal(pytorch_copy_of(wpe_grad_bf16_tensor.to(DType::Float32)), usual_wpe_grad_bf16_tensor) &&
                        difference_val(pytorch_copy_of(wte_grad_bf16_tensor.to(DType::Float32)), usual_wte_grad_bf16_tensor) <=
                            BFLOAT16_TOLERANCE;
        if (same_val)
            launches_with_the_same_result_val++;

        const char *usual_ptr = blocks_val == ENCODER_BLOCKS && threads_val == ENCODER_FORWARD_THREADS
                                    ? "   <- what encoder_forward launches"
                                : blocks_val == ENCODER_BLOCKS && threads_val == ENCODER_THREADS
                                    ? "   <- what encoder_backward launches"
                                    : "";
        printf("                %6d   %7d   %11d   %7.3f   %12.3f   %12.3f   %12.3f   %17.3f   %17.3f   %-11s%s\n",
               blocks_val, threads_val, blocks_val * threads_val, times_list[0], times_list[1], times_list[2], times_list[3],
               times_list[4], times_list[5], same_val ? "yes" : "NO", usual_ptr);
    }
    for (int kernel_index_val = 0; kernel_index_val < 6; kernel_index_val++)
        printf("    fastest %-17s  %4d blocks x %4d threads: %.3f ms\n", kernel_names_list[kernel_index_val],
               fastest_blocks_list[kernel_index_val], fastest_threads_list[kernel_index_val],
               fastest_ms_list[kernel_index_val]);

    // 6. every launch gave the same results
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ══ part 3: experiments on what part 2 found ══════════════════════════════════

/// The wte backward as encoder.cuh had it before: one float of encoded_grad per load, then one
/// float atomicAdd, 3,145,728 jobs. Kept here for part 3, to compare with encoder.cuh's kernel,
/// which loads a float4 (16 bytes in flight per thread instead of 4) and then does four adds.
///
///   wte_grad_ptr, encoded_grad_ptr, token_ids_ptr   as in encoder_backward_wte_kernel, float
///   batch_size_val, seq_len_val, embed_dim_val      encoded_grad's shape
__global__ void encoder_backward_wte_float_kernel(float *wte_grad_ptr, const float *encoded_grad_ptr,
                                                  const int *token_ids_ptr, size_t batch_size_val, size_t seq_len_val,
                                                  size_t embed_dim_val)
{
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_values_val = batch_size_val * seq_len_val * embed_dim_val;
    // Thread n takes values n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_values_val; index_val += (size_t)gridDim.x * blockDim.x)
    {
        // 1. which token, which value of its row
        size_t token_index_val = index_val / embed_dim_val;
        size_t embed_index_val = index_val % embed_dim_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. add it onto the token's row
        atomicAdd(&wte_grad_ptr[token_id_val * embed_dim_val + embed_index_val], encoded_grad_ptr[index_val]);
    }
}

static void time_wte_backward_with_float4_loads(void)
{
    // Prints the wte backward with float loads (above, as encoder.cuh had it) and with float4
    // loads (encoder.cuh now), each launch size timed for both, one after the other, on GPT-2
    // small's sizes.
    //
    // Expect, if too few bytes in flight is why the wte backward slowed down so much with few
    // threads: with float4 loads it keeps 4x the bytes in flight per thread, so at 5,120 threads it
    // slows down far less than with float loads. Measured on a T4 in 5 runs: 2.18-2.21x with float
    // loads, 1.01-1.03x with float4 loads. Both must give the same wte_grad as the usual launch,
    // within TOLERANCE.

    // 1. GPT-2 small's sizes; the usual launch's result, to compare each launch with
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t vocab_size_val = 50304;
    size_t embed_dim_val = 768;
    Tensor token_ids_tensor =
        ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_grad_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});
    Tensor wte_grad_tensor = Tensor::zeros({vocab_size_val, embed_dim_val});
    encoder_backward_wte_kernel<float><<<ENCODER_BLOCKS, ENCODER_THREADS>>>(
        wte_grad_tensor.data(), encoded_grad_tensor.data(), token_ids_tensor.data_ptr<int>(), batch_size_val, seq_len_val,
        embed_dim_val);
    cudaCheck(cudaGetLastError());
    torch::Tensor usual_wte_grad_tensor = pytorch_copy_of(wte_grad_tensor);

    // 2. the two launches, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto float_loads_val = [&] {
        encoder_backward_wte_float_kernel<<<blocks_val, threads_val>>>(wte_grad_tensor.data(), encoded_grad_tensor.data(),
                                                                       token_ids_tensor.data_ptr<int>(), batch_size_val,
                                                                       seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    auto float4_loads_val = [&] {
        encoder_backward_wte_kernel<float><<<blocks_val, threads_val>>>(wte_grad_tensor.data(), encoded_grad_tensor.data(),
                                                                        token_ids_tensor.data_ptr<int>(), batch_size_val,
                                                                        seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    // tells if wte_grad, after one backward from 0 with `run_val`, equals the usual launch's
    auto same_as_usual_val = [&](auto run_val) {
        wte_grad_tensor.zero_();
        run_val();
        torch::Tensor difference_tensor =
            (pytorch_copy_of(wte_grad_tensor) - usual_wte_grad_tensor).abs().max() / usual_wte_grad_tensor.abs().max();
        return difference_tensor.item<float>() <= TOLERANCE;
    };
    blocks_val = ENCODER_BLOCKS;
    threads_val = ENCODER_THREADS;
    warm_up_gpu(float4_loads_val);

    // 3. time both at every launch, and check them
    int block_counts_list[] = {40, 80, 160, 320, 512, 1024, 2048, 4096};
    int thread_counts_list[] = {4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    float float_ms_list[8][4];
    float float4_ms_list[8][4];
    int launches_with_the_same_result_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   all threads   float loads   float4 loads   same result\n");
    for (int block_index_val = 0; block_index_val < 8; block_index_val++)
    {
        for (int thread_index_val = 0; thread_index_val < 4; thread_index_val++)
        {
            blocks_val = block_counts_list[block_index_val];
            threads_val = thread_counts_list[thread_index_val];
            float_ms_list[block_index_val][thread_index_val] = average_milliseconds(float_loads_val);
            float4_ms_list[block_index_val][thread_index_val] = average_milliseconds(float4_loads_val);
            bool same_val = same_as_usual_val(float_loads_val) && same_as_usual_val(float4_loads_val);
            if (same_val)
                launches_with_the_same_result_val++;
            printf("                %6d   %7d   %11d   %11.3f   %12.3f   %s\n", blocks_val, threads_val,
                   blocks_val * threads_val, float_ms_list[block_index_val][thread_index_val],
                   float4_ms_list[block_index_val][thread_index_val], same_val ? "yes" : "NO");
        }
    }

    // 4. print the slow-downs: 40 x 128 is [0][0]; 512 x 1024 is [4][3]
    printf("    slow-down at 5,120 threads against 512 x 1024:   float loads %.2fx   float4 loads %.2fx\n",
           float_ms_list[0][0] / float_ms_list[4][3], float4_ms_list[0][0] / float4_ms_list[4][3]);
    printf("    slow-down at 10,240 threads (40 x 256):           float loads %.2fx   float4 loads %.2fx\n",
           float_ms_list[0][1] / float_ms_list[4][3], float4_ms_list[0][1] / float4_ms_list[4][3]);

    // 5. every launch gave the same result
    ASSERT_EQ(launches_with_the_same_result_val, 8 * 4);
}

static void time_forward_with_blocks_of_128_around_512(void)
{
    // Prints the forward with blocks of 128 threads and 256 to 1,024 blocks in steps of 64, each
    // timed right after 512 x 1024 so the two are compared in the same moment. In part 2's 6 runs,
    // 512 x 128 was the fastest forward every time, 4-6% ahead of 512 x 1024.
    //
    // Expect one of two shapes: a smooth dip around 512 blocks (a real effect of the launch size)
    // or one low point at 512 alone (something special about that one launch). The jobs per thread
    // are printed too: 512 x 128 gives every thread exactly 12 of the 786,432 jobs, but part 2's
    // launches with exactly 3 each were not faster, so an even share alone is not expected to matter.

    // 1. GPT-2 small's sizes; the usual launch's result
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t vocab_size_val = 50304;
    size_t embed_dim_val = 768;
    Tensor wte_tensor = Tensor::randn({vocab_size_val, embed_dim_val}); // made on the CPU: a few seconds
    Tensor wpe_tensor = Tensor::randn({seq_len_val, embed_dim_val});
    Tensor token_ids_tensor =
        ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    torch::Tensor usual_encoded_tensor = pytorch_copy_of(encoded_tensor);

    // 2. the forward, at the launch the loop sets
    int blocks_val = 0;
    int threads_val = 0;
    auto forward_val = [&] {
        encoder_forward_kernel<float><<<blocks_val, threads_val>>>(encoded_tensor.data(), token_ids_tensor.data_ptr<int>(),
                                                                   wte_tensor.data(), wpe_tensor.data(), batch_size_val,
                                                                   seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
    };
    blocks_val = ENCODER_BLOCKS;
    threads_val = ENCODER_THREADS;
    warm_up_gpu(forward_val);

    // 3. each block count, timed right after 512 x 1024, and checked
    const int small_block_val = 4 * THREADS_PER_WARP;        // 128
    int launches_with_the_same_result_val = 0;
    int launches_val = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   jobs per thread   forward   512 x 1024 just before   ratio\n");
    for (int block_count_val = 256; block_count_val <= 1024; block_count_val += 64)
    {
        blocks_val = ENCODER_BLOCKS;
        threads_val = ENCODER_THREADS;
        float usual_ms_val = average_milliseconds(forward_val);
        blocks_val = block_count_val;
        threads_val = small_block_val;
        float ms_val = average_milliseconds(forward_val);

        encoded_tensor.zero_();
        forward_val();
        bool same_val = torch::equal(pytorch_copy_of(encoded_tensor), usual_encoded_tensor);
        if (same_val)
            launches_with_the_same_result_val++;
        launches_val++;
        double jobs_per_thread_val =
            (double)(batch_size_val * seq_len_val * embed_dim_val / VALUES_PER_FLOAT4) / (blocks_val * threads_val);
        printf("                %6d   %7d   %15.2f   %7.3f   %22.3f   %5.3f%s\n", blocks_val, threads_val,
               jobs_per_thread_val, ms_val, usual_ms_val, ms_val / usual_ms_val, same_val ? "" : "   NOT THE SAME RESULT");
    }

    // 4. every launch gave the same result
    ASSERT_EQ(launches_with_the_same_result_val, launches_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tunes/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_steps_on_gpt2_small_size();
    time_20_steps_on_gpt2_small_size_in_tensor_buffer();
    time_each_kernel_with_other_block_and_thread_counts();
    time_wte_backward_with_float4_loads();
    time_forward_with_blocks_of_128_around_512();

    printf("\nall tunes done\n");
    return 0;
}
