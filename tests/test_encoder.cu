// test_encoder.cu — tests for llmc/kernels/encoder.cuh (encoder_forward, encoder_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's embedding, with autograd for the backward; part 2
// checks numbers worked out by hand; part 3 checks that each kernel's registers fit a block
// of ENCODER_THREADS; part 4 times each kernel with other block and thread counts; part 5 tests
// what part 4 found (a wte backward with float4 loads, a finer sweep of the forward). (The C#
// trainer had no encoder test: TorchSharp's embedding was trusted.)
//
// The tests use a Tensor for each weight, the simplest to write. The timing runs twice: with a
// Tensor for each, and with wte and wpe in one TensorBuffer, as the trainer keeps them.
#include <torch/types.h>              // torch::tensor, torch::embedding, backward(); libtorch before any CUDA header
#include <random>                     // std::mt19937: random token ids
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: wte and wpe in one allocation, as the trainer keeps them
#include "../llmc/kernels/encoder.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape)
{
    return Tensor::randn(shape).tolist();
}

/// Makes `count` random token ids from 0 .. highest_id, as a CPU list. The same seed gives
/// the same ids every run.
static std::vector<int> random_token_ids(size_t count, int highest_id)
{
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> pick(0, highest_id);
    std::vector<int> ids(count);
    for (int &id : ids)
        id = pick(generator);
    return ids;
}

/// Makes my tensor on the GPU from values in row order: floats, or ints for token ids.
template <typename T>
static Tensor my_tensor_of(const std::vector<T> &row_values, const Shape &shape)
{
    Tensor tensor = Tensor::empty(shape, std::is_same<T, int>::value ? DType::Int32 : DType::Float32);
    tensor.copy_from_cpu(row_values.data());
    return tensor;
}

/// Gives a Shape as the sizes PyTorch takes: {2, 4} as int64_t.
static std::vector<int64_t> sizes_of(const Shape &shape)
{
    return std::vector<int64_t>(shape.begin(), shape.end());
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU. Token ids
/// become int64, the type PyTorch's embedding is usually given.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    bool ints = tensor.dtype() == DType::Int32;
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(ints ? torch::kInt : torch::kFloat).device(torch::kCUDA);
    void *values = ints ? (void *)tensor.data_ptr<int>() : (void *)tensor.data();
    torch::Tensor copy = torch::from_blob(values, sizes_of(tensor.shape()), on_gpu).clone();
    return ints ? copy.to(torch::kLong) : copy;
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu.data_ptr<float>(), on_cpu.data_ptr<float>() + on_cpu.numel());
}

/// Measures how long `run` takes on the GPU, in milliseconds: a mark on the GPU's queue
/// before it and one after; the GPU writes down the time as it reaches each.
template <typename Run>
static float gpu_milliseconds(Run run)
{
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    run();
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end)); // wait until the GPU has passed the second mark
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    return milliseconds;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

static void forward_matches_pytorch(void)
{
    // Expect: a tiny GPT-2's encoder (vocab 64, positions 16, 16 wide), batch_size = 4 rows of
    // seq_len = 8 random ids: encoded equals PyTorch's embedding(wte, ids) + wpe[0 .. 7], exactly.
    // Each value is one float add of the same two numbers on both sides, so it rounds the same.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 8;
    Tensor wte = my_tensor_of(random_values({64, 16}), {64, 16});
    Tensor wpe = my_tensor_of(random_values({16, 16}), {16, 16});
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 63), {batch_size, seq_len});
    Tensor encoded = Tensor::zeros({batch_size, seq_len, 16});
    encoder_forward(encoded, token_ids, wte, wpe);

    torch::Tensor pytorch_encoded = torch::embedding(pytorch_copy_of(wte), pytorch_copy_of(token_ids)) +
                                pytorch_copy_of(wpe).narrow(0, 0, seq_len);
    ASSERT_EQ(encoded.tolist(), list_of(pytorch_encoded));
}

static void backward_matches_pytorch(void)
{
    // Expect: the same tiny encoder, but the ids only from 0 .. 15, so most of them come up 2 or
    // more times in the 32 and their wte_grad rows are sums. Random encoded_grad, backward from 0:
    // wte_grad and wpe_grad equal PyTorch's .grad after autograd's backward, within 1e-5 (the
    // two add a repeated token's rows in a different order). wte_grad's rows 16 .. 63 (ids never
    // read) stay 0, and so do wpe_grad's rows 8 .. 15 (positions after seq_len).
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 8;
    Tensor wte = my_tensor_of(random_values({64, 16}), {64, 16});
    Tensor wpe = my_tensor_of(random_values({16, 16}), {16, 16});
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 15), {batch_size, seq_len});
    Tensor encoded_grad = my_tensor_of(random_values({batch_size, seq_len, 16}), {batch_size, seq_len, 16});
    Tensor wte_grad = Tensor::zeros({64, 16}), wpe_grad = Tensor::zeros({16, 16});
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);

    torch::Tensor pytorch_wte = pytorch_copy_of(wte).requires_grad_(true);
    torch::Tensor pytorch_wpe = pytorch_copy_of(wpe).requires_grad_(true);
    torch::Tensor pytorch_encoded =
        torch::embedding(pytorch_wte, pytorch_copy_of(token_ids)) + pytorch_wpe.narrow(0, 0, seq_len);
    pytorch_encoded.backward(pytorch_copy_of(encoded_grad));
    ASSERT_NEAR(wte_grad.tolist(), list_of(pytorch_wte.grad()), TOLERANCE);
    ASSERT_NEAR(wpe_grad.tolist(), list_of(pytorch_wpe.grad()), TOLERANCE);
    ASSERT_EQ(wte_grad.narrow(0, 16, 48).tolist(), std::vector<float>(48 * 16, 0.0f));
    ASSERT_EQ(wpe_grad.narrow(0, 8, 8).tolist(), std::vector<float>(8 * 16, 0.0f));
}

/// Times 20 forwards and 20 backwards of the encoder, mine against PyTorch's embedding with
/// autograd, and prints a table under `test`'s name. wte and wpe hold the weights, wte_grad and
/// wpe_grad start at 0; the batch is 4 rows of seq_len = wpe.size(0) random ids. Both sides add
/// 20 backwards onto their gradients; at the end they must still agree.
static void time_20_steps(const char *test, const Tensor &wte, const Tensor &wpe, const Tensor &wte_grad,
                          const Tensor &wpe_grad)
{
    size_t batch_size = 4, seq_len = wpe.size(0), embed_dim = wte.size(1);
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 50256),   // GPT-2's real ids: 0 .. 50256
                                    {batch_size, seq_len});
    Tensor encoded_grad = Tensor::randn({batch_size, seq_len, embed_dim});
    Tensor encoded = Tensor::zeros({batch_size, seq_len, embed_dim});

    torch::Tensor pytorch_wte = pytorch_copy_of(wte).requires_grad_(true);
    torch::Tensor pytorch_wpe = pytorch_copy_of(wpe).requires_grad_(true);
    torch::Tensor pytorch_ids = pytorch_copy_of(token_ids), pytorch_encoded_grad = pytorch_copy_of(encoded_grad);
    torch::Tensor pytorch_encoded;

    // Step 1 does first-time work (CUDA loads each side's kernels), so the average is of 2..20.
    print_test_name(test);
    printf("    time in ms:   step   my forward   my backward   PyTorch forward   PyTorch backward\n");
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { encoder_forward(encoded, token_ids, wte, wpe); });
        times[1] = gpu_milliseconds([&] { encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids); });
        times[2] = gpu_milliseconds([&] {
            pytorch_encoded = torch::embedding(pytorch_wte, pytorch_ids) + pytorch_wpe.narrow(0, 0, seq_len);
        });
        times[3] = gpu_milliseconds([&] { pytorch_encoded.backward(pytorch_encoded_grad); });
        printf("                  %4d   %10.3f   %11.3f   %15.3f   %16.3f\n", t, times[0], times[1], times[2],
               times[3]);
        if (t == 1)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    printf("    average of steps 2..20:  %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 19, totals[1] / 19,
           totals[2] / 19, totals[3] / 19);
    printf("    PyTorch takes, against mine:                         %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_EQ(encoded.tolist(), list_of(pytorch_encoded));
    ASSERT_NEAR(wte_grad.tolist(), list_of(pytorch_wte.grad()), TOLERANCE);
    ASSERT_NEAR(wpe_grad.tolist(), list_of(pytorch_wpe.grad()), TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times on GPT-2 small's encoder: wte {50304, 768}, wpe {1024, 768}, batch_size =
    // 4 rows of seq_len = 1024 ids, with wte, wpe and their gradients each a Tensor of its own. A
    // forward reads 4096 rows of wte and of wpe and writes 4096 rows of encoded: 3 x 12.6 MB =
    // 37.7 MB (wpe's 3 MB is read again for each of the 4 rows: the T4's 4 MB L2 cache most likely
    // does not keep it while wte's rows stream through), so at a T4's 320 GB/s at least 0.12 ms.
    Tensor::manual_seed(0);
    Tensor wte = Tensor::randn({50304, 768});                          // made on the CPU: a few seconds
    Tensor wpe = Tensor::randn({1024, 768});
    Tensor wte_grad = Tensor::zeros({50304, 768}), wpe_grad = Tensor::zeros({1024, 768});
    time_20_steps(__func__, wte, wpe, wte_grad, wpe_grad);
}

static void time_20_steps_on_gpt2_small_size_in_tensor_buffer(void)
{
    // Prints the times on the same sizes, with wte and wpe one after the other in one
    // TensorBuffer and their gradients in a second one laid out the same, as the trainer keeps
    // them. The kernels get the same pointers either way, so the times should match the test
    // above; a write past wte_grad's end would land in wpe_grad and fail its check.
    Tensor::manual_seed(0);
    TensorBuffer weights;
    int wte_index = weights.add({50304, 768});
    int wpe_index = weights.add({1024, 768});
    weights.allocate();
    weights.flat().normal_(0.0f, 1.0f);                                // both, made on the CPU: a few seconds
    TensorBuffer grads = TensorBuffer::zeros_like(weights);
    time_20_steps(__func__, weights.tensor(wte_index), weights.tensor(wpe_index), grads.tensor(wte_index),
                  grads.tensor(wpe_index));
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the same wte and wpe, 4 wide: wte's rows count up from 0, so each
// value shows which token row it came from; wpe's rows are 100s, so each shows its position.
//
//     wte   token 0: 0 1 2 3        wpe   position 0: 100 100 100 100
//           token 1: 4 5 6 7              position 1: 200 200 200 200
//           token 2: 8 9 10 11            position 2: 300 300 300 300
//                                         position 3: 400 400 400 400

/// Makes the wte of part 2: 3 tokens, rows 0 1 2 3 | 4 5 6 7 | 8 9 10 11.
static Tensor small_wte(void)
{
    return my_tensor_of(sample_data(12), {3, 4});
}

/// Makes the wpe of part 2: 4 positions, rows of 100s, 200s, 300s, 400s.
static Tensor small_wpe(void)
{
    std::vector<float> rows = {100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400};
    return my_tensor_of(rows, {4, 4});
}

static void forward_adds_the_tokens_row_and_the_positions_row(void)
{
    // Expect: ids {2, 0}: encoded row 0 is token 2's row + position 0's = 8 9 10 11 + 100 =
    // 108 109 110 111; encoded row 1 is token 0's + position 1's = 0 1 2 3 + 200 = 200 201 202 203.
    Tensor token_ids = my_tensor_of(std::vector<int>{2, 0}, {1, 2});
    Tensor encoded = Tensor::zeros({1, 2, 4});
    encoder_forward(encoded, token_ids, small_wte(), small_wpe());
    ASSERT_EQ(encoded.tolist(), std::vector<float>({108, 109, 110, 111, 200, 201, 202, 203}));
}

static void the_same_token_at_another_position_gets_another_row(void)
{
    // Expect: ids {1, 1, 1}, token 1 three times: 4 5 6 7 plus 100, then 200, then 300. The
    // position row is what tells the model the 3 apart.
    Tensor token_ids = my_tensor_of(std::vector<int>{1, 1, 1}, {1, 3});
    Tensor encoded = Tensor::zeros({1, 3, 4});
    encoder_forward(encoded, token_ids, small_wte(), small_wpe());
    ASSERT_EQ(encoded.tolist(), std::vector<float>({104, 105, 106, 107, 204, 205, 206, 207, 304, 305, 306, 307}));
}

static void every_row_of_the_batch_starts_again_at_position_0(void)
{
    // Expect: batch_size = 2 rows, ids {{0, 1}, {2, 0}}. Each row of the batch is its own text,
    // so its first token is at position 0 again:
    //
    //     b 0:  token 0 + position 0 = 100 101 102 103     token 1 + position 1 = 204 205 206 207
    //     b 1:  token 2 + position 0 = 108 109 110 111     token 0 + position 1 = 200 201 202 203
    Tensor token_ids = my_tensor_of(std::vector<int>{0, 1, 2, 0}, {2, 2});
    Tensor encoded = Tensor::zeros({2, 2, 4});
    encoder_forward(encoded, token_ids, small_wte(), small_wpe());
    ASSERT_EQ(encoded.tolist(),
              std::vector<float>({100, 101, 102, 103, 204, 205, 206, 207, 108, 109, 110, 111, 200, 201, 202, 203}));
}

static void backward_adds_a_repeated_tokens_rows_together(void)
{
    // Expect: ids {1, 1, 1} and encoded_grad rows of 1s, 2s, 3s. Token 1 made all 3 rows, so its
    // wte_grad row gets all 3: 1 + 2 + 3 = 6. Tokens 0 and 2 were not read: 0. Each position
    // made one row, so wpe_grad's rows are 1, 2, 3, and position 3 (after seq_len = 3) stays 0.
    Tensor token_ids = my_tensor_of(std::vector<int>{1, 1, 1}, {1, 3});
    Tensor encoded_grad = my_tensor_of(std::vector<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), {1, 3, 4});
    Tensor wte_grad = Tensor::zeros({3, 4}), wpe_grad = Tensor::zeros({4, 4});
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    ASSERT_EQ(wte_grad.tolist(), std::vector<float>({0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0}));
    ASSERT_EQ(wpe_grad.tolist(), std::vector<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0}));
}

static void backward_adds_the_batch_rows_at_each_position(void)
{
    // Expect: batch_size = 2 rows, ids {{0, 1}, {2, 0}}, encoded_grad counting up from 0:
    //
    //     (b 0, t 0)  0  1  2  3    token 0      (b 1, t 0)  8  9 10 11    token 2
    //     (b 0, t 1)  4  5  6  7    token 1      (b 1, t 1) 12 13 14 15    token 0
    //
    //     wpe_grad position 0 = (b 0, t 0) + (b 1, t 0) =  8 10 12 14
    //              position 1 = (b 0, t 1) + (b 1, t 1) = 16 18 20 22      positions 2, 3: 0
    //     wte_grad token 0    = (b 0, t 0) + (b 1, t 1) = 12 14 16 18
    //              token 1    = (b 0, t 1)              =  4  5  6  7
    //              token 2    = (b 1, t 0)              =  8  9 10 11
    Tensor token_ids = my_tensor_of(std::vector<int>{0, 1, 2, 0}, {2, 2});
    Tensor encoded_grad = my_tensor_of(sample_data(16), {2, 2, 4});
    Tensor wte_grad = Tensor::zeros({3, 4}), wpe_grad = Tensor::zeros({4, 4});
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    ASSERT_EQ(wte_grad.tolist(), std::vector<float>({12, 14, 16, 18, 4, 5, 6, 7, 8, 9, 10, 11}));
    ASSERT_EQ(wpe_grad.tolist(), std::vector<float>({8, 10, 12, 14, 16, 18, 20, 22, 0, 0, 0, 0, 0, 0, 0, 0}));
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there). One
    // token, id 0, encoded_grad 1 2 3 4: row 0 of both becomes 11 12 13 14 and every other row stays
    // 10. A second backward adds again: 12 14 16 18. Gradient accumulation relies on this.
    Tensor token_ids = my_tensor_of(std::vector<int>{0}, {1, 1});
    Tensor encoded_grad = my_tensor_of(std::vector<float>({1, 2, 3, 4}), {1, 1, 4});
    Tensor wte_grad = Tensor::zeros({3, 4}).fill_(10), wpe_grad = Tensor::zeros({4, 4}).fill_(10);
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    ASSERT_EQ(wte_grad.tolist(), std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 10, 10, 10, 10}));
    ASSERT_EQ(wpe_grad.tolist(), std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}));
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    ASSERT_EQ(wte_grad[0].tolist(), std::vector<float>({12, 14, 16, 18}));
    ASSERT_EQ(wpe_grad[0].tolist(), std::vector<float>({12, 14, 16, 18}));
}

static void many_threads_adding_onto_one_row_lose_nothing(void)
{
    // Expect: GPT-2 small's sizes, batch_size = 4 rows of seq_len = 1024 ids, every id 5,
    // encoded_grad all 1. The wte backward's 524,288 threads take 786,432 float4s, 3,145,728 adds,
    // and the 4096 adds of each column all go onto the same value of token 5's row, many at the
    // same moment. With atomicAdd no add is lost: 4096 x 1 = 4096, exactly (whole numbers this
    // small add exactly in floats); with a plain += most of them would be. Tokens 0 .. 4 stay 0.
    // Every position has 4 rows of 1s: wpe_grad is 4 everywhere.
    size_t batch_size = 4, seq_len = 1024, embed_dim = 768;
    Tensor token_ids = Tensor::zeros({batch_size, seq_len}, DType::Int32).fill_(5);
    Tensor encoded_grad = Tensor::ones({batch_size, seq_len, embed_dim});
    Tensor wte_grad = Tensor::zeros({6, embed_dim}), wpe_grad = Tensor::zeros({seq_len, embed_dim});
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    ASSERT_EQ(wte_grad[5].tolist(), std::vector<float>(embed_dim, 4096.0f));
    ASSERT_EQ(wte_grad.narrow(0, 0, 5).tolist(), std::vector<float>(5 * embed_dim, 0.0f));
    ASSERT_EQ(wpe_grad.tolist(), std::vector<float>(seq_len * embed_dim, 4.0f));
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_a_block_of_encoder_threads(void)
{
    // Expect: each of the 3 kernels can be launched with ENCODER_THREADS (1024) threads per block,
    // and nothing is spilled. The compiler picks each kernel's registers per thread; a block of
    // 1024 needs at most 65,536 / 1024 = 64 each, or the launch fails with "too many resources
    // requested for launch". cudaFuncGetAttributes reads what the compiler wrote down, without
    // running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with: 1024 if it fits
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)encoder_forward_kernel, (const void *)encoder_backward_wte_kernel,
                             (const void *)encoder_backward_wpe_kernel};
    const char *names[] = {"encoder_forward_kernel", "encoder_backward_wte_kernel", "encoder_backward_wpe_kernel"};
    for (int k = 0; k < 3; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= ENCODER_THREADS, true);
        ASSERT_EQ(attributes.localSizeBytes, (size_t)0);
    }
}

// ══ part 4: other block and thread counts ═════════════════════════════════════

/// Runs `run` 20 times on the GPU and gives the average time of runs 2..20, in milliseconds;
/// run 1 does first-time work, so it is left out.
template <typename Run>
static float average_milliseconds(Run run)
{
    float total = 0.0f;
    for (int t = 1; t <= 20; t++)
    {
        float milliseconds = gpu_milliseconds(run);
        if (t > 1)
            total += milliseconds;
    }
    return total / 19;
}

/// Keeps the GPU busy with `run` for about a second before timing. While the CPU makes random
/// numbers for seconds the GPU idles and its clock drops; without this, the first launches timed
/// run while the clock is still rising, and with few threads they come out slower by a different
/// amount every run.
template <typename Run>
static void warm_up_gpu(Run run)
{
    float total = 0.0f;
    while (total < 1000.0f)
        total += gpu_milliseconds(run);
}

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of each of the 3 kernels on GPT-2 small's sizes (batch_size = 4, seq_len =
    // 1024) for 8 block counts x 4 thread counts, each the average of runs 2..20. The kernels are
    // launched here directly, <<<blocks, threads>>>, instead of through encoder_forward and
    // encoder_backward, which always launch ENCODER_BLOCKS x ENCODER_THREADS (512 x 1024).
    //
    // Expect: every launch gives the same result as 512 x 1024 (the grid-stride loop covers
    // every value whatever the count), so only the time changes. A T4 runs at most 40 SMs x 1024
    // = 40,960 threads at once: with fewer, too few warps are left to run while the others wait
    // on memory, so the time should go up; with more, the extra blocks wait their turn and the
    // time should stay about the same.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, vocab_size = 50304, embed_dim = 768;
    Tensor wte = Tensor::randn({vocab_size, embed_dim});              // made on the CPU: a few seconds
    Tensor wpe = Tensor::randn({seq_len, embed_dim});
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 50256),   // GPT-2's real ids: 0 .. 50256
                                    {batch_size, seq_len});
    Tensor encoded_grad = Tensor::randn({batch_size, seq_len, embed_dim});
    Tensor encoded = Tensor::zeros({batch_size, seq_len, embed_dim});
    Tensor wte_grad = Tensor::zeros({vocab_size, embed_dim}), wpe_grad = Tensor::zeros({seq_len, embed_dim});

    // The usual launch's results, kept on the GPU as PyTorch tensors to compare each launch with.
    encoder_forward(encoded, token_ids, wte, wpe);
    encoder_backward(wte_grad, wpe_grad, encoded_grad, token_ids);
    torch::Tensor usual_encoded = pytorch_copy_of(encoded);
    torch::Tensor usual_wte_grad = pytorch_copy_of(wte_grad), usual_wpe_grad = pytorch_copy_of(wpe_grad);

    int blocks = 0, threads = 0; // the launch the 3 below use, set by the loop
    auto forward = [&] {
        encoder_forward_kernel<<<blocks, threads>>>((float4 *)encoded.data(), token_ids.data_ptr<int>(),
                                                    (const float4 *)wte.data(), (const float4 *)wpe.data(),
                                                    batch_size, seq_len, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wte = [&] {
        encoder_backward_wte_kernel<<<blocks, threads>>>(wte_grad.data(), (const float4 *)encoded_grad.data(),
                                                         token_ids.data_ptr<int>(), batch_size, seq_len, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    auto backward_wpe = [&] {
        encoder_backward_wpe_kernel<<<blocks, threads>>>((float4 *)wpe_grad.data(), (const float4 *)encoded_grad.data(),
                                                         batch_size, seq_len, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    blocks = ENCODER_BLOCKS;
    threads = ENCODER_THREADS;
    warm_up_gpu(forward);

    int block_counts[] = {40, 80, 160, 320, 512, 1024, 2048, 4096};
    int thread_counts[] = {128, 256, 512, 1024};
    const char *kernel_names[] = {"forward", "wte backward", "wpe backward"};
    float fastest_ms[3] = {1e9f, 1e9f, 1e9f};
    int fastest_blocks[3] = {0, 0, 0}, fastest_threads[3] = {0, 0, 0};
    int launches_with_the_same_result = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   all threads   forward   wte backward   wpe backward   same result\n");
    for (int block_count : block_counts)
    {
        for (int thread_count : thread_counts)
        {
            blocks = block_count;
            threads = thread_count;
            float times[3];
            times[0] = average_milliseconds(forward);
            times[1] = average_milliseconds(backward_wte);
            times[2] = average_milliseconds(backward_wpe);
            for (int k = 0; k < 3; k++)
            {
                if (times[k] < fastest_ms[k])
                {
                    fastest_ms[k] = times[k];
                    fastest_blocks[k] = blocks;
                    fastest_threads[k] = threads;
                }
            }

            // One forward and one backward from 0 with this launch, against the usual launch's:
            // encoded and wpe_grad to the last bit, wte_grad within TOLERANCE (atomicAdd's order).
            encoded.zero_();
            wte_grad.zero_();
            wpe_grad.zero_();
            forward();
            backward_wte();
            backward_wpe();
            torch::Tensor wte_grad_difference =
                (pytorch_copy_of(wte_grad) - usual_wte_grad).abs().max() / usual_wte_grad.abs().max();
            bool same = torch::equal(pytorch_copy_of(encoded), usual_encoded) &&
                        torch::equal(pytorch_copy_of(wpe_grad), usual_wpe_grad) &&
                        wte_grad_difference.item<float>() <= TOLERANCE;
            if (same)
                launches_with_the_same_result++;

            bool usual = blocks == ENCODER_BLOCKS && threads == ENCODER_THREADS;
            printf("                %6d   %7d   %11d   %7.3f   %12.3f   %12.3f   %-11s%s\n", blocks, threads,
                   blocks * threads, times[0], times[1], times[2], same ? "yes" : "NO",
                   usual ? "   <- what encoder.cuh launches" : "");
        }
    }
    for (int k = 0; k < 3; k++)
        printf("    fastest %-12s  %4d blocks x %4d threads: %.3f ms\n", kernel_names[k], fastest_blocks[k],
               fastest_threads[k], fastest_ms[k]);
    ASSERT_EQ(launches_with_the_same_result, 8 * 4);
}

// ══ part 5: experiments on what part 4 found ══════════════════════════════════

/// The wte backward as encoder.cuh had it before: one float of encoded_grad per load, then one
/// float atomicAdd, 3,145,728 jobs. Kept here for part 5, to compare with encoder.cuh's kernel,
/// which loads a float4 (16 bytes in flight per thread instead of 4) and then does four adds.
__global__ void encoder_backward_wte_float_kernel(float *wte_grad, const float *encoded_grad, const int *token_ids,
                                                  size_t batch_size, size_t seq_len, size_t embed_dim)
{
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_values = batch_size * seq_len * embed_dim;
    for (size_t index = thread_number; index < total_values; index += (size_t)gridDim.x * blockDim.x)
    {
        size_t token_index = index / embed_dim;
        size_t embed_index = index % embed_dim;
        size_t token_id = token_ids[token_index];
        atomicAdd(&wte_grad[token_id * embed_dim + embed_index], encoded_grad[index]);
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
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, vocab_size = 50304, embed_dim = 768;
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 50256), {batch_size, seq_len});
    Tensor encoded_grad = Tensor::randn({batch_size, seq_len, embed_dim});
    Tensor wte_grad = Tensor::zeros({vocab_size, embed_dim});

    // The usual launch's result, to compare each launch with.
    encoder_backward_wte_kernel<<<ENCODER_BLOCKS, ENCODER_THREADS>>>(wte_grad.data(),
                                                                     (const float4 *)encoded_grad.data(),
                                                                     token_ids.data_ptr<int>(), batch_size, seq_len,
                                                                     embed_dim);
    cudaCheck(cudaGetLastError());
    torch::Tensor usual_wte_grad = pytorch_copy_of(wte_grad);

    int blocks = 0, threads = 0; // the launch the 2 below use, set by the loop
    auto float_loads = [&] {
        encoder_backward_wte_float_kernel<<<blocks, threads>>>(wte_grad.data(), encoded_grad.data(),
                                                               token_ids.data_ptr<int>(), batch_size, seq_len,
                                                               embed_dim);
        cudaCheck(cudaGetLastError());
    };
    auto float4_loads = [&] {
        encoder_backward_wte_kernel<<<blocks, threads>>>(wte_grad.data(), (const float4 *)encoded_grad.data(),
                                                         token_ids.data_ptr<int>(), batch_size, seq_len, embed_dim);
        cudaCheck(cudaGetLastError());
    };
    // Tells if wte_grad, after one backward from 0 with `run`, equals the usual launch's.
    auto same_as_usual = [&](auto run) {
        wte_grad.zero_();
        run();
        torch::Tensor difference =
            (pytorch_copy_of(wte_grad) - usual_wte_grad).abs().max() / usual_wte_grad.abs().max();
        return difference.item<float>() <= TOLERANCE;
    };
    blocks = ENCODER_BLOCKS;
    threads = ENCODER_THREADS;
    warm_up_gpu(float4_loads);

    int block_counts[] = {40, 80, 160, 320, 512, 1024, 2048, 4096};
    int thread_counts[] = {128, 256, 512, 1024};
    float float_ms[8][4], float4_ms[8][4];
    int launches_with_the_same_result = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   all threads   float loads   float4 loads   same result\n");
    for (int block_index = 0; block_index < 8; block_index++)
    {
        for (int thread_index = 0; thread_index < 4; thread_index++)
        {
            blocks = block_counts[block_index];
            threads = thread_counts[thread_index];
            float_ms[block_index][thread_index] = average_milliseconds(float_loads);
            float4_ms[block_index][thread_index] = average_milliseconds(float4_loads);
            bool same = same_as_usual(float_loads) && same_as_usual(float4_loads);
            if (same)
                launches_with_the_same_result++;
            printf("                %6d   %7d   %11d   %11.3f   %12.3f   %s\n", blocks, threads, blocks * threads,
                   float_ms[block_index][thread_index], float4_ms[block_index][thread_index], same ? "yes" : "NO");
        }
    }
    // 40 x 128 is [0][0]; 512 x 1024 is [4][3]
    printf("    slow-down at 5,120 threads against 512 x 1024:   float loads %.2fx   float4 loads %.2fx\n",
           float_ms[0][0] / float_ms[4][3], float4_ms[0][0] / float4_ms[4][3]);
    printf("    slow-down at 10,240 threads (40 x 256):           float loads %.2fx   float4 loads %.2fx\n",
           float_ms[0][1] / float_ms[4][3], float4_ms[0][1] / float4_ms[4][3]);
    ASSERT_EQ(launches_with_the_same_result, 8 * 4);
}

static void time_forward_with_blocks_of_128_around_512(void)
{
    // Prints the forward with blocks of 128 threads and 256 to 1,024 blocks in steps of 64, each
    // timed right after 512 x 1024 so the two are compared in the same moment. In part 4's 6 runs,
    // 512 x 128 was the fastest forward every time, 4-6% ahead of 512 x 1024.
    //
    // Expect one of two shapes: a smooth dip around 512 blocks (a real effect of the launch size)
    // or one low point at 512 alone (something special about that one launch). The jobs per thread
    // are printed too: 512 x 128 gives every thread exactly 12 of the 786,432 jobs, but part 4's
    // launches with exactly 3 each were not faster, so an even share alone is not expected to matter.
    Tensor::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, vocab_size = 50304, embed_dim = 768;
    Tensor wte = Tensor::randn({vocab_size, embed_dim}); // made on the CPU: a few seconds
    Tensor wpe = Tensor::randn({seq_len, embed_dim});
    Tensor token_ids = my_tensor_of(random_token_ids(batch_size * seq_len, 50256), {batch_size, seq_len});
    Tensor encoded = Tensor::zeros({batch_size, seq_len, embed_dim});
    encoder_forward(encoded, token_ids, wte, wpe);
    torch::Tensor usual_encoded = pytorch_copy_of(encoded);

    int blocks = 0, threads = 0; // the launch the forward below uses
    auto forward = [&] {
        encoder_forward_kernel<<<blocks, threads>>>((float4 *)encoded.data(), token_ids.data_ptr<int>(),
                                                    (const float4 *)wte.data(), (const float4 *)wpe.data(),
                                                    batch_size, seq_len, embed_dim);
        cudaCheck(cudaGetLastError());
    };

    blocks = ENCODER_BLOCKS;
    threads = ENCODER_THREADS;
    warm_up_gpu(forward);

    int launches_with_the_same_result = 0, launches = 0;
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   jobs per thread   forward   512 x 1024 just before   ratio\n");
    for (int block_count = 256; block_count <= 1024; block_count += 64)
    {
        blocks = ENCODER_BLOCKS;
        threads = ENCODER_THREADS;
        float usual_ms = average_milliseconds(forward);
        blocks = block_count;
        threads = 128;
        float ms = average_milliseconds(forward);

        encoded.zero_();
        forward();
        bool same = torch::equal(pytorch_copy_of(encoded), usual_encoded);
        if (same)
            launches_with_the_same_result++;
        launches++;
        double jobs_per_thread = (double)(batch_size * seq_len * embed_dim / 4) / (blocks * threads);
        printf("                %6d   %7d   %15.2f   %7.3f   %22.3f   %5.3f%s\n", blocks, threads, jobs_per_thread,
               ms, usual_ms, ms / usual_ms, same ? "" : "   NOT THE SAME RESULT");
    }
    ASSERT_EQ(launches_with_the_same_result, launches);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    forward_matches_pytorch();
    backward_matches_pytorch();
    time_20_steps_on_gpt2_small_size();
    time_20_steps_on_gpt2_small_size_in_tensor_buffer();

    forward_adds_the_tokens_row_and_the_positions_row();
    the_same_token_at_another_position_gets_another_row();
    every_row_of_the_batch_starts_again_at_position_0();
    backward_adds_a_repeated_tokens_rows_together();
    backward_adds_the_batch_rows_at_each_position();
    backward_adds_onto_the_gradients_already_there();
    many_threads_adding_onto_one_row_lose_nothing();

    every_kernel_fits_a_block_of_encoder_threads();

    time_each_kernel_with_other_block_and_thread_counts();

    time_wte_backward_with_float4_loads();
    time_forward_with_blocks_of_128_around_512();

    printf("\nall tests passed\n");
    return 0;
}
