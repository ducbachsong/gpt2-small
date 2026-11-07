// test_encoder.cu — tests for llmc/kernels/encoder.cuh (encoder_forward, encoder_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's embedding, with autograd for the backward; part 2
// checks numbers worked out by hand; part 3 checks that each kernel's registers fit a block
// of ENCODER_THREADS, and that any block size gives the same result. The timings are in
// tunes/tune_encoder.cu. (The C# trainer had no encoder test: TorchSharp's embedding was trusted.)
//
// The tests use a Tensor for each weight, the simplest to write.
#include <torch/types.h>              // torch::tensor, torch::embedding, backward(); libtorch before any CUDA header
#include <random>                     // std::mt19937: random token ids
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: wte and wpe in one allocation, as the trainer keeps them
#include "../llmc/kernels/encoder.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it. A repeated token's wte_grad row is rounded
// once per add, a few times here.
static const double BFLOAT16_TOLERANCE = 2e-2;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

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

/// Makes a tensor of `shape_list` holding `values_list` (float) on the GPU.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes an int tensor of `shape_list` holding `ids_list` on the GPU: token ids.
static Tensor ids_of(const Shape &shape_list, const std::vector<int> &ids_list)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

static void forward_matches_pytorch(void)
{
    // Expect: a tiny GPT-2's encoder (vocab 64, positions 16, 16 wide), batch_size = 4 rows of
    // seq_len = 8 random ids: encoded equals PyTorch's embedding(wte, ids) + wpe[0 .. 7], exactly.
    // Each value is one float add of the same two numbers on both sides, so it rounds the same.

    // 1. a tiny encoder and random ids; mine
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = tensor_of({64, 16}, random_values({64, 16}));
    Tensor wpe_tensor = tensor_of({16, 16}, random_values({16, 16}));
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 63));
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16});
    encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);

    // 2. PyTorch's
    torch::Tensor pytorch_encoded_tensor = torch::embedding(pytorch_copy_of(wte_tensor), pytorch_copy_of(token_ids_tensor)) +
                                           pytorch_copy_of(wpe_tensor).narrow(0, 0, seq_len_val);

    // 3. the same bits
    ASSERT_EQ(encoded_tensor.tolist(), list_of(pytorch_encoded_tensor));
}

static void backward_matches_pytorch(void)
{
    // Expect: the same tiny encoder, but the ids only from 0 .. 15, so most of them come up 2 or
    // more times in the 32 and their wte_grad rows are sums. Random encoded_grad, backward from 0:
    // wte_grad and wpe_grad equal PyTorch's .grad after autograd's backward, within 1e-5 (the
    // two add a repeated token's rows in a different order). wte_grad's rows 16 .. 63 (ids never
    // read) stay 0, and so do wpe_grad's rows 8 .. 15 (positions after seq_len).

    // 1. a tiny encoder, ids from 0 .. 15, random encoded_grad; mine
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = tensor_of({64, 16}, random_values({64, 16}));
    Tensor wpe_tensor = tensor_of({16, 16}, random_values({16, 16}));
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
    Tensor encoded_grad_tensor = tensor_of({batch_size_val, seq_len_val, 16}, random_values({batch_size_val, seq_len_val, 16}));
    Tensor wte_grad_tensor = Tensor::zeros({64, 16});
    Tensor wpe_grad_tensor = Tensor::zeros({16, 16});
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

    // 2. PyTorch's, with autograd
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_wpe_tensor = pytorch_copy_of(wpe_tensor).requires_grad_(true);
    torch::Tensor pytorch_encoded_tensor =
        torch::embedding(pytorch_wte_tensor, pytorch_copy_of(token_ids_tensor)) + pytorch_wpe_tensor.narrow(0, 0, seq_len_val);
    pytorch_encoded_tensor.backward(pytorch_copy_of(encoded_grad_tensor));

    // 3. check; the rows never read stay 0
    ASSERT_NEAR(wte_grad_tensor.tolist(), list_of(pytorch_wte_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(wpe_grad_tensor.tolist(), list_of(pytorch_wpe_tensor.grad()), TOLERANCE);
    ASSERT_EQ(wte_grad_tensor.narrow(0, 16, 48).tolist(), std::vector<float>(48 * 16, 0.0f));
    ASSERT_EQ(wpe_grad_tensor.narrow(0, 8, 8).tolist(), std::vector<float>(8 * 16, 0.0f));
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: forward_matches_pytorch's and backward_matches_pytorch's tiny encoder, with ids from 0 .. 15,
    // in bfloat16 tensors. encoded is each pair of bfloat16s added in float and rounded once: PyTorch's float
    // add of the same values, rounded to bfloat16 by PyTorch, to the last bit. wpe_grad adds the 4 rows in
    // float and rounds once; wte_grad rounds after each add onto a repeated token's row (the 8-byte
    // compare-and-swap of atomic_add4_from_float): both within BFLOAT16_TOLERANCE of PyTorch's float sums. The rows
    // never read stay 0.

    // 1. a tiny encoder in bfloat16; mine
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = tensor_of({64, 16}, random_values({64, 16})).to(DType::BFloat16);
    Tensor wpe_tensor = tensor_of({16, 16}, random_values({16, 16})).to(DType::BFloat16);
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
    Tensor encoded_grad_tensor =
        tensor_of({batch_size_val, seq_len_val, 16}, random_values({batch_size_val, seq_len_val, 16})).to(DType::BFloat16);
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, DType::BFloat16);
    Tensor wte_grad_tensor = Tensor::zeros({64, 16}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({16, 16}, DType::BFloat16);
    encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

    // 2. PyTorch's, in float on the same values
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_wpe_tensor = pytorch_copy_of(wpe_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_encoded_tensor =
        torch::embedding(pytorch_wte_tensor, pytorch_copy_of(token_ids_tensor)) + pytorch_wpe_tensor.narrow(0, 0, seq_len_val);
    pytorch_encoded_tensor.backward(pytorch_copy_of(encoded_grad_tensor.to(DType::Float32)));

    // 3. check; the rows never read stay 0
    ASSERT_EQ(encoded_tensor.to(DType::Float32).tolist(), list_of(pytorch_encoded_tensor.to(torch::kBFloat16).to(torch::kFloat)));
    ASSERT_NEAR(wte_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_wte_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(wpe_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_wpe_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_EQ(wte_grad_tensor.narrow(0, 16, 48).to(DType::Float32).tolist(), std::vector<float>(48 * 16, 0.0f));
    ASSERT_EQ(wpe_grad_tensor.narrow(0, 8, 8).to(DType::Float32).tolist(), std::vector<float>(8 * 16, 0.0f));
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
    return tensor_of({3, 4}, sample_data(12));
}

/// Makes the wpe of part 2: 4 positions, rows of 100s, 200s, 300s, 400s.
static Tensor small_wpe(void)
{
    return tensor_of({4, 4}, {100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400});
}

static void forward_adds_the_tokens_row_and_the_positions_row(void)
{
    // Expect: ids {2, 0}: encoded row 0 is token 2's row + position 0's = 8 9 10 11 + 100 =
    // 108 109 110 111; encoded row 1 is token 0's + position 1's = 0 1 2 3 + 200 = 200 201 202 203.
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({1, 2, 4});
    encoder_forward(encoded_tensor, ids_of({1, 2}, {2, 0}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(encoded_tensor.tolist(), std::vector<float>({108, 109, 110, 111, 200, 201, 202, 203}));
}

static void the_same_token_at_another_position_gets_another_row(void)
{
    // Expect: ids {1, 1, 1}, token 1 three times: 4 5 6 7 plus 100, then 200, then 300. The
    // position row is what tells the model the 3 apart.
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({1, 3, 4});
    encoder_forward(encoded_tensor, ids_of({1, 3}, {1, 1, 1}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(encoded_tensor.tolist(), std::vector<float>({104, 105, 106, 107, 204, 205, 206, 207, 304, 305, 306, 307}));
}

static void every_row_of_the_batch_starts_again_at_position_0(void)
{
    // Expect: batch_size = 2 rows, ids {{0, 1}, {2, 0}}. Each row of the batch is its own text,
    // so its first token is at position 0 again:
    //
    //     b 0:  token 0 + position 0 = 100 101 102 103     token 1 + position 1 = 204 205 206 207
    //     b 1:  token 2 + position 0 = 108 109 110 111     token 0 + position 1 = 200 201 202 203
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({2, 2, 4});
    encoder_forward(encoded_tensor, ids_of({2, 2}, {0, 1, 2, 0}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(encoded_tensor.tolist(),
              std::vector<float>({100, 101, 102, 103, 204, 205, 206, 207, 108, 109, 110, 111, 200, 201, 202, 203}));
}

static void backward_adds_a_repeated_tokens_rows_together(void)
{
    // Expect: ids {1, 1, 1} and encoded_grad rows of 1s, 2s, 3s. Token 1 made all 3 rows, so its
    // wte_grad row gets all 3: 1 + 2 + 3 = 6. Tokens 0 and 2 were not read: 0. Each position
    // made one row, so wpe_grad's rows are 1, 2, 3, and position 3 (after seq_len = 3) stays 0.
    // 1. the backward
    Tensor encoded_grad_tensor = tensor_of({1, 3, 4}, {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});
    Tensor wte_grad_tensor = Tensor::zeros({3, 4});
    Tensor wpe_grad_tensor = Tensor::zeros({4, 4});
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, ids_of({1, 3}, {1, 1, 1}));
    // 2. check
    ASSERT_EQ(wte_grad_tensor.tolist(), std::vector<float>({0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0}));
    ASSERT_EQ(wpe_grad_tensor.tolist(), std::vector<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0}));
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
    // 1. the backward
    Tensor wte_grad_tensor = Tensor::zeros({3, 4});
    Tensor wpe_grad_tensor = Tensor::zeros({4, 4});
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, tensor_of({2, 2, 4}, sample_data(16)), ids_of({2, 2}, {0, 1, 2, 0}));
    // 2. check
    ASSERT_EQ(wte_grad_tensor.tolist(), std::vector<float>({12, 14, 16, 18, 4, 5, 6, 7, 8, 9, 10, 11}));
    ASSERT_EQ(wpe_grad_tensor.tolist(), std::vector<float>({8, 10, 12, 14, 16, 18, 20, 22, 0, 0, 0, 0, 0, 0, 0, 0}));
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there). One
    // token, id 0, encoded_grad 1 2 3 4: row 0 of both becomes 11 12 13 14 and every other row stays
    // 10. A second backward adds again: 12 14 16 18. Gradient accumulation relies on this.
    // 1. gradients that start at 10
    Tensor token_ids_tensor = ids_of({1, 1}, {0});
    Tensor encoded_grad_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor wte_grad_tensor = Tensor::zeros({3, 4}).fill_(10);
    Tensor wpe_grad_tensor = Tensor::zeros({4, 4}).fill_(10);
    // 2. one backward: added onto
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    ASSERT_EQ(wte_grad_tensor.tolist(), std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 10, 10, 10, 10}));
    ASSERT_EQ(wpe_grad_tensor.tolist(), std::vector<float>({11, 12, 13, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}));
    // 3. a second backward: added again
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    ASSERT_EQ(wte_grad_tensor[0].tolist(), std::vector<float>({12, 14, 16, 18}));
    ASSERT_EQ(wpe_grad_tensor[0].tolist(), std::vector<float>({12, 14, 16, 18}));
}

static void many_threads_adding_onto_one_row_lose_nothing(void)
{
    // Expect: GPT-2 small's sizes, batch_size = 4 rows of seq_len = 1024 ids, every id 5,
    // encoded_grad all 1. The wte backward's 524,288 threads take 786,432 float4s, 3,145,728 adds,
    // and the 4096 adds of each column all go onto the same value of token 5's row, many at the
    // same moment. With atomicAdd no add is lost: 4096 x 1 = 4096, exactly (whole numbers this
    // small add exactly in floats); with a plain += most of them would be. Tokens 0 .. 4 stay 0.
    // Every position has 4 rows of 1s: wpe_grad is 4 everywhere.
    // 1. every id 5, encoded_grad all 1; the backward
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t embed_dim_val = 768;
    Tensor token_ids_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32).fill_(5);
    Tensor encoded_grad_tensor = Tensor::ones({batch_size_val, seq_len_val, embed_dim_val});
    Tensor wte_grad_tensor = Tensor::zeros({6, embed_dim_val});
    Tensor wpe_grad_tensor = Tensor::zeros({seq_len_val, embed_dim_val});
    encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    // 2. check
    ASSERT_EQ(wte_grad_tensor[5].tolist(), std::vector<float>(embed_dim_val, 4096.0f));
    ASSERT_EQ(wte_grad_tensor.narrow(0, 0, 5).tolist(), std::vector<float>(5 * embed_dim_val, 0.0f));
    ASSERT_EQ(wpe_grad_tensor.tolist(), std::vector<float>(seq_len_val * embed_dim_val, 4.0f));
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

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
    const void *kernels_list[] = {(const void *)encoder_forward_kernel<float>,
                                  (const void *)encoder_backward_wte_kernel<float>,
                                  (const void *)encoder_backward_wpe_kernel<float>};
    const char *names_list[] = {"encoder_forward_kernel", "encoder_backward_wte_kernel", "encoder_backward_wpe_kernel"};
    for (int kernel_index_val = 0; kernel_index_val < 3; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= ENCODER_THREADS, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_result(void)
{
    // Expect: encoder_forward and encoder_backward called with threads_per_block 32, 128, 512 and 1024 (one
    // warp up to MAX_THREADS_PER_BLOCK) give the same result as with the defaults, ENCODER_FORWARD_THREADS
    // and ENCODER_THREADS, in float and in bfloat16: encoded and wpe_grad to the last bit (each value worked
    // out alone, wpe_grad's batch rows added in order), wte_grad within TOLERANCE (BFLOAT16_TOLERANCE in
    // bfloat16): its atomic adds onto a repeated token's row can land in another order, as from run to run.
    // The tiny encoder of part 1, ids from 0 .. 15 so most tokens repeat.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. a tiny encoder; the default launches' results
        Tensor::manual_seed(0);
        size_t batch_size_val = 4;
        size_t seq_len_val = 8;
        double tolerance_val = dtype_val == DType::BFloat16 ? BFLOAT16_TOLERANCE : TOLERANCE;
        Tensor wte_tensor = tensor_of({64, 16}, random_values({64, 16})).to(dtype_val);
        Tensor wpe_tensor = tensor_of({16, 16}, random_values({16, 16})).to(dtype_val);
        Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
        Tensor encoded_grad_tensor =
            tensor_of({batch_size_val, seq_len_val, 16}, random_values({batch_size_val, seq_len_val, 16})).to(dtype_val);
        Tensor usual_encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, dtype_val);
        Tensor usual_wte_grad_tensor = Tensor::zeros({64, 16}, dtype_val);
        Tensor usual_wpe_grad_tensor = Tensor::zeros({16, 16}, dtype_val);
        encoder_forward(usual_encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
        encoder_backward(usual_wte_grad_tensor, usual_wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

        // 2. each block size, against the defaults'
        for (int threads_per_block_val : block_sizes_list)
        {
            Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, dtype_val);
            Tensor wte_grad_tensor = Tensor::zeros({64, 16}, dtype_val);
            Tensor wpe_grad_tensor = Tensor::zeros({16, 16}, dtype_val);
            encoder_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor, threads_per_block_val);
            encoder_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor, threads_per_block_val);
            ASSERT_EQ(encoded_tensor.to(DType::Float32).tolist(), usual_encoded_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(wpe_grad_tensor.to(DType::Float32).tolist(), usual_wpe_grad_tensor.to(DType::Float32).tolist());
            ASSERT_NEAR(wte_grad_tensor.to(DType::Float32).tolist(), usual_wte_grad_tensor.to(DType::Float32).tolist(),
                        tolerance_val);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_matches_pytorch();
    backward_matches_pytorch();
    forward_and_backward_in_bfloat16();

    forward_adds_the_tokens_row_and_the_positions_row();
    the_same_token_at_another_position_gets_another_row();
    every_row_of_the_batch_starts_again_at_position_0();
    backward_adds_a_repeated_tokens_rows_together();
    backward_adds_the_batch_rows_at_each_position();
    backward_adds_onto_the_gradients_already_there();
    many_threads_adding_onto_one_row_lose_nothing();

    every_kernel_fits_a_block_of_encoder_threads();
    any_block_size_gives_the_same_result();

    printf("\nall tests passed\n");
    return 0;
}
