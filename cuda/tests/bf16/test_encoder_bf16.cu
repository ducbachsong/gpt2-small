// test_encoder_bf16.cu — tests for cuda/kernels/bf16/encoder_bf16.cuh (encoder_bf16_forward, encoder_bf16_backward):
// GPT-2's first layer in bfloat16, 8 values a thread (one 16-byte load). Every check prints its test case, input,
// expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks against PyTorch's embedding (with autograd for the backward) and against fp32/encoder.cuh's bfloat16
// path (4 values a thread): the same bits; part 2 checks numbers worked out by hand; part 3 checks that each kernel's
// registers fit a block of ENCODER_BF16_THREADS, and that any block size gives the same result. Any GPU. The timings
// are in cuda/tests/bf16/tune_encoder_bf16.cu.
#include <torch/types.h>              // torch::embedding, torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include "../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../../kernels/fp32/encoder.cuh"       // the 4-value path, to compare with
#include "../../kernels/bf16/encoder_bf16.cuh"

// Against PyTorch in float on the same bfloat16 values: a bfloat16 keeps 8 significant bits, so each rounding moves
// a value by up to 2^-8, about 0.4% of it. A repeated token's wte_grad row is rounded once per add, a few times here.
// The largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 2e-2;

#define GPT2_HIGHEST_TOKEN_ID 50256   // GPT-2's real ids: 0 .. 50256
#define GPT2_PADDED_VOCAB_SIZE 50304  // GPT-2 small's sizes
#define GPT2_MAX_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_BATCH_SIZE 4

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

/// Makes `count_val` random token ids from 0 .. highest_id_val, as a CPU list.
///
///   count_val        how many ids
///   highest_id_val   the largest id
static std::vector<int> random_token_ids(size_t count_val, int highest_id_val)
{
    // The same seed gives the same ids every run.
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, highest_id_val);
    std::vector<int> ids_list(count_val);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    return ids_list;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values.
///
///   tensor   token ids or bfloat16 values, on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    // GPU to GPU. Token ids become int64, the type PyTorch's embedding is usually given; bfloat16 values become
    // floats, exactly.
    bool ints_val = tensor.dtype() == DType::Int32;
    Tensor source_tensor = ints_val ? tensor : tensor.to(DType::Float32);   // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(ints_val ? torch::kInt : torch::kFloat).device(torch::kCUDA);
    void *values_ptr = ints_val ? (void *)source_tensor.data_ptr<int>() : (void *)source_tensor.data();
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

/// Gives every value of my tensor as a CPU list of floats (bfloat16 turned into floats, exactly).
static std::vector<float> floats_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Makes a bfloat16 tensor of `shape_list` holding `values_list` on the GPU, each value rounded to bfloat16.
static Tensor bf16_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Makes an int tensor of `shape_list` holding `ids_list` on the GPU: token ids.
static Tensor ids_of(const Shape &shape_list, const std::vector<int> &ids_list)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

/// Gives `count_val` copies of `value_val`, as a CPU list.
static std::vector<float> all_of(size_t count_val, float value_val)
{
    return std::vector<float>(count_val, value_val);
}

// ══ part 1: mine against PyTorch's and against the 4-value path ═══════════════

static void forward_matches_pytorch(void)
{
    // Expect: a tiny GPT-2's encoder in bfloat16 (vocab 64, positions 16, 16 wide: 2 loads a row), batch_size = 4 rows
    // of seq_len = 8 random ids: encoded equals PyTorch's embedding(wte, ids) + wpe[0 .. 7] in float on the same
    // values, rounded to bfloat16 by PyTorch, to the last bit: each value is one float add of the same two numbers,
    // rounded once to the nearest.

    // 1. a tiny encoder and random ids; mine
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, 16});
    Tensor wpe_tensor = random_bf16_tensor({16, 16});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 63));
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, DType::BFloat16);
    encoder_bf16_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);

    // 2. PyTorch's, in float, rounded to bfloat16 at the end
    torch::Tensor pytorch_encoded_tensor = torch::embedding(pytorch_copy_of(wte_tensor), pytorch_copy_of(token_ids_tensor)) +
                                           pytorch_copy_of(wpe_tensor).narrow(0, 0, seq_len_val);

    // 3. the same bits
    ASSERT_EQ(floats_of(encoded_tensor), list_of(pytorch_encoded_tensor.to(torch::kBFloat16).to(torch::kFloat)));
}

static void backward_matches_pytorch(void)
{
    // Expect: the same tiny encoder, but the ids only from 0 .. 15, so most of them come up 2 or more times in the 32
    // and their wte_grad rows are sums. Random encoded_grad, backward from 0: wte_grad and wpe_grad within
    // BFLOAT16_TOLERANCE of PyTorch's float .grad after autograd's backward (wpe_grad adds the 4 rows in float and
    // rounds once; wte_grad rounds after each add onto a repeated token's row). wte_grad's rows 16 .. 63 (ids never
    // read) stay 0, and so do wpe_grad's rows 8 .. 15 (positions after seq_len).

    // 1. a tiny encoder, ids from 0 .. 15, random encoded_grad; mine
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, 16});
    Tensor wpe_tensor = random_bf16_tensor({16, 16});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
    Tensor encoded_grad_tensor = random_bf16_tensor({batch_size_val, seq_len_val, 16});
    Tensor wte_grad_tensor = Tensor::zeros({64, 16}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({16, 16}, DType::BFloat16);
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

    // 2. PyTorch's, in float on the same values, with autograd
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_wpe_tensor = pytorch_copy_of(wpe_tensor).requires_grad_(true);
    torch::Tensor pytorch_encoded_tensor =
        torch::embedding(pytorch_wte_tensor, pytorch_copy_of(token_ids_tensor)) + pytorch_wpe_tensor.narrow(0, 0, seq_len_val);
    pytorch_encoded_tensor.backward(pytorch_copy_of(encoded_grad_tensor));

    // 3. check; the rows never read stay 0
    ASSERT_NEAR(floats_of(wte_grad_tensor), list_of(pytorch_wte_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(wpe_grad_tensor), list_of(pytorch_wpe_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_EQ(floats_of(wte_grad_tensor.narrow(0, 16, 48)), all_of(48 * 16, 0.0f));
    ASSERT_EQ(floats_of(wpe_grad_tensor.narrow(0, 8, 8)), all_of(8 * 16, 0.0f));
}

static void the_same_bits_as_the_4_value_path_on_gpt2_small_size(void)
{
    // Expect: GPT-2 small's encoder in bfloat16 (wte {50304, 768}, wpe {1024, 768}), batch_size = 4 rows of seq_len =
    // 1024 random ids, random encoded_grad. 8 values a thread here, 4 in fp32/encoder.cuh's bfloat16 path: the same
    // adds in float, each rounded once as it is stored, so encoded and wpe_grad are the same to the last bit.
    // wte_grad within BFLOAT16_TOLERANCE: a token that comes up twice has its row's adds land in any order, on
    // either side, each add rounded to bfloat16.

    // 1. GPT-2 small's sizes, random values and ids
    torch::manual_seed(0);
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN},
                                     random_token_ids(GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_grad_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});

    // 2. both sides' forward and backward, from 0
    Tensor tensors_list[2][3];                               // [side][encoded, wte_grad, wpe_grad]: 0 here, 1 the 4-value path
    for (int side_index_val = 0; side_index_val < 2; side_index_val++)
    {
        tensors_list[side_index_val][0] = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
        tensors_list[side_index_val][1] = Tensor::zeros({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM}, DType::BFloat16);
        tensors_list[side_index_val][2] = Tensor::zeros({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    }
    encoder_bf16_forward(tensors_list[0][0], token_ids_tensor, wte_tensor, wpe_tensor);
    encoder_bf16_backward(tensors_list[0][1], tensors_list[0][2], encoded_grad_tensor, token_ids_tensor);
    encoder_forward(tensors_list[1][0], token_ids_tensor, wte_tensor, wpe_tensor);
    encoder_backward(tensors_list[1][1], tensors_list[1][2], encoded_grad_tensor, token_ids_tensor);

    // 3. check, on the GPU: encoded and wpe_grad the same bits, wte_grad within BFLOAT16_TOLERANCE of the largest value
    torch::Tensor wte_grad_tensor = pytorch_copy_of(tensors_list[0][1]);
    torch::Tensor other_wte_grad_tensor = pytorch_copy_of(tensors_list[1][1]);
    double wte_grad_difference_val =
        ((wte_grad_tensor - other_wte_grad_tensor).abs().max() / other_wte_grad_tensor.abs().max()).item<double>();
    print_test_name(__func__);
    printf("    wte_grad's largest difference from the 4-value path: %.2e of its largest value\n", wte_grad_difference_val);
    ASSERT_EQ(torch::equal(pytorch_copy_of(tensors_list[0][0]), pytorch_copy_of(tensors_list[1][0])), true);
    ASSERT_EQ(torch::equal(pytorch_copy_of(tensors_list[0][2]), pytorch_copy_of(tensors_list[1][2])), true);
    ASSERT_EQ(wte_grad_difference_val <= BFLOAT16_TOLERANCE, true);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the same wte and wpe, 8 wide (one load a row): wte's rows count up from 0, so each value shows
// which token row it came from; wpe's rows are 32s, 64s, ..., so each shows its position. Every sum stays at or below
// 256, where bfloat16 holds every whole number exactly: the results are exact.
//
//     wte   token 0:  0  1  2  3  4  5  6  7       wpe   position 0:  32 x 8
//           token 1:  8  9 10 11 12 13 14 15             position 1:  64 x 8
//           token 2: 16 17 18 19 20 21 22 23             position 2:  96 x 8
//                                                        position 3: 128 x 8

/// Makes the wte of part 2: 3 tokens, rows 0 .. 7 | 8 .. 15 | 16 .. 23.
static Tensor small_wte(void)
{
    return bf16_of({3, 8}, sample_data(24));
}

/// Makes the wpe of part 2: 4 positions, rows of 32s, 64s, 96s, 128s.
static Tensor small_wpe(void)
{
    std::vector<float> values_list;
    for (int position_index_val = 0; position_index_val < 4; position_index_val++)
        for (int value_index_val = 0; value_index_val < 8; value_index_val++)
            values_list.push_back(32.0f * (position_index_val + 1));
    return bf16_of({4, 8}, values_list);
}

static void forward_adds_the_tokens_row_and_the_positions_row(void)
{
    // Expect: ids {2, 0}: encoded row 0 is token 2's row + position 0's = 16 .. 23 + 32 = 48 .. 55; encoded row 1 is
    // token 0's + position 1's = 0 .. 7 + 64 = 64 .. 71.
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({1, 2, 8}, DType::BFloat16);
    encoder_bf16_forward(encoded_tensor, ids_of({1, 2}, {2, 0}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(floats_of(encoded_tensor), std::vector<float>({48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71}));
}

static void the_same_token_at_another_position_gets_another_row(void)
{
    // Expect: ids {1, 1, 1}, token 1 three times: 8 .. 15 plus 32, then 64, then 96. The position row is what tells
    // the model the 3 apart.
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({1, 3, 8}, DType::BFloat16);
    encoder_bf16_forward(encoded_tensor, ids_of({1, 3}, {1, 1, 1}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(floats_of(encoded_tensor), std::vector<float>({40, 41, 42, 43, 44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79,
                                                             104, 105, 106, 107, 108, 109, 110, 111}));
}

static void every_row_of_the_batch_starts_again_at_position_0(void)
{
    // Expect: batch_size = 2 rows, ids {{0, 1}, {2, 0}}. Each row of the batch is its own text, so its first token is
    // at position 0 again:
    //
    //     b 0:  token 0 + position 0 = 32 .. 39     token 1 + position 1 = 72 .. 79
    //     b 1:  token 2 + position 0 = 48 .. 55     token 0 + position 1 = 64 .. 71
    // 1. the forward
    Tensor encoded_tensor = Tensor::zeros({2, 2, 8}, DType::BFloat16);
    encoder_bf16_forward(encoded_tensor, ids_of({2, 2}, {0, 1, 2, 0}), small_wte(), small_wpe());
    // 2. check
    ASSERT_EQ(floats_of(encoded_tensor), std::vector<float>({32, 33, 34, 35, 36, 37, 38, 39, 72, 73, 74, 75, 76, 77, 78, 79,
                                                             48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71}));
}

static void backward_adds_a_repeated_tokens_rows_together(void)
{
    // Expect: ids {1, 1, 1} and encoded_grad rows of 1s, 2s, 3s. Token 1 made all 3 rows, so its wte_grad row gets
    // all 3: 1 + 2 + 3 = 6. Tokens 0 and 2 were not read: 0. Each position made one row, so wpe_grad's rows are 1, 2,
    // 3, and position 3 (after seq_len = 3) stays 0.
    // 1. the backward
    std::vector<float> grad_list;
    for (float value_val : {1.0f, 2.0f, 3.0f})
        for (int value_index_val = 0; value_index_val < 8; value_index_val++)
            grad_list.push_back(value_val);
    Tensor wte_grad_tensor = Tensor::zeros({3, 8}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({4, 8}, DType::BFloat16);
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, bf16_of({1, 3, 8}, grad_list), ids_of({1, 3}, {1, 1, 1}));
    // 2. check
    std::vector<float> expected_wte_grad_list = all_of(8, 0.0f);
    for (float value_val : {6.0f, 0.0f})
        for (int value_index_val = 0; value_index_val < 8; value_index_val++)
            expected_wte_grad_list.push_back(value_val);
    std::vector<float> expected_wpe_grad_list = grad_list;
    for (int value_index_val = 0; value_index_val < 8; value_index_val++)
        expected_wpe_grad_list.push_back(0.0f);
    ASSERT_EQ(floats_of(wte_grad_tensor), expected_wte_grad_list);
    ASSERT_EQ(floats_of(wpe_grad_tensor), expected_wpe_grad_list);
}

static void backward_adds_the_batch_rows_at_each_position(void)
{
    // Expect: batch_size = 2 rows, ids {{0, 1}, {2, 0}}, encoded_grad counting up from 0:
    //
    //     (b 0, t 0)   0 ..  7    token 0      (b 1, t 0)  16 .. 23    token 2
    //     (b 0, t 1)   8 .. 15    token 1      (b 1, t 1)  24 .. 31    token 0
    //
    //     wpe_grad position 0 = (b 0, t 0) + (b 1, t 0) = 16 18 20 ... 30
    //              position 1 = (b 0, t 1) + (b 1, t 1) = 32 34 36 ... 46      positions 2, 3: 0
    //     wte_grad token 0    = (b 0, t 0) + (b 1, t 1) = 24 26 28 ... 38
    //              token 1    = (b 0, t 1)              =  8  9 10 ... 15
    //              token 2    = (b 1, t 0)              = 16 17 18 ... 23
    // 1. the backward
    Tensor wte_grad_tensor = Tensor::zeros({3, 8}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({4, 8}, DType::BFloat16);
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, bf16_of({2, 2, 8}, sample_data(32)), ids_of({2, 2}, {0, 1, 2, 0}));
    // 2. check
    ASSERT_EQ(floats_of(wte_grad_tensor), std::vector<float>({24, 26, 28, 30, 32, 34, 36, 38, 8, 9, 10, 11, 12, 13, 14, 15,
                                                              16, 17, 18, 19, 20, 21, 22, 23}));
    std::vector<float> expected_wpe_grad_list = {16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46};
    for (int value_index_val = 0; value_index_val < 2 * 8; value_index_val++)
        expected_wpe_grad_list.push_back(0.0f);
    ASSERT_EQ(floats_of(wpe_grad_tensor), expected_wpe_grad_list);
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch had left it there). One token, id 0,
    // encoded_grad 1 .. 8: row 0 of both becomes 11 .. 18 and every other row stays 10. A second backward adds again:
    // 12 14 ... 26. Gradient accumulation relies on this.
    // 1. gradients that start at 10
    Tensor token_ids_tensor = ids_of({1, 1}, {0});
    Tensor encoded_grad_tensor = bf16_of({1, 1, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor wte_grad_tensor = Tensor::zeros({3, 8}, DType::BFloat16).fill_(10);
    Tensor wpe_grad_tensor = Tensor::zeros({4, 8}, DType::BFloat16).fill_(10);
    // 2. one backward: added onto
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    std::vector<float> once_list = {11, 12, 13, 14, 15, 16, 17, 18};
    std::vector<float> expected_wte_grad_list = once_list;
    std::vector<float> expected_wpe_grad_list = once_list;
    for (int value_index_val = 0; value_index_val < 2 * 8; value_index_val++)
        expected_wte_grad_list.push_back(10.0f);
    for (int value_index_val = 0; value_index_val < 3 * 8; value_index_val++)
        expected_wpe_grad_list.push_back(10.0f);
    ASSERT_EQ(floats_of(wte_grad_tensor), expected_wte_grad_list);
    ASSERT_EQ(floats_of(wpe_grad_tensor), expected_wpe_grad_list);
    // 3. a second backward: added again
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    ASSERT_EQ(floats_of(wte_grad_tensor[0]), std::vector<float>({12, 14, 16, 18, 20, 22, 24, 26}));
    ASSERT_EQ(floats_of(wpe_grad_tensor[0]), std::vector<float>({12, 14, 16, 18, 20, 22, 24, 26}));
}

static void many_threads_adding_onto_one_row_lose_nothing(void)
{
    // Expect: GPT-2 small's width, batch_size = 4 rows of seq_len = 64 ids, every id 5, encoded_grad all 1. The wte
    // backward takes 4 x 64 x 96 = 24,576 loads, one thread each at the default launch, and the 256 adds of each
    // column all go onto the same value of token 5's row, many at the same moment. With the compare-and-swap of
    // atomic_add4_from_float no add is lost: 256 x 1 = 256, exactly. (Not GPT-2's seq_len 1024: bfloat16 holds every
    // whole number only up to 256; 256 + 1 = 257 needs 9 significant bits and rounds back to 256.) Tokens 0 .. 4 stay
    // 0. Every position has 4 rows of 1s: wpe_grad is 4 everywhere.
    // 1. every id 5, encoded_grad all 1; the backward
    size_t batch_size_val = 4;
    size_t seq_len_val = 64;
    Tensor token_ids_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32).fill_(5);
    Tensor encoded_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, GPT2_EMBED_DIM}, DType::BFloat16).fill_(1);
    Tensor wte_grad_tensor = Tensor::zeros({6, GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({seq_len_val, GPT2_EMBED_DIM}, DType::BFloat16);
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    // 2. check
    ASSERT_EQ(floats_of(wte_grad_tensor[5]), all_of(GPT2_EMBED_DIM, 256.0f));
    ASSERT_EQ(floats_of(wte_grad_tensor.narrow(0, 0, 5)), all_of(5 * GPT2_EMBED_DIM, 0.0f));
    ASSERT_EQ(floats_of(wpe_grad_tensor), all_of(seq_len_val * GPT2_EMBED_DIM, 4.0f));
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_a_block_of_encoder_bf16_threads(void)
{
    // Expect: each of the 3 kernels can be launched with ENCODER_BF16_THREADS (1024) threads per block, and nothing is
    // spilled: a thread's 8 values (and the 16-byte load they are read into) stay in registers. cudaFuncGetAttributes
    // reads what the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with: 1024 if it fits
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)encoder_bf16_forward_kernel, (const void *)encoder_bf16_backward_wte_kernel,
                                  (const void *)encoder_bf16_backward_wpe_kernel};
    const char *names_list[] = {"encoder_bf16_forward_kernel", "encoder_bf16_backward_wte_kernel",
                                "encoder_bf16_backward_wpe_kernel"};
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
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= ENCODER_BF16_THREADS, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_result(void)
{
    // Expect: encoder_bf16_forward and encoder_bf16_backward called with threads_per_block 32, 128, 512 and 1024 (one
    // warp up to HARDWARE_MAX_THREADS_PER_BLOCK) give the same result as with the defaults, ENCODER_BF16_FORWARD_THREADS
    // and ENCODER_BF16_THREADS: encoded and wpe_grad to the last bit (each value worked out alone, wpe_grad's batch
    // rows added in order), wte_grad within BFLOAT16_TOLERANCE: its atomic adds onto a repeated token's row can land
    // in another order, as from run to run. The tiny encoder of part 1, ids from 0 .. 15 so most tokens repeat.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                    HARDWARE_MAX_THREADS_PER_BLOCK};

    // 1. a tiny encoder; the default launches' results
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, 16});
    Tensor wpe_tensor = random_bf16_tensor({16, 16});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
    Tensor encoded_grad_tensor = random_bf16_tensor({batch_size_val, seq_len_val, 16});
    Tensor usual_encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, DType::BFloat16);
    Tensor usual_wte_grad_tensor = Tensor::zeros({64, 16}, DType::BFloat16);
    Tensor usual_wpe_grad_tensor = Tensor::zeros({16, 16}, DType::BFloat16);
    encoder_bf16_forward(usual_encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    encoder_bf16_backward(usual_wte_grad_tensor, usual_wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

    // 2. each block size, against the defaults'
    for (int threads_per_block_val : block_sizes_list)
    {
        Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, 16}, DType::BFloat16);
        Tensor wte_grad_tensor = Tensor::zeros({64, 16}, DType::BFloat16);
        Tensor wpe_grad_tensor = Tensor::zeros({16, 16}, DType::BFloat16);
        encoder_bf16_forward(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor, threads_per_block_val);
        encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor, threads_per_block_val);
        ASSERT_EQ(floats_of(encoded_tensor), floats_of(usual_encoded_tensor));
        ASSERT_EQ(floats_of(wpe_grad_tensor), floats_of(usual_wpe_grad_tensor));
        ASSERT_NEAR(floats_of(wte_grad_tensor), floats_of(usual_wte_grad_tensor), BFLOAT16_TOLERANCE);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_matches_pytorch();
    backward_matches_pytorch();
    the_same_bits_as_the_4_value_path_on_gpt2_small_size();

    forward_adds_the_tokens_row_and_the_positions_row();
    the_same_token_at_another_position_gets_another_row();
    every_row_of_the_batch_starts_again_at_position_0();
    backward_adds_a_repeated_tokens_rows_together();
    backward_adds_the_batch_rows_at_each_position();
    backward_adds_onto_the_gradients_already_there();
    many_threads_adding_onto_one_row_lose_nothing();

    every_kernel_fits_a_block_of_encoder_bf16_threads();
    any_block_size_gives_the_same_result();

    printf("\nall tests passed\n");
    return 0;
}
