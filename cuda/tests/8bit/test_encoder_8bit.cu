// test_encoder_8bit.cu — tests for cuda/kernels/8bit/encoder_8bit.cuh (encoder_8bit_forward, encoder_8bit_backward):
// GPT-2's first layer for a bfloat16 model with wte read from its 8-bit copy (FP8 E4M3, one scale per row). Every
// check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the forward against PyTorch on the copy turned back (float8_e4m3fn x scale), the copy against
// round_8bit's, the forward against the bfloat16 encoder within 8-bit rounding, and the backward against the bfloat16
// encoder's; part 2 checks numbers worked out by hand; part 3 checks that the gather's registers fit a block of
// ENCODER_8BIT_THREADS, and that any block size gives the same bits. Any GPU. The timings are in
// cuda/tests/8bit/tune_encoder_8bit.cu.
#include <torch/types.h>              // torch::embedding, torch::randn, float8_e4m3fn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include "../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../../kernels/8bit/encoder_8bit.cuh"

// Against the bfloat16 encoder: each E4M3 is off by up to 2^-4 (6%) of its value, so encoded moves by a few percent
// of wte's values. Against PyTorch in float: a repeated token's wte_grad row is rounded once per add. Each tolerance
// is the largest difference, as a fraction of the largest value.
static const double EIGHT_BIT_TOLERANCE = 1e-1;
static const double BFLOAT16_TOLERANCE = 2e-2;

#define GPT2_HIGHEST_TOKEN_ID 50256   // GPT-2's real ids: 0 .. 50256
#define GPT2_PADDED_VOCAB_SIZE 50304  // GPT-2 small's sizes
#define GPT2_MAX_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_BATCH_SIZE 4
#define SMALL_EMBED_DIM ROUND_8BIT_ROW_LENGTH_MULTIPLE   // the narrowest wte round_8bit takes: 16

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

/// wte's 8-bit copy, the tensors encoder_8bit_forward writes it into.
struct WteCopy
{
    // {vocab_size, embed_dim} E4M3 and {vocab_size} float, made for a wte of `vocab_size_val` rows of
    // `embed_dim_val`.
    Tensor wte_8bit_tensor;
    Tensor wte_scales_tensor;
    size_t vocab_size_val;
    size_t embed_dim_val;

    WteCopy(size_t vocab_size, size_t embed_dim)
        : wte_8bit_tensor(Tensor::zeros({vocab_size, embed_dim}, DType::E4M3)),
          wte_scales_tensor(Tensor::zeros({vocab_size})), vocab_size_val(vocab_size), embed_dim_val(embed_dim)
    {
    }

    /// Gives the copy turned back by PyTorch: each E4M3 times its row's scale, exact.
    torch::Tensor turned_back() const
    {
        // Gives the copy turned back by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times its row's scale,
        // exact. {vocab_size, embed_dim} float.
        torch::Tensor e4m3_tensor = torch::from_blob(wte_8bit_tensor.data_ptr(), {(int64_t)vocab_size_val, (int64_t)embed_dim_val},
                                                     torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(wte_scales_tensor).unsqueeze(1);
    }
};

// ══ part 1: mine against PyTorch's, round_8bit's and the bfloat16 encoder's ═══

static void forward_matches_pytorch_on_the_turned_back_copy(void)
{
    // Expect: a tiny GPT-2's encoder (vocab 64, positions 16, SMALL_EMBED_DIM = 16 wide), batch_size = 4 rows of
    // seq_len = 8 random ids. encoded equals PyTorch's embedding(the copy turned back, ids) + wpe[0 .. 7] in float,
    // rounded to bfloat16 by PyTorch, to the last bit: turning an E4M3 back (x a power of 2) is exact, then each
    // value is one float add of the same two numbers, rounded once to the nearest.

    // 1. a tiny encoder and random ids; mine
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, SMALL_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({16, SMALL_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 63));
    Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, SMALL_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(64, SMALL_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, token_ids_tensor, wte_tensor,
                         wpe_tensor);

    // 2. PyTorch's on the copy turned back, in float, rounded to bfloat16 at the end
    torch::Tensor pytorch_encoded_tensor = torch::embedding(copy_val.turned_back(), pytorch_copy_of(token_ids_tensor)) +
                                           pytorch_copy_of(wpe_tensor).narrow(0, 0, seq_len_val);

    // 3. the same bits
    ASSERT_EQ(floats_of(encoded_tensor), list_of(pytorch_encoded_tensor.to(torch::kBFloat16).to(torch::kFloat)));
}

static void the_copy_is_round_8bit_s(void)
{
    // Expect: GPT-2 small's wte {50304, 768}, random; one forward on 4 rows of 1024 random ids. The copy it leaves is
    // round_8bit's on all of wte, every row (also the ones no id read), to the last bit: the copy the head can take
    // in place of rounding wte again.

    // 1. GPT-2 small's sizes; mine
    torch::manual_seed(0);
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN},
                                     random_token_ids(GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, token_ids_tensor, wte_tensor,
                         wpe_tensor);

    // 2. round_8bit's own
    WteCopy rounded_val(GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM);
    round_8bit(rounded_val.wte_8bit_tensor, rounded_val.wte_scales_tensor, wte_tensor);

    // 3. the same bits
    ASSERT_EQ(copy_val.wte_8bit_tensor.equal(rounded_val.wte_8bit_tensor), true);
    ASSERT_EQ(copy_val.wte_scales_tensor.tolist(), rounded_val.wte_scales_tensor.tolist());
}

static void forward_is_within_8_bit_rounding_of_the_bfloat16_encoder_on_gpt2_small_size(void)
{
    // Expect: GPT-2 small's encoder, random wte and wpe, batch_size = 4 rows of seq_len = 1024 random ids. encoded
    // within EIGHT_BIT_TOLERANCE of encoder_bf16_forward's: only wte's values moved, each by up to 2^-4 (6%) of
    // itself. The difference is printed.

    // 1. GPT-2 small's sizes; mine and the bfloat16 encoder's
    torch::manual_seed(0);
    Tensor wte_tensor = random_bf16_tensor({GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN},
                                     random_token_ids(GPT2_BATCH_SIZE * GPT2_MAX_SEQ_LEN, GPT2_HIGHEST_TOKEN_ID));
    Tensor encoded_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor bf16_encoded_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_MAX_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, token_ids_tensor, wte_tensor,
                         wpe_tensor);
    encoder_bf16_forward(bf16_encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);

    // 2. the largest difference, on the GPU, as a fraction of the bfloat16 encoder's largest value
    torch::Tensor bf16_tensor = pytorch_copy_of(bf16_encoded_tensor);
    double difference_val = ((pytorch_copy_of(encoded_tensor) - bf16_tensor).abs().max() / bf16_tensor.abs().max()).item<double>();
    print_test_name(__func__);
    printf("    encoded's largest difference from the bfloat16 encoder: %.2e of its largest value\n", difference_val);

    // 3. check
    ASSERT_EQ(difference_val <= EIGHT_BIT_TOLERANCE, true);
}

static void backward_is_the_bfloat16_encoder_s(void)
{
    // Expect: the tiny encoder of forward_matches_pytorch_on_the_turned_back_copy, ids from 0 .. 15 so most repeat,
    // random encoded_grad, backward from 0. The forward's rounding passes the gradient through (straight-through), so
    // the backward is encoder_bf16_backward's: wpe_grad the same bits, wte_grad within BFLOAT16_TOLERANCE (its atomic
    // adds onto a repeated token's row land in any order). Both within BFLOAT16_TOLERANCE of PyTorch's float .grad
    // of embedding(wte, ids) + wpe, the rounding left out.

    // 1. a tiny encoder; mine and the bfloat16 encoder's
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, SMALL_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({16, SMALL_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 15));
    Tensor encoded_grad_tensor = random_bf16_tensor({batch_size_val, seq_len_val, SMALL_EMBED_DIM});
    Tensor wte_grad_tensor = Tensor::zeros({64, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor wpe_grad_tensor = Tensor::zeros({16, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor bf16_wte_grad_tensor = Tensor::zeros({64, SMALL_EMBED_DIM}, DType::BFloat16);
    Tensor bf16_wpe_grad_tensor = Tensor::zeros({16, SMALL_EMBED_DIM}, DType::BFloat16);
    encoder_8bit_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);
    encoder_bf16_backward(bf16_wte_grad_tensor, bf16_wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor);

    // 2. PyTorch's, in float on the same values, with autograd
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_wpe_tensor = pytorch_copy_of(wpe_tensor).requires_grad_(true);
    torch::Tensor pytorch_encoded_tensor =
        torch::embedding(pytorch_wte_tensor, pytorch_copy_of(token_ids_tensor)) + pytorch_wpe_tensor.narrow(0, 0, seq_len_val);
    pytorch_encoded_tensor.backward(pytorch_copy_of(encoded_grad_tensor));

    // 3. check
    ASSERT_EQ(floats_of(wpe_grad_tensor), floats_of(bf16_wpe_grad_tensor));
    ASSERT_NEAR(floats_of(wte_grad_tensor), floats_of(bf16_wte_grad_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(wte_grad_tensor), list_of(pytorch_wte_tensor.grad()), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(wpe_grad_tensor), list_of(pytorch_wpe_tensor.grad()), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// E4M3 keeps 4 significant bits (1 + 3 stored), so between 16 and 32 it steps by 2 and between 128 and 256 by 16.
// A row's scale is the smallest power of 2 with largest |value| / scale <= 448.

static void forward_reads_whole_numbers_up_to_15_exactly(void)
{
    // Expect: wte {3, 16}: token 0's row 0 .. 15, token 1's 15 .. 0, token 2's all 7; wpe rows of 32s and 64s. Whole
    // numbers up to 15 have at most 4 significant bits, so E4M3 holds them over any power of 2:
    //
    //     token   largest   scale                      stored largest
    //     0, 1    15        2^-4 (15 / 2^-5 = 480 > 448)   15 x 16 = 240
    //     2       7         2^-6 (7 / 448 = 2^-6 itself)   7 x 64 = 448, E4M3's largest
    //
    // ids {2, 1}: encoded row 0 = 7 + 32 = 39 everywhere; row 1 = 15 .. 0 + 64 = 79 .. 64. Exactly the bfloat16
    // encoder's.
    // 1. the forward
    std::vector<float> wte_list = sample_data(16);
    for (int value_index_val = 15; value_index_val >= 0; value_index_val--)
        wte_list.push_back((float)value_index_val);
    for (int value_index_val = 0; value_index_val < 16; value_index_val++)
        wte_list.push_back(7.0f);
    std::vector<float> wpe_list(16, 32.0f);
    for (int value_index_val = 0; value_index_val < 16; value_index_val++)
        wpe_list.push_back(64.0f);
    Tensor encoded_tensor = Tensor::zeros({1, 2, SMALL_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(3, SMALL_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, ids_of({1, 2}, {2, 1}),
                         bf16_of({3, SMALL_EMBED_DIM}, wte_list), bf16_of({2, SMALL_EMBED_DIM}, wpe_list));
    // 2. check: the scales, encoded
    std::vector<float> expected_list(16, 39.0f);
    for (int value_index_val = 15; value_index_val >= 0; value_index_val--)
        expected_list.push_back(64.0f + value_index_val);
    ASSERT_EQ(copy_val.wte_scales_tensor.tolist(), std::vector<float>({0.0625f, 0.0625f, 0.015625f}));
    ASSERT_EQ(floats_of(encoded_tensor), expected_list);
}

static void forward_rounds_a_value_e4m3_cannot_hold(void)
{
    // Expect: wte {2, 16}: token 0's row 256, 17, then 0s; token 1's row 1000, then 0s; wpe all 0, so encoded shows
    // the copy itself. ids {0, 1}:
    //
    //     token 0: largest 256, scale 1 (256 / 448 < 1). 17 sits halfway between 16 and 18, E4M3's neighbours there;
    //              a tie goes to the even last bit: 16. 256 stays.
    //     token 1: largest 1000, scale 4 (1000 / 2 = 500 > 448). 1000 / 4 = 250 sits between 240 and 256, nearer 256:
    //              256 x 4 = 1024.
    //
    // encoded row 0 = 256, 16, 0 ...; row 1 = 1024, 0 .... The bfloat16 encoder would give 17 and 1000: this is the
    // 8 bits' cost.
    // 1. the forward
    std::vector<float> wte_list(2 * SMALL_EMBED_DIM, 0.0f);
    wte_list[0] = 256.0f;
    wte_list[1] = 17.0f;
    wte_list[SMALL_EMBED_DIM] = 1000.0f;
    Tensor encoded_tensor = Tensor::zeros({1, 2, SMALL_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(2, SMALL_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, ids_of({1, 2}, {0, 1}),
                         bf16_of({2, SMALL_EMBED_DIM}, wte_list), Tensor::zeros({2, SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. check: the scales, encoded
    std::vector<float> expected_list(2 * SMALL_EMBED_DIM, 0.0f);
    expected_list[0] = 256.0f;
    expected_list[1] = 16.0f;
    expected_list[SMALL_EMBED_DIM] = 1024.0f;
    ASSERT_EQ(copy_val.wte_scales_tensor.tolist(), std::vector<float>({1.0f, 4.0f}));
    ASSERT_EQ(floats_of(encoded_tensor), expected_list);
}

static void the_copy_holds_every_row_of_wte_not_only_the_ones_read(void)
{
    // Expect: wte {3, 16} counting up by halves, 0, 0.5, ..., 23.5 (some, like 23.5, need more than E4M3's 4
    // significant bits, so the copy is checked against PyTorch's rounding, not against wte). One id, 0: the gather
    // reads only row 0, but the copy has all 3 rows, as the head needs: turned back, every row equals PyTorch's
    // float8_e4m3fn rounding of wte over its scale, times the scale, to the last bit. Rows 1 and 2 are not 0.
    // 1. the forward on one id
    std::vector<float> wte_list = sample_data(3 * SMALL_EMBED_DIM);
    for (float &value_val : wte_list)
        value_val *= 0.5f;
    Tensor wte_tensor = bf16_of({3, SMALL_EMBED_DIM}, wte_list);
    Tensor encoded_tensor = Tensor::zeros({1, 1, SMALL_EMBED_DIM}, DType::BFloat16);
    WteCopy copy_val(3, SMALL_EMBED_DIM);
    encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, ids_of({1, 1}, {0}), wte_tensor,
                         Tensor::zeros({1, SMALL_EMBED_DIM}, DType::BFloat16));
    // 2. PyTorch's rounding of every row with my scales
    torch::Tensor scales_tensor = pytorch_copy_of(copy_val.wte_scales_tensor).unsqueeze(1);
    torch::Tensor expected_tensor =
        (pytorch_copy_of(wte_tensor) / scales_tensor).to(torch::kFloat8_e4m3fn).to(torch::kFloat) * scales_tensor;
    // 3. check: every row, the same bits; rows 1 and 2 hold values
    torch::Tensor turned_back_tensor = copy_val.turned_back();
    ASSERT_EQ(list_of(turned_back_tensor), list_of(expected_tensor));
    ASSERT_EQ(turned_back_tensor.narrow(0, 1, 2).abs().sum().item<float>() > 0.0f, true);
}

// ══ part 3: the gather fits its launch, and any block size works ═════════════

static void the_gather_fits_a_block_of_encoder_8bit_threads(void)
{
    // Expect: encoder_8bit_forward_kernel can be launched with ENCODER_8BIT_THREADS (1024) threads per block, and
    // nothing is spilled: a thread's 8 values stay in registers. (The rounding is round_8bit's, tested in
    // cuda/tests/8bit/test_round_8bit.cu; the backward is encoder_bf16's, in cuda/tests/bf16/test_encoder_bf16.cu.)
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with: 1024 if it fits
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    // 1. what the compiler wrote down
    cudaFuncAttributes attributes_val;
    cudaCheck(cudaFuncGetAttributes(&attributes_val, (const void *)encoder_8bit_forward_kernel));
    print_test_name(__func__);
    printf("    encoder_8bit_forward_kernel: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
           attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
    // 2. check
    ASSERT_EQ(attributes_val.maxThreadsPerBlock >= ENCODER_8BIT_THREADS, true);
    ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: encoder_8bit_forward called with threads_per_block 32, 128, 512 and 1024 (one warp up to
    // HARDWARE_MAX_THREADS_PER_BLOCK) gives the same encoded and the same copy as with the default,
    // ENCODER_8BIT_FORWARD_THREADS, to the last bit: each value is worked out alone. The tiny encoder of part 1.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                    HARDWARE_MAX_THREADS_PER_BLOCK};

    // 1. a tiny encoder; the default launch's results
    torch::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 8;
    Tensor wte_tensor = random_bf16_tensor({64, SMALL_EMBED_DIM});
    Tensor wpe_tensor = random_bf16_tensor({16, SMALL_EMBED_DIM});
    Tensor token_ids_tensor = ids_of({batch_size_val, seq_len_val}, random_token_ids(batch_size_val * seq_len_val, 63));
    Tensor usual_encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, SMALL_EMBED_DIM}, DType::BFloat16);
    WteCopy usual_copy_val(64, SMALL_EMBED_DIM);
    encoder_8bit_forward(usual_encoded_tensor, usual_copy_val.wte_8bit_tensor, usual_copy_val.wte_scales_tensor, token_ids_tensor,
                         wte_tensor, wpe_tensor);

    // 2. each block size, against the default's
    for (int threads_per_block_val : block_sizes_list)
    {
        Tensor encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, SMALL_EMBED_DIM}, DType::BFloat16);
        WteCopy copy_val(64, SMALL_EMBED_DIM);
        encoder_8bit_forward(encoded_tensor, copy_val.wte_8bit_tensor, copy_val.wte_scales_tensor, token_ids_tensor, wte_tensor,
                             wpe_tensor, threads_per_block_val);
        ASSERT_EQ(floats_of(encoded_tensor), floats_of(usual_encoded_tensor));
        ASSERT_EQ(copy_val.wte_8bit_tensor.equal(usual_copy_val.wte_8bit_tensor), true);
        ASSERT_EQ(copy_val.wte_scales_tensor.tolist(), usual_copy_val.wte_scales_tensor.tolist());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_matches_pytorch_on_the_turned_back_copy();
    the_copy_is_round_8bit_s();
    forward_is_within_8_bit_rounding_of_the_bfloat16_encoder_on_gpt2_small_size();
    backward_is_the_bfloat16_encoder_s();

    forward_reads_whole_numbers_up_to_15_exactly();
    forward_rounds_a_value_e4m3_cannot_hold();
    the_copy_holds_every_row_of_wte_not_only_the_ones_read();

    the_gather_fits_a_block_of_encoder_8bit_threads();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
