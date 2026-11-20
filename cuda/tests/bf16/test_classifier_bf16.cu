// test_classifier_bf16.cu — tests for cuda/kernels/bf16/classifier_bf16.cuh (classifier_bf16_forward,
// classifier_bf16_backward): GPT-2's head (logits = lnf_output · wteᵀ on an H100's tensor cores, WGMMA fed by TMA)
// with the softmax's row sums worked out as each tile of logits is stored, then the loss; the backward's logits_grad
// and the head's two products. Every check prints its test case, input, expected and actual value, and ok; the
// first FAIL stops the program.
//
// Part 1 checks against PyTorch's float head and cross_entropy on the same bfloat16 values, with autograd, and
// against the unfused path (linear_bf16_forward's logits, to the bit, then fp32/classifier.cuh's softmax on them);
// part 2 checks numbers worked out by hand; part 3 checks that each kernel's registers fit its block, and other
// block sizes. Run it on an H100 (compute capability 9.0), built with -arch=sm_90a in place of -arch=native. The
// timings are in cuda/tests/bf16/tune_classifier_bf16.cu.
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <cmath>                      // std::log, std::exp: the hand-worked losses
#include "../common/test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../kernels/bf16/classifier_bf16.cuh"
#include "../../kernels/fp32/classifier.cuh"      // the unfused softmax, to compare with

// Against PyTorch in float on the same bfloat16 lnf_output and wte: each logit is rounded to bfloat16 as it is
// stored, by up to 2^-8 of it, and the softmax works from the rounded logits; the backward rounds logits_grad and
// the two gradients on top. Each tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;
// Against fp32/classifier.cuh on the same stored logits: the same math in float, each row added up in another
// order.
static const double SAME_LOGITS_TOLERANCE = 1e-5;
static const float WTE_STD = 0.02f;   // GPT-2's starting std: logits of about 0.02 x sqrt(embed_dim), as in training

#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_VOCAB_SIZE 50257
#define GPT2_PADDED_VOCAB_SIZE 50304

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation `std_val`, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
///   std_val      the standard deviation; 1 if not given
static Tensor random_bf16_tensor(const Shape &shape_list, float std_val = 1.0f)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * std_val;
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded.
static Tensor bf16_tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Makes an int tensor of `shape_list` on the GPU holding `values_list`: the targets.
static Tensor targets_of(const Shape &shape_list, const std::vector<int> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes an int tensor of `shape_list` with random targets 0 .. vocab_size - 1, made by PyTorch on the GPU.
static Tensor random_targets_on_gpu(const Shape &shape_list, size_t vocab_size_val)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    torch::Tensor values_tensor = torch::randint((int64_t)vocab_size_val, {(int64_t)tensor.numel()}, ints_on_gpu_val);
    cudaCheck(cudaMemcpy(tensor.data_ptr<int>(), values_tensor.data_ptr<int>(), tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Gives a tensor's values as a CPU list of floats, exactly (a bfloat16 tensor's turned into floats first).
static std::vector<float> values_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU. A bfloat16 tensor's values are turned into floats first, exactly.
    Tensor floats_tensor = source_tensor.to(DType::Float32);  // the source_tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets_tensor)
{
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(targets_tensor.shape().begin(), targets_tensor.shape().end());
    return torch::from_blob(targets_tensor.data_ptr<int>(), sizes_list, ints_on_gpu_val).to(torch::kLong);
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
///
///   mine_tensor      my result
///   pytorch_tensor   PyTorch's, the same shape
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    // Worked out on the GPU: GPT-2's logits are 3.2M values a row of 64 tokens.
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor classifier_bf16_forward and _backward take, for one size.
struct ClassifierTensors
{
    // Made with 0s, lnf_output, wte and targets filled by the test; losses_grad 1 / rows in every row (the mean
    // loss).
    size_t vocab_size_val;
    // forward
    Tensor lnf_output_tensor;
    Tensor wte_tensor;
    Tensor targets_tensor;
    Tensor logits_tensor;
    Tensor losses_tensor;
    Tensor max_of_each_row_tensor;
    Tensor sum_of_each_row_tensor;
    Tensor tile_max_tensor;
    Tensor tile_sum_tensor;
    // backward
    Tensor losses_grad_tensor;
    Tensor logits_grad_tensor;
    Tensor lnf_output_grad_tensor;
    Tensor wte_grad_tensor;

    /// Makes them for batch_size x seq_len rows, embed_dim, and vocab_size words padded to padded_vocab_size.
    ClassifierTensors(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val, size_t vocab_size, size_t padded_vocab_size_val)
        : vocab_size_val(vocab_size)
    {
        Shape per_row_list = {batch_size_val, seq_len_val};
        lnf_output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        wte_tensor = Tensor::zeros({padded_vocab_size_val, embed_dim_val}, DType::BFloat16);
        targets_tensor = Tensor::zeros(per_row_list, DType::Int32);
        logits_tensor = Tensor::zeros({batch_size_val, seq_len_val, padded_vocab_size_val}, DType::BFloat16);
        losses_tensor = Tensor::zeros(per_row_list);
        max_of_each_row_tensor = Tensor::zeros(per_row_list);
        sum_of_each_row_tensor = Tensor::zeros(per_row_list);
        tile_max_tensor = Tensor::zeros({batch_size_val, seq_len_val, classifier_bf16_tile_columns(padded_vocab_size_val)});
        tile_sum_tensor = Tensor::zeros(tile_max_tensor.shape());
        losses_grad_tensor = Tensor::zeros(per_row_list).fill_(1.0 / (batch_size_val * seq_len_val));
        logits_grad_tensor = Tensor::zeros(logits_tensor.shape(), DType::BFloat16);
        lnf_output_grad_tensor = Tensor::zeros(lnf_output_tensor.shape(), DType::BFloat16);
        wte_grad_tensor = Tensor::zeros(wte_tensor.shape(), DType::BFloat16);
    }

    /// Runs classifier_bf16_forward at a loss block size and logits tile order.
    ///
    ///   threads_per_block_val     the loss kernel's block size; the default if not given
    ///   tile_rows_per_group_val   the logits kernel's tile order; the default if not given
    void forward(int threads_per_block_val = CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK,
                 int tile_rows_per_group_val = CLASSIFIER_BF16_TILE_ROWS_PER_GROUP)
    {
        classifier_bf16_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, tile_max_tensor, tile_sum_tensor,
                                logits_tensor, lnf_output_tensor, wte_tensor, targets_tensor, vocab_size_val, threads_per_block_val,
                                tile_rows_per_group_val);
    }

    /// Runs classifier_bf16_backward into logits_grad, logits_grad at this block size (the default if not given).
    void backward(int threads_per_block_val = CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK)
    {
        classifier_bf16_backward(lnf_output_grad_tensor, wte_grad_tensor, logits_grad_tensor, losses_grad_tensor, logits_tensor,
                                 lnf_output_tensor, wte_tensor, targets_tensor, max_of_each_row_tensor, sum_of_each_row_tensor,
                                 vocab_size_val, threads_per_block_val);
    }
};

// ══ part 1: mine against PyTorch's, and against the unfused path ═══════════════

/// Runs one forward and one backward on random values and checks them against PyTorch and the reference kernels.
///
///   test_ptr                the test's name, printed with the sizes
///   batch_size_val          rows of the batch
///   seq_len_val             tokens a row
///   embed_dim_val           the model's width
///   vocab_size_val          real columns of the logits
///   padded_vocab_size_val   the logits' columns
static void forward_and_backward_match_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val,
                                               size_t embed_dim_val, size_t vocab_size_val, size_t padded_vocab_size_val)
{
    // Runs one forward and one backward on random bfloat16 lnf_output (std 1) and wte (std WTE_STD) and random
    // targets, then checks: logits, losses, row_max, row_sum against PyTorch's float head and cross_entropy over the
    // first vocab_size columns within BFLOAT16_TOLERANCE, lnf_output_grad and wte_grad against its autograd within
    // BFLOAT16_GRAD_TOLERANCE; logits equal to linear_bf16_forward's to the bit, and losses, row_max and row_sum
    // within SAME_LOGITS_TOLERANCE of fp32/classifier.cuh's on those logits. Prints the sizes under `test_ptr`'s name.
    // 1. random values; mine
    torch::manual_seed(0);
    size_t rows_val = batch_size_val * seq_len_val;
    ClassifierTensors mine_val(batch_size_val, seq_len_val, embed_dim_val, vocab_size_val, padded_vocab_size_val);
    mine_val.lnf_output_tensor = random_bf16_tensor(mine_val.lnf_output_tensor.shape());
    mine_val.wte_tensor = random_bf16_tensor(mine_val.wte_tensor.shape(), WTE_STD);
    mine_val.targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    mine_val.forward();
    mine_val.backward();

    // 2. the unfused path on the same values: the head by linear_bf16_forward, then fp32/classifier.cuh's softmax
    Tensor unfused_logits_tensor = Tensor::zeros(mine_val.logits_tensor.shape(), DType::BFloat16);
    linear_bf16_forward(unfused_logits_tensor, mine_val.lnf_output_tensor, mine_val.wte_tensor, Tensor());
    Tensor unfused_losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor unfused_max_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor unfused_sum_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    classifier_forward(unfused_losses_tensor, unfused_max_tensor, unfused_sum_tensor, unfused_logits_tensor,
                       mine_val.targets_tensor, vocab_size_val);

    // 3. PyTorch's float head and cross_entropy over the words, with autograd; its row maxes and sums
    torch::Tensor pytorch_lnf_output_tensor = pytorch_copy_of(mine_val.lnf_output_tensor).reshape({(int64_t)rows_val, -1}).requires_grad_(true);
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(mine_val.wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_logits_tensor = pytorch_lnf_output_tensor.matmul(pytorch_wte_tensor.t());
    torch::Tensor word_logits_tensor = pytorch_logits_tensor.narrow(1, 0, (int64_t)vocab_size_val);
    torch::Tensor pytorch_losses_tensor = torch::cross_entropy_loss(
        word_logits_tensor, pytorch_targets_of(mine_val.targets_tensor).reshape({-1}), {}, at::Reduction::None);
    pytorch_losses_tensor.backward(pytorch_copy_of(mine_val.losses_grad_tensor).reshape({-1}));
    torch::Tensor pytorch_max_tensor = std::get<0>(word_logits_tensor.max(1)).detach();
    torch::Tensor pytorch_sum_tensor = (word_logits_tensor - pytorch_max_tensor.unsqueeze(1)).exp().sum(1).detach();

    // 4. print the sizes, then check: against PyTorch, then against the unfused path
    print_test_name(test_ptr);
    printf("    %zu rows, embed_dim %zu, vocab_size %zu padded to %zu (%zu tiles across)\n", rows_val, embed_dim_val,
           vocab_size_val, padded_vocab_size_val, classifier_bf16_tile_columns(padded_vocab_size_val));
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.logits_tensor, pytorch_logits_tensor.detach()), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.losses_tensor, pytorch_losses_tensor.detach()), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.max_of_each_row_tensor, pytorch_max_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.sum_of_each_row_tensor, pytorch_sum_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.lnf_output_grad_tensor, pytorch_lnf_output_tensor.grad()), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.wte_grad_tensor, pytorch_wte_tensor.grad()), 0.0, BFLOAT16_GRAD_TOLERANCE);
    ASSERT_EQ(values_of(mine_val.logits_tensor), values_of(unfused_logits_tensor));
    ASSERT_NEAR(mine_val.losses_tensor.tolist(), unfused_losses_tensor.tolist(), SAME_LOGITS_TOLERANCE);
    ASSERT_NEAR(mine_val.max_of_each_row_tensor.tolist(), unfused_max_tensor.tolist(), SAME_LOGITS_TOLERANCE);
    ASSERT_NEAR(mine_val.sum_of_each_row_tensor.tolist(), unfused_sum_tensor.tolist(), SAME_LOGITS_TOLERANCE);
}

static void matches_pytorch_on_gpt2_sizes(void)
{
    // Expect: 64 rows of GPT-2 small: embed_dim 768, vocab_size 50257 padded to 50304, 786 tiles across, the last
    // with 17 words and 47 padding columns. Within the tolerances of PyTorch, and of the unfused path.
    forward_and_backward_match_pytorch(__func__, 1, 64, GPT2_EMBED_DIM, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
}

static void matches_pytorch_on_a_part_full_last_tile(void)
{
    // Expect: 2 x 50 = 100 rows (a tile of 64 down, then one of 36), embed_dim 96 (a step of 64, then 32: TMA
    // gives 0s past the edge), vocab_size 1000 with no padding: 16 tiles across, the last 40 columns wide, its
    // other 24 past the row's end, never stored. The same checks.
    forward_and_backward_match_pytorch(__func__, 2, 50, 96, 1000, 1000);
}

static void matches_pytorch_with_a_tile_of_padding_only(void)
{
    // Expect: 128 rows, embed_dim 768, vocab_size 900 padded to 1024: tile 14 (columns 896..959) holds 4 words and
    // 60 padding columns, tile 15 (960..1023) only padding, so its rows' (max, sum) are (-FLT_MAX, 0) and add
    // nothing to the row. The same checks; the padding's logits_grad is 0.
    forward_and_backward_match_pytorch(__func__, 2, 64, GPT2_EMBED_DIM, 900, 1024);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════

static void equal_logits_give_log_of_the_vocabulary_size(void)
{
    // Expect: lnf_output all 0, so every logit is 0 whatever wte holds: 64 rows, vocab_size 1000 padded to 1024.
    // row_max 0; row_sum = 1000 x exp(0) = 1000, exactly (whole numbers in float); loss = log(1000) = 6.9077553
    // whatever the target; logits_grad = 1/1000 x losses_grad (1/64) at every word, minus losses_grad at the target,
    // and 0 at the 24 padding columns; lnf_output_grad = logits_grad · wte with wte all 1: each row's
    // logits_grad adds up to 0 (the probs to 1, less the 1 at the target), so 0 up to rounding.
    // 1. the forward and the backward
    ClassifierTensors mine_val(1, 64, 64, 1000, 1024);
    mine_val.wte_tensor.fill_(1.0);
    std::vector<int> targets_list(64);
    for (int row_index_val = 0; row_index_val < 64; row_index_val++)
        targets_list[row_index_val] = row_index_val * 15;                  // 0, 15, ..., 945: spread over the words
    mine_val.targets_tensor = targets_of({1, 64}, targets_list);
    mine_val.forward();
    mine_val.backward();
    // 2. check the forward
    ASSERT_EQ(mine_val.max_of_each_row_tensor.tolist(), std::vector<float>(64, 0.0f));
    ASSERT_EQ(mine_val.sum_of_each_row_tensor.tolist(), std::vector<float>(64, 1000.0f));
    ASSERT_NEAR(mine_val.losses_tensor.tolist(), std::vector<float>(64, 6.9077553f), 1e-6);
    // 3. check logits_grad: row 0's target 0, a word, and a padding column; every row's padding 0
    std::vector<float> logits_grad_list = values_of(mine_val.logits_grad_tensor);
    std::vector<float> expected_row_0_list = {(1.0f / 1000 - 1.0f) / 64, 1.0f / 1000 / 64, 0.0f};
    std::vector<float> row_0_list = {logits_grad_list[0], logits_grad_list[1], logits_grad_list[1000]};
    ASSERT_NEAR(row_0_list, expected_row_0_list, 1e-2);                    // bfloat16: one rounding, 2^-8 of each
    size_t padding_values_val = 0;
    for (size_t row_index_val = 0; row_index_val < 64; row_index_val++)
        for (size_t column_index_val = 1000; column_index_val < 1024; column_index_val++)
            padding_values_val += logits_grad_list[row_index_val * 1024 + column_index_val] != 0.0f;
    ASSERT_EQ(padding_values_val, (size_t)0);
    ASSERT_NEAR(values_of(mine_val.lnf_output_grad_tensor), std::vector<float>(64 * 64, 0.0f), 1e-3);
}

static void each_row_sees_its_own_word_and_target(void)
{
    // Expect: 64 rows, embed_dim 64, vocab_size 64 (one tile): lnf_output row r = 1 at column r, 0 elsewhere (the
    // identity), wte = 4 x the identity, so logit[r][v] = 4 if v = r, else 0. Row r's target is r (the even rows,
    // the sure word) or r + 1 (the odd rows, a word at 0). row_max 4; row_sum = exp(0) + 63 exp(-4) = 2.1538...;
    // loss = log(row_sum) for the sure word, log(row_sum) + 4 for the other. All worked out in double.
    // 1. the forward
    ClassifierTensors mine_val(1, 64, 64, 64, 64);
    std::vector<float> identity_list(64 * 64, 0.0f), wte_list(64 * 64, 0.0f);
    std::vector<int> targets_list(64);
    for (int row_index_val = 0; row_index_val < 64; row_index_val++)
    {
        identity_list[row_index_val * 64 + row_index_val] = 1.0f;
        wte_list[row_index_val * 64 + row_index_val] = 4.0f;
        targets_list[row_index_val] = row_index_val % 2 == 0 ? row_index_val : (row_index_val + 1) % 64;
    }
    mine_val.lnf_output_tensor = bf16_tensor_of({1, 64, 64}, identity_list);
    mine_val.wte_tensor = bf16_tensor_of({64, 64}, wte_list);
    mine_val.targets_tensor = targets_of({1, 64}, targets_list);
    mine_val.forward();
    // 2. the expected numbers
    double row_sum_val = 1.0 + 63.0 * std::exp(-4.0);
    std::vector<float> losses_list(64);
    for (int row_index_val = 0; row_index_val < 64; row_index_val++)
        losses_list[row_index_val] = (float)(std::log(row_sum_val) + (row_index_val % 2 == 0 ? 0.0 : 4.0));
    // 3. check
    ASSERT_EQ(mine_val.max_of_each_row_tensor.tolist(), std::vector<float>(64, 4.0f));
    ASSERT_NEAR(mine_val.sum_of_each_row_tensor.tolist(), std::vector<float>(64, (float)row_sum_val), 1e-6);
    ASSERT_NEAR(mine_val.losses_tensor.tolist(), losses_list, 1e-6);
}

static void logits_grad_can_be_the_logits_themselves(void)
{
    // Expect: the backward with logits_grad = logits (no 412 MB for GPT-2's logits_grad): each float4 of logits is
    // read before the same float4 of logits_grad is written, so logits_grad, lnf_output_grad and wte_grad are the
    // same bits as with logits_grad apart. 128 rows, embed_dim 768, vocab_size 1000 padded to 1024.
    // 1. random values; the forward, then a backward into logits_grad apart
    torch::manual_seed(0);
    ClassifierTensors apart_val(2, 64, GPT2_EMBED_DIM, 1000, 1024);
    apart_val.lnf_output_tensor = random_bf16_tensor(apart_val.lnf_output_tensor.shape());
    apart_val.wte_tensor = random_bf16_tensor(apart_val.wte_tensor.shape(), WTE_STD);
    apart_val.targets_tensor = random_targets_on_gpu({2, 64}, 1000);
    apart_val.forward();
    apart_val.backward();
    // 2. the same backward into logits itself (in_place_val shares apart_val's tensors: logits is overwritten
    //    only now, after apart_val's results are in)
    ClassifierTensors in_place_val = apart_val;
    in_place_val.lnf_output_grad_tensor = Tensor::zeros(apart_val.lnf_output_tensor.shape(), DType::BFloat16);
    in_place_val.wte_grad_tensor = Tensor::zeros(apart_val.wte_tensor.shape(), DType::BFloat16);
    in_place_val.logits_grad_tensor = in_place_val.logits_tensor;
    in_place_val.backward();
    // 3. check: the same bits
    ASSERT_EQ(values_of(in_place_val.logits_tensor), values_of(apart_val.logits_grad_tensor));
    ASSERT_EQ(values_of(in_place_val.lnf_output_grad_tensor), values_of(apart_val.lnf_output_grad_tensor));
    ASSERT_EQ(values_of(in_place_val.wte_grad_tensor), values_of(apart_val.wte_grad_tensor));
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (the logits kernel's one warpgroup, 128; the
    // loss kernel's 256; logits_grad's 128), and nothing is spilled. cudaFuncGetAttributes reads what the compiler
    // wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // The logits kernel holds the tile's 32 sums and, as it stores them, 16 rounded logits of a row.
    const int kernel_count_val = 3;
    const void *kernels_list[kernel_count_val] = {(const void *)classifier_bf16_logits_kernel,
                                                  (const void *)classifier_bf16_loss_kernel,
                                                  (const void *)classifier_bf16_logits_grad_kernel};
    const char *names_list[kernel_count_val] = {"classifier_bf16_logits_kernel", "classifier_bf16_loss_kernel",
                                                "classifier_bf16_logits_grad_kernel"};
    int threads_list[kernel_count_val] = {HARDWARE_WARPGROUP_THREADS, CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK,
                                          CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK};
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: random values, 128 rows, embed_dim 768, vocab_size 1000 padded to 1024. The loss kernel and
    // logits_grad with their block size set to 32, 128, 512 and 1024 (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK)
    // give the defaults' losses, row_max, row_sum, logits_grad, lnf_output_grad and wte_grad to the last bit: each row
    // of the loss kernel is one warp's whatever the block, and each value of logits_grad one thread's.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                    HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random values; the defaults' results
    torch::manual_seed(0);
    ClassifierTensors usual_val(2, 64, GPT2_EMBED_DIM, 1000, 1024);
    usual_val.lnf_output_tensor = random_bf16_tensor(usual_val.lnf_output_tensor.shape());
    usual_val.wte_tensor = random_bf16_tensor(usual_val.wte_tensor.shape(), WTE_STD);
    usual_val.targets_tensor = random_targets_on_gpu({2, 64}, 1000);
    usual_val.forward();
    usual_val.backward();
    // 2. each block size, from the same values, against the defaults'
    for (int threads_per_block_val : block_sizes_list)
    {
        ClassifierTensors tensors_val(2, 64, GPT2_EMBED_DIM, 1000, 1024);
        tensors_val.lnf_output_tensor = usual_val.lnf_output_tensor;
        tensors_val.wte_tensor = usual_val.wte_tensor;
        tensors_val.targets_tensor = usual_val.targets_tensor;
        tensors_val.forward(threads_per_block_val);
        tensors_val.backward(threads_per_block_val);
        ASSERT_EQ(tensors_val.losses_tensor.tolist(), usual_val.losses_tensor.tolist());
        ASSERT_EQ(tensors_val.max_of_each_row_tensor.tolist(), usual_val.max_of_each_row_tensor.tolist());
        ASSERT_EQ(tensors_val.sum_of_each_row_tensor.tolist(), usual_val.sum_of_each_row_tensor.tolist());
        ASSERT_EQ(values_of(tensors_val.logits_grad_tensor), values_of(usual_val.logits_grad_tensor));
        ASSERT_EQ(values_of(tensors_val.lnf_output_grad_tensor), values_of(usual_val.lnf_output_grad_tensor));
        ASSERT_EQ(values_of(tensors_val.wte_grad_tensor), values_of(usual_val.wte_grad_tensor));
    }
}

static void every_tile_order_gives_the_same_bits(void)
{
    // Expect: random values, 320 rows (5 rows of tiles), embed_dim 768, vocab_size 1000 padded to 1024 (16 tiles
    // across). The logits kernel's blocks in groups of 1 (row after row), 2, 3, 4 and 64 rows of tiles, down a group
    // first (the last group gets the rows left): every tile is worked out the same way whichever block runs it when,
    // and each tile's (max, sum) is stored in its own place, so logits, losses, row_max and row_sum are the
    // default order's to the last bit.
    // 1. random values; the default order's results
    const int tile_rows_per_group_list[] = {1, 2, 3, 4, 64};
    torch::manual_seed(11);
    ClassifierTensors usual_val(5, 64, GPT2_EMBED_DIM, 1000, 1024);
    usual_val.lnf_output_tensor = random_bf16_tensor(usual_val.lnf_output_tensor.shape());
    usual_val.wte_tensor = random_bf16_tensor(usual_val.wte_tensor.shape(), WTE_STD);
    usual_val.targets_tensor = random_targets_on_gpu({5, 64}, 1000);
    usual_val.forward();
    print_test_name(__func__);
    // 2. each order, from the same values, against the default's
    for (int tile_rows_per_group_val : tile_rows_per_group_list)
    {
        ClassifierTensors tensors_val(5, 64, GPT2_EMBED_DIM, 1000, 1024);
        tensors_val.lnf_output_tensor = usual_val.lnf_output_tensor;
        tensors_val.wte_tensor = usual_val.wte_tensor;
        tensors_val.targets_tensor = usual_val.targets_tensor;
        tensors_val.forward(CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK, tile_rows_per_group_val);
        printf("    groups of %d rows of tiles:\n", tile_rows_per_group_val);
        ASSERT_EQ(values_of(tensors_val.logits_tensor), values_of(usual_val.logits_tensor));
        ASSERT_EQ(tensors_val.losses_tensor.tolist(), usual_val.losses_tensor.tolist());
        ASSERT_EQ(tensors_val.max_of_each_row_tensor.tolist(), usual_val.max_of_each_row_tensor.tolist());
        ASSERT_EQ(tensors_val.sum_of_each_row_tensor.tolist(), usual_val.sum_of_each_row_tensor.tolist());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_sizes();
    matches_pytorch_on_a_part_full_last_tile();
    matches_pytorch_with_a_tile_of_padding_only();

    equal_logits_give_log_of_the_vocabulary_size();
    each_row_sees_its_own_word_and_target();
    logits_grad_can_be_the_logits_themselves();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();
    every_tile_order_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
