// test_classifier_8bit.cu — tests for llmc/kernels/8bit/classifier_8bit.cuh (classifier_8bit_forward,
// classifier_8bit_backward): GPT-2's head with lnf_output and wte rounded to E4M3 (round_8bit), the product on an
// H100's E4M3 tensor cores (WGMMA fed by TMA) with the softmax's row sums worked out as each tile of logits is
// stored, then the loss; the backward's logits_grad and the head's two products in 8 bits (linear_8bit_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the logits to the bit against wgmma_matmul_8bit of the very E4M3 copies the forward made (read back
// from the scratch), the softmax against fp32/classifier.cuh's on those logits, the gradients against PyTorch's
// float products of the backward's copies, and everything against PyTorch's float head and cross_entropy on the
// values before rounding (how far 8 bits move it); part 2 checks numbers worked out by hand; part 3 checks that the
// logits kernel's registers fit its block. The loss and logits_grad kernels are classifier_bf16.cuh's, tested with
// other block sizes in tests/test_classifier_bf16.cu. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a in place of -arch=native. The timings are in tests/8bit/tune_classifier_8bit.cu.
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, float8_e4m3fn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <cmath>                      // std::log, std::exp: the hand-worked losses
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/8bit/classifier_8bit.cuh"
#include "../../llmc/kernels/fp32/classifier.cuh"      // the unfused softmax, to compare with

// Against PyTorch in float on the same E4M3 copies: the sums are float on both sides, then mine are rounded to
// bfloat16, by up to 2^-9 of each value. Against the values before rounding: each E4M3 is off by up to 2^-4 (6%) of
// its value, so the logits and gradients move by a few percent. Each tolerance is the largest difference, as a
// fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double EIGHT_BIT_TOLERANCE = 1e-1;
// Against fp32/classifier.cuh on the same stored logits: the same math in float, each row added up in another order.
static const double SAME_LOGITS_TOLERANCE = 1e-5;
static const float WTE_STD = 0.02f;   // GPT-2's starting std: logits of about 0.02 x sqrt(embed_dim), as in training

#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_VOCAB_SIZE 50257
#define GPT2_PADDED_VOCAB_SIZE 50304

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation `std_val`, made by
/// PyTorch on the GPU, then rounded to bfloat16.
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

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
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

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives an 8-bit copy turned back, by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times its row's scale,
/// exact. {rows, k} float.
///
///   rounded_tensor   {rows, k} E4M3, as round_8bit wrote it
///   scales_tensor    {rows} float
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scales_tensor)
{
    int64_t rows_val = (int64_t)rounded_tensor.size(0);
    int64_t k_val = (int64_t)rounded_tensor.size(1);
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, k_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
}

/// Gives PyTorch's float product left · rightᵀ of one of the scratch's products, its E4M3 copies turned back.
///
///   copies_val   one of the scratch's three products
static torch::Tensor product_of_copies(const Linear8bitProductCopies &copies_val)
{
    return turned_back(copies_val.left_8bit_tensor, copies_val.left_scales_tensor)
        .matmul(turned_back(copies_val.right_8bit_tensor, copies_val.right_scales_tensor).t());
}

/// Every tensor classifier_8bit_forward and _backward take, for one size: made with 0s, lnf_output, wte and
/// targets filled by the test; losses_grad 1 / rows in every row (the mean loss); the scratch for the head's sizes.
struct ClassifierTensors
{
    size_t vocab_size_val;
    Linear8bitScratch scratch_val;
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
        : vocab_size_val(vocab_size), scratch_val(batch_size_val * seq_len_val, embed_dim_val, padded_vocab_size_val)
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

    /// Runs classifier_8bit_forward.
    void forward()
    {
        classifier_8bit_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, tile_max_tensor, tile_sum_tensor,
                                logits_tensor, lnf_output_tensor, wte_tensor, targets_tensor, vocab_size_val, scratch_val);
    }

    /// Runs classifier_8bit_backward into logits_grad.
    void backward()
    {
        classifier_8bit_backward(lnf_output_grad_tensor, wte_grad_tensor, logits_grad_tensor, losses_grad_tensor, logits_tensor,
                                 lnf_output_tensor, wte_tensor, targets_tensor, max_of_each_row_tensor, sum_of_each_row_tensor,
                                 vocab_size_val, scratch_val);
    }
};

// ══ part 1: mine against the copies' products, the unfused softmax and PyTorch ═

/// Runs one forward and one backward on random bfloat16 lnf_output (std 1) and wte (std WTE_STD) and random
/// targets, then checks: the logits to the bit against wgmma_matmul_8bit of the forward's copies; losses, row_max and
/// row_sum within SAME_LOGITS_TOLERANCE of fp32/classifier.cuh's on the stored logits; lnf_output_grad and wte_grad
/// within BFLOAT16_TOLERANCE of PyTorch's float products of the backward's copies; and logits, losses,
/// lnf_output_grad and wte_grad within EIGHT_BIT_TOLERANCE of PyTorch's float head and cross_entropy with autograd
/// on the values before rounding. Prints the sizes under `test_ptr`'s name.
static void forward_and_backward_match_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val,
                                               size_t embed_dim_val, size_t vocab_size_val, size_t padded_vocab_size_val)
{
    // 1. random values; mine
    torch::manual_seed(0);
    size_t rows_val = batch_size_val * seq_len_val;
    ClassifierTensors mine_val(batch_size_val, seq_len_val, embed_dim_val, vocab_size_val, padded_vocab_size_val);
    mine_val.lnf_output_tensor = random_bf16_tensor(mine_val.lnf_output_tensor.shape());
    mine_val.wte_tensor = random_bf16_tensor(mine_val.wte_tensor.shape(), WTE_STD);
    mine_val.targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    mine_val.forward();

    // 2. the unfused path on the forward's copies: wgmma_matmul_8bit, then fp32/classifier.cuh's softmax on my logits.
    //    Promotion 0: the fused head keeps its embed_dim-long sum in the tensor cores, never moved into float, so
    //    the unfused product adds the same way and the logits match to the last bit
    const Linear8bitProductCopies &copies_val = mine_val.scratch_val.output_copies;
    Tensor unfused_logits_tensor = Tensor::zeros({rows_val, padded_vocab_size_val}, DType::BFloat16);
    wgmma_matmul_8bit(unfused_logits_tensor, copies_val.left_8bit_tensor, copies_val.left_scales_tensor,
                      copies_val.right_8bit_tensor, copies_val.right_scales_tensor, Tensor(), false, WgmmaMatmul8bitStore(), 0);
    Tensor unfused_losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor unfused_max_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor unfused_sum_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    classifier_forward(unfused_losses_tensor, unfused_max_tensor, unfused_sum_tensor, mine_val.logits_tensor,
                       mine_val.targets_tensor, vocab_size_val);

    // 3. the backward, and PyTorch's float products of its copies
    mine_val.backward();
    torch::Tensor copies_lnf_output_grad_tensor = product_of_copies(mine_val.scratch_val.input_grad_copies);
    torch::Tensor copies_wte_grad_tensor = product_of_copies(mine_val.scratch_val.weight_grad_copies);

    // 4. PyTorch's float head and cross_entropy over the words, with autograd, on the values before rounding
    torch::Tensor pytorch_lnf_output_tensor = pytorch_copy_of(mine_val.lnf_output_tensor).reshape({(int64_t)rows_val, -1}).requires_grad_(true);
    torch::Tensor pytorch_wte_tensor = pytorch_copy_of(mine_val.wte_tensor).requires_grad_(true);
    torch::Tensor pytorch_logits_tensor = pytorch_lnf_output_tensor.matmul(pytorch_wte_tensor.t());
    torch::Tensor pytorch_losses_tensor = torch::cross_entropy_loss(
        pytorch_logits_tensor.narrow(1, 0, (int64_t)vocab_size_val), pytorch_targets_of(mine_val.targets_tensor).reshape({-1}),
        {}, at::Reduction::None);
    pytorch_losses_tensor.backward(pytorch_copy_of(mine_val.losses_grad_tensor).reshape({-1}));

    // 5. print the sizes, then check: the copies' products and the unfused softmax, then PyTorch before rounding
    print_test_name(test_ptr);
    printf("    %zu rows, embed_dim %zu, vocab_size %zu padded to %zu (%zu tiles across)\n", rows_val, embed_dim_val,
           vocab_size_val, padded_vocab_size_val, classifier_bf16_tile_columns(padded_vocab_size_val));
    ASSERT_EQ(values_of(mine_val.logits_tensor), values_of(unfused_logits_tensor));
    ASSERT_NEAR(mine_val.losses_tensor.tolist(), unfused_losses_tensor.tolist(), SAME_LOGITS_TOLERANCE);
    ASSERT_NEAR(mine_val.max_of_each_row_tensor.tolist(), unfused_max_tensor.tolist(), SAME_LOGITS_TOLERANCE);
    ASSERT_NEAR(mine_val.sum_of_each_row_tensor.tolist(), unfused_sum_tensor.tolist(), SAME_LOGITS_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.lnf_output_grad_tensor, copies_lnf_output_grad_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.wte_grad_tensor, copies_wte_grad_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.logits_tensor, pytorch_logits_tensor.detach()), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.losses_tensor, pytorch_losses_tensor.detach()), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.lnf_output_grad_tensor, pytorch_lnf_output_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.wte_grad_tensor, pytorch_wte_tensor.grad()), 0.0, EIGHT_BIT_TOLERANCE);
}

static void matches_pytorch_on_gpt2_sizes(void)
{
    // Expect: 64 rows of GPT-2 small: embed_dim 768 (6 steps of 128), vocab_size 50257 padded to 50304 (786 tiles
    // across, the last with 17 words and 47 padding columns). The checks above.
    forward_and_backward_match_pytorch(__func__, 1, 64, GPT2_EMBED_DIM, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
}

static void matches_pytorch_on_a_part_full_last_tile(void)
{
    // Expect: 2 x 40 = 80 rows (a tile of 64 down, then one of 16), embed_dim 96 (less than one step of 128: TMA
    // gives 0s past the edge), vocab_size 1000 padded to 1008 (multiples of 16, as every 8-bit size must be): 16
    // tiles across, the last 48 columns wide with 40 words. The same checks.
    forward_and_backward_match_pytorch(__func__, 2, 40, 96, 1000, 1008);
}

static void matches_pytorch_with_a_tile_of_padding_only(void)
{
    // Expect: 128 rows, embed_dim 768, vocab_size 900 padded to 1024: tile 15 (columns 960..1023) only padding, its
    // rows' (max, sum) (-FLT_MAX, 0), adding nothing. The same checks.
    forward_and_backward_match_pytorch(__func__, 2, 64, GPT2_EMBED_DIM, 900, 1024);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// The values below are 0, 1 and 4: whole numbers E4M3 holds exactly at any power-of-2 scale a row of them gets (a
// row whose largest is 4 has scale 2^-6, 4 x 64 = 256 <= 448; whose largest is 1, 2^-8). So the 8-bit products are
// exact, and the hand-worked numbers are those of bf16/classifier_bf16.cuh.

static void equal_logits_give_log_of_the_vocabulary_size(void)
{
    // Expect: lnf_output all 0, so every logit is 0: 64 rows, embed_dim 64, vocab_size 1000 padded to 1008. row_max 0;
    // row_sum = 1000, exactly; loss = log(1000) = 6.9077553 whatever the target; the padding's logits_grad 0.
    // 1. the forward and the backward
    ClassifierTensors mine_val(1, 64, 64, 1000, 1008);
    mine_val.wte_tensor.fill_(1.0);
    std::vector<int> targets_list(64);
    for (int row_index_val = 0; row_index_val < 64; row_index_val++)
        targets_list[row_index_val] = row_index_val * 15;                  // 0, 15, ..., 945: spread over the words
    mine_val.targets_tensor = targets_of({1, 64}, targets_list);
    mine_val.forward();
    mine_val.backward();
    // 2. check
    ASSERT_EQ(mine_val.max_of_each_row_tensor.tolist(), std::vector<float>(64, 0.0f));
    ASSERT_EQ(mine_val.sum_of_each_row_tensor.tolist(), std::vector<float>(64, 1000.0f));
    ASSERT_NEAR(mine_val.losses_tensor.tolist(), std::vector<float>(64, 6.9077553f), 1e-6);
    std::vector<float> logits_grad_list = values_of(mine_val.logits_grad_tensor);
    size_t padding_values_val = 0;
    for (size_t row_index_val = 0; row_index_val < 64; row_index_val++)
        for (size_t column_index_val = 1000; column_index_val < 1008; column_index_val++)
            padding_values_val += logits_grad_list[row_index_val * 1008 + column_index_val] != 0.0f;
    ASSERT_EQ(padding_values_val, (size_t)0);
}

static void each_row_sees_its_own_word_and_target(void)
{
    // Expect: 64 rows, embed_dim 64, vocab_size 64 (one tile): lnf_output the identity, wte = 4 x the identity, so
    // logit[r][v] = 4 if v = r, else 0, exactly after rounding to 8 bits. Row r's target is r (the even rows) or
    // r + 1 (the odd rows). row_max 4; row_sum = 1 + 63 exp(-4); loss = log(row_sum), + 4 for the odd rows.
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

// ══ part 3: the kernel fits its launch ═════════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: classifier_8bit_logits_kernel can be launched with one warpgroup (128 threads), and nothing is
    // spilled. cudaFuncGetAttributes reads what the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // The loss and logits_grad kernels are checked in tests/test_classifier_bf16.cu.
    // 1. what the compiler wrote down
    cudaFuncAttributes attributes_val;
    cudaCheck(cudaFuncGetAttributes(&attributes_val, (const void *)classifier_8bit_logits_kernel));
    print_test_name(__func__);
    printf("    classifier_8bit_logits_kernel: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
           attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
    // 2. check
    ASSERT_EQ(attributes_val.maxThreadsPerBlock >= HARDWARE_WARPGROUP_THREADS, true);
    ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
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

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
