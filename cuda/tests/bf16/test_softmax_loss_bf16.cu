// test_softmax_loss_bf16.cu — tests for cuda/kernels/bf16/softmax_loss_bf16.cuh (softmax_loss_bf16_forward, _backward,
// _forward_backward): the softmax, the cross-entropy loss and its gradient over bfloat16 logits. Every check prints its
// test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks against PyTorch's cross_entropy in float on the same values, with autograd for the gradient; part 2
// checks numbers worked out by hand; part 3 checks that the three launches agree to the last bit (the fused one, the
// two alone, the gradient over the logits themselves) and that any block size works; part 4 that each kernel fits its
// block. The timings are in cuda/tests/bf16/tune_softmax_loss_bf16.cu.
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../kernels/bf16/softmax_loss_bf16.cuh"

// Against PyTorch in float on the same bfloat16 logits: the loss, row max and row sum are float on both sides, each
// exp one ex2.approx (about 2 ulp) here; the largest difference as a fraction of the largest value.
static const double FLOAT_TOLERANCE = 1e-5;
// logits_grad is rounded to bfloat16 here, by up to 2^-9 of each value: twice that, as a fraction of the largest.
static const double BFLOAT16_TOLERANCE = 8e-3;
// With the softcap each capped logit is c · tanh.approx (about 2^-11 relative), off by up to ~4e-3 at |capped| 8; the
// exps carry that into the sum and the gradient: within 1e-2 of the largest value.
static const double SOFTCAP_TOLERANCE = 1e-2;
#define NANOCHAT_LOGIT_SOFTCAP 15.0f  // the 8-bit model's c (TRAIN_8BIT_LOGIT_SOFTCAP)

#define GPT2_VOCAB_SIZE 50257         // GPT-2's real words
#define GPT2_PADDED_VOCAB_SIZE 50304  // the row length: the words, then padding up to a multiple of 128

// ── helpers ───────────────────────────────────────────────────────────────────

/// Everything one launch reads and writes, for batch_size x seq_len rows of padded_vocab_size.
struct SoftmaxLossTensors
{
    // Random bfloat16 logits (std 2, padding random too, so a padding value read by mistake shows), random targets
    // under vocab_size.
    size_t vocab_size_val;
    Tensor logits_tensor;
    Tensor targets_tensor;
    Tensor losses_tensor;
    Tensor max_of_each_row_tensor;
    Tensor sum_of_each_row_tensor;
    Tensor logits_grad_tensor;

    /// Makes them, the logits and targets random from the current torch seed.
    SoftmaxLossTensors(size_t batch_size_val, size_t seq_len_val, size_t model_vocab_size_val, size_t padded_vocab_size_val)
        : vocab_size_val(model_vocab_size_val)
    {
        torch::TensorOptions on_gpu_val = torch::TensorOptions().device(torch::kCUDA);
        torch::Tensor values_tensor = 2.0f * torch::randn({(int64_t)(batch_size_val * seq_len_val * padded_vocab_size_val)}, on_gpu_val);
        Tensor floats_tensor = Tensor::empty({batch_size_val, seq_len_val, padded_vocab_size_val});
        cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        logits_tensor = floats_tensor.to(DType::BFloat16);
        targets_tensor = Tensor::empty({batch_size_val, seq_len_val}, DType::Int32);
        torch::Tensor targets_values_tensor = torch::randint((int64_t)model_vocab_size_val, {(int64_t)(batch_size_val * seq_len_val)},
                                                             torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        cudaCheck(cudaMemcpy(targets_tensor.data_ptr<int>(), targets_values_tensor.data_ptr<int>(),
                             targets_tensor.numel() * sizeof(int), cudaMemcpyDeviceToDevice));
        losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
        max_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
        sum_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
        logits_grad_tensor = Tensor::zeros(logits_tensor.shape(), DType::BFloat16);
    }

    /// Runs the fused launch: losses, row max and sum, and logits_grad for the mean over the rows.
    void forward_backward(int threads_per_block_val = SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, float softcap_val = 0.0f)
    {
        softmax_loss_bf16_forward_backward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_grad_tensor,
                                           logits_tensor, targets_tensor, vocab_size_val, mean_loss_grad(),
                                           threads_per_block_val, softcap_val);
    }

    /// Gives d(mean loss) / d(each row's loss): 1 / rows.
    float mean_loss_grad() const { return 1.0f / (float)losses_tensor.numel(); }
};

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU (a bfloat16 tensor's values turned into floats first, exactly).
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets_tensor)
{
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(targets_tensor.shape().begin(), targets_tensor.shape().end());
    return torch::from_blob(targets_tensor.data_ptr<int>(), sizes_list, ints_on_gpu_val).to(torch::kLong).reshape({-1});
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
///
///   mine_tensor      my result
///   pytorch_tensor   PyTorch's, the same shape
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    // Worked out on the GPU.
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives my tensor's values as a CPU float list in row order.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs the fused launch on random logits and targets and checks it against PyTorch's cross_entropy.
///
///   test_ptr                the test's name
///   batch_size_val          rows of the batch
///   seq_len_val             tokens a row
///   vocab_size_val          real columns of the logits
///   padded_vocab_size_val   the logits' columns
///   softcap_val             the logit softcap; 0: none
static void fused_matches_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val, size_t vocab_size_val,
                                  size_t padded_vocab_size_val, float softcap_val = 0.0f)
{
    // Runs the fused launch on random logits and targets, and PyTorch's cross_entropy over the first vocab_size columns
    // in float with autograd (the mean over the rows), the logits capped first to c · tanh(logit / c) if softcap_val > 0;
    // checks losses, row max (mine times ln 2 back to natural units), row sum within FLOAT_TOLERANCE (with the softcap:
    // SOFTCAP_TOLERANCE), logits_grad's words within BFLOAT16_TOLERANCE (SOFTCAP_TOLERANCE) and its padding exactly 0.
    // 1. mine
    torch::manual_seed(0);
    SoftmaxLossTensors mine_val(batch_size_val, seq_len_val, vocab_size_val, padded_vocab_size_val);
    mine_val.forward_backward(SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, softcap_val);
    // 2. PyTorch's, in float on the same values: the words only, capped first with a softcap (autograd through it)
    int64_t rows_val = (int64_t)(batch_size_val * seq_len_val);
    torch::Tensor word_logits_tensor = pytorch_copy_of(mine_val.logits_tensor)
                                           .reshape({rows_val, (int64_t)padded_vocab_size_val})
                                           .narrow(1, 0, (int64_t)vocab_size_val)
                                           .clone()
                                           .requires_grad_(true);
    torch::Tensor capped_tensor = softcap_val > 0.0f ? softcap_val * torch::tanh(word_logits_tensor / softcap_val) : word_logits_tensor;
    torch::Tensor targets_tensor = pytorch_targets_of(mine_val.targets_tensor);
    torch::Tensor losses_tensor = torch::cross_entropy_loss(capped_tensor, targets_tensor, {}, at::Reduction::None);
    losses_tensor.mean().backward();
    torch::Tensor max_tensor = std::get<0>(capped_tensor.detach().max(1));
    torch::Tensor sum_tensor = (capped_tensor.detach() - max_tensor.unsqueeze(1)).exp().sum(1);
    // 3. check
    double float_tolerance_val = softcap_val > 0.0f ? SOFTCAP_TOLERANCE : FLOAT_TOLERANCE;
    double grad_tolerance_val = softcap_val > 0.0f ? SOFTCAP_TOLERANCE : BFLOAT16_TOLERANCE;
    print_test_name(test_ptr);
    printf("    %zu rows of %zu words, %zu columns, softcap %g\n", (size_t)rows_val, vocab_size_val, padded_vocab_size_val,
           softcap_val);
    torch::Tensor mine_max_tensor = pytorch_copy_of(mine_val.max_of_each_row_tensor).reshape({-1}) * SOFTMAX_LOSS_BF16_LN_2;
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.losses_tensor, losses_tensor.detach()), 0.0, float_tolerance_val);
    ASSERT_NEAR(((mine_max_tensor - max_tensor).abs().max() / max_tensor.abs().max()).item<double>(), 0.0, float_tolerance_val);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.sum_of_each_row_tensor, sum_tensor), 0.0, float_tolerance_val);
    torch::Tensor mine_grad_tensor = pytorch_copy_of(mine_val.logits_grad_tensor).reshape({rows_val, (int64_t)padded_vocab_size_val});
    torch::Tensor mine_word_grad_tensor = mine_grad_tensor.narrow(1, 0, (int64_t)vocab_size_val);
    torch::Tensor word_grad_tensor = word_logits_tensor.grad();
    ASSERT_NEAR(((mine_word_grad_tensor - word_grad_tensor).abs().max() / word_grad_tensor.abs().max()).item<double>(), 0.0,
                grad_tolerance_val);
    if (padded_vocab_size_val > vocab_size_val)
        ASSERT_EQ(mine_grad_tensor.narrow(1, (int64_t)vocab_size_val, (int64_t)(padded_vocab_size_val - vocab_size_val))
                      .abs().max().item<float>(),
                  0.0f);
}

static void fused_matches_pytorch_on_gpt2_rows(void)
{
    // Expect: 4 x 256 rows of GPT-2's 50257 words padded to 50304, random logits std 2 (padding random too): losses,
    // row max and row sum within 1e-5 of PyTorch's float cross_entropy, logits_grad within the bfloat16 tolerance,
    // the padding's gradient 0 exactly (never a padding value in the softmax).
    fused_matches_pytorch(__func__, 4, 256, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
}

static void fused_matches_pytorch_with_a_part_load_of_words(void)
{
    // Expect: 3 x 5 rows of 50 words padded to 64: the 7th load holds 2 words and 6 padding values, the 8th only
    // padding. The same checks.
    fused_matches_pytorch(__func__, 3, 5, 50, 64);
}

static void softcap_matches_pytorch_on_d24_rows(void)
{
    // Expect: 4 x 256 rows of the d24 model's 32768 words (no padding), random logits std 2 (the largest about 9, where
    // tanh already bends), softcap 15: losses, row max, row sum and logits_grad within SOFTCAP_TOLERANCE of PyTorch's
    // cross_entropy of 15 · tanh(logits / 15), its gradient by autograd through the tanh.
    fused_matches_pytorch(__func__, 4, 256, 32768, 32768, NANOCHAT_LOGIT_SOFTCAP);
}

static void softcap_matches_pytorch_with_a_part_load_of_words(void)
{
    // Expect: 3 x 5 rows of 50 words padded to 64, softcap 2 (most logits past the bend: tanh' far from 1): the same
    // checks, the padding's gradient exactly 0.
    fused_matches_pytorch(__func__, 3, 5, 50, 64, 2.0f);
}

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void one_row_by_hand(void)
{
    // Expect: the row 1 2 3 4, then 4 padding values of 100 (which would win the softmax if read), target 3, loss_grad
    // 1: row_max 4 (kept as 4 log2 e), row_sum e^-3 + e^-2 + e^-1 + 1 = 1.5530018, loss log(1.5530018) - 0 =
    // 0.4401897; the gradient the probs 0.0320586, 0.0871443, 0.2368828, 0.6439143 with 1 off the last, rounded to
    // bfloat16 (within 2^-8 of each: half a step of 8 bits, a step 2^-7 of the power of 2 below), the padding 0.
    // 1. the row
    std::vector<float> logits_list = {1, 2, 3, 4, 100, 100, 100, 100};
    Tensor floats_tensor = Tensor::empty({1, 1, 8});
    floats_tensor.copy_from_cpu(logits_list.data());
    Tensor logits_tensor = floats_tensor.to(DType::BFloat16);
    std::vector<int> target_list = {3};
    Tensor targets_tensor = Tensor::empty({1, 1}, DType::Int32);
    targets_tensor.copy_from_cpu(target_list.data());
    Tensor losses_tensor = Tensor::zeros({1, 1}), max_tensor = Tensor::zeros({1, 1}), sum_tensor = Tensor::zeros({1, 1});
    Tensor logits_grad_tensor = Tensor::zeros({1, 1, 8}, DType::BFloat16);
    // 2. the fused launch, loss_grad 1
    softmax_loss_bf16_forward_backward(losses_tensor, max_tensor, sum_tensor, logits_grad_tensor, logits_tensor,
                                       targets_tensor, 4, 1.0f);
    // 3. check
    print_test_name(__func__);
    ASSERT_NEAR(max_tensor.tolist()[0], 4.0 * 1.4426950408889634, 1e-5);
    ASSERT_NEAR(sum_tensor.tolist()[0], 1.5530018, 1e-5);
    ASSERT_NEAR(losses_tensor.tolist()[0], 0.4401897, 1e-5);
    std::vector<float> expected_list = {0.0320586f, 0.0871443f, 0.2368828f, 0.6439143f - 1.0f, 0, 0, 0, 0};
    std::vector<float> grad_list = list_of(logits_grad_tensor);
    for (size_t index_val = 0; index_val < expected_list.size(); index_val++)
        ASSERT_NEAR(grad_list[index_val], expected_list[index_val], std::fabs(expected_list[index_val]) / 256.0 + 1e-7);
}

// ══ part 3: the launches agree, and any block size works ══════════════════════

static void fused_gives_forward_then_backward_bits(void)
{
    // Expect: 2 x 512 rows of GPT-2's: the fused launch, and softmax_loss_bf16_forward then softmax_loss_bf16_backward
    // on the same logits, give the same losses, row max, row sum and logits_grad to the last bit: pass 2 runs the same
    // code on the same row max and sum, kept in memory or in shared memory.
    // 1. fused
    torch::manual_seed(1);
    SoftmaxLossTensors mine_val(2, 512, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
    mine_val.forward_backward();
    std::vector<float> losses_list = list_of(mine_val.losses_tensor);
    std::vector<float> max_list = list_of(mine_val.max_of_each_row_tensor);
    std::vector<float> sum_list = list_of(mine_val.sum_of_each_row_tensor);
    torch::Tensor grad_tensor = pytorch_copy_of(mine_val.logits_grad_tensor);
    // 2. the forward, then the backward, onto 7s
    mine_val.losses_tensor.fill_(7.0), mine_val.max_of_each_row_tensor.fill_(7.0), mine_val.sum_of_each_row_tensor.fill_(7.0);
    mine_val.logits_grad_tensor.fill_(7.0);
    softmax_loss_bf16_forward(mine_val.losses_tensor, mine_val.max_of_each_row_tensor, mine_val.sum_of_each_row_tensor,
                              mine_val.logits_tensor, mine_val.targets_tensor, mine_val.vocab_size_val);
    softmax_loss_bf16_backward(mine_val.logits_grad_tensor, mine_val.logits_tensor, mine_val.targets_tensor,
                               mine_val.max_of_each_row_tensor, mine_val.sum_of_each_row_tensor, mine_val.vocab_size_val,
                               mine_val.mean_loss_grad());
    // 3. check
    print_test_name(__func__);
    ASSERT_EQ(list_of(mine_val.losses_tensor) == losses_list, true);
    ASSERT_EQ(list_of(mine_val.max_of_each_row_tensor) == max_list, true);
    ASSERT_EQ(list_of(mine_val.sum_of_each_row_tensor) == sum_list, true);
    ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.logits_grad_tensor), grad_tensor), true);
}

static void gradient_over_the_logits_gives_the_same_bits(void)
{
    // Expect: 2 x 512 rows of GPT-2's: the fused launch writing the gradient over the logits themselves gives the same
    // gradient, losses, row max and row sum as into a tensor of its own: each value is read before the same thread
    // writes it, and pass 1 is done with the row before pass 2 writes it.
    // 1. into a tensor of its own
    torch::manual_seed(2);
    SoftmaxLossTensors mine_val(2, 512, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
    mine_val.forward_backward();
    torch::Tensor grad_tensor = pytorch_copy_of(mine_val.logits_grad_tensor);
    std::vector<float> losses_list = list_of(mine_val.losses_tensor);
    // 2. over the logits
    softmax_loss_bf16_forward_backward(mine_val.losses_tensor, mine_val.max_of_each_row_tensor, mine_val.sum_of_each_row_tensor,
                                       mine_val.logits_tensor, mine_val.logits_tensor, mine_val.targets_tensor,
                                       mine_val.vocab_size_val, mine_val.mean_loss_grad());
    print_test_name(__func__);
    ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.logits_tensor), grad_tensor), true);
    ASSERT_EQ(list_of(mine_val.losses_tensor) == losses_list, true);
}

static void any_block_size_works(void)
{
    // Expect: 2 x 512 rows of GPT-2's, the fused launch at 32, 64, 128, 256, 512 and 1024 threads a block (one warp up
    // to HARDWARE_MAX_THREADS_PER_BLOCK): losses, row max and row sum within 1e-6 of the default's (another size adds
    // each row in another order, so the last bits can change), logits_grad within 2^-8 of it. And the backward alone at
    // each size, from the default's row max and sum, gives the default's logits_grad to the last bit: each value
    // is worked out alone.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP,
                                    8 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. the default's
    torch::manual_seed(3);
    SoftmaxLossTensors usual_val(2, 512, GPT2_VOCAB_SIZE, GPT2_PADDED_VOCAB_SIZE);
    usual_val.forward_backward();
    torch::Tensor usual_grad_tensor = pytorch_copy_of(usual_val.logits_grad_tensor);
    print_test_name(__func__);
    for (int threads_per_block_val : block_sizes_list)
    {
        // 2. the fused launch at this size: within float rounding
        SoftmaxLossTensors mine_val = usual_val;
        mine_val.losses_tensor = Tensor::zeros(usual_val.losses_tensor.shape());
        mine_val.max_of_each_row_tensor = Tensor::zeros(usual_val.losses_tensor.shape());
        mine_val.sum_of_each_row_tensor = Tensor::zeros(usual_val.losses_tensor.shape());
        mine_val.logits_grad_tensor = Tensor::zeros(usual_val.logits_tensor.shape(), DType::BFloat16);
        mine_val.forward_backward(threads_per_block_val);
        printf("    %d threads a block:\n", threads_per_block_val);
        ASSERT_NEAR(list_of(mine_val.losses_tensor), list_of(usual_val.losses_tensor), 1e-6);
        ASSERT_NEAR(list_of(mine_val.max_of_each_row_tensor), list_of(usual_val.max_of_each_row_tensor), 1e-6);
        ASSERT_NEAR(list_of(mine_val.sum_of_each_row_tensor), list_of(usual_val.sum_of_each_row_tensor), 1e-6);
        ASSERT_NEAR(relative_difference_on_gpu(mine_val.logits_grad_tensor, usual_grad_tensor), 0.0, 1.0 / 256.0);
        // 3. the backward alone at this size, from the default's row max and sum: the same bits
        softmax_loss_bf16_backward(mine_val.logits_grad_tensor, usual_val.logits_tensor, usual_val.targets_tensor,
                                   usual_val.max_of_each_row_tensor, usual_val.sum_of_each_row_tensor, usual_val.vocab_size_val,
                                   usual_val.mean_loss_grad(), threads_per_block_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(mine_val.logits_grad_tensor), usual_grad_tensor), true);
    }
}

// ══ part 4: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the three kernels (forward, backward, both) can be launched with HARDWARE_MAX_THREADS_PER_BLOCK threads,
    // the most any_block_size_works gives them, and nothing is spilled: cudaFuncGetAttributes' maxThreadsPerBlock and
    // localSizeBytes, read without running anything.
    const void *kernels_list[] = {(const void *)softmax_loss_bf16_kernel<true, false>,
                                  (const void *)softmax_loss_bf16_kernel<false, true>,
                                  (const void *)softmax_loss_bf16_kernel<true, true>};
    const char *names_list[] = {"softmax_loss_bf16_kernel<forward>", "softmax_loss_bf16_kernel<backward>",
                                "softmax_loss_bf16_kernel<forward, backward>"};
    for (int kernel_index_val = 0; kernel_index_val < 3; kernel_index_val++)
    {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val],
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= HARDWARE_MAX_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    fused_matches_pytorch_on_gpt2_rows();
    fused_matches_pytorch_with_a_part_load_of_words();
    softcap_matches_pytorch_on_d24_rows();
    softcap_matches_pytorch_with_a_part_load_of_words();

    one_row_by_hand();

    fused_gives_forward_then_backward_bits();
    gradient_over_the_logits_gives_the_same_bits();
    any_block_size_works();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
