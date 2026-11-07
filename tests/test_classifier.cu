// test_classifier.cu — tests for llmc/kernels/classifier.cuh (classifier_forward, classifier_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's cross_entropy, with autograd for the backward; part 2 checks
// numbers worked out by hand; part 3 checks that each kernel's registers fit its block, and any block size.
// The timings are in tunes/tune_classifier.cu.
// (The C# trainer had no classifier test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/classifier.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so one
// rounding moves a value by up to 2^-8, about 0.4% of it. Twice that.
static const double BFLOAT16_TOLERANCE = 8e-3;

#define GPT2_VOCAB_SIZE 50257         // GPT-2's real words
#define GPT2_PADDED_VOCAB_SIZE 50304  // the row length: the words, then padding up to a multiple of 128

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes an int tensor of `shape_list` on the GPU holding `values_list`: the targets.
static Tensor targets_of(const Shape &shape_list, const std::vector<int> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list, DType::Int32);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes minutes for the logits' 206M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
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

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets_tensor)
{
    torch::TensorOptions ints_on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(targets_tensor.shape().begin(), targets_tensor.shape().end());
    return torch::from_blob(targets_tensor.data_ptr<int>(), sizes_list, ints_on_gpu_val).to(torch::kLong);
}

/// Gives the first vocab_size columns of pytorch_logits {batch_size, seq_len, padded_vocab_size} as
/// {rows, vocab_size}: the words, without the padding, the way cross_entropy takes them.
static torch::Tensor word_logits_of(const torch::Tensor &pytorch_logits_tensor, size_t vocab_size_val)
{
    int64_t rows_val = pytorch_logits_tensor.size(0) * pytorch_logits_tensor.size(1);
    return pytorch_logits_tensor.narrow(2, 0, (int64_t)vocab_size_val).reshape({rows_val, (int64_t)vocab_size_val});
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random logits and targets, with losses_grad 1 / rows in every
/// row (the mean), mine and PyTorch's cross_entropy over the first vocab_size columns with autograd.
/// Checks that the losses, max_of_each_row, sum_of_each_row and logits_grad agree within TOLERANCE, the padding's
/// gradient 0 on both sides.
static void forward_and_backward_match_pytorch(const Shape &logits_shape_list, size_t vocab_size_val)
{
    // 1. random logits and targets; mine
    torch::manual_seed(0);
    size_t batch_size_val = logits_shape_list[0];
    size_t seq_len_val = logits_shape_list[1];
    size_t rows_val = batch_size_val * seq_len_val;
    Tensor logits_tensor = random_tensor_on_gpu(logits_shape_list);
    Tensor targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    Tensor losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor max_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor sum_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor losses_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val}).fill_(1.0 / rows_val);
    Tensor logits_grad_tensor = Tensor::zeros(logits_shape_list);
    classifier_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor, targets_tensor,
                       vocab_size_val);
    classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor, max_of_each_row_tensor,
                        sum_of_each_row_tensor, vocab_size_val);

    // 2. PyTorch's cross_entropy over the words, with autograd; its row maxes and sums
    torch::Tensor pytorch_logits_tensor = pytorch_copy_of(logits_tensor).requires_grad_(true);
    torch::Tensor word_logits_tensor = word_logits_of(pytorch_logits_tensor, vocab_size_val);
    torch::Tensor pytorch_losses_tensor = torch::cross_entropy_loss(
        word_logits_tensor, pytorch_targets_of(targets_tensor).reshape({-1}), {}, at::Reduction::None);
    pytorch_losses_tensor.backward(pytorch_copy_of(losses_grad_tensor).reshape({-1}));
    torch::Tensor pytorch_max_of_each_row_tensor = std::get<0>(word_logits_tensor.max(1));
    torch::Tensor pytorch_sum_of_each_row_tensor =
        (word_logits_tensor - pytorch_max_of_each_row_tensor.unsqueeze(1)).exp().sum(1);

    // 3. check
    ASSERT_NEAR(losses_tensor.tolist(), list_of(pytorch_losses_tensor), TOLERANCE);
    ASSERT_NEAR(max_of_each_row_tensor.tolist(), list_of(pytorch_max_of_each_row_tensor), TOLERANCE);
    ASSERT_NEAR(sum_of_each_row_tensor.tolist(), list_of(pytorch_sum_of_each_row_tensor), TOLERANCE);
    ASSERT_NEAR(logits_grad_tensor.tolist(), list_of(pytorch_logits_tensor.grad()), TOLERANCE);
}

static void matches_pytorch_on_gpt2_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens, GPT-2's padded_vocab_size = 50304 with
    // vocab_size = 50257 words: 16 blocks of the forward, each of the 256 threads taking about 49 of a
    // row's 12,576 float4s; float4 12,564 holds the last word and 3 padding columns. Random logits and
    // targets: the losses (about log(50257) = 10.8 each, as random logits guess about evenly),
    // max_of_each_row, sum_of_each_row and logits_grad equal PyTorch's within 1e-5.
    forward_and_backward_match_pytorch({2, 8, GPT2_PADDED_VOCAB_SIZE}, GPT2_VOCAB_SIZE);
}

static void matches_pytorch_with_a_small_vocabulary(void)
{
    // Expect: {3, 5, 12} with vocab_size = 10: 3 float4s per row for 256 threads, so 253 threads of
    // each forward block have nothing, and float4 2 holds words 8, 9 and padding 10, 11. The backward's
    // 45 float4s are 1 part-full block. The same as PyTorch within 1e-5.
    forward_and_backward_match_pytorch({3, 5, 12}, 10);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_rows's {2, 8, 50304} logits and targets, with logits and logits_grad
    // in bfloat16. The forward reads each logit as a float and works in float, so the losses,
    // max_of_each_row and sum_of_each_row are PyTorch's float cross_entropy of the same values within 1e-5:
    // bfloat16 changes only what is read, not the math. logits_grad is rounded once as it is stored: within
    // BFLOAT16_TOLERANCE. The 47 padding columns get 0 exactly.

    // 1. random logits in bfloat16 and targets; mine
    torch::manual_seed(0);
    size_t batch_size_val = 2;
    size_t seq_len_val = 8;
    size_t rows_val = batch_size_val * seq_len_val;
    size_t vocab_size_val = GPT2_VOCAB_SIZE;
    Shape logits_shape_list = {batch_size_val, seq_len_val, GPT2_PADDED_VOCAB_SIZE};
    Tensor logits_tensor = random_tensor_on_gpu(logits_shape_list).to(DType::BFloat16);
    Tensor targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, vocab_size_val);
    Tensor losses_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor max_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor sum_of_each_row_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor losses_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val}).fill_(1.0 / rows_val);
    Tensor logits_grad_tensor = Tensor::zeros(logits_shape_list, DType::BFloat16);
    classifier_forward(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor, targets_tensor,
                       vocab_size_val);
    classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor, max_of_each_row_tensor,
                        sum_of_each_row_tensor, vocab_size_val);

    // 2. PyTorch's, in float on the same values
    torch::Tensor pytorch_logits_tensor = pytorch_copy_of(logits_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor word_logits_tensor = word_logits_of(pytorch_logits_tensor, vocab_size_val);
    torch::Tensor pytorch_losses_tensor = torch::cross_entropy_loss(
        word_logits_tensor, pytorch_targets_of(targets_tensor).reshape({-1}), {}, at::Reduction::None);
    pytorch_losses_tensor.backward(pytorch_copy_of(losses_grad_tensor).reshape({-1}));
    torch::Tensor pytorch_max_of_each_row_tensor = std::get<0>(word_logits_tensor.max(1));
    torch::Tensor pytorch_sum_of_each_row_tensor =
        (word_logits_tensor - pytorch_max_of_each_row_tensor.unsqueeze(1)).exp().sum(1);

    // 3. check: the forward's float results within TOLERANCE, logits_grad within one rounding
    ASSERT_NEAR(losses_tensor.tolist(), list_of(pytorch_losses_tensor), TOLERANCE);
    ASSERT_NEAR(max_of_each_row_tensor.tolist(), list_of(pytorch_max_of_each_row_tensor), TOLERANCE);
    ASSERT_NEAR(sum_of_each_row_tensor.tolist(), list_of(pytorch_sum_of_each_row_tensor), TOLERANCE);
    ASSERT_NEAR(logits_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_logits_tensor.grad()), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the row 1 2 3 4 with target 3 (one float4: thread 0 of the block has it all).
// It works out as:
//
//     row_max = 4
//     row_sum = e^-3 + e^-2 + e^-1 + e^0 = 0.0497871 + 0.1353353 + 0.3678794 + 1 = 1.5530018
//     probs   = each / 1.5530018         = 0.0320586  0.0871443  0.2368828  0.6439143
//     loss    = log(1.5530018) - (4 - 4) = 0.4401897 = -log(0.6439143)
//     grad    = probs - 1 at column 3    = 0.0320586  0.0871443  0.2368828  -0.3560857    (losses_grad 1)

static const std::vector<float> GRAD_OF_1_2_3_4_TARGET_3 = {0.0320586f, 0.0871443f, 0.2368828f, -0.3560857f};

/// The forward's three results for a few rows, all 0 at first.
struct RowResults
{
    Tensor losses_tensor;
    Tensor max_of_each_row_tensor;
    Tensor sum_of_each_row_tensor;
};

/// Makes the forward's results for `shape_list` rows, all 0.
static RowResults zero_row_results(const Shape &shape_list)
{
    return {Tensor::zeros(shape_list), Tensor::zeros(shape_list), Tensor::zeros(shape_list)};
}

static void forward_of_1_2_3_4(void)
{
    // Expect: the row 1 2 3 4, target 3: loss 0.4401897, max_of_each_row 4 exactly, sum_of_each_row 1.5530018.
    // 1. the forward
    RowResults results_val = zero_row_results({1, 1});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       tensor_of({1, 1, 4}, {1, 2, 3, 4}), targets_of({1, 1}, {3}), 4);
    // 2. check
    ASSERT_NEAR(results_val.losses_tensor.tolist()[0], 0.4401897, 1e-6);
    ASSERT_EQ(results_val.max_of_each_row_tensor.tolist(), std::vector<float>({4}));
    ASSERT_NEAR(results_val.sum_of_each_row_tensor.tolist()[0], 1.5530018, 1e-6);
}

static void backward_of_1_2_3_4(void)
{
    // Expect: the row 1 2 3 4, target 3, losses_grad 1: the grad above, within 1e-6. It pushes the
    // target's logit up (-0.36: the loss falls as it rises) and every other logit down, each by its prob.
    // 1. the forward, then the backward
    Tensor logits_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor targets_tensor = targets_of({1, 1}, {3});
    RowResults results_val = zero_row_results({1, 1});
    Tensor logits_grad_tensor = Tensor::zeros({1, 1, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, Tensor::ones({1, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_NEAR(logits_grad_tensor.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
}

static void equal_logits_give_log_of_the_vocabulary_size(void)
{
    // Expect: the row 0 0 0 0, target 1: every prob is 1 / 4, so the loss is log(4) = 1.3862944, the
    // loss of an even guess (for GPT-2's 50257 words, log(50257) = 10.82, where training starts).
    // sum_of_each_row is 4 and the grad 0.25 -0.75 0.25 0.25, both exactly: e^0 = 1, and 1 / 4 has no rounding.
    // 1. the forward, then the backward
    Tensor logits_tensor = Tensor::zeros({1, 1, 4});
    Tensor targets_tensor = targets_of({1, 1}, {1});
    RowResults results_val = zero_row_results({1, 1});
    Tensor logits_grad_tensor = Tensor::zeros({1, 1, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, Tensor::ones({1, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_NEAR(results_val.losses_tensor.tolist()[0], 1.3862944, 1e-6);
    ASSERT_EQ(results_val.sum_of_each_row_tensor.tolist(), std::vector<float>({4}));
    ASSERT_EQ(logits_grad_tensor.tolist(), std::vector<float>({0.25f, -0.75f, 0.25f, 0.25f}));
}

static void large_logits_do_not_overflow(void)
{
    // Expect: the row 1000 1001 1002 1003, target 3. exp(1003) is infinity in a float, but the kernels
    // take exp(logit - row_max), exp(-3) .. exp(0), so sum_of_each_row, loss and grad are those of 1 2 3 4:
    // adding the same number to every logit changes no prob. max_of_each_row is 1003.
    // 1. the forward, then the backward
    Tensor logits_tensor = tensor_of({1, 1, 4}, {1000, 1001, 1002, 1003});
    Tensor targets_tensor = targets_of({1, 1}, {3});
    RowResults results_val = zero_row_results({1, 1});
    Tensor logits_grad_tensor = Tensor::zeros({1, 1, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, Tensor::ones({1, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_EQ(results_val.max_of_each_row_tensor.tolist(), std::vector<float>({1003}));
    ASSERT_NEAR(results_val.sum_of_each_row_tensor.tolist()[0], 1.5530018, 1e-6);
    ASSERT_NEAR(results_val.losses_tensor.tolist()[0], 0.4401897, 1e-6);
    ASSERT_NEAR(logits_grad_tensor.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
}

static void padding_columns_are_left_out_and_get_grad_0(void)
{
    // Expect: vocab_size = 5 of padded_vocab_size = 8, the row 1 2 3 4 5 and 100 100 100 in the padding,
    // target 0. If the padding counted, its 100s would take nearly all the prob. Left out:
    //
    //     row_max 5, row_sum = e^-4 + e^-3 + e^-2 + e^-1 + 1 = 1.5713174
    //     loss  = log(1.5713174) - (1 - 5) = 4.4519144
    //     grad  = -0.9883438 0.0316849 0.0861285 0.2341217 0.6364086, then 0 0 0 exactly
    //
    // Float4 1 holds word 4 and padding 5, 6, 7: the padding is left out inside one float4.
    // 1. the forward, then the backward, logits_grad from 1s
    Tensor logits_tensor = tensor_of({1, 1, 8}, {1, 2, 3, 4, 5, 100, 100, 100});
    Tensor targets_tensor = targets_of({1, 1}, {0});
    RowResults results_val = zero_row_results({1, 1});
    Tensor logits_grad_tensor = Tensor::ones({1, 1, 8});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 5);
    classifier_backward(logits_grad_tensor, Tensor::ones({1, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 5);
    // 2. check: the words' grads, then the padding's 0s
    ASSERT_EQ(results_val.max_of_each_row_tensor.tolist(), std::vector<float>({5}));
    ASSERT_NEAR(results_val.losses_tensor.tolist()[0], 4.4519144, 1e-6);
    std::vector<float> grad_list = logits_grad_tensor.tolist();
    ASSERT_NEAR(std::vector<float>(grad_list.begin(), grad_list.begin() + 5),
                std::vector<float>({-0.9883438f, 0.0316849f, 0.0861285f, 0.2341217f, 0.6364086f}), 1e-6);
    ASSERT_EQ(std::vector<float>(grad_list.begin() + 5, grad_list.end()), std::vector<float>({0, 0, 0}));
}

static void a_sure_right_answer_costs_0_and_a_sure_wrong_one_costs_20(void)
{
    // Expect: two rows of 0 0 20 0, targets 2 and 0. The other logits have prob e^-20 = 2.1e-9 each,
    // so row_sum = 1 + 6.2e-9, which is 1 in a float.
    //
    //     row 0, target 2:  loss = log(1) - (20 - 20) = 0     grad about 0 0 0 0 (2.1e-9 and 0)
    //     row 1, target 0:  loss = log(1) - (0 - 20)  = 20    grad about -1 0 1 0
    //
    // Both losses exactly. A wrong guess made with certainty costs the most, and gets the largest push.
    // 1. the forward, then the backward
    Tensor logits_tensor = tensor_of({1, 2, 4}, {0, 0, 20, 0, 0, 0, 20, 0});
    Tensor targets_tensor = targets_of({1, 2}, {2, 0});
    RowResults results_val = zero_row_results({1, 2});
    Tensor logits_grad_tensor = Tensor::zeros({1, 2, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, Tensor::ones({1, 2}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_EQ(results_val.losses_tensor.tolist(), std::vector<float>({0, 20}));
    ASSERT_NEAR(logits_grad_tensor[0][0].tolist(), std::vector<float>({0, 0, 0, 0}), 1e-6);
    ASSERT_NEAR(logits_grad_tensor[0][1].tolist(), std::vector<float>({-1, 0, 1, 0}), 1e-6);
}

static void each_row_uses_its_own_target(void)
{
    // Expect: two rows of 1 2 3 4, targets 3 and 0. Same probs, other target:
    //
    //     row 0, target 3:  loss 0.4401897                         grad 0.0320586 0.0871443 0.2368828 -0.3560857
    //     row 1, target 0:  loss 0.4401897 + (4 - 1) = 3.4401897   grad -0.9679414 0.0871443 0.2368828 0.6439143
    // 1. the forward, then the backward
    Tensor logits_tensor = tensor_of({2, 1, 4}, {1, 2, 3, 4, 1, 2, 3, 4});
    Tensor targets_tensor = targets_of({2, 1}, {3, 0});
    RowResults results_val = zero_row_results({2, 1});
    Tensor logits_grad_tensor = Tensor::zeros({2, 1, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, Tensor::ones({2, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_NEAR(results_val.losses_tensor.tolist(), std::vector<float>({0.4401897f, 3.4401897f}), 1e-6);
    ASSERT_NEAR(logits_grad_tensor[0][0].tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
    ASSERT_NEAR(logits_grad_tensor[1][0].tolist(), std::vector<float>({-0.9679414f, 0.0871443f, 0.2368828f, 0.6439143f}),
                1e-6);
}

static void losses_grad_scales_each_row_on_its_own(void)
{
    // Expect: two rows of 1 2 3 4, target 3, losses_grad 1 and 0.25: row 0's grad is the one above, row
    // 1's a quarter of it, 0.0080147 0.0217861 0.0592207 -0.0890214. In training every row gets the
    // same, 1 / rows, for the mean.
    // 1. the forward, then the backward with losses_grad 1 and 0.25
    Tensor logits_tensor = tensor_of({2, 1, 4}, {1, 2, 3, 4, 1, 2, 3, 4});
    Tensor targets_tensor = targets_of({2, 1}, {3, 3});
    RowResults results_val = zero_row_results({2, 1});
    Tensor logits_grad_tensor = Tensor::zeros({2, 1, 4});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_grad_tensor, tensor_of({2, 1}, {1, 0.25f}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_NEAR(logits_grad_tensor[0][0].tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
    ASSERT_NEAR(logits_grad_tensor[1][0].tolist(), std::vector<float>({0.0080147f, 0.0217861f, 0.0592207f, -0.0890214f}),
                1e-6);
}

static void each_row_of_logits_grad_adds_up_to_0(void)
{
    // Expect: {2, 3, 50304} random logits, vocab_size = 50257, losses_grad 1. In each row the probs add
    // up to 1, and the target takes 1 away: each row of logits_grad adds up to 0, within 1e-5 (50257
    // rounded probs added). Raising every logit of a row by the same amount changes no prob, so the loss
    // cannot move that way.
    // 1. random logits; the forward, then the backward
    torch::manual_seed(0);
    size_t padded_vocab_size_val = GPT2_PADDED_VOCAB_SIZE;
    size_t vocab_size_val = GPT2_VOCAB_SIZE;
    Tensor logits_tensor = random_tensor_on_gpu({2, 3, padded_vocab_size_val});
    Tensor targets_tensor = random_targets_on_gpu({2, 3}, vocab_size_val);
    RowResults results_val = zero_row_results({2, 3});
    Tensor logits_grad_tensor = Tensor::zeros({2, 3, padded_vocab_size_val});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, vocab_size_val);
    classifier_backward(logits_grad_tensor, Tensor::ones({2, 3}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, vocab_size_val);
    // 2. each row's sum, in double
    std::vector<float> grad_list = logits_grad_tensor.tolist();
    std::vector<float> row_sums_list(6, 0.0f);
    for (size_t row_index_val = 0; row_index_val < 6; row_index_val++)
    {
        double sum_val = 0.0;
        for (size_t column_val = 0; column_val < padded_vocab_size_val; column_val++)
            sum_val += grad_list[row_index_val * padded_vocab_size_val + column_val];
        row_sums_list[row_index_val] = (float)sum_val;
    }
    // 3. check
    ASSERT_NEAR(row_sums_list, std::vector<float>(6, 0.0f), 1e-5);
}

static void logits_grad_can_be_the_logits_themselves(void)
{
    // Expect: the row 1 2 3 4, target 3, with the logits tensor passed as logits_grad too: the logits
    // become their gradient, the same as GRAD_OF_1_2_3_4_TARGET_3, since each thread reads its float4
    // before it writes it.
    // 1. the forward, then the backward into the logits
    Tensor logits_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor targets_tensor = targets_of({1, 1}, {3});
    RowResults results_val = zero_row_results({1, 1});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_tensor, 4);
    classifier_backward(logits_tensor, Tensor::ones({1, 1}), logits_tensor, targets_tensor,
                        results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor, 4);
    // 2. check
    ASSERT_NEAR(logits_tensor.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
}

static void forward_leaves_the_logits_as_they_are(void)
{
    // Expect: after classifier_forward on the row 1 2 3 4, the logits are still 1 2 3 4, exactly: the
    // validation loss can be worked out with the logits kept, for sampling.
    // 1. the forward
    Tensor logits_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    RowResults results_val = zero_row_results({1, 1});
    classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor, results_val.sum_of_each_row_tensor,
                       logits_tensor, targets_of({1, 1}, {3}), 4);
    // 2. check
    ASSERT_EQ(logits_tensor.tolist(), std::vector<float>({1, 2, 3, 4}));
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the forward can be launched with CLASSIFIER_FORWARD_THREADS_PER_BLOCK (256) and the backward
    // with CLASSIFIER_BACKWARD_THREADS_PER_BLOCK (128), and nothing is spilled. The compiler picks each
    // kernel's registers per thread; cudaFuncGetAttributes reads what it wrote down, without running
    // anything:
    //
    //     numRegs              registers per thread: at most 64 for 1024 threads (a T4 has 65,536)
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)classifier_forward_kernel<float>,
                                  (const void *)classifier_backward_kernel<float>};
    const char *names_list[] = {"classifier_forward_kernel", "classifier_backward_kernel"};
    int threads_list[] = {CLASSIFIER_FORWARD_THREADS_PER_BLOCK, CLASSIFIER_BACKWARD_THREADS_PER_BLOCK};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++)
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

static void any_block_size_gives_the_same_result(void)
{
    // Expect: {2, 8, 50304} random logits, in float and in bfloat16, both launchers called with
    // threads_per_block 32, 128, 512 and 1024 (one warp up to MAX_THREADS_PER_BLOCK). The forward gives the
    // defaults' losses, max_of_each_row and sum_of_each_row within TOLERANCE, not to the last bit: another
    // block size adds each row's exps in another order. The backward, from the default forward's row maxes
    // and sums, gives the default's logits_grad to the last bit: each value is worked out alone.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random logits and targets; the defaults' results
        torch::manual_seed(0);
        size_t batch_size_val = 2;
        size_t seq_len_val = 8;
        size_t rows_val = batch_size_val * seq_len_val;
        Shape logits_shape_list = {batch_size_val, seq_len_val, GPT2_PADDED_VOCAB_SIZE};
        Tensor logits_tensor = random_tensor_on_gpu(logits_shape_list).to(dtype_val);
        Tensor targets_tensor = random_targets_on_gpu({batch_size_val, seq_len_val}, GPT2_VOCAB_SIZE);
        Tensor losses_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val}).fill_(1.0 / rows_val);
        RowResults usual_val = zero_row_results({batch_size_val, seq_len_val});
        Tensor usual_logits_grad_tensor = Tensor::zeros(logits_shape_list, dtype_val);
        classifier_forward(usual_val.losses_tensor, usual_val.max_of_each_row_tensor, usual_val.sum_of_each_row_tensor,
                           logits_tensor, targets_tensor, GPT2_VOCAB_SIZE);
        classifier_backward(usual_logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor,
                            usual_val.max_of_each_row_tensor, usual_val.sum_of_each_row_tensor, GPT2_VOCAB_SIZE);

        // 2. each block size, against the defaults'
        for (int threads_per_block_val : block_sizes_list)
        {
            RowResults results_val = zero_row_results({batch_size_val, seq_len_val});
            Tensor logits_grad_tensor = Tensor::zeros(logits_shape_list, dtype_val);
            classifier_forward(results_val.losses_tensor, results_val.max_of_each_row_tensor,
                               results_val.sum_of_each_row_tensor, logits_tensor, targets_tensor, GPT2_VOCAB_SIZE,
                               threads_per_block_val);
            classifier_backward(logits_grad_tensor, losses_grad_tensor, logits_tensor, targets_tensor,
                                usual_val.max_of_each_row_tensor, usual_val.sum_of_each_row_tensor, GPT2_VOCAB_SIZE,
                                threads_per_block_val);
            ASSERT_NEAR(results_val.losses_tensor.tolist(), usual_val.losses_tensor.tolist(), TOLERANCE);
            ASSERT_NEAR(results_val.max_of_each_row_tensor.tolist(), usual_val.max_of_each_row_tensor.tolist(), TOLERANCE);
            ASSERT_NEAR(results_val.sum_of_each_row_tensor.tolist(), usual_val.sum_of_each_row_tensor.tolist(), TOLERANCE);
            ASSERT_EQ(logits_grad_tensor.to(DType::Float32).tolist(), usual_logits_grad_tensor.to(DType::Float32).tolist());
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_rows();
    matches_pytorch_with_a_small_vocabulary();
    forward_and_backward_in_bfloat16();

    forward_of_1_2_3_4();
    backward_of_1_2_3_4();
    equal_logits_give_log_of_the_vocabulary_size();
    large_logits_do_not_overflow();
    padding_columns_are_left_out_and_get_grad_0();
    a_sure_right_answer_costs_0_and_a_sure_wrong_one_costs_20();
    each_row_uses_its_own_target();
    losses_grad_scales_each_row_on_its_own();
    each_row_of_logits_grad_adds_up_to_0();
    logits_grad_can_be_the_logits_themselves();
    forward_leaves_the_logits_as_they_are();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_result();

    printf("\nall tests passed\n");
    return 0;
}
