// test_classifier.cu — tests for llmc/kernels/classifier.cuh (classifier_forward, classifier_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's cross_entropy, with autograd for the backward, and times them
// on GPT-2 small's sizes; part 2 checks numbers worked out by hand; part 3 checks that each kernel's
// registers fit its block; part 4 times each kernel with other block and thread counts. (The C#
// trainer had no classifier test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::cross_entropy_loss, torch::randn, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/classifier.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so one
// rounding moves a value by up to 2^-8, about 0.4% of it. Twice that.
static const double BFLOAT16_TOLERANCE = 8e-3;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape` on the GPU holding `values`, given in row order.
static Tensor tensor_of(const Shape &shape, const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty(shape);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes an int tensor of `shape` on the GPU holding `values`: the targets.
static Tensor targets_of(const Shape &shape, const std::vector<int> &values)
{
    Tensor tensor = Tensor::empty(shape, DType::Int32);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes a tensor of `shape` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes minutes for the logits' 206M values.
static Tensor random_tensor_on_gpu(const Shape &shape)
{
    Tensor tensor = Tensor::empty(shape);
    torch::Tensor values = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes an int tensor of `shape` with random targets 0 .. vocab_size - 1, made by PyTorch on the GPU.
static Tensor random_targets_on_gpu(const Shape &shape, size_t vocab_size)
{
    Tensor tensor = Tensor::empty(shape, DType::Int32);
    torch::TensorOptions ints_on_gpu = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    torch::Tensor values = torch::randint((int64_t)vocab_size, {(int64_t)tensor.numel()}, ints_on_gpu);
    cudaCheck(cudaMemcpy(tensor.data_ptr<int>(), values.data_ptr<int>(), tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes, on_gpu).clone();
}

/// Makes a PyTorch copy of my int targets as int64, the type cross_entropy wants its targets in.
static torch::Tensor pytorch_targets_of(const Tensor &targets)
{
    torch::TensorOptions ints_on_gpu = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes(targets.shape().begin(), targets.shape().end());
    return torch::from_blob(targets.data_ptr<int>(), sizes, ints_on_gpu).to(torch::kLong);
}

/// Gives the first vocab_size columns of pytorch_logits {batch_size, seq_len, padded_vocab_size} as
/// {rows, vocab_size}: the words, without the padding, the way cross_entropy takes them.
static torch::Tensor word_logits_of(const torch::Tensor &pytorch_logits, size_t vocab_size)
{
    int64_t rows = pytorch_logits.size(0) * pytorch_logits.size(1);
    return pytorch_logits.narrow(2, 0, (int64_t)vocab_size).reshape({rows, (int64_t)vocab_size});
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu.data_ptr<float>(), on_cpu.data_ptr<float>() + on_cpu.numel());
}

/// Gives the largest difference between my tensor and a PyTorch tensor of the same values, as a
/// fraction of the PyTorch tensor's largest value, worked out on the GPU: for tensors too big to copy
/// to the CPU and compare there.
static double relative_difference_on_gpu(const Tensor &mine, const torch::Tensor &pytorch)
{
    torch::Tensor difference = (pytorch_copy_of(mine).reshape(pytorch.sizes()) - pytorch).abs().max();
    return difference.item<double>() / pytorch.abs().max().item<double>();
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

/// Runs one forward and one backward on random logits and targets, with losses_grad 1 / rows in every
/// row (the mean), mine and PyTorch's cross_entropy over the first vocab_size columns with autograd.
/// Checks that the losses, max_of_each_row, sum_of_each_row and logits_grad agree within TOLERANCE, the padding's
/// gradient 0 on both sides.
static void forward_and_backward_match_pytorch(const Shape &logits_shape, size_t vocab_size)
{
    torch::manual_seed(0);
    size_t batch_size = logits_shape[0], seq_len = logits_shape[1], rows = batch_size * seq_len;
    Tensor logits = random_tensor_on_gpu(logits_shape);
    Tensor targets = random_targets_on_gpu({batch_size, seq_len}, vocab_size);
    Tensor losses = Tensor::zeros({batch_size, seq_len});
    Tensor max_of_each_row = Tensor::zeros({batch_size, seq_len}), sum_of_each_row = Tensor::zeros({batch_size, seq_len});
    Tensor losses_grad = Tensor::zeros({batch_size, seq_len}).fill_(1.0 / rows);
    Tensor logits_grad = Tensor::zeros(logits_shape);
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    classifier_backward(logits_grad, losses_grad, logits, targets, max_of_each_row, sum_of_each_row, vocab_size);

    torch::Tensor pytorch_logits = pytorch_copy_of(logits).requires_grad_(true);
    torch::Tensor word_logits = word_logits_of(pytorch_logits, vocab_size);
    torch::Tensor pytorch_losses = torch::cross_entropy_loss(word_logits, pytorch_targets_of(targets).reshape({-1}),
                                                             {}, at::Reduction::None);
    pytorch_losses.backward(pytorch_copy_of(losses_grad).reshape({-1}));
    torch::Tensor pytorch_max_of_each_row = std::get<0>(word_logits.max(1));
    torch::Tensor pytorch_sum_of_each_row = (word_logits - pytorch_max_of_each_row.unsqueeze(1)).exp().sum(1);
    ASSERT_NEAR(losses.tolist(), list_of(pytorch_losses), TOLERANCE);
    ASSERT_NEAR(max_of_each_row.tolist(), list_of(pytorch_max_of_each_row), TOLERANCE);
    ASSERT_NEAR(sum_of_each_row.tolist(), list_of(pytorch_sum_of_each_row), TOLERANCE);
    ASSERT_NEAR(logits_grad.tolist(), list_of(pytorch_logits.grad()), TOLERANCE);
}

static void matches_pytorch_on_gpt2_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens, GPT-2's padded_vocab_size = 50304 with
    // vocab_size = 50257 words: 16 blocks of the forward, each of the 1024 threads taking about 12 of a
    // row's 12,576 float4s; float4 12,564 holds the last word and 3 padding columns. Random logits and
    // targets: the losses (about log(50257) = 10.8 each, as random logits guess about evenly),
    // max_of_each_row, sum_of_each_row and logits_grad equal PyTorch's within 1e-5.
    forward_and_backward_match_pytorch({2, 8, 50304}, 50257);
}

static void matches_pytorch_with_a_small_vocabulary(void)
{
    // Expect: {3, 5, 12} with vocab_size = 10: 3 float4s per row for 1024 threads, so 1021 threads of
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
    torch::manual_seed(0);
    size_t batch_size = 2, seq_len = 8, rows = batch_size * seq_len, vocab_size = 50257;
    Shape logits_shape = {batch_size, seq_len, 50304};
    Tensor logits = random_tensor_on_gpu(logits_shape).to(DType::BFloat16);
    Tensor targets = random_targets_on_gpu({batch_size, seq_len}, vocab_size);
    Tensor losses = Tensor::zeros({batch_size, seq_len});
    Tensor max_of_each_row = Tensor::zeros({batch_size, seq_len}), sum_of_each_row = Tensor::zeros({batch_size, seq_len});
    Tensor losses_grad = Tensor::zeros({batch_size, seq_len}).fill_(1.0 / rows);
    Tensor logits_grad = Tensor::zeros(logits_shape, DType::BFloat16);
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    classifier_backward(logits_grad, losses_grad, logits, targets, max_of_each_row, sum_of_each_row, vocab_size);

    torch::Tensor pytorch_logits = pytorch_copy_of(logits.to(DType::Float32)).requires_grad_(true);
    torch::Tensor word_logits = word_logits_of(pytorch_logits, vocab_size);
    torch::Tensor pytorch_losses = torch::cross_entropy_loss(word_logits, pytorch_targets_of(targets).reshape({-1}),
                                                             {}, at::Reduction::None);
    pytorch_losses.backward(pytorch_copy_of(losses_grad).reshape({-1}));
    torch::Tensor pytorch_max_of_each_row = std::get<0>(word_logits.max(1));
    torch::Tensor pytorch_sum_of_each_row = (word_logits - pytorch_max_of_each_row.unsqueeze(1)).exp().sum(1);
    ASSERT_NEAR(losses.tolist(), list_of(pytorch_losses), TOLERANCE);
    ASSERT_NEAR(max_of_each_row.tolist(), list_of(pytorch_max_of_each_row), TOLERANCE);
    ASSERT_NEAR(sum_of_each_row.tolist(), list_of(pytorch_sum_of_each_row), TOLERANCE);
    ASSERT_NEAR(logits_grad.to(DType::Float32).tolist(), list_of(pytorch_logits.grad()), BFLOAT16_TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's sizes, batch_size = 4 rows of
    // seq_len = 1024 tokens, padded_vocab_size = 50304: logits of 206M floats, 824 MB. Mine against
    // PyTorch's cross_entropy (mean) and autograd. My forward reads the logits once: at a T4's 320 GB/s
    // at least 2.58 ms. My backward reads them once and writes logits_grad: 2 x 824 MB, at least
    // 5.15 ms. PyTorch's forward includes copying the 50257 word columns out of the padded rows (narrow,
    // then reshape), as a PyTorch user would write it. Both sides write the gradient over each step; at
    // the end they must agree.
    torch::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, vocab_size = 50257, padded_vocab_size = 50304;
    size_t rows = batch_size * seq_len;
    Tensor logits = random_tensor_on_gpu({batch_size, seq_len, padded_vocab_size});
    Tensor targets = random_targets_on_gpu({batch_size, seq_len}, vocab_size);
    Tensor losses = Tensor::zeros({batch_size, seq_len});
    Tensor max_of_each_row = Tensor::zeros({batch_size, seq_len}), sum_of_each_row = Tensor::zeros({batch_size, seq_len});
    Tensor losses_grad = Tensor::zeros({batch_size, seq_len}).fill_(1.0 / rows);
    Tensor logits_grad = Tensor::zeros({batch_size, seq_len, padded_vocab_size});

    torch::Tensor pytorch_logits = pytorch_copy_of(logits).requires_grad_(true);
    torch::Tensor pytorch_targets = pytorch_targets_of(targets).reshape({-1});
    torch::Tensor pytorch_losses, pytorch_loss;

    // Steps 1 and 2 do first-time work, so the averages are of 3..20: in step 1 CUDA loads each side's
    // kernels.
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size); });
        times[1] = gpu_milliseconds([&] {
            classifier_backward(logits_grad, losses_grad, logits, targets, max_of_each_row, sum_of_each_row, vocab_size);
        });
        times[2] = gpu_milliseconds([&] {
            pytorch_losses = torch::cross_entropy_loss(word_logits_of(pytorch_logits, vocab_size), pytorch_targets,
                                                       {}, at::Reduction::None);
            pytorch_loss = pytorch_losses.mean();
        });
        pytorch_logits.mutable_grad() = torch::Tensor(); // the gradient written, as mine, not added onto
        times[3] = gpu_milliseconds([&] { pytorch_loss.backward(); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    double logits_gb = (double)logits.numel() * sizeof(float) / 1e9;     // 0.824 GB
    double moved_gb[2] = {logits_gb, 2 * logits_gb};
    print_test_name(__func__);
    printf("    average of steps 3..20:   my forward   my backward   PyTorch forward   PyTorch backward\n");
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("        GB/s                  %10.1f   %11.1f\n", moved_gb[0] / (totals[0] / 18 / 1e3),
           moved_gb[1] / (totals[1] / 18 / 1e3));
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_NEAR(relative_difference_on_gpu(losses, pytorch_losses), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(logits_grad, pytorch_logits.grad()), 0.0, TOLERANCE);
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

static void forward_of_1_2_3_4(void)
{
    // Expect: the row 1 2 3 4, target 3: loss 0.4401897, max_of_each_row 4 exactly, sum_of_each_row 1.5530018.
    Tensor logits = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets_of({1, 1}, {3}), 4);
    ASSERT_NEAR(losses.tolist()[0], 0.4401897, 1e-6);
    ASSERT_EQ(max_of_each_row.tolist(), std::vector<float>({4}));
    ASSERT_NEAR(sum_of_each_row.tolist()[0], 1.5530018, 1e-6);
}

static void backward_of_1_2_3_4(void)
{
    // Expect: the row 1 2 3 4, target 3, losses_grad 1: the grad above, within 1e-6. It pushes the
    // target's logit up (-0.36: the loss falls as it rises) and every other logit down, each by its prob.
    Tensor logits = tensor_of({1, 1, 4}, {1, 2, 3, 4}), targets = targets_of({1, 1}, {3});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    Tensor logits_grad = Tensor::zeros({1, 1, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, Tensor::ones({1, 1}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_NEAR(logits_grad.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
}

static void equal_logits_give_log_of_the_vocabulary_size(void)
{
    // Expect: the row 0 0 0 0, target 1: every prob is 1 / 4, so the loss is log(4) = 1.3862944, the
    // loss of an even guess (for GPT-2's 50257 words, log(50257) = 10.82, where training starts).
    // sum_of_each_row is 4 and the grad 0.25 -0.75 0.25 0.25, both exactly: e^0 = 1, and 1 / 4 has no rounding.
    Tensor logits = Tensor::zeros({1, 1, 4}), targets = targets_of({1, 1}, {1});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    Tensor logits_grad = Tensor::zeros({1, 1, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, Tensor::ones({1, 1}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_NEAR(losses.tolist()[0], 1.3862944, 1e-6);
    ASSERT_EQ(sum_of_each_row.tolist(), std::vector<float>({4}));
    ASSERT_EQ(logits_grad.tolist(), std::vector<float>({0.25f, -0.75f, 0.25f, 0.25f}));
}

static void large_logits_do_not_overflow(void)
{
    // Expect: the row 1000 1001 1002 1003, target 3. exp(1003) is infinity in a float, but the kernels
    // take exp(logit - row_max), exp(-3) .. exp(0), so sum_of_each_row, loss and grad are those of 1 2 3 4:
    // adding the same number to every logit changes no prob. max_of_each_row is 1003.
    Tensor logits = tensor_of({1, 1, 4}, {1000, 1001, 1002, 1003}), targets = targets_of({1, 1}, {3});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    Tensor logits_grad = Tensor::zeros({1, 1, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, Tensor::ones({1, 1}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_EQ(max_of_each_row.tolist(), std::vector<float>({1003}));
    ASSERT_NEAR(sum_of_each_row.tolist()[0], 1.5530018, 1e-6);
    ASSERT_NEAR(losses.tolist()[0], 0.4401897, 1e-6);
    ASSERT_NEAR(logits_grad.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
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
    Tensor logits = tensor_of({1, 1, 8}, {1, 2, 3, 4, 5, 100, 100, 100}), targets = targets_of({1, 1}, {0});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    Tensor logits_grad = Tensor::ones({1, 1, 8});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 5);
    classifier_backward(logits_grad, Tensor::ones({1, 1}), logits, targets, max_of_each_row, sum_of_each_row, 5);
    ASSERT_EQ(max_of_each_row.tolist(), std::vector<float>({5}));
    ASSERT_NEAR(losses.tolist()[0], 4.4519144, 1e-6);
    std::vector<float> grad = logits_grad.tolist();
    ASSERT_NEAR(std::vector<float>(grad.begin(), grad.begin() + 5),
                std::vector<float>({-0.9883438f, 0.0316849f, 0.0861285f, 0.2341217f, 0.6364086f}), 1e-6);
    ASSERT_EQ(std::vector<float>(grad.begin() + 5, grad.end()), std::vector<float>({0, 0, 0}));
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
    Tensor logits = tensor_of({1, 2, 4}, {0, 0, 20, 0, 0, 0, 20, 0}), targets = targets_of({1, 2}, {2, 0});
    Tensor losses = Tensor::zeros({1, 2}), max_of_each_row = Tensor::zeros({1, 2}), sum_of_each_row = Tensor::zeros({1, 2});
    Tensor logits_grad = Tensor::zeros({1, 2, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, Tensor::ones({1, 2}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_EQ(losses.tolist(), std::vector<float>({0, 20}));
    ASSERT_NEAR(logits_grad[0][0].tolist(), std::vector<float>({0, 0, 0, 0}), 1e-6);
    ASSERT_NEAR(logits_grad[0][1].tolist(), std::vector<float>({-1, 0, 1, 0}), 1e-6);
}

static void each_row_uses_its_own_target(void)
{
    // Expect: two rows of 1 2 3 4, targets 3 and 0. Same probs, other target:
    //
    //     row 0, target 3:  loss 0.4401897                         grad 0.0320586 0.0871443 0.2368828 -0.3560857
    //     row 1, target 0:  loss 0.4401897 + (4 - 1) = 3.4401897   grad -0.9679414 0.0871443 0.2368828 0.6439143
    Tensor logits = tensor_of({2, 1, 4}, {1, 2, 3, 4, 1, 2, 3, 4}), targets = targets_of({2, 1}, {3, 0});
    Tensor losses = Tensor::zeros({2, 1}), max_of_each_row = Tensor::zeros({2, 1}), sum_of_each_row = Tensor::zeros({2, 1});
    Tensor logits_grad = Tensor::zeros({2, 1, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, Tensor::ones({2, 1}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_NEAR(losses.tolist(), std::vector<float>({0.4401897f, 3.4401897f}), 1e-6);
    ASSERT_NEAR(logits_grad[0][0].tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
    ASSERT_NEAR(logits_grad[1][0].tolist(), std::vector<float>({-0.9679414f, 0.0871443f, 0.2368828f, 0.6439143f}),
                1e-6);
}

static void losses_grad_scales_each_row_on_its_own(void)
{
    // Expect: two rows of 1 2 3 4, target 3, losses_grad 1 and 0.25: row 0's grad is the one above, row
    // 1's a quarter of it, 0.0080147 0.0217861 0.0592207 -0.0890214. In training every row gets the
    // same, 1 / rows, for the mean.
    Tensor logits = tensor_of({2, 1, 4}, {1, 2, 3, 4, 1, 2, 3, 4}), targets = targets_of({2, 1}, {3, 3});
    Tensor losses = Tensor::zeros({2, 1}), max_of_each_row = Tensor::zeros({2, 1}), sum_of_each_row = Tensor::zeros({2, 1});
    Tensor logits_grad = Tensor::zeros({2, 1, 4});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits_grad, tensor_of({2, 1}, {1, 0.25f}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_NEAR(logits_grad[0][0].tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
    ASSERT_NEAR(logits_grad[1][0].tolist(), std::vector<float>({0.0080147f, 0.0217861f, 0.0592207f, -0.0890214f}),
                1e-6);
}

static void each_row_of_logits_grad_adds_up_to_0(void)
{
    // Expect: {2, 3, 50304} random logits, vocab_size = 50257, losses_grad 1. In each row the probs add
    // up to 1, and the target takes 1 away: each row of logits_grad adds up to 0, within 1e-5 (50257
    // rounded probs added). Raising every logit of a row by the same amount changes no prob, so the loss
    // cannot move that way.
    torch::manual_seed(0);
    size_t padded_vocab_size = 50304, vocab_size = 50257;
    Tensor logits = random_tensor_on_gpu({2, 3, padded_vocab_size});
    Tensor targets = random_targets_on_gpu({2, 3}, vocab_size);
    Tensor losses = Tensor::zeros({2, 3}), max_of_each_row = Tensor::zeros({2, 3}), sum_of_each_row = Tensor::zeros({2, 3});
    Tensor logits_grad = Tensor::zeros({2, 3, padded_vocab_size});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    classifier_backward(logits_grad, Tensor::ones({2, 3}), logits, targets, max_of_each_row, sum_of_each_row, vocab_size);
    std::vector<float> grad = logits_grad.tolist();
    std::vector<float> row_sums(6, 0.0f);
    for (size_t row = 0; row < 6; row++)
    {
        double sum = 0.0;
        for (size_t column = 0; column < padded_vocab_size; column++)
            sum += grad[row * padded_vocab_size + column];
        row_sums[row] = (float)sum;
    }
    ASSERT_NEAR(row_sums, std::vector<float>(6, 0.0f), 1e-5);
}

static void logits_grad_can_be_the_logits_themselves(void)
{
    // Expect: the row 1 2 3 4, target 3, with the logits tensor passed as logits_grad too: the logits
    // become their gradient, the same as GRAD_OF_1_2_3_4_TARGET_3, since each thread reads its float4
    // before it writes it.
    Tensor logits = tensor_of({1, 1, 4}, {1, 2, 3, 4}), targets = targets_of({1, 1}, {3});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, 4);
    classifier_backward(logits, Tensor::ones({1, 1}), logits, targets, max_of_each_row, sum_of_each_row, 4);
    ASSERT_NEAR(logits.tolist(), GRAD_OF_1_2_3_4_TARGET_3, 1e-6);
}

static void forward_leaves_the_logits_as_they_are(void)
{
    // Expect: after classifier_forward on the row 1 2 3 4, the logits are still 1 2 3 4, exactly: the
    // validation loss can be worked out with the logits kept, for sampling.
    Tensor logits = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor losses = Tensor::zeros({1, 1}), max_of_each_row = Tensor::zeros({1, 1}), sum_of_each_row = Tensor::zeros({1, 1});
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets_of({1, 1}, {3}), 4);
    ASSERT_EQ(logits.tolist(), std::vector<float>({1, 2, 3, 4}));
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the forward can be launched with CLASSIFIER_FORWARD_THREADS_PER_BLOCK (1024) and the backward
    // with CLASSIFIER_BACKWARD_THREADS_PER_BLOCK (256), and nothing is spilled. The compiler picks each
    // kernel's registers per thread; cudaFuncGetAttributes reads what it wrote down, without running
    // anything:
    //
    //     numRegs              registers per thread: at most 64 for 1024 threads (a T4 has 65,536)
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels[] = {(const void *)classifier_forward_kernel<float>,
                             (const void *)classifier_backward_kernel<float>};
    const char *names[] = {"classifier_forward_kernel", "classifier_backward_kernel"};
    int threads[] = {CLASSIFIER_FORWARD_THREADS_PER_BLOCK, CLASSIFIER_BACKWARD_THREADS_PER_BLOCK};
    for (int k = 0; k < 2; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= threads[k], true);
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

/// Keeps the GPU busy with `run` for about a second before timing, so the first launches timed do not
/// run while its clock is still rising from idle.
template <typename Run>
static void warm_up_gpu(Run run)
{
    float total = 0.0f;
    while (total < 1000.0f)
        total += gpu_milliseconds(run);
}

static void time_each_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the forward and the backward on GPT-2 small's sizes (4096 rows of 50304), each
    // for 7 block counts x 4 thread counts, each the average of runs 2..20. The kernels are launched here
    // directly, <<<blocks, threads>>>, instead of through classifier_forward and classifier_backward,
    // which always launch 4096 x 1024 (one row per block) and 201,216 x 256 (one float4 per thread).
    //
    // Expect: every forward launch gives the usual losses, max_of_each_row and sum_of_each_row within TOLERANCE, not
    // to the last bit: another thread count adds each row's exps in another order. Every backward launch
    // gives the usual logits_grad to the last bit: each value is worked out alone, from the same numbers.
    // A T4 runs at most 1024 threads per SM, 40,960 in all.
    torch::manual_seed(0);
    size_t batch_size = 4, seq_len = 1024, vocab_size = 50257, padded_vocab_size = 50304;
    size_t rows = batch_size * seq_len;
    Tensor logits = random_tensor_on_gpu({batch_size, seq_len, padded_vocab_size});
    Tensor targets = random_targets_on_gpu({batch_size, seq_len}, vocab_size);
    Tensor losses = Tensor::zeros({batch_size, seq_len});
    Tensor max_of_each_row = Tensor::zeros({batch_size, seq_len}), sum_of_each_row = Tensor::zeros({batch_size, seq_len});
    Tensor losses_grad = Tensor::zeros({batch_size, seq_len}).fill_(1.0 / rows);
    Tensor logits_grad = Tensor::zeros({batch_size, seq_len, padded_vocab_size});

    // The usual launches' results, kept on the GPU as PyTorch tensors to compare each launch with.
    classifier_forward(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    classifier_backward(logits_grad, losses_grad, logits, targets, max_of_each_row, sum_of_each_row, vocab_size);
    torch::Tensor usual_losses = pytorch_copy_of(losses);
    torch::Tensor usual_max_of_each_row = pytorch_copy_of(max_of_each_row), usual_sum_of_each_row = pytorch_copy_of(sum_of_each_row);
    torch::Tensor usual_logits_grad = pytorch_copy_of(logits_grad);
    int usual_backward_blocks = (int)((logits.numel() / 4 + CLASSIFIER_BACKWARD_THREADS_PER_BLOCK - 1) /
                                      CLASSIFIER_BACKWARD_THREADS_PER_BLOCK);   // 201,216

    int blocks = 0, threads = 0;                             // the launch the 2 below use, set by the loops
    auto forward = [&] {
        classifier_forward_kernel<float><<<blocks, threads>>>(losses.data(), max_of_each_row.data(),
                                                              sum_of_each_row.data(), logits.data(),
                                                              targets.data_ptr<int>(), rows, vocab_size,
                                                              padded_vocab_size);
        cudaCheck(cudaGetLastError());
    };
    auto backward = [&] {
        classifier_backward_kernel<float><<<blocks, threads>>>(logits_grad.data(), losses_grad.data(), logits.data(),
                                                               targets.data_ptr<int>(), max_of_each_row.data(),
                                                               sum_of_each_row.data(), rows, vocab_size,
                                                               padded_vocab_size);
        cudaCheck(cudaGetLastError());
    };
    blocks = (int)rows;
    threads = CLASSIFIER_FORWARD_THREADS_PER_BLOCK;
    warm_up_gpu(forward);

    // The forward: losses, max_of_each_row and sum_of_each_row within TOLERANCE of the usual launch's.
    int forward_block_counts[] = {40, 80, 160, 320, 640, 1280, 4096};
    int thread_counts[] = {128, 256, 512, 1024};
    float fastest_ms = 1e9f;
    int fastest_blocks = 0, fastest_threads = 0, launches_with_the_same_result = 0;
    print_test_name(__func__);
    printf("    forward, time in ms:    blocks   threads   rows per block   time   same result\n");
    for (int block_count : forward_block_counts)
    {
        for (int thread_count : thread_counts)
        {
            blocks = block_count;
            threads = thread_count;
            float milliseconds = average_milliseconds(forward);
            if (milliseconds < fastest_ms)
            {
                fastest_ms = milliseconds;
                fastest_blocks = blocks;
                fastest_threads = threads;
            }
            losses.zero_();
            max_of_each_row.zero_();
            sum_of_each_row.zero_();
            forward();
            bool same = relative_difference_on_gpu(losses, usual_losses) <= TOLERANCE &&
                        relative_difference_on_gpu(max_of_each_row, usual_max_of_each_row) <= TOLERANCE &&
                        relative_difference_on_gpu(sum_of_each_row, usual_sum_of_each_row) <= TOLERANCE;
            if (same)
                launches_with_the_same_result++;
            int rows_per_block = (int)((rows + blocks - 1) / blocks); // the most any block takes
            const char *usual = blocks == (int)rows && threads == CLASSIFIER_FORWARD_THREADS_PER_BLOCK
                                    ? "   <- what classifier_forward launches"
                                    : "";
            printf("                            %6d   %7d   %14d   %6.3f   %-11s%s\n", blocks, threads,
                   rows_per_block, milliseconds, same ? "yes" : "NO", usual);
        }
    }
    printf("    fastest forward   %6d blocks x %4d threads: %.3f ms\n", fastest_blocks, fastest_threads, fastest_ms);
    ASSERT_EQ(launches_with_the_same_result, 7 * 4);

    // The backward: logits_grad equal to the usual launch's, to the last bit. It reads the usual
    // forward's max_of_each_row and sum_of_each_row, which the last forward above wrote again within TOLERANCE, so
    // put those back first.
    cudaCheck(cudaMemcpy(max_of_each_row.data(), usual_max_of_each_row.data_ptr<float>(), rows * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(sum_of_each_row.data(), usual_sum_of_each_row.data_ptr<float>(), rows * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    int backward_block_counts[] = {40, 160, 640, 2560, 10240, 40960, usual_backward_blocks};
    fastest_ms = 1e9f;
    launches_with_the_same_result = 0;
    printf("    backward, time in ms:   blocks   threads   float4s per thread   time   same result\n");
    for (int block_count : backward_block_counts)
    {
        for (int thread_count : thread_counts)
        {
            blocks = block_count;
            threads = thread_count;
            float milliseconds = average_milliseconds(backward);
            if (milliseconds < fastest_ms)
            {
                fastest_ms = milliseconds;
                fastest_blocks = blocks;
                fastest_threads = threads;
            }
            logits_grad.zero_();
            backward();
            bool same = torch::equal(pytorch_copy_of(logits_grad), usual_logits_grad);
            if (same)
                launches_with_the_same_result++;
            size_t threads_launched = (size_t)blocks * threads;
            int float4s_per_thread = (int)((logits.numel() / 4 + threads_launched - 1) / threads_launched); // the most
            const char *usual = blocks == usual_backward_blocks && threads == CLASSIFIER_BACKWARD_THREADS_PER_BLOCK
                                    ? "   <- what classifier_backward launches"
                                    : "";
            printf("                            %6d   %7d   %18d   %6.3f   %-11s%s\n", blocks, threads,
                   float4s_per_thread, milliseconds, same ? "yes" : "NO", usual);
        }
    }
    printf("    fastest backward  %6d blocks x %4d threads: %.3f ms\n", fastest_blocks, fastest_threads, fastest_ms);
    ASSERT_EQ(launches_with_the_same_result, 7 * 4);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_on_gpt2_rows();
    matches_pytorch_with_a_small_vocabulary();
    forward_and_backward_in_bfloat16();
    time_20_steps_on_gpt2_small_size();

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

    time_each_kernel_with_other_block_and_thread_counts();

    printf("\nall tests passed\n");
    return 0;
}
