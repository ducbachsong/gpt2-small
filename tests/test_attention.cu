// test_attention.cu — tests for llmc/kernels/attention.cuh (attention_forward, attention_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's scaled_dot_product_attention (causal), with autograd for the
// backward, and the weights against PyTorch's softmax of the masked scores, and times them on GPT-2
// small's size; part 1b checks flash attention (bfloat16, sm_80 or later) against PyTorch and against the
// attention steps in bfloat16, and times all three; part 2 checks numbers worked out by hand and what the mask must give; part 3 checks that
// each kernel's registers fit its block; part 4 times the two softmax kernels with other block and thread
// counts. (The C# trainer had no attention test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/attention.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values. A bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it, and attention rounds several times in a row:
// the scores before the softmax (a score of 4 by up to 0.008, which moves its weight by 0.8%), the weights,
// the outputs. The backward rounds da, ds and the three gradients on top, and ds = a (da - row_dot) takes
// two close numbers apart, so its errors are larger next to its own size.
static const double BFLOAT16_TOLERANCE = 2e-2;
static const double BFLOAT16_GRAD_TOLERANCE = 5e-2;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape` on the GPU holding `values`, given in row order.
static Tensor tensor_of(const Shape &shape, const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty(shape);
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Makes a tensor of `shape` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 9.4M qkv values.
static Tensor random_tensor_on_gpu(const Shape &shape)
{
    Tensor tensor = Tensor::empty(shape);
    torch::Tensor values = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats = tensor.to(DType::Float32);  // the tensor itself if it is float already
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats.data(), sizes, on_gpu).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu.data_ptr<float>(), on_cpu.data_ptr<float>() + on_cpu.numel());
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: GPT-2's weights are 50M values, too many to copy to the CPU and print.
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

/// Every tensor attention_forward and attention_backward take, for one size, made with 0s except qkv and
/// output_grad, which the test fills.
struct AttentionTensors
{
    Tensor qkv, output, query, key, value, weights, head_outputs;                        // forward
    Tensor output_grad, qkv_grad, head_outputs_grad, weights_grad, query_grad, key_grad, value_grad;   // backward

    /// Makes them for batch_size x num_heads heads of seq_len tokens and head_size columns, all of dtype.
    AttentionTensors(size_t batch_size, size_t num_heads, size_t seq_len, size_t head_size,
                     DType dtype = DType::Float32)
    {
        size_t embed_dim = num_heads * head_size;
        Shape per_head = {batch_size, num_heads, seq_len, head_size};
        Shape per_pair = {batch_size, num_heads, seq_len, seq_len};
        qkv = Tensor::zeros({batch_size, seq_len, 3 * embed_dim}, dtype);
        output = Tensor::zeros({batch_size, seq_len, embed_dim}, dtype);
        query = Tensor::zeros(per_head, dtype), key = Tensor::zeros(per_head, dtype);
        value = Tensor::zeros(per_head, dtype);
        weights = Tensor::zeros(per_pair, dtype), head_outputs = Tensor::zeros(per_head, dtype);
        output_grad = Tensor::zeros(output.shape(), dtype), qkv_grad = Tensor::zeros(qkv.shape(), dtype);
        head_outputs_grad = Tensor::zeros(per_head, dtype), weights_grad = Tensor::zeros(per_pair, dtype);
        query_grad = Tensor::zeros(per_head, dtype), key_grad = Tensor::zeros(per_head, dtype);
        value_grad = Tensor::zeros(per_head, dtype);
    }

    /// Runs attention_forward on these tensors.
    void forward() { attention_forward(output, query, key, value, weights, head_outputs, qkv); }

    /// Runs attention_backward on these tensors.
    void backward()
    {
        attention_backward(qkv_grad, output_grad, query, key, value, weights, head_outputs_grad, weights_grad,
                           query_grad, key_grad, value_grad);
    }
};

/// PyTorch's answers for one qkv and output_grad, the plain PyTorch way, as GPT-2's model.py does it: qkv
/// split into q, k, v, each cut into heads, then scaled_dot_product_attention with is_causal, and autograd.
/// weights is worked out on the side as GPT-2's original code does: softmax of the masked scores.
struct PytorchAttention
{
    torch::Tensor qkv, output, weights, logsumexp;                 // logsumexp of each row of masked scores

    /// Works out output, qkv.grad() and weights from my qkv and output_grad.
    PytorchAttention(const AttentionTensors &mine)
    {
        int64_t batch_size = mine.query.size(0), num_heads = mine.query.size(1), seq_len = mine.query.size(2);
        int64_t head_size = mine.query.size(3), embed_dim = num_heads * head_size;
        qkv = pytorch_copy_of(mine.qkv).requires_grad_(true);
        std::vector<torch::Tensor> parts = qkv.split(embed_dim, 2);
        auto heads = [&](const torch::Tensor &part) {
            return part.view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
        };
        torch::Tensor q = heads(parts[0]), k = heads(parts[1]), v = heads(parts[2]);
        torch::Tensor y = torch::scaled_dot_product_attention(q, k, v, {}, 0.0, true);
        output = y.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim});
        output.backward(pytorch_copy_of(mine.output_grad));

        torch::NoGradGuard no_grad;
        torch::Tensor scores = q.matmul(k.transpose(-2, -1)) / std::sqrt((double)head_size);
        torch::Tensor after = torch::ones({seq_len, seq_len}, scores.options().dtype(torch::kBool)).triu(1);
        weights = scores.masked_fill(after, -INFINITY).softmax(-1);
        logsumexp = scores.masked_fill(after, -INFINITY).logsumexp(-1);
    }
};

/// Every tensor attention_flash_forward and attention_flash_backward take, for one size, all bfloat16 but
/// logsumexp and row_dot (float), made with 0s except qkv and output_grad, which the test fills.
struct FlashTensors
{
    Tensor qkv, output, query, key, value, logsumexp;              // forward
    Tensor output_grad, qkv_grad, row_dot;                         // backward

    /// Makes them for batch_size x num_heads heads of seq_len tokens of 64 columns.
    FlashTensors(size_t batch_size, size_t num_heads, size_t seq_len)
    {
        size_t embed_dim = num_heads * FLASH_HEAD_SIZE;
        Shape per_head = {batch_size, num_heads, seq_len, FLASH_HEAD_SIZE};
        qkv = Tensor::zeros({batch_size, seq_len, 3 * embed_dim}, DType::BFloat16);
        output = Tensor::zeros({batch_size, seq_len, embed_dim}, DType::BFloat16);
        query = Tensor::zeros(per_head, DType::BFloat16), key = Tensor::zeros(per_head, DType::BFloat16);
        value = Tensor::zeros(per_head, DType::BFloat16);
        logsumexp = Tensor::zeros({batch_size, num_heads, seq_len});
        output_grad = Tensor::zeros(output.shape(), DType::BFloat16);
        qkv_grad = Tensor::zeros(qkv.shape(), DType::BFloat16);
        row_dot = Tensor::zeros({batch_size, num_heads, seq_len});
    }

    /// Runs attention_flash_forward on these tensors.
    void forward() { attention_flash_forward(output, query, key, value, logsumexp, qkv); }

    /// Runs attention_flash_backward on these tensors.
    void backward() { attention_flash_backward(qkv_grad, output_grad, output, query, key, value, logsumexp, row_dot); }
};

/// Tells if this GPU runs flash attention (sm_80 or later); if not, prints why the test that asked is skipped.
static bool flash_runs_here(const char *test)
{
    if (attention_flash_fits(DType::BFloat16, 1024, FLASH_HEAD_SIZE))
        return true;
    print_test_name(test);
    printf("    skipped: flash attention needs a GPU of compute capability 8.0 or more (A100, H100)\n");
    return false;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random qkv and output_grad, mine and PyTorch's, and checks that
/// output, weights and qkv_grad agree within TOLERANCE.
static void forward_and_backward_match_pytorch(size_t batch_size, size_t num_heads, size_t seq_len, size_t head_size)
{
    torch::manual_seed(0);
    AttentionTensors mine(batch_size, num_heads, seq_len, head_size);
    mine.qkv = random_tensor_on_gpu(mine.qkv.shape());
    mine.output_grad = random_tensor_on_gpu(mine.output.shape());
    mine.forward();
    mine.backward();

    PytorchAttention pytorch(mine);
    ASSERT_NEAR(mine.output.tolist(), list_of(pytorch.output), TOLERANCE);
    ASSERT_NEAR(mine.weights.tolist(), list_of(pytorch.weights), TOLERANCE);
    ASSERT_NEAR(mine.qkv_grad.tolist(), list_of(pytorch.qkv.grad()), TOLERANCE);
}

static void matches_pytorch_on_gpt2_heads(void)
{
    // Expect: batch_size = 2, 12 heads of 64 as GPT-2 small, seq_len = 64 tokens: qkv {2, 64, 2304}, weights
    // {2, 12, 64, 64}. output, weights and qkv_grad equal PyTorch's within 1e-5.
    forward_and_backward_match_pytorch(2, 12, 64, 64);
}

static void matches_pytorch_with_a_seq_len_of_37(void)
{
    // Expect: {3, 2, 37, 8}: rows of 37 weights, so in each warp's row, lane 5 has 2 columns and lane 6 one;
    // tokens 0 .. 30 leave some lanes with no column at all, which must add nothing. The same as PyTorch
    // within 1e-5.
    forward_and_backward_match_pytorch(3, 2, 37, 8);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_heads's {2, 12, 64, 64} with every tensor bfloat16: the six products
    // by cublasGemmStridedBatchedEx (CUDA_R_16BF, float sums), the kernels reading and writing bfloat16.
    // PyTorch is given the same bfloat16 values as floats and works in float: output and weights within
    // BFLOAT16_TOLERANCE, qkv_grad within BFLOAT16_GRAD_TOLERANCE.
    torch::manual_seed(0);
    AttentionTensors mine(2, 12, 64, 64, DType::BFloat16);
    mine.qkv = random_tensor_on_gpu(mine.qkv.shape()).to(DType::BFloat16);
    mine.output_grad = random_tensor_on_gpu(mine.output.shape()).to(DType::BFloat16);
    mine.forward();
    mine.backward();

    PytorchAttention pytorch(mine);
    ASSERT_NEAR(mine.output.to(DType::Float32).tolist(), list_of(pytorch.output), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine.weights.to(DType::Float32).tolist(), list_of(pytorch.weights), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(mine.qkv_grad.to(DType::Float32).tolist(), list_of(pytorch.qkv.grad()), BFLOAT16_GRAD_TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size: {4, 1024} tokens, 12 heads of
    // 64. Mine against PyTorch's scaled_dot_product_attention (causal) with autograd, which on a T4 is a
    // fused kernel that never writes the 201 MB of weights. Each side writes its gradient over each step;
    // at the end they must agree.
    torch::manual_seed(0);
    AttentionTensors mine(4, 12, 1024, 64);
    mine.qkv = random_tensor_on_gpu(mine.qkv.shape());
    mine.output_grad = random_tensor_on_gpu(mine.output.shape());

    torch::Tensor pytorch_qkv = pytorch_copy_of(mine.qkv).requires_grad_(true);
    torch::Tensor pytorch_output_grad = pytorch_copy_of(mine.output_grad);
    torch::Tensor pytorch_output;
    auto pytorch_forward = [&] {
        std::vector<torch::Tensor> parts = pytorch_qkv.split(768, 2);
        auto heads = [](const torch::Tensor &part) { return part.view({4, 1024, 12, 64}).transpose(1, 2); };
        torch::Tensor y = torch::scaled_dot_product_attention(heads(parts[0]), heads(parts[1]), heads(parts[2]), {},
                                                              0.0, true);
        pytorch_output = y.transpose(1, 2).contiguous().view({4, 1024, 768});
    };

    // Steps 1 and 2 do first-time work, so the averages are of 3..20: in step 1 CUDA loads each side's
    // kernels and cuBLAS picks its own.
    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { mine.forward(); });
        times[1] = gpu_milliseconds([&] { mine.backward(); });
        times[2] = gpu_milliseconds(pytorch_forward);
        pytorch_qkv.mutable_grad() = torch::Tensor(); // qkv's gradient written, as mine, not added onto
        times[3] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 4; column++)
            totals[column] += times[column];
    }
    print_test_name(__func__);
    printf("    average of steps 3..20:   my forward   my backward   PyTorch forward   PyTorch backward\n");
    printf("        ms                    %10.3f   %11.3f   %15.3f   %16.3f\n", totals[0] / 18, totals[1] / 18,
           totals[2] / 18, totals[3] / 18);
    printf("    PyTorch takes, against mine:                          %14.2fx   %15.2fx\n", totals[2] / totals[0],
           totals[3] / totals[1]);
    ASSERT_NEAR(relative_difference_on_gpu(mine.output, pytorch_output), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine.qkv_grad, pytorch_qkv.grad()), 0.0, TOLERANCE);
}

// ══ part 1b: flash attention against PyTorch, and against the attention steps ════

/// Runs flash attention's forward and backward on random bfloat16 qkv and output_grad, and the attention
/// steps on the same values in bfloat16, and PyTorch in float on them too; checks that flash attention's
/// output and logsumexp are within BFLOAT16_TOLERANCE of PyTorch's, and its qkv_grad within
/// BFLOAT16_GRAD_TOLERANCE, and that it is as close to the steps' bfloat16 results.
static void flash_matches_pytorch_and_the_steps(size_t batch_size, size_t num_heads, size_t seq_len)
{
    torch::manual_seed(0);
    AttentionTensors steps(batch_size, num_heads, seq_len, FLASH_HEAD_SIZE, DType::BFloat16);
    steps.qkv = random_tensor_on_gpu(steps.qkv.shape()).to(DType::BFloat16);
    steps.output_grad = random_tensor_on_gpu(steps.output.shape()).to(DType::BFloat16);
    steps.forward();
    steps.backward();
    FlashTensors flash(batch_size, num_heads, seq_len);
    flash.qkv.copy_(steps.qkv);
    flash.output_grad.copy_(steps.output_grad);
    flash.forward();
    flash.backward();

    PytorchAttention pytorch(steps);
    ASSERT_NEAR(relative_difference_on_gpu(flash.output, pytorch.output), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash.logsumexp, pytorch.logsumexp), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash.qkv_grad, pytorch.qkv.grad()), 0.0, BFLOAT16_GRAD_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash.output, pytorch_copy_of(steps.output)), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash.qkv_grad, pytorch_copy_of(steps.qkv_grad)), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

static void flash_matches_on_one_tile(void)
{
    // Expect: {1, 2, 64}: one tile of 64 queries and 64 keys, all on the diagonal, so every score goes through
    // the mask. Within the bfloat16 tolerances of PyTorch and of the steps.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(1, 2, 64);
}

static void flash_matches_on_gpt2_heads_of_256_tokens(void)
{
    // Expect: {2, 12, 256}: 4 tiles per head, so tiles below the diagonal (no mask), the online softmax's
    // max moving from tile to tile, the backward's key blocks walking 4, 3, 2, 1 query tiles. The same.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(2, 12, 256);
}

static void flash_matches_on_gpt2_small_size(void)
{
    // Expect: {4, 12, 1024}, as training runs it: 16 tiles per head, 768 blocks. The same.
    if (flash_runs_here(__func__))
        flash_matches_pytorch_and_the_steps(4, 12, 1024);
}

static void flash_needs_bfloat16_head_size_64_and_whole_tiles(void)
{
    // Expect: attention_flash_fits says no to float, to a seq_len that is not a multiple of 64 (1000) and to a
    // head_size other than 64 (32); and yes to bfloat16, 1024 and 64 exactly when the GPU is sm_80 or later.
    int device = 0, major = 0;
    cudaCheck(cudaGetDevice(&device));
    cudaCheck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    ASSERT_EQ(attention_flash_fits(DType::Float32, 1024, 64), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, 1000, 64), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, 1024, 32), false);
    ASSERT_EQ(attention_flash_fits(DType::BFloat16, 1024, 64), major >= 8);
}

static void flash_backward_writes_over_qkv_grad_the_same_every_time(void)
{
    // Expect: qkv_grad filled with 7s first: after a backward, no 7 is left (every value of its three parts
    // is written), and a second backward gives the same values to the last bit (no two blocks add onto one
    // value, so nothing depends on which block finishes first).
    if (!flash_runs_here(__func__))
        return;
    torch::manual_seed(0);
    FlashTensors flash(2, 12, 256);
    flash.qkv = random_tensor_on_gpu(flash.qkv.shape()).to(DType::BFloat16);
    flash.output_grad = random_tensor_on_gpu(flash.output.shape()).to(DType::BFloat16);
    flash.forward();
    flash.qkv_grad.fill_(7.0);
    flash.backward();
    torch::Tensor first = pytorch_copy_of(flash.qkv_grad);
    ASSERT_EQ((first == 7.0f).sum().item<int64_t>(), (int64_t)0);
    flash.backward();
    ASSERT_EQ(torch::equal(pytorch_copy_of(flash.qkv_grad), first), true);
}

static void time_flash_against_the_steps_and_pytorch(void)
{
    // Prints the times of 20 forwards and 20 backwards on GPT-2 small's size, {4, 1024} tokens, 12 heads of
    // 64, all in bfloat16: flash attention, the attention steps (cuBLAS products, the softmax kernels, the 201
    // MB of weights) and PyTorch's scaled_dot_product_attention (causal), which in bfloat16 on an sm_80 GPU is
    // its own flash attention. Averages of steps 3..20. Expect flash attention faster than the steps, and at
    // the end its output and qkv_grad within the bfloat16 tolerances of PyTorch's.
    if (!flash_runs_here(__func__))
        return;
    torch::manual_seed(0);
    AttentionTensors steps(4, 12, 1024, 64, DType::BFloat16);
    steps.qkv = random_tensor_on_gpu(steps.qkv.shape()).to(DType::BFloat16);
    steps.output_grad = random_tensor_on_gpu(steps.output.shape()).to(DType::BFloat16);
    FlashTensors flash(4, 12, 1024);
    flash.qkv.copy_(steps.qkv);
    flash.output_grad.copy_(steps.output_grad);

    torch::Tensor pytorch_qkv = pytorch_copy_of(steps.qkv).to(torch::kBFloat16).requires_grad_(true);
    torch::Tensor pytorch_output_grad = pytorch_copy_of(steps.output_grad).to(torch::kBFloat16);
    torch::Tensor pytorch_output;
    auto pytorch_forward = [&] {
        std::vector<torch::Tensor> parts = pytorch_qkv.split(768, 2);
        auto heads = [](const torch::Tensor &part) { return part.view({4, 1024, 12, 64}).transpose(1, 2); };
        torch::Tensor y = torch::scaled_dot_product_attention(heads(parts[0]), heads(parts[1]), heads(parts[2]), {},
                                                              0.0, true);
        pytorch_output = y.transpose(1, 2).contiguous().view({4, 1024, 768});
    };

    double totals[6] = {};
    for (int t = 1; t <= 20; t++)
    {
        float times[6];
        times[0] = gpu_milliseconds([&] { flash.forward(); });
        times[1] = gpu_milliseconds([&] { flash.backward(); });
        times[2] = gpu_milliseconds([&] { steps.forward(); });
        times[3] = gpu_milliseconds([&] { steps.backward(); });
        times[4] = gpu_milliseconds(pytorch_forward);
        pytorch_qkv.mutable_grad() = torch::Tensor();
        times[5] = gpu_milliseconds([&] { pytorch_output.backward(pytorch_output_grad); });
        if (t <= 2)
            continue;
        for (int column = 0; column < 6; column++)
            totals[column] += times[column];
    }
    print_test_name(__func__);
    printf("    average of steps 3..20, ms:   forward   backward   both\n");
    const char *sides[] = {"flash attention", "the steps", "PyTorch (flash)"};
    for (int side = 0; side < 3; side++)
        printf("        %-22s %10.3f %10.3f %6.3f\n", sides[side], totals[2 * side] / 18, totals[2 * side + 1] / 18,
               (totals[2 * side] + totals[2 * side + 1]) / 18);
    printf("    the steps take %.2fx as long as flash attention, PyTorch %.2fx\n",
           (totals[2] + totals[3]) / (totals[0] + totals[1]), (totals[4] + totals[5]) / (totals[0] + totals[1]));
    ASSERT_EQ(totals[0] + totals[1] < totals[2] + totals[3], true);
    ASSERT_NEAR(relative_difference_on_gpu(flash.output, pytorch_output.to(torch::kFloat)), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(flash.qkv_grad, pytorch_qkv.grad().to(torch::kFloat)), 0.0,
                BFLOAT16_GRAD_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example: one head of 3 tokens with head_size 4, so qkv is {1, 3, 12}, each
// row q | k | v. scale = 1 / sqrt(4) = 0.5.
//
//     token   q            k            v            scores (0.5 q · k), columns 0 .. token   weights
//     0       2 0 0 0      1 0 0 0      1 0 0 0      1                                        1
//     1       0 2 0 0      0 1 0 0      0 1 0 0      0  1                                     0.2689414  0.7310586
//     2       2 2 0 0      1 1 0 0      0 0 1 0      1  1  2                                  0.2119416  0.2119416  0.5761169
//
// Token 1: e^0 / (e^0 + e^1) = 1 / 3.7182818 = 0.2689414. Each v has one 1, so output row t is weights row t
// followed by 0s. With output_grad 1 0 0 0 in every row (the loss wants output column 0 up: the weight on
// token 0):
//
//     da = output_grad · vᵀ        1 0 0 in every row
//     ds = a (da - row_dot)        token 1: 0.2689414 x (1 - 0.2689414) = 0.1966119, and -0.1966119
//                                  token 2: 0.1670223  -0.0449192  -0.1221031         (token 0: 0)
//     dq = 0.5 ds · k,  dk = 0.5 dsᵀ · q,  dv = aᵀ · output_grad

static const std::vector<float> NOTE_QKV = {2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                            0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                            2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0};
static const std::vector<float> NOTE_WEIGHTS = {1, 0, 0,
                                                0.2689414f, 0.7310586f, 0,
                                                0.2119416f, 0.2119416f, 0.5761169f};

/// Makes the note's example: one head of 3 tokens, head_size 4, qkv as above, output_grad 1 0 0 0 per row.
static AttentionTensors note_example(void)
{
    AttentionTensors note(1, 1, 3, 4);
    note.qkv = tensor_of({1, 3, 12}, NOTE_QKV);
    note.output_grad = tensor_of({1, 3, 4}, {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0});
    return note;
}

static void forward_of_the_note_s_example(void)
{
    // Expect: the weights above, and output row t = weights row t with a 0 after it, within 1e-6.
    AttentionTensors note = note_example();
    note.forward();
    ASSERT_NEAR(note.weights.tolist(), NOTE_WEIGHTS, 1e-6);
    ASSERT_NEAR(note.output.tolist(), std::vector<float>({1, 0, 0, 0,
                                                          0.2689414f, 0.7310586f, 0, 0,
                                                          0.2119416f, 0.2119416f, 0.5761169f, 0}), 1e-6);
}

static void backward_of_the_note_s_example(void)
{
    // Expect: qkv_grad row t = dq | dk | dv of token t, worked out above:
    //
    //     token   dq                          dk                          dv
    //     0       0          0          0 0   0.1670223  0.3636343  0 0   1.4808830  0 0 0
    //     1       0.0983060  -0.0983060 0 0   -0.0449192 -0.2415312 0 0   0.9430001  0 0 0
    //     2       0.0224596  -0.0835112 0 0   -0.1221031 -0.1221031 0 0   0.5761169  0 0 0
    //
    // dv of token 0 is 1 + 0.2689414 + 0.2119416: every token's weight on token 0. And weights_grad ends
    // holding ds. Within 1e-6.
    AttentionTensors note = note_example();
    note.forward();
    note.backward();
    ASSERT_NEAR(note.qkv_grad.tolist(),
                std::vector<float>({0, 0, 0, 0, 0.1670223f, 0.3636343f, 0, 0, 1.4808830f, 0, 0, 0,
                                    0.0983060f, -0.0983060f, 0, 0, -0.0449192f, -0.2415312f, 0, 0, 0.9430001f, 0, 0, 0,
                                    0.0224596f, -0.0835112f, 0, 0, -0.1221031f, -0.1221031f, 0, 0, 0.5761169f, 0, 0, 0}),
                1e-6);
    ASSERT_NEAR(note.weights_grad.tolist(), std::vector<float>({0, 0, 0,
                                                                0.1966119f, -0.1966119f, 0,
                                                                0.1670223f, -0.0449192f, -0.1221031f}), 1e-6);
}

static void weights_after_the_token_are_0_and_each_row_adds_up_to_1(void)
{
    // Expect: random qkv, {1, 2, 8, 4}: in every row t of weights, columns t + 1 .. 7 are exactly 0 (F3),
    // and the row adds up to 1 within 1e-6 (F4).
    torch::manual_seed(1);
    AttentionTensors tensors(1, 2, 8, 4);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.forward();
    std::vector<float> weights = tensors.weights.tolist();
    std::vector<float> after_the_token, row_sums;
    for (size_t row = 0; row < 2 * 8; row++)
    {
        size_t token = row % 8;
        float row_sum = 0.0f;
        for (size_t column = 0; column < 8; column++)
        {
            float weight = weights[row * 8 + column];
            if (column > token)
                after_the_token.push_back(weight);
            row_sum += weight;
        }
        row_sums.push_back(row_sum);
    }
    ASSERT_EQ(after_the_token, std::vector<float>(after_the_token.size(), 0.0f));
    ASSERT_NEAR(row_sums, std::vector<float>(16, 1.0f), 1e-6);
}

static void the_first_token_sees_only_itself(void)
{
    // Expect: token 0 sees column 0 only, so its weight is exp(0) / exp(0) = 1 exactly, and its output is its
    // own v, exactly: 1 x v + 0 x the others. Random qkv, {1, 2, 8, 4}: output row 0 = v of token 0 for both
    // heads, qkv row 0's last 8 values.
    torch::manual_seed(2);
    AttentionTensors tensors(1, 2, 8, 4);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.forward();
    std::vector<float> qkv = tensors.qkv.tolist(), output = tensors.output.tolist();
    ASSERT_EQ(std::vector<float>(output.begin(), output.begin() + 8), std::vector<float>(qkv.begin() + 16, qkv.begin() + 24));
}

static void a_later_token_does_not_change_earlier_outputs(void)
{
    // Expect: random qkv, {1, 2, 8, 8}; then token 7's whole qkv row is set to 100 and the forward run again.
    // Tokens 0 .. 6 cannot see token 7, so their output rows are the same to the last bit: token 7's scores
    // are in their rows only after them, where the weights are 0, and 0 x 100 adds 0.
    torch::manual_seed(3);
    AttentionTensors tensors(1, 2, 8, 8);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.forward();
    std::vector<float> before = tensors.output.tolist();
    tensors.qkv[0][7].fill_(100);
    tensors.forward();
    std::vector<float> after = tensors.output.tolist();
    size_t first_7_rows = 7 * 16;
    ASSERT_EQ(std::vector<float>(after.begin(), after.begin() + first_7_rows),
              std::vector<float>(before.begin(), before.begin() + first_7_rows));
}

static void equal_scores_give_the_mean_of_the_values(void)
{
    // Expect: q = 0, so every score is 0 and token t gives each of its t + 1 columns 1 / (t + 1): its output is
    // the mean of v over tokens 0 .. t. With v of token t = t t t t, output row t is the mean of 0 .. t = t / 2:
    // 0, 0.5, 1, 1.5 in all 4 columns, within 1e-6. One head of 4 tokens, head_size 4.
    AttentionTensors tensors(1, 1, 4, 4);
    std::vector<float> qkv(4 * 12, 0.0f);
    for (int token = 0; token < 4; token++)
        for (int column = 8; column < 12; column++)
            qkv[token * 12 + column] = (float)token;
    tensors.qkv = tensor_of({1, 4, 12}, qkv);
    tensors.forward();
    ASSERT_NEAR(tensors.output.tolist(), std::vector<float>({0, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f,
                                                             1, 1, 1, 1, 1.5f, 1.5f, 1.5f, 1.5f}), 1e-6);
}

static void large_scores_do_not_overflow(void)
{
    // Expect: q = 1000 0 0 0 for every token, k of token j = j 0 0 0, so token 2's scores are 0.5 x 1000 x j =
    // 0, 500, 1000. exp(1000) is infinity in a float, but after taking the row's max, exp(-1000) and exp(-500)
    // are 0 and exp(0) is 1: weights 0 0 1, exactly. Each token puts all its weight on itself, and its output
    // is its own v = (token + 1) 0 0 0: 1, 2, 3 in column 0, exactly.
    std::vector<float> qkv(3 * 12, 0.0f);
    for (int token = 0; token < 3; token++)
    {
        qkv[token * 12 + 0] = 1000;
        qkv[token * 12 + 4] = (float)token;
        qkv[token * 12 + 8] = (float)(token + 1);
    }
    AttentionTensors tensors(1, 1, 3, 4);
    tensors.qkv = tensor_of({1, 3, 12}, qkv);
    tensors.forward();
    ASSERT_EQ(tensors.weights.tolist(), std::vector<float>({1, 0, 0, 0, 1, 0, 0, 0, 1}));
    ASSERT_EQ(tensors.output.tolist(), std::vector<float>({1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}));
}

static void heads_do_not_mix(void)
{
    // Expect: random qkv, {1, 2, 8, 4}; then head 1's columns of q, k and v are set to 100 for every token
    // and the forward run again. Head 0's half of every output row (its first 4 columns) is the same to the
    // last bit: each head's scores, weights and mix use its own columns only.
    torch::manual_seed(4);
    AttentionTensors tensors(1, 2, 8, 4);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.forward();
    std::vector<float> before = tensors.output.tolist();
    std::vector<float> qkv = tensors.qkv.tolist();
    for (size_t token = 0; token < 8; token++)
        for (size_t part = 0; part < 3; part++)          // q, k, v: 8 columns each, head 1 the last 4
            for (size_t column = 4; column < 8; column++)
                qkv[token * 24 + part * 8 + column] = 100;
    tensors.qkv.copy_from_cpu(qkv.data());
    tensors.forward();
    std::vector<float> after = tensors.output.tolist(), head_0_before, head_0_after;
    for (size_t token = 0; token < 8; token++)
        for (size_t column = 0; column < 4; column++)
        {
            head_0_before.push_back(before[token * 8 + column]);
            head_0_after.push_back(after[token * 8 + column]);
        }
    ASSERT_EQ(head_0_after, head_0_before);
}

static void a_gradient_on_the_first_token_reaches_only_its_v(void)
{
    // Expect: random qkv, {1, 2, 8, 4}, and output_grad random in row 0, 0 in the others. Output row 0 is
    // 1 x v of token 0 (the_first_token_sees_only_itself), so the only gradient is v's of token 0, which is
    // output_grad row 0 itself; q and k of token 0 get ds = 1 x (da - 1 x da) = 0, and every other token 0,
    // all exactly.
    torch::manual_seed(5);
    AttentionTensors tensors(1, 2, 8, 4);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    std::vector<float> output_grad(8 * 8, 0.0f), row_0 = Tensor::randn({8}).tolist();
    std::copy(row_0.begin(), row_0.end(), output_grad.begin());
    tensors.output_grad = tensor_of({1, 8, 8}, output_grad);
    tensors.forward();
    tensors.backward();
    std::vector<float> expected(8 * 24, 0.0f);                // row 0: q 0, k 0, v = output_grad row 0
    std::copy(row_0.begin(), row_0.end(), expected.begin() + 16);
    ASSERT_EQ(tensors.qkv_grad.tolist(), expected);
}

static void each_row_of_scores_grad_adds_up_to_0(void)
{
    // Expect: random qkv and output_grad, {1, 2, 8, 4}. After the backward, weights_grad holds ds, and each
    // row adds up to 0 within 1e-6: adding one number to every score of a row changes no weight (note 5).
    // The columns after each token are exactly 0.
    torch::manual_seed(6);
    AttentionTensors tensors(1, 2, 8, 4);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.output_grad = random_tensor_on_gpu(tensors.output.shape());
    tensors.forward();
    tensors.backward();
    std::vector<float> scores_grad = tensors.weights_grad.tolist(), row_sums, after_the_token;
    for (size_t row = 0; row < 16; row++)
    {
        float row_sum = 0.0f;
        for (size_t column = 0; column < 8; column++)
        {
            row_sum += scores_grad[row * 8 + column];
            if (column > row % 8)
                after_the_token.push_back(scores_grad[row * 8 + column]);
        }
        row_sums.push_back(row_sum);
    }
    ASSERT_NEAR(row_sums, std::vector<float>(16, 0.0f), 1e-6);
    ASSERT_EQ(after_the_token, std::vector<float>(after_the_token.size(), 0.0f));
}

static void backward_writes_over_qkv_grad(void)
{
    // Expect: qkv_grad starts at 10 (left from somewhere else); the backward writes over it, so after it
    // qkv_grad is the note's, with no 10 added: v of token 0 gets 1.4808830, not 11.4808830.
    AttentionTensors note = note_example();
    note.qkv_grad.fill_(10);
    note.forward();
    note.backward();
    ASSERT_NEAR(note.qkv_grad[0][0].tolist()[8], 1.4808830, 1e-6);
    ASSERT_NEAR(note.qkv_grad[0][1].tolist()[0], 0.0983060, 1e-6);
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (256; flash attention's 128), and nothing
    // is spilled. The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it
    // wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    //
    // Flash attention's key_grad kernel holds the most: its 16 keys and values as tiles, key_grad, value_grad,
    // the weights and scores_grad, 160 registers of them, under the 255 a thread may have. On a GPU older than
    // sm_80 the flash kernels are empty, and fit trivially.
    const void *kernels[] = {(const void *)attention_split_qkv_kernel<float>,
                             (const void *)attention_softmax_forward_kernel<float>,
                             (const void *)attention_merge_heads_kernel<float>,
                             (const void *)attention_split_heads_kernel<float>,
                             (const void *)attention_softmax_backward_kernel<float>,
                             (const void *)attention_merge_qkv_grad_kernel<float>,
                             (const void *)attention_flash_forward_kernel,
                             (const void *)attention_flash_row_dot_kernel,
                             (const void *)attention_flash_key_grad_kernel,
                             (const void *)attention_flash_query_grad_kernel};
    const char *names[] = {"attention_split_qkv_kernel", "attention_softmax_forward_kernel",
                           "attention_merge_heads_kernel", "attention_split_heads_kernel",
                           "attention_softmax_backward_kernel", "attention_merge_qkv_grad_kernel",
                           "attention_flash_forward_kernel", "attention_flash_row_dot_kernel",
                           "attention_flash_key_grad_kernel", "attention_flash_query_grad_kernel"};
    int threads[] = {ATTENTION_THREADS_PER_BLOCK, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK,
                     ATTENTION_THREADS_PER_BLOCK, ATTENTION_THREADS_PER_BLOCK,
                     ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK, ATTENTION_THREADS_PER_BLOCK,
                     FLASH_THREADS, 256, FLASH_THREADS, FLASH_THREADS};
    for (int k = 0; k < 10; k++)
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

/// Keeps the GPU busy with `run` for about a second before timing, so the first launches timed do not
/// run while its clock is still rising from idle.
template <typename Run>
static void warm_up_gpu(Run run)
{
    float total = 0.0f;
    while (total < 1000.0f)
        total += gpu_milliseconds(run);
}

static void time_each_softmax_kernel_with_other_block_and_thread_counts(void)
{
    // Prints the time of the two softmax kernels on GPT-2 small's weights, {4, 12, 1024, 1024} (49,152 rows),
    // in float and in bfloat16, for 9 block counts x 4 thread counts, each the average of 20 runs: the 7 picked
    // on a T4, then this GPU's SM count x 1 and x 2 (114 and 228 on an H100 PCIe, 132 and 264 on an SXM;
    // on a T4 they repeat 40 and 80). The 36 launches take turns: 20 rounds, each timing every launch once (the forward and the backward in float,
    // then both in bfloat16). One launch's 20 runs in a row would all land in the same stretch of the test,
    // and a T4's clock can drift as it heats up (the first launch timed would get the fastest clock); taking
    // turns gives every launch runs from early, middle and late alike. Run 1 is counted too: warm_up_gpu has
    // already done the first-time work. They are launched here directly, <<<blocks, threads>>>, instead of
    // through attention_forward and attention_backward. Their launches are the fastest in bfloat16, the dtype
    // training runs in, on an H100 PCIe: both one warp per row, 6144 x ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK
    // and 6144 x ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK (256 each). A T4's sweep had picked 80 x 128 for
    // the backward (320 warps, 154 rows each), 5.3x slower on the H100. The two cuBLAS products around them are
    // cuBLAS's to launch, and the four kernels that move values are the same kind as gelu.cuh's. A bfloat16
    // row is 2 KB instead of 4 KB, so twice as many warps' rows fit in the T4's 4 MB L2: its best launch
    // may differ.
    //
    // Both kernels work in place, so the timed runs change the values; before each check of the result,
    // the scores (or da) are copied back from a kept copy. Expect: every launch gives the same weights and
    // ds as the usual ones of its dtype, to the last bit: each row is worked out by one warp, the same way
    // whichever.
    torch::manual_seed(0);
    AttentionTensors tensors(4, 12, 1024, 64);
    tensors.qkv = random_tensor_on_gpu(tensors.qkv.shape());
    tensors.output_grad = random_tensor_on_gpu(tensors.output.shape());
    tensors.forward();
    tensors.backward();                                      // fills weights, and the scratch the steps need

    size_t rows = 4 * 12 * 1024, seq_len = 1024;
    // The scores (F2's product) and da (B2's), kept to start each check from; then the usual launches' results.
    Tensor scores = Tensor::empty(tensors.weights.shape()), weights_grad_before = Tensor::empty(tensors.weights.shape());
    float scale = 0.125f, one = 1.0f, zero = 0.0f;
    attentionCublasCheck(cublasSgemmStridedBatched(attention_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1024, 64, &scale,
                                                   tensors.key.data(), 64, 1024 * 64, tensors.query.data(), 64,
                                                   1024 * 64, &zero, scores.data(), 1024, 1024 * 1024, 48));
    attentionCublasCheck(cublasSgemmStridedBatched(attention_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1024, 64, &one,
                                                   tensors.value.data(), 64, 1024 * 64, tensors.head_outputs_grad.data(),
                                                   64, 1024 * 64, &zero, weights_grad_before.data(), 1024, 1024 * 1024,
                                                   48));
    torch::Tensor usual_weights = pytorch_copy_of(tensors.weights), usual_scores_grad = pytorch_copy_of(tensors.weights_grad);
    int one_warp_per_row_blocks = attention_blocks(rows, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK, 32);  // 6144

    // The same scores and da in bfloat16, and the usual launches' bfloat16 weights and ds from them.
    Tensor scores_bf16 = scores.to(DType::BFloat16), weights_grad_before_bf16 = weights_grad_before.to(DType::BFloat16);
    Tensor weights_bf16 = scores_bf16.clone(), weights_grad_bf16 = weights_grad_before_bf16.clone();
    attention_softmax_forward_kernel<nv_bfloat16><<<one_warp_per_row_blocks, ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK>>>(
        weights_bf16.data_ptr<nv_bfloat16>(), rows, seq_len);
    attention_softmax_backward_kernel<nv_bfloat16>
        <<<one_warp_per_row_blocks, ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK>>>(
            weights_grad_bf16.data_ptr<nv_bfloat16>(), weights_bf16.data_ptr<nv_bfloat16>(), rows, seq_len);
    cudaCheck(cudaGetLastError());
    torch::Tensor usual_weights_bf16 = pytorch_copy_of(weights_bf16);
    torch::Tensor usual_scores_grad_bf16 = pytorch_copy_of(weights_grad_bf16);

    int blocks = 0, threads = 0;                             // the launch the 4 below use, set by the loop
    auto forward = [&] {
        attention_softmax_forward_kernel<float><<<blocks, threads>>>(tensors.weights.data(), rows, seq_len);
        cudaCheck(cudaGetLastError());
    };
    auto backward = [&] {
        attention_softmax_backward_kernel<float><<<blocks, threads>>>(tensors.weights_grad.data(),
                                                                      tensors.weights.data(), rows, seq_len);
        cudaCheck(cudaGetLastError());
    };
    auto forward_bf16 = [&] {
        attention_softmax_forward_kernel<nv_bfloat16><<<blocks, threads>>>(weights_bf16.data_ptr<nv_bfloat16>(), rows,
                                                                           seq_len);
        cudaCheck(cudaGetLastError());
    };
    auto backward_bf16 = [&] {
        attention_softmax_backward_kernel<nv_bfloat16><<<blocks, threads>>>(
            weights_grad_bf16.data_ptr<nv_bfloat16>(), weights_bf16.data_ptr<nv_bfloat16>(), rows, seq_len);
        cudaCheck(cudaGetLastError());
    };
    blocks = one_warp_per_row_blocks;
    threads = ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK;
    warm_up_gpu(backward_bf16);

    int sms = 0;                                             // this GPU's SMs: 40 on a T4, 114 on an H100 PCIe
    cudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int block_counts[] = {40, 80, 160, 320, 640, 1280, one_warp_per_row_blocks, sms, 2 * sms};
    int thread_counts[] = {128, 256, 512, 1024};
    const int launches = 9 * 4, rounds = 20;
    // launch k is block_counts[k / 4] x thread_counts[k % 4]; one time per kernel, in kernel_names' order
    float total_ms[launches][4] = {};
    for (int round = 1; round <= rounds; round++)
    {
        for (int k = 0; k < launches; k++)
        {
            blocks = block_counts[k / 4];
            threads = thread_counts[k % 4];
            total_ms[k][0] += gpu_milliseconds(forward);
            total_ms[k][1] += gpu_milliseconds(backward);
            total_ms[k][2] += gpu_milliseconds(forward_bf16);
            total_ms[k][3] += gpu_milliseconds(backward_bf16);
        }
    }

    const char *kernel_names[] = {"forward", "backward", "bf16 forward", "bf16 backward"};
    float fastest_ms[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    int fastest_blocks[4] = {}, fastest_threads[4] = {};
    int launches_with_the_same_result = 0;
    size_t weights_bytes = tensors.weights.numel() * sizeof(float);
    print_test_name(__func__);
    printf("    time in ms:   blocks   threads   rows per warp   forward   backward   bf16 forward   bf16 backward   "
           "same result\n");
    for (int k = 0; k < launches; k++)
    {
        blocks = block_counts[k / 4];
        threads = thread_counts[k % 4];
        float times[4];
        for (int j = 0; j < 4; j++)
        {
            times[j] = total_ms[k][j] / rounds;
            if (times[j] < fastest_ms[j])
            {
                fastest_ms[j] = times[j];
                fastest_blocks[j] = blocks;
                fastest_threads[j] = threads;
            }
        }

        // One forward from the scores and one backward from da with this launch in each dtype, against the
        // usual.
        cudaCheck(cudaMemcpy(tensors.weights.data(), scores.data(), weights_bytes, cudaMemcpyDeviceToDevice));
        forward();
        cudaCheck(cudaMemcpy(tensors.weights_grad.data(), weights_grad_before.data(), weights_bytes,
                             cudaMemcpyDeviceToDevice));
        backward();
        weights_bf16.copy_(scores_bf16);
        forward_bf16();
        weights_grad_bf16.copy_(weights_grad_before_bf16);
        backward_bf16();
        bool same = torch::equal(pytorch_copy_of(tensors.weights), usual_weights) &&
                    torch::equal(pytorch_copy_of(tensors.weights_grad), usual_scores_grad) &&
                    torch::equal(pytorch_copy_of(weights_bf16), usual_weights_bf16) &&
                    torch::equal(pytorch_copy_of(weights_grad_bf16), usual_scores_grad_bf16);
        if (same)
            launches_with_the_same_result++;

        size_t warps_launched = (size_t)blocks * threads / 32;
        int rows_per_warp = (int)((rows + warps_launched - 1) / warps_launched);   // the most
        const char *usual =
            blocks == one_warp_per_row_blocks && threads == ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK
                ? "   <- what attention_forward and attention_backward launch: one warp per row"
            : blocks == 80 && threads == 128
                ? "   <- a T4's pick for the backward, what it launched before"
                : "";
        printf("                  %6d   %7d   %13d   %7.3f   %8.3f   %12.3f   %13.3f   %-11s%s\n", blocks, threads,
               rows_per_warp, times[0], times[1], times[2], times[3], same ? "yes" : "NO", usual);
    }
    for (int j = 0; j < 4; j++)
        printf("    fastest %-13s  %5d blocks x %4d threads: %.3f ms\n", kernel_names[j], fastest_blocks[j],
               fastest_threads[j], fastest_ms[j]);
    ASSERT_EQ(launches_with_the_same_result, launches);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_on_gpt2_heads();
    matches_pytorch_with_a_seq_len_of_37();
    forward_and_backward_in_bfloat16();
    time_20_steps_on_gpt2_small_size();

    flash_needs_bfloat16_head_size_64_and_whole_tiles();
    flash_matches_on_one_tile();
    flash_matches_on_gpt2_heads_of_256_tokens();
    flash_matches_on_gpt2_small_size();
    flash_backward_writes_over_qkv_grad_the_same_every_time();
    time_flash_against_the_steps_and_pytorch();

    forward_of_the_note_s_example();
    backward_of_the_note_s_example();
    weights_after_the_token_are_0_and_each_row_adds_up_to_1();
    the_first_token_sees_only_itself();
    a_later_token_does_not_change_earlier_outputs();
    equal_scores_give_the_mean_of_the_values();
    large_scores_do_not_overflow();
    heads_do_not_mix();
    a_gradient_on_the_first_token_reaches_only_its_v();
    each_row_of_scores_grad_adds_up_to_0();
    backward_writes_over_qkv_grad();

    every_kernel_fits_its_block();

    time_each_softmax_kernel_with_other_block_and_thread_counts();

    printf("\nall tests passed\n");
    return 0;
}
