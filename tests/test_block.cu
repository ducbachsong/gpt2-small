// test_block.cu — tests for llmc/block.cuh (BlockParams, block_init_params, block_forward, block_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks a block against the same block written in PyTorch, with autograd for the backward, in float
// and in bfloat16, and times one block on GPT-2 small's size in bfloat16, the dtype training runs in; part 2
// checks what can be worked out by hand.
#include <torch/types.h>              // torch::layer_norm, torch::linear, scaled_dot_product_attention; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/block.cuh"

// A block is ten kernels in a row, each within 1e-5 of PyTorch's; the differences add up through them.
static const double TOLERANCE = 1e-4;
// In bfloat16, against PyTorch in float on the same values: each kernel rounds its outputs once (up to 2^-8,
// 0.4%), and the next kernel reads those rounded values, ten times in a row.
static const double BFLOAT16_TOLERANCE = 3e-2;

static const char *PARAM_NAMES[12] = {"qkv_weight", "attention_out_weight", "mlp_up_weight", "mlp_down_weight",
                                      "ln1_weight", "ln1_bias",             "qkv_bias",      "attention_out_bias",
                                      "ln2_weight", "ln2_bias",             "mlp_up_bias",   "mlp_down_bias"};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Fills my tensor (float or bfloat16) with random values, std `scale` around `shift`, made by PyTorch on the
/// GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's sizes.
static void fill_random(const Tensor &tensor, float scale, float shift = 0.0f)
{
    torch::Tensor values =
        torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * scale + shift;
    Tensor floats = Tensor::empty(tensor.shape());
    cudaCheck(cudaMemcpy(floats.data(), values.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    tensor.copy_(floats);                       // rounded, if tensor is bfloat16
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values; bfloat16 is turned into float
/// first, exactly.
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

/// Gives every value of my tensor as a float list on the CPU, whatever its dtype.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
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

/// Everything one block needs, for one size and dtype: its parameters and their gradients in a TensorBuffer
/// each, as the optimizer keeps them, its activations and scratch, and input, output and their gradients.
/// All 0 to start.
struct BlockTensors
{
    size_t batch_size, seq_len, embed_dim, num_heads;
    TensorBuffer param_buffer, grad_buffer;
    BlockParams params, grads;
    BlockActivations activations;
    BlockScratch scratch;
    Tensor input, output, output_grad, input_grad;

    BlockTensors(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                 DType dtype = DType::Float32)
        : batch_size(batch_size), seq_len(seq_len), embed_dim(embed_dim), num_heads(num_heads),
          activations(batch_size, seq_len, embed_dim, num_heads, dtype),
          scratch(batch_size, seq_len, embed_dim, num_heads, dtype)
    {
        int group = param_buffer.add_many(BlockParams::shapes(embed_dim));
        param_buffer.allocate(dtype);
        grad_buffer = TensorBuffer::zeros_like(param_buffer);
        std::vector<Tensor> param_tensors, grad_tensors;
        for (int k = 0; k < 12; k++)
        {
            param_tensors.push_back(param_buffer.tensor(group, k));
            grad_tensors.push_back(grad_buffer.tensor(group, k));
        }
        params = BlockParams(param_tensors);
        grads = BlockParams(grad_tensors);
        Shape rows = {batch_size, seq_len, embed_dim};
        input = Tensor::zeros(rows, dtype), output = Tensor::zeros(rows, dtype);
        output_grad = Tensor::zeros(rows, dtype), input_grad = Tensor::zeros(rows, dtype);
    }

    /// Fills the parameters, input and output_grad with random values. Not GPT-2's starting values: there
    /// the biases are 0 and the projections tiny, which would hide a mistake in them. Each matrix gets std
    /// 1 / sqrt(in_features), so every linear's output has std about 1; the biases std 0.1; the LayerNorm
    /// weights 1 plus std 0.1.
    void fill_random_values()
    {
        fill_random(params.qkv_weight, 1.0f / sqrtf((float)embed_dim));
        fill_random(params.attention_out_weight, 1.0f / sqrtf((float)embed_dim));
        fill_random(params.mlp_up_weight, 1.0f / sqrtf((float)embed_dim));
        fill_random(params.mlp_down_weight, 1.0f / sqrtf(4.0f * embed_dim));
        for (const Tensor &bias : {params.ln1_bias, params.qkv_bias, params.attention_out_bias, params.ln2_bias,
                                   params.mlp_up_bias, params.mlp_down_bias})
            fill_random(bias, 0.1f);
        fill_random(params.ln1_weight, 0.1f, 1.0f);
        fill_random(params.ln2_weight, 0.1f, 1.0f);
        fill_random(input, 1.0f);
        fill_random(output_grad, 1.0f);
    }

    void forward() { block_forward(output, input, params, activations, scratch); }
    void backward() { block_backward(input_grad, grads, output_grad, input, params, activations, scratch); }
};

/// The same block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and
/// the 12 parameters in BlockParams::shapes()' order, gives the output.
static torch::Tensor pytorch_block_forward(const torch::Tensor &input, const std::vector<torch::Tensor> &p,
                                           int64_t num_heads)
{
    int64_t batch_size = input.size(0), seq_len = input.size(1), embed_dim = input.size(2);
    int64_t head_size = embed_dim / num_heads;
    std::vector<int64_t> normalized_shape = {embed_dim};
    // p: 0 qkv_weight, 1 attention_out_weight, 2 mlp_up_weight, 3 mlp_down_weight, 4 ln1_weight, 5 ln1_bias,
    //    6 qkv_bias, 7 attention_out_bias, 8 ln2_weight, 9 ln2_bias, 10 mlp_up_bias, 11 mlp_down_bias
    torch::Tensor qkv = torch::linear(torch::layer_norm(input, normalized_shape, p[4], p[5], 1e-5), p[0], p[6]);
    std::vector<torch::Tensor> parts = qkv.split(embed_dim, 2);
    auto heads = [&](const torch::Tensor &part) {
        return part.view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
    };
    torch::Tensor y = torch::scaled_dot_product_attention(heads(parts[0]), heads(parts[1]), heads(parts[2]), {},
                                                          0.0, true);
    y = y.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim});
    torch::Tensor residual2 = input + torch::linear(y, p[1], p[7]);
    torch::Tensor ln2_output = torch::layer_norm(residual2, normalized_shape, p[8], p[9], 1e-5);
    torch::Tensor mlp_up = torch::linear(ln2_output, p[2], p[10]);
    return residual2 + torch::linear(torch::gelu(mlp_up, "tanh"), p[3], p[11]);
}

/// PyTorch's answers for my block's input, parameters and output_grad: output, input.grad() and each
/// parameter's .grad(), with autograd.
struct PytorchBlock
{
    torch::Tensor input, output;
    std::vector<torch::Tensor> params;

    PytorchBlock(const BlockTensors &mine)
    {
        input = pytorch_copy_of(mine.input).requires_grad_(true);
        for (const Tensor &param : mine.params.tensors())
            params.push_back(pytorch_copy_of(param).requires_grad_(true));
        output = pytorch_block_forward(input, params, (int64_t)mine.num_heads);
        output.backward(pytorch_copy_of(mine.output_grad));
    }
};

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward of a block of these sizes and dtype on random values, mine and PyTorch's,
/// and checks output, input_grad and the 12 parameter gradients within `tolerance`.
static void forward_and_backward_match_pytorch(const char *test, size_t batch_size, size_t seq_len,
                                               size_t embed_dim, size_t num_heads, DType dtype, double tolerance)
{
    torch::manual_seed(0);
    BlockTensors mine(batch_size, seq_len, embed_dim, num_heads, dtype);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    PytorchBlock pytorch(mine);

    print_test_name(test);
    printf("    {%zu, %zu, %zu}, %zu heads, %s\n", batch_size, seq_len, embed_dim, num_heads,
           dtype == DType::BFloat16 ? "bfloat16" : "float");
    ASSERT_NEAR(list_of(mine.output), list_of(pytorch.output), tolerance);
    ASSERT_NEAR(list_of(mine.input_grad), list_of(pytorch.input.grad()), tolerance);
    std::vector<Tensor> grads = mine.grads.tensors();
    for (int k = 0; k < 12; k++)
    {
        printf("    %s's gradient:\n", PARAM_NAMES[k]);
        ASSERT_NEAR(list_of(grads[k]), list_of(pytorch.params[k].grad()), tolerance);
    }
}

static void matches_pytorch_on_gpt2_widths(void)
{
    // Expect: 2 rows of 64 tokens, GPT-2 small's widths (768, 12 heads of 64, MLP 3072), float: the output,
    // input_grad and all 12 parameter gradients equal PyTorch's within 1e-4.
    forward_and_backward_match_pytorch(__func__, 2, 64, 768, 12, DType::Float32, TOLERANCE);
}

static void matches_pytorch_with_small_odd_sizes(void)
{
    // Expect: {3, 37, 64} with 4 heads of 16: a seq_len no warp or square divides, and rows of 64, so
    // every kernel's part-full edges are crossed. The same as PyTorch within 1e-4.
    forward_and_backward_match_pytorch(__func__, 3, 37, 64, 4, DType::Float32, TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_widths's block with every tensor bfloat16 (mean and rstd float).
    // PyTorch is given the same bfloat16 values as floats and works in float: within BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, 2, 64, 768, 12, DType::BFloat16, BFLOAT16_TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of one block on GPT-2 small's size, {4, 1024, 768}
    // with 12 heads, in bfloat16, the dtype training runs in: mine against PyTorch's block with autograd,
    // also in bfloat16, on the same values. Steps 1 and 2 do first-time work (loading kernels, cuBLAS
    // picking its own), so the averages are of 3..20. Each side starts every backward from gradients of 0,
    // as a training step does. A T4 has no bfloat16 tensor cores (they came with the A100), so neither
    // side's products get them.
    //
    // Expect: at the end, my output and input_grad still match PyTorch's block in float on the same
    // bfloat16 values within BFLOAT16_TOLERANCE, as in forward_and_backward_in_bfloat16 at this size.
    torch::manual_seed(0);
    BlockTensors mine(4, 1024, 768, 12, DType::BFloat16);
    mine.fill_random_values();
    torch::Tensor pytorch_input = pytorch_copy_of(mine.input).to(torch::kBFloat16).requires_grad_(true);
    std::vector<torch::Tensor> pytorch_params;
    for (const Tensor &param : mine.params.tensors())
        pytorch_params.push_back(pytorch_copy_of(param).to(torch::kBFloat16).requires_grad_(true));
    torch::Tensor pytorch_output_grad = pytorch_copy_of(mine.output_grad).to(torch::kBFloat16), pytorch_output;

    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { mine.forward(); });
        mine.grad_buffer.flat().zero_();
        times[1] = gpu_milliseconds([&] { mine.backward(); });
        times[2] = gpu_milliseconds([&] { pytorch_output = pytorch_block_forward(pytorch_input, pytorch_params, 12); });
        pytorch_input.mutable_grad() = torch::Tensor();
        for (torch::Tensor &param : pytorch_params)
            param.mutable_grad() = torch::Tensor();
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
    PytorchBlock pytorch_in_float(mine);               // once, not timed: the float answer to check against
    ASSERT_NEAR(list_of(mine.output), list_of(pytorch_in_float.output), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(list_of(mine.input_grad), list_of(pytorch_in_float.input.grad()), BFLOAT16_TOLERANCE);
}

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void shapes_are_gpt2s_with_the_matrices_first(void)
{
    // Expect: for embed_dim 768, the 4 matrices {2304, 768}, {768, 768}, {3072, 768}, {768, 3072}, then the 8
    // vectors {768}, {768}, {2304}, {768}, {768}, {768}, {3072}, {768}. The matrices hold 3 + 1 + 4 + 4 = 12
    // times 768² = 7,077,888 values and the vectors 1 + 1 + 3 + 1 + 1 + 1 + 4 + 1 = 13 times 768 = 9,984, so
    // a block has 7,087,872, and GPT-2 small's 12 blocks 85,054,464.
    std::vector<Shape> shapes = BlockParams::shapes(768);
    std::vector<Shape> expected = {{2304, 768}, {768, 768}, {3072, 768}, {768, 3072}, {768}, {768},
                                   {2304},      {768},      {768},       {768},       {3072}, {768}};
    ASSERT_EQ(shapes.size(), (size_t)12);
    size_t matrix_values = 0, vector_values = 0;
    for (size_t k = 0; k < 12; k++)
    {
        ASSERT_EQ(shapes[k], expected[k]);
        size_t numel = 1;
        for (size_t size : shapes[k])
            numel *= size;
        (shapes[k].size() == 2 ? matrix_values : vector_values) += numel;
    }
    ASSERT_EQ(matrix_values, (size_t)7077888);
    ASSERT_EQ(vector_values, (size_t)9984);
}

static void init_gives_gpt2s_starting_values(void)
{
    // Expect: block_init_params for a model of 12 layers: the LayerNorm weights all 1, the 6 biases all 0;
    // qkv_weight and mlp_up_weight mean 0 and std 0.02; the two projections std 0.02 / sqrt(24) = 0.0040825. Each
    // std within 2% (a matrix holds 589,824 or more values, so by chance about 0.1%), each mean within 0.0005.
    Tensor::manual_seed(0);
    BlockTensors mine(1, 4, 768, 12);
    block_init_params(mine.params, 12);
    ASSERT_EQ(list_of(mine.params.ln1_weight), std::vector<float>(768, 1.0f));
    ASSERT_EQ(list_of(mine.params.ln2_weight), std::vector<float>(768, 1.0f));
    for (const Tensor &bias : {mine.params.ln1_bias, mine.params.qkv_bias, mine.params.attention_out_bias,
                               mine.params.ln2_bias, mine.params.mlp_up_bias, mine.params.mlp_down_bias})
        ASSERT_EQ(list_of(bias), std::vector<float>(bias.numel(), 0.0f));
    float projection_std = 0.02f / sqrtf(24.0f);
    const Tensor matrices[] = {mine.params.qkv_weight, mine.params.attention_out_weight, mine.params.mlp_up_weight,
                               mine.params.mlp_down_weight};
    float stds[] = {0.02f, projection_std, 0.02f, projection_std};
    for (int k = 0; k < 4; k++)
    {
        std::vector<float> values = list_of(matrices[k]);
        double sum = 0.0, sum_squares = 0.0;
        for (float value : values)
            sum += value;
        double mean = sum / values.size();
        for (float value : values)
            sum_squares += (value - mean) * (value - mean);
        ASSERT_NEAR(mean, 0.0, 0.0005);
        ASSERT_NEAR(sqrt(sum_squares / values.size()), stds[k], 0.02 * stds[k]);
    }
}

static void zero_projections_pass_the_input_through(void)
{
    // Expect: random parameters, but the two projections' weights and biases 0. Both branches then give 0,
    // so output = input + 0 + 0 = input, to the last bit. Backward: the branches' input gradients are
    // output_grad times a 0 weight, 0, so nothing reaches the LayerNorms, the attention, the qkv or MLP up linear:
    // input_grad = output_grad to the last bit, and the gradients of ln1, qkv, ln2 and mlp_up are all exactly 0.
    // Only the projections learn: each bias's gradient is output_grad summed over the 2 x 64 rows, the same
    // for both, as both see output_grad.
    torch::manual_seed(0);
    BlockTensors mine(2, 64, 768, 12);
    mine.fill_random_values();
    for (const Tensor &projection : {mine.params.attention_out_weight, mine.params.attention_out_bias,
                                     mine.params.mlp_down_weight, mine.params.mlp_down_bias})
        projection.zero_();
    mine.forward();
    mine.backward();
    ASSERT_EQ(list_of(mine.output), list_of(mine.input));
    ASSERT_EQ(list_of(mine.input_grad), list_of(mine.output_grad));
    for (const Tensor &grad : {mine.grads.ln1_weight, mine.grads.ln1_bias, mine.grads.qkv_weight,
                               mine.grads.qkv_bias, mine.grads.ln2_weight, mine.grads.ln2_bias,
                               mine.grads.mlp_up_weight, mine.grads.mlp_up_bias})
        ASSERT_EQ(list_of(grad), std::vector<float>(grad.numel(), 0.0f));
    std::vector<float> column_sums = list_of(pytorch_copy_of(mine.output_grad).sum(torch::IntArrayRef{0, 1}));
    ASSERT_NEAR(list_of(mine.grads.mlp_down_bias), column_sums, 1e-5);
    ASSERT_EQ(list_of(mine.grads.attention_out_bias), list_of(mine.grads.mlp_down_bias));
}

static void backward_adds_onto_the_param_grads_and_writes_input_grad(void)
{
    // Expect: two backwards in a row, the gradients not set back to 0 in between: each of the 12 parameter
    // gradients is twice one backward's, within 1e-6 (LayerNorm adds its weight_grad and bias_grad with
    // atomics, in whatever order); input_grad is written over, so the same both times, to the last bit.
    torch::manual_seed(0);
    BlockTensors mine(2, 64, 768, 12);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    std::vector<std::vector<float>> once;
    for (const Tensor &grad : mine.grads.tensors())
        once.push_back(list_of(grad));
    std::vector<float> input_grad_once = list_of(mine.input_grad);
    mine.backward();
    std::vector<Tensor> grads = mine.grads.tensors();
    for (int k = 0; k < 12; k++)
    {
        std::vector<float> twice_once = once[k];
        for (float &value : twice_once)
            value *= 2.0f;
        printf("    %s's gradient:\n", PARAM_NAMES[k]);
        ASSERT_NEAR(list_of(grads[k]), twice_once, 1e-6);
    }
    ASSERT_EQ(list_of(mine.input_grad), input_grad_once);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_on_gpt2_widths();
    matches_pytorch_with_small_odd_sizes();
    forward_and_backward_in_bfloat16();
    time_20_steps_on_gpt2_small_size();

    shapes_are_gpt2s_with_the_matrices_first();
    init_gives_gpt2s_starting_values();
    zero_projections_pass_the_input_through();
    backward_adds_onto_the_param_grads_and_writes_input_grad();

    printf("\nall tests passed\n");
    return 0;
}
