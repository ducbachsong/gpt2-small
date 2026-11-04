// test_gpt2.cu — tests for llmc/gpt2.cuh (GPT2Params, gpt2_init_params, gpt2_forward, gpt2_backward,
// gpt2_mean_loss). Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks the model against the same model written in PyTorch, with autograd for the backward, in float
// and in bfloat16, and times GPT-2 small in bfloat16, the dtype training runs in; part 2 checks what can be
// worked out by hand.
#include <algorithm>                  // std::min
#include <string>
#include <torch/types.h>              // torch::embedding, torch::linear, cross_entropy_loss, ...; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed, torch::NoGradGuard
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the parameters in one allocation, as the optimizer keeps them
#include "../llmc/gpt2.cuh"

// Two blocks are twenty kernels in a row, then the last LayerNorm, the head and the classifier; the block's
// test allows 1e-4 for its ten (each list as a fraction of its largest value).
static const double TOLERANCE = 1e-4;
// In bfloat16, against PyTorch in float on the same values: every kernel rounds its outputs once (up to 2^-8,
// 0.4%), and the next one reads them. The block's test allows 3e-2 for ten kernels; here are 25.
static const double BFLOAT16_TOLERANCE = 5e-2;

// A small model whose sizes no warp or square divides: seq_len 37, 100 words padded to 128, rows of 64.
static GPT2Config small_config()
{
    GPT2Config config;
    config.max_seq_len = 64, config.vocab_size = 100, config.padded_vocab_size = 128;
    config.num_layers = 2, config.num_heads = 4, config.embed_dim = 64;
    return config;
}

// GPT-2 small's widths and vocabulary, with 2 blocks instead of 12.
static GPT2Config two_layer_config()
{
    GPT2Config config;
    config.num_layers = 2;
    return config;
}

static const char *BLOCK_PARAM_NAMES[12] = {"qkv_weight", "attention_out_weight", "mlp_up_weight", "mlp_down_weight",
                                            "ln1_weight", "ln1_bias",             "qkv_bias",      "attention_out_bias",
                                            "ln2_weight", "ln2_bias",             "mlp_up_bias",   "mlp_down_bias"};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives every parameter's name in GPT2Params::shapes()' order, to print with its check: "wte", "wpe",
/// "block 0 qkv_weight", ..., "lnf_bias".
static std::vector<std::string> param_names(const GPT2Config &config)
{
    std::vector<std::string> names = {"wte", "wpe"};
    for (size_t l = 0; l < config.num_layers; l++)
        for (int k = 0; k < 4; k++)
            names.push_back("block " + std::to_string(l) + " " + BLOCK_PARAM_NAMES[k]);
    for (size_t l = 0; l < config.num_layers; l++)
        for (int k = 4; k < 12; k++)
            names.push_back("block " + std::to_string(l) + " " + BLOCK_PARAM_NAMES[k]);
    names.push_back("lnf_weight");
    names.push_back("lnf_bias");
    return names;
}

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

/// Fills my int tensor with random token ids 0 .. vocab_size - 1, made by PyTorch on the GPU.
static void fill_random_ids(const Tensor &ids, size_t vocab_size)
{
    torch::Tensor values = torch::randint((int64_t)vocab_size, {(int64_t)ids.numel()},
                                          torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    cudaCheck(cudaMemcpy(ids.data_ptr(), values.data_ptr<int>(), ids.numel() * sizeof(int), cudaMemcpyDeviceToDevice));
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

/// Makes a PyTorch tensor of int64 ids on the GPU from my int tensor: embedding and cross_entropy want int64.
static torch::Tensor pytorch_ids_of(const Tensor &ids)
{
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes(ids.shape().begin(), ids.shape().end());
    return torch::from_blob(ids.data_ptr(), sizes, on_gpu).to(torch::kLong);
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu = tensor.detach().to(torch::kFloat).contiguous().cpu();
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

/// Everything one model needs, for one config, size and dtype: its parameters and their gradients in a
/// TensorBuffer each, as the optimizer keeps them, its activations and scratch, and token_ids and targets.
/// All 0 to start.
struct GPT2Tensors
{
    GPT2Config config;
    size_t batch_size, seq_len;
    TensorBuffer param_buffer, grad_buffer;
    GPT2Params params, grads;
    GPT2Activations activations;
    GPT2Scratch scratch;
    Tensor token_ids, targets;

    GPT2Tensors(const GPT2Config &config, size_t batch_size, size_t seq_len, DType dtype = DType::Float32)
        : config(config), batch_size(batch_size), seq_len(seq_len), activations(batch_size, seq_len, config, dtype),
          scratch(batch_size, seq_len, config, dtype)
    {
        int group = param_buffer.add_many(GPT2Params::shapes(config));
        param_buffer.allocate(dtype);
        grad_buffer = TensorBuffer::zeros_like(param_buffer);
        std::vector<Tensor> param_tensors, grad_tensors;
        for (int k = 0; k < param_buffer.group_size(group); k++)
        {
            param_tensors.push_back(param_buffer.tensor(group, k));
            grad_tensors.push_back(grad_buffer.tensor(group, k));
        }
        params = GPT2Params(param_tensors, config);
        grads = GPT2Params(grad_tensors, config);
        token_ids = Tensor::zeros({batch_size, seq_len}, DType::Int32);
        targets = Tensor::zeros({batch_size, seq_len}, DType::Int32);
    }

    /// Fills the parameters, token_ids and targets with random values. Not GPT-2's starting values: there
    /// the biases are 0 and the projections tiny, which would hide a mistake in them. wte and wpe get std
    /// 1 / sqrt(embed_dim), so the head's logits have std about 1; each block's matrices std
    /// 1 / sqrt(in_features); the biases std 0.1; the LayerNorm weights 1 plus std 0.1.
    void fill_random_values()
    {
        float embed_std = 1.0f / sqrtf((float)config.embed_dim);
        fill_random(params.wte, embed_std);
        fill_random(params.wpe, embed_std);
        for (const BlockParams &block : params.blocks)
        {
            fill_random(block.qkv_weight, embed_std);
            fill_random(block.attention_out_weight, embed_std);
            fill_random(block.mlp_up_weight, embed_std);
            fill_random(block.mlp_down_weight, 1.0f / sqrtf(4.0f * config.embed_dim));
            for (const Tensor &bias : {block.ln1_bias, block.qkv_bias, block.attention_out_bias, block.ln2_bias,
                                       block.mlp_up_bias, block.mlp_down_bias})
                fill_random(bias, 0.1f);
            fill_random(block.ln1_weight, 0.1f, 1.0f);
            fill_random(block.ln2_weight, 0.1f, 1.0f);
        }
        fill_random(params.lnf_weight, 0.1f, 1.0f);
        fill_random(params.lnf_bias, 0.1f);
        fill_random_ids(token_ids, config.vocab_size);
        fill_random_ids(targets, config.vocab_size);
    }

    void forward() { gpt2_forward(activations, token_ids, targets, params, scratch, config); }
    void backward() { gpt2_backward(grads, token_ids, targets, params, activations, scratch, config); }
};

/// The block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and the 12
/// parameters in BlockParams::shapes()' order, gives the output. The same as test_block.cu's.
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

/// The same model the plain PyTorch way, on PyTorch tensors: from int64 token_ids {B, T} and the parameters
/// in GPT2Params::shapes()' order, gives the logits {B, T, padded_vocab_size}.
static torch::Tensor pytorch_gpt2_logits(const torch::Tensor &token_ids, const std::vector<torch::Tensor> &p,
                                         const GPT2Config &config)
{
    int64_t num_layers = (int64_t)config.num_layers, seq_len = token_ids.size(1);
    torch::Tensor x = torch::embedding(p[0], token_ids) + p[1].narrow(0, 0, seq_len);
    for (int64_t l = 0; l < num_layers; l++)
    {
        std::vector<torch::Tensor> block;                    // block l's 12, gathered as GPT2Params does
        for (int64_t k = 0; k < 4; k++)
            block.push_back(p[2 + 4 * l + k]);
        for (int64_t k = 0; k < 8; k++)
            block.push_back(p[2 + 4 * num_layers + 8 * l + k]);
        x = pytorch_block_forward(x, block, (int64_t)config.num_heads);
    }
    x = torch::layer_norm(x, {(int64_t)config.embed_dim}, p[2 + 12 * num_layers], p[3 + 12 * num_layers], 1e-5);
    return torch::linear(x, p[0]);                           // the head: wte, no bias
}

/// PyTorch's loss for these logits: cross-entropy over the first vocab_size columns (the padding left out),
/// the mean over all B x T rows.
static torch::Tensor pytorch_loss(const torch::Tensor &logits, const torch::Tensor &targets, size_t vocab_size)
{
    torch::Tensor words = logits.narrow(2, 0, (int64_t)vocab_size);
    return torch::cross_entropy_loss(words.reshape({-1, (int64_t)vocab_size}), targets.reshape({-1}));
}

/// PyTorch's answers for my model's parameters, token_ids and targets: logits, loss and each parameter's
/// .grad(), with autograd.
struct PytorchGPT2
{
    torch::Tensor logits, loss;
    std::vector<torch::Tensor> params;

    PytorchGPT2(const GPT2Tensors &mine)
    {
        for (const Tensor &param : mine.params.tensors())
            params.push_back(pytorch_copy_of(param).requires_grad_(true));
        logits = pytorch_gpt2_logits(pytorch_ids_of(mine.token_ids), params, mine.config);
        loss = pytorch_loss(logits, pytorch_ids_of(mine.targets), mine.config.vocab_size);
        loss.backward();
    }
};

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward of a model of this config, size and dtype on random values, mine and
/// PyTorch's, and checks the loss, the logits and every parameter's gradient within `tolerance`.
static void forward_and_backward_match_pytorch(const char *test, const GPT2Config &config, size_t batch_size,
                                               size_t seq_len, DType dtype, double tolerance)
{
    torch::manual_seed(0);
    GPT2Tensors mine(config, batch_size, seq_len, dtype);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    PytorchGPT2 pytorch(mine);

    print_test_name(test);
    printf("    {%zu, %zu}, %zu layers, %zu heads, embed_dim %zu, vocab %zu padded to %zu, %s\n", batch_size, seq_len,
           config.num_layers, config.num_heads, config.embed_dim, config.vocab_size, config.padded_vocab_size,
           dtype == DType::BFloat16 ? "bfloat16" : "float");
    ASSERT_NEAR((double)gpt2_mean_loss(mine.activations), pytorch.loss.item<double>(), tolerance);
    ASSERT_NEAR(list_of(mine.activations.logits), list_of(pytorch.logits), tolerance);
    std::vector<Tensor> grads = mine.grads.tensors();
    std::vector<std::string> names = param_names(config);
    for (size_t k = 0; k < grads.size(); k++)
    {
        printf("    %s's gradient:\n", names[k].c_str());
        ASSERT_NEAR(list_of(grads[k]), list_of(pytorch.params[k].grad()), tolerance);
    }
}

static void matches_pytorch_with_small_odd_sizes(void)
{
    // Expect: small_config's model on {3, 37}: a seq_len no warp or square divides, 100 words padded to 128.
    // The loss, the logits and all 28 parameter gradients the same as PyTorch's within 1e-4, in float.
    forward_and_backward_match_pytorch(__func__, small_config(), 3, 37, DType::Float32, TOLERANCE);
}

static void matches_pytorch_on_gpt2_widths(void)
{
    // Expect: 2 blocks of GPT-2 small's widths (768, 12 heads of 64, MLP 3072) and its vocabulary (50257
    // padded to 50304), 2 rows of 64 tokens, float: the same as PyTorch's within 1e-4.
    forward_and_backward_match_pytorch(__func__, two_layer_config(), 2, 64, DType::Float32, TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: matches_pytorch_on_gpt2_widths's model with every tensor bfloat16 (mean, rstd and the losses
    // float). PyTorch is given the same bfloat16 values as floats and works in float: within
    // BFLOAT16_TOLERANCE.
    forward_and_backward_match_pytorch(__func__, two_layer_config(), 2, 64, DType::BFloat16, BFLOAT16_TOLERANCE);
}

static void time_20_steps_on_gpt2_small_size(void)
{
    // Prints the times of 20 forwards and 20 backwards of GPT-2 small, 12 blocks on {4, 1024}, in bfloat16,
    // the dtype training runs in: mine against PyTorch's model with autograd, also in bfloat16, on the same
    // values. Steps 1 and 2 do first-time work (loading kernels, cuBLAS picking its own), so the averages
    // are of 3..20. Each side starts every backward from gradients of 0, as a training step does.
    //
    // Expect: my loss still matches PyTorch's model in float on the same bfloat16 values within
    // BFLOAT16_TOLERANCE.
    GPT2Config config;
    torch::manual_seed(0);
    GPT2Tensors mine(config, 4, 1024, DType::BFloat16);
    mine.fill_random_values();
    torch::Tensor token_ids = pytorch_ids_of(mine.token_ids), targets = pytorch_ids_of(mine.targets);
    std::vector<torch::Tensor> pytorch_params;
    for (const Tensor &param : mine.params.tensors())
        pytorch_params.push_back(pytorch_copy_of(param).to(torch::kBFloat16).requires_grad_(true));
    torch::Tensor pytorch_loss_value;

    double totals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int t = 1; t <= 20; t++)
    {
        float times[4];
        times[0] = gpu_milliseconds([&] { mine.forward(); });
        mine.grad_buffer.flat().zero_();
        times[1] = gpu_milliseconds([&] { mine.backward(); });
        times[2] = gpu_milliseconds([&] {
            pytorch_loss_value = pytorch_loss(pytorch_gpt2_logits(token_ids, pytorch_params, config), targets,
                                              config.vocab_size);
        });
        for (torch::Tensor &param : pytorch_params)
            param.mutable_grad() = torch::Tensor();
        times[3] = gpu_milliseconds([&] { pytorch_loss_value.backward(); });
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
    double loss_in_float;
    {
        torch::NoGradGuard no_grad;                          // once, not timed: the float answer to check against
        std::vector<torch::Tensor> float_params;
        for (const Tensor &param : mine.params.tensors())
            float_params.push_back(pytorch_copy_of(param));
        loss_in_float = pytorch_loss(pytorch_gpt2_logits(token_ids, float_params, config), targets,
                                     config.vocab_size).item<double>();
    }
    ASSERT_NEAR((double)gpt2_mean_loss(mine.activations), loss_in_float, BFLOAT16_TOLERANCE);
}

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void shapes_are_gpt2s_with_the_matrices_first(void)
{
    // Expect: GPT-2 small has 4 + 12 x 12 = 148 parameter tensors: wte {50304, 768}, wpe {1024, 768}, the
    // blocks' 48 matrices, then their 96 vectors, then lnf's {768}, {768}. Every 2-D one comes before every
    // 1-D one. The 50 matrices hold 38,633,472 + 786,432 + 12 x 7,077,888 = 124,354,560 values and the 98
    // vectors 12 x 9,984 + 2 x 768 = 121,344, so 124,475,904 in all (llm.c's count for the padded vocabulary).
    GPT2Config config;
    std::vector<Shape> shapes = GPT2Params::shapes(config);
    ASSERT_EQ(shapes.size(), (size_t)148);
    ASSERT_EQ(shapes[0], Shape({50304, 768}));
    ASSERT_EQ(shapes[1], Shape({1024, 768}));
    ASSERT_EQ(shapes[2], Shape({2304, 768}));               // block 0's qkv_weight
    ASSERT_EQ(shapes[49], Shape({768, 3072}));              // block 11's mlp_down_weight, the last matrix
    ASSERT_EQ(shapes[50], Shape({768}));                    // block 0's ln1_weight, the first vector
    ASSERT_EQ(shapes[52], Shape({2304}));                   // block 0's qkv_bias
    ASSERT_EQ(shapes[147], Shape({768}));                   // lnf_bias
    size_t matrix_values = 0, vector_values = 0, last_matrix = 0, first_vector = shapes.size();
    for (size_t k = 0; k < shapes.size(); k++)
    {
        size_t numel = 1;
        for (size_t size : shapes[k])
            numel *= size;
        if (shapes[k].size() == 2)
            matrix_values += numel, last_matrix = k;
        else
            vector_values += numel, first_vector = std::min(first_vector, k);
    }
    ASSERT_EQ(last_matrix, (size_t)49);
    ASSERT_EQ(first_vector, (size_t)50);
    ASSERT_EQ(matrix_values, (size_t)124354560);
    ASSERT_EQ(vector_values, (size_t)121344);
    ASSERT_EQ(matrix_values + vector_values, (size_t)124475904);
}

static void params_are_views_in_shapes_order(void)
{
    // Expect: a model of 3 layers has 4 + 36 = 40 tensors. Made from the buffer's 40 views, block 1's
    // qkv_weight is tensor 2 + 4 = 6 and its mlp_down_weight tensor 9; its ln1_weight tensor 2 + 12 + 8 = 22;
    // block 2's mlp_down_bias tensor 2 + 12 + 16 + 7 = 37; lnf_weight 38 and lnf_bias 39. tensors() gives
    // back all 40 in the buffer's order: the same memory, so each starts at the same address.
    GPT2Config config = small_config();
    config.num_layers = 3;
    GPT2Tensors mine(config, 1, 4);
    const TensorBuffer &buffer = mine.param_buffer;
    ASSERT_EQ(mine.params.blocks.size(), (size_t)3);
    ASSERT_EQ((const void *)mine.params.wte.data_ptr(), (const void *)buffer.tensor(0, 0).data_ptr());
    ASSERT_EQ((const void *)mine.params.wpe.data_ptr(), (const void *)buffer.tensor(0, 1).data_ptr());
    ASSERT_EQ((const void *)mine.params.blocks[1].qkv_weight.data_ptr(), (const void *)buffer.tensor(0, 6).data_ptr());
    ASSERT_EQ((const void *)mine.params.blocks[1].mlp_down_weight.data_ptr(),
              (const void *)buffer.tensor(0, 9).data_ptr());
    ASSERT_EQ((const void *)mine.params.blocks[1].ln1_weight.data_ptr(), (const void *)buffer.tensor(0, 22).data_ptr());
    ASSERT_EQ((const void *)mine.params.blocks[2].mlp_down_bias.data_ptr(),
              (const void *)buffer.tensor(0, 37).data_ptr());
    ASSERT_EQ((const void *)mine.params.lnf_weight.data_ptr(), (const void *)buffer.tensor(0, 38).data_ptr());
    ASSERT_EQ((const void *)mine.params.lnf_bias.data_ptr(), (const void *)buffer.tensor(0, 39).data_ptr());
    std::vector<Tensor> tensors = mine.params.tensors();
    ASSERT_EQ(tensors.size(), (size_t)40);
    for (int k = 0; k < 40; k++)
        ASSERT_EQ((const void *)tensors[k].data_ptr(), (const void *)buffer.tensor(0, k).data_ptr());
}

static void init_gives_gpt2s_starting_values(void)
{
    // Expect: gpt2_init_params for GPT-2 small's widths with 2 layers: wte's 50257 word rows and wpe mean 0
    // and std 0.02, each mean within 0.0005 and std within 2% (38.6M and 786,432 values, so by chance
    // about 0.1%); wte's 47 padding rows all 0; lnf_weight all 1 and lnf_bias all 0; and the blocks set for
    // 2 layers: block 1's attention_out_weight std 0.02 / sqrt(2 x 2) = 0.01.
    Tensor::manual_seed(0);
    GPT2Config config = two_layer_config();
    GPT2Tensors mine(config, 1, 4);
    gpt2_init_params(mine.params, config);
    const Tensor random[] = {mine.params.wte.narrow(0, 0, 50257), mine.params.wpe,
                             mine.params.blocks[1].attention_out_weight};
    float stds[] = {0.02f, 0.02f, 0.01f};
    for (int k = 0; k < 3; k++)
    {
        std::vector<float> values = list_of(random[k]);
        double sum = 0.0, sum_squares = 0.0;
        for (float value : values)
            sum += value;
        double mean = sum / values.size();
        for (float value : values)
            sum_squares += (value - mean) * (value - mean);
        ASSERT_NEAR(mean, 0.0, 0.0005);
        ASSERT_NEAR(sqrt(sum_squares / values.size()), stds[k], 0.02 * stds[k]);
    }
    ASSERT_EQ(list_of(mine.params.wte.narrow(0, 50257, 47)), std::vector<float>(47 * 768, 0.0f));
    ASSERT_EQ(list_of(mine.params.lnf_weight), std::vector<float>(768, 1.0f));
    ASSERT_EQ(list_of(mine.params.lnf_bias), std::vector<float>(768, 0.0f));
}

static void zero_wte_gives_every_word_the_same_chance(void)
{
    // Expect: random parameters but wte all 0. The head's logits are lnf_output · 0 = 0 exactly, so every
    // one of the 100 words gets prob 1/100 in every row, whatever the target: each row's loss is
    // log(100) - 0 = 4.6051702, and so is the mean, within 1e-6.
    torch::manual_seed(0);
    GPT2Tensors mine(small_config(), 3, 37);
    mine.fill_random_values();
    mine.params.wte.zero_();
    mine.forward();
    ASSERT_EQ(list_of(mine.activations.logits), std::vector<float>(3 * 37 * 128, 0.0f));
    ASSERT_NEAR(list_of(mine.activations.losses), std::vector<float>(3 * 37, 4.6051702f), 1e-6);
    ASSERT_NEAR((double)gpt2_mean_loss(mine.activations), 4.6051702, 1e-6);
}

static void padding_rows_never_change_the_loss_or_learn(void)
{
    // Expect: small_config's 28 padding rows of wte (rows 100..127) are read by no token id, and their
    // logits are left out of the loss. So their gradient is exactly 0, and giving them other random values
    // changes no row's loss, to the last bit: each word's logit uses only its own row of wte.
    torch::manual_seed(0);
    GPT2Tensors mine(small_config(), 3, 37);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    ASSERT_EQ(list_of(mine.grads.wte.narrow(0, 100, 28)), std::vector<float>(28 * 64, 0.0f));
    std::vector<float> losses = list_of(mine.activations.losses);
    fill_random(mine.params.wte.narrow(0, 100, 28), 10.0f);
    mine.forward();
    ASSERT_EQ(list_of(mine.activations.losses), losses);
}

static void backward_adds_onto_the_param_grads(void)
{
    // Expect: two backwards in a row, the gradients not set back to 0 in between: every parameter's gradient
    // is twice one backward's, within 1e-6 (LayerNorm and the encoder add theirs with atomics, in whatever
    // order).
    torch::manual_seed(0);
    GPT2Tensors mine(small_config(), 3, 37);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    std::vector<std::vector<float>> once;
    for (const Tensor &grad : mine.grads.tensors())
        once.push_back(list_of(grad));
    mine.backward();
    std::vector<Tensor> grads = mine.grads.tensors();
    std::vector<std::string> names = param_names(mine.config);
    print_test_name(__func__);                               // before the first name line, not after it
    for (size_t k = 0; k < grads.size(); k++)
    {
        std::vector<float> twice_once = once[k];
        for (float &value : twice_once)
            value *= 2.0f;
        printf("    %s's gradient:\n", names[k].c_str());
        ASSERT_NEAR(list_of(grads[k]), twice_once, 1e-6);
    }
}

static void micro_batches_divide_each_gradient(void)
{
    // Expect: a backward for a step of 4 micro-batches gives each row's loss 1 / (4 x B x T) instead of
    // 1 / (B x T), so every parameter's gradient is a quarter of a backward's for 1, within 1e-6 (the
    // atomics add in whatever order); 4 of them add up to the gradient of the mean over all 4 x B x T rows.
    torch::manual_seed(0);
    GPT2Tensors mine(small_config(), 3, 37);
    mine.fill_random_values();
    mine.forward();
    mine.backward();
    std::vector<std::vector<float>> for_1;
    for (const Tensor &grad : mine.grads.tensors())
        for_1.push_back(list_of(grad));
    mine.grad_buffer.flat().zero_();
    gpt2_backward(mine.grads, mine.token_ids, mine.targets, mine.params, mine.activations, mine.scratch, mine.config,
                  4);
    std::vector<Tensor> grads = mine.grads.tensors();
    std::vector<std::string> names = param_names(mine.config);
    print_test_name(__func__);
    for (size_t k = 0; k < grads.size(); k++)
    {
        std::vector<float> quarter = for_1[k];
        for (float &value : quarter)
            value /= 4.0f;
        printf("    %s's gradient:\n", names[k].c_str());
        ASSERT_NEAR(list_of(grads[k]), quarter, 1e-6);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    matches_pytorch_with_small_odd_sizes();
    matches_pytorch_on_gpt2_widths();
    forward_and_backward_in_bfloat16();
    time_20_steps_on_gpt2_small_size();

    shapes_are_gpt2s_with_the_matrices_first();
    params_are_views_in_shapes_order();
    init_gives_gpt2s_starting_values();
    zero_wte_gives_every_word_the_same_chance();
    padding_rows_never_change_the_loss_or_learn();
    backward_adds_onto_the_param_grads();
    micro_batches_divide_each_gradient();

    printf("\nall tests passed\n");
    return 0;
}
