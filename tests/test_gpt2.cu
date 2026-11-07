// test_gpt2.cu — tests for llmc/gpt2.cuh (GPT2Params, gpt2_init_params, gpt2_forward, gpt2_backward,
// gpt2_mean_loss). Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks the model against the same model written in PyTorch, with autograd for the backward, in float
// and in bfloat16; part 2 checks what can be worked out by hand. The timing is in tunes/tune_gpt2.cu.
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

#define BLOCK_PARAMS 12               // a block's parameter tensors: 4 matrices, then 8 vectors
#define BLOCK_MATRICES 4              // of them matrices
#define GPT2_VOCAB_SIZE 50257         // GPT-2's real words
#define GPT2_PADDED_VOCAB_SIZE 50304  // wte's rows
#define GPT2_EMBED_DIM 768            // its row width

// A small model whose sizes no warp or square divides: seq_len 37, 100 words padded to 128, rows of 64.
static GPT2Config small_config()
{
    GPT2Config config_val;
    config_val.max_seq_len = 64, config_val.vocab_size = 100, config_val.padded_vocab_size = 128;
    config_val.num_layers = 2, config_val.num_heads = 4, config_val.embed_dim = 64;
    return config_val;
}

// GPT-2 small's widths and vocabulary, with 2 blocks instead of 12.
static GPT2Config two_layer_config()
{
    GPT2Config config_val;
    config_val.num_layers = 2;
    return config_val;
}

static const char *BLOCK_PARAM_NAMES[BLOCK_PARAMS] = {
    "qkv_weight", "attention_out_weight", "mlp_up_weight", "mlp_down_weight",
    "ln1_weight", "ln1_bias",             "qkv_bias",      "attention_out_bias",
    "ln2_weight", "ln2_bias",             "mlp_up_bias",   "mlp_down_bias"};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives every parameter's name in GPT2Params::shapes()' order, to print with its check: "wte", "wpe",
/// "block 0 qkv_weight", ..., "lnf_bias".
static std::vector<std::string> param_names(const GPT2Config &config_val)
{
    std::vector<std::string> names_list = {"wte", "wpe"};
    for (size_t layer_val = 0; layer_val < config_val.num_layers; layer_val++)
        for (int param_index_val = 0; param_index_val < BLOCK_MATRICES; param_index_val++)
            names_list.push_back("block " + std::to_string(layer_val) + " " + BLOCK_PARAM_NAMES[param_index_val]);
    for (size_t layer_val = 0; layer_val < config_val.num_layers; layer_val++)
        for (int param_index_val = BLOCK_MATRICES; param_index_val < BLOCK_PARAMS; param_index_val++)
            names_list.push_back("block " + std::to_string(layer_val) + " " + BLOCK_PARAM_NAMES[param_index_val]);
    names_list.push_back("lnf_weight");
    names_list.push_back("lnf_bias");
    return names_list;
}

/// Fills my tensor (float or bfloat16) with random values, std `scale_val` around `shift_val`, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's sizes.
static void fill_random(const Tensor &tensor, float scale_val, float shift_val = 0.0f)
{
    torch::Tensor values_tensor =
        torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * scale_val + shift_val;
    Tensor floats_tensor = Tensor::empty(tensor.shape());
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    tensor.copy_(floats_tensor);                // rounded, if tensor is bfloat16
}

/// Fills my int tensor with random token ids 0 .. vocab_size - 1, made by PyTorch on the GPU.
static void fill_random_ids(const Tensor &ids_tensor, size_t vocab_size_val)
{
    torch::Tensor values_tensor = torch::randint((int64_t)vocab_size_val, {(int64_t)ids_tensor.numel()},
                                                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    cudaCheck(cudaMemcpy(ids_tensor.data_ptr(), values_tensor.data_ptr<int>(), ids_tensor.numel() * sizeof(int),
                         cudaMemcpyDeviceToDevice));
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values; bfloat16 is turned into float
/// first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);  // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Makes a PyTorch tensor of int64 ids on the GPU from my int tensor: embedding and cross_entropy want int64.
static torch::Tensor pytorch_ids_of(const Tensor &ids_tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(ids_tensor.shape().begin(), ids_tensor.shape().end());
    return torch::from_blob(ids_tensor.data_ptr(), sizes_list, on_gpu_val).to(torch::kLong);
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().to(torch::kFloat).contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Gives every value of my tensor as a float list on the CPU, whatever its dtype.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Gives the mean and the standard deviation of a list of values, in double.
static std::pair<double, double> mean_and_std(const std::vector<float> &values_list)
{
    double sum_val = 0.0;
    double sum_squares_val = 0.0;
    for (float value_val : values_list)
        sum_val += value_val;
    double mean_val = sum_val / values_list.size();
    for (float value_val : values_list)
        sum_squares_val += (value_val - mean_val) * (value_val - mean_val);
    return {mean_val, sqrt(sum_squares_val / values_list.size())};
}

/// Everything one model needs, for one config, size and dtype: its parameters and their gradients in a
/// TensorBuffer each, as the optimizer keeps them, its activations and scratch, and token_ids and targets.
/// All 0 to start.
struct GPT2Tensors
{
    GPT2Config config_val;
    size_t batch_size_val;
    size_t seq_len_val;
    TensorBuffer param_buffer_val;
    TensorBuffer grad_buffer_val;
    GPT2Params params_val;
    GPT2Params grads_val;
    GPT2Activations activations_val;
    GPT2Scratch scratch_val;
    Tensor token_ids_tensor;
    Tensor targets_tensor;

    GPT2Tensors(const GPT2Config &model_config_val, size_t model_batch_size_val, size_t model_seq_len_val,
                DType dtype_val = DType::Float32)
        : config_val(model_config_val), batch_size_val(model_batch_size_val), seq_len_val(model_seq_len_val),
          activations_val(model_batch_size_val, model_seq_len_val, model_config_val, dtype_val),
          scratch_val(model_batch_size_val, model_seq_len_val, model_config_val, dtype_val)
    {
        // 1. the parameters and their gradients, in a TensorBuffer each
        int group_val = param_buffer_val.add_many(GPT2Params::shapes(config_val));
        param_buffer_val.allocate(dtype_val);
        grad_buffer_val = TensorBuffer::zeros_like(param_buffer_val);
        std::vector<Tensor> param_tensors_list;
        std::vector<Tensor> grad_tensors_list;
        for (int param_index_val = 0; param_index_val < param_buffer_val.group_size(group_val); param_index_val++)
        {
            param_tensors_list.push_back(param_buffer_val.tensor(group_val, param_index_val));
            grad_tensors_list.push_back(grad_buffer_val.tensor(group_val, param_index_val));
        }
        params_val = GPT2Params(param_tensors_list, config_val);
        grads_val = GPT2Params(grad_tensors_list, config_val);
        // 2. token_ids and targets
        token_ids_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32);
        targets_tensor = Tensor::zeros({batch_size_val, seq_len_val}, DType::Int32);
    }

    /// Fills the parameters, token_ids and targets with random values. Not GPT-2's starting values: there
    /// the biases are 0 and the projections tiny, which would hide a mistake in them. wte and wpe get std
    /// 1 / sqrt(embed_dim), so the head's logits have std about 1; each block's matrices std
    /// 1 / sqrt(in_features); the biases std 0.1; the LayerNorm weights 1 plus std 0.1.
    void fill_random_values()
    {
        float embed_std_val = 1.0f / sqrtf((float)config_val.embed_dim);
        fill_random(params_val.wte, embed_std_val);
        fill_random(params_val.wpe, embed_std_val);
        for (const BlockParams &block_val : params_val.blocks)
        {
            fill_random(block_val.qkv_weight, embed_std_val);
            fill_random(block_val.attention_out_weight, embed_std_val);
            fill_random(block_val.mlp_up_weight, embed_std_val);
            fill_random(block_val.mlp_down_weight, 1.0f / sqrtf(4.0f * config_val.embed_dim));
            for (const Tensor &bias_tensor : {block_val.ln1_bias, block_val.qkv_bias, block_val.attention_out_bias,
                                              block_val.ln2_bias, block_val.mlp_up_bias, block_val.mlp_down_bias})
                fill_random(bias_tensor, 0.1f);
            fill_random(block_val.ln1_weight, 0.1f, 1.0f);
            fill_random(block_val.ln2_weight, 0.1f, 1.0f);
        }
        fill_random(params_val.lnf_weight, 0.1f, 1.0f);
        fill_random(params_val.lnf_bias, 0.1f);
        fill_random_ids(token_ids_tensor, config_val.vocab_size);
        fill_random_ids(targets_tensor, config_val.vocab_size);
    }

    /// Runs gpt2_forward on these tensors.
    void forward() { gpt2_forward(activations_val, token_ids_tensor, targets_tensor, params_val, scratch_val, config_val); }

    /// Runs gpt2_backward on these tensors.
    void backward()
    {
        gpt2_backward(grads_val, token_ids_tensor, targets_tensor, params_val, activations_val, scratch_val, config_val);
    }

    /// Gives every parameter's gradient as a CPU list, in GPT2Params::shapes()' order.
    std::vector<std::vector<float>> grad_lists() const
    {
        std::vector<std::vector<float>> lists_list;
        for (const Tensor &grad_tensor : grads_val.tensors())
            lists_list.push_back(list_of(grad_tensor));
        return lists_list;
    }
};

/// Runs the block the plain PyTorch way, as GPT-2's model.py writes it, on PyTorch tensors: from input and the 12
/// parameters in BlockParams::shapes()' order, gives the output. The same as test_block.cu's.
static torch::Tensor pytorch_block_forward(const torch::Tensor &input_tensor, const std::vector<torch::Tensor> &p_list,
                                           int64_t num_heads_val)
{
    // 1. the sizes
    int64_t batch_size_val = input_tensor.size(0);
    int64_t seq_len_val = input_tensor.size(1);
    int64_t embed_dim_val = input_tensor.size(2);
    int64_t head_size_val = embed_dim_val / num_heads_val;
    std::vector<int64_t> normalized_shape_list = {embed_dim_val};
    // p: 0 qkv_weight, 1 attention_out_weight, 2 mlp_up_weight, 3 mlp_down_weight, 4 ln1_weight, 5 ln1_bias,
    //    6 qkv_bias, 7 attention_out_bias, 8 ln2_weight, 9 ln2_bias, 10 mlp_up_bias, 11 mlp_down_bias
    // 2. ln1, qkv, the causal attention, its projection, the residual add
    torch::Tensor qkv_tensor =
        torch::linear(torch::layer_norm(input_tensor, normalized_shape_list, p_list[4], p_list[5], 1e-5), p_list[0], p_list[6]);
    std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
    auto heads_val = [&](const torch::Tensor &part_tensor) {
        return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, head_size_val}).transpose(1, 2);
    };
    torch::Tensor y_tensor = torch::scaled_dot_product_attention(heads_val(parts_list[0]), heads_val(parts_list[1]),
                                                                 heads_val(parts_list[2]), {}, 0.0, true);
    y_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
    torch::Tensor residual2_tensor = input_tensor + torch::linear(y_tensor, p_list[1], p_list[7]);
    // 3. ln2, the MLP, the residual add
    torch::Tensor ln2_output_tensor = torch::layer_norm(residual2_tensor, normalized_shape_list, p_list[8], p_list[9], 1e-5);
    torch::Tensor mlp_up_tensor = torch::linear(ln2_output_tensor, p_list[2], p_list[10]);
    return residual2_tensor + torch::linear(torch::gelu(mlp_up_tensor, "tanh"), p_list[3], p_list[11]);
}

/// Runs the same model the plain PyTorch way, on PyTorch tensors: from int64 token_ids {B, T} and the parameters
/// in GPT2Params::shapes()' order, gives the logits {B, T, padded_vocab_size}.
static torch::Tensor pytorch_gpt2_logits(const torch::Tensor &token_ids_tensor, const std::vector<torch::Tensor> &p_list,
                                         const GPT2Config &config_val)
{
    // 1. the encoder
    int64_t num_layers_val = (int64_t)config_val.num_layers;
    int64_t seq_len_val = token_ids_tensor.size(1);
    torch::Tensor x_tensor = torch::embedding(p_list[0], token_ids_tensor) + p_list[1].narrow(0, 0, seq_len_val);
    // 2. the blocks, each's 12 gathered as GPT2Params does
    for (int64_t layer_val = 0; layer_val < num_layers_val; layer_val++)
    {
        std::vector<torch::Tensor> block_list;
        for (int64_t param_index_val = 0; param_index_val < BLOCK_MATRICES; param_index_val++)
            block_list.push_back(p_list[2 + BLOCK_MATRICES * layer_val + param_index_val]);
        for (int64_t param_index_val = 0; param_index_val < BLOCK_PARAMS - BLOCK_MATRICES; param_index_val++)
            block_list.push_back(p_list[2 + BLOCK_MATRICES * num_layers_val + (BLOCK_PARAMS - BLOCK_MATRICES) * layer_val +
                                        param_index_val]);
        x_tensor = pytorch_block_forward(x_tensor, block_list, (int64_t)config_val.num_heads);
    }
    // 3. lnf, then the head: wte, no bias
    x_tensor = torch::layer_norm(x_tensor, {(int64_t)config_val.embed_dim}, p_list[2 + BLOCK_PARAMS * num_layers_val],
                                 p_list[3 + BLOCK_PARAMS * num_layers_val], 1e-5);
    return torch::linear(x_tensor, p_list[0]);
}

/// Gives PyTorch's loss for these logits: cross-entropy over the first vocab_size columns (the padding left out),
/// the mean over all B x T rows.
static torch::Tensor pytorch_loss(const torch::Tensor &logits_tensor, const torch::Tensor &targets_tensor,
                                  size_t vocab_size_val)
{
    torch::Tensor words_tensor = logits_tensor.narrow(2, 0, (int64_t)vocab_size_val);
    return torch::cross_entropy_loss(words_tensor.reshape({-1, (int64_t)vocab_size_val}), targets_tensor.reshape({-1}));
}

/// PyTorch's answers for my model's parameters, token_ids and targets: logits, loss and each parameter's
/// .grad(), with autograd.
struct PytorchGPT2
{
    torch::Tensor logits_tensor;
    torch::Tensor loss_tensor;
    std::vector<torch::Tensor> params_list;

    PytorchGPT2(const GPT2Tensors &mine_val)
    {
        for (const Tensor &param_tensor : mine_val.params_val.tensors())
            params_list.push_back(pytorch_copy_of(param_tensor).requires_grad_(true));
        logits_tensor = pytorch_gpt2_logits(pytorch_ids_of(mine_val.token_ids_tensor), params_list, mine_val.config_val);
        loss_tensor = pytorch_loss(logits_tensor, pytorch_ids_of(mine_val.targets_tensor), mine_val.config_val.vocab_size);
        loss_tensor.backward();
    }
};

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward of a model of this config, size and dtype on random values, mine and
/// PyTorch's, and checks the loss, the logits and every parameter's gradient within `tolerance_val`.
static void forward_and_backward_match_pytorch(const char *test_ptr, const GPT2Config &config_val, size_t batch_size_val,
                                               size_t seq_len_val, DType dtype_val, double tolerance_val)
{
    // 1. random values; mine, then PyTorch's
    torch::manual_seed(0);
    GPT2Tensors mine_val(config_val, batch_size_val, seq_len_val, dtype_val);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    PytorchGPT2 pytorch_val(mine_val);

    // 2. print the sizes, then check the loss, the logits and every parameter's gradient
    print_test_name(test_ptr);
    printf("    {%zu, %zu}, %zu layers, %zu heads, embed_dim %zu, vocab %zu padded to %zu, %s\n", batch_size_val,
           seq_len_val, config_val.num_layers, config_val.num_heads, config_val.embed_dim, config_val.vocab_size,
           config_val.padded_vocab_size, dtype_val == DType::BFloat16 ? "bfloat16" : "float");
    ASSERT_NEAR((double)gpt2_mean_loss(mine_val.activations_val), pytorch_val.loss_tensor.item<double>(), tolerance_val);
    ASSERT_NEAR(list_of(mine_val.activations_val.logits), list_of(pytorch_val.logits_tensor), tolerance_val);
    std::vector<Tensor> grads_list = mine_val.grads_val.tensors();
    std::vector<std::string> names_list = param_names(config_val);
    for (size_t param_index_val = 0; param_index_val < grads_list.size(); param_index_val++)
    {
        printf("    %s's gradient:\n", names_list[param_index_val].c_str());
        ASSERT_NEAR(list_of(grads_list[param_index_val]), list_of(pytorch_val.params_list[param_index_val].grad()),
                    tolerance_val);
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

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void shapes_are_gpt2s_with_the_matrices_first(void)
{
    // Expect: GPT-2 small has 4 + 12 x 12 = 148 parameter tensors: wte {50304, 768}, wpe {1024, 768}, the
    // blocks' 48 matrices, then their 96 vectors, then lnf's {768}, {768}. Every 2-D one comes before every
    // 1-D one. The 50 matrices hold 38,633,472 + 786,432 + 12 x 7,077,888 = 124,354,560 values and the 98
    // vectors 12 x 9,984 + 2 x 768 = 121,344, so 124,475,904 in all (llm.c's count for the padded vocabulary).
    // 1. the shapes, a few of them checked
    GPT2Config config_val;
    std::vector<Shape> shapes_list = GPT2Params::shapes(config_val);
    ASSERT_EQ(shapes_list.size(), (size_t)148);
    ASSERT_EQ(shapes_list[0], Shape({50304, 768}));
    ASSERT_EQ(shapes_list[1], Shape({1024, 768}));
    ASSERT_EQ(shapes_list[2], Shape({2304, 768}));               // block 0's qkv_weight
    ASSERT_EQ(shapes_list[49], Shape({768, 3072}));              // block 11's mlp_down_weight, the last matrix
    ASSERT_EQ(shapes_list[50], Shape({768}));                    // block 0's ln1_weight, the first vector
    ASSERT_EQ(shapes_list[52], Shape({2304}));                   // block 0's qkv_bias
    ASSERT_EQ(shapes_list[147], Shape({768}));                   // lnf_bias
    // 2. where the matrices end and the vectors start, and the values each holds
    size_t matrix_values_val = 0;
    size_t vector_values_val = 0;
    size_t last_matrix_val = 0;
    size_t first_vector_val = shapes_list.size();
    for (size_t param_index_val = 0; param_index_val < shapes_list.size(); param_index_val++)
    {
        size_t numel_val = 1;
        for (size_t size_val : shapes_list[param_index_val])
            numel_val *= size_val;
        if (shapes_list[param_index_val].size() == 2)
            matrix_values_val += numel_val, last_matrix_val = param_index_val;
        else
            vector_values_val += numel_val, first_vector_val = std::min(first_vector_val, param_index_val);
    }
    ASSERT_EQ(last_matrix_val, (size_t)49);
    ASSERT_EQ(first_vector_val, (size_t)50);
    ASSERT_EQ(matrix_values_val, (size_t)124354560);
    ASSERT_EQ(vector_values_val, (size_t)121344);
    ASSERT_EQ(matrix_values_val + vector_values_val, (size_t)124475904);
}

static void params_are_views_in_shapes_order(void)
{
    // Expect: a model of 3 layers has 4 + 36 = 40 tensors. Made from the buffer's 40 views, block 1's
    // qkv_weight is tensor 2 + 4 = 6 and its mlp_down_weight tensor 9; its ln1_weight tensor 2 + 12 + 8 = 22;
    // block 2's mlp_down_bias tensor 2 + 12 + 16 + 7 = 37; lnf_weight 38 and lnf_bias 39. tensors() gives
    // back all 40 in the buffer's order: the same memory, so each starts at the same address.
    // 1. a model of 3 layers
    GPT2Config config_val = small_config();
    config_val.num_layers = 3;
    GPT2Tensors mine_val(config_val, 1, 4);
    const TensorBuffer &buffer_val = mine_val.param_buffer_val;
    ASSERT_EQ(mine_val.params_val.blocks.size(), (size_t)3);
    // 2. each named parameter at its view's address
    ASSERT_EQ((const void *)mine_val.params_val.wte.data_ptr(), (const void *)buffer_val.tensor(0, 0).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.wpe.data_ptr(), (const void *)buffer_val.tensor(0, 1).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.blocks[1].qkv_weight.data_ptr(), (const void *)buffer_val.tensor(0, 6).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.blocks[1].mlp_down_weight.data_ptr(),
              (const void *)buffer_val.tensor(0, 9).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.blocks[1].ln1_weight.data_ptr(),
              (const void *)buffer_val.tensor(0, 22).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.blocks[2].mlp_down_bias.data_ptr(),
              (const void *)buffer_val.tensor(0, 37).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.lnf_weight.data_ptr(), (const void *)buffer_val.tensor(0, 38).data_ptr());
    ASSERT_EQ((const void *)mine_val.params_val.lnf_bias.data_ptr(), (const void *)buffer_val.tensor(0, 39).data_ptr());
    // 3. tensors() in the buffer's order
    std::vector<Tensor> tensors_list = mine_val.params_val.tensors();
    ASSERT_EQ(tensors_list.size(), (size_t)40);
    for (int param_index_val = 0; param_index_val < 40; param_index_val++)
        ASSERT_EQ((const void *)tensors_list[param_index_val].data_ptr(),
                  (const void *)buffer_val.tensor(0, param_index_val).data_ptr());
}

static void init_gives_gpt2s_starting_values(void)
{
    // Expect: gpt2_init_params for GPT-2 small's widths with 2 layers: wte's 50257 word rows and wpe mean 0
    // and std 0.02, each mean within 0.0005 and std within 2% (38.6M and 786,432 values, so by chance
    // about 0.1%); wte's 47 padding rows all 0; lnf_weight all 1 and lnf_bias all 0; and the blocks set for
    // 2 layers: block 1's attention_out_weight std 0.02 / sqrt(2 x 2) = 0.01.
    // 1. init
    Tensor::manual_seed(0);
    GPT2Config config_val = two_layer_config();
    GPT2Tensors mine_val(config_val, 1, 4);
    gpt2_init_params(mine_val.params_val, config_val);
    // 2. the random ones' mean and std
    const Tensor random_list[] = {mine_val.params_val.wte.narrow(0, 0, GPT2_VOCAB_SIZE), mine_val.params_val.wpe,
                                  mine_val.params_val.blocks[1].attention_out_weight};
    float stds_list[] = {0.02f, 0.02f, 0.01f};
    for (int tensor_index_val = 0; tensor_index_val < 3; tensor_index_val++)
    {
        std::pair<double, double> mean_std_val = mean_and_std(list_of(random_list[tensor_index_val]));
        ASSERT_NEAR(mean_std_val.first, 0.0, 0.0005);
        ASSERT_NEAR(mean_std_val.second, stds_list[tensor_index_val], 0.02 * stds_list[tensor_index_val]);
    }
    // 3. the padding rows 0, lnf 1 and 0
    const size_t padding_rows_val = GPT2_PADDED_VOCAB_SIZE - GPT2_VOCAB_SIZE;   // 47
    ASSERT_EQ(list_of(mine_val.params_val.wte.narrow(0, GPT2_VOCAB_SIZE, padding_rows_val)),
              std::vector<float>(padding_rows_val * GPT2_EMBED_DIM, 0.0f));
    ASSERT_EQ(list_of(mine_val.params_val.lnf_weight), std::vector<float>(GPT2_EMBED_DIM, 1.0f));
    ASSERT_EQ(list_of(mine_val.params_val.lnf_bias), std::vector<float>(GPT2_EMBED_DIM, 0.0f));
}

static void zero_wte_gives_every_word_the_same_chance(void)
{
    // Expect: random parameters but wte all 0. The head's logits are lnf_output · 0 = 0 exactly, so every
    // one of the 100 words gets prob 1/100 in every row, whatever the target: each row's loss is
    // log(100) - 0 = 4.6051702, and so is the mean, within 1e-6.
    // 1. random values, wte 0; a forward
    torch::manual_seed(0);
    GPT2Tensors mine_val(small_config(), 3, 37);
    mine_val.fill_random_values();
    mine_val.params_val.wte.zero_();
    mine_val.forward();
    // 2. check
    ASSERT_EQ(list_of(mine_val.activations_val.logits), std::vector<float>(3 * 37 * 128, 0.0f));
    ASSERT_NEAR(list_of(mine_val.activations_val.losses), std::vector<float>(3 * 37, 4.6051702f), 1e-6);
    ASSERT_NEAR((double)gpt2_mean_loss(mine_val.activations_val), 4.6051702, 1e-6);
}

static void padding_rows_never_change_the_loss_or_learn(void)
{
    // Expect: small_config's 28 padding rows of wte (rows 100..127) are read by no token id, and their
    // logits are left out of the loss. So their gradient is exactly 0, and giving them other random values
    // changes no row's loss, to the last bit: each word's logit uses only its own row of wte.
    // 1. a forward and a backward: the padding rows' gradient 0
    torch::manual_seed(0);
    GPT2Tensors mine_val(small_config(), 3, 37);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    ASSERT_EQ(list_of(mine_val.grads_val.wte.narrow(0, 100, 28)), std::vector<float>(28 * 64, 0.0f));
    // 2. other padding rows, the forward again: the same losses
    std::vector<float> losses_list = list_of(mine_val.activations_val.losses);
    fill_random(mine_val.params_val.wte.narrow(0, 100, 28), 10.0f);
    mine_val.forward();
    ASSERT_EQ(list_of(mine_val.activations_val.losses), losses_list);
}

static void backward_adds_onto_the_param_grads(void)
{
    // Expect: two backwards in a row, the gradients not set back to 0 in between: every parameter's gradient
    // is twice one backward's, within 1e-6 (LayerNorm and the encoder add theirs with atomics, in whatever
    // order).
    // 1. a forward and one backward; its gradients kept
    torch::manual_seed(0);
    GPT2Tensors mine_val(small_config(), 3, 37);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    std::vector<std::vector<float>> once_list = mine_val.grad_lists();
    // 2. a second backward: twice each gradient
    mine_val.backward();
    std::vector<Tensor> grads_list = mine_val.grads_val.tensors();
    std::vector<std::string> names_list = param_names(mine_val.config_val);
    print_test_name(__func__);                               // before the first name line, not after it
    for (size_t param_index_val = 0; param_index_val < grads_list.size(); param_index_val++)
    {
        std::vector<float> twice_once_list = once_list[param_index_val];
        for (float &value_val : twice_once_list)
            value_val *= 2.0f;
        printf("    %s's gradient:\n", names_list[param_index_val].c_str());
        ASSERT_NEAR(list_of(grads_list[param_index_val]), twice_once_list, 1e-6);
    }
}

static void micro_batches_divide_each_gradient(void)
{
    // Expect: a backward for a step of 4 micro-batches gives each row's loss 1 / (4 x B x T) instead of
    // 1 / (B x T), so every parameter's gradient is a quarter of a backward's for 1, within 1e-6 (the
    // atomics add in whatever order); 4 of them add up to the gradient of the mean over all 4 x B x T rows.
    // 1. a forward and a backward for 1 micro-batch; its gradients kept
    const int micro_batches_val = 4;
    torch::manual_seed(0);
    GPT2Tensors mine_val(small_config(), 3, 37);
    mine_val.fill_random_values();
    mine_val.forward();
    mine_val.backward();
    std::vector<std::vector<float>> for_1_list = mine_val.grad_lists();
    // 2. a backward for 4, from 0: a quarter of each
    mine_val.grad_buffer_val.flat().zero_();
    gpt2_backward(mine_val.grads_val, mine_val.token_ids_tensor, mine_val.targets_tensor, mine_val.params_val,
                  mine_val.activations_val, mine_val.scratch_val, mine_val.config_val, micro_batches_val);
    std::vector<Tensor> grads_list = mine_val.grads_val.tensors();
    std::vector<std::string> names_list = param_names(mine_val.config_val);
    print_test_name(__func__);
    for (size_t param_index_val = 0; param_index_val < grads_list.size(); param_index_val++)
    {
        std::vector<float> quarter_list = for_1_list[param_index_val];
        for (float &value_val : quarter_list)
            value_val /= (float)micro_batches_val;
        printf("    %s's gradient:\n", names_list[param_index_val].c_str());
        ASSERT_NEAR(list_of(grads_list[param_index_val]), quarter_list, 1e-6);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_with_small_odd_sizes();
    matches_pytorch_on_gpt2_widths();
    forward_and_backward_in_bfloat16();

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
