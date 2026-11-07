// gpt2.cuh — the whole GPT-2 model: the encoder, num_layers blocks, a last LayerNorm, the head and the loss:
//     token_ids ─ encoder ─ block 0 ─ ... ─ block 11 ─ LayerNorm f ─ head ─ classifier ─ losses
//
//     token_ids, targets                {batch_size, seq_len} ints                  {4, 1024}
//     encoded, each block's output      {batch_size, seq_len, embed_dim}            {4, 1024, 768}
//     logits                            {batch_size, seq_len, padded_vocab_size}    {4, 1024, 50304}
//     losses                            {batch_size, seq_len} float                 {4, 1024}
// The head has no bias and its weight is wte ("tied"), so wte's gradient gets both parts. The loss is the mean
// of the B x T row losses. All float or all bfloat16 (mean, rstd and the losses float).
#ifndef GPT2_CUH
#define GPT2_CUH

#include <utility>
#include <vector>
#include "tensor.cuh"
#include "block.cuh"
#include "kernels/classifier.cuh"
#include "kernels/encoder.cuh"
#include "kernels/layer_norm.cuh"
#include "kernels/linear_cublas.cuh"   // the head; kernels/linear.cuh has the same functions

/// GPT-2's sizes. The defaults are GPT-2 small's: 124,475,904 parameters with the padded vocabulary.
struct GPT2Config {
    size_t max_seq_len = 1024;           // wpe's rows: the longest seq_len
    size_t vocab_size = 50257;           // the words the tokenizer has
    size_t padded_vocab_size = 50304;    // wte's rows: vocab_size rounded up to a multiple of 128, for the head
    size_t num_layers = 12;
    size_t num_heads = 12;
    size_t embed_dim = 768;
};

/// The model's parameter tensors, or their gradients: views of the optimizer's buffer. Every matrix first:
///     wte, wpe, each block's 4 matrices, each block's 8 vectors, lnf_weight, lnf_bias
struct GPT2Params {
    Tensor wte;                          // {padded_vocab_size, D}: a row per word, also the head's weight
    Tensor wpe;                          // {max_seq_len, D}: a row per position
    std::vector<BlockParams> blocks;     // num_layers of them
    Tensor lnf_weight;                   // {D}: the last LayerNorm
    Tensor lnf_bias;                     // {D}: the last LayerNorm

    GPT2Params() = default;
    GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config);  // in shapes()' order
    static std::vector<Shape> shapes(const GPT2Config& config);                // 4 + 12 num_layers shapes
    std::vector<Tensor> tensors() const;                                       // in the same order
};

/// What the forward keeps for the backward, for the whole model. mean, rstd and the per-row tensors float, the
/// rest of dtype.
struct GPT2Activations {
    Tensor encoded;                                         // {B, T, D}: the encoder's output, block 0's input
    std::vector<BlockActivations> blocks;                   // num_layers of them
    std::vector<Tensor> block_outputs;                      // {B, T, D} each: block l's output, block l + 1's input
    Tensor lnf_output;                                      // {B, T, D}
    Tensor lnf_mean;                                        // {B, T}
    Tensor lnf_rstd;                                        // {B, T}
    Tensor logits;                                          // {B, T, padded_vocab_size}
    Tensor losses;                                          // {B, T}
    Tensor max_of_each_row;                                 // {B, T}
    Tensor sum_of_each_row;                                 // {B, T}

    GPT2Activations(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype = DType::Float32);
};

/// What lives only during one forward or backward.
struct GPT2Scratch {
    BlockScratch block;                                     // shared by all the blocks
    Tensor losses_grad;                                     // {B, T} float: 1 / (B x T) each
    Tensor logits_grad;                                     // {B, T, padded_vocab_size}
    Tensor lnf_output_grad;                                 // {B, T, D}
    Tensor residual_grads[2];                               // {B, T, D}: a block's output_grad and input_grad, in turns

    GPT2Scratch(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype = DType::Float32);
};

// ── the model ─────────────────────────────────────────────────────────────────
void gpt2_init_params(const GPT2Params& params, const GPT2Config& config);
void gpt2_forward(const GPT2Activations& activations, const Tensor& token_ids, const Tensor& targets,
                  const GPT2Params& params, const GPT2Scratch& scratch, const GPT2Config& config);
void gpt2_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                   const GPT2Params& params, const GPT2Activations& activations, const GPT2Scratch& scratch,
                   const GPT2Config& config, size_t micro_batches = 1);
float gpt2_mean_loss(const GPT2Activations& activations);

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── GPT2Params ────────────────────────────────────────────────────────────────

/// Makes the model's views; block l's 4 matrices are tensors 2 + 4l .., its 8 vectors 2 + 4L + 8l ...
///
///   tensors   4 + 12 num_layers, in shapes()' order
///   config    the sizes
inline GPT2Params::GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config) {
    size_t num_layers = config.num_layers;
    if (tensors.size() != 4 + 12 * num_layers) {
        fprintf(stderr, "error: a GPT-2 of %zu layers has %zu parameter tensors, got %zu\n", num_layers,
                4 + 12 * num_layers, tensors.size());
        exit(EXIT_FAILURE);
    }
    wte = tensors[0], wpe = tensors[1];
    size_t first_vector = 2 + 4 * num_layers;               // block 0's first vector
    for (size_t l = 0; l < num_layers; l++) {
        std::vector<Tensor> block(tensors.begin() + 2 + 4 * l, tensors.begin() + 2 + 4 * l + 4);
        block.insert(block.end(), tensors.begin() + first_vector + 8 * l, tensors.begin() + first_vector + 8 * l + 8);
        blocks.push_back(BlockParams(block));
    }
    lnf_weight = tensors[2 + 12 * num_layers], lnf_bias = tensors[3 + 12 * num_layers];
}

/// Gives the model's shapes, every matrix first: 148 shapes, 124,475,904 values for GPT-2 small.
///
///   config   the sizes
inline std::vector<Shape> GPT2Params::shapes(const GPT2Config& config) {
    size_t D = config.embed_dim;
    std::vector<Shape> block = BlockParams::shapes(D);
    std::vector<Shape> shapes = {{config.padded_vocab_size, D}, {config.max_seq_len, D}};
    for (size_t l = 0; l < config.num_layers; l++) shapes.insert(shapes.end(), block.begin(), block.begin() + 4);
    for (size_t l = 0; l < config.num_layers; l++) shapes.insert(shapes.end(), block.begin() + 4, block.end());
    shapes.push_back({D});                                  // lnf_weight
    shapes.push_back({D});                                  // lnf_bias
    return shapes;
}

/// Gives the model's tensors in shapes()' order.
inline std::vector<Tensor> GPT2Params::tensors() const {
    std::vector<Tensor> all = {wte, wpe};
    for (const BlockParams& block : blocks) {
        std::vector<Tensor> its = block.tensors();
        all.insert(all.end(), its.begin(), its.begin() + 4);
    }
    for (const BlockParams& block : blocks) {
        std::vector<Tensor> its = block.tensors();
        all.insert(all.end(), its.begin() + 4, its.end());
    }
    all.push_back(lnf_weight);
    all.push_back(lnf_bias);
    return all;
}

// ── GPT2Activations and GPT2Scratch ───────────────────────────────────────────

/// Makes the model's kept activations, all 0.
///
///   batch_size, seq_len   B and T; T at most config.max_seq_len
///   config                the sizes
///   dtype                 Float32 or BFloat16
inline GPT2Activations::GPT2Activations(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = config.embed_dim;
    encoded = Tensor::zeros({B, T, D}, dtype);
    for (size_t l = 0; l < config.num_layers; l++) {
        blocks.push_back(BlockActivations(B, T, D, config.num_heads, dtype));
        block_outputs.push_back(Tensor::zeros({B, T, D}, dtype));
    }
    lnf_output = Tensor::zeros({B, T, D}, dtype);
    lnf_mean = Tensor::zeros({B, T}), lnf_rstd = Tensor::zeros({B, T});
    logits = Tensor::zeros({B, T, config.padded_vocab_size}, dtype);
    losses = Tensor::zeros({B, T}), max_of_each_row = Tensor::zeros({B, T}), sum_of_each_row = Tensor::zeros({B, T});
}

/// Makes the scratch tensors, all 0.
///
///   batch_size, seq_len, config, dtype   as in GPT2Activations
inline GPT2Scratch::GPT2Scratch(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype)
    : block(batch_size, seq_len, config.embed_dim, config.num_heads, dtype) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = config.embed_dim;
    losses_grad = Tensor::zeros({B, T});
    logits_grad = Tensor::zeros({B, T, config.padded_vocab_size}, dtype);
    lnf_output_grad = Tensor::zeros({B, T, D}, dtype);
    residual_grads[0] = Tensor::zeros({B, T, D}, dtype), residual_grads[1] = Tensor::zeros({B, T, D}, dtype);
}

// ── the model ─────────────────────────────────────────────────────────────────

/// Sets GPT-2's starting values: wte and wpe normal std 0.02 (wte's padding rows 0, and they stay 0), each block
/// as block_init_params, lnf_weight 1, lnf_bias 0.
///
///   params   the model's
///   config   the sizes
inline void gpt2_init_params(const GPT2Params& params, const GPT2Config& config) {
    params.wte.normal_(0.0f, 0.02f);
    if (config.padded_vocab_size > config.vocab_size)
        params.wte.narrow(0, config.vocab_size, config.padded_vocab_size - config.vocab_size).zero_();
    params.wpe.normal_(0.0f, 0.02f);
    for (const BlockParams& block : params.blocks) block_init_params(block, (int)config.num_layers);
    params.lnf_weight.fill_(1.0), params.lnf_bias.zero_();
}

/// Works out each row's loss into activations.losses, keeping what the backward needs.
///
///   activations          written
///   token_ids, targets   {batch_size, seq_len} ints, every id < vocab_size
///   params               the model's
///   scratch, config      the scratch and the sizes
inline void gpt2_forward(const GPT2Activations& activations, const Tensor& token_ids, const Tensor& targets,
                         const GPT2Params& params, const GPT2Scratch& scratch, const GPT2Config& config) {
    const GPT2Activations& a = activations;
    size_t num_layers = params.blocks.size();
    encoder_forward(a.encoded, token_ids, params.wte, params.wpe);
    for (size_t l = 0; l < num_layers; l++) {
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_forward(a.block_outputs[l], input, params.blocks[l], a.blocks[l], scratch.block);
    }
    layer_norm_forward(a.lnf_output, a.lnf_mean, a.lnf_rstd, a.block_outputs[num_layers - 1], params.lnf_weight,
                       params.lnf_bias);
    linear_forward(a.logits, a.lnf_output, params.wte, Tensor());          // the head: no bias
    classifier_forward(a.losses, a.max_of_each_row, a.sum_of_each_row, a.logits, targets, config.vocab_size);
}

/// Adds every parameter's gradient onto param_grads for the mean loss: the forward's steps the other way.
///
///   param_grads                   added onto: 0 before a step's first backward
///   token_ids, targets            the forward's
///   params, activations           the forward's
///   scratch, config               the scratch and the sizes
///   micro_batches                 per step: each row's losses_grad is 1 / (micro_batches x B x T)
inline void gpt2_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                          const GPT2Params& params, const GPT2Activations& activations, const GPT2Scratch& scratch,
                          const GPT2Config& config, size_t micro_batches) {
    const GPT2Activations& a = activations;
    const GPT2Scratch& s = scratch;
    size_t num_layers = params.blocks.size();
    // loss = sum of the row losses / (micro_batches x B x T)
    s.losses_grad.fill_(1.0 / ((double)micro_batches * a.losses.numel()));
    classifier_backward(s.logits_grad, s.losses_grad, a.logits, targets, a.max_of_each_row, a.sum_of_each_row,
                        config.vocab_size);
    linear_backward(s.lnf_output_grad, param_grads.wte, Tensor(), s.logits_grad, a.lnf_output, params.wte);
    Tensor output_grad = s.residual_grads[0];
    Tensor input_grad = s.residual_grads[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto it
    layer_norm_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, s.lnf_output_grad,
                        a.block_outputs[num_layers - 1], params.lnf_weight, a.lnf_mean, a.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], a.blocks[l],
                       s.block);
        std::swap(output_grad, input_grad);                 // block l - 1's output_grad
    }
    encoder_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Gives the mean of the row losses the last gpt2_forward wrote. Waits for the GPU.
///
///   activations   the forward's
inline float gpt2_mean_loss(const GPT2Activations& activations) {
    std::vector<float> losses = activations.losses.tolist();
    double sum = 0.0;
    for (float loss : losses) sum += loss;
    return (float)(sum / losses.size());
}

#endif // GPT2_CUH
