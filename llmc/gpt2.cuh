// gpt2.cuh — the whole GPT-2 model: the encoder, num_layers blocks (block.cuh), a last LayerNorm, the head
// and the loss. Built only from block.cuh and the kernels in kernels/:
//
//     token_ids ─ encoder ─ block 0 ─ block 1 ─ ... ─ block 11 ─ LayerNorm f ─ head ─ classifier ─ losses
//                 wte, wpe                                                    wte    targets
//
//     token_ids, targets                {batch_size, seq_len} ints                  {4, 1024}
//     encoded, each block's output      {batch_size, seq_len, embed_dim}            {4, 1024, 768}
//     logits                            {batch_size, seq_len, padded_vocab_size}    {4, 1024, 50304}
//     losses                            {batch_size, seq_len} float                 {4, 1024}
//
// The head is a linear layer with no bias whose weight is wte, the encoder's own matrix ("tied"): a word's
// row is read on the way in and scored against on the way out, so wte's gradient gets both parts added.
// The loss of a step is the mean of the B x T row losses, so each row's losses_grad is 1 / (B x T).
//
// Four structs hold the tensors, as in block.cuh:
//
//     GPT2Config        the sizes: GPT-2 small's by default
//     GPT2Params        the parameter tensors, or their gradients: views of memory kept elsewhere
//     GPT2Activations   what the forward keeps for the backward: one per block, and the model's own
//     GPT2Scratch       what lives only during one call
//
// The tensors are all float or all bfloat16 (mean, rstd and the losses always float).
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

/// The model's parameter tensors, or their gradients: views of memory kept elsewhere (the optimizer's
/// buffer). In shapes()' order every matrix comes before every vector, as AdamW wants:
///
///     wte, wpe, block 0's 4 matrices, ..., block 11's 4 matrices,
///     block 0's 8 vectors, ..., block 11's 8 vectors, lnf_weight, lnf_bias
///
/// so a block's 12 tensors are not next to each other: its BlockParams gathers them.
struct GPT2Params {
    Tensor wte;                          // {padded_vocab_size, D}: a row per word, also the head's weight
    Tensor wpe;                          // {max_seq_len, D}: a row per position
    std::vector<BlockParams> blocks;     // num_layers of them
    Tensor lnf_weight, lnf_bias;         // {D}: the last LayerNorm

    GPT2Params() = default;
    GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config);  // in shapes()' order
    static std::vector<Shape> shapes(const GPT2Config& config);                // 4 + 12 num_layers shapes
    std::vector<Tensor> tensors() const;                                       // in the same order
};

/// What the forward keeps for the backward, for the whole model. mean, rstd and the three per-row tensors
/// are float, the rest of dtype. For GPT-2 small at {4, 1024} in bfloat16: each block's 195 MB (most of
/// it attention_weights, 101 MB), 2.3 GB for the 12, and 412 MB of logits.
struct GPT2Activations {
    Tensor encoded;                                         // {B, T, D}: the encoder's output, block 0's input
    std::vector<BlockActivations> blocks;                   // num_layers of them
    std::vector<Tensor> block_outputs;                      // {B, T, D} each: block l's output, block l + 1's input
    Tensor lnf_output, lnf_mean, lnf_rstd;                  // {B, T, D}, {B, T}, {B, T}
    Tensor logits;                                          // {B, T, padded_vocab_size}
    Tensor losses, max_of_each_row, sum_of_each_row;        // {B, T} each

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

/// Makes the model's views from tensors in shapes()' order. Block l's 4 matrices are tensors 2 + 4l .. 2 + 4l + 3,
/// its 8 vectors 2 + 4L + 8l .. 2 + 4L + 8l + 7, with L = num_layers: for GPT-2 small block 1's qkv_weight is
/// tensor 6 and its ln1_weight tensor 58.
inline GPT2Params::GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config) {
    size_t num_layers = config.num_layers;
    if (tensors.size() != 4 + 12 * num_layers) {
        fprintf(stderr, "error: a GPT-2 of %zu layers has %zu parameter tensors, got %zu\n", num_layers,
                4 + 12 * num_layers, tensors.size());
        exit(EXIT_FAILURE);
    }
    wte = tensors[0], wpe = tensors[1];
    size_t first_vector = 2 + 4 * num_layers;               // where block 0's vectors start
    for (size_t l = 0; l < num_layers; l++) {
        std::vector<Tensor> block(tensors.begin() + 2 + 4 * l, tensors.begin() + 2 + 4 * l + 4);
        block.insert(block.end(), tensors.begin() + first_vector + 8 * l, tensors.begin() + first_vector + 8 * l + 8);
        blocks.push_back(BlockParams(block));
    }
    lnf_weight = tensors[2 + 12 * num_layers], lnf_bias = tensors[3 + 12 * num_layers];
}

/// Gives the model's shapes, every matrix first. For GPT-2 small, 148 shapes: the 50 matrices hold
/// 124,354,560 values (wte 38,633,472, wpe 786,432, the blocks' 12 x 7,077,888) and the 98 vectors 121,344
/// (the blocks' 12 x 9,984 and lnf's 2 x 768), so 124,475,904 in all.
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

/// Gives the model's tensors in shapes()' order, to fill or compare them one by one.
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

/// Makes the model's kept activations, all 0, for these sizes: seq_len at most config.max_seq_len.
inline GPT2Activations::GPT2Activations(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype) {
    size_t B = batch_size, T = seq_len, D = config.embed_dim;
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

/// Makes the scratch tensors, all 0, for these sizes.
inline GPT2Scratch::GPT2Scratch(size_t batch_size, size_t seq_len, const GPT2Config& config, DType dtype)
    : block(batch_size, seq_len, config.embed_dim, config.num_heads, dtype) {
    size_t B = batch_size, T = seq_len, D = config.embed_dim;
    losses_grad = Tensor::zeros({B, T});
    logits_grad = Tensor::zeros({B, T, config.padded_vocab_size}, dtype);
    lnf_output_grad = Tensor::zeros({B, T, D}, dtype);
    residual_grads[0] = Tensor::zeros({B, T, D}, dtype), residual_grads[1] = Tensor::zeros({B, T, D}, dtype);
}

// ── the model ─────────────────────────────────────────────────────────────────

/// Sets the parameters to GPT-2's starting values: wte and wpe random, mean 0 and std 0.02, each block as
/// block_init_params sets it, lnf_weight 1 and lnf_bias 0. wte's padding rows (vocab_size and after) are 0:
/// no token id reads them and their logits are left out of the loss, so their gradient is always 0 and
/// they stay 0. The random numbers come from Tensor's generator: Tensor::manual_seed makes them the same
/// every run.
inline void gpt2_init_params(const GPT2Params& params, const GPT2Config& config) {
    params.wte.normal_(0.0f, 0.02f);
    if (config.padded_vocab_size > config.vocab_size)
        params.wte.narrow(0, config.vocab_size, config.padded_vocab_size - config.vocab_size).zero_();
    params.wpe.normal_(0.0f, 0.02f);
    for (const BlockParams& block : params.blocks) block_init_params(block, (int)config.num_layers);
    params.lnf_weight.fill_(1.0), params.lnf_bias.zero_();
}

/// Works out each row's loss into activations.losses, from token_ids and the right next words, targets
/// (both {batch_size, seq_len} ints, every id < vocab_size), and keeps in activations what the backward
/// needs. gpt2_mean_loss reads the step's loss after it.
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

/// Adds every parameter's gradient onto param_grads, for the loss = the mean of the B x T row losses, from
/// what gpt2_forward read and kept. param_grads must be 0 before the first backward of a step. A step of
/// several micro-batches (forward and backward each, the gradients adding up) trains on the mean over all
/// their rows: with micro_batches = 8, each row's losses_grad is 1 / (8 x B x T), so the 8 backwards add up to
/// the gradient of that mean. The forward's steps the other way round:
///
///     classifier → head (wte_grad += the head's part) → LayerNorm f → block 11 → ... → block 0
///     → encoder (wte_grad += the encoder's part, wpe_grad)
///
/// Each block's input_grad is the output_grad of the block before it; the two residual_grads take turns.
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
    Tensor output_grad = s.residual_grads[0], input_grad = s.residual_grads[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto its input_grad
    layer_norm_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, s.lnf_output_grad,
                        a.block_outputs[num_layers - 1], params.lnf_weight, a.lnf_mean, a.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], a.blocks[l],
                       s.block);
        std::swap(output_grad, input_grad);                 // block l's input_grad: block l - 1's output_grad
    }
    encoder_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Gives the step's loss: the mean of the row losses the last gpt2_forward wrote. It copies them to the
/// CPU, so it waits for the GPU to finish the forward.
inline float gpt2_mean_loss(const GPT2Activations& activations) {
    std::vector<float> losses = activations.losses.tolist();
    double sum = 0.0;
    for (float loss : losses) sum += loss;
    return (float)(sum / losses.size());
}

#endif // GPT2_CUH
