// model_fp32.cuh — the whole GPT-2 model in float: the exact reference the bfloat16 and 8-bit models
// (model_bf16.cuh, model_8bit.cuh) are checked against. Params, grads and activations float:
//     token_ids ─ encoder ─ block 0 ─ ... ─ block 11 ─ LayerNorm f ─ head ─ classifier ─ losses
//
//     token_ids, targets                {batch_size, seq_len} ints                  {4, 1024}
//     encoded, each block's output      {batch_size, seq_len, embed_dim}            {4, 1024, 768}
//     logits                            {batch_size, seq_len, padded_vocab_size}    {4, 1024, 50304}
//     losses                            {batch_size, seq_len} float                 {4, 1024}
// The head has no bias and its weight is wte ("tied"), so wte's gradient gets both parts. The loss is the mean
// of the B x T row losses.
//
// One block: attention then an MLP, each with a LayerNorm before and a residual add around:
//     input ─┬─ LayerNorm 1 ─ qkv linear ─ attention ─ attention out linear ─┐
//            └──────────────────────────────────────────────────────────── + ── residual2
//     residual2 ─┬─ LayerNorm 2 ─ MLP up linear ─ GELU ─ MLP down linear ─┐
//                └─────────────────────────────────────────────────────── + ── output
//
//     input, residual2, output       {batch_size, seq_len, embed_dim}        {4, 1024, 768}
//     qkv                            {batch_size, seq_len, 3 * embed_dim}    {4, 1024, 2304}
//     mlp_up_output, gelu_output     {batch_size, seq_len, 4 * embed_dim}    {4, 1024, 3072}
// The attention is flash attention (fp32/attention.cuh): head_size 64, seq_len a multiple of 32, any GPU.
// GPT2Activations: everything the model works in; a block's kept tensors as lists, one per layer, and with the
// _scratch suffix what is used only inside one forward or backward, shared by all the layers.
// The params and grads live in AdamW, with AdamW's m and v for this GPU's slice; the model reads and writes views
// of them (GPT2Params):
//     buffer   [ params | grads | m | v | grad_norm ]   float
// One launch of adamw_update (kernels/fp32/adamw.cuh) a step; the params are their own master.
#ifndef MODEL_FP32_CUH
#define MODEL_FP32_CUH

#include <utility>
#include <vector>
#include "common/tensor.cuh"
#include "common/model_params.cuh"          // GPT2Config, BlockParams, GPT2Params, the starting values, the mean loss
#include "kernels/fp32/attention.cuh"
#include "kernels/fp32/classifier.cuh"
#include "kernels/fp32/encoder.cuh"
#include "kernels/fp32/gelu.cuh"
#include "kernels/fp32/layer_norm.cuh"
#include "kernels/fp32/linear.cuh"
#include "kernels/fp32/residual.cuh"
#include "kernels/fp32/adamw.cuh"             // AdamW's update

/// Everything the model works in, all float. Kept for the backward: the model's tensors, and a block's as lists,
/// one tensor per layer (ln1_output[l] is layer l's). With the _scratch suffix: used only inside one forward or
/// backward, one of each, shared by all the layers.
struct GPT2Activations {
    // ── kept for the backward: the model's ──
    Tensor encoded;                                         // {B, T, D}: the encoder's output, block 0's input
    std::vector<Tensor> block_outputs;                      // {B, T, D} each: block l's output, block l + 1's input
    Tensor lnf_output;                                      // {B, T, D}
    Tensor lnf_mean;                                        // {B, T}
    Tensor lnf_rstd;                                        // {B, T}
    Tensor logits;                                          // {B, T, padded_vocab_size}
    Tensor losses;                                          // {B, T}
    Tensor max_of_each_row;                                 // {B, T}
    Tensor sum_of_each_row;                                 // {B, T}
    // ── kept for the backward: one per layer ──
    std::vector<Tensor> ln1_output;                         // {B, T, D}
    std::vector<Tensor> ln1_mean;                           // {B, T}
    std::vector<Tensor> ln1_rstd;                           // {B, T}
    std::vector<Tensor> qkv;                                // {B, T, 3D}: attention reads query, key, value here
    std::vector<Tensor> attention_logsumexp;                // {B, NH, T}: flash attention's, in place of the weights
    std::vector<Tensor> attention_output;                   // {B, T, D}: the heads merged, before the projection
    std::vector<Tensor> residual2;                          // {B, T, D}: input + the attention branch
    std::vector<Tensor> ln2_output;                         // {B, T, D}
    std::vector<Tensor> ln2_mean;                           // {B, T}
    std::vector<Tensor> ln2_rstd;                           // {B, T}
    std::vector<Tensor> mlp_up_output;                      // {B, T, 4D}
    std::vector<Tensor> gelu_output;                        // {B, T, 4D}
    // ── only inside one forward or backward, shared by all the layers ──
    Tensor branch_output_scratch;                           // {B, T, D}: a branch's output, before its residual add
    Tensor residual2_grad_scratch;                          // {B, T, D}
    Tensor branch_output_grad_scratch;                      // {B, T, D}
    Tensor ln_output_grad_scratch;                          // {B, T, D}
    Tensor attention_output_grad_scratch;                   // {B, T, D}
    Tensor gelu_output_grad_scratch;                        // {B, T, 4D}
    Tensor mlp_up_output_grad_scratch;                      // {B, T, 4D}
    Tensor qkv_grad_scratch;                                // {B, T, 3D}
    Tensor attention_row_dot_scratch;                       // {B, NH, T}: flash attention's backward
    Tensor losses_grad_scratch;                             // {B, T}: 1 / (B x T) each
    Tensor logits_grad_scratch;                             // {B, T, padded_vocab_size}
    Tensor lnf_output_grad_scratch;                         // {B, T, D}
    Tensor residual_grads_scratch[2];                       // {B, T, D}: a block's output_grad and input_grad, in turns

    GPT2Activations(size_t batch_size, size_t seq_len, const GPT2Config& config);
};

// ── one block ─────────────────────────────────────────────────────────────────
void block_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                   const GPT2Activations& activations, size_t layer);
void block_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                    const Tensor& input, const BlockParams& params, const GPT2Activations& activations, size_t layer);

// ── the whole model ───────────────────────────────────────────────────────────
void gpt2_forward(const GPT2Activations& activations, const Tensor& token_ids, const Tensor& targets,
                  const GPT2Params& params, const GPT2Config& config);
void gpt2_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                   const GPT2Params& params, const GPT2Activations& activations, const GPT2Config& config,
                   size_t micro_batches = 1);
float gpt2_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                      const GPT2Params& params, const GPT2Activations& activations, const GPT2Config& config,
                      size_t micro_batches = 1);

// ── the params, grads and AdamW ──────────────────────────────────────────────
/// The model's params and grads, all of them on every GPU, and AdamW's m and v for this GPU's slice, in one float
/// buffer; the update: one launch of adamw_update.
class AdamW : public Optimizer {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config,  // the matrices first
          int rank = 0, int world_size = 1);

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& param(int k) const override;                          // parameter tensor k: float
    const Tensor& grad(int k) const override;                           // its gradient: float
    const Tensor& all_params() const override;                          // every param(k), one flat tensor
    const Tensor& all_grads() const override;                           // every grad(k), one flat tensor
    Tensor master_slice() const override;                               // this slice of all_params()
    const Tensor& m(int k) const;                                       // AdamW's m for parameter k
    const Tensor& v(int k) const;                                       // ...and v
    const Tensor& m_slice() const override;                             // m for this GPU's slice, flat
    const Tensor& v_slice() const override;                             // ...and v

protected:
    void update(int t) override;                                        // AdamW's step t: one launch

private:
    int params_group_index_;             // buffer_'s group of the params
    int grads_group_index_;              // ...of the grads
    int m_group_index_;                  // ...of m, this slice only
    int v_group_index_;                  // ...of v
};

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── the activations ───────────────────────────────────────────────────────────

/// Makes everything the model works in, all 0: the model's kept tensors, a block's for each layer, and the
/// scratch.
///
///   batch_size, seq_len   B and T; T at most config.max_seq_len
///   config                the sizes
inline GPT2Activations::GPT2Activations(size_t batch_size, size_t seq_len, const GPT2Config& config) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = config.embed_dim;
    size_t NH = config.num_heads;
    // 1. kept: the model's
    encoded = Tensor::zeros({B, T, D});
    lnf_output = Tensor::zeros({B, T, D});
    lnf_mean = Tensor::zeros({B, T}), lnf_rstd = Tensor::zeros({B, T});
    logits = Tensor::zeros({B, T, config.padded_vocab_size});
    losses = Tensor::zeros({B, T}), max_of_each_row = Tensor::zeros({B, T}), sum_of_each_row = Tensor::zeros({B, T});
    // 2. kept: one per layer
    for (size_t l = 0; l < config.num_layers; l++) {
        block_outputs.push_back(Tensor::zeros({B, T, D}));
        ln1_output.push_back(Tensor::zeros({B, T, D}));
        ln1_mean.push_back(Tensor::zeros({B, T})), ln1_rstd.push_back(Tensor::zeros({B, T}));
        qkv.push_back(Tensor::zeros({B, T, 3 * D}));
        attention_logsumexp.push_back(Tensor::zeros({B, NH, T}));
        attention_output.push_back(Tensor::zeros({B, T, D}));
        residual2.push_back(Tensor::zeros({B, T, D}));
        ln2_output.push_back(Tensor::zeros({B, T, D}));
        ln2_mean.push_back(Tensor::zeros({B, T})), ln2_rstd.push_back(Tensor::zeros({B, T}));
        mlp_up_output.push_back(Tensor::zeros({B, T, 4 * D})), gelu_output.push_back(Tensor::zeros({B, T, 4 * D}));
    }
    // 3. scratch
    branch_output_scratch = Tensor::zeros({B, T, D});
    residual2_grad_scratch = Tensor::zeros({B, T, D}), branch_output_grad_scratch = Tensor::zeros({B, T, D});
    ln_output_grad_scratch = Tensor::zeros({B, T, D}), attention_output_grad_scratch = Tensor::zeros({B, T, D});
    gelu_output_grad_scratch = Tensor::zeros({B, T, 4 * D}), mlp_up_output_grad_scratch = Tensor::zeros({B, T, 4 * D});
    qkv_grad_scratch = Tensor::zeros({B, T, 3 * D});
    attention_row_dot_scratch = Tensor::zeros({B, NH, T});
    losses_grad_scratch = Tensor::zeros({B, T});
    logits_grad_scratch = Tensor::zeros({B, T, config.padded_vocab_size});
    lnf_output_grad_scratch = Tensor::zeros({B, T, D});
    residual_grads_scratch[0] = Tensor::zeros({B, T, D}), residual_grads_scratch[1] = Tensor::zeros({B, T, D});
}

// ── one block ─────────────────────────────────────────────────────────────────

/// Writes output = block `layer` applied to input, keeping what the backward needs in that layer's activations.
///
///   output, input   {batch_size, seq_len, embed_dim}; not the same tensor (the backward reads input)
///   params          the block's
///   activations     the layer's lists written, the scratch used
///   layer           which block: 0 .. num_layers - 1
inline void block_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                          const GPT2Activations& activations, size_t layer) {
    const GPT2Activations& a = activations;
    size_t l = layer;
    // the attention branch
    layer_norm_forward(a.ln1_output[l], a.ln1_mean[l], a.ln1_rstd[l], input, params.ln1_weight, params.ln1_bias);
    linear_forward(a.qkv[l], a.ln1_output[l], params.qkv_weight, params.qkv_bias);
    attention_flash_forward(a.attention_output[l], a.attention_logsumexp[l], a.qkv[l]);
    linear_forward(a.branch_output_scratch, a.attention_output[l], params.attention_out_weight,
                   params.attention_out_bias);
    residual_forward(a.residual2[l], input, a.branch_output_scratch);
    // the MLP branch
    layer_norm_forward(a.ln2_output[l], a.ln2_mean[l], a.ln2_rstd[l], a.residual2[l], params.ln2_weight,
                       params.ln2_bias);
    linear_forward(a.mlp_up_output[l], a.ln2_output[l], params.mlp_up_weight, params.mlp_up_bias);
    gelu_forward(a.gelu_output[l], a.mlp_up_output[l]);
    linear_forward(a.branch_output_scratch, a.gelu_output[l], params.mlp_down_weight, params.mlp_down_bias);
    residual_forward(output, a.residual2[l], a.branch_output_scratch);
}

/// Writes input_grad and adds block `layer`'s parameter gradients onto param_grads: the forward's steps the
/// other way.
///
///   input_grad                     written over
///   param_grads                    added onto: 0 before a step's first backward
///   output_grad                    the gradient from the next block
///   input, params, activations     the forward's; activations' scratch used
///   layer                          which block: 0 .. num_layers - 1
inline void block_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                           const Tensor& input, const BlockParams& params, const GPT2Activations& activations,
                           size_t layer) {
    const GPT2Activations& a = activations;
    size_t l = layer;
    // the MLP branch; LayerNorm 2 adds its part onto residual2_grad
    residual_backward(a.residual2_grad_scratch, a.branch_output_grad_scratch, output_grad);
    linear_backward(a.gelu_output_grad_scratch, param_grads.mlp_down_weight, param_grads.mlp_down_bias,
                    a.branch_output_grad_scratch, a.gelu_output[l], params.mlp_down_weight);
    gelu_backward(a.mlp_up_output_grad_scratch, a.gelu_output_grad_scratch, a.mlp_up_output[l]);
    linear_backward(a.ln_output_grad_scratch, param_grads.mlp_up_weight, param_grads.mlp_up_bias,
                    a.mlp_up_output_grad_scratch, a.ln2_output[l], params.mlp_up_weight);
    layer_norm_backward(a.residual2_grad_scratch, param_grads.ln2_weight, param_grads.ln2_bias,
                        a.ln_output_grad_scratch, a.residual2[l], params.ln2_weight, a.ln2_mean[l], a.ln2_rstd[l]);
    // the attention branch; LayerNorm 1 adds its part onto input_grad
    residual_backward(input_grad, a.branch_output_grad_scratch, a.residual2_grad_scratch);
    linear_backward(a.attention_output_grad_scratch, param_grads.attention_out_weight, param_grads.attention_out_bias,
                    a.branch_output_grad_scratch, a.attention_output[l], params.attention_out_weight);
    attention_flash_backward(a.qkv_grad_scratch, a.attention_output_grad_scratch, a.attention_output[l], a.qkv[l],
                             a.attention_logsumexp[l], a.attention_row_dot_scratch);
    linear_backward(a.ln_output_grad_scratch, param_grads.qkv_weight, param_grads.qkv_bias, a.qkv_grad_scratch,
                    a.ln1_output[l], params.qkv_weight);
    layer_norm_backward(input_grad, param_grads.ln1_weight, param_grads.ln1_bias, a.ln_output_grad_scratch, input,
                        params.ln1_weight, a.ln1_mean[l], a.ln1_rstd[l]);
}

// ── the whole model ───────────────────────────────────────────────────────────

/// Works out each row's loss into activations.losses, keeping what the backward needs.
///
///   activations          written
///   token_ids, targets   {batch_size, seq_len} ints, every id < vocab_size
///   params               the model's
///   config               the sizes
inline void gpt2_forward(const GPT2Activations& activations, const Tensor& token_ids, const Tensor& targets,
                         const GPT2Params& params, const GPT2Config& config) {
    const GPT2Activations& a = activations;
    size_t num_layers = params.blocks.size();
    encoder_forward(a.encoded, token_ids, params.wte, params.wpe);
    for (size_t l = 0; l < num_layers; l++) {
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_forward(a.block_outputs[l], input, params.blocks[l], a, l);
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
///   params, activations           the forward's; activations' scratch used
///   config                        the sizes
///   micro_batches                 per step: each row's losses_grad is 1 / (micro_batches x B x T)
inline void gpt2_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                          const GPT2Params& params, const GPT2Activations& activations, const GPT2Config& config,
                          size_t micro_batches) {
    const GPT2Activations& a = activations;
    size_t num_layers = params.blocks.size();
    // loss = sum of the row losses / (micro_batches x B x T)
    a.losses_grad_scratch.fill_(1.0 / ((double)micro_batches * a.losses.numel()));
    classifier_backward(a.logits_grad_scratch, a.losses_grad_scratch, a.logits, targets, a.max_of_each_row,
                        a.sum_of_each_row, config.vocab_size);
    linear_backward(a.lnf_output_grad_scratch, param_grads.wte, Tensor(), a.logits_grad_scratch, a.lnf_output,
                    params.wte);
    Tensor output_grad = a.residual_grads_scratch[0];
    Tensor input_grad = a.residual_grads_scratch[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto it
    layer_norm_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, a.lnf_output_grad_scratch,
                        a.block_outputs[num_layers - 1], params.lnf_weight, a.lnf_mean, a.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], a, l);
        std::swap(output_grad, input_grad);                 // block l - 1's output_grad
    }
    encoder_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Runs one micro-batch of a training step: the forward, then the backward adding every parameter's
/// gradient onto param_grads. Gives the micro-batch's mean loss, so waits for the GPU.
///
///   param_grads, token_ids, targets, params, activations, config, micro_batches   as in gpt2_backward
inline float gpt2_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                             const GPT2Params& params, const GPT2Activations& activations, const GPT2Config& config,
                             size_t micro_batches) {
    gpt2_forward(activations, token_ids, targets, params, config);
    gpt2_backward(param_grads, token_ids, targets, params, activations, config, micro_batches);
    return gpt2_mean_loss(activations);
}

// ── AdamW: making it ─────────────────────────────────────────────────────────

/// Makes a float AdamW: params, grads, then m and v for this slice, then grad_norm; all 0.
///
///   param_shapes        the matrices first
///   config              the settings
///   rank, world_size    this GPU of how many; the values must split into slices of a multiple of 4
inline AdamW::AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config, int rank, int world_size)
    : Optimizer(param_shapes, config, DType::Float32, rank, world_size) {
    params_group_index_ = buffer_.add_many(param_shapes);
    grads_group_index_ = buffer_.add_many(param_shapes);
    m_group_index_ = buffer_.add_many(slice_shapes());
    v_group_index_ = buffer_.add_many(slice_shapes());
    allocate();
}

// ── AdamW: reading ───────────────────────────────────────────────────────────

/// Gives parameter tensor k.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::param(int k) const { return buffer_.tensor(params_group_index_, k); }

/// Gives the gradient of parameter k.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::grad(int k) const { return buffer_.tensor(grads_group_index_, k); }

/// Gives every param(k) as one flat tensor.
inline const Tensor& AdamW::all_params() const { return buffer_.group(params_group_index_); }

/// Gives every grad(k) as one flat tensor: one launch zeroes them all.
inline const Tensor& AdamW::all_grads() const { return buffer_.group(grads_group_index_); }

/// Gives this slice's params as one flat tensor: in float they are their own master. A checkpoint saves it.
inline Tensor AdamW::master_slice() const { return all_params().narrow(0, slice_start(), slice_size()); }

/// Gives m for parameter k: the running average of its gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::m(int k) const {
    stop_unless_one_gpu("m(k)");
    return buffer_.tensor(m_group_index_, k);
}

/// Gives v for parameter k: the running average of its squared gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::v(int k) const {
    stop_unless_one_gpu("v(k)");
    return buffer_.tensor(v_group_index_, k);
}

/// Gives m for this slice as one flat tensor: a checkpoint saves it.
inline const Tensor& AdamW::m_slice() const { return buffer_.group(m_group_index_); }

/// Gives v for this slice as one flat tensor: a checkpoint saves it.
inline const Tensor& AdamW::v_slice() const { return buffer_.group(v_group_index_); }

// ── AdamW: the update ────────────────────────────────────────────────────────

/// Does AdamW's step t on this slice with one launch of adamw_update. grad_norm is the norm of all the
/// gradients, so every GPU clips alike.
///
///   t   the step number: 1, 2, 3, ...
inline void AdamW::update(int t) {
    size_t start = slice_start();
    adamw_update(all_params().data() + start, all_grads().data() + start, m_slice().data(), v_slice().data(),
                 slice_size(), decayed_in_slice(), grad_norm().data(), config_.max_norm, config_.learning_rate,
                 config_.beta1, config_.beta2, config_.eps, config_.weight_decay, t);
}

#endif // MODEL_FP32_CUH
