// block.cuh — one GPT-2 block: attention then an MLP, each with a LayerNorm before and a residual add around:
//     input ─┬─ LayerNorm 1 ─ qkv linear ─ attention ─ attention out linear ─┐
//            └──────────────────────────────────────────────────────────── + ── residual2
//     residual2 ─┬─ LayerNorm 2 ─ MLP up linear ─ GELU ─ MLP down linear ─┐
//                └─────────────────────────────────────────────────────── + ── output
//
//     input, residual2, output       {batch_size, seq_len, embed_dim}        {4, 1024, 768}
//     qkv                            {batch_size, seq_len, 3 * embed_dim}    {4, 1024, 2304}
//     mlp_up_output, gelu_output     {batch_size, seq_len, 4 * embed_dim}    {4, 1024, 3072}
// BlockParams: the 12 params or grads. BlockActivations: kept for the backward, one per block. BlockScratch:
// used only inside a call, one for all blocks. All float or all bfloat16 (mean and rstd float).
#ifndef BLOCK_CUH
#define BLOCK_CUH

#include <cmath>
#include <vector>
#include "tensor.cuh"
#include "kernels/attention.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layer_norm.cuh"
#include "kernels/linear_cublas.cuh"   // the four linears; kernels/linear.cuh has the same functions
#include "kernels/residual.cuh"

/// A block's 12 parameter tensors, or their gradients: views of the optimizer's buffer, the 4 matrices first.
/// D is embed_dim.
struct BlockParams {
    Tensor qkv_weight;                   // {3D, D}
    Tensor attention_out_weight;         // {D, D}
    Tensor mlp_up_weight;                // {4D, D}
    Tensor mlp_down_weight;              // {D, 4D}
    Tensor ln1_weight;                   // {D}
    Tensor ln1_bias;                     // {D}
    Tensor qkv_bias;                     // {3D}
    Tensor attention_out_bias;           // {D}
    Tensor ln2_weight;                   // {D}
    Tensor ln2_bias;                     // {D}
    Tensor mlp_up_bias;                  // {4D}
    Tensor mlp_down_bias;                // {D}

    BlockParams() = default;
    explicit BlockParams(const std::vector<Tensor>& tensors);           // 12 tensors, in shapes()' order
    static std::vector<Shape> shapes(size_t embed_dim);                 // the 12 shapes, matrices first
    std::vector<Tensor> tensors() const;                                // the 12, in the same order
};

/// What the forward keeps for the backward, for one block. mean, rstd and attention_logsumexp float, the rest
/// of dtype. With flash attention, attention_logsumexp in place of attention_weights.
struct BlockActivations {
    bool flash_attention;                                   // attention_flash_forward and _backward, or the steps
    Tensor ln1_output;                                      // {B, T, D}
    Tensor ln1_mean;                                        // {B, T}
    Tensor ln1_rstd;                                        // {B, T}
    Tensor query;                                           // {B, NH, T, HS}
    Tensor key;                                             // {B, NH, T, HS}
    Tensor value;                                           // {B, NH, T, HS}
    Tensor attention_weights;                               // {B, NH, T, T}; none with flash attention
    Tensor attention_logsumexp;                             // {B, NH, T}: flash attention's, in the weights' place
    Tensor attention_output;                                // {B, T, D}: the heads merged, before the projection
    Tensor residual2;                                       // {B, T, D}: input + the attention branch
    Tensor ln2_output;                                      // {B, T, D}
    Tensor ln2_mean;                                        // {B, T}
    Tensor ln2_rstd;                                        // {B, T}
    Tensor mlp_up_output;                                   // {B, T, 4D}
    Tensor gelu_output;                                     // {B, T, 4D}

    BlockActivations(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                     DType dtype = DType::Float32);
};

/// What lives only during one forward or backward: one for all the blocks of a model.
struct BlockScratch {
    Tensor branch_output;                                   // {B, T, D}: a branch's output, before its residual add
    Tensor qkv;                                             // {B, T, 3D}: split into query, key, value, then unread
    Tensor head_outputs;                                   // {B, NH, T, HS}: attention's, before the merge; not flash's
    Tensor residual2_grad;                                  // {B, T, D}
    Tensor branch_output_grad;                              // {B, T, D}
    Tensor ln_output_grad;                                  // {B, T, D}
    Tensor attention_output_grad;                           // {B, T, D}
    Tensor gelu_output_grad;                                // {B, T, 4D}
    Tensor mlp_up_output_grad;                              // {B, T, 4D}
    Tensor qkv_grad;                                        // {B, T, 3D}
    // the attention steps' scratch (none with flash attention):
    Tensor head_outputs_grad;                                     // {B, NH, T, HS}
    Tensor query_grad;                                            // {B, NH, T, HS}
    Tensor key_grad;                                              // {B, NH, T, HS}
    Tensor value_grad;                                            // {B, NH, T, HS}
    Tensor weights_grad;                                    // {B, NH, T, T}
    // flash attention's (none without):
    Tensor attention_row_dot;                               // {B, NH, T} float

    BlockScratch(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                 DType dtype = DType::Float32);
};

// ── the block ─────────────────────────────────────────────────────────────────
void block_init_params(const BlockParams& params, int num_layers);
void block_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                   const BlockActivations& activations, const BlockScratch& scratch);
void block_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                    const Tensor& input, const BlockParams& params, const BlockActivations& activations,
                    const BlockScratch& scratch);

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── BlockParams ───────────────────────────────────────────────────────────────

/// Makes the block's views.
///
///   tensors   12, in shapes()' order: the 4 matrices, then the 8 vectors
inline BlockParams::BlockParams(const std::vector<Tensor>& tensors) {
    if (tensors.size() != 12) {
        fprintf(stderr, "error: a block has 12 parameter tensors, got %zu\n", tensors.size());
        exit(EXIT_FAILURE);
    }
    qkv_weight = tensors[0], attention_out_weight = tensors[1];
    mlp_up_weight = tensors[2], mlp_down_weight = tensors[3];
    ln1_weight = tensors[4], ln1_bias = tensors[5], qkv_bias = tensors[6], attention_out_bias = tensors[7];
    ln2_weight = tensors[8], ln2_bias = tensors[9], mlp_up_bias = tensors[10], mlp_down_bias = tensors[11];
}

/// Gives the 12 shapes, matrices first: 12 D² + 13 D = 7,087,872 values for D = 768.
///
///   embed_dim   D
inline std::vector<Shape> BlockParams::shapes(size_t embed_dim) {
    size_t D = embed_dim;
    return {{3 * D, D}, {D, D}, {4 * D, D}, {D, 4 * D},       // qkv, attention_out, mlp_up, mlp_down weights
            {D}, {D}, {3 * D}, {D}, {D}, {D}, {4 * D}, {D}};  // ln1, qkv b, attention_out b, ln2, mlp_up b, mlp_down b
}

/// Gives the 12 tensors in shapes()' order.
inline std::vector<Tensor> BlockParams::tensors() const {
    return {qkv_weight, attention_out_weight, mlp_up_weight, mlp_down_weight, ln1_weight, ln1_bias,
            qkv_bias,   attention_out_bias,   ln2_weight,    ln2_bias,        mlp_up_bias, mlp_down_bias};
}

// ── BlockActivations and BlockScratch ─────────────────────────────────────────

/// Makes a block's kept activations, all 0.
///
///   batch_size, seq_len, embed_dim, num_heads   B, T, D, NH; head_size = D / NH
///   dtype                                       Float32 or BFloat16
inline BlockActivations::BlockActivations(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                                          DType dtype) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = embed_dim;
    size_t NH = num_heads;
    size_t HS = embed_dim / num_heads;
    flash_attention = attention_flash_fits(dtype, T, HS);
    ln1_output = Tensor::zeros({B, T, D}, dtype);
    ln1_mean = Tensor::zeros({B, T}), ln1_rstd = Tensor::zeros({B, T});
    query =Tensor::zeros({B, NH, T, HS}, dtype), key = Tensor::zeros({B, NH, T, HS}, dtype);
    value = Tensor::zeros({B, NH, T, HS}, dtype);
    if (flash_attention) attention_logsumexp = Tensor::zeros({B, NH, T});
    else attention_weights = Tensor::zeros({B, NH, T, T}, dtype);
    attention_output = Tensor::zeros({B, T, D}, dtype);
    residual2 = Tensor::zeros({B, T, D}, dtype);
    ln2_output = Tensor::zeros({B, T, D}, dtype);
    ln2_mean = Tensor::zeros({B, T}), ln2_rstd = Tensor::zeros({B, T});
    mlp_up_output = Tensor::zeros({B, T, 4 * D}, dtype), gelu_output = Tensor::zeros({B, T, 4 * D}, dtype);
}

/// Makes the scratch tensors, all 0: with flash attention, attention_row_dot in place of the attention steps' six.
///
///   batch_size, seq_len, embed_dim, num_heads   B, T, D, NH
///   dtype                                       Float32 or BFloat16
inline BlockScratch::BlockScratch(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                                  DType dtype) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = embed_dim;
    size_t NH = num_heads;
    size_t HS = embed_dim / num_heads;
    bool flash_attention = attention_flash_fits(dtype, T, HS);
    branch_output = Tensor::zeros({B, T, D}, dtype);
    qkv = Tensor::zeros({B, T, 3 * D}, dtype);
    residual2_grad = Tensor::zeros({B, T, D}, dtype), branch_output_grad = Tensor::zeros({B, T, D}, dtype);
    ln_output_grad = Tensor::zeros({B, T, D}, dtype), attention_output_grad = Tensor::zeros({B, T, D}, dtype);
    gelu_output_grad = Tensor::zeros({B, T, 4 * D}, dtype), mlp_up_output_grad = Tensor::zeros({B, T, 4 * D}, dtype);
    qkv_grad = Tensor::zeros({B, T, 3 * D}, dtype);
    if (flash_attention) {
        attention_row_dot = Tensor::zeros({B, NH, T});
        return;
    }
    head_outputs = Tensor::zeros({B, NH, T, HS}, dtype);
    head_outputs_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    query_grad = Tensor::zeros({B, NH, T, HS}, dtype), key_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    value_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    weights_grad = Tensor::zeros({B, NH, T, T}, dtype);
}

// ── the block ─────────────────────────────────────────────────────────────────

/// Sets GPT-2's starting values: matrices normal std 0.02, the two projections std 0.02 / sqrt(2 num_layers),
/// biases 0, LayerNorm weights 1.
///
///   params       the block's
///   num_layers   12 for GPT-2 small
inline void block_init_params(const BlockParams& params, int num_layers) {
    float projection_std = 0.02f / sqrtf(2.0f * num_layers);
    params.qkv_weight.normal_(0.0f, 0.02f);
    params.attention_out_weight.normal_(0.0f, projection_std);
    params.mlp_up_weight.normal_(0.0f, 0.02f);
    params.mlp_down_weight.normal_(0.0f, projection_std);
    params.ln1_weight.fill_(1.0), params.ln2_weight.fill_(1.0);
    params.ln1_bias.zero_(), params.qkv_bias.zero_(), params.attention_out_bias.zero_();
    params.ln2_bias.zero_(), params.mlp_up_bias.zero_(), params.mlp_down_bias.zero_();
}

/// Writes output = the block applied to input, keeping what the backward needs.
///
///   output, input   {batch_size, seq_len, embed_dim}; not the same tensor (the backward reads input)
///   params          the block's
///   activations     kept for the backward
///   scratch         shared by all blocks
inline void block_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                          const BlockActivations& activations, const BlockScratch& scratch) {
    const BlockActivations& a = activations;
    // the attention branch
    layer_norm_forward(a.ln1_output, a.ln1_mean, a.ln1_rstd, input, params.ln1_weight, params.ln1_bias);
    linear_forward(scratch.qkv, a.ln1_output, params.qkv_weight, params.qkv_bias);
    if (a.flash_attention)
        attention_flash_forward(a.attention_output, a.query, a.key, a.value, a.attention_logsumexp, scratch.qkv);
    else
        attention_forward(a.attention_output, a.query, a.key, a.value, a.attention_weights, scratch.head_outputs,
                          scratch.qkv);
    linear_forward(scratch.branch_output, a.attention_output, params.attention_out_weight, params.attention_out_bias);
    residual_forward(a.residual2, input, scratch.branch_output);
    // the MLP branch
    layer_norm_forward(a.ln2_output, a.ln2_mean, a.ln2_rstd, a.residual2, params.ln2_weight, params.ln2_bias);
    linear_forward(a.mlp_up_output, a.ln2_output, params.mlp_up_weight, params.mlp_up_bias);
    gelu_forward(a.gelu_output, a.mlp_up_output);
    linear_forward(scratch.branch_output, a.gelu_output, params.mlp_down_weight, params.mlp_down_bias);
    residual_forward(output, a.residual2, scratch.branch_output);
}

/// Writes input_grad and adds the parameters' gradients onto param_grads: the forward's steps the other way.
///
///   input_grad                     written over
///   param_grads                    added onto: 0 before a step's first backward
///   output_grad                    the gradient from the next block
///   input, params, activations     the forward's
///   scratch                        shared by all blocks
inline void block_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                           const Tensor& input, const BlockParams& params, const BlockActivations& activations,
                           const BlockScratch& scratch) {
    const BlockActivations& a = activations;
    const BlockScratch& s = scratch;
    // the MLP branch; LayerNorm 2 adds its part onto residual2_grad
    residual_backward(s.residual2_grad, s.branch_output_grad, output_grad);
    linear_backward(s.gelu_output_grad, param_grads.mlp_down_weight, param_grads.mlp_down_bias, s.branch_output_grad,
                    a.gelu_output, params.mlp_down_weight);
    gelu_backward(s.mlp_up_output_grad, s.gelu_output_grad, a.mlp_up_output);
    linear_backward(s.ln_output_grad, param_grads.mlp_up_weight, param_grads.mlp_up_bias, s.mlp_up_output_grad,
                    a.ln2_output, params.mlp_up_weight);
    layer_norm_backward(s.residual2_grad, param_grads.ln2_weight, param_grads.ln2_bias, s.ln_output_grad,
                        a.residual2, params.ln2_weight, a.ln2_mean, a.ln2_rstd);
    // the attention branch; LayerNorm 1 adds its part onto input_grad
    residual_backward(input_grad, s.branch_output_grad, s.residual2_grad);
    linear_backward(s.attention_output_grad, param_grads.attention_out_weight, param_grads.attention_out_bias,
                    s.branch_output_grad, a.attention_output, params.attention_out_weight);
    if (a.flash_attention)
        attention_flash_backward(s.qkv_grad, s.attention_output_grad, a.attention_output, a.query, a.key, a.value,
                                 a.attention_logsumexp, s.attention_row_dot);
    else
        attention_backward(s.qkv_grad, s.attention_output_grad, a.query, a.key, a.value, a.attention_weights,
                           s.head_outputs_grad, s.weights_grad, s.query_grad, s.key_grad, s.value_grad);
    linear_backward(s.ln_output_grad, param_grads.qkv_weight, param_grads.qkv_bias, s.qkv_grad, a.ln1_output,
                    params.qkv_weight);
    layer_norm_backward(input_grad, param_grads.ln1_weight, param_grads.ln1_bias, s.ln_output_grad, input,
                        params.ln1_weight, a.ln1_mean, a.ln1_rstd);
}

#endif // BLOCK_CUH
