// block.cuh — one GPT-2 block: attention, then an MLP, each with a LayerNorm before it and a residual add
// around it. GPT-2 small stacks 12 of them. Built only from the kernels in kernels/:
//
//     input ─┬─ LayerNorm 1 ─ qkv linear ─ attention ─ attention out linear ─┐
//            └──────────────────────────────────────────────────────────── + ── residual2
//     residual2 ─┬─ LayerNorm 2 ─ MLP up linear ─ GELU ─ MLP down linear ─┐
//                └─────────────────────────────────────────────────────── + ── output
//
//     input, residual2, output       {batch_size, seq_len, embed_dim}        {4, 1024, 768}
//     qkv                            {batch_size, seq_len, 3 * embed_dim}    {4, 1024, 2304}
//     mlp_up_output, gelu_output     {batch_size, seq_len, 4 * embed_dim}    {4, 1024, 3072}
//
// Backward runs the same steps the other way. The gradient reaches input by two roads, through the
// attention branch and straight through the residual adds: residual_backward writes input_grad, then
// LayerNorm 1's backward adds its part onto it.
//
// Three structs hold the tensors, so a model of 12 blocks can share what can be shared:
//
//     BlockParams        the 12 parameter tensors, or their 12 gradients: views of memory kept elsewhere
//     BlockActivations   what the forward keeps for the backward: one per block
//     BlockScratch       what lives only during one call: one for all the blocks
//
// The tensors are all float or all bfloat16 (mean and rstd always float); each kernel works in float.
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

/// A block's 12 parameter tensors, or their gradients: views of memory kept elsewhere (the optimizer's
/// buffer). The 4 matrices first, in the order shapes() gives: AdamW decays the matrices and wants them
/// first. D is embed_dim.
struct BlockParams {
    Tensor qkv_weight;                   // {3D, D}
    Tensor attention_out_weight;         // {D, D}
    Tensor mlp_up_weight;                // {4D, D}
    Tensor mlp_down_weight;              // {D, 4D}
    Tensor ln1_weight, ln1_bias;         // {D}
    Tensor qkv_bias;                     // {3D}
    Tensor attention_out_bias;           // {D}
    Tensor ln2_weight, ln2_bias;         // {D}
    Tensor mlp_up_bias;                  // {4D}
    Tensor mlp_down_bias;                // {D}

    BlockParams() = default;
    explicit BlockParams(const std::vector<Tensor>& tensors);           // 12 tensors, in shapes()' order
    static std::vector<Shape> shapes(size_t embed_dim);                 // the 12 shapes, matrices first
    std::vector<Tensor> tensors() const;                                // the 12, in the same order
};

/// What the forward keeps for the backward, for one block. mean and rstd are float, the rest of dtype.
struct BlockActivations {
    Tensor ln1_output, ln1_mean, ln1_rstd;                  // {B, T, D}, {B, T}, {B, T}
    Tensor query, key, value;                               // {B, NH, T, HS} each
    Tensor attention_weights;                               // {B, NH, T, T}
    Tensor attention_output;                                // {B, T, D}: the heads merged, before the projection
    Tensor residual2;                                       // {B, T, D}: input + the attention branch
    Tensor ln2_output, ln2_mean, ln2_rstd;                  // {B, T, D}, {B, T}, {B, T}
    Tensor mlp_up_output, gelu_output;                      // {B, T, 4D} each

    BlockActivations(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                     DType dtype = DType::Float32);
};

/// What lives only during one forward or backward: one for all the blocks of a model.
struct BlockScratch {
    Tensor branch_output;                                   // {B, T, D}: a branch's output, before its residual add
    Tensor qkv;                                             // {B, T, 3D}: split into query, key, value, then unread
    Tensor head_outputs;                                   // {B, NH, T, HS}: attention's, before the merge
    Tensor residual2_grad, branch_output_grad;              // {B, T, D}
    Tensor ln_output_grad, attention_output_grad;           // {B, T, D}
    Tensor gelu_output_grad, mlp_up_output_grad;            // {B, T, 4D}
    Tensor qkv_grad;                                        // {B, T, 3D}
    Tensor head_outputs_grad, query_grad, key_grad, value_grad;   // {B, NH, T, HS}
    Tensor weights_grad;                                    // {B, NH, T, T}

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

/// Makes the block's views from 12 tensors in shapes()' order: the 4 matrices, then the 8 vectors.
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

/// Gives the 12 shapes for embed_dim D, matrices first. For D = 768: the 4 matrices hold 12 D² = 7,077,888
/// values and the 8 vectors 13 D = 9,984, so 7,087,872 per block.
inline std::vector<Shape> BlockParams::shapes(size_t embed_dim) {
    size_t D = embed_dim;
    return {{3 * D, D}, {D, D}, {4 * D, D}, {D, 4 * D},       // qkv, attention_out, mlp_up, mlp_down weights
            {D}, {D}, {3 * D}, {D}, {D}, {D}, {4 * D}, {D}};  // ln1, qkv b, attention_out b, ln2, mlp_up b, mlp_down b
}

/// Gives the 12 tensors in shapes()' order, to fill or compare them one by one.
inline std::vector<Tensor> BlockParams::tensors() const {
    return {qkv_weight, attention_out_weight, mlp_up_weight, mlp_down_weight, ln1_weight, ln1_bias,
            qkv_bias,   attention_out_bias,   ln2_weight,    ln2_bias,        mlp_up_bias, mlp_down_bias};
}

// ── BlockActivations and BlockScratch ─────────────────────────────────────────

/// Makes a block's kept activations, all 0, for these sizes. head_size = embed_dim / num_heads.
inline BlockActivations::BlockActivations(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                                          DType dtype) {
    size_t B = batch_size, T = seq_len, D = embed_dim, NH = num_heads, HS = embed_dim / num_heads;
    ln1_output = Tensor::zeros({B, T, D}, dtype);
    ln1_mean = Tensor::zeros({B, T}), ln1_rstd = Tensor::zeros({B, T});
    query =Tensor::zeros({B, NH, T, HS}, dtype), key = Tensor::zeros({B, NH, T, HS}, dtype);
    value = Tensor::zeros({B, NH, T, HS}, dtype);
    attention_weights = Tensor::zeros({B, NH, T, T}, dtype);
    attention_output = Tensor::zeros({B, T, D}, dtype);
    residual2 = Tensor::zeros({B, T, D}, dtype);
    ln2_output = Tensor::zeros({B, T, D}, dtype);
    ln2_mean = Tensor::zeros({B, T}), ln2_rstd = Tensor::zeros({B, T});
    mlp_up_output = Tensor::zeros({B, T, 4 * D}, dtype), gelu_output = Tensor::zeros({B, T, 4 * D}, dtype);
}

/// Makes the scratch tensors, all 0, for these sizes: big enough for any block of them.
inline BlockScratch::BlockScratch(size_t batch_size, size_t seq_len, size_t embed_dim, size_t num_heads,
                                  DType dtype) {
    size_t B = batch_size, T = seq_len, D = embed_dim, NH = num_heads, HS = embed_dim / num_heads;
    branch_output = Tensor::zeros({B, T, D}, dtype);
    qkv = Tensor::zeros({B, T, 3 * D}, dtype);
    head_outputs = Tensor::zeros({B, NH, T, HS}, dtype);
    residual2_grad = Tensor::zeros({B, T, D}, dtype), branch_output_grad = Tensor::zeros({B, T, D}, dtype);
    ln_output_grad = Tensor::zeros({B, T, D}, dtype), attention_output_grad = Tensor::zeros({B, T, D}, dtype);
    gelu_output_grad = Tensor::zeros({B, T, 4 * D}, dtype), mlp_up_output_grad = Tensor::zeros({B, T, 4 * D}, dtype);
    qkv_grad = Tensor::zeros({B, T, 3 * D}, dtype);
    head_outputs_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    query_grad = Tensor::zeros({B, NH, T, HS}, dtype), key_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    value_grad = Tensor::zeros({B, NH, T, HS}, dtype);
    weights_grad = Tensor::zeros({B, NH, T, T}, dtype);
}

// ── the block ─────────────────────────────────────────────────────────────────

/// Sets a block's parameters to GPT-2's starting values: the matrices random, mean 0 and std 0.02, the two
/// projections (attention_out, mlp_down) with std 0.02 / sqrt(2 num_layers), as each adds onto the residual
/// stream and a model has 2 of them per layer; the biases 0; the LayerNorm weights 1. The random numbers
/// come from Tensor's generator: Tensor::manual_seed makes them the same every run.
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

/// Writes output = the block applied to input, and keeps in activations what the backward needs. input and
/// output are {batch_size, seq_len, embed_dim}; they must not be the same tensor (the second residual add
/// reads residual2, not input, but the backward reads input). Ten launches, one per box of the head comment.
inline void block_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                          const BlockActivations& activations, const BlockScratch& scratch) {
    const BlockActivations& a = activations;
    // the attention branch
    layer_norm_forward(a.ln1_output, a.ln1_mean, a.ln1_rstd, input, params.ln1_weight, params.ln1_bias);
    linear_forward(scratch.qkv, a.ln1_output, params.qkv_weight, params.qkv_bias);
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

/// Writes input_grad, and adds the parameters' gradients onto param_grads, from output_grad and what the
/// forward read and kept: input, params and activations. param_grads must be 0 before the first backward of
/// a step (each linear and LayerNorm adds onto them); input_grad is written over. The forward's steps the
/// other way round, each kernel's backward:
///
///     MLP branch         residual 2 → MLP down linear → GELU → MLP up linear → LayerNorm 2, onto residual2_grad
///     attention branch   residual 1 → attention out linear → attention → qkv linear → LayerNorm 1, onto input_grad
inline void block_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                           const Tensor& input, const BlockParams& params, const BlockActivations& activations,
                           const BlockScratch& scratch) {
    const BlockActivations& a = activations;
    const BlockScratch& s = scratch;
    // the MLP branch: residual2's gradient is output_grad (the add), then LayerNorm 2 adds the branch's part
    residual_backward(s.residual2_grad, s.branch_output_grad, output_grad);
    linear_backward(s.gelu_output_grad, param_grads.mlp_down_weight, param_grads.mlp_down_bias, s.branch_output_grad,
                    a.gelu_output, params.mlp_down_weight);
    gelu_backward(s.mlp_up_output_grad, s.gelu_output_grad, a.mlp_up_output);
    linear_backward(s.ln_output_grad, param_grads.mlp_up_weight, param_grads.mlp_up_bias, s.mlp_up_output_grad,
                    a.ln2_output, params.mlp_up_weight);
    layer_norm_backward(s.residual2_grad, param_grads.ln2_weight, param_grads.ln2_bias, s.ln_output_grad,
                        a.residual2, params.ln2_weight, a.ln2_mean, a.ln2_rstd);
    // the attention branch: input's gradient is residual2's (the add), then LayerNorm 1 adds the branch's part
    residual_backward(input_grad, s.branch_output_grad, s.residual2_grad);
    linear_backward(s.attention_output_grad, param_grads.attention_out_weight, param_grads.attention_out_bias,
                    s.branch_output_grad, a.attention_output, params.attention_out_weight);
    attention_backward(s.qkv_grad, s.attention_output_grad, a.query, a.key, a.value, a.attention_weights,
                       s.head_outputs_grad, s.weights_grad, s.query_grad, s.key_grad, s.value_grad);
    linear_backward(s.ln_output_grad, param_grads.qkv_weight, param_grads.qkv_bias, s.qkv_grad, a.ln1_output,
                    params.qkv_weight);
    layer_norm_backward(input_grad, param_grads.ln1_weight, param_grads.ln1_bias, s.ln_output_grad, input,
                        params.ln1_weight, a.ln1_mean, a.ln1_rstd);
}

#endif // BLOCK_CUH
