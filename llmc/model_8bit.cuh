// model_8bit.cuh — the whole GPT-2 model with its products in 8 bits (FP8 E4M3) on an H100's tensor cores: the params
// (the master the 8-bit copies are made from), grads and activations stay bfloat16, as in model_bf16.cuh; only each
// product's two sides are rounded to E4M3 (one power-of-2 scale per row) just before it, and added up in float.
// The params and grads live in AdamW8bit, with AdamW's float master for this GPU's slice and m and sqrt(v) in
// 8 bits (FP8 E4M3, one float scale a warp of ADAMW_VALUES_PER_SCALE values); the model reads and writes views
// of them (GPT2Params):
//     model_buffer   [ params | grads ]             bfloat16
//     buffer         [ master params | grad_norm ]  float
//     m_bytes, m_scales, sqrt_v_bytes, sqrt_v_scales
// One launch of adamw_update_8bit (kernels/8bit/adamw_8bit.cuh) a step. No checkpoints yet: m_slice() and
// v_slice() stop.
//     token_ids ─ encoder ─ block 0 ─ ... ─ block 11 ─ LayerNorm f ─ head ─ classifier ─ losses
//
//     step                              kernel                          in 8 bits
//     encoder, LayerNorms, residuals    bf16/                           nothing: no product
//     the 4 linears of a block          8bit/prod/<product>_8bit.cuh    output, input_grad, weight_grad: a kernel each
//     GELU                              8bit/prod/mlp_up_8bit.cuh,      inside mlp_up's forward, mlp_down's backward
//                                       mlp_down_input_grad_8bit.cuh
//     attention                         8bit/attention_flash_8bit.cuh   query · keyᵀ; query, key, value kept in E4M3
//     head                              8bit/prod/head*_8bit.cuh        logits, lnf_output_grad, wte_grad
//     loss                              bf16/softmax_loss_bf16.cuh      nothing: from the bfloat16 logits
// With GPT2Config::linear_bias false the 4 linears have no bias, as in model_bf16.cuh: Tensor() biases the products
// take as none, and no linear_bf16_bias_grad in the backward.
// Each E4M3 is off by up to 2^-4 (6%) of its value, against bfloat16's 2^-8: the losses land a little off
// model_bf16.cuh's. Every product rounds its own two sides (round_8bit), so the LayerNorms and the encoder run in
// bfloat16: layer_norm_8bit.cuh's and encoder_8bit.cuh's copies would be made again by the product after them.
// Needs an H100 (WGMMA and TMA), built with -arch=sm_90a; head_size FLASH_HEAD_SIZE (128), seq_len a multiple of 64, and batch_size x
// seq_len, embed_dim and padded_vocab_size multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE (16).
#ifndef MODEL_8BIT_CUH
#define MODEL_8BIT_CUH

#include <utility>
#include <vector>
#include "common/tensor.cuh"
#include "common/model_params.cuh"                  // GPT2Config, BlockParams, GPT2Params, the starting values, the mean loss
#include "kernels/bf16/encoder_bf16.cuh"
#include "kernels/bf16/layer_norm_bf16.cuh"
#include "kernels/bf16/residual_bf16.cuh"
#include "kernels/bf16/linear_bf16.cuh"            // linear_bf16_bias_grad: the linears' bias gradients, in float from bfloat16
#include "kernels/bf16/softmax_loss_bf16.cuh"      // the softmax, the loss and its gradient from the bfloat16 logits
#include "kernels/8bit/attention_flash_8bit.cuh"   // attention_8bit_forward, _backward
#include "kernels/8bit/adamw_8bit.cuh"            // AdamW8bit's update
// GPT-2's 15 matrix products in 8 bits, each a kernel of its own with its own setting (its file's defines, from its
// tests/8bit/prod/tune_<product>_8bit.cu) and its own 8-bit copies: a layer's forward, its backward's input_grad and
// weight_grad, and the head's
#include "kernels/8bit/prod/qkv_8bit.cuh"
#include "kernels/8bit/prod/qkv_input_grad_8bit.cuh"
#include "kernels/8bit/prod/qkv_weight_grad_8bit.cuh"
#include "kernels/8bit/prod/attention_output_8bit.cuh"
#include "kernels/8bit/prod/attention_output_input_grad_8bit.cuh"
#include "kernels/8bit/prod/attention_output_weight_grad_8bit.cuh"
#include "kernels/8bit/prod/mlp_up_8bit.cuh"                    // with GELU: gelu_output written beside mlp_up_output
#include "kernels/8bit/prod/mlp_up_input_grad_8bit.cuh"
#include "kernels/8bit/prod/mlp_up_weight_grad_8bit.cuh"
#include "kernels/8bit/prod/mlp_down_8bit.cuh"
#include "kernels/8bit/prod/mlp_down_input_grad_8bit.cuh"       // with GELU's backward: times gelu'(mlp_up_output)
#include "kernels/8bit/prod/mlp_down_weight_grad_8bit.cuh"
#include "kernels/8bit/prod/head_8bit.cuh"
#include "kernels/8bit/prod/head_input_grad_8bit.cuh"
#include "kernels/8bit/prod/head_weight_grad_8bit.cuh"
#include "kernels/8bit/round_8bit.cuh"                          // round_8bit_rows_and_columns: output_grad's two copies in two reads

// The backward rounds each linear's operands once, as NVIDIA Transformer Engine's FP8 training does: output_grad into
// both its copies by one round_8bit_rows_and_columns (the input_grad's left as it sits, the weight_grad's left turned),
// the weight turned (the input_grad's right) and the input turned (the weight_grad's right), then both products on
// those copies (<product>_rounded). MODEL_8BIT_WEIGHT_GRAD_BF16 1: the weight_grads in bfloat16 instead
// (bf16/prod/*_weight_grad_bf16.cuh, from the bfloat16 output_grad and input), as torchao's float8 recipe
// rowwise_with_gw_hp does: no turned copies at all, and the weight gradients without E4M3's rounding.
#ifndef MODEL_8BIT_WEIGHT_GRAD_BF16
#define MODEL_8BIT_WEIGHT_GRAD_BF16 0
#endif
#if MODEL_8BIT_WEIGHT_GRAD_BF16
#include "kernels/bf16/prod/qkv_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/attention_output_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/mlp_up_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/mlp_down_weight_grad_bf16.cuh"
#include "kernels/bf16/prod/head_weight_grad_bf16.cuh"
#endif

/// Everything the model works in: mean, rstd, the scales, attention_logsumexp and the per-row tensors float;
/// query, key and value in E4M3 (4 in each int of an Int32 tensor) with a float scale per head row; the rest
/// bfloat16. Kept for the backward: the model's tensors, and a block's as lists, one tensor per layer
/// (ln1_output[l] is layer l's); qkv itself is not kept: the backward turns query, key and value back from their
/// 8-bit copies. With the _scratch suffix: used only inside one forward or backward, one of each, shared by all
/// the layers, with the 8-bit copies of each product (one set per product: its file's copies struct).
struct GPT2Activations8bit {
    // ── kept for the backward: the model's ──
    Tensor encoded;                                         // {B, T, D}: the encoder's output, block 0's input
    std::vector<Tensor> block_outputs;                      // {B, T, D} each: block l's output, block l + 1's input
    Tensor lnf_output;                                      // {B, T, D}
    Tensor lnf_mean;                                        // {B, T}
    Tensor lnf_rstd;                                        // {B, T}
    Tensor logits;                                          // {B, T, padded_vocab_size}; the backward writes logits_grad
                                                            // over it (softmax_loss_bf16_backward allows it): 6.6 GB less
                                                            // at {64, 1024}
    Tensor losses;                                          // {B, T}
    Tensor max_of_each_row;                                 // {B, T}
    Tensor sum_of_each_row;                                 // {B, T}
    // ── kept for the backward: one per layer ──
    std::vector<Tensor> ln1_output;                         // {B, T, D}
    std::vector<Tensor> ln1_mean;                           // {B, T}
    std::vector<Tensor> ln1_rstd;                           // {B, T}
    std::vector<Tensor> query_8bit;                         // {B, NH, T, HS / 4} Int32
    std::vector<Tensor> key_8bit;                           // {B, NH, T, HS / 4} Int32
    std::vector<Tensor> value_8bit;                         // {B, NH, T, HS / 4} Int32
    std::vector<Tensor> query_scales;                       // {B, NH, T}
    std::vector<Tensor> key_scales;                         // {B, NH, T}
    std::vector<Tensor> value_scales;                       // {B, NH, T}
    std::vector<Tensor> attention_logsumexp;                // {B, NH, T}
    std::vector<Tensor> attention_output;                   // {B, T, D}: the heads merged, before the projection
    std::vector<Tensor> residual2;                          // {B, T, D}: input + the attention branch
    std::vector<Tensor> ln2_output;                         // {B, T, D}
    std::vector<Tensor> ln2_mean;                           // {B, T}
    std::vector<Tensor> ln2_rstd;                           // {B, T}
    std::vector<Tensor> mlp_up_output;                      // {B, T, 4D}
    std::vector<Tensor> gelu_output;                        // {B, T, 4D}
    // ── only inside one forward or backward, shared by all the layers ──
    Tensor qkv_scratch;                                     // {B, T, 3D}: forward, the qkv linear's output; backward,
                                                            // query, key, value turned back from 8 bits
    Tensor branch_output_scratch;                           // {B, T, D}: a branch's output, before its residual add
    Tensor residual2_grad_scratch;                          // {B, T, D}
    Tensor branch_output_grad_scratch;                      // {B, T, D}
    Tensor ln_output_grad_scratch;                          // {B, T, D}
    Tensor attention_output_grad_scratch;                   // {B, T, D}
    Tensor mlp_up_output_grad_scratch;                      // {B, T, 4D}
    Tensor qkv_grad_scratch;                                // {B, T, 3D}
    Tensor attention_row_dot_scratch;                       // {B, NH, T} float
    // each product's 8-bit copies {rows of output, its columns, its sum's length}
    Qkv8bitCopies qkv_copies_scratch;                                        // {rows, 3D, D}
    QkvInputGrad8bitCopies qkv_input_grad_copies_scratch;                    // {rows, D, 3D}
    QkvWeightGrad8bitCopies qkv_weight_grad_copies_scratch;                  // {3D, D, rows}
    AttentionOutput8bitCopies attention_output_copies_scratch;               // {rows, D, D}
    AttentionOutputInputGrad8bitCopies attention_output_input_grad_copies_scratch;     // {rows, D, D}
    AttentionOutputWeightGrad8bitCopies attention_output_weight_grad_copies_scratch;   // {D, D, rows}
    MlpUp8bitCopies mlp_up_copies_scratch;                                   // {rows, 4D, D}
    MlpUpInputGrad8bitCopies mlp_up_input_grad_copies_scratch;               // {rows, D, 4D}
    MlpUpWeightGrad8bitCopies mlp_up_weight_grad_copies_scratch;             // {4D, D, rows}
    MlpDown8bitCopies mlp_down_copies_scratch;                               // {rows, D, 4D}
    MlpDownInputGrad8bitCopies mlp_down_input_grad_copies_scratch;           // {rows, 4D, D}
    MlpDownWeightGrad8bitCopies mlp_down_weight_grad_copies_scratch;         // {D, 4D, rows}
    Head8bitCopies head_copies_scratch;                                      // {rows, padded_vocab_size, D}
    HeadInputGrad8bitCopies head_input_grad_copies_scratch;                  // {rows, D, padded_vocab_size}
    HeadWeightGrad8bitCopies head_weight_grad_copies_scratch;                // {padded_vocab_size, D, rows}
    Tensor lnf_output_grad_scratch;                         // {B, T, D}
    Tensor residual_grads_scratch[2];                       // {B, T, D}: a block's output_grad and input_grad, in turns

    GPT2Activations8bit(size_t batch_size, size_t seq_len, const GPT2Config& config);
};

// ── one block ─────────────────────────────────────────────────────────────────
void block_8bit_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                        const GPT2Activations8bit& activations, size_t layer);
void block_8bit_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                         const Tensor& input, const BlockParams& params, const GPT2Activations8bit& activations,
                         size_t layer);

// ── the whole model ───────────────────────────────────────────────────────────
void gpt2_8bit_forward(const GPT2Activations8bit& activations, const Tensor& token_ids, const Tensor& targets,
                       const GPT2Params& params, const GPT2Config& config);
void gpt2_8bit_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                        const GPT2Params& params, const GPT2Activations8bit& activations, const GPT2Config& config,
                        size_t micro_batches = 1);
float gpt2_8bit_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                           const GPT2Params& params, const GPT2Activations8bit& activations, const GPT2Config& config,
                           size_t micro_batches = 1);

// ── the params, grads and AdamW ──────────────────────────────────────────────
/// The model's params and grads (bfloat16, all of them on every GPU), AdamW's float master for this GPU's slice,
/// and m and sqrt(v) in 8 bits; the update: one launch of adamw_update_8bit.
class AdamW8bit : public Optimizer {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    AdamW8bit(const std::vector<Shape>& param_shapes, const AdamWConfig& config,  // the matrices first
              int rank = 0, int world_size = 1);
    void copy_params_to_master() override;                              // once, after filling param(k)

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& param(int k) const override;                          // parameter tensor k: bfloat16
    const Tensor& grad(int k) const override;                           // its gradient: bfloat16
    const Tensor& all_params() const override;                          // every param(k), one flat tensor
    const Tensor& all_grads() const override;                           // every grad(k), one flat tensor
    const Tensor& master_param(int k) const;                            // the float copy the update changes
    Tensor master_slice() const override;                               // this slice's master, flat
    const Tensor& m_slice() const override;                             // stops: no float or bfloat16 m
    const Tensor& v_slice() const override;                             // ...nor v

protected:
    void update(int t) override;                                        // AdamW's step t: one launch

private:
    TensorBuffer model_buffer_;          // bfloat16: the params and grads
    int params_group_index_;             // model_buffer_'s group of the params
    int grads_group_index_;              // ...of the grads
    int master_group_index_;             // buffer_'s group of the master params, this slice only
    Tensor m_bytes_;                     // m / scale, FP8_PER_INT bytes to an int, this slice only
    Tensor m_scales_;                    // ...one float scale per ADAMW_VALUES_PER_SCALE values
    Tensor sqrt_v_bytes_;                // sqrt(v) / scale, the same
    Tensor sqrt_v_scales_;
    [[noreturn]] void stop_no_moments_tensor(const char* what) const;   // m and v as tensors: not in 8 bits
};

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── the activations ───────────────────────────────────────────────────────────

/// Makes everything the model works in, all 0: the model's kept tensors, a block's for each layer, and the
/// scratch with the 8-bit copies of each product.
///
///   batch_size, seq_len   B and T; T at most config.max_seq_len
///   config                the sizes; head_size HS = D / NH
inline GPT2Activations8bit::GPT2Activations8bit(size_t batch_size, size_t seq_len, const GPT2Config& config)
    : qkv_copies_scratch(batch_size * seq_len, 3 * config.embed_dim, config.embed_dim),
      qkv_input_grad_copies_scratch(batch_size * seq_len, config.embed_dim, 3 * config.embed_dim),
      qkv_weight_grad_copies_scratch(3 * config.embed_dim, config.embed_dim, batch_size * seq_len),
      attention_output_copies_scratch(batch_size * seq_len, config.embed_dim, config.embed_dim),
      attention_output_input_grad_copies_scratch(batch_size * seq_len, config.embed_dim, config.embed_dim),
      attention_output_weight_grad_copies_scratch(config.embed_dim, config.embed_dim, batch_size * seq_len),
      mlp_up_copies_scratch(batch_size * seq_len, 4 * config.embed_dim, config.embed_dim),
      mlp_up_input_grad_copies_scratch(batch_size * seq_len, config.embed_dim, 4 * config.embed_dim),
      mlp_up_weight_grad_copies_scratch(4 * config.embed_dim, config.embed_dim, batch_size * seq_len),
      mlp_down_copies_scratch(batch_size * seq_len, config.embed_dim, 4 * config.embed_dim),
      mlp_down_input_grad_copies_scratch(batch_size * seq_len, 4 * config.embed_dim, config.embed_dim),
      mlp_down_weight_grad_copies_scratch(config.embed_dim, 4 * config.embed_dim, batch_size * seq_len),
      head_copies_scratch(batch_size * seq_len, config.padded_vocab_size, config.embed_dim),
      head_input_grad_copies_scratch(batch_size * seq_len, config.embed_dim, config.padded_vocab_size),
      head_weight_grad_copies_scratch(config.padded_vocab_size, config.embed_dim, batch_size * seq_len) {
    size_t B = batch_size;
    size_t T = seq_len;
    size_t D = config.embed_dim;
    size_t NH = config.num_heads;
    size_t HS = D / NH;
    DType bf16 = DType::BFloat16;
    Shape head_rows_8bit = {B, NH, T, HS / ATTENTION_8BIT_VALUES_PER_INT};
    // 1. kept: the model's
    encoded = Tensor::zeros({B, T, D}, bf16);
    lnf_output = Tensor::zeros({B, T, D}, bf16);
    lnf_mean = Tensor::zeros({B, T}), lnf_rstd = Tensor::zeros({B, T});
    logits = Tensor::zeros({B, T, config.padded_vocab_size}, bf16);
    losses = Tensor::zeros({B, T}), max_of_each_row = Tensor::zeros({B, T}), sum_of_each_row = Tensor::zeros({B, T});
    // 2. kept: one per layer
    for (size_t l = 0; l < config.num_layers; l++) {
        block_outputs.push_back(Tensor::zeros({B, T, D}, bf16));
        ln1_output.push_back(Tensor::zeros({B, T, D}, bf16));
        ln1_mean.push_back(Tensor::zeros({B, T})), ln1_rstd.push_back(Tensor::zeros({B, T}));
        query_8bit.push_back(Tensor::zeros(head_rows_8bit, DType::Int32));
        key_8bit.push_back(Tensor::zeros(head_rows_8bit, DType::Int32));
        value_8bit.push_back(Tensor::zeros(head_rows_8bit, DType::Int32));
        query_scales.push_back(Tensor::zeros({B, NH, T})), key_scales.push_back(Tensor::zeros({B, NH, T}));
        value_scales.push_back(Tensor::zeros({B, NH, T}));
        attention_logsumexp.push_back(Tensor::zeros({B, NH, T}));
        attention_output.push_back(Tensor::zeros({B, T, D}, bf16));
        residual2.push_back(Tensor::zeros({B, T, D}, bf16));
        ln2_output.push_back(Tensor::zeros({B, T, D}, bf16));
        ln2_mean.push_back(Tensor::zeros({B, T})), ln2_rstd.push_back(Tensor::zeros({B, T}));
        mlp_up_output.push_back(Tensor::zeros({B, T, 4 * D}, bf16));
        gelu_output.push_back(Tensor::zeros({B, T, 4 * D}, bf16));
    }
    // 3. scratch (the 8-bit copies made above)
    qkv_scratch = Tensor::zeros({B, T, 3 * D}, bf16);
    branch_output_scratch = Tensor::zeros({B, T, D}, bf16);
    residual2_grad_scratch = Tensor::zeros({B, T, D}, bf16), branch_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    ln_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    attention_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    mlp_up_output_grad_scratch = Tensor::zeros({B, T, 4 * D}, bf16);
    qkv_grad_scratch = Tensor::zeros({B, T, 3 * D}, bf16);
    attention_row_dot_scratch = Tensor::zeros({B, NH, T});
    lnf_output_grad_scratch = Tensor::zeros({B, T, D}, bf16);
    residual_grads_scratch[0] = Tensor::zeros({B, T, D}, bf16), residual_grads_scratch[1] = Tensor::zeros({B, T, D}, bf16);
}

// ── one block ─────────────────────────────────────────────────────────────────

/// Writes output = block `layer` applied to input with its products in 8 bits, keeping what the backward needs in
/// that layer's activations.
///
///   output, input   {batch_size, seq_len, embed_dim} bfloat16; not the same tensor (the backward reads input)
///   params          the block's, bfloat16
///   activations     the layer's lists written, the scratch used
///   layer           which block: 0 .. num_layers - 1
inline void block_8bit_forward(const Tensor& output, const Tensor& input, const BlockParams& params,
                               const GPT2Activations8bit& activations, size_t layer) {
    const GPT2Activations8bit& a = activations;
    size_t l = layer;
    // the attention branch; attention keeps query, key, value in 8 bits, so qkv is only scratch
    layer_norm_bf16_forward(a.ln1_output[l], a.ln1_mean[l], a.ln1_rstd[l], input, params.ln1_weight, params.ln1_bias);
    qkv_8bit(a.qkv_scratch, a.ln1_output[l], params.qkv_weight, params.qkv_bias, a.qkv_copies_scratch);
    attention_8bit_forward(a.attention_output[l], a.query_8bit[l], a.key_8bit[l], a.value_8bit[l], a.query_scales[l],
                           a.key_scales[l], a.value_scales[l], a.attention_logsumexp[l], a.qkv_scratch);
    attention_output_8bit(a.branch_output_scratch, a.attention_output[l], params.attention_out_weight, params.attention_out_bias,
                          a.attention_output_copies_scratch);
    residual_bf16_forward(a.residual2[l], input, a.branch_output_scratch);
    // the MLP branch; mlp_up and GELU in one product: GELU as each pair of mlp_up_output is stored
    layer_norm_bf16_forward(a.ln2_output[l], a.ln2_mean[l], a.ln2_rstd[l], a.residual2[l], params.ln2_weight,
                            params.ln2_bias);
    mlp_up_8bit(a.mlp_up_output[l], a.gelu_output[l], a.ln2_output[l], params.mlp_up_weight, params.mlp_up_bias,
                a.mlp_up_copies_scratch);
    mlp_down_8bit(a.branch_output_scratch, a.gelu_output[l], params.mlp_down_weight, params.mlp_down_bias, a.mlp_down_copies_scratch);
    residual_bf16_forward(output, a.residual2[l], a.branch_output_scratch);
}

/// Rounds a linear layer's backward operands once each into its two products' copies: output_grad both ways in two
/// reads (round_8bit_rows_and_columns: the input_grad's left as it sits, one scale a token; the weight_grad's left
/// turned, one scale an output feature), the weight turned (the input_grad's right) and the input turned (the
/// weight_grad's right). With MODEL_8BIT_WEIGHT_GRAD_BF16, only the input_grad's: its weight_grad is bfloat16.
///
///   input_grad_copies_val    the input_grad product's copies: written
///   weight_grad_copies_val   the weight_grad product's: written (not with MODEL_8BIT_WEIGHT_GRAD_BF16)
///   output_grad              {B, T, out} bfloat16
///   weight                   {out, in} bfloat16
///   input                    {B, T, in} bfloat16: the forward's input of the linear
/// INPUT_GRAD_COPIES, WEIGHT_GRAD_COPIES: the products' copies types, e.g. QkvInputGrad8bitCopies, QkvWeightGrad8bitCopies.
template <typename INPUT_GRAD_COPIES, typename WEIGHT_GRAD_COPIES>
inline void linear_8bit_round_backward(const INPUT_GRAD_COPIES& input_grad_copies_val, const WEIGHT_GRAD_COPIES& weight_grad_copies_val,
                                       const Tensor& output_grad, const Tensor& weight, const Tensor& input) {
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    (void)weight_grad_copies_val, (void)input;
    round_8bit(input_grad_copies_val.left_8bit_tensor, input_grad_copies_val.left_scales_tensor, output_grad);
#else
    round_8bit_rows_and_columns(input_grad_copies_val.left_8bit_tensor, input_grad_copies_val.left_scales_tensor,
                                weight_grad_copies_val.left_8bit_tensor, weight_grad_copies_val.left_scales_tensor, output_grad);
    round_8bit(weight_grad_copies_val.right_8bit_tensor, weight_grad_copies_val.right_scales_tensor, input, true);
#endif
    round_8bit(input_grad_copies_val.right_8bit_tensor, input_grad_copies_val.right_scales_tensor, weight, true);
}

/// Writes input_grad and adds block `layer`'s parameter gradients onto param_grads, the products in 8 bits: the
/// forward's steps the other way. Each linear's operands rounded once (linear_8bit_round_backward), then its two
/// products on the copies.
///
///   input_grad                     written over
///   param_grads                    added onto: 0 before a step's first backward
///   output_grad                    the gradient from the next block
///   input, params, activations     the forward's; activations' scratch used
///   layer                          which block: 0 .. num_layers - 1
inline void block_8bit_backward(const Tensor& input_grad, const BlockParams& param_grads, const Tensor& output_grad,
                                const Tensor& input, const BlockParams& params, const GPT2Activations8bit& activations,
                                size_t layer) {
    const GPT2Activations8bit& a = activations;
    size_t l = layer;
    // the MLP branch; LayerNorm 2 adds its part onto residual2_grad
    residual_bf16_backward(a.residual2_grad_scratch, a.branch_output_grad_scratch, output_grad);
    // mlp_down's backward and GELU's in one: its input_grad times gelu' as each pair is stored
    linear_8bit_round_backward(a.mlp_down_input_grad_copies_scratch, a.mlp_down_weight_grad_copies_scratch, a.branch_output_grad_scratch,
                               params.mlp_down_weight, a.gelu_output[l]);
    mlp_down_input_grad_8bit_rounded(a.mlp_up_output_grad_scratch, a.mlp_up_output[l], a.mlp_down_input_grad_copies_scratch);
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    mlp_down_weight_grad_bf16(param_grads.mlp_down_weight, a.branch_output_grad_scratch, a.gelu_output[l]);
#else
    mlp_down_weight_grad_8bit_rounded(param_grads.mlp_down_weight, a.mlp_down_weight_grad_copies_scratch);
#endif
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.mlp_down_bias, a.branch_output_grad_scratch);
    linear_8bit_round_backward(a.mlp_up_input_grad_copies_scratch, a.mlp_up_weight_grad_copies_scratch, a.mlp_up_output_grad_scratch,
                               params.mlp_up_weight, a.ln2_output[l]);
    mlp_up_input_grad_8bit_rounded(a.ln_output_grad_scratch, a.mlp_up_input_grad_copies_scratch);
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    mlp_up_weight_grad_bf16(param_grads.mlp_up_weight, a.mlp_up_output_grad_scratch, a.ln2_output[l]);
#else
    mlp_up_weight_grad_8bit_rounded(param_grads.mlp_up_weight, a.mlp_up_weight_grad_copies_scratch);
#endif
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.mlp_up_bias, a.mlp_up_output_grad_scratch);
    layer_norm_bf16_backward(a.residual2_grad_scratch, param_grads.ln2_weight, param_grads.ln2_bias,
                             a.ln_output_grad_scratch, a.residual2[l], params.ln2_weight, a.ln2_mean[l], a.ln2_rstd[l]);
    // the attention branch; LayerNorm 1 adds its part onto input_grad
    residual_bf16_backward(input_grad, a.branch_output_grad_scratch, a.residual2_grad_scratch);
    linear_8bit_round_backward(a.attention_output_input_grad_copies_scratch, a.attention_output_weight_grad_copies_scratch,
                               a.branch_output_grad_scratch, params.attention_out_weight, a.attention_output[l]);
    attention_output_input_grad_8bit_rounded(a.attention_output_grad_scratch, a.attention_output_input_grad_copies_scratch);
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    attention_output_weight_grad_bf16(param_grads.attention_out_weight, a.branch_output_grad_scratch, a.attention_output[l]);
#else
    attention_output_weight_grad_8bit_rounded(param_grads.attention_out_weight, a.attention_output_weight_grad_copies_scratch);
#endif
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.attention_out_bias, a.branch_output_grad_scratch);
    attention_8bit_backward(a.qkv_grad_scratch, a.attention_output_grad_scratch, a.attention_output[l],
                            a.query_8bit[l], a.key_8bit[l], a.value_8bit[l], a.query_scales[l], a.key_scales[l],
                            a.value_scales[l], a.attention_logsumexp[l], a.qkv_scratch, a.attention_row_dot_scratch);
    linear_8bit_round_backward(a.qkv_input_grad_copies_scratch, a.qkv_weight_grad_copies_scratch, a.qkv_grad_scratch, params.qkv_weight,
                               a.ln1_output[l]);
    qkv_input_grad_8bit_rounded(a.ln_output_grad_scratch, a.qkv_input_grad_copies_scratch);
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    qkv_weight_grad_bf16(param_grads.qkv_weight, a.qkv_grad_scratch, a.ln1_output[l]);
#else
    qkv_weight_grad_8bit_rounded(param_grads.qkv_weight, a.qkv_weight_grad_copies_scratch);
#endif
    if (param_grads.has_linear_bias()) linear_bf16_bias_grad(param_grads.qkv_bias, a.qkv_grad_scratch);
    layer_norm_bf16_backward(input_grad, param_grads.ln1_weight, param_grads.ln1_bias, a.ln_output_grad_scratch, input,
                             params.ln1_weight, a.ln1_mean[l], a.ln1_rstd[l]);
}

// ── the whole model ───────────────────────────────────────────────────────────

/// Works out each row's loss into activations.losses with the products in 8 bits, keeping what the backward needs.
///
///   activations          written
///   token_ids, targets   {batch_size, seq_len} ints, every id < vocab_size
///   params               the model's, bfloat16
///   config               the sizes
inline void gpt2_8bit_forward(const GPT2Activations8bit& activations, const Tensor& token_ids, const Tensor& targets,
                              const GPT2Params& params, const GPT2Config& config) {
    const GPT2Activations8bit& a = activations;
    size_t num_layers = params.blocks.size();
    encoder_bf16_forward(a.encoded, token_ids, params.wte, params.wpe);
    for (size_t l = 0; l < num_layers; l++) {
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_8bit_forward(a.block_outputs[l], input, params.blocks[l], a, l);
    }
    layer_norm_bf16_forward(a.lnf_output, a.lnf_mean, a.lnf_rstd, a.block_outputs[num_layers - 1], params.lnf_weight,
                            params.lnf_bias);
    head_8bit(a.logits, a.lnf_output, params.wte, a.head_copies_scratch);   // the head: no bias
    softmax_loss_bf16_forward(a.losses, a.max_of_each_row, a.sum_of_each_row, a.logits, targets, config.vocab_size);
}

/// Adds every parameter's gradient onto param_grads for the mean loss, the products in 8 bits: the forward's
/// steps the other way.
///
///   param_grads                   added onto: 0 before a step's first backward
///   token_ids, targets            the forward's
///   params, activations           the forward's; activations' scratch used
///   config                        the sizes
///   micro_batches                 per step: each row's losses_grad is 1 / (micro_batches x B x T)
inline void gpt2_8bit_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                               const GPT2Params& params, const GPT2Activations8bit& activations,
                               const GPT2Config& config, size_t micro_batches) {
    const GPT2Activations8bit& a = activations;
    size_t num_layers = params.blocks.size();
    // loss = sum of the row losses / (micro_batches x B x T); logits_grad written over the logits, then the head's
    // backward: lnf_output_grad, and its part added onto wte_grad
    float loss_grad_val = (float)(1.0 / ((double)micro_batches * a.losses.numel()));
    softmax_loss_bf16_backward(a.logits, a.logits, targets, a.max_of_each_row, a.sum_of_each_row, config.vocab_size, loss_grad_val);
    linear_8bit_round_backward(a.head_input_grad_copies_scratch, a.head_weight_grad_copies_scratch, a.logits, params.wte, a.lnf_output);
    head_input_grad_8bit_rounded(a.lnf_output_grad_scratch, a.head_input_grad_copies_scratch);
#if MODEL_8BIT_WEIGHT_GRAD_BF16
    head_weight_grad_bf16(param_grads.wte, a.logits, a.lnf_output);
#else
    head_weight_grad_8bit_rounded(param_grads.wte, a.head_weight_grad_copies_scratch);
#endif
    Tensor output_grad = a.residual_grads_scratch[0];
    Tensor input_grad = a.residual_grads_scratch[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto it
    layer_norm_bf16_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, a.lnf_output_grad_scratch,
                             a.block_outputs[num_layers - 1], params.lnf_weight, a.lnf_mean, a.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? a.encoded : a.block_outputs[l - 1];
        block_8bit_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], a, l);
        std::swap(output_grad, input_grad);                 // block l - 1's output_grad
    }
    encoder_bf16_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Runs one micro-batch of a training step, the products in 8 bits: the forward, then the backward adding every parameter's
/// gradient onto param_grads. Gives the micro-batch's mean loss, so waits for the GPU.
///
///   param_grads, token_ids, targets, params, activations, config, micro_batches   as in gpt2_8bit_backward
inline float gpt2_8bit_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                  const GPT2Params& params, const GPT2Activations8bit& activations, const GPT2Config& config,
                                  size_t micro_batches) {
    gpt2_8bit_forward(activations, token_ids, targets, params, config);
    gpt2_8bit_backward(param_grads, token_ids, targets, params, activations, config, micro_batches);
    return gpt2_mean_loss(activations);
}

// ── AdamW8bit: making it ─────────────────────────────────────────────────────

/// Makes an 8-bit AdamW: the master for this slice and grad_norm in the float buffer; the params and grads in
/// the bfloat16 one; m and sqrt(v) for this slice in 8 bits with their scales; all 0.
///
///   param_shapes        the matrices first
///   config              the settings
///   rank, world_size    this GPU of how many; the values must split into slices of a multiple of 4
inline AdamW8bit::AdamW8bit(const std::vector<Shape>& param_shapes, const AdamWConfig& config, int rank,
                            int world_size)
    : Optimizer(param_shapes, config, DType::BFloat16, rank, world_size) {
    // 1. the master and grad_norm; the params and grads
    master_group_index_ = buffer_.add_many(slice_shapes());
    allocate();
    params_group_index_ = model_buffer_.add_many(param_shapes);
    grads_group_index_ = model_buffer_.add_many(param_shapes);
    model_buffer_.allocate(DType::BFloat16);

    // 2. m and sqrt(v): 1 byte a value, a float scale a warp
    size_t bytes_as_ints = slice_size() / FP8_PER_INT;                  // whole: slice_size is a multiple of 4
    size_t scales = CEIL_DIV(slice_size(), (size_t)ADAMW_VALUES_PER_SCALE);
    m_bytes_ = Tensor::zeros({bytes_as_ints}, DType::Int32);
    m_scales_ = Tensor::zeros({scales});
    sqrt_v_bytes_ = Tensor::zeros({bytes_as_ints}, DType::Int32);
    sqrt_v_scales_ = Tensor::zeros({scales});
}

/// Copies this slice of the model's params into the float master, once, after filling param(k) and before the
/// first step. Exact: every bfloat16 is a float.
inline void AdamW8bit::copy_params_to_master() {
    buffer_.group(master_group_index_).copy_(all_params().narrow(0, slice_start(), slice_size()));
}

// ── AdamW8bit: reading ───────────────────────────────────────────────────────

/// Gives parameter tensor k, bfloat16.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW8bit::param(int k) const { return model_buffer_.tensor(params_group_index_, k); }

/// Gives the gradient of parameter k, bfloat16.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW8bit::grad(int k) const { return model_buffer_.tensor(grads_group_index_, k); }

/// Gives every param(k) as one flat tensor.
inline const Tensor& AdamW8bit::all_params() const { return model_buffer_.group(params_group_index_); }

/// Gives every grad(k) as one flat tensor: one launch zeroes them all.
inline const Tensor& AdamW8bit::all_grads() const { return model_buffer_.group(grads_group_index_); }

/// Gives the float copy of parameter k the update changes. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW8bit::master_param(int k) const {
    stop_unless_one_gpu("master_param(k)");
    return buffer_.tensor(master_group_index_, k);
}

/// Gives this slice's master params as one flat tensor.
inline Tensor AdamW8bit::master_slice() const { return buffer_.group(master_group_index_); }

/// Stops: 8-bit m has no float or bfloat16 tensor to give (checkpoints do not save 8-bit m and v yet).
inline const Tensor& AdamW8bit::m_slice() const { stop_no_moments_tensor("m_slice()"); }

/// Stops: 8-bit v has no float or bfloat16 tensor to give.
inline const Tensor& AdamW8bit::v_slice() const { stop_no_moments_tensor("v_slice()"); }

/// Stops with a message: m and v are bytes and scales here, not a tensor of values.
///
///   what   the call, for the message
inline void AdamW8bit::stop_no_moments_tensor(const char* what) const {
    fprintf(stderr, "error: %s is not there with 8-bit m and v\n", what);
    exit(EXIT_FAILURE);
}

// ── AdamW8bit: the update ────────────────────────────────────────────────────

/// Does AdamW's step t on this slice with one launch of adamw_update_8bit: m and sqrt(v) unpacked from 8 bits,
/// the master moved in float, then rounded into this slice of the params, m and sqrt(v) packed again. grad_norm
/// is the norm of all the gradients, so every GPU clips alike.
///
///   t   the step number: 1, 2, 3, ...
inline void AdamW8bit::update(int t) {
    size_t start = slice_start();
    adamw_update_8bit(all_params().data_ptr<nv_bfloat16>() + start, all_grads().data_ptr<nv_bfloat16>() + start,
                      (__nv_fp8_e4m3*)m_bytes_.data_ptr<int>(), m_scales_.data(),
                      (__nv_fp8_e4m3*)sqrt_v_bytes_.data_ptr<int>(), sqrt_v_scales_.data(), slice_size(),
                      decayed_in_slice(), grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                      config_.beta2, config_.eps, config_.weight_decay, t, buffer_.group(master_group_index_).data());
}

#endif // MODEL_8BIT_CUH
