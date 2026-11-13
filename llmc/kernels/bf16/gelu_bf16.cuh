// gelu_bf16.cuh — GPT-2's MLP GELU in bfloat16, done inside the linear layers around it on an H100's tensor cores,
// with no pass of its own over memory:
//     forward:    gelu_input  = input · weightᵀ + bias            the MLP's first linear (mlp_up)
//                 gelu_output = gelu(gelu_input)                  in the same product, as each pair is stored
//     backward:   gelu_input_grad = (output_grad · weight) · gelu'(gelu_input)
//                                                                 the second linear's (mlp_down's) input_grad,
//                                                                 times gelu' as each pair is stored
//                 weight_grad += output_gradᵀ · gelu_output       that linear's weight_grad and bias_grad, as usual
//
//     tensor                                 type in memory   shape (GPT-2's MLP)
//     input (the forward's)                  bfloat16         {batch_size, seq_len, embed_dim}
//     gelu_input, gelu_output,
//     gelu_input_grad                        bfloat16         {batch_size, seq_len, 4 · embed_dim}
//     output_grad (the backward's)           bfloat16         {batch_size, seq_len, embed_dim}
//     sums                                   float            in registers
//
// How: wgmma_matmul_bf16.cuh's epilogue, a functor its kernel applies to each pair of results before storing it.
// GeluBf16ForwardEpilogue writes gelu of the pair to gelu_output beside it; GeluBf16BackwardEpilogue reads the pair's
// gelu_input and multiplies by gelu'. Each takes its pair rounded to bfloat16 first, as the unfused path stores it
// between its two kernels (linear_bf16_forward then gelu.cuh's gelu_forward; linear_bf16_backward then
// gelu_backward), and uses gelu.cuh's gelu_value and gelu_slope: the results are that path's to the bit, without
// its passes over memory (the forward's gelu_input read again, the backward's 25 MB gelu_output_grad written and
// read for GPT-2 small's micro-batch). Needs an H100 (WGMMA and TMA), built with -arch=sm_90a, and what
// linear_bf16.cuh needs. GELU alone, float or bfloat16, any GPU: gelu.cuh; with the products in 8 bits:
// 8bit/gelu_8bit.cuh (the same epilogues).
// Math: https://ducbachsong.github.io/posts/gelu-backward/
#ifndef GELU_BF16_CUH
#define GELU_BF16_CUH

#include "../cuda_common.h"          // cudaCheck
#include "../../common/tensor.cuh"
#include "../fp32/gelu.cuh"          // gelu_value, gelu_slope: the same math as the unfused GELU
#include "linear_bf16.cuh"           // the two linear layers, and their checks: linear_bf16_forward, _backward

/// The forward's epilogue: stores gelu of each pair of the linear's results in gelu_output, at the same index; the
/// pair itself is stored in gelu_input by the product.
struct GeluBf16ForwardEpilogue {
    nv_bfloat16* gelu_output_ptr;                    // gelu_input's shape: written

    /// Writes gelu of the pair, rounded to bfloat16 first as gelu_input holds it; gives the pair back to be stored.
    ///
    ///   pair_val    two neighbouring results of the linear (bias added), in float
    ///   index_val   the first one's index in gelu_input and gelu_output
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const {
        // 1. the pair as gelu_input will hold it
        float2 x_val = __bfloat1622float2(__floats2bfloat162_rn(pair_val.x, pair_val.y));
        // 2. gelu of each, stored beside it
        *reinterpret_cast<nv_bfloat162*>(gelu_output_ptr + index_val) = __floats2bfloat162_rn(gelu_value(x_val.x), gelu_value(x_val.y));
        return pair_val;
    }
};

/// The backward's epilogue: multiplies each pair of the second linear's input_grad (gelu_output's gradient) by
/// gelu' of the forward's gelu_input at the same index, so what is stored is gelu_input's gradient.
struct GeluBf16BackwardEpilogue {
    const nv_bfloat16* gelu_input_ptr;               // the forward's gelu_input: read

    /// Gives the pair, rounded to bfloat16 first as the unfused path stores gelu_output_grad, times gelu'(gelu_input).
    ///
    ///   pair_val    two neighbouring values of gelu_output's gradient, in float
    ///   index_val   the first one's index in gelu_input and gelu_input_grad
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const {
        // 1. gelu_output's gradient as the unfused path stores it, and the pair's gelu_input
        float2 output_grad_val = __bfloat1622float2(__floats2bfloat162_rn(pair_val.x, pair_val.y));
        float2 x_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(gelu_input_ptr + index_val));
        // 2. times gelu'
        return make_float2(output_grad_val.x * gelu_slope(x_val.x), output_grad_val.y * gelu_slope(x_val.y));
    }
};

// ── forward and backward ──────────────────────────────────────────────────────
template <typename SETTING = WgmmaMatmulBf16Setting<>>
void gelu_bf16_forward(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor,
                       const Tensor& weight_tensor, const Tensor& bias_tensor);
template <typename INPUT_GRAD_SETTING = WgmmaMatmulBf16Setting<>, typename WEIGHT_GRAD_SETTING = WgmmaMatmulBf16Setting<>>
void gelu_bf16_backward(const Tensor& gelu_input_grad_tensor, const Tensor& weight_grad_tensor,
                        const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                        const Tensor& gelu_output_tensor, const Tensor& weight_tensor, const Tensor& gelu_input_tensor);

// ── for both, and for 8bit/gelu_8bit.cuh ──────────────────────────────────────
void gelu_bf16_check_beside(const char* name_ptr, const Tensor& beside_tensor, const char* beside_name_ptr,
                            const Tensor& tensor, const char* tensor_name_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes gelu_input = input · weightᵀ + bias and gelu_output = gelu(gelu_input) in one product on the tensor
/// cores: checks gelu_output, then linear_bf16_forward with GeluBf16ForwardEpilogue (it checks the rest).
///
///   gelu_input_tensor    {..., out_features} bfloat16: the linear's output, GELU's input; kept for the backward
///   gelu_output_tensor   gelu_input's shape, bfloat16: written
///   input_tensor         {..., in_features} bfloat16: ln2's output in GPT-2's MLP
///   weight_tensor        {out_features, in_features} bfloat16
///   bias_tensor          {out_features} bfloat16, or Tensor() for none
/// SETTING: the product's WgmmaMatmulBf16Setting (wgmma_matmul_bf16.cuh's); not given: the defaults.
template <typename SETTING>
inline void gelu_bf16_forward(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor,
                              const Tensor& weight_tensor, const Tensor& bias_tensor) {
    // 1. check gelu_output beside gelu_input; any of it wrong: stop with an error
    gelu_bf16_check_beside("gelu_bf16_forward", gelu_output_tensor, "gelu_output", gelu_input_tensor, "gelu_input");

    // 2. gelu_input = input · weightᵀ + bias, each pair's gelu into gelu_output as it is stored
    linear_bf16_forward<SETTING>(gelu_input_tensor, input_tensor, weight_tensor, bias_tensor,
                        GeluBf16ForwardEpilogue{gelu_output_tensor.data_ptr<nv_bfloat16>()});
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the backward of the linear after GELU and of GELU itself: checks gelu_input, then
/// linear_bf16_backward with GeluBf16BackwardEpilogue (it checks the rest): gelu_input_grad = (output_grad · weight)
/// · gelu'(gelu_input), written over; weight_grad += output_gradᵀ · gelu_output and bias_grad += the columns' sums
/// of output_grad, added onto.
///
///   gelu_input_grad_tensor   gelu_input's shape, bfloat16: written; may be gelu_input itself (each pair is read
///                            before the same pair is written, by the same thread)
///   weight_grad_tensor       weight's shape, bfloat16: added onto
///   bias_grad_tensor         {out_features} bfloat16, or Tensor() for none: added onto
///   output_grad_tensor       {..., out_features} bfloat16: the gradient from the layer above
///   gelu_output_tensor       {..., in_features} bfloat16: the linear's input, the forward's GELU output
///   weight_tensor            {out_features, in_features} bfloat16: mlp_down's weight in GPT-2's MLP
///   gelu_input_tensor        gelu_output's shape, bfloat16: the forward's GELU input
/// INPUT_GRAD_SETTING, WEIGHT_GRAD_SETTING: the two products' WgmmaMatmulBf16Setting; not given: the defaults.
template <typename INPUT_GRAD_SETTING, typename WEIGHT_GRAD_SETTING>
inline void gelu_bf16_backward(const Tensor& gelu_input_grad_tensor, const Tensor& weight_grad_tensor,
                               const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                               const Tensor& gelu_output_tensor, const Tensor& weight_tensor, const Tensor& gelu_input_tensor) {
    // 1. check gelu_input beside gelu_output; any of it wrong: stop with an error
    gelu_bf16_check_beside("gelu_bf16_backward", gelu_input_tensor, "gelu_input", gelu_output_tensor, "gelu_output");

    // 2. the linear's backward, its input_grad times gelu' as each pair is stored
    linear_bf16_backward<INPUT_GRAD_SETTING, WEIGHT_GRAD_SETTING>(gelu_input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                                                                  output_grad_tensor, gelu_output_tensor,
                         weight_tensor, GeluBf16BackwardEpilogue{gelu_input_tensor.data_ptr<nv_bfloat16>()});
}

// ── for both, and for 8bit/gelu_8bit.cuh ──────────────────────────────────────

/// Stops with a message unless a tensor the epilogue reads or writes at the product's indices fits: the same shape
/// as the product's tensor, bfloat16, starting on a pair of bfloat16s (an epilogue moves two at a time).
///
///   name_ptr                         the caller's name, for the message
///   beside_tensor, beside_name_ptr   the epilogue's tensor, and its name: gelu_output (forward) or gelu_input
///                                    (backward)
///   tensor, tensor_name_ptr          the tensor whose shape it must have, and its name: gelu_input (forward) or
///                                    gelu_output (backward)
inline void gelu_bf16_check_beside(const char* name_ptr, const Tensor& beside_tensor, const char* beside_name_ptr,
                                   const Tensor& tensor, const char* tensor_name_ptr) {
    if (beside_tensor.shape() != tensor.shape() || beside_tensor.dtype() != DType::BFloat16 ||
        (size_t)beside_tensor.data_ptr() % sizeof(nv_bfloat162) != 0) {
        fprintf(stderr, "error: %s needs %s bfloat16, the same shape as %s, starting on %zu bytes\n", name_ptr,
                beside_name_ptr, tensor_name_ptr, sizeof(nv_bfloat162));
        exit(EXIT_FAILURE);
    }
}

#endif // GELU_BF16_CUH
