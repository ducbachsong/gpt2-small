// gelu_8bit.cuh — GPT-2's MLP GELU for a bfloat16 model whose linear layers multiply in 8 bits, done inside those
// layers' products.
//
//     tensor                                 type in memory
//     input (the forward's)                  bfloat16
//     gelu_input, gelu_output,
//     gelu_input_grad                        bfloat16
//     output_grad (the backward's)           bfloat16
//     the 8-bit copies                       E4M3, a float scale per row
#ifndef GELU_8BIT_CUH
#define GELU_8BIT_CUH

#include "../cuda_common.h"          // cudaCheck
#include "../../common/tensor.cuh"
#include "linear_8bit.cuh"           // the two linear layers in 8 bits: linear_8bit_forward, _backward, Linear8bitScratch
#include "../bf16/gelu_bf16.cuh"     // the epilogues, and gelu_bf16_check_beside

// ── forward and backward ──────────────────────────────────────────────────────
void gelu_8bit_forward(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor,
                       const Tensor& weight_tensor, const Tensor& bias_tensor, const Linear8bitScratch& scratch_val);
void gelu_8bit_backward(const Tensor& gelu_input_grad_tensor, const Tensor& weight_grad_tensor,
                        const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                        const Tensor& gelu_output_tensor, const Tensor& weight_tensor, const Tensor& gelu_input_tensor,
                        const Linear8bitScratch& scratch_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes gelu_input = input · weightᵀ + bias (the product in 8 bits) and gelu_output = gelu(gelu_input).
///
///   gelu_input_tensor, gelu_output_tensor, input_tensor, weight_tensor, bias_tensor   as in gelu_bf16_forward
///   scratch_val          Linear8bitScratch(rows, in_features, out_features): its output_copies are written
inline void gelu_8bit_forward(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor,
                              const Tensor& weight_tensor, const Tensor& bias_tensor, const Linear8bitScratch& scratch_val) {
    // In one product: checks gelu_output, then linear_8bit_forward with GeluBf16ForwardEpilogue (it checks the
    // rest).
    // 1. check gelu_output beside gelu_input; any of it wrong: stop with an error
    gelu_bf16_check_beside("gelu_8bit_forward", gelu_output_tensor, "gelu_output", gelu_input_tensor, "gelu_input");

    // 2. gelu_input = input · weightᵀ + bias in 8 bits, each pair's gelu into gelu_output as it is stored
    linear_8bit_forward(gelu_input_tensor, input_tensor, weight_tensor, bias_tensor, scratch_val,
                        GeluBf16ForwardEpilogue{gelu_output_tensor.data_ptr<nv_bfloat16>()});
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the backward of the linear after GELU and of GELU itself, the products in 8 bits.
///
///   gelu_input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, gelu_output_tensor,
///   weight_tensor, gelu_input_tensor   as in gelu_bf16_backward
///   scratch_val          Linear8bitScratch(rows, in_features, out_features) of the linear after GELU: its
///                        input_grad_copies and weight_grad_copies are written
inline void gelu_8bit_backward(const Tensor& gelu_input_grad_tensor, const Tensor& weight_grad_tensor,
                               const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                               const Tensor& gelu_output_tensor, const Tensor& weight_tensor, const Tensor& gelu_input_tensor,
                               const Linear8bitScratch& scratch_val) {
    // Checks gelu_input, then linear_8bit_backward with GeluBf16BackwardEpilogue (it checks the rest):
    // gelu_input_grad = (output_grad · weight) · gelu'(gelu_input), written over; weight_grad += output_gradᵀ ·
    // gelu_output and bias_grad += the columns' sums of output_grad, added onto.
    // 1. check gelu_input beside gelu_output; any of it wrong: stop with an error
    gelu_bf16_check_beside("gelu_8bit_backward", gelu_input_tensor, "gelu_input", gelu_output_tensor, "gelu_output");

    // 2. the linear's backward in 8 bits, its input_grad times gelu' as each pair is stored
    linear_8bit_backward(gelu_input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, gelu_output_tensor,
                         weight_tensor, scratch_val, GeluBf16BackwardEpilogue{gelu_input_tensor.data_ptr<nv_bfloat16>()});
}

#endif // GELU_8BIT_CUH
