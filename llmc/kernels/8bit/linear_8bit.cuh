// linear_8bit.cuh — a linear layer of a bfloat16 model with its three products in 8 bits (FP8 E4M3) on an
// H100's tensor cores, as PyTorch's nn.Linear:
//     forward:    output = input · weightᵀ + bias
//     backward:   bias_grad   += sum over rows of output_grad      db = dyᵀ · 1
//                 weight_grad += output_gradᵀ · input              dW = dyᵀ x
//                 input_grad   = output_grad · weight              dx = dy W
//
//     tensor                     type in memory                   shape                                 why
//     input, input_grad          bfloat16                         {batch_size, seq_len, in_features}    the model's
//     weight, weight_grad        bfloat16                         {out_features, in_features}           the model's
//     bias, bias_grad            bfloat16                         {out_features}                        the model's
//     output, output_grad        bfloat16                         {batch_size, seq_len, out_features}   the model's
//     the 8-bit copies           E4M3, 4 in each int of an        Linear8bitScratch's                   scratch, made
//                                Int32 tensor, a float scale                                            again each call
//                                per row
//     sums                       float                            in registers
//
// Before each product its two sides are rounded to E4M3 by round_8bit, one power-of-2 scale per row; then
// wgmma_matmul_8bit adds up the E4M3 products in float (WGMMA m64n64k32, the tiles brought by TMA) and applies
// both scales once, as each sum is stored in bfloat16. FP8 WGMMA reads only K-major tiles (no transpose bit), so a
// side that sits turned round is turned while it is rounded (round_8bit's turned), once:
//     product      left (rows x k)                         right (columns x k)                    bias   added onto
//     output       input {rows, in}                        weight {out, in}                       bias   no
//     input_grad   output_grad {rows, out}                 weight turned: {in, out}               -      no
//     weight_grad  output_grad turned: {out, rows}         input turned: {in, rows}               -      yes
// Each E4M3 is off by up to 2^-4 (6%) of its value, against bfloat16's 2^-8, so the results move by a few
// percent; the bias and bias_grad are not rounded (bias_grad: linear_bf16.cuh's sum in float). An epilogue
// (wgmma_matmul_8bit.cuh's) can be given for output in the forward and for input_grad in the backward:
// 8bit/gelu_8bit.cuh passes GELU's.
// Needs an H100 (WGMMA and TMA), built with -arch=sm_90a, and rows, in_features and out_features multiples of
// ROUND_8BIT_ROW_LENGTH_MULTIPLE: each is the k of one product, read by TMA in whole 16-byte pieces. The same
// layer in bfloat16: linear_bf16.cuh; in float: linear.cuh.
// Math: https://ducbachsong.github.io/posts/linear-backward
#ifndef LINEAR_8BIT_CUH
#define LINEAR_8BIT_CUH

#include "../cuda_common.h"             // cudaCheck
#include "../../tensor.cuh"
#include "round_8bit.cuh"            // the 8-bit copies: round_8bit, ROUND_8BIT_VALUES_PER_INT, ROUND_8BIT_ROW_LENGTH_MULTIPLE
#include "wgmma_matmul_8bit.cuh"     // the products
#include "../bf16/linear_bf16.cuh"           // the shape checks and bias_grad: linear_bf16_check_forward, _check_backward, linear_bf16_bias_grad

/// The 8-bit copies of one product's two sides: left {rows, k} and right {columns, k}, each E4M3 (4 in each int)
/// with one float scale per row, as round_8bit writes them and wgmma_matmul_8bit reads them.
struct Linear8bitProductCopies {
    Tensor left_8bit_tensor;             // {rows, k / ROUND_8BIT_VALUES_PER_INT} Int32
    Tensor left_scales_tensor;           // {rows} float
    Tensor right_8bit_tensor;            // {columns, k / ROUND_8BIT_VALUES_PER_INT} Int32
    Tensor right_scales_tensor;          // {columns} float

    Linear8bitProductCopies() = default;
    Linear8bitProductCopies(size_t rows_val, size_t columns_val, size_t k_val);
};

/// What the layer's 8-bit copies live in, one set per product: made once for a size, used by every call. Only
/// used inside a call, so one is enough for all the layers of a model that have the same sizes.
struct Linear8bitScratch {
    Linear8bitProductCopies output_copies;        // output:      input {rows, in},        weight {out, in}
    Linear8bitProductCopies input_grad_copies;    // input_grad:  output_grad {rows, out}, weight turned {in, out}
    Linear8bitProductCopies weight_grad_copies;   // weight_grad: output_grad turned {out, rows}, input turned {in, rows}

    Linear8bitScratch() = default;
    Linear8bitScratch(size_t rows_val, size_t in_features_val, size_t out_features_val);
};

// ── forward and backward ──────────────────────────────────────────────────────
template <typename EPILOGUE = WgmmaMatmul8bitStore>
void linear_8bit_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                         const Tensor& bias_tensor, const Linear8bitScratch& scratch_val,
                         const EPILOGUE& output_epilogue_val = EPILOGUE());
template <typename EPILOGUE = WgmmaMatmul8bitStore>
void linear_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                          const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                          const Linear8bitScratch& scratch_val, const EPILOGUE& input_grad_epilogue_val = EPILOGUE());

// ── for both ──────────────────────────────────────────────────────────────────
void linear_8bit_check_sizes(const char* name_ptr, const Tensor& input_tensor, const Tensor& weight_tensor,
                             const Linear8bitScratch& scratch_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the scratch ───────────────────────────────────────────────────────────────

/// Makes one product's 8-bit copies, all 0.
///
///   rows_val, columns_val   left's and right's rows: the product's shape
///   k_val                   the length summed over: a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE
inline Linear8bitProductCopies::Linear8bitProductCopies(size_t rows_val, size_t columns_val, size_t k_val) {
    left_8bit_tensor = Tensor::zeros({rows_val, k_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    left_scales_tensor = Tensor::zeros({rows_val});
    right_8bit_tensor = Tensor::zeros({columns_val, k_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    right_scales_tensor = Tensor::zeros({columns_val});
}

/// Makes the copies of the layer's three products, all 0: (rows x out_features + out_features x in_features) +
/// (rows x out_features + in_features x out_features) + (out_features x rows + in_features x rows) bytes of E4M3s,
/// and a float per row of each.
///
///   rows_val                            batch_size x seq_len
///   in_features_val, out_features_val   the layer's
inline Linear8bitScratch::Linear8bitScratch(size_t rows_val, size_t in_features_val, size_t out_features_val)
    : output_copies(rows_val, out_features_val, in_features_val),
      input_grad_copies(rows_val, in_features_val, out_features_val),
      weight_grad_copies(out_features_val, in_features_val, rows_val) {}

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes output = input · weightᵀ + bias with the product in 8 bits: checks, rounds input and weight to E4M3
/// (both sit k next to each other already), then wgmma_matmul_8bit.
///
///   output_tensor   {..., out_features} bfloat16: written over
///   input_tensor    {..., in_features} bfloat16: {batch_size, seq_len, in_features} in GPT-2
///   weight_tensor   {out_features, in_features} bfloat16
///   bias_tensor     {out_features} bfloat16, or Tensor() for none (the head)
///   scratch_val     made for these sizes: its output_copies are written
///   output_epilogue_val   what is done to each pair of output as it is stored (wgmma_matmul_8bit.cuh's); not
///                         given: nothing
template <typename EPILOGUE>
inline void linear_8bit_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                const Tensor& bias_tensor, const Linear8bitScratch& scratch_val,
                                const EPILOGUE& output_epilogue_val) {
    // 1. check the tensors and the scratch; any of them wrong: stop with an error
    linear_bf16_check_forward("linear_8bit_forward", output_tensor, input_tensor, weight_tensor, bias_tensor);
    linear_8bit_check_sizes("linear_8bit_forward", input_tensor, weight_tensor, scratch_val);
    size_t out_features_val = weight_tensor.size(0);
    size_t rows_val = input_tensor.numel() / weight_tensor.size(1);

    // 2. the 8-bit copies: input {rows, in} and weight {out, in}
    const Linear8bitProductCopies& copies_val = scratch_val.output_copies;
    round_8bit(copies_val.left_8bit_tensor, copies_val.left_scales_tensor, input_tensor);
    round_8bit(copies_val.right_8bit_tensor, copies_val.right_scales_tensor, weight_tensor);

    // 3. output = input · weightᵀ + bias, output seen as {rows, out_features} as its left copy is
    wgmma_matmul_8bit(output_tensor.view({rows_val, out_features_val}), copies_val.left_8bit_tensor, copies_val.left_scales_tensor,
                      copies_val.right_8bit_tensor, copies_val.right_scales_tensor, bias_tensor, false, output_epilogue_val);
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of input, weight and bias with the two products in 8 bits: checks, rounds each
/// product's sides to E4M3 (weight, output_grad and input turned while rounding where they sit turned round), the
/// two products, then bias_grad in float (linear_bf16_bias_grad). input_grad is written over; weight_grad and
/// bias_grad are added onto, so they must be 0 before a step's first backward.
///
///   input_grad_tensor              input's shape, bfloat16
///   weight_grad_tensor             weight's shape, bfloat16
///   bias_grad_tensor               {out_features} bfloat16, or Tensor() for none
///   output_grad_tensor             the gradient from the layer above, bfloat16
///   input_tensor, weight_tensor    the forward's
///   scratch_val                    made for these sizes: its input_grad_copies and weight_grad_copies are written
///   input_grad_epilogue_val        what is done to each pair of input_grad as it is stored; not given: nothing
template <typename EPILOGUE>
inline void linear_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                 const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                 const Tensor& input_tensor, const Tensor& weight_tensor,
                                 const Linear8bitScratch& scratch_val, const EPILOGUE& input_grad_epilogue_val) {
    // 1. check the tensors and the scratch; any of them wrong: stop with an error
    linear_bf16_check_backward("linear_8bit_backward", input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                               output_grad_tensor, input_tensor, weight_tensor);
    linear_8bit_check_sizes("linear_8bit_backward", input_tensor, weight_tensor, scratch_val);
    size_t in_features_val = weight_tensor.size(1);
    size_t rows_val = input_tensor.numel() / in_features_val;

    // 2. input_grad = output_grad · weight: output_grad {rows, out} as it sits, weight {out, in} turned to {in, out}
    const Linear8bitProductCopies& input_grad_copies_val = scratch_val.input_grad_copies;
    round_8bit(input_grad_copies_val.left_8bit_tensor, input_grad_copies_val.left_scales_tensor, output_grad_tensor);
    round_8bit(input_grad_copies_val.right_8bit_tensor, input_grad_copies_val.right_scales_tensor, weight_tensor, true);
    wgmma_matmul_8bit(input_grad_tensor.view({rows_val, in_features_val}), input_grad_copies_val.left_8bit_tensor,
                      input_grad_copies_val.left_scales_tensor, input_grad_copies_val.right_8bit_tensor,
                      input_grad_copies_val.right_scales_tensor, Tensor(), false, input_grad_epilogue_val);

    // 3. weight_grad += output_gradᵀ · input: output_grad {rows, out} turned to {out, rows}, input {rows, in}
    //    turned to {in, rows}
    const Linear8bitProductCopies& weight_grad_copies_val = scratch_val.weight_grad_copies;
    round_8bit(weight_grad_copies_val.left_8bit_tensor, weight_grad_copies_val.left_scales_tensor, output_grad_tensor, true);
    round_8bit(weight_grad_copies_val.right_8bit_tensor, weight_grad_copies_val.right_scales_tensor, input_tensor, true);
    wgmma_matmul_8bit(weight_grad_tensor, weight_grad_copies_val.left_8bit_tensor, weight_grad_copies_val.left_scales_tensor,
                      weight_grad_copies_val.right_8bit_tensor, weight_grad_copies_val.right_scales_tensor, Tensor(), true);

    // 4. bias_grad += the columns' sums of output_grad, in float from the bfloat16 values (not rounded to 8 bits)
    if (bias_grad_tensor.data_ptr() != nullptr) linear_bf16_bias_grad(bias_grad_tensor, output_grad_tensor);
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless the layer's sizes fit 8 bits and the scratch was made for them: rows, in_features
/// and out_features multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE (each is one product's k), and each of the
/// scratch's copies the shape round_8bit writes for its side. The shapes of the tensors themselves are checked
/// by linear_bf16_check_forward or _check_backward first.
///
///   name_ptr                       the caller's name, for the message
///   input_tensor, weight_tensor    the layer's
///   scratch_val                    as in linear_8bit_forward
inline void linear_8bit_check_sizes(const char* name_ptr, const Tensor& input_tensor, const Tensor& weight_tensor,
                                    const Linear8bitScratch& scratch_val) {
    // 1. the sizes: each a product's k
    size_t in_features_val = weight_tensor.size(1);
    size_t out_features_val = weight_tensor.size(0);
    size_t rows_val = input_tensor.numel() / in_features_val;
    if (rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 || in_features_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 ||
        out_features_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0) {
        fprintf(stderr, "error: %s needs rows, in_features and out_features multiples of %d (each is the length one "
                        "8-bit product sums over); got %zu, %zu, %zu\n",
                name_ptr, ROUND_8BIT_ROW_LENGTH_MULTIPLE, rows_val, in_features_val, out_features_val);
        exit(EXIT_FAILURE);
    }
    // 2. the scratch: made for these sizes
    const Linear8bitProductCopies* copies_list[3] = {&scratch_val.output_copies, &scratch_val.input_grad_copies,
                                                      &scratch_val.weight_grad_copies};
    size_t shapes_list[3][3] = {{rows_val, out_features_val, in_features_val},    // rows, columns, k of each product
                                {rows_val, in_features_val, out_features_val},
                                {out_features_val, in_features_val, rows_val}};
    bool fits_val = true;
    for (int product_index_val = 0; product_index_val < 3; product_index_val++) {
        const Linear8bitProductCopies& copies_val = *copies_list[product_index_val];
        size_t product_rows_val = shapes_list[product_index_val][0];
        size_t product_columns_val = shapes_list[product_index_val][1];
        size_t k_ints_val = shapes_list[product_index_val][2] / ROUND_8BIT_VALUES_PER_INT;
        fits_val = fits_val && copies_val.left_8bit_tensor.shape() == Shape({product_rows_val, k_ints_val}) &&
                   copies_val.right_8bit_tensor.shape() == Shape({product_columns_val, k_ints_val}) &&
                   copies_val.left_scales_tensor.shape() == Shape({product_rows_val}) &&
                   copies_val.right_scales_tensor.shape() == Shape({product_columns_val});
    }
    if (!fits_val) {
        fprintf(stderr, "error: %s's scratch was not made for rows %zu, in_features %zu, out_features %zu: make it "
                        "with Linear8bitScratch(rows, in_features, out_features)\n",
                name_ptr, rows_val, in_features_val, out_features_val);
        exit(EXIT_FAILURE);
    }
}

#endif // LINEAR_8BIT_CUH
