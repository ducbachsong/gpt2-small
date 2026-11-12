// linear.cuh — a linear layer in float, as PyTorch's nn.Linear:
//     forward:    output = input · weightᵀ + bias
//     backward:   bias_grad   += sum over rows of output_grad      db = dyᵀ · 1
//                 weight_grad += output_gradᵀ · input              dW = dyᵀ x
//                 input_grad   = output_grad · weight              dx = dy W
//
//     input    {batch_size, seq_len, in_features}
//     weight   {out_features, in_features}
//     bias     {out_features}
//     output   {batch_size, seq_len, out_features}
// The three products are the matmul tools' output = left · rightᵀ, with the tensors as they sit (turned: the
// tensor holds its matrix turned round):
//     product      left                     right                  bias   added onto
//     output       input                    weight                 bias   no
//     input_grad   output_grad              weight, turned         -      no
//     weight_grad  output_grad, turned      input, turned          -      yes
// All three by fma_matmul_bf16.cuh in float on the CUDA cores: the exact reference, any GPU. bias_grad:
// linear_bias_grad_kernel below, a sum down each column of output_grad. All float; the sums in float. The same
// layer in bfloat16 on an H100's tensor cores: linear_bf16.cuh; with its products in 8 bits: linear_8bit.cuh.
// Math: https://ducbachsong.github.io/posts/linear-backward
#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"
#include "../bf16/fma_matmul_bf16.cuh"       // the products, in float

// bias_grad: a block takes HARDWARE_THREADS_PER_WARP neighbouring columns of output_grad, one per lane, and its
// warps share the rows; then warp 0 adds up the warps' sums in order. Fixed, so the sum's order (and its bits)
// never changes.
#define LINEAR_BIAS_GRAD_WARPS 8                                                         // warps of a block: each takes every 8th row
#define LINEAR_BIAS_GRAD_THREADS_PER_BLOCK (LINEAR_BIAS_GRAD_WARPS * HARDWARE_THREADS_PER_WARP)

// ── forward and backward ──────────────────────────────────────────────────────
void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                    const Tensor& bias_tensor);
void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes output = input · weightᵀ + bias in float: checks the shapes and types, then the product
/// (fma_matmul_bf16), which checks the rest.
///
///   output_tensor   {..., out_features}
///   input_tensor    {..., in_features}: {batch_size, seq_len, in_features} in GPT-2
///   weight_tensor   {out_features, in_features}
///   bias_tensor     {out_features}, or Tensor() for none (the head)
inline void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                           const Tensor& bias_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == weight_tensor.size(0) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer needs input {..., in_features}, output {..., out_features} with the "
                        "same rows, weight {out_features, in_features}, bias {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // types: all float (bfloat16: linear_bf16.cuh)
    if (input_tensor.dtype() != DType::Float32 || output_tensor.dtype() != DType::Float32 ||
        weight_tensor.dtype() != DType::Float32 || (has_bias_val && bias_tensor.dtype() != DType::Float32)) {
        fprintf(stderr, "error: linear_forward needs output, input, weight and bias all float; for bfloat16, "
                        "linear_bf16.cuh\n");
        exit(EXIT_FAILURE);
    }

    // 2. output = input · weightᵀ + bias
    fma_matmul_bf16(output_tensor, input_tensor, weight_tensor, bias_tensor);
}

// ── backward ──────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(LINEAR_BIAS_GRAD_THREADS_PER_BLOCK)
linear_bias_grad_kernel(float* bias_grad_ptr, const float* output_grad_ptr, size_t rows_val, size_t out_features_val);

/// Works out the gradients of input, weight and bias in float: checks the shapes and types, then the two products
/// (each checks the rest) and linear_bias_grad_kernel. input_grad is written over; weight_grad and bias_grad are
/// added onto, so they must be 0 before a step's first backward.
///
///   input_grad_tensor              input's shape
///   weight_grad_tensor             weight's shape
///   bias_grad_tensor               {out_features}, or Tensor() for none
///   output_grad_tensor             the gradient from the layer above
///   input_tensor, weight_tensor    the forward's
inline void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                            const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight, each grad its tensor's shape
    bool has_bias_val = bias_grad_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_grad_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_grad_tensor.size(output_grad_tensor.dim() - 1) == weight_tensor.size(0) &&
                    input_grad_tensor.shape() == input_tensor.shape() && weight_grad_tensor.shape() == weight_tensor.shape() &&
                    (!has_bias_val || bias_grad_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_grad_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer's backward needs input and input_grad {..., in_features}, output_grad "
                        "{..., out_features} with the same rows, weight and weight_grad {out_features, in_features}, "
                        "bias_grad {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // types: all float (bfloat16: linear_bf16.cuh)
    if (input_tensor.dtype() != DType::Float32 || input_grad_tensor.dtype() != DType::Float32 ||
        weight_grad_tensor.dtype() != DType::Float32 || output_grad_tensor.dtype() != DType::Float32 ||
        weight_tensor.dtype() != DType::Float32 || (has_bias_val && bias_grad_tensor.dtype() != DType::Float32)) {
        fprintf(stderr, "error: linear_backward needs its tensors all float; for bfloat16, linear_bf16.cuh\n");
        exit(EXIT_FAILURE);
    }

    // 2. input_grad = output_grad · weight (weight {out, in} is that product's right turned round);
    //    weight_grad += output_gradᵀ · input (both sit turned round: {rows, out} and {rows, in})
    fma_matmul_bf16(input_grad_tensor, output_grad_tensor, weight_tensor, Tensor(), false, false, true);
    fma_matmul_bf16(weight_grad_tensor, output_grad_tensor, input_tensor, Tensor(), true, true, true);

    // 3. bias_grad += the columns' sums of output_grad: one block per HARDWARE_THREADS_PER_WARP columns
    if (!has_bias_val) return;
    size_t out_features_val = weight_tensor.size(0);
    size_t rows_val = output_grad_tensor.numel() / out_features_val;
    unsigned blocks_val = (unsigned)CEIL_DIV(out_features_val, (size_t)HARDWARE_THREADS_PER_WARP);
    linear_bias_grad_kernel<<<blocks_val, LINEAR_BIAS_GRAD_THREADS_PER_BLOCK>>>(bias_grad_tensor.data(), output_grad_tensor.data(),
                                                                                rows_val, out_features_val);
    cudaCheck(cudaGetLastError());
}

/// Adds output_grad's columns' sums onto bias_grad (db = dyᵀ · 1). A block takes HARDWARE_THREADS_PER_WARP
/// neighbouring columns, one per lane (a warp reads neighbouring values of a row at once); warp w adds up rows
/// w, w + LINEAR_BIAS_GRAD_WARPS, ...; then warp 0 adds the warps' sums in order and adds that onto bias_grad. No
/// atomics: the same result every run.
///
///   bias_grad_ptr                {out_features}: added onto
///   output_grad_ptr              {rows, out_features}
///   rows_val, out_features_val   output_grad's shape
__global__ void __launch_bounds__(LINEAR_BIAS_GRAD_THREADS_PER_BLOCK)
linear_bias_grad_kernel(float* bias_grad_ptr, const float* output_grad_ptr, size_t rows_val, size_t out_features_val) {
    __shared__ float warp_sums_list[LINEAR_BIAS_GRAD_WARPS][HARDWARE_THREADS_PER_WARP];
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    size_t column_index_val = (size_t)blockIdx.x * HARDWARE_THREADS_PER_WARP + lane_index_val;
    // 1. this warp's rows of the column
    float sum_val = 0.0f;
    if (column_index_val < out_features_val)
        for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += LINEAR_BIAS_GRAD_WARPS)
            sum_val += output_grad_ptr[row_index_val * out_features_val + column_index_val];
    warp_sums_list[warp_index_val][lane_index_val] = sum_val;
    __syncthreads();
    // 2. the warps' sums in order, onto bias_grad
    if (warp_index_val == 0 && column_index_val < out_features_val) {
        float total_val = 0.0f;
        for (int other_warp_index_val = 0; other_warp_index_val < LINEAR_BIAS_GRAD_WARPS; other_warp_index_val++)
            total_val += warp_sums_list[other_warp_index_val][lane_index_val];
        bias_grad_ptr[column_index_val] += total_val;
    }
}

#endif // LINEAR_CUH
