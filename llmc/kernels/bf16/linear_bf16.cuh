// linear_bf16.cuh — a linear layer in bfloat16 on an H100's tensor cores, as PyTorch's nn.Linear:
//     forward:    output = input · weightᵀ + bias
//     backward:   bias_grad   += sum over rows of output_grad      db = dyᵀ · 1
//                 weight_grad += output_gradᵀ · input              dW = dyᵀ x
//                 input_grad   = output_grad · weight              dx = dy W
//
//     tensor                     type in memory   shape
//     input, input_grad          bfloat16         {batch_size, seq_len, in_features}
//     weight, weight_grad        bfloat16         {out_features, in_features}
//     bias, bias_grad            bfloat16         {out_features}
//     output, output_grad        bfloat16         {batch_size, seq_len, out_features}
//     sums                       float            in registers
//
// The three products are wgmma_matmul_bf16's output = left · rightᵀ (WGMMA m64n64k16, the tiles brought by TMA),
// with the tensors as they sit (turned: the tensor holds its matrix turned round, and the WGMMA reads that tile
// with its transpose bit):
//     product      left                     right                  bias   added onto
//     output       input                    weight                 bias   no
//     input_grad   output_grad              weight, turned         -      no
//     weight_grad  output_grad, turned      input, turned          -      yes
// Each product can be given its own setting (wgmma_matmul_bf16.cuh's WgmmaMatmulBf16Setting: the square, the sets of
// tiles, the splits of the sum, the block order); not given: the defaults. model_bf16.cuh gives each GPT-2 product
// the one tune_wgmma_matmul_bf16 found fastest.
// An epilogue (wgmma_matmul_bf16.cuh's) can be given for output in the forward and for input_grad in the backward:
// bf16/gelu_bf16.cuh passes GELU's, so the GELU after a linear layer is done as its output is stored.
// bias_grad: linear_bf16_bias_grad_kernel below, a sum down each column of output_grad in float. Needs an H100
// (WGMMA and TMA exist only on Hopper), built with -arch=sm_90a, and every tensor's last dim a multiple of
// WGMMA_MATMUL_BF16_VALUE_MULTIPLE (wgmma_matmul_bf16 checks). The same layer in float: linear.cuh; with its
// products in 8 bits: linear_8bit.cuh.
// Math: https://ducbachsong.github.io/posts/linear-backward
#ifndef LINEAR_BF16_CUH
#define LINEAR_BF16_CUH

#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"
#include "wgmma_matmul_bf16.cuh"     // the products

// bias_grad: a block takes LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS neighbouring columns of output_grad and one chunk of
// its rows, its warps sharing the chunk's rows; then warp 0 adds up the warps' sums in order into the chunk's row of
// a scratch, and a second kernel adds the chunks in order onto bias_grad. Fixed for a launch, so the sum's order
// (and its bits) never changes between runs. The chunks spread a column sum over many blocks: one block per 32
// columns (as before) is 24 blocks for 768 columns, on 132 SMs.
#define LINEAR_BF16_BIAS_GRAD_WARPS 8                                                              // warps of a block: each takes every 8th row of the chunk
#define LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK (LINEAR_BF16_BIAS_GRAD_WARPS * HARDWARE_THREADS_PER_WARP)
#define LINEAR_BF16_BIAS_GRAD_COLUMNS_PER_LANE 2                                                   // neighbouring columns a lane reads: one 4-byte bfloat162
#define LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS (LINEAR_BF16_BIAS_GRAD_COLUMNS_PER_LANE * HARDWARE_THREADS_PER_WARP)   // a block's columns
#define LINEAR_BF16_BIAS_GRAD_ROW_CHUNKS 64                                                        // the default chunks of rows (tune_linear_bf16 sweeps it)

// ── forward and backward ──────────────────────────────────────────────────────
template <typename SETTING = WgmmaMatmulBf16Setting<>, typename EPILOGUE = WgmmaMatmulBf16Store>
void linear_bf16_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                         const Tensor& bias_tensor, const EPILOGUE& output_epilogue_val = EPILOGUE());
template <typename INPUT_GRAD_SETTING = WgmmaMatmulBf16Setting<>, typename WEIGHT_GRAD_SETTING = WgmmaMatmulBf16Setting<>,
          typename EPILOGUE = WgmmaMatmulBf16Store>
void linear_bf16_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                          const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                          const EPILOGUE& input_grad_epilogue_val = EPILOGUE());

// ── for both, and for linear_8bit.cuh ─────────────────────────────────────────
void linear_bf16_check_forward(const char* name_ptr, const Tensor& output_tensor, const Tensor& input_tensor,
                               const Tensor& weight_tensor, const Tensor& bias_tensor);
void linear_bf16_check_backward(const char* name_ptr, const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                const Tensor& input_tensor, const Tensor& weight_tensor);
void linear_bf16_bias_grad(const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                          int row_chunks_val = LINEAR_BF16_BIAS_GRAD_ROW_CHUNKS);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes output = input · weightᵀ + bias in bfloat16: checks the shapes and types, then the product
/// (wgmma_matmul_bf16), which checks the rest.
///
///   output_tensor   {..., out_features}
///   input_tensor    {..., in_features}: {batch_size, seq_len, in_features} in GPT-2
///   weight_tensor   {out_features, in_features}
///   bias_tensor     {out_features}, or Tensor() for none (the head)
///   output_epilogue_val   what is done to each pair of output as it is stored (wgmma_matmul_bf16.cuh's); not
///                         given: nothing
/// SETTING: the product's WgmmaMatmulBf16Setting; not given: the defaults.
template <typename SETTING, typename EPILOGUE>
inline void linear_bf16_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                const Tensor& bias_tensor, const EPILOGUE& output_epilogue_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    linear_bf16_check_forward("linear_bf16_forward", output_tensor, input_tensor, weight_tensor, bias_tensor);

    // 2. output = input · weightᵀ + bias
    wgmma_matmul_bf16<SETTING>(output_tensor, input_tensor, weight_tensor, bias_tensor, false, false, false, output_epilogue_val);
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of input, weight and bias in bfloat16: checks the shapes and types, then the two
/// products (each checks the rest) and bias_grad. input_grad is written over; weight_grad and bias_grad are added
/// onto, so they must be 0 before a step's first backward.
///
///   input_grad_tensor              input's shape
///   weight_grad_tensor             weight's shape
///   bias_grad_tensor               {out_features}, or Tensor() for none
///   output_grad_tensor             the gradient from the layer above
///   input_tensor, weight_tensor    the forward's
///   input_grad_epilogue_val        what is done to each pair of input_grad as it is stored; not given: nothing
/// INPUT_GRAD_SETTING, WEIGHT_GRAD_SETTING: the two products' WgmmaMatmulBf16Setting; not given: the defaults.
template <typename INPUT_GRAD_SETTING, typename WEIGHT_GRAD_SETTING, typename EPILOGUE>
inline void linear_bf16_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                 const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                 const Tensor& input_tensor, const Tensor& weight_tensor,
                                 const EPILOGUE& input_grad_epilogue_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    linear_bf16_check_backward("linear_bf16_backward", input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                               output_grad_tensor, input_tensor, weight_tensor);

    // 2. input_grad = output_grad · weight (weight {out, in} is that product's right turned round);
    //    weight_grad += output_gradᵀ · input (both sit turned round: {rows, out} and {rows, in})
    wgmma_matmul_bf16<INPUT_GRAD_SETTING>(input_grad_tensor, output_grad_tensor, weight_tensor, Tensor(), false, false, true,
                                          input_grad_epilogue_val);
    wgmma_matmul_bf16<WEIGHT_GRAD_SETTING>(weight_grad_tensor, output_grad_tensor, input_tensor, Tensor(), true, true, true);

    // 3. bias_grad += the columns' sums of output_grad
    if (bias_grad_tensor.data_ptr() != nullptr) linear_bf16_bias_grad(bias_grad_tensor, output_grad_tensor);
}

// ── for both, and for linear_8bit.cuh ─────────────────────────────────────────

/// Stops with a message unless a forward's tensors fit: input {..., in_features}, output {..., out_features} with
/// the same rows, weight {out_features, in_features}, bias {out_features} or none; all bfloat16.
///
///   name_ptr                                               the caller's name, for the message
///   output_tensor, input_tensor, weight_tensor, bias_tensor   as in linear_bf16_forward
inline void linear_bf16_check_forward(const char* name_ptr, const Tensor& output_tensor, const Tensor& input_tensor,
                                      const Tensor& weight_tensor, const Tensor& bias_tensor) {
    // 1. shapes: the same rows, the features matching weight
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == weight_tensor.size(0) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: %s needs input {..., in_features}, output {..., out_features} with the same rows, "
                        "weight {out_features, in_features}, bias {out_features} or Tensor()\n", name_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16 (float: linear.cuh)
    if (input_tensor.dtype() != DType::BFloat16 || output_tensor.dtype() != DType::BFloat16 ||
        weight_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: %s needs output, input, weight and bias all bfloat16; for float, linear.cuh\n", name_ptr);
        exit(EXIT_FAILURE);
    }
}

/// Stops with a message unless a backward's tensors fit: as linear_bf16_check_forward, and each grad its tensor's
/// shape; all bfloat16.
///
///   name_ptr     the caller's name, for the message
///   the others   as in linear_bf16_backward
inline void linear_bf16_check_backward(const char* name_ptr, const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                       const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                       const Tensor& input_tensor, const Tensor& weight_tensor) {
    // 1. shapes: the same rows, the features matching weight, each grad its tensor's shape
    bool has_bias_val = bias_grad_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_grad_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_grad_tensor.size(output_grad_tensor.dim() - 1) == weight_tensor.size(0) &&
                    input_grad_tensor.shape() == input_tensor.shape() && weight_grad_tensor.shape() == weight_tensor.shape() &&
                    (!has_bias_val || bias_grad_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_grad_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: %s needs input and input_grad {..., in_features}, output_grad {..., out_features} with "
                        "the same rows, weight and weight_grad {out_features, in_features}, bias_grad {out_features} "
                        "or Tensor()\n", name_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16 (float: linear.cuh)
    if (input_tensor.dtype() != DType::BFloat16 || input_grad_tensor.dtype() != DType::BFloat16 ||
        weight_grad_tensor.dtype() != DType::BFloat16 || output_grad_tensor.dtype() != DType::BFloat16 ||
        weight_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_grad_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: %s needs its tensors all bfloat16; for float, linear.cuh\n", name_ptr);
        exit(EXIT_FAILURE);
    }
}

__global__ void __launch_bounds__(LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK)
linear_bf16_bias_grad_kernel(float* chunk_sums_ptr, const nv_bfloat16* output_grad_ptr, size_t rows_val,
                             size_t out_features_val, size_t rows_per_chunk_val);
__global__ void linear_bf16_bias_grad_sums_kernel(nv_bfloat16* bias_grad_ptr, const float* chunk_sums_ptr, int row_chunks_val,
                                                  size_t out_features_val);

/// Adds output_grad's columns' sums onto bias_grad: launches linear_bf16_bias_grad_kernel, one block per
/// LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS columns and chunk of rows (row_chunks_val chunks), each block's sums into its
/// chunk's row of a scratch; then linear_bf16_bias_grad_sums_kernel adds the chunks in order onto bias_grad. The
/// block size and the chunks are fixed for a launch, so the order of the sum, and its bits, never change between
/// runs. The scratch, row_chunks x out_features floats, is made the first time and grows with the largest launch.
/// The callers have checked the shapes (out_features even: a lane reads two columns).
///
///   bias_grad_tensor     {out_features} bfloat16: added onto
///   output_grad_tensor   {..., out_features} bfloat16
///   row_chunks_val       chunks the rows are cut into, one block row each: at least 1; not given:
///                        LINEAR_BF16_BIAS_GRAD_ROW_CHUNKS
inline void linear_bf16_bias_grad(const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor, int row_chunks_val) {
    // 1. the sizes and the chunks; the scratch
    if (row_chunks_val < 1) {
        fprintf(stderr, "error: linear_bf16_bias_grad's row_chunks must be 1 or more; got %d\n", row_chunks_val);
        exit(EXIT_FAILURE);
    }
    size_t out_features_val = bias_grad_tensor.numel();
    size_t rows_val = output_grad_tensor.numel() / out_features_val;
    size_t rows_per_chunk_val = CEIL_DIV(rows_val, (size_t)row_chunks_val);
    static Tensor chunk_sums_tensor;
    if ((size_t)chunk_sums_tensor.numel() < (size_t)row_chunks_val * out_features_val)
        chunk_sums_tensor = Tensor::zeros({(size_t)row_chunks_val * out_features_val});
    // 2. each chunk's sums: columns across, chunks down
    dim3 grid_val((unsigned)CEIL_DIV(out_features_val, (size_t)LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS), (unsigned)row_chunks_val);
    linear_bf16_bias_grad_kernel<<<grid_val, LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK>>>(
        chunk_sums_tensor.data(), output_grad_tensor.data_ptr<nv_bfloat16>(), rows_val, out_features_val, rows_per_chunk_val);
    cudaCheck(cudaGetLastError());
    // 3. the chunks in order, onto bias_grad: one thread per column
    linear_bf16_bias_grad_sums_kernel<<<(unsigned)CEIL_DIV(out_features_val, (size_t)LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK),
                                        LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK>>>(
        bias_grad_tensor.data_ptr<nv_bfloat16>(), chunk_sums_tensor.data(), row_chunks_val, out_features_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one chunk's sums of LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS columns of output_grad (db = dyᵀ · 1), in
/// float: a lane takes LINEAR_BF16_BIAS_GRAD_COLUMNS_PER_LANE neighbouring columns (one 4-byte read; a warp reads
/// 128 bytes of a row at once), warp w the chunk's rows w, w + LINEAR_BF16_BIAS_GRAD_WARPS, ...; then warp 0 adds
/// the warps' sums in order into the chunk's row of chunk_sums. No atomics.
///
///   chunk_sums_ptr               {row_chunks, out_features}: this block's chunk and columns written
///   output_grad_ptr              {rows, out_features}
///   rows_val, out_features_val   output_grad's shape
///   rows_per_chunk_val           rows of a chunk (the last one may have fewer)
__global__ void __launch_bounds__(LINEAR_BF16_BIAS_GRAD_THREADS_PER_BLOCK)
linear_bf16_bias_grad_kernel(float* chunk_sums_ptr, const nv_bfloat16* output_grad_ptr, size_t rows_val,
                             size_t out_features_val, size_t rows_per_chunk_val) {
    __shared__ float2 warp_sums_list[LINEAR_BF16_BIAS_GRAD_WARPS][HARDWARE_THREADS_PER_WARP];
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    size_t column_index_val = (size_t)blockIdx.x * LINEAR_BF16_BIAS_GRAD_BLOCK_COLUMNS + LINEAR_BF16_BIAS_GRAD_COLUMNS_PER_LANE * lane_index_val;
    size_t first_row_index_val = (size_t)blockIdx.y * rows_per_chunk_val;
    size_t end_row_index_val = first_row_index_val + rows_per_chunk_val < rows_val ? first_row_index_val + rows_per_chunk_val : rows_val;
    // 1. this warp's rows of the chunk, two neighbouring columns at a time
    float2 sum_val = make_float2(0.0f, 0.0f);
    if (column_index_val < out_features_val)
        for (size_t row_index_val = first_row_index_val + warp_index_val; row_index_val < end_row_index_val;
             row_index_val += LINEAR_BF16_BIAS_GRAD_WARPS) {
            float2 pair_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(output_grad_ptr + row_index_val * out_features_val + column_index_val));
            sum_val.x += pair_val.x;
            sum_val.y += pair_val.y;
        }
    warp_sums_list[warp_index_val][lane_index_val] = sum_val;
    __syncthreads();
    // 2. the warps' sums in order, into the chunk's row
    if (warp_index_val == 0 && column_index_val < out_features_val) {
        float2 total_val = make_float2(0.0f, 0.0f);
        for (int other_warp_index_val = 0; other_warp_index_val < LINEAR_BF16_BIAS_GRAD_WARPS; other_warp_index_val++) {
            total_val.x += warp_sums_list[other_warp_index_val][lane_index_val].x;
            total_val.y += warp_sums_list[other_warp_index_val][lane_index_val].y;
        }
        float* row_ptr = chunk_sums_ptr + (size_t)blockIdx.y * out_features_val + column_index_val;
        row_ptr[0] = total_val.x;
        row_ptr[1] = total_val.y;
    }
}

/// Adds the chunks' sums of each column up, in order, in float, then onto bias_grad, rounded to bfloat16 once: one
/// thread per column.
///
///   bias_grad_ptr      {out_features}: added onto
///   chunk_sums_ptr     {row_chunks, out_features}
///   row_chunks_val     the chunks
///   out_features_val   the columns
__global__ void linear_bf16_bias_grad_sums_kernel(nv_bfloat16* bias_grad_ptr, const float* chunk_sums_ptr, int row_chunks_val,
                                                  size_t out_features_val) {
    size_t column_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (column_index_val >= out_features_val) return;
    float sum_val = 0.0f;
    for (int chunk_index_val = 0; chunk_index_val < row_chunks_val; chunk_index_val++)
        sum_val += chunk_sums_ptr[(size_t)chunk_index_val * out_features_val + column_index_val];
    bias_grad_ptr[column_index_val] = __float2bfloat16(__bfloat162float(bias_grad_ptr[column_index_val]) + sum_val);
}

#endif // LINEAR_BF16_CUH
