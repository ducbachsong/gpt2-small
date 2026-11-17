// fma_matmul_8bit.cuh — matrix products on 8-bit (FP8 E4M3) copies, on the CUDA cores: fma_matmul_bf16.cuh's
// product with left and right rounded to E4M3 first (round_8bit.cuh), one float scale per row:
//     output[i][j] = left_scale[i] · right_scale[j] · sum over k of left_8bit[i][k] · right_8bit[j][k]
//                    (+ bias[j]) (+ output[i][j])
// Each E4M3 is turned into a float as it is copied into shared memory, so the math is fma_matmul_bf16's, in float;
// both scales are applied once, as each sum is stored. Any GPU. What 8 bits save here is GPU memory reads (one byte
// a value), not math: the CUDA cores have no faster E4M3 multiply.
//
//     tensor                    type in memory                   shape                               read with
//     left_8bit, right_8bit     E4M3                             {rows, sum_length},                  4 E4M3s (4 bytes) at a time
//                                                                {columns, sum_length}
//     left_scales, right_scales float                            {rows}, {columns}                    as the sums are stored
//     bias                      bfloat16                         {columns}, or none                   the same
//     output                    bfloat16                         {rows, columns}                      written over, or added onto
//
// Both copies sit k next to each other (K-major), as round_8bit writes them; a matrix that sits turned round (a
// linear layer's backward) is turned while rounding (round_8bit's turned). The squares, steps and outer products
// are fma_matmul_bf16.cuh's; the block size is fixed by the square, so this takes none.
#ifndef FMA_MATMUL_8BIT_CUH
#define FMA_MATMUL_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"
#include "../../common/tensor.cuh"
#include "../bf16/fma_matmul_bf16.cuh"       // the squares, the pieces, fma_matmul_bf16_add_outer_products
#include "round_8bit.cuh"            // ROUND_8BIT_ROW_LENGTH_MULTIPLE

// ── the product ───────────────────────────────────────────────────────────────
void fma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                     const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor = Tensor(),
                     bool add_onto_val = false);

// ── for it ────────────────────────────────────────────────────────────────────
void fma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                                  const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(FMA_MATMUL_BF16_THREADS_PER_BLOCK, 2)
fma_matmul_8bit_kernel(nv_bfloat16* output_ptr, const __nv_fp8_e4m3* left_ptr, const float* left_scales_ptr,
                       const __nv_fp8_e4m3* right_ptr, const float* right_scales_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                       size_t columns_val, size_t sum_length_val, bool add_onto_val);
__device__ __forceinline__ float4 fma_matmul_8bit_load_piece(const __nv_fp8_e4m3* matrix_ptr, size_t lines_val, size_t sum_length_val,
                                                             size_t block_first_line_index_val, size_t step_first_k_index_val);

/// Writes output = left_scale · right_scale · (left_8bit · right_8bitᵀ) (+ bias) (+ output): checks, then launches
/// fma_matmul_8bit_kernel, one block per square of output, as one line of blocks.
///
///   output_tensor                           {..., columns} bfloat16: written over, or added onto
///   left_8bit_tensor, left_scales_tensor    round_8bit's of left {..., sum_length}: x in a linear layer
///   right_8bit_tensor, right_scales_tensor  round_8bit's of right {columns, sum_length}: W
///   bias_tensor                             {columns} bfloat16, or Tensor() for none
///   add_onto_val                            true: added onto what output holds; false: written over
/// sum_length must be a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE (16) and columns of VALUES_PER_FLOAT4.
inline void fma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                            const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor,
                            bool add_onto_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    fma_matmul_8bit_check_shapes(output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor,
                                 bias_tensor);
    // 2. launch: squares across and down, rounded up, as one line of blocks
    size_t sum_length_val = right_8bit_tensor.size(1);
    size_t rows_val = left_8bit_tensor.numel() / sum_length_val;
    size_t columns_val = right_8bit_tensor.size(0);
    unsigned blocks_val = (unsigned)(CEIL_DIV(rows_val, (size_t)FMA_MATMUL_BF16_BLOCK_ROWS) *
                                     CEIL_DIV(columns_val, (size_t)FMA_MATMUL_BF16_BLOCK_COLUMNS));
    fma_matmul_8bit_kernel<<<blocks_val, FMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
        output_tensor.data_ptr<nv_bfloat16>(), reinterpret_cast<const __nv_fp8_e4m3*>(left_8bit_tensor.data_ptr()),
        left_scales_tensor.data(), reinterpret_cast<const __nv_fp8_e4m3*>(right_8bit_tensor.data_ptr()), right_scales_tensor.data(),
        bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr, rows_val, columns_val, sum_length_val,
        add_onto_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output: fma_matmul_bf16_kernel's walk, the E4M3s turned into floats as they are copied;
/// at the end each sum is times its row's and column's scales.
///
///   output_ptr, left_ptr, left_scales_ptr, right_ptr, right_scales_ptr, bias_ptr   as in fma_matmul_8bit; bias nullptr for none
///   rows_val, columns_val, sum_length_val                                           the product's sizes
///   add_onto_val                                                                    as in fma_matmul_8bit
__global__ void __launch_bounds__(FMA_MATMUL_BF16_THREADS_PER_BLOCK, 2)
fma_matmul_8bit_kernel(nv_bfloat16* output_ptr, const __nv_fp8_e4m3* left_ptr, const float* left_scales_ptr,
                       const __nv_fp8_e4m3* right_ptr, const float* right_scales_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                       size_t columns_val, size_t sum_length_val, bool add_onto_val) {
    // this step's values of k as floats, as the rows: [set][k][r] = left[r][k], [set][k][c] = right[c][k]
    __shared__ __align__(16) float left_pieces_list[2][FMA_MATMUL_BF16_STEP_K][FMA_MATMUL_BF16_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float right_pieces_list[2][FMA_MATMUL_BF16_STEP_K][FMA_MATMUL_BF16_PIECE_ROW_LENGTH];
    // 1. the block's square and the thread's part of it
    size_t block_first_row_index_val = 0, block_first_column_index_val = 0;
    fma_matmul_bf16_block_square(rows_val, columns_val, &block_first_row_index_val, &block_first_column_index_val);
    int thread_first_row_index_val = (threadIdx.x / FMA_MATMUL_BF16_THREADS_ACROSS) * FMA_MATMUL_BF16_THREAD_ROWS;
    int thread_first_column_index_val = (threadIdx.x % FMA_MATMUL_BF16_THREADS_ACROSS) * (FMA_MATMUL_BF16_THREAD_COLUMNS / 2);

    // 2. walk the sum: copy each step's pieces (as floats), read the next step's, add the outer products
    float sums_list[FMA_MATMUL_BF16_THREAD_ROWS][FMA_MATMUL_BF16_THREAD_COLUMNS] = {};
    size_t total_steps_val = CEIL_DIV(sum_length_val, (size_t)FMA_MATMUL_BF16_STEP_K);
    float4 left_val = fma_matmul_8bit_load_piece(left_ptr, rows_val, sum_length_val, block_first_row_index_val, 0);
    float4 right_val = fma_matmul_8bit_load_piece(right_ptr, columns_val, sum_length_val, block_first_column_index_val, 0);
    for (size_t step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        int piece_set_index_val = (int)(step_index_val % 2);
        fma_matmul_bf16_store_piece<false>(left_pieces_list[piece_set_index_val], left_val);
        fma_matmul_bf16_store_piece<false>(right_pieces_list[piece_set_index_val], right_val);
        __syncthreads();
        size_t next_step_first_k_index_val = (step_index_val + 1) * FMA_MATMUL_BF16_STEP_K;
        if (next_step_first_k_index_val < sum_length_val) {
            left_val = fma_matmul_8bit_load_piece(left_ptr, rows_val, sum_length_val, block_first_row_index_val, next_step_first_k_index_val);
            right_val = fma_matmul_8bit_load_piece(right_ptr, columns_val, sum_length_val, block_first_column_index_val,
                                                   next_step_first_k_index_val);
        }
        fma_matmul_bf16_add_outer_products(sums_list, left_pieces_list[piece_set_index_val], right_pieces_list[piece_set_index_val],
                                           thread_first_row_index_val, thread_first_column_index_val);
    }

    // 3. output = left_scale · right_scale · sums (+ bias) (+ output), one float4 at a time; nothing past the edges
#pragma unroll
    for (int r_index_val = 0; r_index_val < FMA_MATMUL_BF16_THREAD_ROWS; r_index_val++) {
        size_t row_index_val = block_first_row_index_val + thread_first_row_index_val + r_index_val;
        if (row_index_val >= rows_val) continue;
        float left_scale_val = left_scales_ptr[row_index_val];
#pragma unroll
        for (int c_index_val = 0; c_index_val < FMA_MATMUL_BF16_THREAD_COLUMNS; c_index_val += VALUES_PER_FLOAT4) {
            int half_index_val = c_index_val / VALUES_PER_FLOAT4;
            size_t column_index_val = block_first_column_index_val + thread_first_column_index_val + half_index_val * FMA_MATMUL_BF16_SECOND_HALF_OFFSET;
            if (column_index_val >= columns_val) continue;
            float4 right_scales_val = *reinterpret_cast<const float4*>(right_scales_ptr + column_index_val);
            float4 bias_val = bias_ptr != nullptr ? read4_as_float(bias_ptr + column_index_val) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 result_val = make_float4(sums_list[r_index_val][c_index_val] * left_scale_val * right_scales_val.x + bias_val.x,
                                            sums_list[r_index_val][c_index_val + 1] * left_scale_val * right_scales_val.y + bias_val.y,
                                            sums_list[r_index_val][c_index_val + 2] * left_scale_val * right_scales_val.z + bias_val.z,
                                            sums_list[r_index_val][c_index_val + 3] * left_scale_val * right_scales_val.w + bias_val.w);
            nv_bfloat16* output_float4_ptr = output_ptr + row_index_val * columns_val + column_index_val;
            if (add_onto_val) {
                float4 old_val = read4_as_float(output_float4_ptr);
                result_val = make_float4(old_val.x + result_val.x, old_val.y + result_val.y, old_val.z + result_val.z,
                                         old_val.w + result_val.w);
            }
            write4_from_float(output_float4_ptr, result_val);
        }
    }
}

/// Gives this thread's float4 of one step's piece of left_8bit or right_8bit: 4 neighbouring k of one line (one int
/// of 4 E4M3s, 2 threads a line), turned into floats; 0 past the matrix's edges. fma_matmul_bf16_store_piece<false>
/// then writes it turned round.
///
///   matrix_ptr                     left_8bit or right_8bit: {lines, sum_length} E4M3
///   lines_val, sum_length_val      its sizes
///   block_first_line_index_val     the block's first line
///   step_first_k_index_val         the step's first k
__device__ __forceinline__ float4 fma_matmul_8bit_load_piece(const __nv_fp8_e4m3* matrix_ptr, size_t lines_val, size_t sum_length_val,
                                                             size_t block_first_line_index_val, size_t step_first_k_index_val) {
    const int threads_per_line_val = FMA_MATMUL_BF16_STEP_K / VALUES_PER_FLOAT4;
    size_t line_index_val = block_first_line_index_val + threadIdx.x / threads_per_line_val;
    size_t k_index_val = step_first_k_index_val + (threadIdx.x % threads_per_line_val) * VALUES_PER_FLOAT4;
    if (line_index_val >= lines_val || k_index_val >= sum_length_val) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return static_cast<float4>(*reinterpret_cast<const __nv_fp8x4_e4m3*>(matrix_ptr + line_index_val * sum_length_val + k_index_val));
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Stops with a message unless the product's tensors fit: the shapes at the top of the file, the types, sum_length
/// a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE and columns of VALUES_PER_FLOAT4, each tensor starting on 16 bytes.
///
///   output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor, bias_tensor
///       as in fma_matmul_8bit
inline void fma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                                         const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor) {
    // 1. shapes: the same rows, the sums' lengths and columns matching right, a scale per row of each
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = left_8bit_tensor.dim() >= 1 && right_8bit_tensor.dim() == 2 && output_tensor.dim() == left_8bit_tensor.dim() &&
                    left_scales_tensor.dim() == left_8bit_tensor.dim() - 1 &&
                    left_8bit_tensor.size(left_8bit_tensor.dim() - 1) == right_8bit_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == right_8bit_tensor.size(0) &&
                    right_scales_tensor.shape() == Shape({right_8bit_tensor.size(0)}) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({right_8bit_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < left_8bit_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == left_8bit_tensor.size(dim_val) && left_scales_tensor.size(dim_val) == left_8bit_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: fma_matmul_8bit needs left_8bit {..., sum_length} with left_scales {...}, output {..., "
                        "columns} with the same rows, right_8bit {columns, sum_length} with right_scales {columns}, "
                        "bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: the 8-bit copies E4M3, the scales float, output and bias bfloat16
    if (left_8bit_tensor.dtype() != DType::E4M3 || right_8bit_tensor.dtype() != DType::E4M3 ||
        left_scales_tensor.dtype() != DType::Float32 || right_scales_tensor.dtype() != DType::Float32 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: fma_matmul_8bit needs left_8bit and right_8bit E4M3 (round_8bit's), the scales float, output "
                        "and bias bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses
    size_t sum_length_val = right_8bit_tensor.size(1);
    size_t columns_val = right_8bit_tensor.size(0);
    const size_t start_bytes_val = sizeof(float4);
    if (sum_length_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 || columns_val % VALUES_PER_FLOAT4 != 0 ||
        (size_t)output_tensor.data_ptr() % start_bytes_val != 0 || (size_t)left_8bit_tensor.data_ptr() % start_bytes_val != 0 ||
        (size_t)right_8bit_tensor.data_ptr() % start_bytes_val != 0 || (size_t)bias_tensor.data_ptr() % start_bytes_val != 0 ||
        (size_t)right_scales_tensor.data_ptr() % start_bytes_val != 0) {
        fprintf(stderr, "error: fma_matmul_8bit needs sum_length a multiple of %d and columns of %d, every tensor starting on "
                        "%zu bytes; got sum_length %zu, columns %zu\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE, VALUES_PER_FLOAT4, start_bytes_val, sum_length_val, columns_val);
        exit(EXIT_FAILURE);
    }
}

#endif // FMA_MATMUL_8BIT_CUH
