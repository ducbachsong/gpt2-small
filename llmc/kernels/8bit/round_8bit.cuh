// round_8bit.cuh — a bfloat16 matrix rounded to FP8 E4M3, one float scale per row: the 8-bit copies every 8-bit
// product reads (wgmma_matmul_8bit.cuh, mma_matmul_8bit.cuh, fma_matmul_8bit.cuh). Any GPU: no tensor cores here.
//
//     tensor     type in memory                  shape                       note
//     input      bfloat16                        {rows, row_length}          or, turned: {row_length, rows}
//     rounded    E4M3, 4 in each int of an       {rows, row_length / 4}      always rows first: k next to each other
//                Int32 tensor                                                (K-major), as the products read it
//     scales     float                           {rows}
//
// Each row is divided by its scale, the smallest power of 2 with
//     largest |value| / scale <= HARDWARE_E4M3_LARGEST (448)
// and rounded to the nearest E4M3: by up to 2^-4 (6%) of a value, against bfloat16's 2^-8. A power of 2, so turning
// back (E4M3 x scale) is exact.
//
// turned: input holds the matrix turned round, {row_length, rows}, as output_grad and input are in a linear layer's
// weight_grad = output_gradᵀ · input. The 8-bit products take only K-major copies (E4M3 WGMMA has no transpose), so
// the turning happens here, once, while rounding: two kernels, the columns' largest |value|s, then 32 x 32 tiles
// turned round through shared memory and rounded.
#ifndef ROUND_8BIT_CUH
#define ROUND_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_max, HARDWARE_E4M3_LARGEST
#include "../../tensor.cuh"

#define ROUND_8BIT_VALUES_PER_INT (sizeof(int) / sizeof(__nv_fp8_e4m3))      // E4M3s in one int of an Int32 tensor: 4
#define ROUND_8BIT_ROW_LENGTH_MULTIPLE 16                                    // row_length's multiple: the products read rows of whole 16-byte pieces
#define ROUND_8BIT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)         // the default: one warp per row (turned: warps share a column's values)
#define ROUND_8BIT_TURN_TILE HARDWARE_THREADS_PER_WARP                       // a turned tile: 32 x 32 values, one lane per value of a tile row
#define ROUND_8BIT_TURN_WARPS 8                                              // warps of a turning block: each takes every 8th tile row
#define ROUND_8BIT_TURN_THREADS (ROUND_8BIT_TURN_WARPS * HARDWARE_THREADS_PER_WARP)   // the turning's block size: fixed by the tile
static_assert(ROUND_8BIT_VALUES_PER_INT == VALUES_PER_FLOAT4, "the rounding writes an int of E4M3s from one float4");
static_assert(ROUND_8BIT_TURN_TILE % ROUND_8BIT_TURN_WARPS == 0, "each warp takes the same number of tile rows");

// ── the rounding ──────────────────────────────────────────────────────────────
void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val = false,
                int threads_per_block_val = ROUND_8BIT_THREADS_PER_BLOCK);

// ══ the code, in the order of the list above ══════════════════════════════════

__global__ void round_8bit_kernel(__nv_fp8_e4m3* rounded_ptr, float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                                  int row_length_val);
__global__ void round_8bit_turned_scales_kernel(float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val, size_t row_length_val);
__global__ void __launch_bounds__(ROUND_8BIT_TURN_THREADS)
round_8bit_turned_kernel(__nv_fp8_e4m3* rounded_ptr, const float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                         size_t row_length_val);
__device__ __forceinline__ float round_8bit_scale(float row_largest_val);

/// Writes input rounded to E4M3, one power-of-2 scale per row: checks, then launches round_8bit_kernel (one warp
/// per row), or for a turned input round_8bit_turned_scales_kernel and round_8bit_turned_kernel.
///
///   rounded_tensor          {rows, row_length / ROUND_8BIT_VALUES_PER_INT} Int32 (more dims allowed when not turned,
///                           as input's): written
///   scales_tensor           {rows} float (input's dims but the last, when not turned): written
///   input_tensor            bfloat16: {..., row_length}, or turned {..., rows} (row_length = all the other dims);
///                           row_length a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE
///   turned_val              true: input holds the matrix turned round
///   threads_per_block_val   the block size of the scales' kernel: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK;
///                           not given: ROUND_8BIT_THREADS_PER_BLOCK. Any size gives the same bits; only the time
///                           changes. The turning's block size is fixed by its tile.
inline void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val,
                       int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "round_8bit");
    bool fits_val = input_tensor.dim() >= 1 && input_tensor.dtype() == DType::BFloat16 &&
                    rounded_tensor.dtype() == DType::Int32 && scales_tensor.dtype() == DType::Float32;
    size_t last_size_val = fits_val ? input_tensor.size(input_tensor.dim() - 1) : 1;
    size_t rows_val = turned_val ? last_size_val : input_tensor.numel() / last_size_val;
    size_t row_length_val = turned_val ? input_tensor.numel() / last_size_val : last_size_val;
    fits_val = fits_val && row_length_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 &&
               rounded_tensor.dim() >= 1 && rounded_tensor.size(rounded_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT == row_length_val &&
               rounded_tensor.numel() * ROUND_8BIT_VALUES_PER_INT == rows_val * row_length_val && scales_tensor.numel() == rows_val &&
               (size_t)input_tensor.data_ptr() % (VALUES_PER_FLOAT4 * sizeof(nv_bfloat16)) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit needs a bfloat16 input {..., row_length} (turned: {..., rows}) with row_length a "
                        "multiple of %d, an Int32 rounded {rows, row_length / %d} and float scales {rows}\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE, (int)ROUND_8BIT_VALUES_PER_INT);
        exit(EXIT_FAILURE);
    }

    // 2. launch: one warp per row; turned, the columns' scales, then 32 x 32 tiles turned round
    __nv_fp8_e4m3* rounded_ptr = reinterpret_cast<__nv_fp8_e4m3*>(rounded_tensor.data_ptr());
    if (!turned_val) {
        int blocks_val = (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val);
        round_8bit_kernel<<<blocks_val, threads_per_block_val>>>(rounded_ptr, scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(),
                                                                 rows_val, (int)row_length_val);
    } else {
        int scale_blocks_val = (int)CEIL_DIV(rows_val, (size_t)HARDWARE_THREADS_PER_WARP);
        round_8bit_turned_scales_kernel<<<scale_blocks_val, threads_per_block_val>>>(
            scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), rows_val, row_length_val);
        cudaCheck(cudaGetLastError());
        unsigned tile_blocks_val = (unsigned)(CEIL_DIV(rows_val, (size_t)ROUND_8BIT_TURN_TILE) * CEIL_DIV(row_length_val, (size_t)ROUND_8BIT_TURN_TILE));
        round_8bit_turned_kernel<<<tile_blocks_val, ROUND_8BIT_TURN_THREADS>>>(
            rounded_ptr, scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), rows_val, row_length_val);
    }
    cudaCheck(cudaGetLastError());
}

/// Rounds each row to E4M3, one warp per row: the row's largest |value| (warp_reduce_max), its scale, then
/// value / scale rounded to the nearest E4M3 (saturating, as __nv_fp8x4_e4m3 does), VALUES_PER_FLOAT4 values per
/// lane at a time.
///
///   rounded_ptr                 {rows, row_length} E4M3: written
///   scales_ptr                  one float per row: written
///   input_ptr                   {rows, row_length} bfloat16
///   rows_val, row_length_val    the shape
__global__ void round_8bit_kernel(__nv_fp8_e4m3* rounded_ptr, float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                                  int row_length_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const nv_bfloat16* row_ptr = input_ptr + row_index_val * row_length_val;
        // 1. the row's largest |value|: this lane's float4s, then the warp's
        float thread_largest_val = 0.0f;
        for (int column_index_val = VALUES_PER_FLOAT4 * lane_index_val; column_index_val < row_length_val;
             column_index_val += VALUES_PER_FLOAT4 * HARDWARE_THREADS_PER_WARP) {
            float4 values_val = read4_as_float(row_ptr + column_index_val);
            thread_largest_val = fmaxf(thread_largest_val, fmaxf(fmaxf(fabsf(values_val.x), fabsf(values_val.y)),
                                                                 fmaxf(fabsf(values_val.z), fabsf(values_val.w))));
        }
        float row_largest_val = warp_reduce_max(thread_largest_val);
        // 2. its scale, and the values over it, rounded
        float scale_val = round_8bit_scale(row_largest_val);
        float to_stored_val = 1.0f / scale_val;                             // exact: a power of 2
        for (int column_index_val = VALUES_PER_FLOAT4 * lane_index_val; column_index_val < row_length_val;
             column_index_val += VALUES_PER_FLOAT4 * HARDWARE_THREADS_PER_WARP) {
            float4 values_val = read4_as_float(row_ptr + column_index_val);
            *reinterpret_cast<__nv_fp8x4_e4m3*>(rounded_ptr + row_index_val * row_length_val + column_index_val) =
                __nv_fp8x4_e4m3(make_float4(values_val.x * to_stored_val, values_val.y * to_stored_val,
                                            values_val.z * to_stored_val, values_val.w * to_stored_val));
        }
        if (lane_index_val == 0) scales_ptr[row_index_val] = scale_val;
    }
}

/// The scales of a turned input: each block takes HARDWARE_THREADS_PER_WARP neighbouring columns (rows of the
/// result), one lane per column, and its warps share the column's values: warp w reads input rows w, w + warps, ...
/// (lanes reading neighbouring values, so the reads are whole), then warp 0 takes the largest of the warps' and
/// writes the scale. The largest is the same whatever the order, so any block size gives the same bits.
///
///   scales_ptr                  one float per row of the result: written
///   input_ptr                   {row_length, rows} bfloat16
///   rows_val, row_length_val    the result's shape
__global__ void round_8bit_turned_scales_kernel(float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val, size_t row_length_val) {
    __shared__ float warp_largest_list[HARDWARE_MAX_WARPS_PER_BLOCK][HARDWARE_THREADS_PER_WARP];
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int warps_val = blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t row_index_val = (size_t)blockIdx.x * HARDWARE_THREADS_PER_WARP + lane_index_val;   // a column of input
    // 1. this warp's share of the column
    float largest_val = 0.0f;
    if (row_index_val < rows_val)
        for (size_t k_index_val = warp_index_val; k_index_val < row_length_val; k_index_val += warps_val)
            largest_val = fmaxf(largest_val, fabsf(__bfloat162float(input_ptr[k_index_val * rows_val + row_index_val])));
    warp_largest_list[warp_index_val][lane_index_val] = largest_val;
    __syncthreads();
    // 2. the warps' together, and the scale
    if (warp_index_val == 0 && row_index_val < rows_val) {
        for (int other_warp_index_val = 1; other_warp_index_val < warps_val; other_warp_index_val++)
            largest_val = fmaxf(largest_val, warp_largest_list[other_warp_index_val][lane_index_val]);
        scales_ptr[row_index_val] = round_8bit_scale(largest_val);
    }
}

/// Rounds a turned input, one ROUND_8BIT_TURN_TILE x ROUND_8BIT_TURN_TILE tile per block: the tile is read along
/// input's rows (lanes on neighbouring values), kept in shared memory, then written turned round along the result's
/// rows, each value over its row's scale rounded to the nearest E4M3 (saturating, as round_8bit_kernel's). The same
/// bits as round_8bit on the turned-round matrix.
///
///   rounded_ptr                 {rows, row_length} E4M3: written
///   scales_ptr                  one float per row, from round_8bit_turned_scales_kernel
///   input_ptr                   {row_length, rows} bfloat16
///   rows_val, row_length_val    the result's shape
__global__ void __launch_bounds__(ROUND_8BIT_TURN_THREADS)
round_8bit_turned_kernel(__nv_fp8_e4m3* rounded_ptr, const float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                         size_t row_length_val) {
    // one spare column: a tile column then sits in different banks, for the turned reads
    __shared__ float tile_list[ROUND_8BIT_TURN_TILE][ROUND_8BIT_TURN_TILE + 1];
    // 1. the block's tile, from its number: tiles along the result's rows first
    size_t tiles_per_row_val = CEIL_DIV(row_length_val, (size_t)ROUND_8BIT_TURN_TILE);
    size_t first_row_index_val = (blockIdx.x / tiles_per_row_val) * ROUND_8BIT_TURN_TILE;   // the result's rows: input's columns
    size_t first_k_index_val = (blockIdx.x % tiles_per_row_val) * ROUND_8BIT_TURN_TILE;     // the result's columns: input's rows
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    // 2. read the tile: tile[k][row] = input[first_k + k][first_row + row]
    for (int k_index_val = warp_index_val; k_index_val < ROUND_8BIT_TURN_TILE; k_index_val += ROUND_8BIT_TURN_WARPS) {
        size_t input_row_index_val = first_k_index_val + k_index_val;
        size_t input_column_index_val = first_row_index_val + lane_index_val;
        tile_list[k_index_val][lane_index_val] = input_row_index_val < row_length_val && input_column_index_val < rows_val
                                                     ? __bfloat162float(input_ptr[input_row_index_val * rows_val + input_column_index_val])
                                                     : 0.0f;
    }
    __syncthreads();
    // 3. write it turned round: rounded[first_row + row][first_k + k] = tile[k][row] / scale, rounded
    for (int row_offset_val = warp_index_val; row_offset_val < ROUND_8BIT_TURN_TILE; row_offset_val += ROUND_8BIT_TURN_WARPS) {
        size_t row_index_val = first_row_index_val + row_offset_val;
        size_t k_index_val = first_k_index_val + lane_index_val;
        if (row_index_val >= rows_val || k_index_val >= row_length_val) continue;
        float to_stored_val = 1.0f / scales_ptr[row_index_val];   // exact: a power of 2
        rounded_ptr[row_index_val * row_length_val + k_index_val] = __nv_fp8_e4m3(tile_list[lane_index_val][row_offset_val] * to_stored_val);
    }
}

/// Gives a row's scale: the smallest power of 2 with row_largest / scale <= HARDWARE_E4M3_LARGEST; 1 for a row of
/// 0s. The row's largest |value| / scale then lands in (HARDWARE_E4M3_LARGEST / 2, HARDWARE_E4M3_LARGEST].
///
///   row_largest_val   the row's largest |value|
__device__ __forceinline__ float round_8bit_scale(float row_largest_val) {
    // 1. a row of 0s: any scale; 1
    if (row_largest_val == 0.0f) return 1.0f;
    // 2. ratio = mantissa x 2^exponent, mantissa in [0.5, 1): the power of 2 at or above it is 2^(exponent - 1)
    //    when mantissa is exactly 0.5 (ratio a power of 2 itself), else 2^exponent
    int exponent_val = 0;
    float mantissa_val = frexpf(row_largest_val / HARDWARE_E4M3_LARGEST, &exponent_val);
    return ldexpf(1.0f, mantissa_val == 0.5f ? exponent_val - 1 : exponent_val);
}

#endif // ROUND_8BIT_CUH
