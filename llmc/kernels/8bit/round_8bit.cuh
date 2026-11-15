// round_8bit.cuh — a bfloat16 matrix rounded to FP8 E4M3, one float scale per row: the 8-bit copies every 8-bit
// product reads (wgmma_matmul_8bit.cuh, mma_matmul_8bit.cuh, fma_matmul_8bit.cuh, 8bit/prod/). Any GPU: no tensor
// cores here.
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
// the turning happens here, while rounding, as NVIDIA Transformer Engine's cast-transpose does: input is read in
// ROUND_8BIT_TILE x ROUND_8BIT_TILE tiles, 16 bytes a thread, twice: first each column's largest |value| (an atomic
// max a tile), then each tile turned round through shared memory and written 16 E4M3s a thread.
//
// round_8bit_rows_and_columns makes both copies of one matrix in those two reads: as it sits (one scale per row, for
// a product summing along its rows, e.g. output_grad · weight) and turned (one scale per column, for one summing
// down its columns, e.g. output_gradᵀ · input). A linear layer's backward needs both of output_grad; each from its
// own call would read it four times.
#ifndef ROUND_8BIT_CUH
#define ROUND_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_max, HARDWARE_E4M3_LARGEST
#include "../../common/tensor.cuh"          // read4_as_float, read8_as_float, VALUES_PER_BF16_LOAD

#define ROUND_8BIT_VALUES_PER_INT (sizeof(int) / sizeof(__nv_fp8_e4m3))      // E4M3s in one int of an Int32 tensor: 4
#define ROUND_8BIT_ROW_LENGTH_MULTIPLE 16                                    // row_length's multiple: the products read rows of whole 16-byte pieces
#define ROUND_8BIT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)         // the default: one warp per row
#define ROUND_8BIT_TILE 128                                                  // the tile kernels' tile: 128 x 128 values, a turned row of it 128 E4M3s (128 bytes)
#define ROUND_8BIT_TILE_THREADS (8 * HARDWARE_THREADS_PER_WARP)              // the tile kernels' block size: fixed by the tile
#define ROUND_8BIT_LOAD_VALUES VALUES_PER_BF16_LOAD                          // bfloat16s a thread reads at once: one 16-byte load, 8
#define ROUND_8BIT_STORE_VALUES ((int)(sizeof(uint4) / sizeof(__nv_fp8_e4m3)))   // E4M3s a thread writes turned at once: one 16-byte store, 16
#define ROUND_8BIT_LOADS_PER_TILE_ROW (ROUND_8BIT_TILE / ROUND_8BIT_LOAD_VALUES)   // a tile row's loads: 16, half a warp's lanes
#define ROUND_8BIT_TILE_ROWS_AT_ONCE (ROUND_8BIT_TILE_THREADS / ROUND_8BIT_LOADS_PER_TILE_ROW)   // tile rows a block reads at once: 16
#define ROUND_8BIT_TILE_PADDING 2                                            // bfloat16s past a tile row in shared memory: rows start in other banks, and a row stays on 4 bytes
#define ROUND_8BIT_STORES_PER_TURNED_ROW (ROUND_8BIT_TILE / ROUND_8BIT_STORE_VALUES)   // a turned tile row's stores: 8
static_assert(ROUND_8BIT_VALUES_PER_INT == VALUES_PER_FLOAT4, "the rounding writes an int of E4M3s from one float4");
static_assert(ROUND_8BIT_ROW_LENGTH_MULTIPLE % ROUND_8BIT_STORE_VALUES == 0 && ROUND_8BIT_ROW_LENGTH_MULTIPLE % ROUND_8BIT_LOAD_VALUES == 0,
              "a row of whole 16-byte pieces is whole loads and whole turned stores");
static_assert(ROUND_8BIT_TILE % ROUND_8BIT_TILE_ROWS_AT_ONCE == 0, "a tile is whole passes of the block's rows");
static_assert(ROUND_8BIT_TILE * ROUND_8BIT_STORES_PER_TURNED_ROW % ROUND_8BIT_TILE_THREADS == 0, "each thread writes as many turned pieces");

// ── the rounding ──────────────────────────────────────────────────────────────
void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val = false,
                int threads_per_block_val = ROUND_8BIT_THREADS_PER_BLOCK);
void round_8bit_rows_and_columns(const Tensor& rows_8bit_tensor, const Tensor& row_scales_tensor, const Tensor& columns_8bit_tensor,
                                 const Tensor& column_scales_tensor, const Tensor& input_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

__global__ void round_8bit_kernel(__nv_fp8_e4m3* rounded_ptr, float* scales_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                                  int row_length_val);
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_largest_kernel(unsigned* row_largest_ptr, unsigned* column_largest_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                          size_t columns_val);
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_tile_kernel(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                       const unsigned* row_largest_ptr, const unsigned* column_largest_ptr, const nv_bfloat16* input_ptr,
                       size_t rows_val, size_t columns_val);
void round_8bit_by_tiles(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                         const nv_bfloat16* input_ptr, size_t rows_val, size_t columns_val);
__device__ __forceinline__ float round_8bit_scale(float row_largest_val);

/// Writes input rounded to E4M3, one power-of-2 scale per row: checks, then launches round_8bit_kernel (one warp
/// per row), or for a turned input the tile kernels (round_8bit_by_tiles).
///
///   rounded_tensor          {rows, row_length / ROUND_8BIT_VALUES_PER_INT} Int32 (more dims allowed when not turned,
///                           as input's): written
///   scales_tensor           {rows} float (input's dims but the last, when not turned): written
///   input_tensor            bfloat16: {..., row_length}, or turned {..., rows} (row_length = all the other dims);
///                           row_length a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE
///   turned_val              true: input holds the matrix turned round
///   threads_per_block_val   round_8bit_kernel's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                           given: ROUND_8BIT_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
///                           The tile kernels' block size is fixed by the tile.
inline void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val,
                       int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "round_8bit");
    bool fits_val = input_tensor.dim() >= 1 && input_tensor.dtype() == DType::BFloat16 &&
                    rounded_tensor.dtype() == DType::Int32 && scales_tensor.dtype() == DType::Float32;
    size_t last_size_val = fits_val ? input_tensor.size(input_tensor.dim() - 1) : 1;
    size_t rows_val = turned_val ? last_size_val : input_tensor.numel() / last_size_val;
    size_t row_length_val = turned_val ? input_tensor.numel() / last_size_val : last_size_val;
    fits_val = fits_val && row_length_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && (!turned_val || rows_val % ROUND_8BIT_LOAD_VALUES == 0) &&
               rounded_tensor.dim() >= 1 && rounded_tensor.size(rounded_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT == row_length_val &&
               rounded_tensor.numel() * ROUND_8BIT_VALUES_PER_INT == rows_val * row_length_val && scales_tensor.numel() == rows_val &&
               (size_t)input_tensor.data_ptr() % (turned_val ? sizeof(uint4) : VALUES_PER_FLOAT4 * sizeof(nv_bfloat16)) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit needs a bfloat16 input {..., row_length} (turned: {..., rows}, rows a multiple of %d, "
                        "starting on 16 bytes) with row_length a multiple of %d, an Int32 rounded {rows, row_length / %d} and float "
                        "scales {rows}\n",
                ROUND_8BIT_LOAD_VALUES, ROUND_8BIT_ROW_LENGTH_MULTIPLE, (int)ROUND_8BIT_VALUES_PER_INT);
        exit(EXIT_FAILURE);
    }

    // 2. launch: one warp per row; turned, the tile kernels writing only the turned copy (input's columns are the
    //    copy's rows)
    __nv_fp8_e4m3* rounded_ptr = reinterpret_cast<__nv_fp8_e4m3*>(rounded_tensor.data_ptr());
    if (!turned_val) {
        int blocks_val = (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val);
        round_8bit_kernel<<<blocks_val, threads_per_block_val>>>(rounded_ptr, scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(),
                                                                 rows_val, (int)row_length_val);
        cudaCheck(cudaGetLastError());
    } else {
        round_8bit_by_tiles(nullptr, nullptr, rounded_ptr, scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), row_length_val,
                            rows_val);
    }
}

/// Writes both 8-bit copies of one matrix, in the tile kernels' two reads of it: as it sits (round_8bit's, one scale
/// per row) and turned round (round_8bit turned's, one scale per column of input). The same bits as those two
/// calls. Either copy may be left out (Tensor()), not both.
///
///   rows_8bit_tensor, row_scales_tensor         {rows, columns / ROUND_8BIT_VALUES_PER_INT} Int32 and {rows} float:
///                                               written; or both Tensor()
///   columns_8bit_tensor, column_scales_tensor   {columns, rows / ROUND_8BIT_VALUES_PER_INT} Int32 and {columns} float:
///                                               written; or both Tensor()
///   input_tensor                                bfloat16 {..., columns} (rows = all the other dims); rows and columns
///                                               multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE
inline void round_8bit_rows_and_columns(const Tensor& rows_8bit_tensor, const Tensor& row_scales_tensor, const Tensor& columns_8bit_tensor,
                                        const Tensor& column_scales_tensor, const Tensor& input_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    bool has_rows_val = rows_8bit_tensor.numel() > 0;
    bool has_columns_val = columns_8bit_tensor.numel() > 0;
    bool fits_val = input_tensor.dim() >= 1 && input_tensor.dtype() == DType::BFloat16 && (has_rows_val || has_columns_val) &&
                    (size_t)input_tensor.data_ptr() % sizeof(uint4) == 0;
    size_t columns_val = fits_val ? input_tensor.size(input_tensor.dim() - 1) : 1;
    size_t rows_val = input_tensor.numel() / columns_val;
    fits_val = fits_val && rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && columns_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0;
    if (has_rows_val)
        fits_val = fits_val && rows_8bit_tensor.dtype() == DType::Int32 && row_scales_tensor.dtype() == DType::Float32 &&
                   rows_8bit_tensor.numel() * ROUND_8BIT_VALUES_PER_INT == rows_val * columns_val &&
                   rows_8bit_tensor.size(rows_8bit_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT == columns_val &&
                   row_scales_tensor.numel() == rows_val;
    if (has_columns_val)
        fits_val = fits_val && columns_8bit_tensor.dtype() == DType::Int32 && column_scales_tensor.dtype() == DType::Float32 &&
                   columns_8bit_tensor.numel() * ROUND_8BIT_VALUES_PER_INT == rows_val * columns_val &&
                   columns_8bit_tensor.size(columns_8bit_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT == rows_val &&
                   column_scales_tensor.numel() == columns_val;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit_rows_and_columns needs a bfloat16 input {rows, columns}, both multiples of %d, starting "
                        "on 16 bytes, and Int32 copies {rows, columns / %d} with float scales {rows} and/or {columns, rows / %d} "
                        "with {columns}\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE, (int)ROUND_8BIT_VALUES_PER_INT, (int)ROUND_8BIT_VALUES_PER_INT);
        exit(EXIT_FAILURE);
    }
    // 2. the two reads
    round_8bit_by_tiles(has_rows_val ? reinterpret_cast<__nv_fp8_e4m3*>(rows_8bit_tensor.data_ptr()) : nullptr,
                        has_rows_val ? row_scales_tensor.data() : nullptr,
                        has_columns_val ? reinterpret_cast<__nv_fp8_e4m3*>(columns_8bit_tensor.data_ptr()) : nullptr,
                        has_columns_val ? column_scales_tensor.data() : nullptr, input_tensor.data_ptr<nv_bfloat16>(), rows_val,
                        columns_val);
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

/// The tile kernels' first read: each row's and each column's largest |value| of input {rows, columns}, one
/// ROUND_8BIT_TILE x ROUND_8BIT_TILE tile per block (tiles across, blockIdx.x; down, blockIdx.y). Thread t reads load
/// t % ROUND_8BIT_LOADS_PER_TILE_ROW of tile rows t / ROUND_8BIT_LOADS_PER_TILE_ROW, + ROUND_8BIT_TILE_ROWS_AT_ONCE,
/// ...: a row's 16 loads are half a warp, whose largest is a shuffle away; a column's are the threads of the same
/// load in each half-warp, met in shared memory. Each block's largest go into the row's and the column's by an atomic
/// max on the float's bits (|value| >= 0, so the bits order as the floats do); both lists must hold 0s first. The
/// largest is the same whatever the order, so the bits are too.
///
///   row_largest_ptr      {rows} the float bits: maxed onto, or nullptr (no row copy wanted)
///   column_largest_ptr   {columns} the same, or nullptr (no turned copy wanted)
///   input_ptr            {rows, columns} bfloat16, on 16 bytes
///   rows_val, columns_val   the shape: columns a multiple of ROUND_8BIT_LOAD_VALUES
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_largest_kernel(unsigned* row_largest_ptr, unsigned* column_largest_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                          size_t columns_val) {
    __shared__ float column_largest_list[ROUND_8BIT_TILE_ROWS_AT_ONCE][ROUND_8BIT_TILE];
    // 1. this thread's load in the tile
    int load_index_val = threadIdx.x % ROUND_8BIT_LOADS_PER_TILE_ROW;
    int first_tile_row_val = threadIdx.x / ROUND_8BIT_LOADS_PER_TILE_ROW;
    size_t first_column_index_val = (size_t)blockIdx.x * ROUND_8BIT_TILE + load_index_val * ROUND_8BIT_LOAD_VALUES;
    bool column_in_val = first_column_index_val < columns_val;
    float thread_column_largest_list[ROUND_8BIT_LOAD_VALUES] = {};
    // 2. its rows of the tile: each row's largest, the half-warp's, onto the row's; each column's kept
    for (int tile_row_val = first_tile_row_val; tile_row_val < ROUND_8BIT_TILE; tile_row_val += ROUND_8BIT_TILE_ROWS_AT_ONCE) {
        size_t row_index_val = (size_t)blockIdx.y * ROUND_8BIT_TILE + tile_row_val;
        float values_list[ROUND_8BIT_LOAD_VALUES] = {};
        if (row_index_val < rows_val && column_in_val)
            read8_as_float(values_list, input_ptr + row_index_val * columns_val + first_column_index_val);
        float load_largest_val = 0.0f;
#pragma unroll
        for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val++) {
            float magnitude_val = fabsf(values_list[value_index_val]);
            load_largest_val = fmaxf(load_largest_val, magnitude_val);
            thread_column_largest_list[value_index_val] = fmaxf(thread_column_largest_list[value_index_val], magnitude_val);
        }
#pragma unroll
        for (int lane_step_val = ROUND_8BIT_LOADS_PER_TILE_ROW / 2; lane_step_val > 0; lane_step_val /= 2)   // within the half-warp
            load_largest_val = fmaxf(load_largest_val, __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, load_largest_val, lane_step_val));
        if (row_largest_ptr != nullptr && load_index_val == 0 && row_index_val < rows_val)
            atomicMax(row_largest_ptr + row_index_val, __float_as_uint(load_largest_val));
    }
    if (column_largest_ptr == nullptr) return;
    // 3. the columns: each thread's into shared memory, then one thread a column takes the largest onto the column's
#pragma unroll
    for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val++)
        column_largest_list[first_tile_row_val][load_index_val * ROUND_8BIT_LOAD_VALUES + value_index_val] = thread_column_largest_list[value_index_val];
    __syncthreads();
    if (threadIdx.x < ROUND_8BIT_TILE) {
        size_t column_index_val = (size_t)blockIdx.x * ROUND_8BIT_TILE + threadIdx.x;
        float largest_val = 0.0f;
        for (int row_group_val = 0; row_group_val < ROUND_8BIT_TILE_ROWS_AT_ONCE; row_group_val++)
            largest_val = fmaxf(largest_val, column_largest_list[row_group_val][threadIdx.x]);
        if (column_index_val < columns_val) atomicMax(column_largest_ptr + column_index_val, __float_as_uint(largest_val));
    }
}

/// The tile kernels' second read: writes the copies of input {rows, columns}, one ROUND_8BIT_TILE x ROUND_8BIT_TILE
/// tile per block, as round_8bit_largest_kernel reads it. Each load is
///     the row copy's: times 1 / its row's scale, rounded, written straight back (8 E4M3s, 8 bytes, a half-warp's
///                     row 128 bytes)
///     the turned copy's: kept in shared memory; then each thread takes ROUND_8BIT_STORE_VALUES tile rows of one
///                     tile column (neighbouring lanes, neighbouring columns: other banks), times 1 / the column's
///                     scale, rounded, and writes them as 16 E4M3s (16 bytes) of the turned row
/// The blocks of the first tile row write the columns' scales, those of the first tile column the rows'. Rounding
/// as round_8bit_kernel does (__nv_fp8x4_e4m3, saturating), so the same bits.
///
///   rows_8bit_ptr, row_scales_ptr           {rows, columns} E4M3 and {rows}: written; or nullptr
///   columns_8bit_ptr, column_scales_ptr     {columns, rows} E4M3 and {columns}: written; or nullptr
///   row_largest_ptr, column_largest_ptr     round_8bit_largest_kernel's (nullptr where it was given nullptr)
///   input_ptr, rows_val, columns_val        as there: rows a multiple of ROUND_8BIT_STORE_VALUES when turned
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_tile_kernel(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                       const unsigned* row_largest_ptr, const unsigned* column_largest_ptr, const nv_bfloat16* input_ptr,
                       size_t rows_val, size_t columns_val) {
    __shared__ __align__(16) nv_bfloat16 tile_list[ROUND_8BIT_TILE][ROUND_8BIT_TILE + ROUND_8BIT_TILE_PADDING];
    // 1. this thread's load in the tile
    int load_index_val = threadIdx.x % ROUND_8BIT_LOADS_PER_TILE_ROW;
    int first_tile_row_val = threadIdx.x / ROUND_8BIT_LOADS_PER_TILE_ROW;
    size_t first_row_index_val = (size_t)blockIdx.y * ROUND_8BIT_TILE;
    size_t first_column_index_val = (size_t)blockIdx.x * ROUND_8BIT_TILE;
    size_t load_column_index_val = first_column_index_val + load_index_val * ROUND_8BIT_LOAD_VALUES;
    bool column_in_val = load_column_index_val < columns_val;

    // 2. the tile's rows: the row copy written, the values kept for the turned copy
    for (int tile_row_val = first_tile_row_val; tile_row_val < ROUND_8BIT_TILE; tile_row_val += ROUND_8BIT_TILE_ROWS_AT_ONCE) {
        size_t row_index_val = first_row_index_val + tile_row_val;
        bool in_val = row_index_val < rows_val && column_in_val;
        float values_list[ROUND_8BIT_LOAD_VALUES] = {};
        if (in_val) read8_as_float(values_list, input_ptr + row_index_val * columns_val + load_column_index_val);
        if (rows_8bit_ptr != nullptr && in_val) {
            float scale_val = round_8bit_scale(__uint_as_float(row_largest_ptr[row_index_val]));
            float to_stored_val = 1.0f / scale_val;                     // exact: a power of 2
            __nv_fp8x4_e4m3 stored_list[ROUND_8BIT_LOAD_VALUES / VALUES_PER_FLOAT4];
#pragma unroll
            for (int four_index_val = 0; four_index_val < ROUND_8BIT_LOAD_VALUES / VALUES_PER_FLOAT4; four_index_val++)
                stored_list[four_index_val] = __nv_fp8x4_e4m3(make_float4(
                    values_list[VALUES_PER_FLOAT4 * four_index_val] * to_stored_val, values_list[VALUES_PER_FLOAT4 * four_index_val + 1] * to_stored_val,
                    values_list[VALUES_PER_FLOAT4 * four_index_val + 2] * to_stored_val, values_list[VALUES_PER_FLOAT4 * four_index_val + 3] * to_stored_val));
            *reinterpret_cast<uint2*>(rows_8bit_ptr + row_index_val * columns_val + load_column_index_val) =
                *reinterpret_cast<const uint2*>(stored_list);
            if (row_scales_ptr != nullptr && blockIdx.x == 0 && load_index_val == 0) row_scales_ptr[row_index_val] = scale_val;
        }
        if (columns_8bit_ptr != nullptr) {
#pragma unroll
            for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val += 2)
                *reinterpret_cast<nv_bfloat162*>(&tile_list[tile_row_val][load_index_val * ROUND_8BIT_LOAD_VALUES + value_index_val]) =
                    __floats2bfloat162_rn(values_list[value_index_val], values_list[value_index_val + 1]);   // exact: they were bfloat16s
        }
    }
    if (columns_8bit_ptr == nullptr) return;
    __syncthreads();

    // 3. the turned copy: piece p of the block is tile column p % ROUND_8BIT_TILE, tile rows
    //    ROUND_8BIT_STORE_VALUES x (p / ROUND_8BIT_TILE) on; a warp's lanes on neighbouring columns
    for (int piece_index_val = threadIdx.x; piece_index_val < ROUND_8BIT_TILE * ROUND_8BIT_STORES_PER_TURNED_ROW;
         piece_index_val += ROUND_8BIT_TILE_THREADS) {
        int tile_column_val = piece_index_val % ROUND_8BIT_TILE;
        int first_tile_row_of_piece_val = (piece_index_val / ROUND_8BIT_TILE) * ROUND_8BIT_STORE_VALUES;
        size_t column_index_val = first_column_index_val + tile_column_val;        // the turned copy's row
        size_t first_row_of_piece_val = first_row_index_val + first_tile_row_of_piece_val;
        if (column_index_val >= columns_val || first_row_of_piece_val >= rows_val) continue;
        float scale_val = round_8bit_scale(__uint_as_float(column_largest_ptr[column_index_val]));
        float to_stored_val = 1.0f / scale_val;                         // exact: a power of 2
        __nv_fp8x4_e4m3 stored_list[ROUND_8BIT_STORE_VALUES / VALUES_PER_FLOAT4];
#pragma unroll
        for (int four_index_val = 0; four_index_val < ROUND_8BIT_STORE_VALUES / VALUES_PER_FLOAT4; four_index_val++) {
            int tile_row_val = first_tile_row_of_piece_val + VALUES_PER_FLOAT4 * four_index_val;
            stored_list[four_index_val] = __nv_fp8x4_e4m3(make_float4(
                __bfloat162float(tile_list[tile_row_val][tile_column_val]) * to_stored_val,
                __bfloat162float(tile_list[tile_row_val + 1][tile_column_val]) * to_stored_val,
                __bfloat162float(tile_list[tile_row_val + 2][tile_column_val]) * to_stored_val,
                __bfloat162float(tile_list[tile_row_val + 3][tile_column_val]) * to_stored_val));
        }
        *reinterpret_cast<uint4*>(columns_8bit_ptr + column_index_val * rows_val + first_row_of_piece_val) =
            *reinterpret_cast<const uint4*>(stored_list);
        if (column_scales_ptr != nullptr && blockIdx.y == 0 && first_tile_row_of_piece_val == 0) column_scales_ptr[column_index_val] = scale_val;
    }
}

/// Runs the tile kernels on input {rows, columns}: the largest lists zeroed, round_8bit_largest_kernel, then
/// round_8bit_tile_kernel, one block per tile. The largest lists are scratch in GPU memory, made the first time and
/// grown with the largest call (rows + columns unsigneds).
///
///   rows_8bit_ptr, row_scales_ptr         the row copy and its scales, or nullptr
///   columns_8bit_ptr, column_scales_ptr   the turned copy and its scales, or nullptr
///   input_ptr, rows_val, columns_val      input and its shape
inline void round_8bit_by_tiles(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                                const nv_bfloat16* input_ptr, size_t rows_val, size_t columns_val) {
    // 1. the largest lists, 0s
    static Tensor largest_tensor;
    if ((size_t)largest_tensor.numel() < rows_val + columns_val) largest_tensor = Tensor::zeros({rows_val + columns_val});
    unsigned* row_largest_ptr = rows_8bit_ptr != nullptr ? reinterpret_cast<unsigned*>(largest_tensor.data()) : nullptr;
    unsigned* column_largest_ptr = columns_8bit_ptr != nullptr ? reinterpret_cast<unsigned*>(largest_tensor.data()) + rows_val : nullptr;
    cudaCheck(cudaMemsetAsync(largest_tensor.data(), 0, (rows_val + columns_val) * sizeof(unsigned)));
    // 2. the two reads, a block a tile
    dim3 grid_val((unsigned)CEIL_DIV(columns_val, (size_t)ROUND_8BIT_TILE), (unsigned)CEIL_DIV(rows_val, (size_t)ROUND_8BIT_TILE));
    round_8bit_largest_kernel<<<grid_val, ROUND_8BIT_TILE_THREADS>>>(row_largest_ptr, column_largest_ptr, input_ptr, rows_val, columns_val);
    cudaCheck(cudaGetLastError());
    round_8bit_tile_kernel<<<grid_val, ROUND_8BIT_TILE_THREADS>>>(rows_8bit_ptr, row_scales_ptr, columns_8bit_ptr, column_scales_ptr,
                                                                  row_largest_ptr, column_largest_ptr, input_ptr, rows_val, columns_val);
    cudaCheck(cudaGetLastError());
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
