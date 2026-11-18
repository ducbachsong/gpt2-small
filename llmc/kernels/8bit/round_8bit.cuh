// round_8bit.cuh — a bfloat16 matrix rounded to FP8 E4M3, one float scale per row: the 8-bit copies every 8-bit
// product reads (wgmma_matmul_8bit.cuh, mma_matmul_8bit.cuh, fma_matmul_8bit.cuh, 8bit/prod/). Any GPU: no tensor
// cores here.
//
//     tensor     type in memory                  shape                       note
//     input      bfloat16                        {rows, row_length}          or, turned: {row_length, rows}
//     rounded    E4M3                            {rows, row_length}          always rows first: k next to each other
//                                                                            (K-major), as the products read it
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
//
// round_8bit_per_tensor: one scale for the whole tensor (torchao's and nanochat's "tensorwise" FP8, NVIDIA Transformer
// Engine's default) in place of one per row: both copies in ONE read by the tile kernel, the scale known before it. The
// scale comes from the tensor's largest |value|: found by a read of its own (exact), or, given a Round8bitHistory,
// taken from the tensor the same place held the call before, times ROUND_8BIT_DELAYED_MARGIN (Transformer Engine's
// delayed scaling: an activation's range moves little from one micro-batch to the next), while this read finds this
// tensor's largest for the next call. A value past the scale's range is clamped to the largest E4M3 (448 x scale).
// The scale is written as one float {1}, both copies' (the products take one scale for each side). What it costs: a row much smaller than the tensor's largest keeps fewer of its bits (E4M3 has 2^-9 .. 448: values under
// 2^-18 of the largest round to 0).
#ifndef ROUND_8BIT_CUH
#define ROUND_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_max, HARDWARE_E4M3_LARGEST
#include "../../common/tensor.cuh"          // read4_as_float, read8_as_float, VALUES_PER_BF16_LOAD

#define ROUND_8BIT_ROW_LENGTH_MULTIPLE 16                                    // row_length's multiple: the products read rows of whole 16-byte pieces
#define ROUND_8BIT_THREADS_PER_BLOCK 0                                       // as it sits: the default block size, one block a row: 0, chosen by the row's width (round_8bit_threads_for)
#define ROUND_8BIT_LOADS_PER_THREAD_AIM 2                                    // ... chosen: the fewest threads, a power of 2, holding the row in this many loads a thread ...
#define ROUND_8BIT_LEAST_THREADS (8 * HARDWARE_THREADS_PER_WARP)             // ... at least 256 ...
#define ROUND_8BIT_MOST_THREADS (16 * HARDWARE_THREADS_PER_WARP)             // ... at most 512: the fastest for rows of 2,304 (256) and 9,216 (512) on an H100 PCIe (tune_round_8bit)
#define ROUND_8BIT_MAX_LOADS_PER_THREAD 16                                   // as it sits: 16-byte loads a thread holds in registers, at most (1, 2, 4, 8 or 16, a kernel each): a row of up to 16 x 8 x the block's threads values, 32,768 at 256
#define ROUND_8BIT_TILE 128                                                  // the tile kernels' tile: 128 x 128 values, a turned row of it 128 E4M3s (128 bytes)
#define ROUND_8BIT_TILE_THREADS (8 * HARDWARE_THREADS_PER_WARP)              // the tile kernels' block size: fixed by the tile
#define ROUND_8BIT_LOAD_VALUES VALUES_PER_BF16_LOAD                          // bfloat16s a thread reads at once: one 16-byte load, 8
#define ROUND_8BIT_STORE_VALUES ((int)(sizeof(uint4) / sizeof(__nv_fp8_e4m3)))   // E4M3s a thread writes turned at once: one 16-byte store, 16
#define ROUND_8BIT_LOADS_PER_TILE_ROW (ROUND_8BIT_TILE / ROUND_8BIT_LOAD_VALUES)   // a tile row's loads: 16, half a warp's lanes
#define ROUND_8BIT_TILE_ROWS_AT_ONCE (ROUND_8BIT_TILE_THREADS / ROUND_8BIT_LOADS_PER_TILE_ROW)   // tile rows a block reads at once: 16
#define ROUND_8BIT_TILE_PADDING 2                                            // bfloat16s past a tile row in shared memory: rows start in other banks, and a row stays on 4 bytes
#define ROUND_8BIT_STORES_PER_TURNED_ROW (ROUND_8BIT_TILE / ROUND_8BIT_STORE_VALUES)   // a turned tile row's stores: 8
#define ROUND_8BIT_DELAYED_MARGIN 2.0f                                       // per tensor, the scale from the call before: room for values this many times its largest (the scale a power of 2: 2 to 4 times)
#define ROUND_8BIT_LARGEST_THREADS (8 * HARDWARE_THREADS_PER_WARP)           // per tensor, the exact largest's block size
#define ROUND_8BIT_LARGEST_LOADS_PER_THREAD 16                               // ... and the 16-byte loads a thread of it reads
static_assert(ROUND_8BIT_ROW_LENGTH_MULTIPLE % ROUND_8BIT_STORE_VALUES == 0 && ROUND_8BIT_ROW_LENGTH_MULTIPLE % ROUND_8BIT_LOAD_VALUES == 0,
              "a row of whole 16-byte pieces is whole loads and whole turned stores");
static_assert(ROUND_8BIT_TILE % ROUND_8BIT_TILE_ROWS_AT_ONCE == 0, "a tile is whole passes of the block's rows");
static_assert(ROUND_8BIT_TILE * ROUND_8BIT_STORES_PER_TURNED_ROW % ROUND_8BIT_TILE_THREADS == 0, "each thread writes as many turned pieces");

// ── the rounding ──────────────────────────────────────────────────────────────
void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val = false,
                int threads_per_block_val = ROUND_8BIT_THREADS_PER_BLOCK);
void round_8bit_rows_and_columns(const Tensor& rows_8bit_tensor, const Tensor& row_scales_tensor, const Tensor& columns_8bit_tensor,
                                 const Tensor& column_scales_tensor, const Tensor& input_tensor);

/// What round_8bit_per_tensor keeps for one place in a model (one tensor of one layer) from call to call: the largest
/// |value| of the tensor it rounded last, and of the one before, in GPU memory (the float's bits, in turns).
struct Round8bitHistory {
    Tensor largest_tensor = Tensor::zeros({2});             // the last two tensors' largest |value|, as float bits
    mutable size_t calls_val = 0;                           // tensors rounded here so far: the last one's slot is (calls - 1) % 2
};
void round_8bit_per_tensor(const Tensor& rows_8bit_tensor, const Tensor& columns_8bit_tensor, const Tensor& scale_tensor,
                           const Tensor& input_tensor, const Round8bitHistory* history_ptr = nullptr, bool same_input_val = false);
bool round_8bit_delayed_largest(const Round8bitHistory& history_val, const unsigned*& scale_largest_ptr, unsigned*& next_largest_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

template <int LOADS_PER_THREAD>
__global__ void round_8bit_rows_kernel(__nv_fp8_e4m3* rounded_ptr, float* scales_ptr, const nv_bfloat16* input_ptr, int row_length_val);
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_largest_kernel(unsigned* row_largest_ptr, unsigned* column_largest_ptr, const nv_bfloat16* input_ptr, size_t rows_val,
                          size_t columns_val);
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_tile_kernel(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                       const unsigned* row_largest_ptr, const unsigned* column_largest_ptr, const nv_bfloat16* input_ptr,
                       size_t rows_val, size_t columns_val, const unsigned* tensor_largest_ptr, float margin_val,
                       unsigned* next_largest_ptr, float* tensor_scale_ptr);
__global__ void round_8bit_tensor_largest_kernel(unsigned* largest_ptr, const nv_bfloat16* input_ptr, size_t loads_val);
void round_8bit_by_tiles(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                         const nv_bfloat16* input_ptr, size_t rows_val, size_t columns_val);
int round_8bit_threads_for(size_t row_length_val);
__device__ __forceinline__ float round_8bit_scale(float row_largest_val);

/// Writes input rounded to E4M3, one power-of-2 scale per row: checks, then launches round_8bit_rows_kernel (one
/// block a row, the row read once into registers), or for a turned input the tile kernels (round_8bit_by_tiles).
///
///   rounded_tensor          {rows, row_length} E4M3 (more dims allowed when not turned,
///                           as input's): written
///   scales_tensor           {rows} float (input's dims but the last, when not turned): written
///   input_tensor            bfloat16: {..., row_length}, or turned {..., rows} (row_length = all the other dims);
///                           row_length a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE; starting on 16 bytes
///   turned_val              true: input holds the matrix turned round
///   threads_per_block_val   a row's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; 0 (not given:
///                           ROUND_8BIT_THREADS_PER_BLOCK): round_8bit_threads_for the row's width. Each thread holds the fewest loads of 1, 2, 4, 8, 16 its row
///                           needs; a row longer than ROUND_8BIT_MAX_LOADS_PER_THREAD loads a thread (the head's logits'
///                           gradient, 50,304 wide: 24.6 at 256) goes to the tile kernels instead. Any size gives the
///                           same bits; only the time changes. The tile kernels' block size is fixed by the tile.
inline void round_8bit(const Tensor& rounded_tensor, const Tensor& scales_tensor, const Tensor& input_tensor, bool turned_val,
                       int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    bool fits_val = input_tensor.dim() >= 1 && input_tensor.dtype() == DType::BFloat16 &&
                    rounded_tensor.dtype() == DType::E4M3 && scales_tensor.dtype() == DType::Float32;
    size_t last_size_val = fits_val ? input_tensor.size(input_tensor.dim() - 1) : 1;
    size_t rows_val = turned_val ? last_size_val : input_tensor.numel() / last_size_val;
    size_t row_length_val = turned_val ? input_tensor.numel() / last_size_val : last_size_val;
    if (threads_per_block_val == 0) threads_per_block_val = round_8bit_threads_for(row_length_val);
    check_threads_per_block(threads_per_block_val, "round_8bit");
    fits_val = fits_val && row_length_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && (!turned_val || rows_val % ROUND_8BIT_LOAD_VALUES == 0) &&
               rounded_tensor.dim() >= 1 && rounded_tensor.size(rounded_tensor.dim() - 1) == row_length_val &&
               rounded_tensor.numel() == rows_val * row_length_val && scales_tensor.numel() == rows_val &&
               (size_t)input_tensor.data_ptr() % sizeof(uint4) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit needs a bfloat16 input {..., row_length} (turned: {..., rows}, rows a multiple of %d) "
                        "starting on 16 bytes, with row_length a multiple of %d, an E4M3 rounded {rows, row_length} and "
                        "float scales {rows}\n",
                ROUND_8BIT_LOAD_VALUES, ROUND_8BIT_ROW_LENGTH_MULTIPLE);
        exit(EXIT_FAILURE);
    }
    size_t row_loads_val = row_length_val / ROUND_8BIT_LOAD_VALUES;
    int loads_per_thread_val = 1;                            // the fewest of 1, 2, 4, 8, 16 that hold the row
    while (loads_per_thread_val < ROUND_8BIT_MAX_LOADS_PER_THREAD && (size_t)loads_per_thread_val * threads_per_block_val < row_loads_val)
        loads_per_thread_val *= 2;
    // the kernel made for that many loads, and the largest block its registers allow (asked of CUDA once a kernel)
    using RowsKernel = void (*)(__nv_fp8_e4m3*, float*, const nv_bfloat16*, int);
    static const RowsKernel kernels_list[] = {round_8bit_rows_kernel<1>, round_8bit_rows_kernel<2>, round_8bit_rows_kernel<4>,
                                              round_8bit_rows_kernel<8>, round_8bit_rows_kernel<16>};
    static_assert(ROUND_8BIT_MAX_LOADS_PER_THREAD == 16, "a kernel above for each count of loads up to the most");
    static int most_threads_list[sizeof(kernels_list) / sizeof(kernels_list[0])] = {};
    int kernel_index_val = __builtin_ctz((unsigned)loads_per_thread_val);   // 1, 2, 4, 8, 16: 0 .. 4
    if (most_threads_list[kernel_index_val] == 0) {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        most_threads_list[kernel_index_val] = attributes_val.maxThreadsPerBlock;
    }
    bool row_fits_block_val = (size_t)loads_per_thread_val * threads_per_block_val >= row_loads_val &&
                              threads_per_block_val <= most_threads_list[kernel_index_val];

    // 2. launch: one block a row, the kernel made for its loads a thread; a row longer than the block holds (or a
    //    block larger than that kernel's registers allow), or turned, the tile kernels writing only the copy asked for
    //    (turned: input's columns are the copy's rows)
    __nv_fp8_e4m3* rounded_ptr = reinterpret_cast<__nv_fp8_e4m3*>(rounded_tensor.data_ptr());
    if (!turned_val && !row_fits_block_val) {
        round_8bit_by_tiles(rounded_ptr, scales_tensor.data(), nullptr, nullptr, input_tensor.data_ptr<nv_bfloat16>(), rows_val,
                            row_length_val);
    } else if (!turned_val) {
        kernels_list[kernel_index_val]<<<(unsigned)rows_val, threads_per_block_val>>>(
            rounded_ptr, scales_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), (int)row_length_val);
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
///   rows_8bit_tensor, row_scales_tensor         {rows, columns} E4M3 and {rows} float:
///                                               written; or both Tensor()
///   columns_8bit_tensor, column_scales_tensor   {columns, rows} E4M3 and {columns} float:
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
        fits_val = fits_val && rows_8bit_tensor.dtype() == DType::E4M3 && row_scales_tensor.dtype() == DType::Float32 &&
                   rows_8bit_tensor.numel() == rows_val * columns_val &&
                   rows_8bit_tensor.size(rows_8bit_tensor.dim() - 1) == columns_val &&
                   row_scales_tensor.numel() == rows_val;
    if (has_columns_val)
        fits_val = fits_val && columns_8bit_tensor.dtype() == DType::E4M3 && column_scales_tensor.dtype() == DType::Float32 &&
                   columns_8bit_tensor.numel() == rows_val * columns_val &&
                   columns_8bit_tensor.size(columns_8bit_tensor.dim() - 1) == rows_val &&
                   column_scales_tensor.numel() == columns_val;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit_rows_and_columns needs a bfloat16 input {rows, columns}, both multiples of %d, starting "
                        "on 16 bytes, and E4M3 copies {rows, columns} with float scales {rows} and/or {columns, rows} "
                        "with {columns}\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE);
        exit(EXIT_FAILURE);
    }
    // 2. the two reads
    round_8bit_by_tiles(has_rows_val ? reinterpret_cast<__nv_fp8_e4m3*>(rows_8bit_tensor.data_ptr()) : nullptr,
                        has_rows_val ? row_scales_tensor.data() : nullptr,
                        has_columns_val ? reinterpret_cast<__nv_fp8_e4m3*>(columns_8bit_tensor.data_ptr()) : nullptr,
                        has_columns_val ? column_scales_tensor.data() : nullptr, input_tensor.data_ptr<nv_bfloat16>(), rows_val,
                        columns_val);
}

/// Rounds one row to E4M3 per block, reading it once: each thread's LOADS_PER_THREAD 16-byte loads (loads t,
/// t + threads, ... of the row) kept in registers, the row's largest |value| (each warp's by warp_reduce_max, the
/// warps' through shared memory), its scale, then those values / scale rounded to the nearest E4M3 (saturating, as
/// __nv_fp8x4_e4m3 does), 8 a store. The largest is the same whatever the order, so any block size gives the same
/// bits.
///
///   rounded_ptr      {rows, row_length} E4M3: written
///   scales_ptr       one float per row: written
///   input_ptr        {rows, row_length} bfloat16, on 16 bytes; rows: the launch's blocks
///   row_length_val   a multiple of ROUND_8BIT_LOAD_VALUES, at most LOADS_PER_THREAD x 8 x the block's threads
/// LOADS_PER_THREAD: the loads a thread holds, the registers its values take (8 floats each).
template <int LOADS_PER_THREAD>
__global__ void round_8bit_rows_kernel(__nv_fp8_e4m3* rounded_ptr, float* scales_ptr, const nv_bfloat16* input_ptr, int row_length_val) {
    __shared__ float warp_largest_list[HARDWARE_MAX_WARPS_PER_BLOCK];
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int row_loads_val = row_length_val / ROUND_8BIT_LOAD_VALUES;
    const nv_bfloat16* row_ptr = input_ptr + (size_t)blockIdx.x * row_length_val;
    // 1. this thread's loads of the row, kept, and their largest |value|
    float values_list[LOADS_PER_THREAD][ROUND_8BIT_LOAD_VALUES];
    float thread_largest_val = 0.0f;
#pragma unroll
    for (int thread_load_index_val = 0; thread_load_index_val < LOADS_PER_THREAD; thread_load_index_val++) {
        int load_index_val = threadIdx.x + thread_load_index_val * blockDim.x;
        if (load_index_val >= row_loads_val) continue;
        read8_as_float(values_list[thread_load_index_val], row_ptr + load_index_val * ROUND_8BIT_LOAD_VALUES);
#pragma unroll
        for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val++)
            thread_largest_val = fmaxf(thread_largest_val, fabsf(values_list[thread_load_index_val][value_index_val]));
    }
    // 2. the row's largest: each warp's, then the warps' (every thread reads them all), and its scale
    float warp_largest_val = warp_reduce_max(thread_largest_val);
    if (lane_index_val == 0) warp_largest_list[warp_index_val] = warp_largest_val;
    __syncthreads();
    float row_largest_val = 0.0f;
    for (int other_warp_index_val = 0; other_warp_index_val < (int)(blockDim.x / HARDWARE_THREADS_PER_WARP); other_warp_index_val++)
        row_largest_val = fmaxf(row_largest_val, warp_largest_list[other_warp_index_val]);
    float scale_val = round_8bit_scale(row_largest_val);
    float to_stored_val = 1.0f / scale_val;                                 // exact: a power of 2
    // 3. the kept values over it, rounded, 8 a store
    __nv_fp8_e4m3* rounded_row_ptr = rounded_ptr + (size_t)blockIdx.x * row_length_val;
#pragma unroll
    for (int thread_load_index_val = 0; thread_load_index_val < LOADS_PER_THREAD; thread_load_index_val++) {
        int load_index_val = threadIdx.x + thread_load_index_val * blockDim.x;
        if (load_index_val >= row_loads_val) continue;
        const float* values_ptr = values_list[thread_load_index_val];
        __nv_fp8x4_e4m3 stored_list[ROUND_8BIT_LOAD_VALUES / VALUES_PER_FLOAT4];
#pragma unroll
        for (int four_index_val = 0; four_index_val < ROUND_8BIT_LOAD_VALUES / VALUES_PER_FLOAT4; four_index_val++)
            stored_list[four_index_val] = __nv_fp8x4_e4m3(make_float4(
                values_ptr[VALUES_PER_FLOAT4 * four_index_val] * to_stored_val, values_ptr[VALUES_PER_FLOAT4 * four_index_val + 1] * to_stored_val,
                values_ptr[VALUES_PER_FLOAT4 * four_index_val + 2] * to_stored_val, values_ptr[VALUES_PER_FLOAT4 * four_index_val + 3] * to_stored_val));
        *reinterpret_cast<uint2*>(rounded_row_ptr + load_index_val * ROUND_8BIT_LOAD_VALUES) = *reinterpret_cast<const uint2*>(stored_list);
    }
    if (threadIdx.x == 0) scales_ptr[blockIdx.x] = scale_val;
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
/// as round_8bit_rows_kernel does (__nv_fp8x4_e4m3, saturating), so the same bits.
///
///   rows_8bit_ptr, row_scales_ptr           {rows, columns} E4M3 and {rows}: written; or nullptr (the scales
///                                           nullptr alone: one scale per tensor, into tensor_scale_ptr)
///   columns_8bit_ptr, column_scales_ptr     {columns, rows} E4M3 and {columns}: written; or nullptr (likewise)
///   row_largest_ptr, column_largest_ptr     round_8bit_largest_kernel's (nullptr where it was given nullptr)
///   input_ptr, rows_val, columns_val        as there: rows a multiple of ROUND_8BIT_STORE_VALUES when turned
///   tensor_largest_ptr, margin_val          one scale per tensor (round_8bit_per_tensor): a largest |value| (float
///                                           bits) and what to multiply it by; every value's scale is the scale of
///                                           that. nullptr: one scale per row and column, from their lists
///   next_largest_ptr                        this tensor's largest |value| maxed onto it (float bits, 0 before), or
///                                           nullptr
///   tensor_scale_ptr                        one scale per tensor: that scale, one float, written by the first block's
///                                           first thread (the scales lists then nullptr); or nullptr
__global__ void __launch_bounds__(ROUND_8BIT_TILE_THREADS)
round_8bit_tile_kernel(__nv_fp8_e4m3* rows_8bit_ptr, float* row_scales_ptr, __nv_fp8_e4m3* columns_8bit_ptr, float* column_scales_ptr,
                       const unsigned* row_largest_ptr, const unsigned* column_largest_ptr, const nv_bfloat16* input_ptr,
                       size_t rows_val, size_t columns_val, const unsigned* tensor_largest_ptr, float margin_val,
                       unsigned* next_largest_ptr, float* tensor_scale_ptr) {
    __shared__ __align__(16) nv_bfloat16 tile_list[ROUND_8BIT_TILE][ROUND_8BIT_TILE + ROUND_8BIT_TILE_PADDING];
    // 1. this thread's load in the tile
    int load_index_val = threadIdx.x % ROUND_8BIT_LOADS_PER_TILE_ROW;
    int first_tile_row_val = threadIdx.x / ROUND_8BIT_LOADS_PER_TILE_ROW;
    size_t first_row_index_val = (size_t)blockIdx.y * ROUND_8BIT_TILE;
    size_t first_column_index_val = (size_t)blockIdx.x * ROUND_8BIT_TILE;
    size_t load_column_index_val = first_column_index_val + load_index_val * ROUND_8BIT_LOAD_VALUES;
    bool column_in_val = load_column_index_val < columns_val;
    bool per_tensor_val = tensor_largest_ptr != nullptr;
    float tensor_scale_val = per_tensor_val ? round_8bit_scale(__uint_as_float(*tensor_largest_ptr) * margin_val) : 1.0f;
    float thread_largest_val = 0.0f;                                    // per tensor: this thread's loads' largest |value|
    if (tensor_scale_ptr != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *tensor_scale_ptr = tensor_scale_val;

    // 2. the tile's rows: the row copy written, the values kept for the turned copy
    for (int tile_row_val = first_tile_row_val; tile_row_val < ROUND_8BIT_TILE; tile_row_val += ROUND_8BIT_TILE_ROWS_AT_ONCE) {
        size_t row_index_val = first_row_index_val + tile_row_val;
        bool in_val = row_index_val < rows_val && column_in_val;
        float values_list[ROUND_8BIT_LOAD_VALUES] = {};
        if (in_val) read8_as_float(values_list, input_ptr + row_index_val * columns_val + load_column_index_val);
        if (next_largest_ptr != nullptr) {
#pragma unroll
            for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val++)
                thread_largest_val = fmaxf(thread_largest_val, fabsf(values_list[value_index_val]));
        }
        if (rows_8bit_ptr != nullptr && in_val) {
            float scale_val = per_tensor_val ? tensor_scale_val : round_8bit_scale(__uint_as_float(row_largest_ptr[row_index_val]));
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
    //    per tensor: the block's largest |value| onto the next call's, a warp at a time
    if (next_largest_ptr != nullptr) {
        float warp_largest_val = warp_reduce_max(thread_largest_val);
        if (threadIdx.x % HARDWARE_THREADS_PER_WARP == 0 && warp_largest_val > 0.0f)
            atomicMax(next_largest_ptr, __float_as_uint(warp_largest_val));
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
        float scale_val = per_tensor_val ? tensor_scale_val : round_8bit_scale(__uint_as_float(column_largest_ptr[column_index_val]));
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
                                                                  row_largest_ptr, column_largest_ptr, input_ptr, rows_val, columns_val,
                                                                  nullptr, 1.0f, nullptr, nullptr);
    cudaCheck(cudaGetLastError());
}

/// Maxes the largest |value| of a whole bfloat16 tensor onto largest_ptr (the float's bits; |value| >= 0, so the
/// bits order as the floats do): each thread reads ROUND_8BIT_LARGEST_LOADS_PER_THREAD 16-byte loads (its block's
/// stretch, a block's width apart), each warp's largest by warp_reduce_max, one atomic max a warp.
///
///   largest_ptr   one unsigned, 0 before: maxed onto
///   input_ptr     the tensor, on 16 bytes
///   loads_val     its 16-byte loads: its values / ROUND_8BIT_LOAD_VALUES
__global__ void round_8bit_tensor_largest_kernel(unsigned* largest_ptr, const nv_bfloat16* input_ptr, size_t loads_val) {
    size_t block_first_load_val = (size_t)blockIdx.x * blockDim.x * ROUND_8BIT_LARGEST_LOADS_PER_THREAD;
    float thread_largest_val = 0.0f;
#pragma unroll
    for (int thread_load_index_val = 0; thread_load_index_val < ROUND_8BIT_LARGEST_LOADS_PER_THREAD; thread_load_index_val++) {
        size_t load_index_val = block_first_load_val + (size_t)thread_load_index_val * blockDim.x + threadIdx.x;
        if (load_index_val >= loads_val) continue;
        float values_list[ROUND_8BIT_LOAD_VALUES];
        read8_as_float(values_list, input_ptr + load_index_val * ROUND_8BIT_LOAD_VALUES);
#pragma unroll
        for (int value_index_val = 0; value_index_val < ROUND_8BIT_LOAD_VALUES; value_index_val++)
            thread_largest_val = fmaxf(thread_largest_val, fabsf(values_list[value_index_val]));
    }
    float warp_largest_val = warp_reduce_max(thread_largest_val);
    if (threadIdx.x % HARDWARE_THREADS_PER_WARP == 0 && warp_largest_val > 0.0f)
        atomicMax(largest_ptr, __float_as_uint(warp_largest_val));
}

/// Writes the 8-bit copies of one matrix with ONE scale for the whole tensor: as it sits and/or turned round (as
/// round_8bit_rows_and_columns: either may be left out), in one read by the tile kernel, and that scale (a power of 2,
/// both copies'). Where the scale comes from:
///     no history                  this tensor's largest |value|, by a read of its own first: exact
///     a history, its first call   the same, kept in the history
///     a history, later calls      the largest of the tensor rounded there the call before, x ROUND_8BIT_DELAYED_MARGIN
///                                 (no read for it); this read finds this tensor's largest for the next call. A value
///                                 past the range is clamped to the largest E4M3
///     same_input_val              the tensor is the one the history's last call rounded (a linear's input, rounded as
///                                 it sits by the forward, now turned for the weight_grad): its largest is known, exact
///
///   rows_8bit_tensor      {rows, columns} E4M3: written; or Tensor()
///   columns_8bit_tensor   {columns, rows} E4M3, turned round: written; or Tensor() (not both)
///   scale_tensor          {1} float: written, both copies' scale
///   input_tensor          bfloat16 {..., columns} (rows = all the other dims); rows and columns multiples of
///                         ROUND_8BIT_ROW_LENGTH_MULTIPLE
///   history_ptr           the place's history, or nullptr (not given)
///   same_input_val        see above; needs a history
inline void round_8bit_per_tensor(const Tensor& rows_8bit_tensor, const Tensor& columns_8bit_tensor, const Tensor& scale_tensor,
                                  const Tensor& input_tensor, const Round8bitHistory* history_ptr, bool same_input_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    bool has_rows_val = rows_8bit_tensor.numel() > 0;
    bool has_columns_val = columns_8bit_tensor.numel() > 0;
    bool fits_val = input_tensor.dim() >= 1 && input_tensor.dtype() == DType::BFloat16 && (has_rows_val || has_columns_val) &&
                    (size_t)input_tensor.data_ptr() % sizeof(uint4) == 0 && scale_tensor.dtype() == DType::Float32 &&
                    scale_tensor.numel() == 1;
    size_t columns_val = fits_val ? input_tensor.size(input_tensor.dim() - 1) : 1;
    size_t rows_val = input_tensor.numel() / columns_val;
    fits_val = fits_val && rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && columns_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0;
    if (has_rows_val)
        fits_val = fits_val && rows_8bit_tensor.dtype() == DType::E4M3 && rows_8bit_tensor.numel() == rows_val * columns_val;
    if (has_columns_val)
        fits_val = fits_val && columns_8bit_tensor.dtype() == DType::E4M3 && columns_8bit_tensor.numel() == rows_val * columns_val;
    if (!fits_val) {
        fprintf(stderr, "error: round_8bit_per_tensor needs a bfloat16 input {rows, columns}, both multiples of %d, starting on 16 "
                        "bytes, E4M3 copies {rows, columns} and/or {columns, rows}, and a float scale {1}\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE);
        exit(EXIT_FAILURE);
    }
    const nv_bfloat16* input_ptr = input_tensor.data_ptr<nv_bfloat16>();
    // 2. the largest the scale comes from, its margin, and where this tensor's largest goes
    static Tensor exact_largest_tensor = Tensor::zeros({1});    // no history: the exact largest, used by this call only
    unsigned* slots_ptr = history_ptr != nullptr ? reinterpret_cast<unsigned*>(history_ptr->largest_tensor.data()) : nullptr;
    const unsigned* scale_largest_ptr = nullptr;
    unsigned* next_largest_ptr = nullptr;
    float margin_val = 1.0f;
    if (history_ptr == nullptr || history_ptr->calls_val == 0) {
        //    exact: a read of its own finds the largest
        unsigned* exact_ptr = history_ptr != nullptr ? slots_ptr : reinterpret_cast<unsigned*>(exact_largest_tensor.data());
        size_t loads_val = rows_val * columns_val / ROUND_8BIT_LOAD_VALUES;
        cudaCheck(cudaMemsetAsync(exact_ptr, 0, sizeof(unsigned)));
        round_8bit_tensor_largest_kernel<<<(unsigned)CEIL_DIV(loads_val, (size_t)(ROUND_8BIT_LARGEST_THREADS * ROUND_8BIT_LARGEST_LOADS_PER_THREAD)),
                                           ROUND_8BIT_LARGEST_THREADS>>>(exact_ptr, input_ptr, loads_val);
        cudaCheck(cudaGetLastError());
        scale_largest_ptr = exact_ptr;
        if (history_ptr != nullptr) history_ptr->calls_val = 1;
    } else if (same_input_val) {
        //    the last call's tensor again: its largest is in the history
        scale_largest_ptr = slots_ptr + (history_ptr->calls_val - 1) % 2;
    } else {
        //    delayed: the scale from the last call's largest, with room; this tensor's largest into the other slot
        round_8bit_delayed_largest(*history_ptr, scale_largest_ptr, next_largest_ptr);
        margin_val = ROUND_8BIT_DELAYED_MARGIN;
    }
    // 3. the one read, a block a tile
    dim3 grid_val((unsigned)CEIL_DIV(columns_val, (size_t)ROUND_8BIT_TILE), (unsigned)CEIL_DIV(rows_val, (size_t)ROUND_8BIT_TILE));
    round_8bit_tile_kernel<<<grid_val, ROUND_8BIT_TILE_THREADS>>>(
        has_rows_val ? reinterpret_cast<__nv_fp8_e4m3*>(rows_8bit_tensor.data_ptr()) : nullptr, nullptr,
        has_columns_val ? reinterpret_cast<__nv_fp8_e4m3*>(columns_8bit_tensor.data_ptr()) : nullptr, nullptr, nullptr, nullptr,
        input_ptr, rows_val, columns_val, scale_largest_ptr, margin_val, next_largest_ptr, scale_tensor.data());
    cudaCheck(cudaGetLastError());
}

/// Takes a delayed scale's turn in a history, for a kernel that rounds a tensor it writes itself with one scale for the
/// tensor (mlp_up_8bit_gelu_8bit's gelu_output), as round_8bit_per_tensor's later calls do: the scale is to come from
/// the largest of the tensor rounded there the call before (x ROUND_8BIT_DELAYED_MARGIN), and this tensor's largest is
/// to be maxed into the other slot, zeroed here; the call is counted. On the history's first call there is no largest
/// yet: nothing is done, false, and the caller rounds that tensor by round_8bit_per_tensor (exact, which starts the
/// history).
///
///   history_val         the place's history
///   scale_largest_ptr   given: the largest the scale comes from (a float's bits, GPU memory)
///   next_largest_ptr    given: where this tensor's largest goes (0 now)
inline bool round_8bit_delayed_largest(const Round8bitHistory& history_val, const unsigned*& scale_largest_ptr, unsigned*& next_largest_ptr) {
    if (history_val.calls_val == 0) return false;
    unsigned* slots_ptr = reinterpret_cast<unsigned*>(history_val.largest_tensor.data());
    scale_largest_ptr = slots_ptr + (history_val.calls_val - 1) % 2;
    next_largest_ptr = slots_ptr + history_val.calls_val % 2;
    cudaCheck(cudaMemsetAsync(next_largest_ptr, 0, sizeof(unsigned)));
    history_val.calls_val++;
    return true;
}

/// Gives round_8bit's block size for rows as they sit, from their width: the fewest threads, a power of 2, that
/// hold the row in ROUND_8BIT_LOADS_PER_THREAD_AIM loads each, between ROUND_8BIT_LEAST_THREADS and
/// ROUND_8BIT_MOST_THREADS: 256 for 2,304 values (1.1 loads a thread), 512 for 9,216 (2.2).
///
///   row_length_val   the row's values
inline int round_8bit_threads_for(size_t row_length_val) {
    size_t wanted_val = CEIL_DIV(row_length_val / ROUND_8BIT_LOAD_VALUES, (size_t)ROUND_8BIT_LOADS_PER_THREAD_AIM);
    int threads_val = ROUND_8BIT_LEAST_THREADS;
    while (threads_val < ROUND_8BIT_MOST_THREADS && (size_t)threads_val < wanted_val) threads_val *= 2;
    return threads_val;
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
