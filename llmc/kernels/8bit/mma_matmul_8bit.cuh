// mma_matmul_8bit.cuh — matrix products in 8 bits (FP8 E4M3) on the tensor cores by mma.sync (sm_89 or later: L4,
// 4090, H100, ...): mma_matmul_bf16.cuh's product with left and right rounded to E4M3 first (round_8bit.cuh), one
// float scale per row:
//     output[i][j] = left_scale[i] · right_scale[j] · sum over k of left_8bit[i][k] · right_8bit[j][k]
//                    (+ bias[j]) (+ output[i][j])
// A row's scale is the same for all its k, so it comes out of the sum: the tensor cores add up the E4M3 products in
// float, and both scales are applied once, as each sum is stored. E4M3 has twice bfloat16's values per byte, so one
// mma.sync m16n8k32 takes twice bfloat16's k16 in the same 32 bytes of a row.
//
//     tensor                    type in memory                   shape                               read with
//     left_8bit, right_8bit     E4M3, 4 in each int of an        {rows, sum_length / 4},              16-byte loads into shared memory
//                               Int32 tensor                     {columns, sum_length / 4}
//     left_scales, right_scales float                            {rows}, {columns}                    as the sums are stored
//     bias                      bfloat16                         {columns}, or none                   the same
//     output                    bfloat16                         {rows, columns}                      written over, or added onto
//
// Both copies sit k next to each other (K-major), as round_8bit writes them; a matrix that sits turned round is
// turned while rounding (round_8bit's turned). The squares and warps are mma_matmul_bf16.cuh's, the step
// MMA_MATMUL_8BIT_BLOCK_K bytes of k; the block size is fixed by the square, so this takes none.
#ifndef MMA_MATMUL_8BIT_CUH
#define MMA_MATMUL_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // HARDWARE_MMA_*, HARDWARE_E4M3_PER_REGISTER
#include "../../tensor.cuh"
#include "../bf16/mma_matmul_bf16.cuh"       // the squares and warps (MMA_MATMUL_BF16_BLOCK_*, _WARP_*, _M_TILES, _N_TILES), mma_matmul_bf16_block_square
#include "round_8bit.cuh"            // ROUND_8BIT_VALUES_PER_INT, ROUND_8BIT_ROW_LENGTH_MULTIPLE

// The steps: our choices. The index math is written for these numbers: change them together with it.
#define MMA_MATMUL_8BIT_BLOCK_K 64                                                          // values (bytes) of k a step copies: 2 mma steps of 32
#define MMA_MATMUL_8BIT_K_STEPS (MMA_MATMUL_8BIT_BLOCK_K / HARDWARE_MMA_K_E4M3)             // mma steps of k a step: 2
#define MMA_MATMUL_8BIT_LOAD_VALUES ((int)sizeof(uint4))                                    // E4M3s per 16-byte load: 16
#define MMA_MATMUL_8BIT_K_ROW_LENGTH (MMA_MATMUL_8BIT_BLOCK_K + MMA_MATMUL_8BIT_LOAD_VALUES)  // a tile row, one spare load for other banks: 80 bytes
#define MMA_MATMUL_8BIT_TILE_BYTES (MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_8BIT_K_ROW_LENGTH)   // one tile: 10,240 bytes
#define MMA_MATMUL_8BIT_LOADS_PER_THREAD (MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_8BIT_BLOCK_K / MMA_MATMUL_8BIT_LOAD_VALUES / MMA_MATMUL_BF16_THREADS_PER_BLOCK)   // 2
#define MMA_MATMUL_8BIT_COLUMN_MULTIPLE 8                                                   // columns' multiple: whole pairs, as mma_matmul_bf16's
static_assert(MMA_MATMUL_8BIT_LOADS_PER_THREAD * MMA_MATMUL_BF16_THREADS_PER_BLOCK * MMA_MATMUL_8BIT_LOAD_VALUES ==
              MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_8BIT_BLOCK_K, "the threads' loads cover a tile exactly");
static_assert(MMA_MATMUL_8BIT_BLOCK_K % HARDWARE_MMA_K_E4M3 == 0, "each step is whole mma steps");
static_assert(ROUND_8BIT_ROW_LENGTH_MULTIPLE % MMA_MATMUL_8BIT_LOAD_VALUES == 0, "round_8bit's rows are whole loads");

// ── the product ───────────────────────────────────────────────────────────────
void mma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                     const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor = Tensor(),
                     bool add_onto_val = false);

// ── for it ────────────────────────────────────────────────────────────────────
void mma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                                  const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(MMA_MATMUL_BF16_THREADS_PER_BLOCK)
mma_matmul_8bit_kernel(nv_bfloat16* output_ptr, const uint8_t* left_ptr, const float* left_scales_ptr, const uint8_t* right_ptr,
                       const float* right_scales_ptr, const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val,
                       size_t sum_length_val, bool add_onto_val);
__device__ __forceinline__ void mma_matmul_8bit_load_tile(uint4 (&loads_list)[MMA_MATMUL_8BIT_LOADS_PER_THREAD], const uint8_t* matrix_ptr,
                                                          size_t lines_val, size_t sum_length_val, size_t block_first_line_index_val,
                                                          size_t step_first_k_index_val);
__device__ __forceinline__ void mma_matmul_8bit_store_tile(uint8_t* tile_ptr, const uint4 (&loads_list)[MMA_MATMUL_8BIT_LOADS_PER_THREAD]);
__device__ __forceinline__ void mma_matmul_8bit_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3], const uint8_t* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val);
__device__ __forceinline__ void mma_matmul_8bit_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3], const uint8_t* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val);
__device__ __forceinline__ void mma_matmul_8bit_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_E4M3],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3]);

/// Writes output = left_scale · right_scale · (left_8bit · right_8bitᵀ) (+ bias) (+ output): checks, then launches
/// mma_matmul_8bit_kernel, one block per square of output, as one line of blocks.
///
///   output_tensor                           {..., columns} bfloat16: written over, or added onto
///   left_8bit_tensor, left_scales_tensor    round_8bit's of left {..., sum_length}: x in a linear layer
///   right_8bit_tensor, right_scales_tensor  round_8bit's of right {columns, sum_length}: W
///   bias_tensor                             {columns} bfloat16, or Tensor() for none
///   add_onto_val                            true: added onto what output holds; false: written over
/// sum_length must be a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE (16) and columns of MMA_MATMUL_8BIT_COLUMN_MULTIPLE (8).
inline void mma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                            const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor,
                            bool add_onto_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    mma_matmul_8bit_check_shapes(output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor,
                                 bias_tensor);
    // 2. launch: squares across and down, rounded up, as one line of blocks
    size_t sum_length_val = right_8bit_tensor.size(1) * ROUND_8BIT_VALUES_PER_INT;
    size_t rows_val = left_8bit_tensor.numel() * ROUND_8BIT_VALUES_PER_INT / sum_length_val;
    size_t columns_val = right_8bit_tensor.size(0);
    unsigned blocks_val = (unsigned)(CEIL_DIV(rows_val, (size_t)MMA_MATMUL_BF16_BLOCK_ROWS) *
                                     CEIL_DIV(columns_val, (size_t)MMA_MATMUL_BF16_BLOCK_COLUMNS));
    mma_matmul_8bit_kernel<<<blocks_val, MMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
        output_tensor.data_ptr<nv_bfloat16>(), reinterpret_cast<const uint8_t*>(left_8bit_tensor.data_ptr()), left_scales_tensor.data(),
        reinterpret_cast<const uint8_t*>(right_8bit_tensor.data_ptr()), right_scales_tensor.data(),
        bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr, rows_val, columns_val, sum_length_val,
        add_onto_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output: mma_matmul_bf16_kernel's walk on E4M3 tiles, mma.sync m16n8k32; at the end each
/// sum is times its row's and column's scales.
///
///   output_ptr, left_ptr, left_scales_ptr, right_ptr, right_scales_ptr, bias_ptr   as in mma_matmul_8bit; bias nullptr for none
///   rows_val, columns_val, sum_length_val                                           the product's sizes
///   add_onto_val                                                                    as in mma_matmul_8bit
__global__ void __launch_bounds__(MMA_MATMUL_BF16_THREADS_PER_BLOCK)
mma_matmul_8bit_kernel(nv_bfloat16* output_ptr, const uint8_t* left_ptr, const float* left_scales_ptr, const uint8_t* right_ptr,
                       const float* right_scales_ptr, const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val,
                       size_t sum_length_val, bool add_onto_val) {
    // this step's tiles, two sets: this step reads one while the next step's loads wait in registers
    __shared__ __align__(16) uint8_t left_tiles_list[2][MMA_MATMUL_8BIT_TILE_BYTES];
    __shared__ __align__(16) uint8_t right_tiles_list[2][MMA_MATMUL_8BIT_TILE_BYTES];
    // 1. the block's square, and this warp's part of it
    size_t block_first_row_index_val = 0, block_first_column_index_val = 0;
    mma_matmul_bf16_block_square(rows_val, columns_val, &block_first_row_index_val, &block_first_column_index_val);
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_first_row_index_val = (warp_index_val / MMA_MATMUL_BF16_WARPS_ACROSS) * MMA_MATMUL_BF16_WARP_ROWS;
    int warp_first_column_index_val = (warp_index_val % MMA_MATMUL_BF16_WARPS_ACROSS) * MMA_MATMUL_BF16_WARP_COLUMNS;

    // 2. walk the sum: copy each step's tiles, read the next step's, run the mmas
    float sums_list[MMA_MATMUL_BF16_M_TILES][MMA_MATMUL_BF16_N_TILES][HARDWARE_MMA_C_VALUES_PER_THREAD_E4M3] = {};
    uint4 left_loads_list[MMA_MATMUL_8BIT_LOADS_PER_THREAD];
    uint4 right_loads_list[MMA_MATMUL_8BIT_LOADS_PER_THREAD];
    mma_matmul_8bit_load_tile(left_loads_list, left_ptr, rows_val, sum_length_val, block_first_row_index_val, 0);
    mma_matmul_8bit_load_tile(right_loads_list, right_ptr, columns_val, sum_length_val, block_first_column_index_val, 0);
    size_t total_steps_val = CEIL_DIV(sum_length_val, (size_t)MMA_MATMUL_8BIT_BLOCK_K);
    for (size_t step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. this step's tiles into shared memory; the next step's loads started
        int tile_set_index_val = (int)(step_index_val % 2);
        mma_matmul_8bit_store_tile(left_tiles_list[tile_set_index_val], left_loads_list);
        mma_matmul_8bit_store_tile(right_tiles_list[tile_set_index_val], right_loads_list);
        __syncthreads();
        size_t next_step_first_k_index_val = (step_index_val + 1) * MMA_MATMUL_8BIT_BLOCK_K;
        if (next_step_first_k_index_val < sum_length_val) {
            mma_matmul_8bit_load_tile(left_loads_list, left_ptr, rows_val, sum_length_val, block_first_row_index_val, next_step_first_k_index_val);
            mma_matmul_8bit_load_tile(right_loads_list, right_ptr, columns_val, sum_length_val, block_first_column_index_val,
                                      next_step_first_k_index_val);
        }
        // b. the tensor cores: for each 32 values of k, this warp's A and B registers, then every A x B
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < MMA_MATMUL_8BIT_K_STEPS; k_step_index_val++) {
            int first_k_index_val = k_step_index_val * HARDWARE_MMA_K_E4M3;
            uint32_t a_list[MMA_MATMUL_BF16_M_TILES][HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3];
            uint32_t b_list[MMA_MATMUL_BF16_N_TILES][HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3];
#pragma unroll
            for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++)
                mma_matmul_8bit_load_a(a_list[m_tile_index_val], left_tiles_list[tile_set_index_val],
                                       warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M, first_k_index_val);
#pragma unroll
            for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++)
                mma_matmul_8bit_load_b(b_list[n_tile_index_val], right_tiles_list[tile_set_index_val],
                                       warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N, first_k_index_val);
#pragma unroll
            for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++)
#pragma unroll
                for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++)
                    mma_matmul_8bit_multiply_add(sums_list[m_tile_index_val][n_tile_index_val], a_list[m_tile_index_val],
                                                 b_list[n_tile_index_val]);
        }
    }

    // 3. output = left_scale · right_scale · sums (+ bias) (+ output), two neighbouring columns at a time; nothing
    //    past the edges. Each C tile: rows group and group + 8, columns pair and pair + 1 (the mma layout)
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
#pragma unroll
    for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++) {
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_MMA_ROWS_PER_THREAD; thread_row_index_val++) {
            size_t row_index_val = block_first_row_index_val + warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M +
                                   thread_row_index_val * HARDWARE_MMA_ROWS_PER_WARP + group_index_val;
            if (row_index_val >= rows_val) continue;
            float left_scale_val = left_scales_ptr[row_index_val];
#pragma unroll
            for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++) {
                size_t column_index_val = block_first_column_index_val + warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N +
                                          pair_column_index_val;
                if (column_index_val >= columns_val) continue;   // columns is even, so column + 1 is in too
                int c_index_val = HARDWARE_MMA_COLUMNS_PER_THREAD * thread_row_index_val;   // this row's 2 sums in the C tile
                float2 right_scales_val = *reinterpret_cast<const float2*>(right_scales_ptr + column_index_val);
                float2 bias_val = bias_ptr != nullptr ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                                      : make_float2(0.0f, 0.0f);
                float2 result_val = make_float2(sums_list[m_tile_index_val][n_tile_index_val][c_index_val] * left_scale_val * right_scales_val.x + bias_val.x,
                                                sums_list[m_tile_index_val][n_tile_index_val][c_index_val + 1] * left_scale_val * right_scales_val.y + bias_val.y);
                nv_bfloat162* output_pair_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + row_index_val * columns_val + column_index_val);
                if (add_onto_val) {
                    float2 old_val = __bfloat1622float2(*output_pair_ptr);
                    result_val = make_float2(result_val.x + old_val.x, result_val.y + old_val.y);
                }
                *output_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
            }
        }
    }
}

/// Reads this thread's 16-byte loads of one step's tile of left_8bit or right_8bit ({lines, sum_length} E4M3) from
/// GPU memory: a load is 16 neighbouring k of one line; 0s past the matrix's edges (a load is all in or all out).
///
///   loads_list                     written: this thread's MMA_MATMUL_8BIT_LOADS_PER_THREAD loads
///   matrix_ptr                     left_8bit or right_8bit
///   lines_val, sum_length_val      its sizes
///   block_first_line_index_val     the block's first line
///   step_first_k_index_val         the step's first k
__device__ __forceinline__ void mma_matmul_8bit_load_tile(uint4 (&loads_list)[MMA_MATMUL_8BIT_LOADS_PER_THREAD], const uint8_t* matrix_ptr,
                                                          size_t lines_val, size_t sum_length_val, size_t block_first_line_index_val,
                                                          size_t step_first_k_index_val) {
    const int loads_per_line_val = MMA_MATMUL_8BIT_BLOCK_K / MMA_MATMUL_8BIT_LOAD_VALUES;
#pragma unroll
    for (int load_index_val = 0; load_index_val < MMA_MATMUL_8BIT_LOADS_PER_THREAD; load_index_val++) {
        int piece_index_val = threadIdx.x + load_index_val * MMA_MATMUL_BF16_THREADS_PER_BLOCK;
        size_t line_index_val = block_first_line_index_val + piece_index_val / loads_per_line_val;
        size_t k_index_val = step_first_k_index_val + (piece_index_val % loads_per_line_val) * MMA_MATMUL_8BIT_LOAD_VALUES;
        loads_list[load_index_val] = line_index_val < lines_val && k_index_val < sum_length_val
                                         ? *reinterpret_cast<const uint4*>(matrix_ptr + line_index_val * sum_length_val + k_index_val)
                                         : make_uint4(0, 0, 0, 0);
    }
}

/// Writes mma_matmul_8bit_load_tile's loads into a tile in shared memory as they sat: tile[line][k], rows
/// MMA_MATMUL_8BIT_K_ROW_LENGTH bytes apart.
///
///   tile_ptr     one set of the shared memory tiles
///   loads_list   what mma_matmul_8bit_load_tile gave
__device__ __forceinline__ void mma_matmul_8bit_store_tile(uint8_t* tile_ptr, const uint4 (&loads_list)[MMA_MATMUL_8BIT_LOADS_PER_THREAD]) {
    const int loads_per_line_val = MMA_MATMUL_8BIT_BLOCK_K / MMA_MATMUL_8BIT_LOAD_VALUES;
#pragma unroll
    for (int load_index_val = 0; load_index_val < MMA_MATMUL_8BIT_LOADS_PER_THREAD; load_index_val++) {
        int piece_index_val = threadIdx.x + load_index_val * MMA_MATMUL_BF16_THREADS_PER_BLOCK;
        int offset_val = (piece_index_val / loads_per_line_val) * MMA_MATMUL_8BIT_K_ROW_LENGTH +
                         (piece_index_val % loads_per_line_val) * MMA_MATMUL_8BIT_LOAD_VALUES;
        *reinterpret_cast<uint4*>(tile_ptr + offset_val) = loads_list[load_index_val];
    }
}

/// Reads this thread's A registers of one mma tile out of left_8bit's tile. With group = lane / 4 and quad =
/// 4 (lane % 4), the registers hold A's (group, quad .. quad+3), (group+8, ..), (group, quad+16 ..), (group+8,
/// quad+16 ..): each 4 neighbouring E4M3s of k, one 4-byte read.
///
///   a_list                                    written: HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3 registers
///   tile_ptr                                  left_8bit's tile
///   first_row_index_val, first_k_index_val    the mma tile's {HARDWARE_MMA_M, HARDWARE_MMA_K_E4M3} corner in the tile
__device__ __forceinline__ void mma_matmul_8bit_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3], const uint8_t* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int quad_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_E4M3_PER_REGISTER;
    const int half_k_val = HARDWARE_MMA_K_E4M3 / 2;               // registers 2 and 3: 16 values of k on
    const uint8_t* top_ptr = tile_ptr + (first_row_index_val + group_index_val) * MMA_MATMUL_8BIT_K_ROW_LENGTH + first_k_index_val + quad_index_val;
    const uint8_t* bottom_ptr = top_ptr + HARDWARE_MMA_ROWS_PER_WARP * MMA_MATMUL_8BIT_K_ROW_LENGTH;
    a_list[0] = *reinterpret_cast<const uint32_t*>(top_ptr);
    a_list[1] = *reinterpret_cast<const uint32_t*>(bottom_ptr);
    a_list[2] = *reinterpret_cast<const uint32_t*>(top_ptr + half_k_val);
    a_list[3] = *reinterpret_cast<const uint32_t*>(bottom_ptr + half_k_val);
}

/// Reads this thread's B registers of one mma tile out of right_8bit's tile: B's rows (quad .. quad+3, column
/// group) and (quad+16 .. quad+19, column group), B[k][n] = right[n][k]: each one 4-byte read.
///
///   b_list                                       written: HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3 registers
///   tile_ptr                                     right_8bit's tile
///   first_column_index_val, first_k_index_val    the mma tile's first column (n) and first k in the tile
__device__ __forceinline__ void mma_matmul_8bit_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3], const uint8_t* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int quad_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_E4M3_PER_REGISTER;
    const int half_k_val = HARDWARE_MMA_K_E4M3 / 2;
    const uint8_t* at_ptr = tile_ptr + (first_column_index_val + group_index_val) * MMA_MATMUL_8BIT_K_ROW_LENGTH + first_k_index_val + quad_index_val;
    b_list[0] = *reinterpret_cast<const uint32_t*>(at_ptr);
    b_list[1] = *reinterpret_cast<const uint32_t*>(at_ptr + half_k_val);
}

/// C += A · B on the E4M3 tensor cores, for the whole warp: mma.sync m16n8k32, E4M3 in, float sums.
///
///   c_list   this thread's HARDWARE_MMA_C_VALUES_PER_THREAD_E4M3 floats of C
///   a_list   this thread's A registers
///   b_list   this thread's B registers
__device__ __forceinline__ void mma_matmul_8bit_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_E4M3],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= HARDWARE_MMA_E4M3_SMALLEST_CAPABILITY * 10
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c_list[0]), "+f"(c_list[1]), "+f"(c_list[2]), "+f"(c_list[3])
                 : "r"(a_list[0]), "r"(a_list[1]), "r"(a_list[2]), "r"(a_list[3]), "r"(b_list[0]), "r"(b_list[1]));
#endif
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Stops with a message unless the product's tensors fit: the shapes at the top of the file, the types, sum_length
/// a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE and columns of MMA_MATMUL_8BIT_COLUMN_MULTIPLE, each tensor
/// starting on 16 bytes.
///
///   output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor, bias_tensor
///       as in mma_matmul_8bit
inline void mma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
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
        fprintf(stderr, "error: mma_matmul_8bit needs left_8bit {..., sum_length / %d} with left_scales {...}, output {..., "
                        "columns} with the same rows, right_8bit {columns, sum_length / %d} with right_scales {columns}, "
                        "bias {columns} or Tensor()\n",
                (int)ROUND_8BIT_VALUES_PER_INT, (int)ROUND_8BIT_VALUES_PER_INT);
        exit(EXIT_FAILURE);
    }
    // 2. types: the 8-bit copies Int32, the scales float, output and bias bfloat16
    if (left_8bit_tensor.dtype() != DType::Int32 || right_8bit_tensor.dtype() != DType::Int32 ||
        left_scales_tensor.dtype() != DType::Float32 || right_scales_tensor.dtype() != DType::Float32 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: mma_matmul_8bit needs left_8bit and right_8bit Int32 (round_8bit's), the scales float, output "
                        "and bias bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte loads, each tensor starting on one
    size_t sum_length_val = right_8bit_tensor.size(1) * ROUND_8BIT_VALUES_PER_INT;
    size_t columns_val = right_8bit_tensor.size(0);
    const size_t load_bytes_val = sizeof(uint4);
    if (sum_length_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 || columns_val % MMA_MATMUL_8BIT_COLUMN_MULTIPLE != 0 ||
        (size_t)output_tensor.data_ptr() % load_bytes_val != 0 || (size_t)left_8bit_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)right_8bit_tensor.data_ptr() % load_bytes_val != 0 || (size_t)bias_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)right_scales_tensor.data_ptr() % sizeof(float2) != 0) {
        fprintf(stderr, "error: mma_matmul_8bit reads rows of whole %zu-byte loads, so sum_length must be a multiple of %d and "
                        "columns of %d, every tensor starting on %zu bytes; got sum_length %zu, columns %zu\n",
                load_bytes_val, ROUND_8BIT_ROW_LENGTH_MULTIPLE, MMA_MATMUL_8BIT_COLUMN_MULTIPLE, load_bytes_val, sum_length_val, columns_val);
        exit(EXIT_FAILURE);
    }
}

#endif // MMA_MATMUL_8BIT_CUH
