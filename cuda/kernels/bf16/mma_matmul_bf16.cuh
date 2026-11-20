// mma_matmul_bf16.cuh — matrix products in bfloat16 on the tensor cores by mma.sync (sm_80 or later).
//
//     tensor   type in memory   shape
//     left     bfloat16         {rows, sum_length}, turned {sum_length, rows}
//     right    bfloat16         {columns, sum_length}, turned {sum_length, columns}
//     bias     bfloat16         {columns}, or none
//     output   bfloat16         {rows, columns}
//     sums     float            in registers
#ifndef MMA_MATMUL_BF16_CUH
#define MMA_MATMUL_BF16_CUH

#include "../cuda_common.h"             // cudaCheck, HARDWARE_MMA_*, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"

// The squares and steps: our choices. The hardware's numbers (HARDWARE_MMA_*) are in cuda_common.h. The index math
// is written for these numbers: change them together with it.
#define MMA_MATMUL_BF16_BLOCK_ROWS 128                 // each block makes 128 rows x 128 columns of output
#define MMA_MATMUL_BF16_BLOCK_COLUMNS 128
#define MMA_MATMUL_BF16_BLOCK_K 32                     // values of k a step copies: 2 mma steps of 16
#define MMA_MATMUL_BF16_WARPS_DOWN 2                   // the block's warps: 2 down x 4 across
#define MMA_MATMUL_BF16_WARPS_ACROSS 4
#define MMA_MATMUL_BF16_THREADS_PER_BLOCK (MMA_MATMUL_BF16_WARPS_DOWN * MMA_MATMUL_BF16_WARPS_ACROSS * HARDWARE_THREADS_PER_WARP)   // 256
#define MMA_MATMUL_BF16_WARP_ROWS (MMA_MATMUL_BF16_BLOCK_ROWS / MMA_MATMUL_BF16_WARPS_DOWN)           // rows of a warp's part: 64
#define MMA_MATMUL_BF16_WARP_COLUMNS (MMA_MATMUL_BF16_BLOCK_COLUMNS / MMA_MATMUL_BF16_WARPS_ACROSS)   // columns of it: 32
#define MMA_MATMUL_BF16_M_TILES (MMA_MATMUL_BF16_WARP_ROWS / HARDWARE_MMA_M)        // mma tiles down a warp's part: 4
#define MMA_MATMUL_BF16_N_TILES (MMA_MATMUL_BF16_WARP_COLUMNS / HARDWARE_MMA_N)     // ... across it: 4 (16 C tiles, 64 sums a thread)
#define MMA_MATMUL_BF16_K_STEPS (MMA_MATMUL_BF16_BLOCK_K / HARDWARE_MMA_K_BF16)     // mma steps of k a step: 2
// The copies: 16-byte loads; a tile row has one spare load, so the rows' reads fall in different banks.
#define MMA_MATMUL_BF16_LOAD_VALUES ((int)(sizeof(uint4) / sizeof(nv_bfloat16)))    // bfloat16s per load: 8
#define MMA_MATMUL_BF16_K_ROW_LENGTH (MMA_MATMUL_BF16_BLOCK_K + MMA_MATMUL_BF16_LOAD_VALUES)         // a tile row as it sits {line, k}: 40
#define MMA_MATMUL_BF16_LINE_ROW_LENGTH (MMA_MATMUL_BF16_BLOCK_ROWS + MMA_MATMUL_BF16_LOAD_VALUES)   // a tile row turned {k, line}: 136
#define MMA_MATMUL_BF16_TILE_VALUES (MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_BF16_K_ROW_LENGTH)      // room for one tile either way: 5,120
#define MMA_MATMUL_BF16_LOADS_PER_THREAD (MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_BF16_BLOCK_K / MMA_MATMUL_BF16_LOAD_VALUES / MMA_MATMUL_BF16_THREADS_PER_BLOCK)   // 2
// The blocks are numbered in groups of 8 rows of squares, down first, as fma_matmul_bf16.cuh's (L2).
#define MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP 8
static_assert(MMA_MATMUL_BF16_BLOCK_ROWS == MMA_MATMUL_BF16_BLOCK_COLUMNS, "the copies are written for a square");
static_assert(MMA_MATMUL_BF16_TILE_VALUES >= MMA_MATMUL_BF16_BLOCK_K * MMA_MATMUL_BF16_LINE_ROW_LENGTH, "a turned tile fits too");
static_assert(MMA_MATMUL_BF16_LOADS_PER_THREAD * MMA_MATMUL_BF16_THREADS_PER_BLOCK * MMA_MATMUL_BF16_LOAD_VALUES ==
              MMA_MATMUL_BF16_BLOCK_ROWS * MMA_MATMUL_BF16_BLOCK_K, "the threads' loads cover a tile exactly");
static_assert(MMA_MATMUL_BF16_BLOCK_K % HARDWARE_MMA_K_BF16 == 0, "each step is whole mma steps");

// ── the product ───────────────────────────────────────────────────────────────
void mma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                     const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                     bool right_turned_val = false);

// ── for it ────────────────────────────────────────────────────────────────────
void mma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                  const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                  size_t* columns_ptr, size_t* sum_length_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

template <bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(MMA_MATMUL_BF16_THREADS_PER_BLOCK)
mma_matmul_bf16_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr, const nv_bfloat16* bias_ptr,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, bool add_onto_val);
__device__ __forceinline__ void mma_matmul_bf16_block_square(size_t rows_val, size_t columns_val, size_t* block_first_row_index_ptr,
                                                             size_t* block_first_column_index_ptr);
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_tile(uint4 (&loads_list)[MMA_MATMUL_BF16_LOADS_PER_THREAD], const nv_bfloat16* matrix_ptr,
                                                          size_t lines_val, size_t sum_length_val, size_t block_first_line_index_val,
                                                          size_t step_first_k_index_val);
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_store_tile(nv_bfloat16* tile_ptr, const uint4 (&loads_list)[MMA_MATMUL_BF16_LOADS_PER_THREAD]);
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val);
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val);
__device__ __forceinline__ void mma_matmul_bf16_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_BF16],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16]);
__device__ __forceinline__ uint32_t mma_matmul_bf16_pack(nv_bfloat16 low_val, nv_bfloat16 high_val);

/// Writes output = left · rightᵀ (+ bias) (+ output).
///
///   output_tensor      {..., columns}, rows x columns values: written over, or added onto
///   left_tensor        {..., sum_length} (rows = the other dims), or turned {..., rows} (sum_length = the other dims)
///   right_tensor       {..., sum_length} (columns = the other dims), or turned {..., columns}
///   bias_tensor        {columns}, or Tensor() for none
///   add_onto_val       true: added onto what output holds; false: written over
///   left_turned_val, right_turned_val   true: that tensor holds its matrix turned round
inline void mma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                            const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val) {
    // Checks, then launches mma_matmul_bf16_kernel made for the two turned flags, one block per square of output,
    // as one line of blocks.
    //
    // All bfloat16; every tensor's last dim a multiple of MMA_MATMUL_BF16_LOAD_VALUES (8) and its start on 16 bytes.
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    mma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                 &rows_val, &columns_val, &sum_length_val);
    // 2. launch: squares across and down, rounded up, as one line of blocks
    unsigned blocks_val = (unsigned)(CEIL_DIV(rows_val, (size_t)MMA_MATMUL_BF16_BLOCK_ROWS) *
                                     CEIL_DIV(columns_val, (size_t)MMA_MATMUL_BF16_BLOCK_COLUMNS));
    nv_bfloat16* output_ptr = output_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* left_ptr = left_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* right_ptr = right_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;
    if (!left_turned_val && !right_turned_val)
        mma_matmul_bf16_kernel<false, false><<<blocks_val, MMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else if (!left_turned_val && right_turned_val)
        mma_matmul_bf16_kernel<false, true><<<blocks_val, MMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else if (left_turned_val && !right_turned_val)
        mma_matmul_bf16_kernel<true, false><<<blocks_val, MMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else
        mma_matmul_bf16_kernel<true, true><<<blocks_val, MMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output.
///
///   output_ptr, left_ptr, right_ptr, bias_ptr    as in mma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes
///   add_onto_val                                 as in mma_matmul_bf16
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(MMA_MATMUL_BF16_THREADS_PER_BLOCK)
mma_matmul_bf16_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* left_ptr, const nv_bfloat16* right_ptr, const nv_bfloat16* bias_ptr,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, bool add_onto_val) {
    // Step by step the threads copy a tile of left and of right into shared memory, then each warp adds its part
    // of their product into its sums on the tensor cores.
    //
    // LEFT_TURNED, RIGHT_TURNED: how left and right sit, so each copy and each register read is made for its layout.
    // this step's tiles, two sets: this step reads one while the next step's loads wait in registers
    __shared__ __align__(16) nv_bfloat16 left_tiles_list[2][MMA_MATMUL_BF16_TILE_VALUES];
    __shared__ __align__(16) nv_bfloat16 right_tiles_list[2][MMA_MATMUL_BF16_TILE_VALUES];
    // 1. the block's square, and this warp's part of it
    size_t block_first_row_index_val = 0, block_first_column_index_val = 0;
    mma_matmul_bf16_block_square(rows_val, columns_val, &block_first_row_index_val, &block_first_column_index_val);
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_first_row_index_val = (warp_index_val / MMA_MATMUL_BF16_WARPS_ACROSS) * MMA_MATMUL_BF16_WARP_ROWS;
    int warp_first_column_index_val = (warp_index_val % MMA_MATMUL_BF16_WARPS_ACROSS) * MMA_MATMUL_BF16_WARP_COLUMNS;

    // 2. walk the sum: copy each step's tiles, read the next step's, run the mmas
    float sums_list[MMA_MATMUL_BF16_M_TILES][MMA_MATMUL_BF16_N_TILES][HARDWARE_MMA_C_VALUES_PER_THREAD_BF16] = {};
    uint4 left_loads_list[MMA_MATMUL_BF16_LOADS_PER_THREAD];
    uint4 right_loads_list[MMA_MATMUL_BF16_LOADS_PER_THREAD];
    mma_matmul_bf16_load_tile<LEFT_TURNED>(left_loads_list, left_ptr, rows_val, sum_length_val, block_first_row_index_val, 0);
    mma_matmul_bf16_load_tile<RIGHT_TURNED>(right_loads_list, right_ptr, columns_val, sum_length_val, block_first_column_index_val, 0);
    size_t total_steps_val = CEIL_DIV(sum_length_val, (size_t)MMA_MATMUL_BF16_BLOCK_K);
    for (size_t step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. this step's tiles into shared memory; the next step's loads started
        int tile_set_index_val = (int)(step_index_val % 2);
        mma_matmul_bf16_store_tile<LEFT_TURNED>(left_tiles_list[tile_set_index_val], left_loads_list);
        mma_matmul_bf16_store_tile<RIGHT_TURNED>(right_tiles_list[tile_set_index_val], right_loads_list);
        __syncthreads();                                        // the tiles whole, and the other set free
        size_t next_step_first_k_index_val = (step_index_val + 1) * MMA_MATMUL_BF16_BLOCK_K;
        if (next_step_first_k_index_val < sum_length_val) {
            mma_matmul_bf16_load_tile<LEFT_TURNED>(left_loads_list, left_ptr, rows_val, sum_length_val, block_first_row_index_val,
                                                   next_step_first_k_index_val);
            mma_matmul_bf16_load_tile<RIGHT_TURNED>(right_loads_list, right_ptr, columns_val, sum_length_val,
                                                    block_first_column_index_val, next_step_first_k_index_val);
        }
        // b. the tensor cores: for each 16 values of k, this warp's A and B registers, then every A x B
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < MMA_MATMUL_BF16_K_STEPS; k_step_index_val++) {
            int first_k_index_val = k_step_index_val * HARDWARE_MMA_K_BF16;
            uint32_t a_list[MMA_MATMUL_BF16_M_TILES][HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16];
            uint32_t b_list[MMA_MATMUL_BF16_N_TILES][HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16];
#pragma unroll
            for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++)
                mma_matmul_bf16_load_a<LEFT_TURNED>(a_list[m_tile_index_val], left_tiles_list[tile_set_index_val],
                                                    warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M, first_k_index_val);
#pragma unroll
            for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++)
                mma_matmul_bf16_load_b<RIGHT_TURNED>(b_list[n_tile_index_val], right_tiles_list[tile_set_index_val],
                                                     warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N, first_k_index_val);
#pragma unroll
            for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++)
#pragma unroll
                for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++)
                    mma_matmul_bf16_multiply_add(sums_list[m_tile_index_val][n_tile_index_val], a_list[m_tile_index_val],
                                                 b_list[n_tile_index_val]);
        }
    }

    // 3. output = sums (+ bias) (+ output), two neighbouring columns at a time; nothing past the edges. Each C tile:
    //    rows group and group + 8, columns pair and pair + 1 (the mma layout)
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
#pragma unroll
    for (int m_tile_index_val = 0; m_tile_index_val < MMA_MATMUL_BF16_M_TILES; m_tile_index_val++) {
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_MMA_ROWS_PER_THREAD; thread_row_index_val++) {
            size_t row_index_val = block_first_row_index_val + warp_first_row_index_val + m_tile_index_val * HARDWARE_MMA_M +
                                   thread_row_index_val * HARDWARE_MMA_ROWS_PER_WARP + group_index_val;
            if (row_index_val >= rows_val) continue;
#pragma unroll
            for (int n_tile_index_val = 0; n_tile_index_val < MMA_MATMUL_BF16_N_TILES; n_tile_index_val++) {
                size_t column_index_val = block_first_column_index_val + warp_first_column_index_val + n_tile_index_val * HARDWARE_MMA_N +
                                          pair_column_index_val;
                if (column_index_val >= columns_val) continue;   // columns is even, so column + 1 is in too
                int c_index_val = HARDWARE_MMA_COLUMNS_PER_THREAD * thread_row_index_val;   // this row's 2 sums in the C tile
                float2 bias_val = bias_ptr != nullptr ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                                      : make_float2(0.0f, 0.0f);
                float2 result_val = make_float2(sums_list[m_tile_index_val][n_tile_index_val][c_index_val] + bias_val.x, sums_list[m_tile_index_val][n_tile_index_val][c_index_val + 1] + bias_val.y);
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

/// Gives the block's square from its number.
///
///   rows_val, columns_val                                     the product's sizes
///   block_first_row_index_ptr, block_first_column_index_ptr   written: the square's first row and column
__device__ __forceinline__ void mma_matmul_bf16_block_square(size_t rows_val, size_t columns_val, size_t* block_first_row_index_ptr,
                                                             size_t* block_first_column_index_ptr) {
    // Groups of MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP rows of squares, down a group first, so the blocks running at
    // once share bands of left and right in L2.
    size_t total_block_columns_val = CEIL_DIV(columns_val, (size_t)MMA_MATMUL_BF16_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)MMA_MATMUL_BF16_BLOCK_ROWS);
    size_t blocks_per_group_val = MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP * total_block_columns_val;
    size_t group_index_val = blockIdx.x / blocks_per_group_val;
    size_t group_first_block_row_index_val = group_index_val * MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP;
    size_t group_block_rows_val = total_block_rows_val - group_first_block_row_index_val;   // fewer in the last group
    if (group_block_rows_val > MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP) group_block_rows_val = MMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP;
    size_t block_index_in_group_val = blockIdx.x % blocks_per_group_val;
    *block_first_row_index_ptr = (group_first_block_row_index_val + block_index_in_group_val % group_block_rows_val) * MMA_MATMUL_BF16_BLOCK_ROWS;
    *block_first_column_index_ptr = (block_index_in_group_val / group_block_rows_val) * MMA_MATMUL_BF16_BLOCK_COLUMNS;
}

/// Reads this thread's 16-byte loads of one step's tile of left or right from GPU memory.
///
///   loads_list                     written: this thread's MMA_MATMUL_BF16_LOADS_PER_THREAD loads
///   matrix_ptr                     left or right
///   lines_val, sum_length_val      its matrix's sizes
///   block_first_line_index_val     the block's first line
///   step_first_k_index_val         the step's first k
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_tile(uint4 (&loads_list)[MMA_MATMUL_BF16_LOADS_PER_THREAD], const nv_bfloat16* matrix_ptr,
                                                          size_t lines_val, size_t sum_length_val, size_t block_first_line_index_val,
                                                          size_t step_first_k_index_val) {
    // Reads this thread's 16-byte loads of one step's tile of left or right from GPU memory; 0s past the matrix's
    // edges (a load is all in or all out: the last dim is a multiple of MMA_MATMUL_BF16_LOAD_VALUES). Each line (a
    // row of left, a column of output for right) has sum_length values of k.
    //     not TURNED: the matrix sits as {lines, sum_length}: a load is 8 neighbouring k of one line
    //     TURNED:     it sits as {sum_length, lines}: a load is 8 neighbouring lines at one k
#pragma unroll
    for (int load_index_val = 0; load_index_val < MMA_MATMUL_BF16_LOADS_PER_THREAD; load_index_val++) {
        int piece_index_val = threadIdx.x + load_index_val * MMA_MATMUL_BF16_THREADS_PER_BLOCK;
        size_t line_index_val, k_index_val;
        if constexpr (TURNED) {
            const int loads_per_k_val = MMA_MATMUL_BF16_BLOCK_ROWS / MMA_MATMUL_BF16_LOAD_VALUES;
            k_index_val = step_first_k_index_val + piece_index_val / loads_per_k_val;
            line_index_val = block_first_line_index_val + (piece_index_val % loads_per_k_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        } else {
            const int loads_per_line_val = MMA_MATMUL_BF16_BLOCK_K / MMA_MATMUL_BF16_LOAD_VALUES;
            line_index_val = block_first_line_index_val + piece_index_val / loads_per_line_val;
            k_index_val = step_first_k_index_val + (piece_index_val % loads_per_line_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        }
        size_t offset_val = TURNED ? k_index_val * lines_val + line_index_val : line_index_val * sum_length_val + k_index_val;
        loads_list[load_index_val] = line_index_val < lines_val && k_index_val < sum_length_val
                                         ? *reinterpret_cast<const uint4*>(matrix_ptr + offset_val)
                                         : make_uint4(0, 0, 0, 0);
    }
}

/// Writes mma_matmul_bf16_load_tile's loads into a tile in shared memory, as they sat.
///
///   tile_ptr     one set of the shared memory tiles
///   loads_list   what mma_matmul_bf16_load_tile gave
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_store_tile(nv_bfloat16* tile_ptr, const uint4 (&loads_list)[MMA_MATMUL_BF16_LOADS_PER_THREAD]) {
    // Not TURNED tile[line][k] (rows MMA_MATMUL_BF16_K_ROW_LENGTH apart), TURNED tile[k][line] (rows
    // MMA_MATMUL_BF16_LINE_ROW_LENGTH apart).
#pragma unroll
    for (int load_index_val = 0; load_index_val < MMA_MATMUL_BF16_LOADS_PER_THREAD; load_index_val++) {
        int piece_index_val = threadIdx.x + load_index_val * MMA_MATMUL_BF16_THREADS_PER_BLOCK;
        int offset_val;
        if constexpr (TURNED) {
            const int loads_per_k_val = MMA_MATMUL_BF16_BLOCK_ROWS / MMA_MATMUL_BF16_LOAD_VALUES;
            offset_val = (piece_index_val / loads_per_k_val) * MMA_MATMUL_BF16_LINE_ROW_LENGTH +
                         (piece_index_val % loads_per_k_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        } else {
            const int loads_per_line_val = MMA_MATMUL_BF16_BLOCK_K / MMA_MATMUL_BF16_LOAD_VALUES;
            offset_val = (piece_index_val / loads_per_line_val) * MMA_MATMUL_BF16_K_ROW_LENGTH +
                         (piece_index_val % loads_per_line_val) * MMA_MATMUL_BF16_LOAD_VALUES;
        }
        *reinterpret_cast<uint4*>(tile_ptr + offset_val) = loads_list[load_index_val];
    }
}

/// Reads this thread's A registers of one mma tile out of left's tile in shared memory.
///
///   a_list                                    written: HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16 registers
///   tile_ptr                                  left's tile
///   first_row_index_val, first_k_index_val    the mma tile's {HARDWARE_MMA_M, HARDWARE_MMA_K_BF16} corner in the tile
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_a(uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_row_index_val, int first_k_index_val) {
    // Reads this thread's A registers of one mma tile out of left's tile in shared memory. With group = lane / 4 and
    // pair = 2 (lane % 4), the registers hold A's (group, pair .. pair+1), (group+8, ..), (group, pair+8 ..),
    // (group+8, pair+8 ..), each 2 neighbouring k.
    //     not TURNED: tile[row][k]: the 2 values sit side by side, one 4-byte read
    //     TURNED:     tile[k][row]: they sit a tile row apart, two reads packed into one register
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
    const int half_k_val = HARDWARE_MMA_K_BF16 / 2;               // registers 2 and 3: 8 values of k on
    if constexpr (TURNED) {
        const int row_val = MMA_MATMUL_BF16_LINE_ROW_LENGTH;      // from one k to the next
        const nv_bfloat16* at_ptr = tile_ptr + (first_k_index_val + pair_index_val) * row_val + first_row_index_val + group_index_val;
        a_list[0] = mma_matmul_bf16_pack(at_ptr[0], at_ptr[row_val]);
        a_list[1] = mma_matmul_bf16_pack(at_ptr[HARDWARE_MMA_ROWS_PER_WARP], at_ptr[row_val + HARDWARE_MMA_ROWS_PER_WARP]);
        a_list[2] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val], at_ptr[(half_k_val + 1) * row_val]);
        a_list[3] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val + HARDWARE_MMA_ROWS_PER_WARP],
                                         at_ptr[(half_k_val + 1) * row_val + HARDWARE_MMA_ROWS_PER_WARP]);
    } else {
        const nv_bfloat16* top_ptr = tile_ptr + (first_row_index_val + group_index_val) * MMA_MATMUL_BF16_K_ROW_LENGTH + first_k_index_val + pair_index_val;
        const nv_bfloat16* bottom_ptr = top_ptr + HARDWARE_MMA_ROWS_PER_WARP * MMA_MATMUL_BF16_K_ROW_LENGTH;
        a_list[0] = *reinterpret_cast<const uint32_t*>(top_ptr);
        a_list[1] = *reinterpret_cast<const uint32_t*>(bottom_ptr);
        a_list[2] = *reinterpret_cast<const uint32_t*>(top_ptr + half_k_val);
        a_list[3] = *reinterpret_cast<const uint32_t*>(bottom_ptr + half_k_val);
    }
}

/// Reads this thread's B registers of one mma tile out of right's tile in shared memory.
///
///   b_list                                       written: HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16 registers
///   tile_ptr                                     right's tile
///   first_column_index_val, first_k_index_val    the mma tile's first column (n) and first k in the tile
template <bool TURNED>
__device__ __forceinline__ void mma_matmul_bf16_load_b(uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16], const nv_bfloat16* tile_ptr,
                                                       int first_column_index_val, int first_k_index_val) {
    // Reads this thread's B registers of one mma tile out of right's tile in shared memory: B's rows (pair .. pair+1,
    // column group) and (pair+8 .. pair+9, column group), B[k][n] = right[n][k].
    //     not TURNED: tile[column][k]: the 2 values sit side by side, one 4-byte read
    //     TURNED:     tile[k][column]: they sit a tile row apart, two reads packed into one register
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_MMA_THREADS_PER_ROW;
    int pair_index_val = (lane_index_val % HARDWARE_MMA_THREADS_PER_ROW) * HARDWARE_MMA_COLUMNS_PER_THREAD;
    const int half_k_val = HARDWARE_MMA_K_BF16 / 2;
    if constexpr (TURNED) {
        const int row_val = MMA_MATMUL_BF16_LINE_ROW_LENGTH;
        const nv_bfloat16* at_ptr = tile_ptr + (first_k_index_val + pair_index_val) * row_val + first_column_index_val + group_index_val;
        b_list[0] = mma_matmul_bf16_pack(at_ptr[0], at_ptr[row_val]);
        b_list[1] = mma_matmul_bf16_pack(at_ptr[half_k_val * row_val], at_ptr[(half_k_val + 1) * row_val]);
    } else {
        const nv_bfloat16* at_ptr = tile_ptr + (first_column_index_val + group_index_val) * MMA_MATMUL_BF16_K_ROW_LENGTH + first_k_index_val + pair_index_val;
        b_list[0] = *reinterpret_cast<const uint32_t*>(at_ptr);
        b_list[1] = *reinterpret_cast<const uint32_t*>(at_ptr + half_k_val);
    }
}

/// C += A · B on the tensor cores, for the whole warp: mma.sync m16n8k16, bfloat16 in, float sums.
///
///   c_list   this thread's HARDWARE_MMA_C_VALUES_PER_THREAD_BF16 floats of C
///   a_list   this thread's A registers
///   b_list   this thread's B registers
__device__ __forceinline__ void mma_matmul_bf16_multiply_add(float (&c_list)[HARDWARE_MMA_C_VALUES_PER_THREAD_BF16],
                                                             const uint32_t (&a_list)[HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16],
                                                             const uint32_t (&b_list)[HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c_list[0]), "+f"(c_list[1]), "+f"(c_list[2]), "+f"(c_list[3])
                 : "r"(a_list[0]), "r"(a_list[1]), "r"(a_list[2]), "r"(a_list[3]), "r"(b_list[0]), "r"(b_list[1]));
#endif
}

/// Gives two bfloat16s as one register, the first in the low half.
///
///   low_val, high_val   the two values
__device__ __forceinline__ uint32_t mma_matmul_bf16_pack(nv_bfloat16 low_val, nv_bfloat16 high_val) {
    return (uint32_t)__bfloat16_as_ushort(low_val) | ((uint32_t)__bfloat16_as_ushort(high_val) << 16);
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit, and gives the product's sizes.
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val   as in mma_matmul_bf16
///   rows_ptr, columns_ptr, sum_length_ptr                                                       written: the sizes
inline void mma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                         const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                         size_t* columns_ptr, size_t* sum_length_ptr) {
    // Left's and right's sum_length the same, output rows x columns with columns last, bias {columns}; all
    // bfloat16; every last dim a multiple of MMA_MATMUL_BF16_LOAD_VALUES and every tensor starting on 16 bytes
    // (the 16-byte loads).
    // 1. shapes: each matrix's last dim and the rest, by how it sits
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = left_tensor.dim() >= 1 && right_tensor.dim() >= 1 && output_tensor.dim() >= 1;
    size_t left_last_val = fits_val ? left_tensor.size(left_tensor.dim() - 1) : 1;
    size_t right_last_val = fits_val ? right_tensor.size(right_tensor.dim() - 1) : 1;
    size_t rows_val = left_turned_val ? left_last_val : left_tensor.numel() / left_last_val;
    size_t sum_length_val = left_turned_val ? left_tensor.numel() / left_last_val : left_last_val;
    size_t columns_val = right_turned_val ? right_last_val : right_tensor.numel() / right_last_val;
    size_t right_sum_length_val = right_turned_val ? right_tensor.numel() / right_last_val : right_last_val;
    fits_val = fits_val && sum_length_val == right_sum_length_val && output_tensor.size(output_tensor.dim() - 1) == columns_val &&
               output_tensor.numel() == rows_val * columns_val && (!has_bias_val || bias_tensor.shape() == Shape({columns_val}));
    if (!fits_val) {
        fprintf(stderr, "error: mma_matmul_bf16 needs left {rows, sum_length} (turned {sum_length, rows}), right {columns, "
                        "sum_length} (turned {sum_length, columns}), output {rows, columns}, bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16
    if (left_tensor.dtype() != DType::BFloat16 || right_tensor.dtype() != DType::BFloat16 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: mma_matmul_bf16 needs output, left, right and bias all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: whole 16-byte loads, each tensor starting on one
    const size_t load_bytes_val = sizeof(uint4);
    if (left_last_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 || right_last_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 ||
        columns_val % MMA_MATMUL_BF16_LOAD_VALUES != 0 || (size_t)output_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)left_tensor.data_ptr() % load_bytes_val != 0 || (size_t)right_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)bias_tensor.data_ptr() % load_bytes_val != 0) {
        fprintf(stderr, "error: mma_matmul_bf16 reads %zu bytes at a time, so every tensor's last dim must be a multiple of %d "
                        "and every tensor must start on %zu bytes; got last dims %zu (left), %zu (right), %zu (output)\n",
                load_bytes_val, MMA_MATMUL_BF16_LOAD_VALUES, load_bytes_val, left_last_val, right_last_val, columns_val);
        exit(EXIT_FAILURE);
    }
    *rows_ptr = rows_val;
    *columns_ptr = columns_val;
    *sum_length_ptr = sum_length_val;
}

#endif // MMA_MATMUL_BF16_CUH
