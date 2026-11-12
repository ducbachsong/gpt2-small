// wgmma_matmul_bf16.cuh — matrix products in bfloat16 on an H100's tensor cores, fed by TMA:
//     output = left · rightᵀ (+ bias) (+ output)
// Sums in float; each value rounded to bfloat16 only as it is stored. Built after Pranjal Shankhdhar's
// "Outperforming cuBLAS on H100: a Worklog", one of his steps at a time.
//
//     tensor   type in memory   shape                                               read with
//     left     bfloat16         {rows, sum_length}, turned {sum_length, rows}       TMA, a tile at a time, into shared memory
//     right    bfloat16         {columns, sum_length}, turned {sum_length, columns} TMA, the same
//     bias     bfloat16         {columns}, or none                                  plain loads, as the sums are stored
//     output   bfloat16         {rows, columns}                                     written over, or added onto
//     sums     float            in registers                                        the tensor cores' C
//
// turned: the tensor holds its matrix turned round (a linear layer's backward; see fma_matmul_bf16.cuh's table).
// A tile is then {64 k, 64 lines} in shared memory, lines next to each other ("MN-major") where it is usually
// {64 lines, 64 k} ("K-major"); the WGMMA's transpose bit for that side says so, and the tensor cores read it
// turned. Both tiles are 64 x 64, so TMA's box is the same shape either way.
//
// Step 1 (his kernel 2, 317 TFLOPS on an H100 SXM at 4096³): each block makes one HARDWARE_WGMMA_M x
// WGMMA_MATMUL_BF16_BLOCK_COLUMNS square of output with one warpgroup (4 warps, 128 threads). It walks the sum
// WGMMA_MATMUL_BF16_BLOCK_K at a time:
//     1. one thread asks TMA for this step's tile of left {64 rows, 64 k} and of right {64 columns, 64 k}; TMA
//        copies them into shared memory in the swizzled layout the tensor cores read, and a barrier tells
//        every thread when both have arrived
//     2. the warpgroup's WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs (m64n64k16): sums {64, 64} += left tile ·
//        right tileᵀ, read straight from shared memory
//     3. wait for them, then let TMA fill the tiles again
// The epilogue: what is done to each pair of results (two neighbouring columns of a row) as they are stored, after
// the bias and before adding onto output. A functor, a template parameter of the kernel, so it costs nothing when it
// does nothing: WgmmaMatmulBf16Store (the default) gives the pair back as it is; bf16/gelu_bf16.cuh's apply GELU
// there, so the MLP's GELU needs no pass of its own over memory. An epilogue is
//     struct { ...its pointers...; __device__ float2 operator()(float2 pair, size_t index) const; }
// called with the pair in float and the index of its first value in output (row x columns + column); what it gives
// back is stored (rounded to bfloat16), and it may read or write other tensors of output's shape at that index.
// WGMMA and TMA exist only on Hopper: run it on an H100 (compute capability 9.0), built with -arch=sm_90a; nothing
// here checks. Other GPUs: mma_matmul_bf16.cuh (sm_80 or later), fma_matmul_bf16.cuh (any). The block size is
// fixed by the tiles, so this takes none.
#ifndef WGMMA_MATMUL_BF16_CUH
#define WGMMA_MATMUL_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"

// The squares and steps: our choices. The hardware's numbers (WARPGROUP_*, WGMMA_*, TMA_*) are in cuda_common.h.
// The index math is written for these numbers: change them together with it.
#define WGMMA_MATMUL_BF16_BLOCK_COLUMNS 64                                                        // columns of output per block: one WGMMA's N (any multiple of 8 to 256)
#define WGMMA_MATMUL_BF16_BLOCK_K ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))       // values of k a step loads: one swizzled row, 64
#define WGMMA_MATMUL_BF16_COLUMN_GROUPS (WGMMA_MATMUL_BF16_BLOCK_COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS)   // 8 groups across: 32 sums a thread
#define WGMMA_MATMUL_BF16_ROW_VALUES ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))    // a TMA box row: one swizzled row, 64 values
#define WGMMA_MATMUL_BF16_VALUE_MULTIPLE ((int)(HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16)))       // a matrix row's multiple: whole 16-byte cells, 8
static_assert(WGMMA_MATMUL_BF16_BLOCK_COLUMNS == 64, "wgmma_matmul_bf16_multiply_add is written for m64n64k16: its 32 sums a thread");
static_assert(WGMMA_MATMUL_BF16_BLOCK_K % HARDWARE_WGMMA_K_BF16 == 0, "each step is whole WGMMAs");
static_assert(HARDWARE_WGMMA_M == WGMMA_MATMUL_BF16_ROW_VALUES && WGMMA_MATMUL_BF16_BLOCK_COLUMNS == WGMMA_MATMUL_BF16_ROW_VALUES,
              "a turned tile's row (64 rows or columns) is one swizzled row, as a tile's row of 64 k is");

/// The plain epilogue: each pair of results is stored as it is.
struct WgmmaMatmulBf16Store {
    /// Gives the pair back unchanged.
    ///
    ///   pair_val    two neighbouring results, in float
    ///   index_val   the first one's index in output (unused)
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const { return pair_val; }
};

// ── the product ───────────────────────────────────────────────────────────────
template <typename EPILOGUE = WgmmaMatmulBf16Store>
void wgmma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                       const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                       bool right_turned_val = false, const EPILOGUE& epilogue_val = EPILOGUE());

// ── for it ────────────────────────────────────────────────────────────────────
CUtensorMap wgmma_matmul_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                         unsigned box_rows_val);
void wgmma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                    const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                    size_t* columns_ptr, size_t* sum_length_ptr);


// ── the product ───────────────────────────────────────────────────────────────

using WgmmaMatmulBf16Barrier = cuda::barrier<cuda::thread_scope_block>;

template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE = WgmmaMatmulBf16Store>
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, int rows_val, int columns_val,
                         int sum_length_val, bool add_onto_val, EPILOGUE epilogue_val);
__device__ __forceinline__ uint64_t wgmma_matmul_bf16_descriptor(const nv_bfloat16* tile_ptr);
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__device__ __forceinline__ void wgmma_matmul_bf16_multiply_add(float (&sums_list)[WGMMA_MATMUL_BF16_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                               const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr);

/// Writes output = left · rightᵀ (+ bias) (+ output): checks, describes left's and right's tiles for TMA, then
/// launches wgmma_matmul_bf16_kernel made for the two turned flags, one block per square of output, as one line of
/// blocks.
///
///   output_tensor      {..., columns}, rows x columns values: written over, or added onto
///   left_tensor        {..., sum_length} (rows = the other dims): x in a linear layer; or turned {..., rows}
///                      (sum_length = the other dims)
///   right_tensor       {..., sum_length} (columns = the other dims): W; or turned {..., columns}
///   bias_tensor        {columns}, or Tensor() for none
///   add_onto_val       true: added onto what output holds; false: written over
///   left_turned_val, right_turned_val   true: that tensor holds its matrix turned round
///   epilogue_val       what is done to each pair of results as it is stored (see the top of the file); not given:
///                      WgmmaMatmulBf16Store, nothing
/// Every tensor's last dim a multiple of 8 (TMA reads rows of whole 16-byte pieces); every tensor bfloat16.
template <typename EPILOGUE>
inline void wgmma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                              const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                              const EPILOGUE& epilogue_val) {
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    wgmma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                   &rows_val, &columns_val, &sum_length_val);

    // 2. TMA's description of left's and right's tiles: {lines, sum_length} in boxes of {64 lines, 64 k}, or
    //    turned {sum_length, lines} in boxes of {64 k, 64 lines}
    const nv_bfloat16* left_ptr = left_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* right_ptr = right_tensor.data_ptr<nv_bfloat16>();
    CUtensorMap left_map_val = left_turned_val ? wgmma_matmul_bf16_tensor_map(left_ptr, sum_length_val, rows_val, WGMMA_MATMUL_BF16_BLOCK_K)
                                               : wgmma_matmul_bf16_tensor_map(left_ptr, rows_val, sum_length_val, HARDWARE_WGMMA_M);
    CUtensorMap right_map_val = right_turned_val ? wgmma_matmul_bf16_tensor_map(right_ptr, sum_length_val, columns_val, WGMMA_MATMUL_BF16_BLOCK_K)
                                                 : wgmma_matmul_bf16_tensor_map(right_ptr, columns_val, sum_length_val, WGMMA_MATMUL_BF16_BLOCK_COLUMNS);

    // 3. launch: squares across and down, rounded up, as one line of blocks
    size_t total_block_columns_val = CEIL_DIV(columns_val, (size_t)WGMMA_MATMUL_BF16_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)HARDWARE_WGMMA_M);
    unsigned total_blocks_val = (unsigned)(total_block_columns_val * total_block_rows_val);
    nv_bfloat16* output_ptr = output_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;
    if (!left_turned_val && !right_turned_val)
        wgmma_matmul_bf16_kernel<false, false, EPILOGUE><<<total_blocks_val, HARDWARE_WARPGROUP_THREADS>>>(
            left_map_val, right_map_val, output_ptr, bias_ptr, (int)rows_val, (int)columns_val, (int)sum_length_val, add_onto_val,
            epilogue_val);
    else if (!left_turned_val && right_turned_val)
        wgmma_matmul_bf16_kernel<false, true, EPILOGUE><<<total_blocks_val, HARDWARE_WARPGROUP_THREADS>>>(
            left_map_val, right_map_val, output_ptr, bias_ptr, (int)rows_val, (int)columns_val, (int)sum_length_val, add_onto_val,
            epilogue_val);
    else if (left_turned_val && !right_turned_val)
        wgmma_matmul_bf16_kernel<true, false, EPILOGUE><<<total_blocks_val, HARDWARE_WARPGROUP_THREADS>>>(
            left_map_val, right_map_val, output_ptr, bias_ptr, (int)rows_val, (int)columns_val, (int)sum_length_val, add_onto_val,
            epilogue_val);
    else
        wgmma_matmul_bf16_kernel<true, true, EPILOGUE><<<total_blocks_val, HARDWARE_WARPGROUP_THREADS>>>(
            left_map_val, right_map_val, output_ptr, bias_ptr, (int)rows_val, (int)columns_val, (int)sum_length_val, add_onto_val,
            epilogue_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output with one warpgroup: step by step, TMA brings a tile of left and of right into
/// shared memory, then the tensor cores add their product into the sums in registers.
///
///   left_map_val, right_map_val                  TMA's descriptions of left's and right's tiles
///   output_ptr, bias_ptr                         as in wgmma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the shapes
///   add_onto_val                                 as in wgmma_matmul_bf16
///   epilogue_val                                 as in wgmma_matmul_bf16
/// LEFT_TURNED, RIGHT_TURNED: how left and right sit: which corner TMA's box starts at, how far apart two WGMMAs'
/// 16 values of k are in a tile, and the WGMMA's transpose bits. EPILOGUE: the epilogue's type.
template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE>
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, int rows_val, int columns_val,
                         int sum_length_val, bool add_onto_val, EPILOGUE epilogue_val) {
    namespace cde = cuda::device::experimental;
    // this step's tiles, in TMA's swizzled layout (on a 1024-byte atom), and the barrier that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 left_tile_list[HARDWARE_WGMMA_M * WGMMA_MATMUL_BF16_BLOCK_K];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 right_tile_list[WGMMA_MATMUL_BF16_BLOCK_COLUMNS * WGMMA_MATMUL_BF16_BLOCK_K];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ WgmmaMatmulBf16Barrier tiles_barrier_val;
    // where a WGMMA's 16 values of k start in a tile: 16 values along a row, or turned 16 rows on (16 x 64 values)
    const int left_k_step_values_val = LEFT_TURNED ? HARDWARE_WGMMA_K_BF16 * HARDWARE_WGMMA_M : HARDWARE_WGMMA_K_BF16;
    const int right_k_step_values_val = RIGHT_TURNED ? HARDWARE_WGMMA_K_BF16 * WGMMA_MATMUL_BF16_BLOCK_COLUMNS : HARDWARE_WGMMA_K_BF16;

    // 1. the block's square, from its number: squares row after row; and the barrier, by one thread
    int total_block_columns_val = CEIL_DIV(columns_val, WGMMA_MATMUL_BF16_BLOCK_COLUMNS);
    int block_row_index_val = blockIdx.x / total_block_columns_val;
    int block_column_index_val = blockIdx.x % total_block_columns_val;
    int block_first_row_index_val = block_row_index_val * HARDWARE_WGMMA_M;
    int block_first_column_index_val = block_column_index_val * WGMMA_MATMUL_BF16_BLOCK_COLUMNS;
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                     // every thread arrives once a step
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();

    // 2. walk the sum: TMA brings the step's tiles, the tensor cores add their product
    float sums_list[WGMMA_MATMUL_BF16_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of the square
    int total_steps_val = CEIL_DIV(sum_length_val, WGMMA_MATMUL_BF16_BLOCK_K);
    for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. one thread asks for both tiles (past the edges TMA gives 0s) and says how many bytes to wait for;
        //    the others arrive; all wait until the bytes are in. A box's corner is (column, row) of the matrix as
        //    it sits: (k, line), or turned (line, k)
        WgmmaMatmulBf16Barrier::arrival_token token_val;
        int step_first_k_index_val = step_index_val * WGMMA_MATMUL_BF16_BLOCK_K;
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(left_tile_list, &left_map_val,
                                                          LEFT_TURNED ? block_first_row_index_val : step_first_k_index_val,
                                                          LEFT_TURNED ? step_first_k_index_val : block_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(right_tile_list, &right_map_val,
                                                          RIGHT_TURNED ? block_first_column_index_val : step_first_k_index_val,
                                                          RIGHT_TURNED ? step_first_k_index_val : block_first_column_index_val, tiles_barrier_val);
            token_val = cuda::device::barrier_arrive_tx(tiles_barrier_val, 1, sizeof(left_tile_list) + sizeof(right_tile_list));
        } else {
            token_val = tiles_barrier_val.arrive();
        }
        tiles_barrier_val.wait(std::move(token_val));

        // b. the tensor cores: WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs, each 16 more values of k
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16; k_step_index_val++)
            wgmma_matmul_bf16_multiply_add<LEFT_TURNED, RIGHT_TURNED>(sums_list, left_tile_list + k_step_index_val * left_k_step_values_val,
                                                                      right_tile_list + k_step_index_val * right_k_step_values_val);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // ... done

        // c. every thread done with the tiles before TMA fills them again
        __syncthreads();
    }

    // 3. output = epilogue(sums (+ bias)) (+ output), two neighbouring columns at a time; nothing past the edges
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int thread_first_row_index_val = block_first_row_index_val + warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
#pragma unroll
    for (int group_index_val = 0; group_index_val < WGMMA_MATMUL_BF16_COLUMN_GROUPS; group_index_val++) {
        int column_index_val = block_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
        if (column_index_val >= columns_val) continue;       // columns is even, so column + 1 is in too
        float2 bias_val = bias_ptr != nullptr ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                              : make_float2(0.0f, 0.0f);
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
            int row_index_val = thread_first_row_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
            if (row_index_val >= rows_val) continue;
            size_t output_index_val = (size_t)row_index_val * columns_val + column_index_val;
            nv_bfloat162* output_pair_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + output_index_val);
            float2 result_val = make_float2(sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] + bias_val.x,
                                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] + bias_val.y);
            result_val = epilogue_val(result_val, output_index_val);
            if (add_onto_val) {
                float2 old_val = __bfloat1622float2(*output_pair_ptr);
                result_val = make_float2(result_val.x + old_val.x, result_val.y + old_val.y);
            }
            *output_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
        }
    }
}

/// Gives the WGMMA descriptor of a tile in shared memory: where it starts and how its rows sit (rows of 128 bytes
/// in TMA's 128-byte swizzle, 8-row atoms HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM apart). The same numbers for both
/// layouts:
///     K-major (rows of k): the stride offset is from 8 lines to the next 8; moving 16 values of k along a row is
///                          32 bytes more on the start address (the swizzle is undone from the address)
///     MN-major (turned, rows of lines): the stride offset is from 8 values of k to the next 8; moving 16 values
///                          of k is 16 rows (2 atoms) more on the start address. One row holds all 64 lines, so the
///                          leading offset (from 64 lines to the next 64) is never used
///
///   tile_ptr   the tile's first value to read, in shared memory
__device__ __forceinline__ uint64_t wgmma_matmul_bf16_descriptor(const nv_bfloat16* tile_ptr) {
    uint64_t address_val = (uint64_t)__cvta_generic_to_shared(tile_ptr);
    uint64_t descriptor_val = ((address_val & HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS) / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT;
    descriptor_val |= HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 << HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT;
    return descriptor_val;
}

/// sums {64, 64} += left tile {64, 16} · right tile {64, 16}ᵀ on the tensor cores, for the whole warpgroup:
/// wgmma.mma_async m64n64k16, bfloat16 in, float sums, read straight from shared memory; asynchronous: done only
/// after wgmma.wait_group. LEFT_TURNED, RIGHT_TURNED: the transpose bits (0: K-major, 1: MN-major), which must be
/// numbers in the instruction itself ("n": written in as constants).
///
///   sums_list                       this thread's 32 sums, in WGMMA's layout
///   left_tile_ptr, right_tile_ptr   the 16 values of k to use, in the shared memory tiles
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__device__ __forceinline__ void wgmma_matmul_bf16_multiply_add(float (&sums_list)[WGMMA_MATMUL_BF16_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                               const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr) {
    uint64_t left_descriptor_val = wgmma_matmul_bf16_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = wgmma_matmul_bf16_descriptor(right_tile_ptr);
    // scale-d 1: add onto the sums; scale-a, scale-b 1; transpose-a, transpose-b: the turned flags
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, %35, %36;\n"
        "}\n"
        : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
          "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
          "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
          "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3])
        : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(1), "n"(LEFT_TURNED ? 1 : 0), "n"(RIGHT_TURNED ? 1 : 0));
}


/// Gives TMA's description of a matrix's tiles: a bfloat16 matrix {matrix_rows, matrix_columns} sitting row after
/// row, read in boxes of {box_rows, WGMMA_MATMUL_BF16_ROW_VALUES} written into shared memory in the 128-byte
/// swizzle; past the edges, 0s. Made by the driver's cuTensorMapEncodeTiled, found at run time, so the program
/// needs no -lcuda.
///
///   matrix_ptr                            its first value, on a 16-byte piece
///   matrix_rows_val, matrix_columns_val   its shape as it sits; matrix_columns a multiple of 8 (rows of whole 16-byte pieces)
///   box_rows_val                          rows of a tile: HARDWARE_WGMMA_M or WGMMA_MATMUL_BF16_BLOCK_COLUMNS (lines), or
///                                         turned WGMMA_MATMUL_BF16_BLOCK_K (values of k)
inline CUtensorMap wgmma_matmul_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                                unsigned box_rows_val) {
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function_ptr, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: wgmma_matmul_bf16 could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: columns, then rows
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {matrix_columns_val, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {matrix_columns_val * sizeof(nv_bfloat16)};            // from one row to the next
    cuuint32_t box_list[2] = {WGMMA_MATMUL_BF16_ROW_VALUES, box_rows_val};
    cuuint32_t element_steps_list[2] = {1, 1};                                            // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bfloat16*>(matrix_ptr),
                                     sizes_list, row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %zu} matrix at %p\n", (int)result_val,
                matrix_rows_val, matrix_columns_val, (const void*)matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Stops with a message unless the tensors fit, and gives the product's sizes: left's and right's sum_length the
/// same, output rows x columns with columns last, bias {columns}; all bfloat16; every last dim a multiple of 8 and
/// every tensor starting on a 16-byte piece.
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val   as in wgmma_matmul_bf16
///   rows_ptr, columns_ptr, sum_length_ptr                                                       written: the sizes
inline void wgmma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                           const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                           size_t* columns_ptr, size_t* sum_length_ptr) {
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
        fprintf(stderr, "error: wgmma_matmul_bf16 needs left {rows, sum_length} (turned {sum_length, rows}), right {columns, "
                        "sum_length} (turned {sum_length, columns}), output {rows, columns}, bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16
    if (left_tensor.dtype() != DType::BFloat16 || right_tensor.dtype() != DType::BFloat16 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: wgmma_matmul_bf16 needs output, left, right and bias all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte pieces, each tensor starting on one
    if (left_last_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 || right_last_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 ||
        columns_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 ||
        (size_t)output_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)left_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 ||
        (size_t)right_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)bias_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0) {
        fprintf(stderr, "error: wgmma_matmul_bf16's TMA reads rows of whole %d-byte pieces, so every tensor's last dim must be a "
                        "multiple of %d and every tensor must start on %d bytes; got last dims %zu (left), %zu (right), %zu (output)\n",
                HARDWARE_TMA_BYTES_PER_CELL, WGMMA_MATMUL_BF16_VALUE_MULTIPLE, HARDWARE_TMA_BYTES_PER_CELL, left_last_val, right_last_val,
                columns_val);
        exit(EXIT_FAILURE);
    }
    *rows_ptr = rows_val;
    *columns_ptr = columns_val;
    *sum_length_ptr = sum_length_val;
}

#endif // WGMMA_MATMUL_BF16_CUH
