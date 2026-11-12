// wgmma_matmul_8bit.cuh — matrix products in 8 bits (FP8 E4M3) on an H100's tensor cores, fed by TMA: wgmma_matmul_bf16.cuh's
// product with left and right rounded to E4M3 first (round_8bit.cuh), one float scale per row:
//     output[i][j] = left_scale[i] · right_scale[j] · sum over k of left_8bit[i][k] · right_8bit[j][k]
//                    (+ bias[j]) (+ output[i][j])
// A row's scale is the same for all its k, so it comes out of the sum: the tensor cores add up the E4M3
// products in float, and both scales are applied once, as each sum is stored. E4M3 has twice bfloat16's values
// per byte, so the tensor cores do twice the products a clock (wgmma m64n64k32 against bfloat16's k16).
//
//     tensor                    type in memory                   shape                              read with
//     left, right (bfloat16)    bfloat16                         {rows, sum_length}, {columns, ...}  round_8bit
//     left_8bit, right_8bit     E4M3, 4 in each int of an        {rows, sum_length / 4},             TMA, a tile at a time
//                               Int32 tensor                     {columns, sum_length / 4}
//     left_scales, right_scales float                            {rows}, {columns}                   as the sums are stored
//     bias                      bfloat16                         {columns}, or none                  the same
//     output                    bfloat16                         {rows, columns}                     written over, or added onto
//     sums                      float                            in registers                        the tensor cores' C
//
// FP8 WGMMA takes only K-major tiles (no transpose bit), so both copies sit k next to each other, as round_8bit
// writes them; a matrix that sits turned round (a linear layer's backward) is turned while rounding (round_8bit's
// turned), not here.
//
// Step 1 (wgmma_matmul_bf16.cuh's step 1, the worklog's kernel 2): each block makes one HARDWARE_WGMMA_M x
// WGMMA_MATMUL_8BIT_BLOCK_COLUMNS square of output with one warpgroup (128 threads), walking the sum WGMMA_MATMUL_8BIT_BLOCK_K
// (128) values at a time: one thread asks TMA for the step's tiles of left_8bit and right_8bit {64 rows, 128
// bytes} into shared memory, swizzled; a barrier says they arrived; 4 WGMMAs m64n64k32 add their product into the
// sums; wait, and again. The epilogue: what is done to each pair of results as they are stored, after the scales
// and the bias and before adding onto output, as in wgmma_matmul_bf16.cuh (the same functors: output is bfloat16 in
// both); WgmmaMatmul8bitStore, the default, does nothing. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a; nothing here checks. The block size of the product is fixed by the tiles. Other GPUs: mma_matmul_8bit.cuh (sm_89 or later),
// fma_matmul_8bit.cuh (any).
#ifndef WGMMA_MATMUL_8BIT_CUH
#define WGMMA_MATMUL_8BIT_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"
#include "round_8bit.cuh"            // the 8-bit copies: ROUND_8BIT_VALUES_PER_INT, ROUND_8BIT_ROW_LENGTH_MULTIPLE

// The squares and steps: our choices. The hardware's numbers (WARPGROUP_*, WGMMA_*, TMA_*) are in cuda_common.h.
// The index math is written for these numbers: change them together with it.
#define WGMMA_MATMUL_8BIT_BLOCK_COLUMNS 64                                                        // columns of output per block: one WGMMA's N (any multiple of 8 to 256)
#define WGMMA_MATMUL_8BIT_BLOCK_K ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(__nv_fp8_e4m3)))     // values of k a step loads: one swizzled row, 128
#define WGMMA_MATMUL_8BIT_COLUMN_GROUPS (WGMMA_MATMUL_8BIT_BLOCK_COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS)   // 8 groups across: 32 sums a thread
static_assert(WGMMA_MATMUL_8BIT_BLOCK_COLUMNS == 64, "wgmma_matmul_8bit_multiply_add is written for m64n64k32: its 32 sums a thread");
static_assert(WGMMA_MATMUL_8BIT_BLOCK_K % HARDWARE_WGMMA_K_E4M3 == 0, "each step is whole WGMMAs");
static_assert(ROUND_8BIT_ROW_LENGTH_MULTIPLE == HARDWARE_TMA_BYTES_PER_CELL, "round_8bit's rows are whole 16-byte pieces, as TMA reads them");

/// The plain epilogue: each pair of results is stored as it is.
struct WgmmaMatmul8bitStore {
    /// Gives the pair back unchanged.
    ///
    ///   pair_val    two neighbouring results, in float
    ///   index_val   the first one's index in output (unused)
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const { return pair_val; }
};

// ── the product ───────────────────────────────────────────────────────────────
template <typename EPILOGUE = WgmmaMatmul8bitStore>
void wgmma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                      const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor,
                      const Tensor& bias_tensor = Tensor(), bool add_onto_val = false,
                      const EPILOGUE& epilogue_val = EPILOGUE());

// ── for it ────────────────────────────────────────────────────────────────────
CUtensorMap wgmma_matmul_8bit_tensor_map(const void* matrix_ptr, size_t matrix_rows_val, size_t matrix_bytes_per_row_val,
                                unsigned box_rows_val);
void wgmma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                           const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

using WgmmaMatmul8bitBarrier = cuda::barrier<cuda::thread_scope_block>;

template <typename EPILOGUE = WgmmaMatmul8bitStore>
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_8bit_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                        nv_bfloat16* output_ptr, const float* left_scales_ptr, const float* right_scales_ptr,
                        const nv_bfloat16* bias_ptr, int rows_val, int columns_val, int sum_length_val, bool add_onto_val,
                        EPILOGUE epilogue_val);
__device__ __forceinline__ uint64_t wgmma_matmul_8bit_descriptor(const uint8_t* tile_ptr);
__device__ __forceinline__ void wgmma_matmul_8bit_multiply_add(float (&sums_list)[WGMMA_MATMUL_8BIT_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                               const uint8_t* left_tile_ptr, const uint8_t* right_tile_ptr);

/// Writes output = left_scale · right_scale · (left_8bit · right_8bitᵀ) (+ bias) (+ output): checks, describes
/// left_8bit's and right_8bit's tiles for TMA, then launches wgmma_matmul_8bit_kernel, one block per square of
/// output, as one line of blocks.
///
///   output_tensor                           {..., columns} bfloat16: written over, or added onto
///   left_8bit_tensor, left_scales_tensor    round_8bit's of left {..., sum_length}: x in a linear layer
///   right_8bit_tensor, right_scales_tensor  round_8bit's of right {columns, sum_length}: W
///   bias_tensor                             {columns} bfloat16, or Tensor() for none
///   add_onto_val                            true: added onto what output holds; false: written over
///   epilogue_val                            what is done to each pair of results as it is stored; not given:
///                                           WgmmaMatmul8bitStore, nothing
/// sum_length must be a multiple of 16 and columns of 8; output bfloat16.
template <typename EPILOGUE>
inline void wgmma_matmul_8bit(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                             const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor,
                             const Tensor& bias_tensor, bool add_onto_val, const EPILOGUE& epilogue_val) {
    // 1. check the tensors; any of them wrong: stop with an error
    wgmma_matmul_8bit_check_shapes(output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor,
                          bias_tensor);

    // 2. the sizes, and TMA's description of left_8bit's and right_8bit's tiles (one byte a value)
    size_t sum_length_val = left_8bit_tensor.size(left_8bit_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT;
    size_t rows_val = left_8bit_tensor.numel() * ROUND_8BIT_VALUES_PER_INT / sum_length_val;
    size_t columns_val = right_8bit_tensor.size(0);
    CUtensorMap left_map_val = wgmma_matmul_8bit_tensor_map(left_8bit_tensor.data_ptr(), rows_val, sum_length_val, HARDWARE_WGMMA_M);
    CUtensorMap right_map_val = wgmma_matmul_8bit_tensor_map(right_8bit_tensor.data_ptr(), columns_val, sum_length_val, WGMMA_MATMUL_8BIT_BLOCK_COLUMNS);

    // 3. launch: squares across and down, rounded up, as one line of blocks
    size_t total_block_columns_val = CEIL_DIV(columns_val, (size_t)WGMMA_MATMUL_8BIT_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)HARDWARE_WGMMA_M);
    unsigned total_blocks_val = (unsigned)(total_block_columns_val * total_block_rows_val);
    wgmma_matmul_8bit_kernel<EPILOGUE><<<total_blocks_val, HARDWARE_WARPGROUP_THREADS>>>(
        left_map_val, right_map_val, output_tensor.data_ptr<nv_bfloat16>(), left_scales_tensor.data(),
        right_scales_tensor.data(), bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr,
        (int)rows_val, (int)columns_val, (int)sum_length_val, add_onto_val, epilogue_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output with one warpgroup: step by step, TMA brings a tile of left_8bit and of
/// right_8bit into shared memory, then the E4M3 tensor cores add their product into the sums in registers; at the
/// end each sum is times its row's and column's scales.
///
///   left_map_val, right_map_val                TMA's descriptions of left_8bit's and right_8bit's tiles
///   output_ptr, left_scales_ptr, right_scales_ptr, bias_ptr   as in wgmma_matmul_8bit; bias nullptr for none
///   rows_val, columns_val, sum_length_val      the shapes
///   add_onto_val, epilogue_val                 as in wgmma_matmul_8bit
template <typename EPILOGUE>
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_8bit_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                        nv_bfloat16* output_ptr, const float* left_scales_ptr, const float* right_scales_ptr,
                        const nv_bfloat16* bias_ptr, int rows_val, int columns_val, int sum_length_val, bool add_onto_val,
                        EPILOGUE epilogue_val) {
    namespace cde = cuda::device::experimental;
    // this step's tiles, in TMA's swizzled layout (on a 1024-byte atom), and the barrier that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t left_tile_list[HARDWARE_WGMMA_M * WGMMA_MATMUL_8BIT_BLOCK_K];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t right_tile_list[WGMMA_MATMUL_8BIT_BLOCK_COLUMNS * WGMMA_MATMUL_8BIT_BLOCK_K];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ WgmmaMatmul8bitBarrier tiles_barrier_val;

    // 1. the block's square, from its number: squares row after row; and the barrier, by one thread
    int total_block_columns_val = CEIL_DIV(columns_val, WGMMA_MATMUL_8BIT_BLOCK_COLUMNS);
    int block_row_index_val = blockIdx.x / total_block_columns_val;
    int block_column_index_val = blockIdx.x % total_block_columns_val;
    int block_first_row_index_val = block_row_index_val * HARDWARE_WGMMA_M;
    int block_first_column_index_val = block_column_index_val * WGMMA_MATMUL_8BIT_BLOCK_COLUMNS;
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                     // every thread arrives once a step
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();

    // 2. walk the sum: TMA brings the step's tiles, the tensor cores add their product
    float sums_list[WGMMA_MATMUL_8BIT_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of the square
    int total_steps_val = CEIL_DIV(sum_length_val, WGMMA_MATMUL_8BIT_BLOCK_K);
    for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. one thread asks for both tiles (past the edges TMA gives 0s) and says how many bytes to wait for;
        //    the others arrive; all wait until the bytes are in
        WgmmaMatmul8bitBarrier::arrival_token token_val;
        int step_first_k_index_val = step_index_val * WGMMA_MATMUL_8BIT_BLOCK_K;
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(left_tile_list, &left_map_val, step_first_k_index_val,
                                                          block_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(right_tile_list, &right_map_val, step_first_k_index_val,
                                                          block_first_column_index_val, tiles_barrier_val);
            token_val = cuda::device::barrier_arrive_tx(tiles_barrier_val, 1, sizeof(left_tile_list) + sizeof(right_tile_list));
        } else {
            token_val = tiles_barrier_val.arrive();
        }
        tiles_barrier_val.wait(std::move(token_val));

        // b. the tensor cores: WGMMA_MATMUL_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3 WGMMAs, each 32 more values (bytes) of k
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < WGMMA_MATMUL_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3; k_step_index_val++)
            wgmma_matmul_8bit_multiply_add(sums_list, left_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                           right_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // ... done

        // c. every thread done with the tiles before TMA fills them again
        __syncthreads();
    }

    // 3. output = epilogue(left_scale · right_scale · sums (+ bias)) (+ output), two neighbouring columns at a
    //    time; nothing past the edges
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int thread_first_row_index_val = block_first_row_index_val + warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
#pragma unroll
    for (int group_index_val = 0; group_index_val < WGMMA_MATMUL_8BIT_COLUMN_GROUPS; group_index_val++) {
        int column_index_val = block_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
        if (column_index_val >= columns_val) continue;       // columns is even, so column + 1 is in too
        float2 right_scales_val = *reinterpret_cast<const float2*>(right_scales_ptr + column_index_val);
        float2 bias_val = bias_ptr != nullptr ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                              : make_float2(0.0f, 0.0f);
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
            int row_index_val = thread_first_row_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
            if (row_index_val >= rows_val) continue;
            float left_scale_val = left_scales_ptr[row_index_val];
            size_t output_index_val = (size_t)row_index_val * columns_val + column_index_val;
            nv_bfloat162* output_pair_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + output_index_val);
            float2 result_val = make_float2(sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] * left_scale_val * right_scales_val.x + bias_val.x,
                                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] * left_scale_val * right_scales_val.y + bias_val.y);
            result_val = epilogue_val(result_val, output_index_val);
            if (add_onto_val) {
                float2 old_val = __bfloat1622float2(*output_pair_ptr);
                result_val = make_float2(result_val.x + old_val.x, result_val.y + old_val.y);
            }
            *output_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
        }
    }
}

/// Gives the WGMMA descriptor of a tile in shared memory, as wgmma_matmul_bf16.cuh's (it counts bytes): K-major rows of
/// 128 bytes in TMA's 128-byte swizzle, 8-row atoms HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM apart. Moving 32 values of k is 32
/// bytes more on the start address.
///
///   tile_ptr   the tile's first value to read, in shared memory
__device__ __forceinline__ uint64_t wgmma_matmul_8bit_descriptor(const uint8_t* tile_ptr) {
    uint64_t address_val = (uint64_t)__cvta_generic_to_shared(tile_ptr);
    uint64_t descriptor_val = ((address_val & HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS) / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT;
    descriptor_val |= HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 << HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT;
    return descriptor_val;
}

/// sums {64, 64} += left tile {64, 32} · right tile {64, 32}ᵀ on the E4M3 tensor cores, for the whole warpgroup:
/// wgmma.mma_async m64n64k32, E4M3 in, float sums. FP8 takes only K-major tiles (no transpose), read straight
/// from shared memory; asynchronous: done only after wgmma.wait_group.
///
///   sums_list                       this thread's 32 sums, in WGMMA's layout
///   left_tile_ptr, right_tile_ptr   the 32 values of k to use, in the shared memory tiles
__device__ __forceinline__ void wgmma_matmul_8bit_multiply_add(float (&sums_list)[WGMMA_MATMUL_8BIT_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                               const uint8_t* left_tile_ptr, const uint8_t* right_tile_ptr) {
    uint64_t left_descriptor_val = wgmma_matmul_8bit_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = wgmma_matmul_8bit_descriptor(right_tile_ptr);
    // scale-d 1: add onto the sums; scale-a, scale-b 1
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1;\n"
        "}\n"
        : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
          "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
          "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
          "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3])
        : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(1));
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Gives TMA's description of an E4M3 matrix's tiles: {matrix_rows, matrix_bytes_per_row} bytes sitting row after
/// row, read in boxes of {box_rows, WGMMA_MATMUL_8BIT_BLOCK_K} written into shared memory in the 128-byte swizzle; past the
/// edges, 0s. Made by the driver's cuTensorMapEncodeTiled, found at run time, so the program needs no -lcuda.
///
///   matrix_ptr                              its first byte, on a 16-byte piece
///   matrix_rows_val, matrix_bytes_per_row_val   its shape; bytes per row (= values: one E4M3 each) a multiple of 16
///   box_rows_val                            rows of a tile: HARDWARE_WGMMA_M or WGMMA_MATMUL_8BIT_BLOCK_COLUMNS
inline CUtensorMap wgmma_matmul_8bit_tensor_map(const void* matrix_ptr, size_t matrix_rows_val, size_t matrix_bytes_per_row_val,
                                       unsigned box_rows_val) {
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function_ptr, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: wgmma_matmul_8bit could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: bytes of a row (k), then rows
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {matrix_bytes_per_row_val, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {matrix_bytes_per_row_val};                         // from one row to the next
    cuuint32_t box_list[2] = {WGMMA_MATMUL_8BIT_BLOCK_K, box_rows_val};
    cuuint32_t element_steps_list[2] = {1, 1};                                         // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(matrix_ptr), sizes_list,
                                     row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %zu}-byte matrix at %p\n", (int)result_val,
                matrix_rows_val, matrix_bytes_per_row_val, matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Stops with a message unless the product's tensors fit: the shapes at the top of the file, the types, sum_length
/// a multiple of 16 and columns of 8, each tensor starting on a 16-byte piece.
///
///   output_tensor, left_8bit_tensor, left_scales_tensor, right_8bit_tensor, right_scales_tensor, bias_tensor
///       as in wgmma_matmul_8bit
inline void wgmma_matmul_8bit_check_shapes(const Tensor& output_tensor, const Tensor& left_8bit_tensor, const Tensor& left_scales_tensor,
                                  const Tensor& right_8bit_tensor, const Tensor& right_scales_tensor, const Tensor& bias_tensor) {
    // 1. shapes: the same rows, the sums' lengths and columns matching right, a scale per row of each
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = left_8bit_tensor.dim() >= 1 && right_8bit_tensor.dim() == 2 &&
                    output_tensor.dim() == left_8bit_tensor.dim() && left_scales_tensor.dim() == left_8bit_tensor.dim() - 1 &&
                    left_8bit_tensor.size(left_8bit_tensor.dim() - 1) == right_8bit_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == right_8bit_tensor.size(0) &&
                    right_scales_tensor.shape() == Shape({right_8bit_tensor.size(0)}) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({right_8bit_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < left_8bit_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == left_8bit_tensor.size(dim_val) &&
                   left_scales_tensor.size(dim_val) == left_8bit_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: wgmma_matmul_8bit needs left_8bit {..., sum_length / %d} with left_scales {...}, output {..., "
                        "columns} with the same rows, right_8bit {columns, sum_length / %d} with right_scales {columns}, "
                        "bias {columns} or Tensor()\n",
                (int)ROUND_8BIT_VALUES_PER_INT, (int)ROUND_8BIT_VALUES_PER_INT);
        exit(EXIT_FAILURE);
    }
    // 2. types: the 8-bit copies Int32, the scales float, output and bias bfloat16
    if (left_8bit_tensor.dtype() != DType::Int32 || right_8bit_tensor.dtype() != DType::Int32 ||
        left_scales_tensor.dtype() != DType::Float32 || right_scales_tensor.dtype() != DType::Float32 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: wgmma_matmul_8bit needs left_8bit and right_8bit Int32 (round_8bit's), the scales float, "
                        "output and bias bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte pieces, each tensor starting on one
    size_t sum_length_val = right_8bit_tensor.size(1) * ROUND_8BIT_VALUES_PER_INT;
    if (sum_length_val % HARDWARE_TMA_BYTES_PER_CELL != 0 || right_8bit_tensor.size(0) % (HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16)) != 0 ||
        (size_t)output_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)left_8bit_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 ||
        (size_t)right_8bit_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)bias_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 ||
        (size_t)right_scales_tensor.data_ptr() % sizeof(float2) != 0) {
        fprintf(stderr, "error: wgmma_matmul_8bit's TMA reads rows of whole %d-byte pieces, so sum_length must be a multiple of %d "
                        "and columns of %zu, every tensor starting on %d bytes; got sum_length %zu, columns %zu\n",
                HARDWARE_TMA_BYTES_PER_CELL, HARDWARE_TMA_BYTES_PER_CELL, HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16), HARDWARE_TMA_BYTES_PER_CELL,
                sum_length_val, right_8bit_tensor.size(0));
        exit(EXIT_FAILURE);
    }
}

#endif // WGMMA_MATMUL_8BIT_CUH
