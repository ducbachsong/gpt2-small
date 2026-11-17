// attention_flash_8bit.cuh — flash attention for a bfloat16 model with query, key and value kept in 8 bits (FP8
// E4M3, one float scale per head row; attention_8bit.cuh): attention_flash_bf16.cuh's flash attention, but
//     F1   query, key, value are rounded to E4M3 once (attention_8bit_round_qkv), and kept that way for the backward
//          (half of bfloat16's bytes)
//     F2   scores = query · keyᵀ on an H100's tensor cores in E4M3 (WGMMA m64n64k32, the tiles brought by TMA): twice
//          bfloat16's values per instruction
//     F5   output += attention_probs · value stays bfloat16: value is turned back to bfloat16 as its tile is loaded, because
//          the attention_probs (0 .. 1, a row adding up to 1) would lose too much in E4M3's 3 mantissa bits
//     backward: query, key, value turned back to bfloat16 into a qkv-shaped scratch (attention_8bit_turn_back),
//          then attention_flash_bf16.cuh's flash backward on it
//
//     tensor                      type in memory                  shape                                           why
//     qkv, qkv_grad               bfloat16                        {batch_size, seq_len, 3 · embed_dim}            the model's
//     output, output_grad         bfloat16                        {batch_size, seq_len, embed_dim}                the model's
//     query_8bit, key_8bit,       E4M3                            {batch_size, num_heads, seq_len, head_size}     kept for backward
//     value_8bit
//     query_scales, key_scales,   float                           {batch_size, num_heads, seq_len}                one per head row
//     value_scales
//     qkv_rounded                 bfloat16                        {batch_size, seq_len, 3 · embed_dim}            backward scratch: the 8-bit
//                                                                                                                 ones turned back, where
//                                                                                                                 qkv has them
//     logsumexp, row_dot          float                           {batch_size, num_heads, seq_len}                as attention_flash_bf16.cuh's
//     in registers                float                           -                                               every sum and the softmax
//
// The scales are powers of 2, so turning back is exact: the backward sees the very query, key and value the
// forward used, and its gradient is the exact gradient of attention on the rounded values; the rounding itself
// is passed through, as if it were not there.
// Needs head_size FLASH_HEAD_SIZE (64 or 128), seq_len a multiple of FLASH_TILE, and an H100 (WGMMA and TMA exist
// only on Hopper), built with -arch=sm_90a.
#ifndef ATTENTION_FLASH_8BIT_CUH
#define ATTENTION_FLASH_8BIT_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, HARDWARE_WGMMA_*, HARDWARE_TMA_*
#include "../../common/tensor.cuh"
#include "attention_8bit.cuh"        // the rounding and the turning back: attention_8bit_round_qkv, attention_8bit_turn_back
#include "../bf16/attention_flash_bf16.cuh"  // FLASH_* tiles, the barrier wait and the WGMMAs for F5, attention_bf16_flash_backward

#define ATTENTION_8BIT_K_STEPS (FLASH_HEAD_SIZE / HARDWARE_WGMMA_K_E4M3)            // WGMMAs across a head row
#define ATTENTION_8BIT_LOAD_VALUES (sizeof(uint4) / sizeof(__nv_fp8_e4m3))          // E4M3s per uint4 load
#define ATTENTION_8BIT_CELLS_PER_LOAD ((int)(ATTENTION_8BIT_LOAD_VALUES * sizeof(nv_bfloat16) / HARDWARE_TMA_BYTES_PER_CELL))   // cells of a bfloat16 tile one load's values fill
#define ATTENTION_8BIT_TILE_BYTES_PER_ROW ((int)(FLASH_HEAD_SIZE * sizeof(__nv_fp8_e4m3)))   // an E4M3 tile's row in shared memory: a head row
#define ATTENTION_8BIT_TILE_BYTES (FLASH_TILE * ATTENTION_8BIT_TILE_BYTES_PER_ROW)           // an E4M3 tile: what one TMA copy brings
#define ATTENTION_8BIT_TILE_BYTES_PER_ATOM (HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN * ATTENTION_8BIT_TILE_BYTES_PER_ROW)   // from one atom's rows to the next's
#define ATTENTION_8BIT_FULL_SWIZZLE_ROW (ATTENTION_8BIT_TILE_BYTES_PER_ROW == HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW)   // head size 128: an E4M3 head row is a whole swizzle row (the 128-byte swizzle); 64: half of one (the 64-byte swizzle)
#define ATTENTION_8BIT_TMA_SWIZZLE (ATTENTION_8BIT_FULL_SWIZZLE_ROW ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B)   // the E4M3 tiles' swizzle, as TMA names it
#define ATTENTION_8BIT_DESCRIPTOR_SWIZZLE (ATTENTION_8BIT_FULL_SWIZZLE_ROW ? HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 : HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_64)   // ... as the WGMMA descriptor names it
#define ATTENTION_8BIT_VALUE_CELLS_PER_BOX_ROW ((int)(FLASH_BOX_VALUES_PER_ROW * sizeof(nv_bfloat16) / HARDWARE_TMA_BYTES_PER_CELL))   // the bfloat16 value tile: cells in a box's row, 8
static_assert(ATTENTION_8BIT_HEAD_SIZE == FLASH_HEAD_SIZE, "the rounding's head rows are flash attention's");
static_assert(2 * ATTENTION_8BIT_TILE_BYTES_PER_ROW == HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW || ATTENTION_8BIT_FULL_SWIZZLE_ROW,
              "an E4M3 head row is half a swizzle row (the 64-byte swizzle) or a whole one (the 128-byte): head size 64 or 128");
static_assert(ATTENTION_8BIT_VALUE_CELLS_PER_BOX_ROW == HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_ROW, "the bfloat16 value tile's box row is one swizzle row: a row's cells XOR the row");
static_assert(ATTENTION_8BIT_VALUE_CELLS_PER_BOX_ROW % ATTENTION_8BIT_CELLS_PER_LOAD == 0, "one load's cells stay in one box");
static_assert(sizeof(uint4) == HARDWARE_TMA_BYTES_PER_CELL, "the value tile is stored one cell, one uint4, at a time");

// ── forward and backward ────────────────────────────────────────────────────────────
bool attention_8bit_fits(size_t seq_len_val, size_t head_size_val);
void attention_8bit_forward(const Tensor& output_tensor, const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor,
                            const Tensor& value_8bit_tensor, const Tensor& query_scales_tensor,
                            const Tensor& key_scales_tensor, const Tensor& value_scales_tensor,
                            const Tensor& logsumexp_tensor, const Tensor& qkv_tensor,
                            int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK, size_t window_val = 0);
void attention_8bit_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& output_tensor,
                             const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor,
                             const Tensor& value_8bit_tensor, const Tensor& query_scales_tensor,
                             const Tensor& key_scales_tensor, const Tensor& value_scales_tensor,
                             const Tensor& logsumexp_tensor, const Tensor& qkv_rounded_tensor,
                             const Tensor& row_dot_tensor, int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK,
                             int row_dot_threads_per_block_val = ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK, size_t window_val = 0,
                             const AttentionFlashQkvGrad8bit* qkv_grad_8bit_ptr = nullptr);

// ── for both ────────────────────────────────────────────────────────────
CUtensorMap attention_8bit_tensor_map(const void* matrix_ptr, size_t matrix_rows_val);
void attention_8bit_check_shapes(const Tensor& output_tensor, const Tensor& query_8bit_tensor,
                                 const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                 const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                 const Tensor& value_scales_tensor, const Tensor& logsumexp_tensor,
                                 const Tensor& qkv_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(FLASH_THREADS)
attention_8bit_forward_kernel(const __grid_constant__ CUtensorMap query_map_val, const __grid_constant__ CUtensorMap key_map_val,
                              nv_bfloat16* output_ptr, float* logsumexp_ptr, const __nv_fp8_e4m3* value_8bit_ptr,
                              const float* query_scales_ptr, const float* key_scales_ptr, const float* value_scales_ptr,
                              int num_heads_val, int seq_len_val, float scale_val, int window_tiles_val);

/// Tells if 8-bit flash attention can run here: head_size FLASH_HEAD_SIZE, seq_len a multiple of FLASH_TILE,
/// and a GPU with WGMMA and TMA (compute capability HARDWARE_WGMMA_CAPABILITY / 10 exactly: an H100).
///
///   seq_len_val, head_size_val   the tensors'
inline bool attention_8bit_fits(size_t seq_len_val, size_t head_size_val) {
    static int capability_val = [] {
        int device_index_val = 0;
        int major_val = 0;
        int minor_val = 0;
        cudaCheck(cudaGetDevice(&device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, device_index_val));
        return 10 * major_val + minor_val;                  // compute capability x 10, as HARDWARE_WGMMA_CAPABILITY
    }();
    return head_size_val == FLASH_HEAD_SIZE && seq_len_val % FLASH_TILE == 0 && capability_val == HARDWARE_WGMMA_CAPABILITY;
}

/// Writes output = causal self-attention of qkv, keeping query, key and value in 8 bits for the backward:
/// checks, F1 (round query, key, value to E4M3, one warp per head row), then one kernel for F2..F6, one block
/// (one warpgroup) per FLASH_TILE queries of a head, its query and key tiles brought by TMA.
///
///   output_tensor                                {batch_size, seq_len, embed_dim} bfloat16
///   query_8bit_tensor, key_8bit_tensor,
///   value_8bit_tensor                            {batch_size, num_heads, seq_len, head_size} E4M3: written
///   query_scales_tensor, key_scales_tensor,
///   value_scales_tensor                          {batch_size, num_heads, seq_len} float: written
///   logsumexp_tensor                             {batch_size, num_heads, seq_len} float: written
///   qkv_tensor                                   {batch_size, seq_len, 3 · embed_dim} bfloat16: the qkv linear's output
///   threads_per_block_val                        F1's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                                                given: ATTENTION_8BIT_THREADS_PER_BLOCK. The tile kernel's is always
///                                                FLASH_THREADS. Any size gives the same bits; only the time changes.
///   window_val                                   the sliding window, as attention_bf16_flash_forward's: a query sees its
///                                                own key and the window_val keys before it; a multiple of FLASH_TILE;
///                                                0 (not given): every key before it
inline void attention_8bit_forward(const Tensor& output_tensor, const Tensor& query_8bit_tensor,
                                   const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                   const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                   const Tensor& value_scales_tensor, const Tensor& logsumexp_tensor,
                                   const Tensor& qkv_tensor, int threads_per_block_val, size_t window_val) {
    // 1. check the tensors; any of them wrong: stop with an error (the launch is checked by the rounding)
    attention_8bit_check_shapes(output_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor,
                                query_scales_tensor, key_scales_tensor, value_scales_tensor, logsumexp_tensor, qkv_tensor);

    // 2. the sizes
    size_t num_heads_val = query_8bit_tensor.size(1);
    size_t seq_len_val = query_8bit_tensor.size(2);
    size_t head_size_val = query_8bit_tensor.size(3);
    size_t heads_in_batch_val = query_8bit_tensor.size(0) * num_heads_val;              // batch_size x num_heads
    size_t rows_val = heads_in_batch_val * seq_len_val;                                 // one head row per (batch, head, token)

    // 3. F1: query, key, value rounded to E4M3, one warp per head row
    attention_8bit_round_qkv(query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor, key_scales_tensor,
                             value_scales_tensor, qkv_tensor, threads_per_block_val);

    // 4. TMA's descriptions of query_8bit's and key_8bit's tiles: each as one matrix of head rows
    CUtensorMap query_map_val = attention_8bit_tensor_map(query_8bit_tensor.data_ptr(), rows_val);
    CUtensorMap key_map_val = attention_8bit_tensor_map(key_8bit_tensor.data_ptr(), rows_val);

    // 5. F2..F6: one block per FLASH_TILE queries of each head
    dim3 grid_val((unsigned)(seq_len_val / FLASH_TILE), (unsigned)heads_in_batch_val);
    attention_8bit_forward_kernel<<<grid_val, FLASH_THREADS>>>(
        query_map_val, key_map_val, output_tensor.data_ptr<nv_bfloat16>(), logsumexp_tensor.data(),
        reinterpret_cast<const __nv_fp8_e4m3*>(value_8bit_tensor.data_ptr()), query_scales_tensor.data(),
        key_scales_tensor.data(), value_scales_tensor.data(), (int)num_heads_val, (int)seq_len_val,
        1.0f / sqrtf((float)head_size_val), attention_bf16_flash_window_tiles(window_val));
    cudaCheck(cudaGetLastError());
}

// WGMMA m64n64k32 in E4M3: C {64, 64} += A {64, 32} · B {32, 64}, sums in float, one instruction for the block's
// whole warpgroup; both tiles are read from shared memory as they sit, k along a token's values (FP8 WGMMA has
// no transpose bit). An E4M3 tile {FLASH_TILE tokens, FLASH_HEAD_SIZE values} has rows of
// ATTENTION_8BIT_TILE_BYTES_PER_ROW bytes, half of bfloat16's: at head size 64 TMA writes it in the 64-byte swizzle
// (4 cells a row), at 128 in the 128-byte swizzle (8 cells, one whole swizzle row), atoms of the same
// HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN rows, and the WGMMA descriptor says which.
// C's layout is attention_flash_bf16.cuh's (WGMMA's, whatever the type of A and B), so the softmax below is the same.

/// Gives TMA's description of an 8-bit tensor's tiles: its E4M3s as a matrix {matrix_rows head rows,
/// FLASH_HEAD_SIZE bytes}, read in boxes of {FLASH_TILE rows, FLASH_HEAD_SIZE bytes} written into shared memory
/// in ATTENTION_8BIT_TMA_SWIZZLE (the 64-byte swizzle at head size 64, the 128-byte at 128). Made by the driver's
/// cuTensorMapEncodeTiled, found at run time, so the program needs no -lcuda.
///
///   matrix_ptr        the tensor's first byte, on a HARDWARE_TMA_BYTES_PER_CELL-byte cell
///   matrix_rows_val   its head rows: batch_size x num_heads x seq_len
inline CUtensorMap attention_8bit_tensor_map(const void* matrix_ptr, size_t matrix_rows_val) {
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function_ptr, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: 8-bit attention could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: bytes of a head row (k), then rows
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {ATTENTION_8BIT_TILE_BYTES_PER_ROW, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {ATTENTION_8BIT_TILE_BYTES_PER_ROW};                // from one row to the next
    cuuint32_t box_list[2] = {ATTENTION_8BIT_TILE_BYTES_PER_ROW, FLASH_TILE};
    cuuint32_t element_steps_list[2] = {1, 1};                                         // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(matrix_ptr), sizes_list,
                                     row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     ATTENTION_8BIT_TMA_SWIZZLE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %d}-byte matrix at %p\n", (int)result_val,
                matrix_rows_val, (int)ATTENTION_8BIT_TILE_BYTES_PER_ROW, matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Gives the WGMMA descriptor of an E4M3 tile in shared memory, as wgmma_matmul_bf16.cuh's but in
/// ATTENTION_8BIT_DESCRIPTOR_SWIZZLE (64-byte or 128-byte): rows of ATTENTION_8BIT_TILE_BYTES_PER_ROW bytes with k
/// along them, the next
/// HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN rows ATTENTION_8BIT_TILE_BYTES_PER_ATOM on. Moving
/// HARDWARE_WGMMA_K_E4M3 values of k is as many bytes more on the start address.
///
///   tile_ptr   the tile's first value to read, in shared memory
__device__ __forceinline__ uint64_t attention_8bit_descriptor(const uint8_t* tile_ptr) {
    uint64_t address_val = (uint64_t)__cvta_generic_to_shared(tile_ptr);
    uint64_t descriptor_val = ((address_val & HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS) / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT;
    descriptor_val |= (uint64_t)(ATTENTION_8BIT_TILE_BYTES_PER_ATOM / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT;
    descriptor_val |= ATTENTION_8BIT_DESCRIPTOR_SWIZZLE << HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT;
    return descriptor_val;
}

/// sums {64, 64} += left tile {64, 32} · right tile {64, 32}ᵀ on the E4M3 tensor cores, for the whole warpgroup:
/// wgmma.mma_async m64n64k32, E4M3 in, float sums, both tiles read straight from shared memory. Asynchronous:
/// done only after flash_wgmma_wait.
///
///   sums_list                       this thread's sums, in WGMMA's layout
///   left_tile_ptr, right_tile_ptr   the HARDWARE_WGMMA_K_E4M3 values of k to use, in the shared memory tiles
__device__ __forceinline__ void attention_8bit_multiply_add(float (&sums_list)[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                            const uint8_t* left_tile_ptr, const uint8_t* right_tile_ptr) {
    uint64_t left_descriptor_val = attention_8bit_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = attention_8bit_descriptor(right_tile_ptr);
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

/// Copies a {FLASH_TILE, FLASH_HEAD_SIZE} tile of E4M3 values into shared memory as bfloat16, each row times its
/// scale: exact, the scale being a power of 2. Written in the layout TMA gives attention_flash_bf16.cuh's tiles
/// (FLASH_HEAD_BOXES boxes of 64 values side by side, FLASH_BOX_VALUES apart, each in the 128-byte swizzle: cell c
/// of a box's row r at slot c XOR (r % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN)), so F5's WGMMA reads it as it
/// reads theirs. ATTENTION_8BIT_LOAD_VALUES E4M3s per load, written as as many bfloat16s: ATTENTION_8BIT_CELLS_PER_LOAD
/// cells, all in one box.
///
///   tile_ptr     shared memory, on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom: FLASH_TILE_BYTES
///   source_ptr   the tile's first value in GPU memory, rows FLASH_HEAD_SIZE apart
///   scales_ptr   the tile's first row's scale; one per row
__device__ __forceinline__ void attention_8bit_load_value_tile(nv_bfloat16* tile_ptr, const __nv_fp8_e4m3* source_ptr,
                                                               const float* scales_ptr) {
    const int loads_per_row_val = FLASH_HEAD_SIZE / ATTENTION_8BIT_LOAD_VALUES;
    for (int piece_index_val = threadIdx.x; piece_index_val < FLASH_TILE * loads_per_row_val; piece_index_val += FLASH_THREADS) {
        int tile_row_index_val = piece_index_val / loads_per_row_val;
        int load_index_val = piece_index_val % loads_per_row_val;
        // 1. ATTENTION_8BIT_LOAD_VALUES E4M3s and their row's scale
        uint4 stored_val = *reinterpret_cast<const uint4*>(source_ptr + tile_row_index_val * FLASH_HEAD_SIZE + load_index_val * ATTENTION_8BIT_LOAD_VALUES);
        float scale_val = scales_ptr[tile_row_index_val];
        const __nv_fp8x4_e4m3* fours_ptr = reinterpret_cast<const __nv_fp8x4_e4m3*>(&stored_val);
        // 2. times the scale, VALUES_PER_FLOAT4 at a time, as pairs of bfloat16s
        __align__(16) nv_bfloat162 pairs_list[ATTENTION_8BIT_LOAD_VALUES / 2];
#pragma unroll
        for (int four_index_val = 0; four_index_val < (int)(ATTENTION_8BIT_LOAD_VALUES / VALUES_PER_FLOAT4); four_index_val++) {
            float4 values_val = (float4)fours_ptr[four_index_val];
            pairs_list[2 * four_index_val] = __floats2bfloat162_rn(values_val.x * scale_val, values_val.y * scale_val);
            pairs_list[2 * four_index_val + 1] = __floats2bfloat162_rn(values_val.z * scale_val, values_val.w * scale_val);
        }
        // 3. into the tile, a cell (one uint4) at a time, each at its swizzled slot of its box's row
        int first_cell_index_val = load_index_val * ATTENTION_8BIT_CELLS_PER_LOAD;            // in the whole head row
        int box_index_val = first_cell_index_val / ATTENTION_8BIT_VALUE_CELLS_PER_BOX_ROW;
        uint4* row_cells_ptr = reinterpret_cast<uint4*>(tile_ptr + box_index_val * FLASH_BOX_VALUES + tile_row_index_val * FLASH_BOX_VALUES_PER_ROW);
        int row_swizzle_val = tile_row_index_val % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN;
#pragma unroll
        for (int load_cell_index_val = 0; load_cell_index_val < ATTENTION_8BIT_CELLS_PER_LOAD; load_cell_index_val++) {
            int cell_index_val = (first_cell_index_val + load_cell_index_val) % ATTENTION_8BIT_VALUE_CELLS_PER_BOX_ROW;   // in the box's row
            row_cells_ptr[cell_index_val ^ row_swizzle_val] = reinterpret_cast<const uint4*>(pairs_list)[load_cell_index_val];
        }
    }
}

/// F2..F6 for FLASH_TILE queries of one head (the longest rows first), attention_bf16_flash_forward_kernel's walk
/// with the scores made in E4M3: the query tile stays in shared memory as E4M3, and for each tile of keys up to
/// the block's own
///     score[t][j]   = (query_8bit[t] · key_8bit[j]) x query_scale[t] x key_scale[j] / sqrt(head_size)
/// then the mask, the online softmax, and output += attention_probs · value in bfloat16, value turned back as it is loaded.
///
///   query_map_val, key_map_val                          TMA's descriptions of query_8bit's and key_8bit's tiles
///   output_ptr                                          {batch_size, seq_len, embed_dim} bfloat16: the head's columns written
///   logsumexp_ptr                                       one float per row: written
///   value_8bit_ptr                                      {batch_size, num_heads, seq_len, FLASH_HEAD_SIZE} E4M3
///   query_scales_ptr, key_scales_ptr, value_scales_ptr  one float per head row
///   num_heads_val, seq_len_val                          the shape
///   scale_val                                           1 / sqrt(FLASH_HEAD_SIZE)
///   window_tiles_val                                    the sliding window in tiles (attention_bf16_flash_window_tiles); 0: none
__global__ void __launch_bounds__(FLASH_THREADS)
attention_8bit_forward_kernel(const __grid_constant__ CUtensorMap query_map_val, const __grid_constant__ CUtensorMap key_map_val,
                              nv_bfloat16* output_ptr, float* logsumexp_ptr, const __nv_fp8_e4m3* value_8bit_ptr,
                              const float* query_scales_ptr, const float* key_scales_ptr, const float* value_scales_ptr,
                              int num_heads_val, int seq_len_val, float scale_val, int window_tiles_val) {
    namespace cde = cuda::device::experimental;
    // the E4M3 tiles as TMA writes them (ATTENTION_8BIT_TMA_SWIZZLE), the value tile in bfloat16 (boxes in the
    // 128-byte swizzle, written by the threads), each on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom, and the
    // barrier that says TMA's have arrived: 8 + 8 + 16 KB at head size 128, inside the 48 KB of static shared memory
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t query_tile_list[ATTENTION_8BIT_TILE_BYTES];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t key_tile_list[ATTENTION_8BIT_TILE_BYTES];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ float key_scale_tile_list[FLASH_TILE];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ FlashBarrier tiles_barrier_val;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW) * HARDWARE_WGMMA_COLUMNS_PER_THREAD;
    int query_block_index_val = gridDim.x - 1 - blockIdx.x;
    size_t head_start_index_val = (size_t)blockIdx.y * seq_len_val * FLASH_HEAD_SIZE;   // this head's first value in value_8bit
    size_t head_first_row_index_val = (size_t)blockIdx.y * seq_len_val;                 // this head's first row: its scale, and its row as TMA sees it
    int warp_first_token_index_val = query_block_index_val * FLASH_TILE + warp_index_val * HARDWARE_WGMMA_M_PER_WARP;

    // 1. the barrier, by one thread; then this block's query tile, kept for the whole walk; this thread's two
    //    rows' query scales, with 1 / sqrt(head_size) in them
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                 // every thread arrives once a load
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();
    if (threadIdx.x == 0)
        cde::cp_async_bulk_tensor_2d_global_to_shared(query_tile_list, &query_map_val, 0,
                                                      (int)head_first_row_index_val + query_block_index_val * FLASH_TILE, tiles_barrier_val);
    flash_wait_for_tiles(tiles_barrier_val, ATTENTION_8BIT_TILE_BYTES);
    float row_query_scale_list[HARDWARE_WGMMA_ROWS_PER_THREAD];
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++)
        row_query_scale_list[thread_row_index_val] =
            query_scales_ptr[head_first_row_index_val + warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP] * scale_val;

    float output_list[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};    // {HARDWARE_WGMMA_M, FLASH_HEAD_SIZE}
    // rows group, group + HARDWARE_WGMMA_ROWS_PER_WARP
    float row_max_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {-INFINITY, -INFINITY};
    float row_sum_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {0.0f, 0.0f};
    // 2. walk the key tiles up to this block's own: from the first, or with a window from the window's oldest tile
    //    (whole tiles: the oldest key a query sees is window_tokens before it, in that tile)
    int window_tokens_val = window_tiles_val > 0 ? window_tiles_val * FLASH_TILE : seq_len_val;   // none: no key is that far back
    int first_key_block_val = window_tiles_val > 0 ? max(query_block_index_val - window_tiles_val, 0) : 0;
    for (int key_block_index_val = first_key_block_val; key_block_index_val <= query_block_index_val; key_block_index_val++) {
        __syncthreads();                                            // every thread is done with the last tiles
        size_t tile_start_index_val = head_start_index_val + (size_t)key_block_index_val * FLASH_TILE * FLASH_HEAD_SIZE;
        size_t tile_first_row_index_val = head_first_row_index_val + (size_t)key_block_index_val * FLASH_TILE;
        if (threadIdx.x == 0)
            cde::cp_async_bulk_tensor_2d_global_to_shared(key_tile_list, &key_map_val, 0, (int)tile_first_row_index_val, tiles_barrier_val);
        attention_8bit_load_value_tile(value_tile_list, value_8bit_ptr + tile_start_index_val, value_scales_ptr + tile_first_row_index_val);
        if (threadIdx.x < FLASH_TILE) key_scale_tile_list[threadIdx.x] = key_scales_ptr[tile_first_row_index_val + threadIdx.x];
        flash_wait_for_tiles(tiles_barrier_val, ATTENTION_8BIT_TILE_BYTES);
        __syncthreads();                                            // the value tile and the key scales are in too

        // F2: scores {HARDWARE_WGMMA_M, FLASH_TILE} = query tile · key tileᵀ in E4M3, ATTENTION_8BIT_K_STEPS WGMMAs
        // across the head row
        float score_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < ATTENTION_8BIT_K_STEPS; k_step_index_val++)
            attention_8bit_multiply_add(score_list, query_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                                        key_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3);
        flash_wgmma_wait();
        // F3: the scales and the mask; then each row's largest score
        float tile_max_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_TILE_GROUPS; group_column_index_val++)
#pragma unroll
            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++) {
                int thread_row_index_val = c_value_index_val / HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                int token_index_val = warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                int column_index_val = group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + pair_column_index_val +
                                       c_value_index_val % HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                int key_token_index_val = key_block_index_val * FLASH_TILE + column_index_val;
                score_list[group_column_index_val][c_value_index_val] =
                    key_token_index_val <= token_index_val && key_token_index_val >= token_index_val - window_tokens_val
                        ? score_list[group_column_index_val][c_value_index_val] * row_query_scale_list[thread_row_index_val] * key_scale_tile_list[column_index_val]
                        : -INFINITY;
                tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], score_list[group_column_index_val][c_value_index_val]);
            }
        // F4: the online softmax: the new max (over the row's HARDWARE_WGMMA_THREADS_PER_ROW lanes), the output and sum
        // moved onto it, then the attention_probs
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
            tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, tile_max_list[thread_row_index_val], 1));
            tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, tile_max_list[thread_row_index_val], 2));
            float new_max_val = fmaxf(row_max_list[thread_row_index_val], tile_max_list[thread_row_index_val]);   // finite: every row sees a key of the first tile walked
            float correction_val = __expf(row_max_list[thread_row_index_val] - new_max_val);     // 0 on the first tile
            row_max_list[thread_row_index_val] = new_max_val;
            row_sum_list[thread_row_index_val] *= correction_val;
#pragma unroll
            for (int group_column_index_val = 0; group_column_index_val < FLASH_HEAD_GROUPS; group_column_index_val++) {
                output_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] *= correction_val;
                output_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] *= correction_val;
            }
        }
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_TILE_GROUPS; group_column_index_val++)
#pragma unroll
            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++) {
                int thread_row_index_val = c_value_index_val / HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                score_list[group_column_index_val][c_value_index_val] =
                    __expf(score_list[group_column_index_val][c_value_index_val] - row_max_list[thread_row_index_val]);   // a masked score gives 0
                row_sum_list[thread_row_index_val] += score_list[group_column_index_val][c_value_index_val];
            }
        // F5: output += attention_probs · value tile in bfloat16: the attention_probs from the score registers as
        // A, the value tile turned round, FLASH_TILE_K_STEPS WGMMAs down its tokens
        uint32_t prob_a_list[FLASH_TILE_K_STEPS][HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16];
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_c_to_a(prob_a_list[k_step_index_val], score_list, k_step_index_val);
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_multiply_add_registers(output_list, prob_a_list[k_step_index_val],
                                         value_tile_list + k_step_index_val * FLASH_TURNED_K_STEP_VALUES);
        flash_wgmma_wait();
    }

    // 3. F6: divide by the row sums, write the head's columns of output, and logsumexp
    int batch_index_val = blockIdx.y / num_heads_val;
    int head_index_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        row_sum_list[thread_row_index_val] += __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_sum_list[thread_row_index_val], 1);
        row_sum_list[thread_row_index_val] += __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_sum_list[thread_row_index_val], 2);
        int token_index_val = warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        nv_bfloat16* row_ptr = output_ptr + ((size_t)batch_index_val * seq_len_val + token_index_val) * embed_dim_val + head_index_val * FLASH_HEAD_SIZE;
        float one_over_sum_val = 1.0f / row_sum_list[thread_row_index_val];
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_HEAD_GROUPS; group_column_index_val++)
            *reinterpret_cast<nv_bfloat162*>(row_ptr + group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + pair_column_index_val) = __floats2bfloat162_rn(
                output_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] * one_over_sum_val,
                output_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] * one_over_sum_val);
        if (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW == 0)
            logsumexp_ptr[head_first_row_index_val + token_index_val] = row_max_list[thread_row_index_val] + logf(row_sum_list[thread_row_index_val]);
    }
}

// ── backward ────────────────────────────────────────────────────────────

/// Writes qkv_grad (written over) for 8-bit attention: checks, turns query, key and value back to bfloat16
/// (exactly) where qkv has them, then runs attention_flash_bf16.cuh's flash backward on that.
///
///   qkv_grad_tensor                              {batch_size, seq_len, 3 · embed_dim} bfloat16: the result
///   output_grad_tensor                           {batch_size, seq_len, embed_dim} bfloat16: from the output linear
///   output_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor,
///   key_scales_tensor, value_scales_tensor,
///   logsumexp_tensor                             the forward's
///   qkv_rounded_tensor                           {batch_size, seq_len, 3 · embed_dim} bfloat16: scratch
///   row_dot_tensor                               {batch_size, num_heads, seq_len} float: scratch
///   threads_per_block_val                        the turn-back's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK;
///                                                not given: ATTENTION_8BIT_THREADS_PER_BLOCK
///   row_dot_threads_per_block_val                row_dot's, the same; not given: ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK.
///                                                Any sizes give the same bits; only the time changes.
///   window_val                                   the forward's sliding window (0, not given: none)
///   qkv_grad_8bit_ptr                            qkv_grad in 8 bits in place of bfloat16 (attention_bf16_flash_backward's);
///                                                nullptr (not given): bfloat16 into qkv_grad_tensor
inline void attention_8bit_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                    const Tensor& output_tensor, const Tensor& query_8bit_tensor,
                                    const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                    const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                    const Tensor& value_scales_tensor, const Tensor& logsumexp_tensor,
                                    const Tensor& qkv_rounded_tensor, const Tensor& row_dot_tensor,
                                    int threads_per_block_val, int row_dot_threads_per_block_val, size_t window_val,
                                    const AttentionFlashQkvGrad8bit* qkv_grad_8bit_ptr) {
    // 1. check the 8-bit tensors as the forward does; the rest is checked by the two calls below
    attention_8bit_check_shapes(output_grad_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor,
                                query_scales_tensor, key_scales_tensor, value_scales_tensor, logsumexp_tensor,
                                qkv_grad_tensor);
    // 2. query, key, value back in bfloat16 where qkv has them: the very values the forward used
    attention_8bit_turn_back(qkv_rounded_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor,
                             query_scales_tensor, key_scales_tensor, value_scales_tensor, threads_per_block_val);
    // 3. the flash backward on them
    attention_bf16_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, qkv_rounded_tensor,
                                  logsumexp_tensor, row_dot_tensor, row_dot_threads_per_block_val, window_val, qkv_grad_8bit_ptr);
}

// ── for both ────────────────────────────────────────────────────────────

/// Stops with a message unless 8-bit attention can run these tensors: the shapes at the top of the file, the
/// types, attention_8bit_fits, the 8-bit tensors starting on a 16-byte load (one of TMA's cells).
///
///   output_tensor                                forward: output; backward: output_grad
///   query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor, key_scales_tensor,
///   value_scales_tensor, logsumexp_tensor        the forward's
///   qkv_tensor                                   forward: qkv; backward: qkv_grad
inline void attention_8bit_check_shapes(const Tensor& output_tensor, const Tensor& query_8bit_tensor,
                                        const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                        const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                        const Tensor& value_scales_tensor, const Tensor& logsumexp_tensor,
                                        const Tensor& qkv_tensor) {
    // 1. shapes, types and addresses together
    bool fits_val = query_8bit_tensor.dim() == 4 && key_8bit_tensor.shape() == query_8bit_tensor.shape() &&
                    value_8bit_tensor.shape() == query_8bit_tensor.shape();
    if (fits_val) {
        size_t batch_size_val = query_8bit_tensor.size(0);
        size_t num_heads_val = query_8bit_tensor.size(1);
        size_t seq_len_val = query_8bit_tensor.size(2);
        size_t head_size_val = query_8bit_tensor.size(3);
        size_t embed_dim_val = num_heads_val * head_size_val;
        Shape per_row_list = {batch_size_val, num_heads_val, seq_len_val};
        fits_val = attention_8bit_fits(seq_len_val, head_size_val) &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val}) &&
                   output_tensor.dtype() == DType::BFloat16 && qkv_tensor.dtype() == DType::BFloat16;
        for (const Tensor* tensor_ptr : {&query_8bit_tensor, &key_8bit_tensor, &value_8bit_tensor})
            fits_val = fits_val && tensor_ptr->dtype() == DType::E4M3 && (size_t)tensor_ptr->data_ptr() % sizeof(uint4) == 0;
        for (const Tensor* tensor_ptr : {&query_scales_tensor, &key_scales_tensor, &value_scales_tensor, &logsumexp_tensor})
            fits_val = fits_val && tensor_ptr->shape() == per_row_list && tensor_ptr->dtype() == DType::Float32;
    }
    // 2. any of them wrong: stop with an error
    if (!fits_val) {
        fprintf(stderr, "error: 8-bit attention needs query_8bit, key_8bit, value_8bit E4M3 {batch_size, num_heads, "
                        "seq_len, %d} starting on %zu bytes, with seq_len a multiple of %d; float "
                        "scales and logsumexp {batch_size, num_heads, seq_len}; bfloat16 output {batch_size, seq_len, "
                        "embed_dim} and qkv {batch_size, seq_len, 3 * embed_dim}; and a GPU of compute capability %d.%d "
                        "(an H100: WGMMA and TMA)\n",
                FLASH_HEAD_SIZE, sizeof(uint4), FLASH_TILE,
                HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_FLASH_8BIT_CUH
