// attention_flash_bf16.cuh — causal self-attention in bfloat16 by flash attention (FlashAttention-2): attention.cuh's
// math without storing the attention_probs, on an H100's tensor cores. The math is in float; each value is
// rounded to bfloat16 only as it is stored.
//
//     tensor                 type in memory   shape                                    why
//     qkv, qkv_grad          bfloat16         {batch_size, seq_len, 3 · embed_dim}     the model's; qkv kept for backward
//     output, output_grad    bfloat16         {batch_size, seq_len, embed_dim}         the model's; output kept for backward
//     logsumexp, row_dot     float            {batch_size, num_heads, seq_len}         one per row, in the attention_probs' place
//     in registers           float            -                                        every sum and the softmax
//
// A block of one warpgroup takes FLASH_TILE queries and walks the keys FLASH_TILE at a time on the tensor cores
// (WGMMA m64n64k16, one instruction for the whole block's {FLASH_TILE, FLASH_TILE} product), with classifier.cuh's
// online softmax; it keeps logsumexp = row_max + log(row_sum) per row, from which the backward makes the
// attention_probs again. No copies: its tiles are brought from qkv by TMA, a head's FLASH_HEAD_SIZE columns of
// FLASH_TILE tokens at a time, into shared memory in the swizzled layout the tensor cores read
// (wgmma_matmul_bf16.cuh's), and output and qkv_grad are written straight into each head's columns. Needs
// head_size FLASH_HEAD_SIZE, seq_len a multiple of FLASH_TILE, and an H100 (WGMMA and TMA exist only on Hopper),
// built with -arch=sm_90a. Its block size is fixed by the tiles (one warpgroup per HARDWARE_WGMMA_M rows), so its
// tile kernels take no block size. Other sizes and GPUs: attention_bf16.cuh (the steps, which keep the
// attention_probs). The same with query, key and value kept in 8 bits: attention_flash_8bit.cuh.
#ifndef ATTENTION_FLASH_BF16_CUH
#define ATTENTION_FLASH_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../../common/tensor.cuh"
#include "wgmma_matmul_bf16.cuh"     // the TMA tile description, the WGMMA descriptor and the tile x tile WGMMA

#define ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)            // row_dot's default: one warp per row

// WGMMA m64n64k16 fed by TMA, the tensor-core product the kernels are built on: its HARDWARE_WGMMA_* and
// HARDWARE_TMA_* numbers are in cuda_common.h. The index math is written for these numbers: change them together
// with it.
#define FLASH_TILE HARDWARE_WGMMA_M                               // queries per block, and keys per step: one WGMMA's rows
#define FLASH_HEAD_SIZE 64                                        // the only head_size the kernels take: GPT-2's
#define FLASH_THREADS HARDWARE_WARPGROUP_THREADS                  // the block size: one warpgroup
#define FLASH_K_STEPS (FLASH_HEAD_SIZE / HARDWARE_WGMMA_K_BF16)   // WGMMAs across a head row: a sum over a head's values
#define FLASH_TILE_K_STEPS (FLASH_TILE / HARDWARE_WGMMA_K_BF16)   // WGMMAs down a tile's tokens: a sum over keys or queries
#define FLASH_HEAD_GROUPS (FLASH_HEAD_SIZE / HARDWARE_WGMMA_GROUP_COLUMNS)   // groups of C across a head row
#define FLASH_TILE_GROUPS (FLASH_TILE / HARDWARE_WGMMA_GROUP_COLUMNS)        // groups of C across a tile of keys or queries
#define FLASH_TILE_BYTES (FLASH_TILE * FLASH_HEAD_SIZE * sizeof(nv_bfloat16))   // a tile in shared memory: what one TMA copy brings
static_assert(HARDWARE_WGMMA_K_BF16 == 2 * HARDWARE_WGMMA_GROUP_COLUMNS, "two groups of C side by side make one A {16, 16}");
static_assert(FLASH_HEAD_SIZE == 2 * HARDWARE_THREADS_PER_WARP, "row_dot: each lane takes 2 values of a head row");
static_assert(FLASH_HEAD_SIZE == WGMMA_MATMUL_BF16_ROW_VALUES, "a tile's row, a head row, is one swizzled row: wgmma_matmul_bf16_tensor_map's box");
static_assert(FLASH_TILE == WGMMA_MATMUL_BF16_ROW_VALUES, "a tile turned round has rows of FLASH_TILE tokens: one swizzled row too");
static_assert(FLASH_HEAD_GROUPS == WGMMA_MATMUL_BF16_COLUMN_GROUPS && FLASH_TILE_GROUPS == WGMMA_MATMUL_BF16_COLUMN_GROUPS,
              "both kinds of product are one WGMMA m64n64k16: the sums wgmma_matmul_bf16_multiply_add takes");

// ── forward and backward ──────────────────────────────────────────────────────
bool attention_bf16_flash_fits(size_t seq_len_val, size_t head_size_val);
void attention_bf16_flash_forward(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor);
void attention_bf16_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                   const Tensor& output_tensor, const Tensor& qkv_tensor, const Tensor& logsumexp_tensor,
                                   const Tensor& row_dot_tensor,
                                   int threads_per_block_val = ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
void attention_bf16_flash_check_shapes(const Tensor& output_tensor, const Tensor& logsumexp_tensor,
                                       const Tensor& qkv_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward and backward ──────────────────────────────────────────────────────

using FlashBarrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_forward_kernel(const __grid_constant__ CUtensorMap qkv_map_val, nv_bfloat16* output_ptr,
                                    float* logsumexp_ptr, int num_heads_val, int seq_len_val, float scale_val);
__global__ void attention_bf16_flash_row_dot_kernel(float* row_dot_ptr, const nv_bfloat16* output_grad_ptr,
                                                    const nv_bfloat16* output_ptr, int num_heads_val, int seq_len_val,
                                                    size_t rows_val);
__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_key_grad_kernel(const __grid_constant__ CUtensorMap qkv_map_val,
                                     const __grid_constant__ CUtensorMap output_grad_map_val, nv_bfloat16* qkv_grad_ptr,
                                     const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                     int seq_len_val, float scale_val);
__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_query_grad_kernel(const __grid_constant__ CUtensorMap qkv_map_val,
                                       const __grid_constant__ CUtensorMap output_grad_map_val, nv_bfloat16* qkv_grad_ptr,
                                       const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                       int seq_len_val, float scale_val);

/// Tells if flash attention can run here: head_size FLASH_HEAD_SIZE, seq_len a multiple of FLASH_TILE, and a
/// GPU with WGMMA and TMA (compute capability HARDWARE_WGMMA_CAPABILITY / 10 exactly: an H100).
///
///   seq_len_val, head_size_val   the tensors'
inline bool attention_bf16_flash_fits(size_t seq_len_val, size_t head_size_val) {
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

/// Writes output = causal self-attention of qkv, as attention_bf16.cuh's attention_bf16_forward, by flash
/// attention: one kernel for F2..F6, one block (one warpgroup) per FLASH_TILE queries of a head, its tiles brought
/// straight from qkv by TMA.
///
///   output_tensor        {batch_size, seq_len, embed_dim}: written, each head its columns; kept for the backward
///   logsumexp_tensor     {batch_size, num_heads, seq_len} float: kept in place of the attention_probs; num_heads comes
///                        from here
///   qkv_tensor           {batch_size, seq_len, 3 · embed_dim}: the qkv linear's output, kept for the backward
/// Its block size is always FLASH_THREADS: the tiles fix it.
inline void attention_bf16_flash_forward(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    attention_bf16_flash_check_shapes(output_tensor, logsumexp_tensor, qkv_tensor);
    size_t batch_size_val = logsumexp_tensor.size(0);
    size_t num_heads_val = logsumexp_tensor.size(1);
    size_t seq_len_val = logsumexp_tensor.size(2);
    // 2. TMA's description of qkv's tiles: qkv as one matrix {batch_size x seq_len tokens, 3 · embed_dim}, read in
    //    boxes of {FLASH_TILE tokens, FLASH_HEAD_SIZE values}: one head's part of query, key or value
    CUtensorMap qkv_map_val = wgmma_matmul_bf16_tensor_map(qkv_tensor.data_ptr<nv_bfloat16>(), batch_size_val * seq_len_val,
                                                           qkv_tensor.size(2), FLASH_TILE);
    // 3. F2..F6: one block per FLASH_TILE queries of each head
    dim3 grid_val((unsigned)(seq_len_val / FLASH_TILE), (unsigned)(batch_size_val * num_heads_val));
    attention_bf16_flash_forward_kernel<<<grid_val, FLASH_THREADS>>>(
        qkv_map_val, output_tensor.data_ptr<nv_bfloat16>(), logsumexp_tensor.data(), (int)num_heads_val,
        (int)seq_len_val, 1.0f / sqrtf((float)FLASH_HEAD_SIZE));
    cudaCheck(cudaGetLastError());
}

/// Writes qkv_grad, as attention_bf16.cuh's attention_bf16_backward, by flash attention: row_dot (one warp per
/// row), then key_grad and value_grad (one block per FLASH_TILE keys), then query_grad (one block per FLASH_TILE
/// queries), every tile brought straight from qkv and output_grad by TMA.
///
///   qkv_grad_tensor                       the result, written over
///   output_grad_tensor                    the gradient from the output linear
///   output_tensor, qkv_tensor,
///   logsumexp_tensor                      the forward's
///   row_dot_tensor                        {batch_size, num_heads, seq_len} float: scratch
///   threads_per_block_val                 row_dot's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                         ATTENTION_BF16_ROW_DOT_THREADS_PER_BLOCK. The tile kernels' is always
///                                         FLASH_THREADS.
inline void attention_bf16_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                          const Tensor& output_tensor, const Tensor& qkv_tensor,
                                          const Tensor& logsumexp_tensor, const Tensor& row_dot_tensor,
                                          int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "flash attention");
    attention_bf16_flash_check_shapes(output_grad_tensor, logsumexp_tensor, qkv_tensor);
    attention_bf16_flash_check_shapes(output_tensor, row_dot_tensor, qkv_grad_tensor);
    size_t batch_size_val = logsumexp_tensor.size(0);
    size_t num_heads_val = logsumexp_tensor.size(1);
    size_t seq_len_val = logsumexp_tensor.size(2);
    size_t rows_val = logsumexp_tensor.numel();                               // one per (batch, head, token)
    float scale_val = 1.0f / sqrtf((float)FLASH_HEAD_SIZE);
    // 2. row_dot, one warp per row
    attention_bf16_flash_row_dot_kernel<<<(int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val),
                                          threads_per_block_val>>>(
        row_dot_tensor.data(), output_grad_tensor.data_ptr<nv_bfloat16>(), output_tensor.data_ptr<nv_bfloat16>(),
        (int)num_heads_val, (int)seq_len_val, rows_val);
    cudaCheck(cudaGetLastError());
    // 3. TMA's descriptions of qkv's and output_grad's tiles: each as one matrix of batch_size x seq_len token
    //    rows, read in boxes of {FLASH_TILE tokens, FLASH_HEAD_SIZE values}
    CUtensorMap qkv_map_val = wgmma_matmul_bf16_tensor_map(qkv_tensor.data_ptr<nv_bfloat16>(), batch_size_val * seq_len_val,
                                                           qkv_tensor.size(2), FLASH_TILE);
    CUtensorMap output_grad_map_val = wgmma_matmul_bf16_tensor_map(output_grad_tensor.data_ptr<nv_bfloat16>(),
                                                                   batch_size_val * seq_len_val, output_grad_tensor.size(2),
                                                                   FLASH_TILE);
    // 4. key_grad and value_grad, then query_grad: one block per tile of each head
    dim3 grid_val((unsigned)(seq_len_val / FLASH_TILE), (unsigned)(batch_size_val * num_heads_val));
    attention_bf16_flash_key_grad_kernel<<<grid_val, FLASH_THREADS>>>(
        qkv_map_val, output_grad_map_val, qkv_grad_tensor.data_ptr<nv_bfloat16>(), logsumexp_tensor.data(),
        row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
    attention_bf16_flash_query_grad_kernel<<<grid_val, FLASH_THREADS>>>(
        qkv_map_val, output_grad_map_val, qkv_grad_tensor.data_ptr<nv_bfloat16>(), logsumexp_tensor.data(),
        row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
}

// WGMMA m64n64k16: C {64, 64} += A {64, 16} · B {16, 64}, one instruction for the block's whole warpgroup. The
// layout is cuda_common.h's: warp w holds rows HARDWARE_WGMMA_M_PER_WARP w and on of C, and thread `lane` of it,
// with group = lane / HARDWARE_WGMMA_THREADS_PER_ROW and pair = HARDWARE_WGMMA_COLUMNS_PER_THREAD (lane %
// HARDWARE_WGMMA_THREADS_PER_ROW), holds in each group of HARDWARE_WGMMA_GROUP_COLUMNS columns:
//     C (float)      4 values: (group, pair), (group, pair+1), (group+8, pair), (group+8, pair+1)
// B is always a tile in shared memory, as TMA wrote it (the 128-byte swizzle), {FLASH_TILE tokens, FLASH_HEAD_SIZE
// values}, read one of two ways:
//     as it sits        B's 64 columns are the tile's tokens, k runs along a token's values ("K-major"):
//                       query · keyᵀ and the like, the sum over a head row. 16 more k: 16 values on
//     turned round      B's 64 columns are the tile's values, k runs down its tokens ("MN-major", the WGMMA's
//                       transpose bit): attention_probs · value and the like, the sum over tokens. 16 more k: 16 rows on
// A is a tile in shared memory too (flash_multiply_add_tiles), or what was just worked out, in registers
// (flash_multiply_add_registers): two groups of C side by side hold exactly a warp's A {16, 16}
//     A (bfloat16)   4 registers of 2 values: (group, pair .. pair+1), (group+8, ..), (group, pair+8 ..), (group+8, pair+8 ..)
// so the attention_probs go into the next product with no move.

/// Waits, the whole block, until TMA has brought the tiles one thread just asked for: that thread says how many
/// bytes are coming, every thread arrives, and all wait for the bytes.
///
///   tiles_barrier_val   the block's barrier, in shared memory
///   bytes_val           the bytes asked for: FLASH_TILE_BYTES each tile
__device__ __forceinline__ void flash_wait_for_tiles(FlashBarrier& tiles_barrier_val, size_t bytes_val) {
    FlashBarrier::arrival_token token_val;
    if (threadIdx.x == 0)
        token_val = cuda::device::barrier_arrive_tx(tiles_barrier_val, 1, bytes_val);
    else
        token_val = tiles_barrier_val.arrive();
    tiles_barrier_val.wait(std::move(token_val));
}

/// Makes the sums in registers ready for the tensor cores: before the first WGMMA of a group.
__device__ __forceinline__ void flash_wgmma_begin() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/// Waits for the WGMMAs issued since flash_wgmma_begin, as one group: their sums can be read only after this.
__device__ __forceinline__ void flash_wgmma_wait() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

/// sums {64, 64} += left tile {64, 16} · right tile {64, 16}ᵀ on the tensor cores, for the whole warpgroup, both
/// tiles read from shared memory as they sit: wgmma_matmul_bf16.cuh's, neither side turned. Asynchronous: done
/// only after flash_wgmma_wait.
///
///   sums_list                       this thread's sums, in WGMMA's layout
///   left_tile_ptr, right_tile_ptr   the HARDWARE_WGMMA_K_BF16 values of k to use, in the shared memory tiles
__device__ __forceinline__ void flash_multiply_add_tiles(float (&sums_list)[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                         const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr) {
    wgmma_matmul_bf16_multiply_add<false, false>(sums_list, left_tile_ptr, right_tile_ptr);
}

/// sums {64, 64} += A {64, 16} · right tile {16, 64} on the tensor cores, for the whole warpgroup: A from
/// registers, the right tile from shared memory turned round (transpose bit 1). wgmma.mma_async m64n64k16,
/// bfloat16 in, float sums. Asynchronous: done only after flash_wgmma_wait; A's registers and the sums must not
/// change until then.
///
///   sums_list        this thread's sums, in WGMMA's layout
///   a_ptr            this thread's HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16 registers of A
///   right_tile_ptr   the HARDWARE_WGMMA_K_BF16 rows of k to use, in the shared memory tile
__device__ __forceinline__ void flash_multiply_add_registers(float (&sums_list)[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                             const uint32_t* a_ptr, const nv_bfloat16* right_tile_ptr) {
    uint64_t right_descriptor_val = wgmma_matmul_bf16_descriptor(right_tile_ptr);
    // scale-d 1: add onto the sums; scale-a, scale-b 1; transpose-b 1: the right tile is turned round
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
          "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
          "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
          "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3])
        : "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "l"(right_descriptor_val), "r"(1));
}

/// Gives two floats as one register of two bfloat16s, rounded: an A register.
///
///   low_val, high_val   the first goes in the low half
__device__ __forceinline__ uint32_t flash_pack(float low_val, float high_val) {
    nv_bfloat162 both_val = __floats2bfloat162_rn(low_val, high_val);
    return *reinterpret_cast<uint32_t*>(&both_val);
}

/// Turns two groups of C side by side into the warp's A {16, 16} of the same HARDWARE_WGMMA_K_BF16 columns,
/// rounded to bfloat16.
///
///   a_ptr              this thread's HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16 registers of A
///   c_ptr              the {HARDWARE_WGMMA_M, FLASH_TILE} product: FLASH_TILE_GROUPS groups of C
///   k_step_index_val   which HARDWARE_WGMMA_K_BF16 columns: groups 2 step and 2 step + 1
__device__ __forceinline__ void flash_c_to_a(uint32_t* a_ptr, float (*c_ptr)[HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP], int k_step_index_val) {
    a_ptr[0] = flash_pack(c_ptr[2 * k_step_index_val][0], c_ptr[2 * k_step_index_val][1]);
    a_ptr[1] = flash_pack(c_ptr[2 * k_step_index_val][2], c_ptr[2 * k_step_index_val][3]);
    a_ptr[2] = flash_pack(c_ptr[2 * k_step_index_val + 1][0], c_ptr[2 * k_step_index_val + 1][1]);
    a_ptr[3] = flash_pack(c_ptr[2 * k_step_index_val + 1][2], c_ptr[2 * k_step_index_val + 1][3]);
}

/// F2..F6 for FLASH_TILE queries of one head (the longest rows first), one warpgroup. The query tile stays in
/// shared memory; the block walks the keys FLASH_TILE at a time up to its own: TMA brings the key and value
/// tiles, then scores, the mask, the online softmax, output += attention_probs · value.
///
///   qkv_map_val                   TMA's description of qkv's tiles: {batch_size x seq_len, 3 · embed_dim}
///   output_ptr                    {batch_size, seq_len, embed_dim}: the head's columns written
///   logsumexp_ptr                 one float per row: written
///   num_heads_val, seq_len_val    the shape
///   scale_val                     1 / sqrt(FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_forward_kernel(const __grid_constant__ CUtensorMap qkv_map_val, nv_bfloat16* output_ptr,
                                    float* logsumexp_ptr, int num_heads_val, int seq_len_val, float scale_val) {
    namespace cde = cuda::device::experimental;
    // the tiles, in TMA's swizzled layout (each on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom), and the barrier
    // that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ FlashBarrier tiles_barrier_val;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW) * HARDWARE_WGMMA_COLUMNS_PER_THREAD;
    int query_block_index_val = gridDim.x - 1 - blockIdx.x;
    int batch_index_val = blockIdx.y / num_heads_val;
    int head_index_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
    // where this head sits in qkv as TMA sees it: its batch row's first token, and its first column of query;
    // its key and value follow, embed_dim apart
    int head_first_row_index_val = batch_index_val * seq_len_val;
    int query_first_column_index_val = head_index_val * FLASH_HEAD_SIZE;
    int key_first_column_index_val = query_first_column_index_val + embed_dim_val;
    int value_first_column_index_val = query_first_column_index_val + 2 * embed_dim_val;
    int warp_first_token_index_val = query_block_index_val * FLASH_TILE + warp_index_val * HARDWARE_WGMMA_M_PER_WARP;

    // 1. the barrier, by one thread; then this block's query tile, kept for the whole walk
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                 // every thread arrives once a load
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();
    if (threadIdx.x == 0)
        cde::cp_async_bulk_tensor_2d_global_to_shared(query_tile_list, &qkv_map_val, query_first_column_index_val,
                                                      head_first_row_index_val + query_block_index_val * FLASH_TILE, tiles_barrier_val);
    flash_wait_for_tiles(tiles_barrier_val, FLASH_TILE_BYTES);

    float output_list[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};    // {HARDWARE_WGMMA_M, FLASH_HEAD_SIZE}
    // rows group, group + HARDWARE_WGMMA_ROWS_PER_WARP
    float row_max_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {-INFINITY, -INFINITY};
    float row_sum_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {0.0f, 0.0f};
    // 2. walk the key tiles up to this block's own
    for (int key_block_index_val = 0; key_block_index_val <= query_block_index_val; key_block_index_val++) {
        __syncthreads();                                            // every thread is done with the last tiles
        if (threadIdx.x == 0) {
            int tile_first_row_index_val = head_first_row_index_val + key_block_index_val * FLASH_TILE;
            cde::cp_async_bulk_tensor_2d_global_to_shared(key_tile_list, &qkv_map_val, key_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(value_tile_list, &qkv_map_val, value_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
        }
        flash_wait_for_tiles(tiles_barrier_val, 2 * FLASH_TILE_BYTES);

        // F2: scores {HARDWARE_WGMMA_M, FLASH_TILE} = query tile · key tileᵀ, FLASH_K_STEPS WGMMAs across the head row
        float score_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_K_STEPS; k_step_index_val++)
            flash_multiply_add_tiles(score_list, query_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                     key_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
        flash_wgmma_wait();
        // F3: the scale and the mask; then each row's largest score
        float tile_max_list[HARDWARE_WGMMA_ROWS_PER_THREAD] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_TILE_GROUPS; group_column_index_val++)
#pragma unroll
            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++) {
                int thread_row_index_val = c_value_index_val / HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                int token_index_val = warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                int key_token_index_val = key_block_index_val * FLASH_TILE + group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS +
                                          pair_column_index_val + c_value_index_val % HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                score_list[group_column_index_val][c_value_index_val] =
                    key_token_index_val <= token_index_val ? score_list[group_column_index_val][c_value_index_val] * scale_val : -INFINITY;
                tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], score_list[group_column_index_val][c_value_index_val]);
            }
        // F4: the online softmax: the new max (over the row's HARDWARE_WGMMA_THREADS_PER_ROW lanes), the output and sum
        // moved onto it, then the attention_probs
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
            tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, tile_max_list[thread_row_index_val], 1));
            tile_max_list[thread_row_index_val] = fmaxf(tile_max_list[thread_row_index_val], __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, tile_max_list[thread_row_index_val], 2));
            float new_max_val = fmaxf(row_max_list[thread_row_index_val], tile_max_list[thread_row_index_val]);   // finite: every row sees key 0
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
        // F5: output += attention_probs · value tile: the attention_probs from the score registers as A, the value
        // tile turned round, FLASH_TILE_K_STEPS WGMMAs down its tokens
        uint32_t prob_a_list[FLASH_TILE_K_STEPS][HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16];
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_c_to_a(prob_a_list[k_step_index_val], score_list, k_step_index_val);
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_multiply_add_registers(output_list, prob_a_list[k_step_index_val],
                                         value_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16 * FLASH_HEAD_SIZE);
        flash_wgmma_wait();
    }

    // 3. F6: divide by the row sums, write the head's columns of output, and logsumexp
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
            logsumexp_ptr[(size_t)blockIdx.y * seq_len_val + token_index_val] = row_max_list[thread_row_index_val] + logf(row_sum_list[thread_row_index_val]);
    }
}

/// Works out row_dot = sum over the head's FLASH_HEAD_SIZE columns of output_grad · output, one warp per row.
///
///   row_dot_ptr                   one float per row: written
///   output_grad_ptr, output_ptr   {batch_size, seq_len, embed_dim}
///   num_heads_val, seq_len_val    the shape
///   rows_val                      batch_size x num_heads x seq_len
__global__ void attention_bf16_flash_row_dot_kernel(float* row_dot_ptr, const nv_bfloat16* output_grad_ptr,
                                                    const nv_bfloat16* output_ptr, int num_heads_val, int seq_len_val,
                                                    size_t rows_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        // 1. where the row is: which (batch, head), which token; each lane takes 2 neighbouring columns
        size_t batch_head_index_val = row_index_val / seq_len_val;                    // batch x num_heads + head
        size_t token_index_val = row_index_val % seq_len_val;
        size_t batch_index_val = batch_head_index_val / num_heads_val;
        size_t head_index_val = batch_head_index_val % num_heads_val;
        size_t lane_value_index_val = (batch_index_val * seq_len_val + token_index_val) * num_heads_val * FLASH_HEAD_SIZE + head_index_val * FLASH_HEAD_SIZE +
                        2 * lane_index_val;
        // 2. this lane's part of the dot, then the warp's
        float2 grad_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(output_grad_ptr + lane_value_index_val));
        float2 out_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(output_ptr + lane_value_index_val));
        float dot_val = grad_val.x * out_val.x + grad_val.y * out_val.y;
        dot_val = warp_reduce_sum(dot_val);
        if (lane_index_val == 0) row_dot_ptr[row_index_val] = dot_val;
    }
}

/// Works out key_grad and value_grad for FLASH_TILE keys of one head, one warpgroup. The key and value tiles stay
/// in shared memory; the block walks the queries FLASH_TILE at a time from its own tile on (TMA brings the query
/// and output_grad tiles), making the attention_probs again, turned round:
///     attention_probsᵀ       = exp(key tile · query tileᵀ / sqrt(head_size) - logsumexp)    0 where query < key
///     attention_probs_gradᵀ  = value tile · output_grad tileᵀ                               B2
///     scores_gradᵀ           = attention_probsᵀ (attention_probs_gradᵀ - row_dot)           B4-B6
///     value_grad            += attention_probsᵀ · output_grad tile                          B3
///     key_grad              += scores_gradᵀ · query tile                                    B8
/// The first two are sums over a head row (both tiles as they sit), issued together; the last two are sums over
/// the queries (A from registers, the tile turned round), issued together.
///
///   qkv_map_val                             TMA's description of qkv's tiles: {batch_size x seq_len, 3 · embed_dim}
///   output_grad_map_val                     and of output_grad's: {batch_size x seq_len, embed_dim}
///   qkv_grad_ptr                            the key and value parts written
///   logsumexp_ptr, row_dot_ptr              one float per row
///   num_heads_val, seq_len_val, scale_val   the shape, and 1 / sqrt(FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_key_grad_kernel(const __grid_constant__ CUtensorMap qkv_map_val,
                                     const __grid_constant__ CUtensorMap output_grad_map_val, nv_bfloat16* qkv_grad_ptr,
                                     const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                     int seq_len_val, float scale_val) {
    namespace cde = cuda::device::experimental;
    // the tiles, in TMA's swizzled layout, and the barrier that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 grad_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ float logsumexp_tile_list[FLASH_TILE];
    __shared__ float row_dot_tile_list[FLASH_TILE];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ FlashBarrier tiles_barrier_val;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW) * HARDWARE_WGMMA_COLUMNS_PER_THREAD;
    int key_block_index_val = blockIdx.x;
    int batch_index_val = blockIdx.y / num_heads_val;
    int head_index_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
    // where this head sits in qkv and output_grad as TMA sees them: its batch row's first token, and its first
    // column of query (and of output_grad); its key and value follow in qkv, embed_dim apart
    int head_first_row_index_val = batch_index_val * seq_len_val;
    int query_first_column_index_val = head_index_val * FLASH_HEAD_SIZE;
    int key_first_column_index_val = query_first_column_index_val + embed_dim_val;
    int value_first_column_index_val = query_first_column_index_val + 2 * embed_dim_val;
    int warp_first_key_index_val = key_block_index_val * FLASH_TILE + warp_index_val * HARDWARE_WGMMA_M_PER_WARP;

    // 1. the barrier, by one thread; then this block's key and value tiles, kept for the whole walk
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                 // every thread arrives once a load
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int tile_first_row_index_val = head_first_row_index_val + key_block_index_val * FLASH_TILE;
        cde::cp_async_bulk_tensor_2d_global_to_shared(key_tile_list, &qkv_map_val, key_first_column_index_val,
                                                      tile_first_row_index_val, tiles_barrier_val);
        cde::cp_async_bulk_tensor_2d_global_to_shared(value_tile_list, &qkv_map_val, value_first_column_index_val,
                                                      tile_first_row_index_val, tiles_barrier_val);
    }
    flash_wait_for_tiles(tiles_barrier_val, 2 * FLASH_TILE_BYTES);

    // 2. walk the query tiles from this block's own on
    float key_grad_list[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
    float value_grad_list[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
    for (int query_block_index_val = key_block_index_val; query_block_index_val < seq_len_val / FLASH_TILE; query_block_index_val++) {
        __syncthreads();                                            // every thread is done with the last tiles
        if (threadIdx.x == 0) {
            int tile_first_row_index_val = head_first_row_index_val + query_block_index_val * FLASH_TILE;
            cde::cp_async_bulk_tensor_2d_global_to_shared(query_tile_list, &qkv_map_val, query_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(grad_tile_list, &output_grad_map_val, query_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
        }
        if (threadIdx.x < FLASH_TILE) {
            size_t row_index_val = (size_t)blockIdx.y * seq_len_val + query_block_index_val * FLASH_TILE + threadIdx.x;
            logsumexp_tile_list[threadIdx.x] = logsumexp_ptr[row_index_val];
            row_dot_tile_list[threadIdx.x] = row_dot_ptr[row_index_val];
        }
        flash_wait_for_tiles(tiles_barrier_val, 2 * FLASH_TILE_BYTES);
        __syncthreads();                                            // the queries' logsumexp and row_dot are in too

        // the scores turned round {HARDWARE_WGMMA_M keys, FLASH_TILE queries} = key tile · query tileᵀ, and B2:
        // attention_probs_gradᵀ = value tile · output_grad tileᵀ
        float attention_prob_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        float score_grad_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_K_STEPS; k_step_index_val++) {
            flash_multiply_add_tiles(attention_prob_list, key_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                     query_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
            flash_multiply_add_tiles(score_grad_list, value_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                     grad_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
        }
        flash_wgmma_wait();
        // the attention_probs turned round, then B4-B6: scores_gradᵀ in attention_probs_gradᵀ's place
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_TILE_GROUPS; group_column_index_val++)
#pragma unroll
            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++) {
                int key_token_index_val = warp_first_key_index_val + group_index_val +
                                          (c_value_index_val / HARDWARE_WGMMA_COLUMNS_PER_THREAD) * HARDWARE_WGMMA_ROWS_PER_WARP;
                int column_index_val = group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + pair_column_index_val +
                                       c_value_index_val % HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                int token_index_val = query_block_index_val * FLASH_TILE + column_index_val;
                attention_prob_list[group_column_index_val][c_value_index_val] =
                    token_index_val >= key_token_index_val
                        ? __expf(attention_prob_list[group_column_index_val][c_value_index_val] * scale_val - logsumexp_tile_list[column_index_val])
                        : 0.0f;
                score_grad_list[group_column_index_val][c_value_index_val] =
                    attention_prob_list[group_column_index_val][c_value_index_val] *
                    (score_grad_list[group_column_index_val][c_value_index_val] - row_dot_tile_list[column_index_val]);
            }
        // B3: value_grad += attention_probsᵀ · output_grad tile, and B8: key_grad += scores_gradᵀ · query tile: A
        // from the registers, the tiles turned round, FLASH_TILE_K_STEPS WGMMAs each down the queries
        uint32_t prob_a_list[FLASH_TILE_K_STEPS][HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16];
        uint32_t score_grad_a_list[FLASH_TILE_K_STEPS][HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16];
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++) {
            flash_c_to_a(prob_a_list[k_step_index_val], attention_prob_list, k_step_index_val);
            flash_c_to_a(score_grad_a_list[k_step_index_val], score_grad_list, k_step_index_val);
        }
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++) {
            flash_multiply_add_registers(value_grad_list, prob_a_list[k_step_index_val],
                                         grad_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16 * FLASH_HEAD_SIZE);
            flash_multiply_add_registers(key_grad_list, score_grad_a_list[k_step_index_val],
                                         query_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16 * FLASH_HEAD_SIZE);
        }
        flash_wgmma_wait();
    }

    // 3. into qkv_grad: each token's row is query, key, value parts of embed_dim each
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        int key_token_index_val = warp_first_key_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        nv_bfloat16* row_ptr = qkv_grad_ptr + ((size_t)batch_index_val * seq_len_val + key_token_index_val) * 3 * embed_dim_val +
                               head_index_val * FLASH_HEAD_SIZE;
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_HEAD_GROUPS; group_column_index_val++) {
            int column_index_val = group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + pair_column_index_val;
            *reinterpret_cast<nv_bfloat162*>(row_ptr + embed_dim_val + column_index_val) = __floats2bfloat162_rn(
                key_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] * scale_val,
                key_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] * scale_val);
            *reinterpret_cast<nv_bfloat162*>(row_ptr + 2 * embed_dim_val + column_index_val) = __floats2bfloat162_rn(
                value_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val],
                value_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1]);
        }
    }
}

/// Works out query_grad for FLASH_TILE queries of one head, one warpgroup. The query and output_grad tiles stay
/// in shared memory; the block walks the keys FLASH_TILE at a time up to its own (TMA brings the key and value
/// tiles), making the attention_probs again:
///     attention_probs        = exp(query tile · key tileᵀ / sqrt(head_size) - logsumexp)    0 where key > query
///     attention_probs_grad   = output_grad tile · value tileᵀ                               B2
///     scores_grad            = attention_probs (attention_probs_grad - row_dot)             B4-B6
///     query_grad            += scores_grad · key tile                                       B7
///
///   qkv_grad_ptr   the query part written
///   the others     as in attention_bf16_flash_key_grad_kernel
__global__ void __launch_bounds__(FLASH_THREADS)
attention_bf16_flash_query_grad_kernel(const __grid_constant__ CUtensorMap qkv_map_val,
                                       const __grid_constant__ CUtensorMap output_grad_map_val, nv_bfloat16* qkv_grad_ptr,
                                       const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                       int seq_len_val, float scale_val) {
    namespace cde = cuda::device::experimental;
    // the tiles, in TMA's swizzled layout, and the barrier that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 grad_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_HEAD_SIZE];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ FlashBarrier tiles_barrier_val;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int group_index_val = lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int pair_column_index_val = (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW) * HARDWARE_WGMMA_COLUMNS_PER_THREAD;
    int query_block_index_val = gridDim.x - 1 - blockIdx.x;
    int batch_index_val = blockIdx.y / num_heads_val;
    int head_index_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
    // where this head sits in qkv and output_grad as TMA sees them: its batch row's first token, and its first
    // column of query (and of output_grad); its key and value follow in qkv, embed_dim apart
    int head_first_row_index_val = batch_index_val * seq_len_val;
    int query_first_column_index_val = head_index_val * FLASH_HEAD_SIZE;
    int key_first_column_index_val = query_first_column_index_val + embed_dim_val;
    int value_first_column_index_val = query_first_column_index_val + 2 * embed_dim_val;
    int warp_first_token_index_val = query_block_index_val * FLASH_TILE + warp_index_val * HARDWARE_WGMMA_M_PER_WARP;

    // 1. the barrier, by one thread; then this block's query and output_grad tiles, kept for the whole walk, and
    //    this thread's rows' logsumexp and row_dot
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                 // every thread arrives once a load
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int tile_first_row_index_val = head_first_row_index_val + query_block_index_val * FLASH_TILE;
        cde::cp_async_bulk_tensor_2d_global_to_shared(query_tile_list, &qkv_map_val, query_first_column_index_val,
                                                      tile_first_row_index_val, tiles_barrier_val);
        cde::cp_async_bulk_tensor_2d_global_to_shared(grad_tile_list, &output_grad_map_val, query_first_column_index_val,
                                                      tile_first_row_index_val, tiles_barrier_val);
    }
    flash_wait_for_tiles(tiles_barrier_val, 2 * FLASH_TILE_BYTES);
    // rows group, group + HARDWARE_WGMMA_ROWS_PER_WARP
    float row_logsumexp_list[HARDWARE_WGMMA_ROWS_PER_THREAD];
    float row_dot_of_row_list[HARDWARE_WGMMA_ROWS_PER_THREAD];
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        size_t row_index_val = (size_t)blockIdx.y * seq_len_val + warp_first_token_index_val + group_index_val +
                               thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        row_logsumexp_list[thread_row_index_val] = logsumexp_ptr[row_index_val];
        row_dot_of_row_list[thread_row_index_val] = row_dot_ptr[row_index_val];
    }

    // 2. walk the key tiles up to this block's own
    float query_grad_list[FLASH_HEAD_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
    for (int key_block_index_val = 0; key_block_index_val <= query_block_index_val; key_block_index_val++) {
        __syncthreads();                                            // every thread is done with the last tiles
        if (threadIdx.x == 0) {
            int tile_first_row_index_val = head_first_row_index_val + key_block_index_val * FLASH_TILE;
            cde::cp_async_bulk_tensor_2d_global_to_shared(key_tile_list, &qkv_map_val, key_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(value_tile_list, &qkv_map_val, value_first_column_index_val,
                                                          tile_first_row_index_val, tiles_barrier_val);
        }
        flash_wait_for_tiles(tiles_barrier_val, 2 * FLASH_TILE_BYTES);

        // the scores {HARDWARE_WGMMA_M queries, FLASH_TILE keys} = query tile · key tileᵀ, and B2:
        // attention_probs_grad = output_grad tile · value tileᵀ
        float attention_prob_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        float score_grad_list[FLASH_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_K_STEPS; k_step_index_val++) {
            flash_multiply_add_tiles(attention_prob_list, query_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                     key_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
            flash_multiply_add_tiles(score_grad_list, grad_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                     value_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
        }
        flash_wgmma_wait();
        // the attention_probs, then B4-B6: scores_grad in attention_probs_grad's place
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_TILE_GROUPS; group_column_index_val++)
#pragma unroll
            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++) {
                int thread_row_index_val = c_value_index_val / HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                int token_index_val = warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                int key_token_index_val = key_block_index_val * FLASH_TILE + group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS +
                                          pair_column_index_val + c_value_index_val % HARDWARE_WGMMA_COLUMNS_PER_THREAD;
                float attention_prob_val =
                    key_token_index_val <= token_index_val
                        ? __expf(attention_prob_list[group_column_index_val][c_value_index_val] * scale_val - row_logsumexp_list[thread_row_index_val])
                        : 0.0f;
                score_grad_list[group_column_index_val][c_value_index_val] =
                    attention_prob_val * (score_grad_list[group_column_index_val][c_value_index_val] - row_dot_of_row_list[thread_row_index_val]);
            }
        // B7: query_grad += scores_grad · key tile: A from the registers, the key tile turned round,
        // FLASH_TILE_K_STEPS WGMMAs down its tokens
        uint32_t score_grad_a_list[FLASH_TILE_K_STEPS][HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16];
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_c_to_a(score_grad_a_list[k_step_index_val], score_grad_list, k_step_index_val);
        flash_wgmma_begin();
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < FLASH_TILE_K_STEPS; k_step_index_val++)
            flash_multiply_add_registers(query_grad_list, score_grad_a_list[k_step_index_val],
                                         key_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16 * FLASH_HEAD_SIZE);
        flash_wgmma_wait();
    }

    // 3. into qkv_grad's query part
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        int token_index_val = warp_first_token_index_val + group_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        nv_bfloat16* row_ptr = qkv_grad_ptr + ((size_t)batch_index_val * seq_len_val + token_index_val) * 3 * embed_dim_val +
                               head_index_val * FLASH_HEAD_SIZE;
#pragma unroll
        for (int group_column_index_val = 0; group_column_index_val < FLASH_HEAD_GROUPS; group_column_index_val++)
            *reinterpret_cast<nv_bfloat162*>(row_ptr + group_column_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + pair_column_index_val) = __floats2bfloat162_rn(
                query_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] * scale_val,
                query_grad_list[group_column_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] * scale_val);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless flash attention can run these tensors: the shapes, bfloat16, logsumexp float,
/// attention_bf16_flash_fits, each bfloat16 tensor starting on one of TMA's HARDWARE_TMA_BYTES_PER_CELL-byte cells.
///
///   output_tensor      forward: output; backward: output_grad or output
///   logsumexp_tensor   {batch_size, num_heads, seq_len} float; backward: also row_dot. num_heads comes from here
///   qkv_tensor         forward: qkv; backward: qkv or qkv_grad
inline void attention_bf16_flash_check_shapes(const Tensor& output_tensor, const Tensor& logsumexp_tensor,
                                              const Tensor& qkv_tensor) {
    // 1. shapes, types and addresses together
    bool fits_val = logsumexp_tensor.dim() == 3 && output_tensor.dim() == 3 && logsumexp_tensor.size(1) > 0;
    if (fits_val) {
        size_t batch_size_val = logsumexp_tensor.size(0);
        size_t num_heads_val = logsumexp_tensor.size(1);
        size_t seq_len_val = logsumexp_tensor.size(2);
        size_t embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
        fits_val = attention_bf16_flash_fits(seq_len_val, output_tensor.size(2) / num_heads_val) &&
                   logsumexp_tensor.dtype() == DType::Float32 &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val});
        for (const Tensor* tensor_ptr : {&output_tensor, &qkv_tensor})
            fits_val = fits_val && tensor_ptr->dtype() == DType::BFloat16 &&
                       (size_t)tensor_ptr->data_ptr() % HARDWARE_TMA_BYTES_PER_CELL == 0;
    }
    // 2. any of them wrong: stop with an error
    if (!fits_val) {
        fprintf(stderr, "error: flash attention needs bfloat16 output {batch_size, seq_len, num_heads * %d} and qkv "
                        "{batch_size, seq_len, 3 * num_heads * %d} with seq_len a multiple of %d, a float logsumexp "
                        "{batch_size, num_heads, seq_len}, output and qkv starting on %d bytes, and a GPU of compute "
                        "capability %d.%d (an H100: WGMMA and TMA)
",
                FLASH_HEAD_SIZE, FLASH_HEAD_SIZE, FLASH_TILE, HARDWARE_TMA_BYTES_PER_CELL, HARDWARE_WGMMA_CAPABILITY / 10,
                HARDWARE_WGMMA_CAPABILITY % 10);
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_FLASH_BF16_CUH
