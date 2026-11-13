// attention_bf16.cuh — causal self-attention in bfloat16 by the steps: attention.cuh's math (F2..F5, B2..B8) with
// the attention_probs written out and kept for the backward, its six matrix products on an H100's tensor cores
// (WGMMA), fed by TMA. The math is in float; each value is rounded to bfloat16 only as it is stored.
//
//     tensor                 type in memory   shape                                         why
//     qkv, qkv_grad          bfloat16         {batch_size, seq_len, 3 · embed_dim}          the model's; qkv kept for backward
//     output, output_grad    bfloat16         {batch_size, seq_len, embed_dim}              the model's
//     attention_probs        bfloat16         {batch_size, num_heads, seq_len, seq_len}     kept for backward
//     attention_probs_grad   bfloat16         {batch_size, num_heads, seq_len, seq_len}     scratch
//     in registers           float            -                                             every sum and the softmax
//
// No copies in GPU memory, as in attention.cuh: query, key and value are read where they sit in qkv, and output
// and qkv_grad written straight into each head's columns. The products by attention_product below: TMA brings a
// tile of each side from where it sits into shared memory, in the swizzled layout the tensor cores read
// (wgmma_matmul_bf16.cuh's), and one WGMMA m64n64k16 adds the tiles' product into the block's sums. The softmax
// kernels read and write bfloat16 in place, one warp per row: no shared memory, so nothing for TMA to bring.
// Needs an H100 (WGMMA and TMA exist only on Hopper), built with -arch=sm_90a, and seq_len and head_size
// multiples of ATTENTION_BF16_SIZE_MULTIPLE: attention_bf16_fits. The same math without storing the
// attention_probs: attention_flash_bf16.cuh (flash attention: head_size 64, seq_len a multiple of 64); that with
// query, key and value kept in 8 bits: attention_flash_8bit.cuh.
#ifndef ATTENTION_BF16_CUH
#define ATTENTION_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum, HARDWARE_WGMMA_*, HARDWARE_TMA_*
#include "../../common/tensor.cuh"
#include "wgmma_matmul_bf16.cuh"     // the WGMMA descriptor and the tile x tile WGMMA: wgmma_matmul_bf16_multiply_add

#define ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)    // F4's default: one warp per row; fastest on an H100
#define ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)   // B4-B6's default: the same
#define ATTENTION_BF16_SIZE_MULTIPLE HARDWARE_WGMMA_K_BF16                                  // seq_len's and head_size's multiple: a sum is whole WGMMAs, 16 values of k each

// The products, as wgmma_matmul_bf16.cuh's: each block (one warpgroup) makes one
// ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS x ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS square of one head's
// product, walking the sum ATTENTION_PRODUCT_BLOCK_K at a time. The hardware's numbers (WARPGROUP_*, WGMMA_*,
// TMA_*) are in cuda_common.h. The index math is written for these numbers: change them together with it.
#define ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS HARDWARE_WGMMA_M                    // each block makes 64 rows: one WGMMA's M
#define ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS WGMMA_MATMUL_BF16_BLOCK_COLUMNS  // x 64 columns of a head's product: its N
#define ATTENTION_PRODUCT_THREADS_PER_BLOCK HARDWARE_WARPGROUP_THREADS              // the block size: one warpgroup, 128
#define ATTENTION_PRODUCT_BLOCK_K WGMMA_MATMUL_BF16_BLOCK_K                         // values of k a step loads: one swizzled row, 64
#define ATTENTION_PRODUCT_BLOCK_K_STEPS (ATTENTION_PRODUCT_BLOCK_K / HARDWARE_WGMMA_K_BF16)        // WGMMAs a step: 64 / 16 = 4
#define ATTENTION_PRODUCT_COLUMN_GROUPS (ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS)   // 8 groups across: 32 sums a thread
// A tile in shared memory, what one TMA copy brings: 64 lines of the tensor x 64 values along each, whichever
// way the side sits ({64 r or c, 64 k}, or turned {64 k, 64 r or c}): the block is a square and k steps by its side.
#define ATTENTION_PRODUCT_TILE_LINES ATTENTION_PRODUCT_BLOCK_K                      // a TMA box's rows: 64
#define ATTENTION_PRODUCT_TILE_LINE_VALUES WGMMA_MATMUL_BF16_ROW_VALUES             // a TMA box's row: one swizzled row, 64 values
#define ATTENTION_PRODUCT_TILE_VALUES (ATTENTION_PRODUCT_TILE_LINES * ATTENTION_PRODUCT_TILE_LINE_VALUES)   // 64 x 64 = 4,096
#define ATTENTION_PRODUCT_TILE_BYTES (ATTENTION_PRODUCT_TILE_VALUES * sizeof(nv_bfloat16))                  // 8 KB
#define ATTENTION_PRODUCT_LINE_MULTIPLE WGMMA_MATMUL_BF16_VALUE_MULTIPLE            // a tensor line's multiple: whole 16-byte cells, 8 values
// Double-buffered: one set of tiles (a left and a right) is read by this step's WGMMAs while TMA fills the other
// with the next step's: 2 sets x 2 tiles x 8 KB = 32 KB of shared memory.
#define ATTENTION_PRODUCT_TILE_SETS 2
static_assert(ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS == ATTENTION_PRODUCT_BLOCK_K &&
              ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS == ATTENTION_PRODUCT_BLOCK_K,
              "a tile is the same box turned or not: the square's side is a step's k");
static_assert(ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS == ATTENTION_PRODUCT_TILE_LINE_VALUES,
              "a turned tile's row (64 rows or columns) is one swizzled row, as a tile's row of 64 k is");
static_assert(ATTENTION_PRODUCT_COLUMN_GROUPS == WGMMA_MATMUL_BF16_COLUMN_GROUPS,
              "the products are one WGMMA m64n64k16: the sums wgmma_matmul_bf16_multiply_add takes");
static_assert(ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS % ATTENTION_BF16_SIZE_MULTIPLE == 0,
              "the mask's first and last sums fall on whole WGMMAs");
static_assert(ATTENTION_PRODUCT_TILE_SETS * 2 * ATTENTION_PRODUCT_TILE_BYTES < HARDWARE_DEFAULT_MAX_SHARED_BYTES_PER_BLOCK,
              "the tiles fit the shared memory a block has without asking");

/// The strides of one head's matrix in a tensor: how many values apart its batch rows, heads, rows and columns
/// sit in memory. Value [row][column] of head (batch, head) is at
///     first + batch · batch_stride + head · head_stride + row · row_stride + column · column_stride
/// The same matrix turned round (transposed) is row_stride and column_stride swapped: nothing moves.
struct AttentionStrides {
    size_t batch_stride_val;
    size_t head_stride_val;
    size_t row_stride_val;
    size_t column_stride_val;
};

/// Which part of a head's product the causal mask leaves to work out; the rest is 0, or never read.
enum AttentionProductMask {
    ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW,   // F2, B2 {token t, token j}: squares with every j after every t skipped
    ATTENTION_PRODUCT_SUM_UP_TO_ROW,       // F5, B7: the sum over key tokens j stops at the square's last t
    ATTENTION_PRODUCT_SUM_FROM_ROW,        // B3, B8: the sum over query tokens t starts at the square's first j
};

// ── forward and backward ─────────────────────────────────────────────────────
void attention_bf16_forward(const Tensor& output_tensor, const Tensor& attention_probs_tensor, const Tensor& qkv_tensor,
                            int softmax_threads_per_block_val = ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK);
void attention_bf16_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& qkv_tensor,
                             const Tensor& attention_probs_tensor, const Tensor& attention_probs_grad_tensor,
                             int softmax_threads_per_block_val = ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK);

// ── the product ────────────────────────────────────────────────────────────────
void attention_product(nv_bfloat16* product_ptr, AttentionStrides product_strides_val, const nv_bfloat16* left_ptr,
                       AttentionStrides left_strides_val, const nv_bfloat16* right_ptr, AttentionStrides right_strides_val,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, size_t batch_size_val,
                       size_t num_heads_val, float scale_val, AttentionProductMask mask_val);

// ── for both ──────────────────────────────────────────────────────────────────
bool attention_bf16_fits(size_t seq_len_val, size_t head_size_val);
int attention_bf16_blocks(size_t total_val, int threads_per_block_val, size_t per_thread_val);
void attention_bf16_check_shapes(const Tensor& output_tensor, const Tensor& attention_probs_tensor, const Tensor& qkv_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ────────────────────────────────────────────────────────────────────

__global__ void attention_bf16_softmax_forward_kernel(nv_bfloat16* attention_probs_ptr, size_t rows_val, size_t seq_len_val);

/// Writes output = causal self-attention of qkv, keeping attention_probs for the backward: checks, then runs F2..F5 in
/// bfloat16, TMA bringing query, key and value from where they sit in qkv to the tensor cores.
///
///   output_tensor                   {batch_size, seq_len, embed_dim}: written, each head its columns
///   attention_probs_tensor                  {batch_size, num_heads, seq_len, seq_len}: written; num_heads comes from here
///   qkv_tensor                      {batch_size, seq_len, 3 · embed_dim}: the qkv linear's output, kept for the
///                                   backward
///   softmax_threads_per_block_val   F4's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                   ATTENTION_BF16_SOFTMAX_FORWARD_THREADS_PER_BLOCK. Any size gives the same bits;
///                                   only the time changes. The products' block is always ATTENTION_PRODUCT_THREADS_PER_BLOCK.
inline void attention_bf16_forward(const Tensor& output_tensor, const Tensor& attention_probs_tensor, const Tensor& qkv_tensor,
                                   int softmax_threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(softmax_threads_per_block_val, "attention's softmax");
    attention_bf16_check_shapes(output_tensor, attention_probs_tensor, qkv_tensor);

    // 2. the sizes, and where each head's matrices sit
    size_t batch_size_val = attention_probs_tensor.size(0);
    size_t num_heads_val = attention_probs_tensor.size(1);
    size_t seq_len_val = attention_probs_tensor.size(2);
    size_t embed_dim_val = output_tensor.size(2);
    size_t head_size_val = embed_dim_val / num_heads_val;
    size_t rows_val = batch_size_val * num_heads_val * seq_len_val;                  // one per (batch, head, token)
    float scale_val = 1.0f / sqrtf((float)head_size_val);
    // each head's strides {batch, head, row, column}; a matrix turned round swaps the last two
    size_t qkv_row_length_val = 3 * embed_dim_val;                                // a token's row: query, key, value
    AttentionStrides qkv_strides_val = {seq_len_val * qkv_row_length_val, head_size_val, qkv_row_length_val, 1};
    AttentionStrides qkv_turned_strides_val = {seq_len_val * qkv_row_length_val, head_size_val, 1, qkv_row_length_val};
    AttentionStrides output_strides_val = {seq_len_val * embed_dim_val, head_size_val, embed_dim_val, 1};
    AttentionStrides attention_probs_strides_val = {num_heads_val * seq_len_val * seq_len_val, seq_len_val * seq_len_val, seq_len_val, 1};
    const nv_bfloat16* query_ptr = qkv_tensor.data_ptr<nv_bfloat16>();     // a qkv row: query, key, value, embed_dim each
    const nv_bfloat16* key_ptr = query_ptr + embed_dim_val;
    const nv_bfloat16* value_ptr = query_ptr + 2 * embed_dim_val;
    nv_bfloat16* attention_probs_ptr = attention_probs_tensor.data_ptr<nv_bfloat16>();

    // 3. F2: scores = scale · query · keyᵀ into attention_probs
    attention_product(attention_probs_ptr, attention_probs_strides_val, query_ptr, qkv_strides_val, key_ptr, qkv_turned_strides_val,
                      seq_len_val, seq_len_val, head_size_val, batch_size_val, num_heads_val, scale_val,
                      ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW);

    // 4. F3, F4: the mask and the softmax, in place, one warp per row
    attention_bf16_softmax_forward_kernel<<<attention_bf16_blocks(rows_val, softmax_threads_per_block_val, HARDWARE_THREADS_PER_WARP),
                                            softmax_threads_per_block_val>>>(attention_probs_ptr, rows_val, seq_len_val);
    cudaCheck(cudaGetLastError());

    // 5. F5: output = attention_probs · value, each head straight into its columns of output
    attention_product(output_tensor.data_ptr<nv_bfloat16>(), output_strides_val, (const nv_bfloat16*)attention_probs_ptr,
                      attention_probs_strides_val, value_ptr, qkv_strides_val, seq_len_val, head_size_val, seq_len_val,
                      batch_size_val, num_heads_val, 1.0f, ATTENTION_PRODUCT_SUM_UP_TO_ROW);
}

/// F3, F4: turns each row of scores into attention_probs in place, one warp per row, with classifier.cuh's online
/// softmax, in float. Columns after the row's token are set to 0; only columns up to it are read.
///
///   attention_probs_ptr    the scores before, the attention_probs after
///   rows_val       batch_size x num_heads x seq_len
///   seq_len_val    the row length
__global__ void attention_bf16_softmax_forward_kernel(nv_bfloat16* attention_probs_ptr, size_t rows_val, size_t seq_len_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        size_t token_index_val = row_index_val % seq_len_val;     // the query's token: it sees columns 0 .. token
        nv_bfloat16* row_attention_probs_ptr = attention_probs_ptr + row_index_val * seq_len_val;

        // 1. this thread's (max, sum), over its columns up to token
        float thread_max_val = -FLT_MAX;
        float thread_sum_val = 0.0f;
        for (size_t column_index_val = lane_index_val; column_index_val <= token_index_val; column_index_val += HARDWARE_THREADS_PER_WARP) {
            float score_val = to_float(row_attention_probs_ptr[column_index_val]);
            float new_max_val = fmaxf(thread_max_val, score_val);
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_max_val) + expf(score_val - new_max_val);
            thread_max_val = new_max_val;
        }

        // 2. the row's: log2(HARDWARE_THREADS_PER_WARP) swaps; a thread with no columns brings (-FLT_MAX, 0), which adds 0
        float row_max_val = thread_max_val;
        float row_sum_val = thread_sum_val;
        for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
            float other_max_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_sum_val, offset_val);
            float new_max_val = fmaxf(row_max_val, other_max_val);
            row_sum_val = row_sum_val * expf(row_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
            row_max_val = new_max_val;
        }

        // 3. the attention_probs, 0 after the token
        float one_over_row_sum_val = 1.0f / row_sum_val;
        for (size_t column_index_val = lane_index_val; column_index_val < seq_len_val; column_index_val += HARDWARE_THREADS_PER_WARP)
            row_attention_probs_ptr[column_index_val] = from_float<nv_bfloat16>(
                column_index_val <= token_index_val ? expf(to_float(row_attention_probs_ptr[column_index_val]) - row_max_val) * one_over_row_sum_val
                                        : 0.0f);
    }
}

// ── backward ────────────────────────────────────────────────────────────────────

__global__ void attention_bf16_softmax_backward_kernel(nv_bfloat16* attention_probs_grad_ptr, const nv_bfloat16* attention_probs_ptr,
                                                       size_t rows_val, size_t seq_len_val);

/// Writes qkv_grad (written over): checks, then runs B2..B8 in bfloat16, TMA bringing query, key and value from
/// where they sit in qkv to the tensor cores, each gradient written straight into its part of qkv_grad.
///
///   qkv_grad_tensor                 {batch_size, seq_len, 3 · embed_dim}: the result
///   output_grad_tensor              {batch_size, seq_len, embed_dim}: the gradient from the output linear
///   qkv_tensor, attention_probs_tensor      the forward's
///   attention_probs_grad_tensor             as attention_probs: scratch, attention_probs_grad after B2, scores_grad after B6
///   softmax_threads_per_block_val   B4-B6's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                   ATTENTION_BF16_SOFTMAX_BACKWARD_THREADS_PER_BLOCK. Any size gives the same
///                                   bits; only the time changes.
inline void attention_bf16_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                    const Tensor& qkv_tensor, const Tensor& attention_probs_tensor,
                                    const Tensor& attention_probs_grad_tensor, int softmax_threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(softmax_threads_per_block_val, "attention's softmax");
    attention_bf16_check_shapes(output_grad_tensor, attention_probs_tensor, qkv_tensor);
    attention_bf16_check_shapes(output_grad_tensor, attention_probs_grad_tensor, qkv_grad_tensor);

    // 2. the sizes, and where each head's matrices sit
    size_t batch_size_val = attention_probs_tensor.size(0);
    size_t num_heads_val = attention_probs_tensor.size(1);
    size_t seq_len_val = attention_probs_tensor.size(2);
    size_t embed_dim_val = output_grad_tensor.size(2);
    size_t head_size_val = embed_dim_val / num_heads_val;
    size_t rows_val = batch_size_val * num_heads_val * seq_len_val;
    float scale_val = 1.0f / sqrtf((float)head_size_val);
    // each head's strides {batch, head, row, column}; a matrix turned round swaps the last two
    size_t qkv_row_length_val = 3 * embed_dim_val;                                // a token's row: query, key, value
    AttentionStrides qkv_strides_val = {seq_len_val * qkv_row_length_val, head_size_val, qkv_row_length_val, 1};
    AttentionStrides qkv_turned_strides_val = {seq_len_val * qkv_row_length_val, head_size_val, 1, qkv_row_length_val};
    AttentionStrides output_strides_val = {seq_len_val * embed_dim_val, head_size_val, embed_dim_val, 1};
    AttentionStrides attention_probs_strides_val = {num_heads_val * seq_len_val * seq_len_val, seq_len_val * seq_len_val, seq_len_val, 1};
    AttentionStrides attention_probs_turned_strides_val = {num_heads_val * seq_len_val * seq_len_val, seq_len_val * seq_len_val, 1, seq_len_val};
    const nv_bfloat16* query_ptr = qkv_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* key_ptr = query_ptr + embed_dim_val;
    const nv_bfloat16* value_ptr = query_ptr + 2 * embed_dim_val;
    nv_bfloat16* query_grad_ptr = qkv_grad_tensor.data_ptr<nv_bfloat16>();          // qkv_grad's parts, as qkv's
    nv_bfloat16* key_grad_ptr = query_grad_ptr + embed_dim_val;
    nv_bfloat16* value_grad_ptr = query_grad_ptr + 2 * embed_dim_val;
    const nv_bfloat16* output_grad_ptr = output_grad_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* attention_probs_ptr = attention_probs_tensor.data_ptr<nv_bfloat16>();
    nv_bfloat16* attention_probs_grad_ptr = attention_probs_grad_tensor.data_ptr<nv_bfloat16>();

    // 3. B2: attention_probs_grad = output_grad · valueᵀ
    attention_product(attention_probs_grad_ptr, attention_probs_strides_val, output_grad_ptr, output_strides_val, value_ptr,
                      qkv_turned_strides_val, seq_len_val, seq_len_val, head_size_val, batch_size_val,
                      num_heads_val, 1.0f, ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW);

    // 4. B3: value_grad = attention_probsᵀ · output_grad, into qkv_grad's value part
    attention_product(value_grad_ptr, qkv_strides_val, attention_probs_ptr, attention_probs_turned_strides_val, output_grad_ptr,
                      output_strides_val, seq_len_val, head_size_val, seq_len_val, batch_size_val, num_heads_val, 1.0f,
                      ATTENTION_PRODUCT_SUM_FROM_ROW);

    // 5. B4-B6: the softmax's backward, in place, one warp per row
    attention_bf16_softmax_backward_kernel<<<attention_bf16_blocks(rows_val, softmax_threads_per_block_val, HARDWARE_THREADS_PER_WARP),
                                             softmax_threads_per_block_val>>>(attention_probs_grad_ptr, attention_probs_ptr, rows_val,
                                                                              seq_len_val);
    cudaCheck(cudaGetLastError());

    // 6. B7: query_grad = scale · scores_grad · key, into qkv_grad's query part
    const nv_bfloat16* scores_grad_ptr = attention_probs_grad_ptr;
    attention_product(query_grad_ptr, qkv_strides_val, scores_grad_ptr, attention_probs_strides_val, key_ptr, qkv_strides_val,
                      seq_len_val, head_size_val, seq_len_val, batch_size_val, num_heads_val, scale_val,
                      ATTENTION_PRODUCT_SUM_UP_TO_ROW);

    // 7. B8: key_grad = scale · scores_gradᵀ · query, into qkv_grad's key part
    attention_product(key_grad_ptr, qkv_strides_val, scores_grad_ptr, attention_probs_turned_strides_val, query_ptr,
                      qkv_strides_val, seq_len_val, head_size_val, seq_len_val, batch_size_val, num_heads_val, scale_val,
                      ATTENTION_PRODUCT_SUM_FROM_ROW);
}

/// B4-B6: turns each row of attention_probs_grad into scores_grad in place, one warp per row, in float:
///     scores_grad = attention_prob · (attention_probs_grad - row_dot), 0 after the row's token
/// Only columns up to the token are read.
///
///   attention_probs_grad_ptr   attention_probs_grad before, scores_grad after
///   attention_probs_ptr        the forward's
///   rows_val           batch_size x num_heads x seq_len
///   seq_len_val        the row length
__global__ void attention_bf16_softmax_backward_kernel(nv_bfloat16* attention_probs_grad_ptr, const nv_bfloat16* attention_probs_ptr,
                                                       size_t rows_val, size_t seq_len_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        size_t token_index_val = row_index_val % seq_len_val;
        nv_bfloat16* row_grad_ptr = attention_probs_grad_ptr + row_index_val * seq_len_val;
        const nv_bfloat16* row_attention_probs_ptr = attention_probs_ptr + row_index_val * seq_len_val;

        // 1. row_dot: this thread's part, then the warp's
        float row_dot_val = 0.0f;
        for (size_t column_index_val = lane_index_val; column_index_val <= token_index_val; column_index_val += HARDWARE_THREADS_PER_WARP)
            row_dot_val += to_float(row_attention_probs_ptr[column_index_val]) * to_float(row_grad_ptr[column_index_val]);
        row_dot_val = warp_reduce_sum(row_dot_val);

        // 2. scores_grad, 0 after the token
        for (size_t column_index_val = lane_index_val; column_index_val < seq_len_val; column_index_val += HARDWARE_THREADS_PER_WARP)
            row_grad_ptr[column_index_val] = from_float<nv_bfloat16>(
                column_index_val <= token_index_val
                    ? to_float(row_attention_probs_ptr[column_index_val]) * (to_float(row_grad_ptr[column_index_val]) - row_dot_val)
                    : 0.0f);
    }
}

// ── the product ───────────────────────────────────────────────────────────────

using AttentionProductBarrier = cuda::barrier<cuda::thread_scope_block>;

template <bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(ATTENTION_PRODUCT_THREADS_PER_BLOCK)
attention_product_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         nv_bfloat16* product_ptr, AttentionStrides product_strides_val, AttentionStrides left_strides_val,
                         AttentionStrides right_strides_val, int rows_val, int columns_val, int sum_length_val,
                         int num_heads_val, float scale_val, AttentionProductMask mask_val);
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__device__ __forceinline__ AttentionProductBarrier::arrival_token attention_product_ask_for_tiles(
        nv_bfloat16* left_tile_ptr, nv_bfloat16* right_tile_ptr, AttentionProductBarrier& tiles_barrier_val,
        const CUtensorMap* left_map_ptr, const CUtensorMap* right_map_ptr, int left_head_first_value_index_val,
        int left_head_first_line_index_val, int right_head_first_value_index_val, int right_head_first_line_index_val,
        int block_first_row_index_val, int block_first_column_index_val, int step_first_k_index_val);
CUtensorMap attention_product_tensor_map(const nv_bfloat16* matrix_ptr, size_t lines_val, size_t line_values_val,
                                         size_t line_stride_val);

// The six products, as product[r][c] = sum over k of left[r][k] · right[k][c], one per head (T = seq_len,
// HS = head_size); "sits as" is which index has its values next to each other in memory, the last one:
//     step   product                         r   c   k (summed)   left, sits as                   right, sits as
//     F2     scores {T, T}                   t   j   d            query {t, d}                    key {j, d}
//     F5     output {T, HS}                  t   d   j            attention_probs {t, j}          value {j, d}: turned
//     B2     attention_probs_grad {T, T}     t   j   d            output_grad {t, d}              value {j, d}
//     B3     value_grad {T, HS}              j   d   t            attention_probs {t, j}: turned  output_grad {t, d}: turned
//     B7     query_grad {T, HS}              t   d   j            scores_grad {t, j}              key {j, d}: turned
//     B8     key_grad {T, HS}                j   d   t            scores_grad {t, j}: turned      query {t, d}: turned
// A side with k's values next to each other, left {r, k} or right {c, k}, is as the tensor cores read it
// ("K-major"): TMA's box is {64 r or c, 64 k}. A side with k's values a line apart, left {k, r} or right {k, c},
// is turned ("MN-major"): TMA's box is {64 k, 64 r or c} and the WGMMA's transpose bit for that side is 1. So
// three kernels are made: neither side turned (F2, B2), right turned (F5, B7), both turned (B3, B8).

/// Writes product = scale · left · right for every head at once, written over, on the tensor cores: checks,
/// describes left's and right's tiles for TMA, then launches attention_product_kernel made for the two turned
/// flags, one block (one warpgroup) per square of each head's product, as one line of blocks: head after head,
/// each head's squares row after row. Each of the three matrices is read or written where it sits (its
/// AttentionStrides), so a head's columns of qkv or output need no copy. The block size is fixed by the tiles,
/// so this takes none.
///
///   product_ptr, left_ptr, right_ptr       each one's first value (a qkv part's first value for query, key, value),
///                                          on one of TMA's HARDWARE_TMA_BYTES_PER_CELL-byte cells
///   product_strides_val                    where each head's product sits: column stride 1
///   left_strides_val, right_strides_val    where each head's matrix sits; each with one stride 1, the other (a
///                                          line of the tensor) a multiple of ATTENTION_PRODUCT_LINE_MULTIPLE
///   rows_val, columns_val                  the product's shape: {rows, columns} per head
///   sum_length_val                         left's columns = right's rows
///                                          (all three multiples of ATTENTION_BF16_SIZE_MULTIPLE)
///   batch_size_val, num_heads_val          the heads: batch_size x num_heads
///   scale_val                              the scale: 1 / sqrt(head_size) or 1
///   mask_val                               which part of the product the causal mask leaves
/// Needs an H100 (WGMMA and TMA); nothing here checks the GPU: attention_bf16_fits does.
inline void attention_product(nv_bfloat16* product_ptr, AttentionStrides product_strides_val, const nv_bfloat16* left_ptr,
                              AttentionStrides left_strides_val, const nv_bfloat16* right_ptr,
                              AttentionStrides right_strides_val, size_t rows_val, size_t columns_val,
                              size_t sum_length_val, size_t batch_size_val, size_t num_heads_val, float scale_val,
                              AttentionProductMask mask_val) {
    // 1. how each side sits: turned when k's values are a line apart (left's column stride, right's row stride);
    //    a line of its tensor is then the other index's stride
    bool left_turned_val = left_strides_val.column_stride_val != 1;
    bool right_turned_val = right_strides_val.row_stride_val != 1;
    size_t left_line_stride_val = left_turned_val ? left_strides_val.column_stride_val : left_strides_val.row_stride_val;
    size_t right_line_stride_val = right_turned_val ? right_strides_val.row_stride_val : right_strides_val.column_stride_val;

    // 2. check the strides, the sizes and the addresses; any of them wrong: stop with an error
    bool fits_val = (left_strides_val.row_stride_val == 1 || left_strides_val.column_stride_val == 1) &&
                    (right_strides_val.row_stride_val == 1 || right_strides_val.column_stride_val == 1) &&
                    product_strides_val.column_stride_val == 1 &&
                    left_line_stride_val % ATTENTION_PRODUCT_LINE_MULTIPLE == 0 &&
                    right_line_stride_val % ATTENTION_PRODUCT_LINE_MULTIPLE == 0 &&
                    rows_val % ATTENTION_BF16_SIZE_MULTIPLE == 0 && columns_val % ATTENTION_BF16_SIZE_MULTIPLE == 0 &&
                    sum_length_val % ATTENTION_BF16_SIZE_MULTIPLE == 0 && sum_length_val > 0 &&
                    (size_t)left_ptr % HARDWARE_TMA_BYTES_PER_CELL == 0 && (size_t)right_ptr % HARDWARE_TMA_BYTES_PER_CELL == 0;
    if (!fits_val) {
        fprintf(stderr, "error: attention_product needs left and right each with a row or column stride of 1 and the "
                        "other a multiple of %d, product with a column stride of 1, rows, columns and sum_length "
                        "multiples of %d, and left and right starting on %d bytes; got rows %zu, columns %zu, "
                        "sum_length %zu, lines of %zu (left) and %zu (right) values\n",
                ATTENTION_PRODUCT_LINE_MULTIPLE, ATTENTION_BF16_SIZE_MULTIPLE, HARDWARE_TMA_BYTES_PER_CELL, rows_val,
                columns_val, sum_length_val, left_line_stride_val, right_line_stride_val);
        exit(EXIT_FAILURE);
    }

    // 3. TMA's description of left's and right's tensors, each as lines of values next to each other. The lines:
    //    every batch row's values / a line's. The values of a line TMA may read: the heads side by side in a
    //    line (num_heads · head stride: query's embed_dim of qkv's 3 · embed_dim), or the whole line when a head
    //    is whole lines (attention_probs); past them, 0s
    size_t left_line_values_val = left_strides_val.head_stride_val * num_heads_val < left_line_stride_val
                                      ? left_strides_val.head_stride_val * num_heads_val : left_line_stride_val;
    size_t right_line_values_val = right_strides_val.head_stride_val * num_heads_val < right_line_stride_val
                                       ? right_strides_val.head_stride_val * num_heads_val : right_line_stride_val;
    CUtensorMap left_map_val = attention_product_tensor_map(
        left_ptr, batch_size_val * left_strides_val.batch_stride_val / left_line_stride_val, left_line_values_val,
        left_line_stride_val);
    CUtensorMap right_map_val = attention_product_tensor_map(
        right_ptr, batch_size_val * right_strides_val.batch_stride_val / right_line_stride_val, right_line_values_val,
        right_line_stride_val);

    // 4. launch: squares across and down, rounded up, for every head, as one line of blocks
    size_t total_block_columns_val = CEIL_DIV(columns_val, (size_t)ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS);
    unsigned total_blocks_val = (unsigned)(total_block_columns_val * total_block_rows_val * batch_size_val * num_heads_val);
    if (!left_turned_val && !right_turned_val)
        attention_product_kernel<false, false><<<total_blocks_val, ATTENTION_PRODUCT_THREADS_PER_BLOCK>>>(
            left_map_val, right_map_val, product_ptr, product_strides_val, left_strides_val, right_strides_val,
            (int)rows_val, (int)columns_val, (int)sum_length_val, (int)num_heads_val, scale_val, mask_val);
    else if (!left_turned_val && right_turned_val)
        attention_product_kernel<false, true><<<total_blocks_val, ATTENTION_PRODUCT_THREADS_PER_BLOCK>>>(
            left_map_val, right_map_val, product_ptr, product_strides_val, left_strides_val, right_strides_val,
            (int)rows_val, (int)columns_val, (int)sum_length_val, (int)num_heads_val, scale_val, mask_val);
    else if (left_turned_val && !right_turned_val)
        attention_product_kernel<true, false><<<total_blocks_val, ATTENTION_PRODUCT_THREADS_PER_BLOCK>>>(
            left_map_val, right_map_val, product_ptr, product_strides_val, left_strides_val, right_strides_val,
            (int)rows_val, (int)columns_val, (int)sum_length_val, (int)num_heads_val, scale_val, mask_val);
    else
        attention_product_kernel<true, true><<<total_blocks_val, ATTENTION_PRODUCT_THREADS_PER_BLOCK>>>(
            left_map_val, right_map_val, product_ptr, product_strides_val, left_strides_val, right_strides_val,
            (int)rows_val, (int)columns_val, (int)sum_length_val, (int)num_heads_val, scale_val, mask_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of one head's product with one warpgroup. The block walks the sum
/// ATTENTION_PRODUCT_BLOCK_K at a time: TMA brings a tile of left and a tile of right into shared memory, then the
/// tensor cores add their product into the sums in registers (WGMMA m64n64k16, read straight from shared memory).
/// Double-buffered: TMA fills the next step's tiles during this step's WGMMAs. Sums in float.
///
///   left_map_val, right_map_val                           TMA's descriptions of left's and right's tensors
///   product_ptr, product_strides_val, left_strides_val,
///   right_strides_val                                     as in attention_product
///   rows_val, columns_val, sum_length_val                 the shapes
///   num_heads_val                                         to tell batch and head from the block's number
///   scale_val, mask_val                                   as in attention_product
/// LEFT_TURNED, RIGHT_TURNED: how left and right sit: which corner TMA's box starts at, how far apart two WGMMAs'
/// HARDWARE_WGMMA_K_BF16 values of k are in a tile, and the WGMMA's transpose bits.
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(ATTENTION_PRODUCT_THREADS_PER_BLOCK)
attention_product_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         nv_bfloat16* product_ptr, AttentionStrides product_strides_val, AttentionStrides left_strides_val,
                         AttentionStrides right_strides_val, int rows_val, int columns_val, int sum_length_val,
                         int num_heads_val, float scale_val, AttentionProductMask mask_val) {
    namespace cde = cuda::device::experimental;
    // two sets of tiles, in TMA's swizzled layout (each tile on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom), and
    // each set's barrier, which says its tiles arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 left_tiles_list[ATTENTION_PRODUCT_TILE_SETS][ATTENTION_PRODUCT_TILE_VALUES];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 right_tiles_list[ATTENTION_PRODUCT_TILE_SETS][ATTENTION_PRODUCT_TILE_VALUES];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ AttentionProductBarrier tiles_barriers_list[ATTENTION_PRODUCT_TILE_SETS];
    // where a WGMMA's HARDWARE_WGMMA_K_BF16 values of k start in a tile: that many values along a row, or turned
    // that many rows on
    const int left_k_step_values_val = LEFT_TURNED ? HARDWARE_WGMMA_K_BF16 * ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS : HARDWARE_WGMMA_K_BF16;
    const int right_k_step_values_val = RIGHT_TURNED ? HARDWARE_WGMMA_K_BF16 * ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS : HARDWARE_WGMMA_K_BF16;

    // 1. the block's head and square, from its number: head after head, each head's squares row after row
    int total_block_columns_val = CEIL_DIV(columns_val, ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS);
    int blocks_per_head_val = total_block_columns_val * CEIL_DIV(rows_val, ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS);
    int head_in_batch_index_val = blockIdx.x / blocks_per_head_val;               // batch x num_heads + head
    int block_index_in_head_val = blockIdx.x % blocks_per_head_val;
    int block_row_index_val = block_index_in_head_val / total_block_columns_val;
    int block_column_index_val = block_index_in_head_val % total_block_columns_val;
    int block_first_row_index_val = block_row_index_val * ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS;             // the block's first r
    int block_first_column_index_val = block_column_index_val * ATTENTION_PRODUCT_VALUES_PER_BLOCK_COLUMNS;    // the block's first c
    size_t batch_index_val = head_in_batch_index_val / num_heads_val;
    size_t head_index_val = head_in_batch_index_val % num_heads_val;

    // 2. what the causal mask leaves: a square of scores all after its tokens is never read (the whole block
    //    leaves at once); a sum over attention_probs or scores_grad starts or stops where they turn 0
    if (mask_val == ATTENTION_PRODUCT_COLUMNS_UP_TO_ROW &&
        block_first_column_index_val > block_first_row_index_val + ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS - 1)
        return;
    int first_k_index_val = mask_val == ATTENTION_PRODUCT_SUM_FROM_ROW ? block_first_row_index_val : 0;
    int k_end_index_val = sum_length_val;                                         // one past the last k
    if (mask_val == ATTENTION_PRODUCT_SUM_UP_TO_ROW && block_first_row_index_val + ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS < sum_length_val)
        k_end_index_val = block_first_row_index_val + ATTENTION_PRODUCT_VALUES_PER_BLOCK_ROWS;

    // 3. where the head sits in left's and right's tensors as TMA sees them: its first value is so many values
    //    from the tensor's first, which is so many whole lines and so many values along a line
    size_t left_line_stride_val = LEFT_TURNED ? left_strides_val.column_stride_val : left_strides_val.row_stride_val;
    size_t right_line_stride_val = RIGHT_TURNED ? right_strides_val.row_stride_val : right_strides_val.column_stride_val;
    size_t left_head_first_val = batch_index_val * left_strides_val.batch_stride_val + head_index_val * left_strides_val.head_stride_val;
    size_t right_head_first_val = batch_index_val * right_strides_val.batch_stride_val + head_index_val * right_strides_val.head_stride_val;
    int left_head_first_line_index_val = (int)(left_head_first_val / left_line_stride_val);
    int left_head_first_value_index_val = (int)(left_head_first_val % left_line_stride_val);
    int right_head_first_line_index_val = (int)(right_head_first_val / right_line_stride_val);
    int right_head_first_value_index_val = (int)(right_head_first_val % right_line_stride_val);

    // 4. the barriers, by one thread
    if (threadIdx.x == 0) {
#pragma unroll
        for (int set_index_val = 0; set_index_val < ATTENTION_PRODUCT_TILE_SETS; set_index_val++)
            init(&tiles_barriers_list[set_index_val], blockDim.x);    // every thread arrives once a step
        cde::fence_proxy_async_shared_cta();                          // the barriers visible to TMA
    }
    __syncthreads();

    // 5. walk the sum: TMA brings each step's tiles one step ahead, the tensor cores add their product
    float sums_list[ATTENTION_PRODUCT_COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of the square
    int total_steps_val = CEIL_DIV(k_end_index_val - first_k_index_val, ATTENTION_PRODUCT_BLOCK_K);
    // each set's token: two variables, not an array, so they stay in registers (an array picked by a step's set,
    // known only at run time, goes to local memory)
    static_assert(ATTENTION_PRODUCT_TILE_SETS == 2, "the tokens are written for two sets of tiles");
    AttentionProductBarrier::arrival_token set_0_token_val;
    AttentionProductBarrier::arrival_token set_1_token_val;
    set_0_token_val = attention_product_ask_for_tiles<LEFT_TURNED, RIGHT_TURNED>(
        left_tiles_list[0], right_tiles_list[0], tiles_barriers_list[0], &left_map_val, &right_map_val,
        left_head_first_value_index_val, left_head_first_line_index_val, right_head_first_value_index_val,
        right_head_first_line_index_val, block_first_row_index_val, block_first_column_index_val, first_k_index_val);
    for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        int set_index_val = step_index_val % ATTENTION_PRODUCT_TILE_SETS;             // this step reads one set; TMA fills the other
        int next_set_index_val = (step_index_val + 1) % ATTENTION_PRODUCT_TILE_SETS;
        int step_first_k_index_val = first_k_index_val + step_index_val * ATTENTION_PRODUCT_BLOCK_K;
        // a. the next step's tiles, asked for now: they arrive during the WGMMAs below. The other set is free:
        //    the step before this one read it, and every thread is past that step's end
        if (step_index_val + 1 < total_steps_val)
            (next_set_index_val == 0 ? set_0_token_val : set_1_token_val) = attention_product_ask_for_tiles<LEFT_TURNED, RIGHT_TURNED>(
                left_tiles_list[next_set_index_val], right_tiles_list[next_set_index_val], tiles_barriers_list[next_set_index_val],
                &left_map_val, &right_map_val, left_head_first_value_index_val, left_head_first_line_index_val,
                right_head_first_value_index_val, right_head_first_line_index_val, block_first_row_index_val,
                block_first_column_index_val, step_first_k_index_val + ATTENTION_PRODUCT_BLOCK_K);

        // b. wait until this step's tiles are in
        tiles_barriers_list[set_index_val].wait(std::move(set_index_val == 0 ? set_0_token_val : set_1_token_val));

        // c. the tensor cores: one WGMMA per HARDWARE_WGMMA_K_BF16 values of k the step has, at most
        //    ATTENTION_PRODUCT_BLOCK_K_STEPS. A tile's values of k past the sum's end are another head's, or
        //    values the mask turned to 0: no WGMMA reads them
        int step_k_values_val = k_end_index_val - step_first_k_index_val < ATTENTION_PRODUCT_BLOCK_K
                                    ? k_end_index_val - step_first_k_index_val : ATTENTION_PRODUCT_BLOCK_K;
        int step_k_steps_val = step_k_values_val / HARDWARE_WGMMA_K_BF16;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < ATTENTION_PRODUCT_BLOCK_K_STEPS; k_step_index_val++)
            if (k_step_index_val < step_k_steps_val)
                wgmma_matmul_bf16_multiply_add<LEFT_TURNED, RIGHT_TURNED>(
                    sums_list, left_tiles_list[set_index_val] + k_step_index_val * left_k_step_values_val,
                    right_tiles_list[set_index_val] + k_step_index_val * right_k_step_values_val);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // ... done

        // d. every thread done with this set before the next step lets TMA fill it again
        __syncthreads();
    }

    // 6. product = scale · sums, written over, two neighbouring columns at a time; nothing past the edges
    nv_bfloat16* head_product_ptr = product_ptr + batch_index_val * product_strides_val.batch_stride_val + head_index_val * product_strides_val.head_stride_val;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int thread_first_row_index_val = block_first_row_index_val + warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
#pragma unroll
    for (int group_index_val = 0; group_index_val < ATTENTION_PRODUCT_COLUMN_GROUPS; group_index_val++) {
        int column_index_val = block_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
        if (column_index_val >= columns_val) continue;       // columns is even, so column + 1 is in too
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
            int row_index_val = thread_first_row_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
            if (row_index_val >= rows_val) continue;
            *reinterpret_cast<nv_bfloat162*>(head_product_ptr + (size_t)row_index_val * product_strides_val.row_stride_val + column_index_val) =
                __floats2bfloat162_rn(scale_val * sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val],
                                      scale_val * sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1]);
        }
    }
}

/// Asks TMA for one step's tile of left and tile of right, to be copied into one set's shared memory, and arrives
/// at that set's barrier: one thread sends the two copies and says how many bytes are coming, the others only
/// arrive. Gives each thread's token, to wait with once the tiles are needed. Past the tensor's edges TMA gives 0s.
/// A box's corner is (value along a line, line) of the tensor as it sits: (k, r or c), or turned (r or c, k).
///
///   left_tile_ptr, right_tile_ptr, tiles_barrier_val     one set's tiles and its barrier
///   left_map_ptr, right_map_ptr                          TMA's descriptions of left's and right's tensors
///   left_head_first_value_index_val, left_head_first_line_index_val,
///   right_head_first_value_index_val, right_head_first_line_index_val   where the head's first value sits in each
///   block_first_row_index_val, block_first_column_index_val             the block's first r and c
///   step_first_k_index_val                                              the step's first k
template <bool LEFT_TURNED, bool RIGHT_TURNED>
__device__ __forceinline__ AttentionProductBarrier::arrival_token attention_product_ask_for_tiles(
        nv_bfloat16* left_tile_ptr, nv_bfloat16* right_tile_ptr, AttentionProductBarrier& tiles_barrier_val,
        const CUtensorMap* left_map_ptr, const CUtensorMap* right_map_ptr, int left_head_first_value_index_val,
        int left_head_first_line_index_val, int right_head_first_value_index_val, int right_head_first_line_index_val,
        int block_first_row_index_val, int block_first_column_index_val, int step_first_k_index_val) {
    namespace cde = cuda::device::experimental;
    if (threadIdx.x != 0) return tiles_barrier_val.arrive();
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        left_tile_ptr, left_map_ptr,
        left_head_first_value_index_val + (LEFT_TURNED ? block_first_row_index_val : step_first_k_index_val),
        left_head_first_line_index_val + (LEFT_TURNED ? step_first_k_index_val : block_first_row_index_val), tiles_barrier_val);
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        right_tile_ptr, right_map_ptr,
        right_head_first_value_index_val + (RIGHT_TURNED ? block_first_column_index_val : step_first_k_index_val),
        right_head_first_line_index_val + (RIGHT_TURNED ? step_first_k_index_val : block_first_column_index_val), tiles_barrier_val);
    return cuda::device::barrier_arrive_tx(tiles_barrier_val, 1, 2 * ATTENTION_PRODUCT_TILE_BYTES);
}

/// Gives TMA's description of a tensor's tiles: bfloat16 lines of values next to each other, read in boxes of
/// {ATTENTION_PRODUCT_TILE_LINES lines, ATTENTION_PRODUCT_TILE_LINE_VALUES values} written into shared memory in
/// the 128-byte swizzle. A line may be only the first part of the tensor's (query's embed_dim of qkv's
/// 3 · embed_dim): TMA reads line_values of each line and jumps line_stride to the next; past line_values or past
/// the last line, 0s. Made by the driver's cuTensorMapEncodeTiled, found at run time, so the program needs no
/// -lcuda. As wgmma_matmul_bf16_tensor_map, with a line's stride apart from its length.
///
///   matrix_ptr          the first line's first value, on a HARDWARE_TMA_BYTES_PER_CELL-byte cell
///   lines_val           the lines
///   line_values_val     the values of a line TMA may read
///   line_stride_val     values from one line to the next: a multiple of ATTENTION_PRODUCT_LINE_MULTIPLE
inline CUtensorMap attention_product_tensor_map(const nv_bfloat16* matrix_ptr, size_t lines_val, size_t line_values_val,
                                                size_t line_stride_val) {
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function_ptr, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: attention_product could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the tensor and its boxes, the fastest-moving dimension first: along a line, then the lines
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {line_values_val, lines_val};
    cuuint64_t line_bytes_list[1] = {line_stride_val * sizeof(nv_bfloat16)};              // from one line to the next
    cuuint32_t box_list[2] = {ATTENTION_PRODUCT_TILE_LINE_VALUES, ATTENTION_PRODUCT_TILE_LINES};
    cuuint32_t element_steps_list[2] = {1, 1};                                            // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bfloat16*>(matrix_ptr),
                                     sizes_list, line_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for %zu lines of %zu values, %zu apart, at %p\n",
                (int)result_val, lines_val, line_values_val, line_stride_val, (const void*)matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Tells if the attention steps can run here: seq_len and head_size multiples of ATTENTION_BF16_SIZE_MULTIPLE
/// (every sum is whole WGMMAs), and a GPU with WGMMA and TMA (compute capability HARDWARE_WGMMA_CAPABILITY / 10
/// exactly: an H100).
///
///   seq_len_val, head_size_val   the tensors'
inline bool attention_bf16_fits(size_t seq_len_val, size_t head_size_val) {
    static int capability_val = [] {
        int device_index_val = 0;
        int major_val = 0;
        int minor_val = 0;
        cudaCheck(cudaGetDevice(&device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, device_index_val));
        return 10 * major_val + minor_val;                  // compute capability x 10, as HARDWARE_WGMMA_CAPABILITY
    }();
    return seq_len_val > 0 && head_size_val > 0 && seq_len_val % ATTENTION_BF16_SIZE_MULTIPLE == 0 &&
           head_size_val % ATTENTION_BF16_SIZE_MULTIPLE == 0 && capability_val == HARDWARE_WGMMA_CAPABILITY;
}

/// Gives the blocks for `total_val` pieces of work, rounded up.
///
///   total_val               pieces of work: rows
///   threads_per_block_val   the launch's
///   per_thread_val          threads per piece: HARDWARE_THREADS_PER_WARP for a row (one warp)
inline int attention_bf16_blocks(size_t total_val, int threads_per_block_val, size_t per_thread_val) {
    size_t threads_val = total_val * per_thread_val;
    return (int)CEIL_DIV(threads_val, (size_t)threads_per_block_val);
}

/// Stops with a message unless the tensors fit: the shapes at the top of the file, all bfloat16,
/// attention_bf16_fits, each tensor starting on one of TMA's HARDWARE_TMA_BYTES_PER_CELL-byte cells.
///
///   output_tensor, attention_probs_tensor, qkv_tensor   forward: these; backward: the grads in their place
inline void attention_bf16_check_shapes(const Tensor& output_tensor, const Tensor& attention_probs_tensor, const Tensor& qkv_tensor) {
    // 1. shapes: as at the top of the file, num_heads from attention_probs, embed_dim a whole number of heads
    bool fits_val = attention_probs_tensor.dim() == 4 && output_tensor.dim() == 3 && attention_probs_tensor.size(1) > 0 &&
                    attention_probs_tensor.size(2) == attention_probs_tensor.size(3);
    size_t num_heads_val = fits_val ? attention_probs_tensor.size(1) : 1;
    size_t seq_len_val = fits_val ? attention_probs_tensor.size(2) : 0;
    size_t embed_dim_val = fits_val ? output_tensor.size(2) : 0;
    if (fits_val) {
        size_t batch_size_val = attention_probs_tensor.size(0);
        fits_val = embed_dim_val > 0 && embed_dim_val % num_heads_val == 0 &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val});
    }
    if (!fits_val) {
        fprintf(stderr, "error: attention needs attention_probs {batch_size, num_heads, seq_len, seq_len}, output {batch_size, "
                        "seq_len, embed_dim} and qkv {batch_size, seq_len, 3 * embed_dim}, embed_dim a multiple of "
                        "num_heads\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16 (float: attention.cuh)
    if (output_tensor.dtype() != DType::BFloat16 || attention_probs_tensor.dtype() != DType::BFloat16 ||
        qkv_tensor.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: attention_bf16 needs its tensors all bfloat16; for float, attention.cuh\n");
        exit(EXIT_FAILURE);
    }
    // 3. the sizes and the GPU the tensor cores and TMA take, and where TMA can start
    fits_val = attention_bf16_fits(seq_len_val, embed_dim_val / num_heads_val);
    for (const Tensor* tensor_ptr : {&output_tensor, &attention_probs_tensor, &qkv_tensor})
        fits_val = fits_val && (size_t)tensor_ptr->data_ptr() % HARDWARE_TMA_BYTES_PER_CELL == 0;
    if (!fits_val) {
        fprintf(stderr, "error: attention_bf16 needs seq_len and head_size multiples of %d (got %zu and %zu), every "
                        "tensor starting on %d bytes, and a GPU of compute capability %d.%d (an H100: WGMMA and TMA)\n",
                ATTENTION_BF16_SIZE_MULTIPLE, seq_len_val, embed_dim_val / num_heads_val, HARDWARE_TMA_BYTES_PER_CELL,
                HARDWARE_WGMMA_CAPABILITY / 10, HARDWARE_WGMMA_CAPABILITY % 10);
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_BF16_CUH
