// attention.cuh — causal self-attention in float by flash attention: each token mixes the values of the tokens
// up to it, per head, without ever storing the attention_probs:
//
//     tensor                 type in memory   shape                                         why
//     qkv, qkv_grad          float            {batch_size, seq_len, 3 · embed_dim}          the model's; qkv kept for backward
//     output, output_grad    float            {batch_size, seq_len, embed_dim}              the model's; output kept for backward
//     logsumexp, row_dot     float            {batch_size, num_heads, seq_len}              one per row: logsumexp kept for
//                                                                                           backward, row_dot scratch
//
// The math, per head (j <= t only: the causal mask):
//     F2   score[t][j]           = query[t] · key[j] / sqrt(head_size)
//     F4   attention_prob[t][j]  = exp(score[t][j] - logsumexp[t]),  logsumexp[t] = log(sum over j of exp(score[t][j]))
//     F5   output[t]             = sum over j of attention_prob[t][j] · value[j]
//     B2   attention_probs_grad[t][j] = output_grad[t] · value[j]
//     B3   value_grad[j]         = sum over t of attention_prob[t][j] · output_grad[t]
//     B4-6 scores_grad[t][j]     = attention_prob[t][j] · (attention_probs_grad[t][j] - row_dot[t]),
//                                  row_dot[t] = output_grad[t] · output[t]
//     B7   query_grad[t]         = sum over j of scores_grad[t][j] · key[j] / sqrt(head_size)
//     B8   key_grad[j]           = sum over t of scores_grad[t][j] · query[t] / sqrt(head_size)
// Math: https://ducbachsong.github.io/posts/attention-backward/
//
// A block of ATTENTION_FLASH_THREADS takes ATTENTION_FLASH_TILE queries and walks the keys ATTENTION_FLASH_TILE at
// a time, with classifier.cuh's online softmax; it keeps logsumexp = row_max + log(row_sum) per row, from which
// the backward makes the attention_probs again, tile by tile. No copies: each head's query, key and value tiles
// are read where they sit in qkv, and its output and gradients written straight into its columns of output and
// qkv_grad. All in float on the CUDA cores (float has no tensor-core product; TF32's would round to 10 bits).
// Needs head_size ATTENTION_FLASH_HEAD_SIZE and seq_len a multiple of ATTENTION_FLASH_TILE, on any GPU; other
// sizes stop with an error. Its block size is fixed by the tiles.
//
// bfloat16 (the steps, which keep the attention_probs, and flash attention on the tensor cores):
// attention_bf16.cuh (the steps), attention_flash_bf16.cuh (flash); with query, key and value in 8 bits: attention_flash_8bit.cuh.
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../../common/tensor.cuh"

// ── flash attention: head_size ATTENTION_FLASH_HEAD_SIZE, seq_len a multiple of ATTENTION_FLASH_TILE ──
// A tile row (one query or key) is shared by ATTENTION_FLASH_ROW_LANES neighbouring lanes of one warp: lane l
// of a row takes the tile's columns l, l + ROW_LANES, ... (keys or queries) and the head's columns l,
// l + ROW_LANES, ... (output, gradients). Neighbouring lanes then read neighbouring banks of shared memory.
#define ATTENTION_FLASH_HEAD_SIZE 64                                                     // the only head_size it takes: GPT-2's
#define ATTENTION_FLASH_TILE 32                                                          // queries per block, and keys per step
#define ATTENTION_FLASH_THREADS (8 * HARDWARE_THREADS_PER_WARP)                          // the block size
#define ATTENTION_FLASH_ROW_LANES (ATTENTION_FLASH_THREADS / ATTENTION_FLASH_TILE)       // lanes per tile row: 8
#define ATTENTION_FLASH_TILE_VALUES (ATTENTION_FLASH_TILE / ATTENTION_FLASH_ROW_LANES)   // scores of a row per lane: 4
#define ATTENTION_FLASH_HEAD_VALUES (ATTENTION_FLASH_HEAD_SIZE / ATTENTION_FLASH_ROW_LANES)   // head columns per lane: 8
#define ATTENTION_FLASH_STRIDE (ATTENTION_FLASH_HEAD_SIZE + 1)                           // a tile row in shared memory: one spare float, rows on different banks
#define ATTENTION_FLASH_SCORES_STRIDE (ATTENTION_FLASH_TILE + 1)                         // a row of a tile's attention_probs, the same
#define ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)        // row_dot's default: one warp per row
static_assert(HARDWARE_THREADS_PER_WARP % ATTENTION_FLASH_ROW_LANES == 0, "a tile row's lanes must be in one warp");
static_assert(ATTENTION_FLASH_HEAD_SIZE % VALUES_PER_FLOAT4 == 0, "the tiles are copied a float4 at a time");
bool attention_flash_fits(size_t seq_len_val, size_t head_size_val);
void attention_flash_forward(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor);
void attention_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& output_tensor,
                              const Tensor& qkv_tensor, const Tensor& logsumexp_tensor, const Tensor& row_dot_tensor,
                              int threads_per_block_val = ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
void attention_flash_check_shapes(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── flash attention ───────────────────────────────────────────────────────────

__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_forward_kernel(float* output_ptr, float* logsumexp_ptr, const float* qkv_ptr, int num_heads_val,
                               int seq_len_val, float scale_val);
__global__ void attention_flash_row_dot_kernel(float* row_dot_ptr, const float* output_grad_ptr, const float* output_ptr,
                                               int num_heads_val, int seq_len_val, size_t rows_val);
__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_key_grad_kernel(float* qkv_grad_ptr, const float* output_grad_ptr, const float* qkv_ptr,
                                const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val, int seq_len_val,
                                float scale_val);
__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_query_grad_kernel(float* qkv_grad_ptr, const float* output_grad_ptr, const float* qkv_ptr,
                                  const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val, int seq_len_val,
                                  float scale_val);

/// Tells if flash attention can run these sizes: head_size ATTENTION_FLASH_HEAD_SIZE, seq_len a multiple of
/// ATTENTION_FLASH_TILE. Any GPU: it uses no tensor cores.
///
///   seq_len_val, head_size_val   the tensors'
inline bool attention_flash_fits(size_t seq_len_val, size_t head_size_val) {
    return head_size_val == ATTENTION_FLASH_HEAD_SIZE && seq_len_val > 0 && seq_len_val % ATTENTION_FLASH_TILE == 0;
}

/// Writes output as attention_forward, by flash attention: one kernel for F2..F6, one block per
/// ATTENTION_FLASH_TILE queries of a head, its tiles read straight from qkv.
///
///   output_tensor        {batch_size, seq_len, embed_dim}: written, each head its columns; kept for the backward
///   logsumexp_tensor     {batch_size, num_heads, seq_len} float: kept in place of the attention_probs; num_heads comes
///                        from here
///   qkv_tensor           {batch_size, seq_len, 3 · embed_dim}: the qkv linear's output, kept for the backward
/// Its block size is always ATTENTION_FLASH_THREADS: the tiles fix it.
inline void attention_flash_forward(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    attention_flash_check_shapes(output_tensor, logsumexp_tensor, qkv_tensor);
    size_t num_heads_val = logsumexp_tensor.size(1);
    size_t seq_len_val = logsumexp_tensor.size(2);
    // 2. F2..F6: one block per ATTENTION_FLASH_TILE queries of each head: rows / ATTENTION_FLASH_TILE blocks
    unsigned total_blocks_val = (unsigned)(logsumexp_tensor.numel() / ATTENTION_FLASH_TILE);
    attention_flash_forward_kernel<<<total_blocks_val, ATTENTION_FLASH_THREADS>>>(
        output_tensor.data(), logsumexp_tensor.data(), qkv_tensor.data(), (int)num_heads_val, (int)seq_len_val,
        1.0f / sqrtf((float)ATTENTION_FLASH_HEAD_SIZE));
    cudaCheck(cudaGetLastError());
}

/// Writes qkv_grad as attention_backward, by flash attention: row_dot (one warp per row), then key_grad and
/// value_grad (one block per ATTENTION_FLASH_TILE keys), then query_grad (one block per ATTENTION_FLASH_TILE
/// queries), every tile read straight from qkv and output_grad.
///
///   qkv_grad_tensor                       the result, written over
///   output_grad_tensor                    the gradient from the output linear
///   output_tensor, qkv_tensor,
///   logsumexp_tensor                      the forward's
///   row_dot_tensor                        {batch_size, num_heads, seq_len} float: scratch
///   threads_per_block_val                 row_dot's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                                         given: ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK. Any size gives the same
///                                         bits. The tile kernels' is always ATTENTION_FLASH_THREADS.
inline void attention_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                     const Tensor& output_tensor, const Tensor& qkv_tensor,
                                     const Tensor& logsumexp_tensor, const Tensor& row_dot_tensor,
                                     int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "flash attention");
    attention_flash_check_shapes(output_grad_tensor, logsumexp_tensor, qkv_tensor);
    attention_flash_check_shapes(output_tensor, row_dot_tensor, qkv_grad_tensor);
    size_t num_heads_val = logsumexp_tensor.size(1);
    size_t seq_len_val = logsumexp_tensor.size(2);
    size_t rows_val = logsumexp_tensor.numel();                               // one per (batch, head, token)
    float scale_val = 1.0f / sqrtf((float)ATTENTION_FLASH_HEAD_SIZE);
    // 2. row_dot, one warp per row: rows x HARDWARE_THREADS_PER_WARP threads, rounded up to whole blocks
    int row_dot_blocks_val = (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val);
    attention_flash_row_dot_kernel<<<row_dot_blocks_val, threads_per_block_val>>>(
        row_dot_tensor.data(), output_grad_tensor.data(), output_tensor.data(), (int)num_heads_val, (int)seq_len_val,
        rows_val);
    cudaCheck(cudaGetLastError());
    // 3. key_grad and value_grad, then query_grad: one block per tile of each head
    unsigned total_blocks_val = (unsigned)(rows_val / ATTENTION_FLASH_TILE);
    attention_flash_key_grad_kernel<<<total_blocks_val, ATTENTION_FLASH_THREADS>>>(
        qkv_grad_tensor.data(), output_grad_tensor.data(), qkv_tensor.data(), logsumexp_tensor.data(),
        row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
    attention_flash_query_grad_kernel<<<total_blocks_val, ATTENTION_FLASH_THREADS>>>(
        qkv_grad_tensor.data(), output_grad_tensor.data(), qkv_tensor.data(), logsumexp_tensor.data(),
        row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
}

/// Copies a {ATTENTION_FLASH_TILE, ATTENTION_FLASH_HEAD_SIZE} tile into shared memory, one float4 per load,
/// its rows ATTENTION_FLASH_STRIDE apart.
///
///   tile_ptr         shared memory
///   source_ptr       the tile's first value in GPU memory
///   row_stride_val   values from row to row: 3 · embed_dim for query, key, value (in qkv); embed_dim for
///                    output_grad
__device__ __forceinline__ void attention_flash_load_tile(float* tile_ptr, const float* source_ptr, size_t row_stride_val) {
    const int float4s_per_row_val = ATTENTION_FLASH_HEAD_SIZE / VALUES_PER_FLOAT4;
    for (int piece_index_val = threadIdx.x; piece_index_val < ATTENTION_FLASH_TILE * float4s_per_row_val; piece_index_val += ATTENTION_FLASH_THREADS) {
        int tile_row_index_val = piece_index_val / float4s_per_row_val;
        int column_index_val = (piece_index_val % float4s_per_row_val) * VALUES_PER_FLOAT4;
        float4 values_val = *reinterpret_cast<const float4*>(source_ptr + tile_row_index_val * row_stride_val + column_index_val);
        float* row_ptr = tile_ptr + tile_row_index_val * ATTENTION_FLASH_STRIDE + column_index_val;   // the spare float breaks float4s: one at a time
        row_ptr[0] = values_val.x;
        row_ptr[1] = values_val.y;
        row_ptr[2] = values_val.z;
        row_ptr[3] = values_val.w;
    }
}

/// Gives the dot product of two tile rows in shared memory, over the head's ATTENTION_FLASH_HEAD_SIZE columns.
///
///   first_row_ptr, second_row_ptr   the rows' first values
__device__ __forceinline__ float attention_flash_dot(const float* first_row_ptr, const float* second_row_ptr) {
    float sum_val = 0.0f;
#pragma unroll
    for (int column_index_val = 0; column_index_val < ATTENTION_FLASH_HEAD_SIZE; column_index_val++)
        sum_val += first_row_ptr[column_index_val] * second_row_ptr[column_index_val];
    return sum_val;
}

/// Gives the largest, or the sum, of a value over the ATTENTION_FLASH_ROW_LANES lanes of one tile row:
/// log2(ATTENTION_FLASH_ROW_LANES) swaps, every lane of the row getting the result.
///
///   value_val   this lane's
///   max_val     true: the largest; false: the sum
__device__ __forceinline__ float attention_flash_row_reduce(float value_val, bool max_val) {
#pragma unroll
    for (int offset_val = ATTENTION_FLASH_ROW_LANES / 2; offset_val > 0; offset_val /= 2) {
        float other_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, value_val, offset_val);
        value_val = max_val ? fmaxf(value_val, other_val) : value_val + other_val;
    }
    return value_val;
}

/// F2..F6 for ATTENTION_FLASH_TILE queries of one head (the longest rows first). Each query row is shared by
/// ATTENTION_FLASH_ROW_LANES lanes, which keep its output columns in registers and walk the keys
/// ATTENTION_FLASH_TILE at a time up to its own: scores, the mask, the online softmax, output += attention_probs · value.
///
///   output_ptr                    {batch_size, seq_len, embed_dim}: the head's columns written
///   logsumexp_ptr                 one float per row: written
///   qkv_ptr                       {batch_size, seq_len, 3 · embed_dim}: the head's columns of query, key, value read
///   num_heads_val, seq_len_val    the shape
///   scale_val                     1 / sqrt(ATTENTION_FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_forward_kernel(float* output_ptr, float* logsumexp_ptr, const float* qkv_ptr, int num_heads_val,
                               int seq_len_val, float scale_val) {
    __shared__ float query_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float key_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float value_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float attention_probs_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_SCORES_STRIDE];
    // 1. where the block and this thread are
    int tiles_per_head_val = seq_len_val / ATTENTION_FLASH_TILE;
    int heads_in_batch_val = gridDim.x / tiles_per_head_val;                     // batch_size x num_heads
    int head_in_batch_index_val = blockIdx.x % heads_in_batch_val;              // batch x num_heads + head
    int tile_order_index_val = blockIdx.x / heads_in_batch_val;                 // every head's tile 0, then every head's tile 1, ...
    int query_block_index_val = tiles_per_head_val - 1 - tile_order_index_val;   // the longest rows first
    int batch_index_val = head_in_batch_index_val / num_heads_val;
    int head_index_val = head_in_batch_index_val % num_heads_val;
    int embed_dim_val = num_heads_val * ATTENTION_FLASH_HEAD_SIZE;
    size_t qkv_row_length_val = 3 * (size_t)embed_dim_val;                       // values from token to token in qkv
    // this head's query of token 0 in qkv; its key and value follow, embed_dim apart
    const float* head_query_ptr = qkv_ptr + (size_t)batch_index_val * seq_len_val * qkv_row_length_val + head_index_val * ATTENTION_FLASH_HEAD_SIZE;
    const float* head_key_ptr = head_query_ptr + embed_dim_val;
    const float* head_value_ptr = head_query_ptr + 2 * embed_dim_val;
    int tile_row_index_val = threadIdx.x / ATTENTION_FLASH_ROW_LANES;            // this thread's query in the tile
    int row_lane_index_val = threadIdx.x % ATTENTION_FLASH_ROW_LANES;            // which of the row's lanes it is
    int token_index_val = query_block_index_val * ATTENTION_FLASH_TILE + tile_row_index_val;
    const float* query_row_ptr = query_tile_list + tile_row_index_val * ATTENTION_FLASH_STRIDE;
    float* attention_probs_row_ptr = attention_probs_tile_list + tile_row_index_val * ATTENTION_FLASH_SCORES_STRIDE;

    // 2. the block's queries into shared memory
    attention_flash_load_tile(query_tile_list, head_query_ptr + (size_t)query_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                              qkv_row_length_val);

    // 3. walk the key tiles up to this block's own
    float output_list[ATTENTION_FLASH_HEAD_VALUES] = {};      // the row's columns row_lane + ROW_LANES · j
    float row_max_val = -INFINITY;
    float thread_sum_val = 0.0f;                              // this lane's part of the row's sum
    for (int key_block_index_val = 0; key_block_index_val <= query_block_index_val; key_block_index_val++) {
        __syncthreads();                                      // the queries copied; every thread done with the last tiles
        attention_flash_load_tile(key_tile_list, head_key_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                                  qkv_row_length_val);
        attention_flash_load_tile(value_tile_list, head_value_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                                  qkv_row_length_val);
        __syncthreads();

        // F2, F3: this lane's scores (keys row_lane + ROW_LANES · i), -infinity after the token; the row's largest
        float score_list[ATTENTION_FLASH_TILE_VALUES];
        float tile_max_val = -INFINITY;
#pragma unroll
        for (int value_index_val = 0; value_index_val < ATTENTION_FLASH_TILE_VALUES; value_index_val++) {
            int key_in_tile_index_val = row_lane_index_val + ATTENTION_FLASH_ROW_LANES * value_index_val;
            int key_token_index_val = key_block_index_val * ATTENTION_FLASH_TILE + key_in_tile_index_val;
            score_list[value_index_val] = key_token_index_val <= token_index_val
                ? scale_val * attention_flash_dot(query_row_ptr, key_tile_list + key_in_tile_index_val * ATTENTION_FLASH_STRIDE)
                : -INFINITY;
            tile_max_val = fmaxf(tile_max_val, score_list[value_index_val]);
        }
        tile_max_val = attention_flash_row_reduce(tile_max_val, true);
        // F4: the online softmax: output and sum moved onto the new max, then the attention_probs
        float new_max_val = fmaxf(row_max_val, tile_max_val);         // finite: every row sees key 0 in the first tile
        float correction_val = expf(row_max_val - new_max_val);       // 0 on the first tile
        row_max_val = new_max_val;
        thread_sum_val *= correction_val;
#pragma unroll
        for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++)
            output_list[head_value_index_val] *= correction_val;
#pragma unroll
        for (int value_index_val = 0; value_index_val < ATTENTION_FLASH_TILE_VALUES; value_index_val++) {
            float attention_prob_val = expf(score_list[value_index_val] - row_max_val);   // a masked score gives 0
            thread_sum_val += attention_prob_val;
            attention_probs_row_ptr[row_lane_index_val + ATTENTION_FLASH_ROW_LANES * value_index_val] = attention_prob_val;
        }
        __syncwarp();                                         // the row's attention_probs whole: its lanes are in this warp
        // F5: output += attention_probs · value tile
        for (int key_in_tile_index_val = 0; key_in_tile_index_val < ATTENTION_FLASH_TILE; key_in_tile_index_val++) {
            float attention_prob_val = attention_probs_row_ptr[key_in_tile_index_val];
            const float* value_row_ptr = value_tile_list + key_in_tile_index_val * ATTENTION_FLASH_STRIDE;
#pragma unroll
            for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++)
                output_list[head_value_index_val] += attention_prob_val * value_row_ptr[row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val];
        }
    }

    // 4. F6: divide by the row's sum, write the head's columns of output, and logsumexp
    float row_sum_val = attention_flash_row_reduce(thread_sum_val, false);
    float one_over_row_sum_val = 1.0f / row_sum_val;
    float* output_row_ptr = output_ptr + ((size_t)batch_index_val * seq_len_val + token_index_val) * embed_dim_val + head_index_val * ATTENTION_FLASH_HEAD_SIZE;
#pragma unroll
    for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++)
        output_row_ptr[row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val] = output_list[head_value_index_val] * one_over_row_sum_val;
    if (row_lane_index_val == 0)
        logsumexp_ptr[(size_t)head_in_batch_index_val * seq_len_val + token_index_val] = row_max_val + logf(row_sum_val);
}

/// Works out row_dot = sum over the head's ATTENTION_FLASH_HEAD_SIZE columns of output_grad · output, one warp
/// per row: B4's row_dot, the same number as sum over j of attention_prob · attention_probs_grad.
///
///   row_dot_ptr                   one float per row: written
///   output_grad_ptr, output_ptr   {batch_size, seq_len, embed_dim}
///   num_heads_val, seq_len_val    the shape
///   rows_val                      batch_size x num_heads x seq_len
__global__ void attention_flash_row_dot_kernel(float* row_dot_ptr, const float* output_grad_ptr, const float* output_ptr,
                                               int num_heads_val, int seq_len_val, size_t rows_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        // 1. where the row is: which (batch, head), which token
        size_t batch_head_index_val = row_index_val / seq_len_val;                    // batch x num_heads + head
        size_t token_index_val = row_index_val % seq_len_val;
        size_t batch_index_val = batch_head_index_val / num_heads_val;
        size_t head_index_val = batch_head_index_val % num_heads_val;
        size_t row_first_index_val = (batch_index_val * seq_len_val + token_index_val) * num_heads_val * ATTENTION_FLASH_HEAD_SIZE +
                                     head_index_val * ATTENTION_FLASH_HEAD_SIZE;
        // 2. this lane's columns of the dot, then the warp's
        float dot_val = 0.0f;
        for (int column_index_val = lane_index_val; column_index_val < ATTENTION_FLASH_HEAD_SIZE; column_index_val += HARDWARE_THREADS_PER_WARP)
            dot_val += output_grad_ptr[row_first_index_val + column_index_val] * output_ptr[row_first_index_val + column_index_val];
        dot_val = warp_reduce_sum(dot_val);
        if (lane_index_val == 0) row_dot_ptr[row_index_val] = dot_val;
    }
}

/// Works out key_grad and value_grad for ATTENTION_FLASH_TILE keys of one head. Each key row is shared by
/// ATTENTION_FLASH_ROW_LANES lanes, which keep its gradients' columns in registers and walk the queries
/// ATTENTION_FLASH_TILE at a time from its own tile on, making the attention_probs again, turned round:
///     attention_probsᵀ       = exp(key · queryᵀ / sqrt(head_size) - logsumexp)    0 where query < key
///     value_grad    += attention_probsᵀ · output_grad tile                        B3
///     attention_probs_gradᵀ  = value · output_grad tileᵀ                          B2
///     scores_gradᵀ   = attention_probsᵀ (attention_probs_gradᵀ - row_dot)                 B4-B6
///     key_grad      += scores_gradᵀ · query tile                          B8
///
///   qkv_grad_ptr                            the key and value parts written
///   output_grad_ptr                         {batch_size, seq_len, embed_dim}
///   qkv_ptr                                 {batch_size, seq_len, 3 · embed_dim}: the forward's
///   logsumexp_ptr, row_dot_ptr              one float per row
///   num_heads_val, seq_len_val, scale_val   the shape, and 1 / sqrt(ATTENTION_FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_key_grad_kernel(float* qkv_grad_ptr, const float* output_grad_ptr, const float* qkv_ptr,
                                const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val, int seq_len_val,
                                float scale_val) {
    __shared__ float key_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float value_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float query_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float grad_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float attention_probs_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_SCORES_STRIDE];       // turned: [key][query]
    __shared__ float scores_grad_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_SCORES_STRIDE];   // the same
    __shared__ float logsumexp_tile_list[ATTENTION_FLASH_TILE];
    __shared__ float row_dot_tile_list[ATTENTION_FLASH_TILE];
    // 1. where the block and this thread are
    int tiles_per_head_val = seq_len_val / ATTENTION_FLASH_TILE;
    int heads_in_batch_val = gridDim.x / tiles_per_head_val;                     // batch_size x num_heads
    int head_in_batch_index_val = blockIdx.x % heads_in_batch_val;              // batch x num_heads + head
    int tile_order_index_val = blockIdx.x / heads_in_batch_val;                 // every head's tile 0, then every head's tile 1, ...
    int key_block_index_val = tile_order_index_val;                             // key tile 0 walks the most query tiles
    int batch_index_val = head_in_batch_index_val / num_heads_val;
    int head_index_val = head_in_batch_index_val % num_heads_val;
    int embed_dim_val = num_heads_val * ATTENTION_FLASH_HEAD_SIZE;
    size_t qkv_row_length_val = 3 * (size_t)embed_dim_val;
    const float* head_query_ptr = qkv_ptr + (size_t)batch_index_val * seq_len_val * qkv_row_length_val + head_index_val * ATTENTION_FLASH_HEAD_SIZE;
    const float* head_key_ptr = head_query_ptr + embed_dim_val;
    const float* head_value_ptr = head_query_ptr + 2 * embed_dim_val;
    int tile_row_index_val = threadIdx.x / ATTENTION_FLASH_ROW_LANES;            // this thread's key in the tile
    int row_lane_index_val = threadIdx.x % ATTENTION_FLASH_ROW_LANES;
    int key_token_index_val = key_block_index_val * ATTENTION_FLASH_TILE + tile_row_index_val;
    const float* key_row_ptr = key_tile_list + tile_row_index_val * ATTENTION_FLASH_STRIDE;
    const float* value_row_ptr = value_tile_list + tile_row_index_val * ATTENTION_FLASH_STRIDE;
    float* attention_probs_row_ptr = attention_probs_tile_list + tile_row_index_val * ATTENTION_FLASH_SCORES_STRIDE;
    float* scores_grad_row_ptr = scores_grad_tile_list + tile_row_index_val * ATTENTION_FLASH_SCORES_STRIDE;

    // 2. the block's keys and their values into shared memory, kept for the whole walk
    attention_flash_load_tile(key_tile_list, head_key_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val, qkv_row_length_val);
    attention_flash_load_tile(value_tile_list, head_value_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val, qkv_row_length_val);

    // 3. walk the query tiles from this block's own on
    float key_grad_list[ATTENTION_FLASH_HEAD_VALUES] = {};    // the row's columns row_lane + ROW_LANES · j
    float value_grad_list[ATTENTION_FLASH_HEAD_VALUES] = {};
    for (int query_block_index_val = key_block_index_val; query_block_index_val < seq_len_val / ATTENTION_FLASH_TILE; query_block_index_val++) {
        __syncthreads();                                      // the keys copied; every thread done with the last tiles
        attention_flash_load_tile(query_tile_list, head_query_ptr + (size_t)query_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                                  qkv_row_length_val);
        attention_flash_load_tile(grad_tile_list,
                                  output_grad_ptr + ((size_t)batch_index_val * seq_len_val + query_block_index_val * ATTENTION_FLASH_TILE) * embed_dim_val +
                                      head_index_val * ATTENTION_FLASH_HEAD_SIZE,
                                  embed_dim_val);
        if (threadIdx.x < ATTENTION_FLASH_TILE) {
            size_t row_index_val = (size_t)head_in_batch_index_val * seq_len_val + query_block_index_val * ATTENTION_FLASH_TILE + threadIdx.x;
            logsumexp_tile_list[threadIdx.x] = logsumexp_ptr[row_index_val];
            row_dot_tile_list[threadIdx.x] = row_dot_ptr[row_index_val];
        }
        __syncthreads();

        // the attention_probs again, B2 and B4-B6, for this lane's queries (row_lane + ROW_LANES · i)
#pragma unroll
        for (int value_index_val = 0; value_index_val < ATTENTION_FLASH_TILE_VALUES; value_index_val++) {
            int query_in_tile_index_val = row_lane_index_val + ATTENTION_FLASH_ROW_LANES * value_index_val;
            int token_index_val = query_block_index_val * ATTENTION_FLASH_TILE + query_in_tile_index_val;
            float attention_prob_val = token_index_val >= key_token_index_val
                ? expf(scale_val * attention_flash_dot(key_row_ptr, query_tile_list + query_in_tile_index_val * ATTENTION_FLASH_STRIDE) -
                       logsumexp_tile_list[query_in_tile_index_val])
                : 0.0f;
            float attention_probs_grad_val = attention_flash_dot(value_row_ptr, grad_tile_list + query_in_tile_index_val * ATTENTION_FLASH_STRIDE);
            attention_probs_row_ptr[query_in_tile_index_val] = attention_prob_val;
            scores_grad_row_ptr[query_in_tile_index_val] = attention_prob_val * (attention_probs_grad_val - row_dot_tile_list[query_in_tile_index_val]);
        }
        __syncwarp();                                         // the row's attention_probs and scores_grad whole
        // B3: value_grad += attention_probsᵀ · output_grad tile; B8: key_grad += scores_gradᵀ · query tile
        for (int query_in_tile_index_val = 0; query_in_tile_index_val < ATTENTION_FLASH_TILE; query_in_tile_index_val++) {
            float attention_prob_val = attention_probs_row_ptr[query_in_tile_index_val];
            float scores_grad_val = scores_grad_row_ptr[query_in_tile_index_val];
            const float* grad_row_ptr = grad_tile_list + query_in_tile_index_val * ATTENTION_FLASH_STRIDE;
            const float* query_row_ptr = query_tile_list + query_in_tile_index_val * ATTENTION_FLASH_STRIDE;
#pragma unroll
            for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++) {
                int column_index_val = row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val;
                value_grad_list[head_value_index_val] += attention_prob_val * grad_row_ptr[column_index_val];
                key_grad_list[head_value_index_val] += scores_grad_val * query_row_ptr[column_index_val];
            }
        }
    }

    // 4. into qkv_grad: each token's row is query, key, value parts of embed_dim each
    float* row_ptr = qkv_grad_ptr + ((size_t)batch_index_val * seq_len_val + key_token_index_val) * qkv_row_length_val +
                     head_index_val * ATTENTION_FLASH_HEAD_SIZE;
#pragma unroll
    for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++) {
        int column_index_val = row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val;
        row_ptr[embed_dim_val + column_index_val] = key_grad_list[head_value_index_val] * scale_val;
        row_ptr[2 * embed_dim_val + column_index_val] = value_grad_list[head_value_index_val];
    }
}

/// Works out query_grad for ATTENTION_FLASH_TILE queries of one head. Each query row is shared by
/// ATTENTION_FLASH_ROW_LANES lanes, which keep its gradient's columns in registers and walk the keys
/// ATTENTION_FLASH_TILE at a time up to its own, making the attention_probs again:
///     attention_probs        = exp(query · key tileᵀ / sqrt(head_size) - logsumexp)    0 where key > query
///     attention_probs_grad   = output_grad · value tileᵀ                               B2
///     scores_grad    = attention_probs (attention_probs_grad - row_dot)                        B4-B6
///     query_grad    += scores_grad · key tile                                  B7
///
///   qkv_grad_ptr   the query part written
///   the others     as in attention_flash_key_grad_kernel
__global__ void __launch_bounds__(ATTENTION_FLASH_THREADS)
attention_flash_query_grad_kernel(float* qkv_grad_ptr, const float* output_grad_ptr, const float* qkv_ptr,
                                  const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val, int seq_len_val,
                                  float scale_val) {
    __shared__ float query_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float grad_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float key_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float value_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_STRIDE];
    __shared__ float scores_grad_tile_list[ATTENTION_FLASH_TILE * ATTENTION_FLASH_SCORES_STRIDE];
    // 1. where the block and this thread are
    int tiles_per_head_val = seq_len_val / ATTENTION_FLASH_TILE;
    int heads_in_batch_val = gridDim.x / tiles_per_head_val;                     // batch_size x num_heads
    int head_in_batch_index_val = blockIdx.x % heads_in_batch_val;              // batch x num_heads + head
    int tile_order_index_val = blockIdx.x / heads_in_batch_val;                 // every head's tile 0, then every head's tile 1, ...
    int query_block_index_val = tiles_per_head_val - 1 - tile_order_index_val;   // the longest rows first
    int batch_index_val = head_in_batch_index_val / num_heads_val;
    int head_index_val = head_in_batch_index_val % num_heads_val;
    int embed_dim_val = num_heads_val * ATTENTION_FLASH_HEAD_SIZE;
    size_t qkv_row_length_val = 3 * (size_t)embed_dim_val;
    const float* head_query_ptr = qkv_ptr + (size_t)batch_index_val * seq_len_val * qkv_row_length_val + head_index_val * ATTENTION_FLASH_HEAD_SIZE;
    const float* head_key_ptr = head_query_ptr + embed_dim_val;
    const float* head_value_ptr = head_query_ptr + 2 * embed_dim_val;
    int tile_row_index_val = threadIdx.x / ATTENTION_FLASH_ROW_LANES;            // this thread's query in the tile
    int row_lane_index_val = threadIdx.x % ATTENTION_FLASH_ROW_LANES;
    int token_index_val = query_block_index_val * ATTENTION_FLASH_TILE + tile_row_index_val;
    const float* query_row_ptr = query_tile_list + tile_row_index_val * ATTENTION_FLASH_STRIDE;
    const float* grad_row_ptr = grad_tile_list + tile_row_index_val * ATTENTION_FLASH_STRIDE;
    float* scores_grad_row_ptr = scores_grad_tile_list + tile_row_index_val * ATTENTION_FLASH_SCORES_STRIDE;

    // 2. the block's queries and output_grad rows into shared memory; this row's logsumexp and row_dot
    attention_flash_load_tile(query_tile_list, head_query_ptr + (size_t)query_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                              qkv_row_length_val);
    attention_flash_load_tile(grad_tile_list,
                              output_grad_ptr + ((size_t)batch_index_val * seq_len_val + query_block_index_val * ATTENTION_FLASH_TILE) * embed_dim_val +
                                  head_index_val * ATTENTION_FLASH_HEAD_SIZE,
                              embed_dim_val);
    size_t row_index_val = (size_t)head_in_batch_index_val * seq_len_val + token_index_val;
    float row_logsumexp_val = logsumexp_ptr[row_index_val];
    float row_dot_of_row_val = row_dot_ptr[row_index_val];

    // 3. walk the key tiles up to this block's own
    float query_grad_list[ATTENTION_FLASH_HEAD_VALUES] = {};  // the row's columns row_lane + ROW_LANES · j
    for (int key_block_index_val = 0; key_block_index_val <= query_block_index_val; key_block_index_val++) {
        __syncthreads();                                      // the queries copied; every thread done with the last tiles
        attention_flash_load_tile(key_tile_list, head_key_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                                  qkv_row_length_val);
        attention_flash_load_tile(value_tile_list, head_value_ptr + (size_t)key_block_index_val * ATTENTION_FLASH_TILE * qkv_row_length_val,
                                  qkv_row_length_val);
        __syncthreads();

        // the attention_probs again, B2 and B4-B6, for this lane's keys (row_lane + ROW_LANES · i)
#pragma unroll
        for (int value_index_val = 0; value_index_val < ATTENTION_FLASH_TILE_VALUES; value_index_val++) {
            int key_in_tile_index_val = row_lane_index_val + ATTENTION_FLASH_ROW_LANES * value_index_val;
            int key_token_index_val = key_block_index_val * ATTENTION_FLASH_TILE + key_in_tile_index_val;
            float attention_prob_val = key_token_index_val <= token_index_val
                ? expf(scale_val * attention_flash_dot(query_row_ptr, key_tile_list + key_in_tile_index_val * ATTENTION_FLASH_STRIDE) -
                       row_logsumexp_val)
                : 0.0f;
            float attention_probs_grad_val = attention_flash_dot(grad_row_ptr, value_tile_list + key_in_tile_index_val * ATTENTION_FLASH_STRIDE);
            scores_grad_row_ptr[key_in_tile_index_val] = attention_prob_val * (attention_probs_grad_val - row_dot_of_row_val);
        }
        __syncwarp();                                         // the row's scores_grad whole
        // B7: query_grad += scores_grad · key tile
        for (int key_in_tile_index_val = 0; key_in_tile_index_val < ATTENTION_FLASH_TILE; key_in_tile_index_val++) {
            float scores_grad_val = scores_grad_row_ptr[key_in_tile_index_val];
            const float* key_row_ptr = key_tile_list + key_in_tile_index_val * ATTENTION_FLASH_STRIDE;
#pragma unroll
            for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++)
                query_grad_list[head_value_index_val] += scores_grad_val * key_row_ptr[row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val];
        }
    }

    // 4. into qkv_grad's query part
    float* row_ptr = qkv_grad_ptr + ((size_t)batch_index_val * seq_len_val + token_index_val) * qkv_row_length_val +
                     head_index_val * ATTENTION_FLASH_HEAD_SIZE;
#pragma unroll
    for (int head_value_index_val = 0; head_value_index_val < ATTENTION_FLASH_HEAD_VALUES; head_value_index_val++)
        row_ptr[row_lane_index_val + ATTENTION_FLASH_ROW_LANES * head_value_index_val] = query_grad_list[head_value_index_val] * scale_val;
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless flash attention can run these tensors: all float, logsumexp {batch_size,
/// num_heads, seq_len}, attention_flash_fits, output and qkv starting on a float4 (the tiles are read a float4 at
/// a time).
///
///   output_tensor      forward: output; backward: output_grad or output
///   logsumexp_tensor   {batch_size, num_heads, seq_len}; backward: also row_dot. num_heads comes from here
///   qkv_tensor         forward: qkv; backward: qkv or qkv_grad
inline void attention_flash_check_shapes(const Tensor& output_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor) {
    // 1. shapes, types and addresses together
    bool fits_val = logsumexp_tensor.dim() == 3 && output_tensor.dim() == 3 && logsumexp_tensor.size(1) > 0;
    if (fits_val) {
        size_t batch_size_val = logsumexp_tensor.size(0);
        size_t num_heads_val = logsumexp_tensor.size(1);
        size_t seq_len_val = logsumexp_tensor.size(2);
        size_t embed_dim_val = num_heads_val * ATTENTION_FLASH_HEAD_SIZE;
        size_t float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(float);
        fits_val = attention_flash_fits(seq_len_val, output_tensor.size(2) / num_heads_val) &&
                   logsumexp_tensor.dtype() == DType::Float32 &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val});
        for (const Tensor* tensor_ptr : {&output_tensor, &qkv_tensor})
            fits_val = fits_val && tensor_ptr->dtype() == DType::Float32 && (size_t)tensor_ptr->data_ptr() % float4_bytes_val == 0;
    }
    // 2. any of them wrong: stop with an error
    if (!fits_val) {
        fprintf(stderr, "error: flash attention needs float output {batch_size, seq_len, num_heads * %d} and qkv "
                        "{batch_size, seq_len, 3 * num_heads * %d} with seq_len a multiple of %d, a float logsumexp "
                        "{batch_size, num_heads, seq_len}, output and qkv starting on %zu bytes; for bfloat16, "
                        "attention_bf16.cuh\n",
                ATTENTION_FLASH_HEAD_SIZE, ATTENTION_FLASH_HEAD_SIZE, ATTENTION_FLASH_TILE, VALUES_PER_FLOAT4 * sizeof(float));
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_CUH
