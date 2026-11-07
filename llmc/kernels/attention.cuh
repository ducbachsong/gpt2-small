// attention.cuh — causal self-attention: each token mixes the values of the tokens up to it, per head:
//     F1   split    query, key, value  = qkv's three parts, cut into heads of head_size columns
//     F2   scores   score[t][j]        = query[t] · key[j] / sqrt(head_size)
//     F3   mask     only j <= t
//     F4   softmax  weight[t][j]       = exp(score[t][j] - row_max) / row_sum
//     F5   mix      head_output[t]     = sum over j <= t of weight[t][j] · value[j]
//     F6   merge    output             = the heads side by side again
//     backward:
//     B1   head_outputs_grad   = output_grad cut into heads
//     B2   weights_grad[t][j]  = head_outputs_grad[t] · value[j]
//     B3   value_grad[j]       = sum over t >= j of weight[t][j] · head_outputs_grad[t]
//     B4-6 scores_grad[t][j]  = weight[t][j] · (weights_grad[t][j] - row_dot[t])
//                                row_dot[t] = sum over j of weight[t][j] · weights_grad[t][j]
//     B7   query_grad[t]       = sum over j <= t of scores_grad[t][j] · key[j] / sqrt(head_size)
//     B8   key_grad[j]         = sum over t >= j of scores_grad[t][j] · query[t] / sqrt(head_size)
//     B9   qkv_grad            = query_grad, key_grad, value_grad put back together
//
//     qkv, qkv_grad          {batch_size, seq_len, 3 · embed_dim}          {4, 1024, 2304}
//     output, output_grad    {batch_size, seq_len, embed_dim}              {4, 1024, 768}
//     query, key, value      {batch_size, num_heads, seq_len, head_size}   {4, 12, 1024, 64}: kept for backward
//     weights                {batch_size, num_heads, seq_len, seq_len}     {4, 12, 1024, 1024}: kept for backward
//     head_outputs, the other _grads: scratch, used only inside a call
// The products (F2, F5, B2, B3, B7, B8) are cuBLAS batched products, one per (batch, head). All float or all
// bfloat16 (T); sums in float. Math: https://ducbachsong.github.io/posts/attention-backward/
//
// Flash attention (attention_flash_forward / _backward): the same math without storing the weights
// (FlashAttention-2). A block of FLASH_WARPS warps takes FLASH_TILE queries and walks the keys FLASH_TILE at a
// time on the tensor cores, with classifier.cuh's online softmax; it keeps logsumexp = row_max + log(row_sum)
// per row, from which the backward makes the weights again. Needs bfloat16, head_size FLASH_HEAD_SIZE, seq_len a
// multiple of FLASH_TILE, sm_80 or later. Its block size is fixed by the tiles (one warp per FLASH_MMA_M rows),
// so its tile kernels take no block size.
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <cublas_v2.h>
#include "cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../tensor.cuh"

#define ATTENTION_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)                    // F1, F6, B1, B9's default: one float4 per thread
#define ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)    // F4's default: one warp per row; fastest in bfloat16 on an H100
#define ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // B4-B6's default: the same
#define ATTENTION_ROW_DOT_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)            // flash row_dot's default: one warp per row

// ── forward and backward ──────────────────────────────────────────────────────
void attention_forward(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                       const Tensor& value_tensor, const Tensor& weights_tensor, const Tensor& head_outputs_tensor,
                       const Tensor& qkv_tensor, int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                       int softmax_threads_per_block_val = ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK);
void attention_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& query_tensor,
                        const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& weights_tensor,
                        const Tensor& head_outputs_grad_tensor, const Tensor& weights_grad_tensor,
                        const Tensor& query_grad_tensor, const Tensor& key_grad_tensor, const Tensor& value_grad_tensor,
                        int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK,
                        int softmax_threads_per_block_val = ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK);

// ── flash attention: bfloat16, head_size FLASH_HEAD_SIZE, seq_len a multiple of FLASH_TILE, sm_80 or later ──
// mma.sync m16n8k16, the tensor-core product the kernels are built on: C {M, N} += A {M, K} · B {K, N}
#define FLASH_MMA_M 16                   // rows of A and C: one warp's rows
#define FLASH_MMA_N 8                    // columns of B and C
#define FLASH_MMA_K 16                   // columns of A, rows of B
#define FLASH_A_REGISTERS 4              // a thread's part of an A tile: 4 registers of 2 bfloat16s
#define FLASH_B_REGISTERS 2              // of a B tile: 2 registers
#define FLASH_C_VALUES 4                 // of a C tile: 4 floats
#define FLASH_GROUP_LANES 4              // lanes that share a row of a tile in mma's layout
#define FLASH_THREAD_ROWS 2              // rows of a C tile one thread holds: group and group + FLASH_MMA_M / 2
#define FLASH_WARPS 4                                       // warps in a block
#define FLASH_TILE (FLASH_WARPS * FLASH_MMA_M)              // queries per block, and keys per step: 64
#define FLASH_HEAD_SIZE 64                                  // the only head_size the kernels take: GPT-2's
#define FLASH_THREADS (FLASH_WARPS * THREADS_PER_WARP)      // the block size
#define FLASH_K_STEPS (FLASH_HEAD_SIZE / FLASH_MMA_K)       // A tiles across a head row: 4
#define FLASH_HEAD_N_TILES (FLASH_HEAD_SIZE / FLASH_MMA_N)  // C tiles across a head row: 8
#define FLASH_TILE_N_TILES (FLASH_TILE / FLASH_MMA_N)       // C tiles across a tile of keys or queries: 8
#define FLASH_TILE_K_STEPS (FLASH_TILE / FLASH_MMA_K)       // A tiles across a tile of keys or queries: 4
#define FLASH_LOAD_VALUES (sizeof(uint4) / sizeof(nv_bfloat16))   // bfloat16s per 16-byte load: 8
#define FLASH_TILE_STRIDE (FLASH_HEAD_SIZE + FLASH_LOAD_VALUES)   // a tile row in shared memory: one spare load, for different banks
static_assert(FLASH_MMA_K == 2 * FLASH_MMA_N, "two C tiles side by side make one A tile");
static_assert(FLASH_HEAD_SIZE == 2 * THREADS_PER_WARP, "row_dot: each lane takes 2 values of a head row");
bool attention_flash_fits(DType dtype_val, size_t seq_len_val, size_t head_size_val);
void attention_flash_forward(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                             const Tensor& value_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor,
                             int threads_per_block_val = ATTENTION_THREADS_PER_BLOCK);
void attention_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& output_tensor,
                              const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                              const Tensor& logsumexp_tensor, const Tensor& row_dot_tensor,
                              int threads_per_block_val = ATTENTION_ROW_DOT_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
cublasHandle_t attention_cublas();
void attention_product(cublasOperation_t op_a_val, cublasOperation_t op_b_val, int m_val, int n_val, int k_val,
                       float alpha_val, const Tensor& a_tensor, int lda_val, long long stride_a_val,
                       const Tensor& b_tensor, int ldb_val, long long stride_b_val, const Tensor& c_tensor, int ldc_val,
                       long long stride_c_val, int batch_val);
__device__ __forceinline__ size_t attention_merged_index(size_t index_val, size_t num_heads_val, size_t seq_len_val,
                                                         size_t head_float4s_val, size_t row_float4s_val);
int attention_blocks(size_t total_val, int threads_per_block_val, size_t per_thread_val);
void attention_check_shapes(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                            const Tensor& value_tensor, const Tensor& weights_tensor, const Tensor& head_outputs_tensor,
                            const Tensor& qkv_tensor);
void attention_flash_check_shapes(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                  const Tensor& value_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor);

/// Stops with the file and line if a cuBLAS call failed.
///
///   status_val            what the call returned
///   file_ptr, line_val    where the call is
static inline void attention_cublas_check(cublasStatus_t status_val, const char* file_ptr, int line_val) {
    if (status_val != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %s\n", file_ptr, line_val, cublasGetStatusString(status_val));
        exit(EXIT_FAILURE);
    }
}
#define attentionCublasCheck(call) attention_cublas_check((call), __FILE__, __LINE__)

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename Value>
void attention_forward_in(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                          const Tensor& value_tensor, const Tensor& weights_tensor, const Tensor& head_outputs_tensor,
                          const Tensor& qkv_tensor, int threads_per_block_val, int softmax_threads_per_block_val);
template <typename T>
__global__ void attention_split_qkv_kernel(T* query_ptr, T* key_ptr, T* value_ptr, const T* qkv_ptr,
                                           size_t total_float4s_val, size_t num_heads_val, size_t seq_len_val,
                                           size_t head_float4s_val);
template <typename T>
__global__ void attention_softmax_forward_kernel(T* weights_ptr, size_t rows_val, size_t seq_len_val);
template <typename T>
__global__ void attention_merge_heads_kernel(T* output_ptr, const T* head_outputs_ptr, size_t total_float4s_val,
                                             size_t num_heads_val, size_t seq_len_val, size_t head_float4s_val);

/// Writes output = causal self-attention of qkv, keeping query, key, value and weights for the backward:
/// checks, then runs F1..F6.
///
///   output_tensor                         {batch_size, seq_len, embed_dim}
///   query_tensor, key_tensor, value_tensor {batch_size, num_heads, seq_len, head_size}: written; num_heads and
///                                         head_size come from here
///   weights_tensor                        {batch_size, num_heads, seq_len, seq_len}: written
///   head_outputs_tensor                   as query: scratch
///   qkv_tensor                            {batch_size, seq_len, 3 · embed_dim}: the qkv linear's output
///   threads_per_block_val                 F1 and F6's block size: whole warps, at most MAX_THREADS_PER_BLOCK; not
///                                         given: ATTENTION_THREADS_PER_BLOCK
///   softmax_threads_per_block_val         F4's: the same; not given: ATTENTION_SOFTMAX_FORWARD_THREADS_PER_BLOCK.
///                                         Any sizes give the same bits; only the time changes.
inline void attention_forward(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                              const Tensor& value_tensor, const Tensor& weights_tensor, const Tensor& head_outputs_tensor,
                              const Tensor& qkv_tensor, int threads_per_block_val, int softmax_threads_per_block_val) {
    // 1. check the launches and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "attention");
    check_threads_per_block(softmax_threads_per_block_val, "attention's softmax");
    attention_check_shapes(output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor, head_outputs_tensor,
                           qkv_tensor);
    // 2. F1..F6 in the tensors' type
    if (qkv_tensor.dtype() == DType::BFloat16)
        attention_forward_in<nv_bfloat16>(output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor,
                                          head_outputs_tensor, qkv_tensor, threads_per_block_val,
                                          softmax_threads_per_block_val);
    else
        attention_forward_in<float>(output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor,
                                    head_outputs_tensor, qkv_tensor, threads_per_block_val, softmax_threads_per_block_val);
}

/// Runs attention_forward's six steps on checked tensors.
///
///   Value                  float or nv_bfloat16 (not T: T is seq_len here)
///   the others             as in attention_forward
template <typename Value>
inline void attention_forward_in(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                 const Tensor& value_tensor, const Tensor& weights_tensor,
                                 const Tensor& head_outputs_tensor, const Tensor& qkv_tensor, int threads_per_block_val,
                                 int softmax_threads_per_block_val) {
    // the sizes, and each head's stride for the batched products
    size_t num_heads_val = query_tensor.size(1);
    size_t seq_len_val = query_tensor.size(2);
    size_t head_size_val = query_tensor.size(3);
    size_t heads_in_batch_val = query_tensor.size(0) * num_heads_val;                     // 48 heads
    size_t rows_val = heads_in_batch_val * seq_len_val;                                   // 49,152 rows
    size_t total_float4s_val = query_tensor.numel() / VALUES_PER_FLOAT4;
    size_t head_float4s_val = head_size_val / VALUES_PER_FLOAT4;
    int seq_len_int_val = (int)seq_len_val;
    int head_size_int_val = (int)head_size_val;
    float scale_val = 1.0f / sqrtf((float)head_size_val);
    long long head_values_val = (long long)seq_len_int_val * head_size_int_val;   // the stride from head to head
    long long head_weights_val = (long long)seq_len_int_val * seq_len_int_val;
    int float4_blocks_val = attention_blocks(total_float4s_val, threads_per_block_val, 1);

    // F1: query, key, value, each head's rows together
    attention_split_qkv_kernel<<<float4_blocks_val, threads_per_block_val>>>(
        query_tensor.data_ptr<Value>(), key_tensor.data_ptr<Value>(), value_tensor.data_ptr<Value>(),
        qkv_tensor.data_ptr<Value>(), total_float4s_val, num_heads_val, seq_len_val, head_float4s_val);
    cudaCheck(cudaGetLastError());

    // F2: scores = scale · query · keyᵀ into weights; cuBLAS reads by columns, so asked for scoresᵀ = key · queryᵀ
    attention_product(CUBLAS_OP_T, CUBLAS_OP_N, seq_len_int_val, seq_len_int_val, head_size_int_val, scale_val,
                      key_tensor, head_size_int_val, head_values_val, query_tensor, head_size_int_val, head_values_val,
                      weights_tensor, seq_len_int_val, head_weights_val, (int)heads_in_batch_val);

    // F3, F4: the mask and the softmax, in place, one warp per row
    attention_softmax_forward_kernel<<<attention_blocks(rows_val, softmax_threads_per_block_val, THREADS_PER_WARP),
                                       softmax_threads_per_block_val>>>(weights_tensor.data_ptr<Value>(), rows_val,
                                                                        seq_len_val);
    cudaCheck(cudaGetLastError());

    // F5: head_outputs = weights · value, asked for as head_outputsᵀ = valueᵀ · weightsᵀ
    attention_product(CUBLAS_OP_N, CUBLAS_OP_N, head_size_int_val, seq_len_int_val, seq_len_int_val, 1.0f, value_tensor,
                      head_size_int_val, head_values_val, weights_tensor, seq_len_int_val, head_weights_val,
                      head_outputs_tensor, head_size_int_val, head_values_val, (int)heads_in_batch_val);

    // F6: the heads side by side again
    attention_merge_heads_kernel<<<float4_blocks_val, threads_per_block_val>>>(
        output_tensor.data_ptr<Value>(), head_outputs_tensor.data_ptr<Value>(), total_float4s_val, num_heads_val,
        seq_len_val, head_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// F1: copies qkv's three parts into query, key and value, one float4 of each per thread.
///
///   query_ptr, key_ptr, value_ptr   {batch_size, num_heads, seq_len, head_size}, float or bfloat16 (T)
///   qkv_ptr                         {batch_size, seq_len, 3 · embed_dim}
///   total_float4s_val               query's values / VALUES_PER_FLOAT4
///   num_heads_val, seq_len_val      query's shape
///   head_float4s_val                head_size / VALUES_PER_FLOAT4
template <typename T>
__global__ void attention_split_qkv_kernel(T* query_ptr, T* key_ptr, T* value_ptr, const T* qkv_ptr,
                                           size_t total_float4s_val, size_t num_heads_val, size_t seq_len_val,
                                           size_t head_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s_val = num_heads_val * head_float4s_val;           // 768 / 4 = 192
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where the query part sits in qkv; key and value follow it, embed_dim apart
        size_t query_part_val = attention_merged_index(index_val, num_heads_val, seq_len_val, head_float4s_val,
                                                       3 * embed_float4s_val);
        // 2. copy all three
        write4_from_float(query_ptr + VALUES_PER_FLOAT4 * index_val, read4_as_float(qkv_ptr + VALUES_PER_FLOAT4 * query_part_val));
        write4_from_float(key_ptr + VALUES_PER_FLOAT4 * index_val,
                          read4_as_float(qkv_ptr + VALUES_PER_FLOAT4 * (query_part_val + embed_float4s_val)));
        write4_from_float(value_ptr + VALUES_PER_FLOAT4 * index_val,
                          read4_as_float(qkv_ptr + VALUES_PER_FLOAT4 * (query_part_val + 2 * embed_float4s_val)));
    }
}

/// F3, F4: turns each row of scores into weights in place, one warp per row, with classifier.cuh's online
/// softmax. Columns after the row's token are set to 0.
///
///   weights_ptr    the scores before, the weights after, float or bfloat16 (T)
///   rows_val       batch_size x num_heads x seq_len
///   seq_len_val    the row length
template <typename T>
__global__ void attention_softmax_forward_kernel(T* weights_ptr, size_t rows_val, size_t seq_len_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        size_t token_val = row_index_val % seq_len_val;     // the query's token: it sees columns 0 .. token
        T* row_weights_ptr = weights_ptr + row_index_val * seq_len_val;

        // 1. this thread's (max, sum), over its columns up to token
        float thread_max_val = -FLT_MAX;
        float thread_sum_val = 0.0f;
        for (size_t column_val = lane_index_val; column_val <= token_val; column_val += THREADS_PER_WARP) {
            float score_val = to_float(row_weights_ptr[column_val]);
            float new_max_val = fmaxf(thread_max_val, score_val);
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_max_val) + expf(score_val - new_max_val);
            thread_max_val = new_max_val;
        }

        // 2. the row's: log2(THREADS_PER_WARP) swaps; a thread with no columns brings (-FLT_MAX, 0), which adds 0
        float row_max_val = thread_max_val;
        float row_sum_val = thread_sum_val;
        for (int offset_val = THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
            float other_max_val = __shfl_xor_sync(WARP_ALL_LANES, row_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(WARP_ALL_LANES, row_sum_val, offset_val);
            float new_max_val = fmaxf(row_max_val, other_max_val);
            row_sum_val = row_sum_val * expf(row_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
            row_max_val = new_max_val;
        }

        // 3. the weights, 0 after the token
        float one_over_row_sum_val = 1.0f / row_sum_val;
        for (size_t column_val = lane_index_val; column_val < seq_len_val; column_val += THREADS_PER_WARP)
            row_weights_ptr[column_val] = from_float<T>(
                column_val <= token_val ? expf(to_float(row_weights_ptr[column_val]) - row_max_val) * one_over_row_sum_val
                                        : 0.0f);
    }
}

/// F6: copies head_outputs into output, one float4 per thread, each head at its place in the row.
///
///   output_ptr                   {batch_size, seq_len, embed_dim}, float or bfloat16 (T)
///   head_outputs_ptr             {batch_size, num_heads, seq_len, head_size}
///   total_float4s_val            head_outputs' values / VALUES_PER_FLOAT4
///   num_heads_val, seq_len_val   head_outputs' shape
///   head_float4s_val             head_size / VALUES_PER_FLOAT4
template <typename T>
__global__ void attention_merge_heads_kernel(T* output_ptr, const T* head_outputs_ptr, size_t total_float4s_val,
                                             size_t num_heads_val, size_t seq_len_val, size_t head_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s_val = num_heads_val * head_float4s_val;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x)
        write4_from_float(output_ptr + VALUES_PER_FLOAT4 * attention_merged_index(index_val, num_heads_val, seq_len_val,
                                                                                  head_float4s_val, embed_float4s_val),
                          read4_as_float(head_outputs_ptr + VALUES_PER_FLOAT4 * index_val));
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename Value>
void attention_backward_in(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& query_tensor,
                           const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& weights_tensor,
                           const Tensor& head_outputs_grad_tensor, const Tensor& weights_grad_tensor,
                           const Tensor& query_grad_tensor, const Tensor& key_grad_tensor,
                           const Tensor& value_grad_tensor, int threads_per_block_val, int softmax_threads_per_block_val);
template <typename T>
__global__ void attention_split_heads_kernel(T* head_outputs_grad_ptr, const T* output_grad_ptr, size_t total_float4s_val,
                                             size_t num_heads_val, size_t seq_len_val, size_t head_float4s_val);
template <typename T>
__global__ void attention_softmax_backward_kernel(T* weights_grad_ptr, const T* weights_ptr, size_t rows_val,
                                                  size_t seq_len_val);
template <typename T>
__global__ void attention_merge_qkv_grad_kernel(T* qkv_grad_ptr, const T* query_grad_ptr, const T* key_grad_ptr,
                                                const T* value_grad_ptr, size_t total_float4s_val, size_t num_heads_val,
                                                size_t seq_len_val, size_t head_float4s_val);

/// Writes qkv_grad (written over): checks, then runs B1..B9.
///
///   qkv_grad_tensor                     {batch_size, seq_len, 3 · embed_dim}: the result
///   output_grad_tensor                  {batch_size, seq_len, embed_dim}: the gradient from the output linear
///   query_tensor, key_tensor,
///   value_tensor, weights_tensor        the forward's
///   head_outputs_grad_tensor            as query: scratch
///   weights_grad_tensor                 as weights: scratch, weights_grad after B2, scores_grad after B6
///   query_grad_tensor, key_grad_tensor,
///   value_grad_tensor                   as query: scratch
///   threads_per_block_val               B1 and B9's block size: whole warps, at most MAX_THREADS_PER_BLOCK; not
///                                       given: ATTENTION_THREADS_PER_BLOCK
///   softmax_threads_per_block_val       B4-B6's: the same; not given: ATTENTION_SOFTMAX_BACKWARD_THREADS_PER_BLOCK.
///                                       Any sizes give the same bits; only the time changes.
inline void attention_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor, const Tensor& query_tensor,
                               const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& weights_tensor,
                               const Tensor& head_outputs_grad_tensor, const Tensor& weights_grad_tensor,
                               const Tensor& query_grad_tensor, const Tensor& key_grad_tensor,
                               const Tensor& value_grad_tensor, int threads_per_block_val,
                               int softmax_threads_per_block_val) {
    // 1. check the launches and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "attention");
    check_threads_per_block(softmax_threads_per_block_val, "attention's softmax");
    attention_check_shapes(output_grad_tensor, query_tensor, key_tensor, value_tensor, weights_tensor,
                           head_outputs_grad_tensor, qkv_grad_tensor);
    attention_check_shapes(output_grad_tensor, query_grad_tensor, key_grad_tensor, value_grad_tensor,
                           weights_grad_tensor, head_outputs_grad_tensor, qkv_grad_tensor);
    // 2. B1..B9 in the tensors' type
    if (qkv_grad_tensor.dtype() == DType::BFloat16)
        attention_backward_in<nv_bfloat16>(qkv_grad_tensor, output_grad_tensor, query_tensor, key_tensor, value_tensor,
                                           weights_tensor, head_outputs_grad_tensor, weights_grad_tensor,
                                           query_grad_tensor, key_grad_tensor, value_grad_tensor, threads_per_block_val,
                                           softmax_threads_per_block_val);
    else
        attention_backward_in<float>(qkv_grad_tensor, output_grad_tensor, query_tensor, key_tensor, value_tensor,
                                     weights_tensor, head_outputs_grad_tensor, weights_grad_tensor, query_grad_tensor,
                                     key_grad_tensor, value_grad_tensor, threads_per_block_val,
                                     softmax_threads_per_block_val);
}

/// Runs attention_backward's nine steps on checked tensors.
///
///   Value        float or nv_bfloat16
///   the others   as in attention_backward
template <typename Value>
inline void attention_backward_in(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                  const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                                  const Tensor& weights_tensor, const Tensor& head_outputs_grad_tensor,
                                  const Tensor& weights_grad_tensor, const Tensor& query_grad_tensor,
                                  const Tensor& key_grad_tensor, const Tensor& value_grad_tensor,
                                  int threads_per_block_val, int softmax_threads_per_block_val) {
    // the sizes, and each head's stride for the batched products
    size_t num_heads_val = query_tensor.size(1);
    size_t seq_len_val = query_tensor.size(2);
    size_t head_size_val = query_tensor.size(3);
    size_t heads_in_batch_val = query_tensor.size(0) * num_heads_val;
    size_t rows_val = heads_in_batch_val * seq_len_val;
    size_t total_float4s_val = query_tensor.numel() / VALUES_PER_FLOAT4;
    size_t head_float4s_val = head_size_val / VALUES_PER_FLOAT4;
    int seq_len_int_val = (int)seq_len_val;
    int head_size_int_val = (int)head_size_val;
    float scale_val = 1.0f / sqrtf((float)head_size_val);
    long long head_values_val = (long long)seq_len_int_val * head_size_int_val;   // the stride from head to head
    long long head_weights_val = (long long)seq_len_int_val * seq_len_int_val;
    int float4_blocks_val = attention_blocks(total_float4s_val, threads_per_block_val, 1);

    // B1: output_grad cut into heads
    attention_split_heads_kernel<<<float4_blocks_val, threads_per_block_val>>>(
        head_outputs_grad_tensor.data_ptr<Value>(), output_grad_tensor.data_ptr<Value>(), total_float4s_val,
        num_heads_val, seq_len_val, head_float4s_val);
    cudaCheck(cudaGetLastError());

    // B2: weights_grad = head_outputs_grad · valueᵀ, asked for turned round, as F2
    attention_product(CUBLAS_OP_T, CUBLAS_OP_N, seq_len_int_val, seq_len_int_val, head_size_int_val, 1.0f, value_tensor,
                      head_size_int_val, head_values_val, head_outputs_grad_tensor, head_size_int_val, head_values_val,
                      weights_grad_tensor, seq_len_int_val, head_weights_val, (int)heads_in_batch_val);

    // B3: value_grad = weightsᵀ · head_outputs_grad, asked for as value_gradᵀ = head_outputs_gradᵀ · weights
    attention_product(CUBLAS_OP_N, CUBLAS_OP_T, head_size_int_val, seq_len_int_val, seq_len_int_val, 1.0f,
                      head_outputs_grad_tensor, head_size_int_val, head_values_val, weights_tensor, seq_len_int_val,
                      head_weights_val, value_grad_tensor, head_size_int_val, head_values_val, (int)heads_in_batch_val);

    // B4-B6: the softmax's backward, in place, one warp per row
    attention_softmax_backward_kernel<<<attention_blocks(rows_val, softmax_threads_per_block_val, THREADS_PER_WARP),
                                        softmax_threads_per_block_val>>>(
        weights_grad_tensor.data_ptr<Value>(), weights_tensor.data_ptr<Value>(), rows_val, seq_len_val);
    cudaCheck(cudaGetLastError());

    // B7: query_grad = scale · scores_grad · key, asked for as query_gradᵀ = keyᵀ · scores_gradᵀ
    attention_product(CUBLAS_OP_N, CUBLAS_OP_N, head_size_int_val, seq_len_int_val, seq_len_int_val, scale_val,
                      key_tensor, head_size_int_val, head_values_val, weights_grad_tensor, seq_len_int_val,
                      head_weights_val, query_grad_tensor, head_size_int_val, head_values_val, (int)heads_in_batch_val);

    // B8: key_grad = scale · scores_gradᵀ · query, asked for as key_gradᵀ = queryᵀ · scores_grad
    attention_product(CUBLAS_OP_N, CUBLAS_OP_T, head_size_int_val, seq_len_int_val, seq_len_int_val, scale_val,
                      query_tensor, head_size_int_val, head_values_val, weights_grad_tensor, seq_len_int_val,
                      head_weights_val, key_grad_tensor, head_size_int_val, head_values_val, (int)heads_in_batch_val);

    // B9: the three gradients into qkv_grad's three parts
    attention_merge_qkv_grad_kernel<<<float4_blocks_val, threads_per_block_val>>>(
        qkv_grad_tensor.data_ptr<Value>(), query_grad_tensor.data_ptr<Value>(), key_grad_tensor.data_ptr<Value>(),
        value_grad_tensor.data_ptr<Value>(), total_float4s_val, num_heads_val, seq_len_val, head_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// B1: copies output_grad into head_outputs_grad, one float4 per thread: F6 the other way round.
///
///   head_outputs_grad_ptr        {batch_size, num_heads, seq_len, head_size}, float or bfloat16 (T)
///   output_grad_ptr              {batch_size, seq_len, embed_dim}
///   total_float4s_val            head_outputs_grad's values / VALUES_PER_FLOAT4
///   num_heads_val, seq_len_val   head_outputs_grad's shape
///   head_float4s_val             head_size / VALUES_PER_FLOAT4
template <typename T>
__global__ void attention_split_heads_kernel(T* head_outputs_grad_ptr, const T* output_grad_ptr, size_t total_float4s_val,
                                             size_t num_heads_val, size_t seq_len_val, size_t head_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s_val = num_heads_val * head_float4s_val;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x)
        write4_from_float(head_outputs_grad_ptr + VALUES_PER_FLOAT4 * index_val,
                          read4_as_float(output_grad_ptr + VALUES_PER_FLOAT4 * attention_merged_index(
                                                               index_val, num_heads_val, seq_len_val, head_float4s_val,
                                                               embed_float4s_val)));
}

/// B4-B6: turns each row of weights_grad into scores_grad in place, one warp per row:
///     scores_grad = weight · (weights_grad - row_dot), 0 after the row's token
///
///   weights_grad_ptr   weights_grad before, scores_grad after, float or bfloat16 (T)
///   weights_ptr        the forward's
///   rows_val           batch_size x num_heads x seq_len
///   seq_len_val        the row length
template <typename T>
__global__ void attention_softmax_backward_kernel(T* weights_grad_ptr, const T* weights_ptr, size_t rows_val,
                                                  size_t seq_len_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        size_t token_val = row_index_val % seq_len_val;
        T* row_grad_ptr = weights_grad_ptr + row_index_val * seq_len_val;
        const T* row_weights_ptr = weights_ptr + row_index_val * seq_len_val;

        // 1. row_dot: this thread's part, then the warp's
        float row_dot_val = 0.0f;
        for (size_t column_val = lane_index_val; column_val <= token_val; column_val += THREADS_PER_WARP)
            row_dot_val += to_float(row_weights_ptr[column_val]) * to_float(row_grad_ptr[column_val]);
        row_dot_val = warp_reduce_sum(row_dot_val);

        // 2. scores_grad, 0 after the token
        for (size_t column_val = lane_index_val; column_val < seq_len_val; column_val += THREADS_PER_WARP)
            row_grad_ptr[column_val] = from_float<T>(
                column_val <= token_val
                    ? to_float(row_weights_ptr[column_val]) * (to_float(row_grad_ptr[column_val]) - row_dot_val)
                    : 0.0f);
    }
}

/// B9: copies query_grad, key_grad and value_grad into qkv_grad's three parts: F1 the other way round.
///
///   qkv_grad_ptr                                      {batch_size, seq_len, 3 · embed_dim}, float or bfloat16 (T)
///   query_grad_ptr, key_grad_ptr, value_grad_ptr      {batch_size, num_heads, seq_len, head_size}
///   total_float4s_val                                 query_grad's values / VALUES_PER_FLOAT4
///   num_heads_val, seq_len_val                        query_grad's shape
///   head_float4s_val                                  head_size / VALUES_PER_FLOAT4
template <typename T>
__global__ void attention_merge_qkv_grad_kernel(T* qkv_grad_ptr, const T* query_grad_ptr, const T* key_grad_ptr,
                                                const T* value_grad_ptr, size_t total_float4s_val, size_t num_heads_val,
                                                size_t seq_len_val, size_t head_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s_val = num_heads_val * head_float4s_val;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where the query part sits in qkv_grad; key and value follow it, embed_dim apart
        size_t query_part_val = attention_merged_index(index_val, num_heads_val, seq_len_val, head_float4s_val,
                                                       3 * embed_float4s_val);
        // 2. copy all three
        write4_from_float(qkv_grad_ptr + VALUES_PER_FLOAT4 * query_part_val,
                          read4_as_float(query_grad_ptr + VALUES_PER_FLOAT4 * index_val));
        write4_from_float(qkv_grad_ptr + VALUES_PER_FLOAT4 * (query_part_val + embed_float4s_val),
                          read4_as_float(key_grad_ptr + VALUES_PER_FLOAT4 * index_val));
        write4_from_float(qkv_grad_ptr + VALUES_PER_FLOAT4 * (query_part_val + 2 * embed_float4s_val),
                          read4_as_float(value_grad_ptr + VALUES_PER_FLOAT4 * index_val));
    }
}

// ── flash attention ───────────────────────────────────────────────────────────

__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_forward_kernel(nv_bfloat16* output_ptr, float* logsumexp_ptr, const nv_bfloat16* query_ptr,
                               const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr, int num_heads_val,
                               int seq_len_val, float scale_val);
__global__ void attention_flash_row_dot_kernel(float* row_dot_ptr, const nv_bfloat16* output_grad_ptr,
                                               const nv_bfloat16* output_ptr, int num_heads_val, int seq_len_val,
                                               size_t rows_val);
__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_key_grad_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* output_grad_ptr, const nv_bfloat16* query_ptr,
                                const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr, const float* logsumexp_ptr,
                                const float* row_dot_ptr, int num_heads_val, int seq_len_val, float scale_val);
__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_query_grad_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* output_grad_ptr,
                                  const nv_bfloat16* query_ptr, const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr,
                                  const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                  int seq_len_val, float scale_val);

/// Tells if flash attention can run here: bfloat16, head_size FLASH_HEAD_SIZE, seq_len a multiple of FLASH_TILE,
/// sm_80 or later.
///
///   dtype_val, seq_len_val, head_size_val   the tensors'
inline bool attention_flash_fits(DType dtype_val, size_t seq_len_val, size_t head_size_val) {
    static int major_val = [] {
        int device_val = 0;
        int value_val = 0;
        cudaCheck(cudaGetDevice(&device_val));
        cudaCheck(cudaDeviceGetAttribute(&value_val, cudaDevAttrComputeCapabilityMajor, device_val));
        return value_val;
    }();
    return dtype_val == DType::BFloat16 && head_size_val == FLASH_HEAD_SIZE && seq_len_val % FLASH_TILE == 0 &&
           major_val >= 8;
}

/// Writes output as attention_forward, by flash attention: F1, then one kernel for F2..F6, one block per
/// FLASH_TILE queries of a head.
///
///   output_tensor, query_tensor, key_tensor,
///   value_tensor, qkv_tensor               as in attention_forward, bfloat16
///   logsumexp_tensor                       {batch_size, num_heads, seq_len} float: kept in place of the weights
///   threads_per_block_val                  F1's block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                                          ATTENTION_THREADS_PER_BLOCK. The tile kernel's is always FLASH_THREADS.
inline void attention_flash_forward(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                    const Tensor& value_tensor, const Tensor& logsumexp_tensor, const Tensor& qkv_tensor,
                                    int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "flash attention");
    attention_flash_check_shapes(output_tensor, query_tensor, key_tensor, value_tensor, logsumexp_tensor, qkv_tensor);
    size_t num_heads_val = query_tensor.size(1);
    size_t seq_len_val = query_tensor.size(2);
    size_t head_size_val = query_tensor.size(3);
    size_t total_float4s_val = query_tensor.numel() / VALUES_PER_FLOAT4;
    size_t head_float4s_val = head_size_val / VALUES_PER_FLOAT4;
    // 2. F1: query, key, value, each head's rows together
    attention_split_qkv_kernel<<<attention_blocks(total_float4s_val, threads_per_block_val, 1), threads_per_block_val>>>(
        query_tensor.data_ptr<nv_bfloat16>(), key_tensor.data_ptr<nv_bfloat16>(), value_tensor.data_ptr<nv_bfloat16>(),
        qkv_tensor.data_ptr<nv_bfloat16>(), total_float4s_val, num_heads_val, seq_len_val, head_float4s_val);
    cudaCheck(cudaGetLastError());
    // 3. F2..F6: one block per FLASH_TILE queries of each head
    dim3 grid_val((unsigned)(seq_len_val / FLASH_TILE), (unsigned)(query_tensor.size(0) * num_heads_val));
    attention_flash_forward_kernel<<<grid_val, FLASH_THREADS>>>(
        output_tensor.data_ptr<nv_bfloat16>(), logsumexp_tensor.data(), query_tensor.data_ptr<nv_bfloat16>(),
        key_tensor.data_ptr<nv_bfloat16>(), value_tensor.data_ptr<nv_bfloat16>(), (int)num_heads_val, (int)seq_len_val,
        1.0f / sqrtf((float)head_size_val));
    cudaCheck(cudaGetLastError());
}

/// Writes qkv_grad as attention_backward, by flash attention: row_dot (one warp per row), then key_grad and
/// value_grad (one block per FLASH_TILE keys), then query_grad (one block per FLASH_TILE queries).
///
///   qkv_grad_tensor                       the result, written over
///   output_grad_tensor                    the gradient from the output linear
///   output_tensor, query_tensor, key_tensor,
///   value_tensor, logsumexp_tensor        the forward's
///   row_dot_tensor                        {batch_size, num_heads, seq_len} float: scratch
///   threads_per_block_val                 row_dot's block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                                         ATTENTION_ROW_DOT_THREADS_PER_BLOCK. The tile kernels' is always FLASH_THREADS.
inline void attention_flash_backward(const Tensor& qkv_grad_tensor, const Tensor& output_grad_tensor,
                                     const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                     const Tensor& value_tensor, const Tensor& logsumexp_tensor,
                                     const Tensor& row_dot_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "flash attention");
    attention_flash_check_shapes(output_grad_tensor, query_tensor, key_tensor, value_tensor, logsumexp_tensor, qkv_grad_tensor);
    attention_flash_check_shapes(output_tensor, query_tensor, key_tensor, value_tensor, row_dot_tensor, qkv_grad_tensor);
    size_t num_heads_val = query_tensor.size(1);
    size_t seq_len_val = query_tensor.size(2);
    size_t head_size_val = query_tensor.size(3);
    size_t rows_val = query_tensor.size(0) * num_heads_val * seq_len_val;     // 49,152 for {4, 12, 1024}
    float scale_val = 1.0f / sqrtf((float)head_size_val);
    // 2. row_dot, one warp per row
    attention_flash_row_dot_kernel<<<attention_blocks(rows_val, threads_per_block_val, THREADS_PER_WARP),
                                     threads_per_block_val>>>(
        row_dot_tensor.data(), output_grad_tensor.data_ptr<nv_bfloat16>(), output_tensor.data_ptr<nv_bfloat16>(),
        (int)num_heads_val, (int)seq_len_val, rows_val);
    cudaCheck(cudaGetLastError());
    // 3. key_grad and value_grad, then query_grad: one block per tile of each head
    dim3 grid_val((unsigned)(seq_len_val / FLASH_TILE), (unsigned)(query_tensor.size(0) * num_heads_val));
    attention_flash_key_grad_kernel<<<grid_val, FLASH_THREADS>>>(
        qkv_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
        query_tensor.data_ptr<nv_bfloat16>(), key_tensor.data_ptr<nv_bfloat16>(), value_tensor.data_ptr<nv_bfloat16>(),
        logsumexp_tensor.data(), row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
    attention_flash_query_grad_kernel<<<grid_val, FLASH_THREADS>>>(
        qkv_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
        query_tensor.data_ptr<nv_bfloat16>(), key_tensor.data_ptr<nv_bfloat16>(), value_tensor.data_ptr<nv_bfloat16>(),
        logsumexp_tensor.data(), row_dot_tensor.data(), (int)num_heads_val, (int)seq_len_val, scale_val);
    cudaCheck(cudaGetLastError());
}

// mma.sync m16n8k16: C {16, 8} += A {16, 16} · B {16, 8}, each of a warp's 32 threads holding a part of
// every tile. The layout is the instruction's own, fixed for 32 lanes. Thread `lane`, with
// group = lane / FLASH_GROUP_LANES and pair = 2 (lane % FLASH_GROUP_LANES), holds:
//     A (bfloat16)   4 registers of 2 values: (group, pair .. pair+1), (group+8, ..), (group, pair+8 ..), (group+8, pair+8 ..)
//     B (bfloat16)   2 registers of 2 values: rows (pair .. pair+1, column group), rows (pair+8 .. pair+9, column group)
//     C (float)      4 values: (group, pair), (group, pair+1), (group+8, pair), (group+8, pair+1)
// Two C tiles side by side hold exactly an A tile's values: weights go into the next product with no move.

/// C += A · B on the tensor cores, for the whole warp.
///
///   c_ptr              this thread's FLASH_C_VALUES floats of C
///   a_ptr              this thread's FLASH_A_REGISTERS registers of A
///   b0_val, b1_val     this thread's FLASH_B_REGISTERS registers of B
__device__ __forceinline__ void flash_mma(float* c_ptr, const uint32_t* a_ptr, uint32_t b0_val, uint32_t b1_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c_ptr[0]), "+f"(c_ptr[1]), "+f"(c_ptr[2]), "+f"(c_ptr[3])
                 : "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(b0_val), "r"(b1_val));
#endif
}

/// Gives two floats as one register of two bfloat16s, rounded: an A or B register.
///
///   low_val, high_val   the first goes in the low half
__device__ __forceinline__ uint32_t flash_pack(float low_val, float high_val) {
    nv_bfloat162 both_val = __floats2bfloat162_rn(low_val, high_val);
    return *reinterpret_cast<uint32_t*>(&both_val);
}

/// Gives two bfloat16s as one register.
///
///   low_val, high_val   the first goes in the low half
__device__ __forceinline__ uint32_t flash_pack(nv_bfloat16 low_val, nv_bfloat16 high_val) {
    return (uint32_t)__bfloat16_as_ushort(low_val) | ((uint32_t)__bfloat16_as_ushort(high_val) << 16);
}

/// Copies a {FLASH_TILE, FLASH_HEAD_SIZE} tile into shared memory, 16 bytes per load.
///
///   tile_ptr         shared memory, rows FLASH_TILE_STRIDE apart
///   source_ptr       the tile's first value in GPU memory
///   row_stride_val   values from row to row: FLASH_HEAD_SIZE for query, key, value; embed_dim for output and
///                    output_grad
__device__ __forceinline__ void flash_load_tile(nv_bfloat16* tile_ptr, const nv_bfloat16* source_ptr, size_t row_stride_val) {
    const int loads_per_row_val = FLASH_HEAD_SIZE / FLASH_LOAD_VALUES;
    for (int piece_val = threadIdx.x; piece_val < FLASH_TILE * loads_per_row_val; piece_val += FLASH_THREADS) {
        int row_val = piece_val / loads_per_row_val;
        int column_val = (piece_val % loads_per_row_val) * FLASH_LOAD_VALUES;
        *reinterpret_cast<uint4*>(tile_ptr + row_val * FLASH_TILE_STRIDE + column_val) =
            *reinterpret_cast<const uint4*>(source_ptr + row_val * row_stride_val + column_val);
    }
}

/// Reads this warp's A tile from shared memory.
///
///   a_ptr                              this thread's FLASH_A_REGISTERS registers
///   tile_ptr                           the tile in shared memory
///   first_row_val, first_column_val    the A tile's {FLASH_MMA_M, FLASH_MMA_K} corner
__device__ __forceinline__ void flash_load_a(uint32_t* a_ptr, const nv_bfloat16* tile_ptr, int first_row_val,
                                             int first_column_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    const nv_bfloat16* top_ptr = tile_ptr + (first_row_val + group_val) * FLASH_TILE_STRIDE + first_column_val + pair_val;
    const nv_bfloat16* bottom_ptr = top_ptr + (FLASH_MMA_M / 2) * FLASH_TILE_STRIDE;
    a_ptr[0] = *reinterpret_cast<const uint32_t*>(top_ptr);
    a_ptr[1] = *reinterpret_cast<const uint32_t*>(bottom_ptr);
    a_ptr[2] = *reinterpret_cast<const uint32_t*>(top_ptr + FLASH_MMA_K / 2);
    a_ptr[3] = *reinterpret_cast<const uint32_t*>(bottom_ptr + FLASH_MMA_K / 2);
}

/// Reads a B tile that is a tile in shared memory turned round (as keyᵀ in query · keyᵀ).
///
///   b_ptr                      this thread's FLASH_B_REGISTERS registers
///   tile_ptr                   the tile in shared memory
///   first_n_val, first_k_val   B's first column (a tile row) and first row (a tile column)
__device__ __forceinline__ void flash_load_b_turned(uint32_t* b_ptr, const nv_bfloat16* tile_ptr, int first_n_val,
                                                    int first_k_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    const nv_bfloat16* at_ptr = tile_ptr + (first_n_val + group_val) * FLASH_TILE_STRIDE + first_k_val + pair_val;
    b_ptr[0] = *reinterpret_cast<const uint32_t*>(at_ptr);
    b_ptr[1] = *reinterpret_cast<const uint32_t*>(at_ptr + FLASH_MMA_K / 2);
}

/// Reads a B tile that is a tile in shared memory as it is (as value in weights · value).
///
///   b_ptr                      this thread's FLASH_B_REGISTERS registers
///   tile_ptr                   the tile in shared memory
///   first_n_val, first_k_val   B's first column and first row
__device__ __forceinline__ void flash_load_b(uint32_t* b_ptr, const nv_bfloat16* tile_ptr, int first_n_val, int first_k_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    const nv_bfloat16* at_ptr = tile_ptr + (first_k_val + pair_val) * FLASH_TILE_STRIDE + first_n_val + group_val;
    b_ptr[0] = flash_pack(at_ptr[0], at_ptr[FLASH_TILE_STRIDE]);
    b_ptr[1] = flash_pack(at_ptr[(FLASH_MMA_K / 2) * FLASH_TILE_STRIDE], at_ptr[(FLASH_MMA_K / 2 + 1) * FLASH_TILE_STRIDE]);
}

/// Turns two C tiles side by side into the A tile of the same FLASH_MMA_K columns, rounded to bfloat16.
///
///   a_ptr      this thread's FLASH_A_REGISTERS registers of A
///   c_ptr      the {FLASH_MMA_M, FLASH_TILE} product: FLASH_TILE_N_TILES C tiles
///   step_val   which FLASH_MMA_K columns: C tiles 2 step and 2 step + 1
__device__ __forceinline__ void flash_c_to_a(uint32_t* a_ptr, float (*c_ptr)[FLASH_C_VALUES], int step_val) {
    a_ptr[0] = flash_pack(c_ptr[2 * step_val][0], c_ptr[2 * step_val][1]);
    a_ptr[1] = flash_pack(c_ptr[2 * step_val][2], c_ptr[2 * step_val][3]);
    a_ptr[2] = flash_pack(c_ptr[2 * step_val + 1][0], c_ptr[2 * step_val + 1][1]);
    a_ptr[3] = flash_pack(c_ptr[2 * step_val + 1][2], c_ptr[2 * step_val + 1][3]);
}

/// F2..F6 for FLASH_TILE queries of one head (the longest rows first). Each warp keeps its FLASH_MMA_M query rows
/// in registers and walks the keys FLASH_TILE at a time up to its own: scores, the mask, the online softmax,
/// output += weights · value.
///
///   output_ptr                    {batch_size, seq_len, embed_dim}: the head's columns written
///   logsumexp_ptr                 one float per row: written
///   query_ptr, key_ptr, value_ptr {batch_size, num_heads, seq_len, FLASH_HEAD_SIZE}
///   num_heads_val, seq_len_val    query's shape
///   scale_val                     1 / sqrt(FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_forward_kernel(nv_bfloat16* output_ptr, float* logsumexp_ptr, const nv_bfloat16* query_ptr,
                               const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr, int num_heads_val,
                               int seq_len_val, float scale_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    int query_block_val = gridDim.x - 1 - blockIdx.x;
    size_t head_start_val = (size_t)blockIdx.y * seq_len_val * FLASH_HEAD_SIZE;   // this head's first value in query
    int warp_first_token_val = query_block_val * FLASH_TILE + warp_index_val * FLASH_MMA_M;

    // 1. this warp's query rows into registers: FLASH_K_STEPS A tiles across the head row
    flash_load_tile(query_tile_list, query_ptr + head_start_val + (size_t)query_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                    FLASH_HEAD_SIZE);
    __syncthreads();
    uint32_t query_a_list[FLASH_K_STEPS][FLASH_A_REGISTERS];
#pragma unroll
    for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++)
        flash_load_a(query_a_list[step_val], query_tile_list, warp_index_val * FLASH_MMA_M, step_val * FLASH_MMA_K);

    float output_list[FLASH_HEAD_N_TILES][FLASH_C_VALUES] = {};    // {FLASH_MMA_M, FLASH_HEAD_SIZE}: C tiles
    // rows group, group + FLASH_MMA_M / 2
    float row_max_list[FLASH_THREAD_ROWS] = {-INFINITY, -INFINITY};
    float row_sum_list[FLASH_THREAD_ROWS] = {0.0f, 0.0f};
    // 2. walk the key tiles up to this block's own
    for (int key_block_val = 0; key_block_val <= query_block_val; key_block_val++) {
        __syncthreads();                                            // every warp is done with the last tiles
        flash_load_tile(key_tile_list, key_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                        FLASH_HEAD_SIZE);
        flash_load_tile(value_tile_list, value_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                        FLASH_HEAD_SIZE);
        __syncthreads();

        // F2: scores {FLASH_MMA_M, FLASH_TILE} = query rows · key tileᵀ, scaled
        float score_list[FLASH_TILE_N_TILES][FLASH_C_VALUES];
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++) {
            score_list[n_val][0] = score_list[n_val][1] = score_list[n_val][2] = score_list[n_val][3] = 0.0f;
#pragma unroll
            for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b_turned(b_list, key_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(score_list[n_val], query_a_list[step_val], b_list[0], b_list[1]);
            }
        }
        // F3: the mask, on the tile of the warp's own tokens only; then each row's largest score
        float tile_max_list[FLASH_THREAD_ROWS] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++)
#pragma unroll
            for (int i_val = 0; i_val < FLASH_C_VALUES; i_val++) {
                int token_val = warp_first_token_val + group_val + (i_val / 2) * (FLASH_MMA_M / 2);
                int key_token_val = key_block_val * FLASH_TILE + n_val * FLASH_MMA_N + pair_val + (i_val & 1);
                score_list[n_val][i_val] = key_token_val <= token_val ? score_list[n_val][i_val] * scale_val : -INFINITY;
                tile_max_list[i_val / 2] = fmaxf(tile_max_list[i_val / 2], score_list[n_val][i_val]);
            }
        // F4: the online softmax: the new max (over the group's FLASH_GROUP_LANES lanes), the output and sum moved
        // onto it, then the weights
#pragma unroll
        for (int r_val = 0; r_val < FLASH_THREAD_ROWS; r_val++) {
            tile_max_list[r_val] = fmaxf(tile_max_list[r_val], __shfl_xor_sync(WARP_ALL_LANES, tile_max_list[r_val], 1));
            tile_max_list[r_val] = fmaxf(tile_max_list[r_val], __shfl_xor_sync(WARP_ALL_LANES, tile_max_list[r_val], 2));
            float new_max_val = fmaxf(row_max_list[r_val], tile_max_list[r_val]);   // finite: every row sees key 0
            float correction_val = __expf(row_max_list[r_val] - new_max_val);     // 0 on the first tile
            row_max_list[r_val] = new_max_val;
            row_sum_list[r_val] *= correction_val;
#pragma unroll
            for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++)
                output_list[n_val][2 * r_val] *= correction_val, output_list[n_val][2 * r_val + 1] *= correction_val;
        }
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++)
#pragma unroll
            for (int i_val = 0; i_val < FLASH_C_VALUES; i_val++) {
                score_list[n_val][i_val] = __expf(score_list[n_val][i_val] - row_max_list[i_val / 2]);   // a masked score gives 0
                row_sum_list[i_val / 2] += score_list[n_val][i_val];
            }
        // F5: output += weights · value tile, the weights straight from the score registers
#pragma unroll
        for (int step_val = 0; step_val < FLASH_TILE_K_STEPS; step_val++) {
            uint32_t a_list[FLASH_A_REGISTERS];
            flash_c_to_a(a_list, score_list, step_val);
#pragma unroll
            for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b(b_list, value_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(output_list[n_val], a_list, b_list[0], b_list[1]);
            }
        }
    }

    // 3. F6: divide by the row sums, write the head's columns of output, and logsumexp
    int batch_val = blockIdx.y / num_heads_val;
    int head_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
#pragma unroll
    for (int r_val = 0; r_val < FLASH_THREAD_ROWS; r_val++) {
        row_sum_list[r_val] += __shfl_xor_sync(WARP_ALL_LANES, row_sum_list[r_val], 1);
        row_sum_list[r_val] += __shfl_xor_sync(WARP_ALL_LANES, row_sum_list[r_val], 2);
        int token_val = warp_first_token_val + group_val + (FLASH_MMA_M / 2) * r_val;
        nv_bfloat16* row_ptr = output_ptr + ((size_t)batch_val * seq_len_val + token_val) * embed_dim_val + head_val * FLASH_HEAD_SIZE;
        float one_over_sum_val = 1.0f / row_sum_list[r_val];
#pragma unroll
        for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++)
            *reinterpret_cast<nv_bfloat162*>(row_ptr + n_val * FLASH_MMA_N + pair_val) = __floats2bfloat162_rn(
                output_list[n_val][2 * r_val] * one_over_sum_val, output_list[n_val][2 * r_val + 1] * one_over_sum_val);
        if (lane_index_val % FLASH_GROUP_LANES == 0)
            logsumexp_ptr[(size_t)blockIdx.y * seq_len_val + token_val] = row_max_list[r_val] + logf(row_sum_list[r_val]);
    }
#endif
}

/// Works out row_dot = sum over the head's FLASH_HEAD_SIZE columns of output_grad · output, one warp per row.
///
///   row_dot_ptr                   one float per row: written
///   output_grad_ptr, output_ptr   {batch_size, seq_len, embed_dim}
///   num_heads_val, seq_len_val    the shape
///   rows_val                      batch_size x num_heads x seq_len
__global__ void attention_flash_row_dot_kernel(float* row_dot_ptr, const nv_bfloat16* output_grad_ptr,
                                               const nv_bfloat16* output_ptr, int num_heads_val, int seq_len_val,
                                               size_t rows_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        // 1. where the row is: which (batch, head), which token; each lane takes 2 neighbouring columns
        size_t head_index_val = row_index_val / seq_len_val;                    // batch x num_heads + head
        size_t token_val = row_index_val % seq_len_val;
        size_t batch_val = head_index_val / num_heads_val;
        size_t head_val = head_index_val % num_heads_val;
        size_t at_val = (batch_val * seq_len_val + token_val) * num_heads_val * FLASH_HEAD_SIZE + head_val * FLASH_HEAD_SIZE +
                        2 * lane_index_val;
        // 2. this lane's part of the dot, then the warp's
        float2 grad_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(output_grad_ptr + at_val));
        float2 out_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(output_ptr + at_val));
        float dot_val = grad_val.x * out_val.x + grad_val.y * out_val.y;
        dot_val = warp_reduce_sum(dot_val);
        if (lane_index_val == 0) row_dot_ptr[row_index_val] = dot_val;
    }
}

/// Works out key_grad and value_grad for FLASH_TILE keys of one head. Each warp keeps its FLASH_MMA_M keys and
/// values in registers and walks the queries FLASH_TILE at a time from its own tile on, making the weights again,
/// turned round:
///     weightsᵀ       = exp(key rows · query tileᵀ / sqrt(head_size) - logsumexp)    0 where query < key
///     value_grad    += weightsᵀ · output_grad tile                                  B3
///     weights_gradᵀ  = value rows · output_grad tileᵀ                               B2
///     scores_gradᵀ   = weightsᵀ (weights_gradᵀ - row_dot)                           B4-B6
///     key_grad      += scores_gradᵀ · query tile                                    B8
///
///   qkv_grad_ptr                            the key and value parts written
///   output_grad_ptr                         {batch_size, seq_len, embed_dim}
///   query_ptr, key_ptr, value_ptr           {batch_size, num_heads, seq_len, FLASH_HEAD_SIZE}
///   logsumexp_ptr, row_dot_ptr              one float per row
///   num_heads_val, seq_len_val, scale_val   the shape, and 1 / sqrt(FLASH_HEAD_SIZE)
__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_key_grad_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* output_grad_ptr, const nv_bfloat16* query_ptr,
                                const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr, const float* logsumexp_ptr,
                                const float* row_dot_ptr, int num_heads_val, int seq_len_val, float scale_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 grad_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ float logsumexp_tile_list[FLASH_TILE];
    __shared__ float row_dot_tile_list[FLASH_TILE];
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    int key_block_val = blockIdx.x;
    int batch_val = blockIdx.y / num_heads_val;
    int head_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
    size_t head_start_val = (size_t)blockIdx.y * seq_len_val * FLASH_HEAD_SIZE;
    int warp_first_key_val = key_block_val * FLASH_TILE + warp_index_val * FLASH_MMA_M;

    // 1. this warp's keys, and their values, into registers
    flash_load_tile(key_tile_list, key_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                    FLASH_HEAD_SIZE);
    flash_load_tile(value_tile_list, value_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                    FLASH_HEAD_SIZE);
    __syncthreads();
    uint32_t key_a_list[FLASH_K_STEPS][FLASH_A_REGISTERS];
    uint32_t value_a_list[FLASH_K_STEPS][FLASH_A_REGISTERS];
#pragma unroll
    for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
        flash_load_a(key_a_list[step_val], key_tile_list, warp_index_val * FLASH_MMA_M, step_val * FLASH_MMA_K);
        flash_load_a(value_a_list[step_val], value_tile_list, warp_index_val * FLASH_MMA_M, step_val * FLASH_MMA_K);
    }

    // 2. walk the query tiles from this block's own on
    float key_grad_list[FLASH_HEAD_N_TILES][FLASH_C_VALUES] = {};
    float value_grad_list[FLASH_HEAD_N_TILES][FLASH_C_VALUES] = {};
    for (int query_block_val = key_block_val; query_block_val < seq_len_val / FLASH_TILE; query_block_val++) {
        __syncthreads();
        flash_load_tile(query_tile_list, query_ptr + head_start_val + (size_t)query_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                        FLASH_HEAD_SIZE);
        flash_load_tile(grad_tile_list,
                        output_grad_ptr + ((size_t)batch_val * seq_len_val + query_block_val * FLASH_TILE) * embed_dim_val +
                            head_val * FLASH_HEAD_SIZE,
                        embed_dim_val);
        if (threadIdx.x < FLASH_TILE) {
            size_t row_index_val = (size_t)blockIdx.y * seq_len_val + query_block_val * FLASH_TILE + threadIdx.x;
            logsumexp_tile_list[threadIdx.x] = logsumexp_ptr[row_index_val];
            row_dot_tile_list[threadIdx.x] = row_dot_ptr[row_index_val];
        }
        __syncthreads();

        // the weights, turned round: {FLASH_MMA_M keys, FLASH_TILE queries}
        float weight_list[FLASH_TILE_N_TILES][FLASH_C_VALUES];
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++) {
            weight_list[n_val][0] = weight_list[n_val][1] = weight_list[n_val][2] = weight_list[n_val][3] = 0.0f;
#pragma unroll
            for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b_turned(b_list, query_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(weight_list[n_val], key_a_list[step_val], b_list[0], b_list[1]);
            }
#pragma unroll
            for (int i_val = 0; i_val < FLASH_C_VALUES; i_val++) {
                int key_token_val = warp_first_key_val + group_val + (i_val / 2) * (FLASH_MMA_M / 2);
                int column_val = n_val * FLASH_MMA_N + pair_val + (i_val & 1);
                int token_val = query_block_val * FLASH_TILE + column_val;
                weight_list[n_val][i_val] = token_val >= key_token_val
                                                ? __expf(weight_list[n_val][i_val] * scale_val - logsumexp_tile_list[column_val])
                                                : 0.0f;
            }
        }
        // B3: value_grad += weightsᵀ · output_grad tile
#pragma unroll
        for (int step_val = 0; step_val < FLASH_TILE_K_STEPS; step_val++) {
            uint32_t a_list[FLASH_A_REGISTERS];
            flash_c_to_a(a_list, weight_list, step_val);
#pragma unroll
            for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b(b_list, grad_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(value_grad_list[n_val], a_list, b_list[0], b_list[1]);
            }
        }
        // B2, B4-B6: weights_gradᵀ = value rows · output_grad tileᵀ, then scores_gradᵀ in its place
        float score_grad_list[FLASH_TILE_N_TILES][FLASH_C_VALUES];
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++) {
            score_grad_list[n_val][0] = score_grad_list[n_val][1] = score_grad_list[n_val][2] = score_grad_list[n_val][3] = 0.0f;
#pragma unroll
            for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b_turned(b_list, grad_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(score_grad_list[n_val], value_a_list[step_val], b_list[0], b_list[1]);
            }
#pragma unroll
            for (int i_val = 0; i_val < FLASH_C_VALUES; i_val++)
                score_grad_list[n_val][i_val] = weight_list[n_val][i_val] *
                    (score_grad_list[n_val][i_val] - row_dot_tile_list[n_val * FLASH_MMA_N + pair_val + (i_val & 1)]);
        }
        // B8: key_grad += scores_gradᵀ · query tile
#pragma unroll
        for (int step_val = 0; step_val < FLASH_TILE_K_STEPS; step_val++) {
            uint32_t a_list[FLASH_A_REGISTERS];
            flash_c_to_a(a_list, score_grad_list, step_val);
#pragma unroll
            for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b(b_list, query_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(key_grad_list[n_val], a_list, b_list[0], b_list[1]);
            }
        }
    }

    // 3. into qkv_grad: each token's row is query, key, value parts of embed_dim each
#pragma unroll
    for (int r_val = 0; r_val < FLASH_THREAD_ROWS; r_val++) {
        int key_token_val = warp_first_key_val + group_val + (FLASH_MMA_M / 2) * r_val;
        nv_bfloat16* row_ptr = qkv_grad_ptr + ((size_t)batch_val * seq_len_val + key_token_val) * 3 * embed_dim_val +
                               head_val * FLASH_HEAD_SIZE;
#pragma unroll
        for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++) {
            *reinterpret_cast<nv_bfloat162*>(row_ptr + embed_dim_val + n_val * FLASH_MMA_N + pair_val) = __floats2bfloat162_rn(
                key_grad_list[n_val][2 * r_val] * scale_val, key_grad_list[n_val][2 * r_val + 1] * scale_val);
            *reinterpret_cast<nv_bfloat162*>(row_ptr + 2 * embed_dim_val + n_val * FLASH_MMA_N + pair_val) =
                __floats2bfloat162_rn(value_grad_list[n_val][2 * r_val], value_grad_list[n_val][2 * r_val + 1]);
        }
    }
#endif
}

/// Works out query_grad for FLASH_TILE queries of one head. Each warp keeps its FLASH_MMA_M query rows and
/// output_grad rows in registers and walks the keys FLASH_TILE at a time up to its own, making the weights again:
///     weights        = exp(query rows · key tileᵀ / sqrt(head_size) - logsumexp)    0 where key > query
///     weights_grad   = output_grad rows · value tileᵀ                               B2
///     scores_grad    = weights (weights_grad - row_dot)                             B4-B6
///     query_grad    += scores_grad · key tile                                       B7
///
///   qkv_grad_ptr   the query part written
///   the others     as in attention_flash_key_grad_kernel
__global__ void __launch_bounds__(FLASH_THREADS)
attention_flash_query_grad_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* output_grad_ptr,
                                  const nv_bfloat16* query_ptr, const nv_bfloat16* key_ptr, const nv_bfloat16* value_ptr,
                                  const float* logsumexp_ptr, const float* row_dot_ptr, int num_heads_val,
                                  int seq_len_val, float scale_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) nv_bfloat16 query_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 grad_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 key_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    __shared__ __align__(16) nv_bfloat16 value_tile_list[FLASH_TILE * FLASH_TILE_STRIDE];
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int group_val = lane_index_val / FLASH_GROUP_LANES;
    int pair_val = (lane_index_val % FLASH_GROUP_LANES) * 2;
    int query_block_val = gridDim.x - 1 - blockIdx.x;
    int batch_val = blockIdx.y / num_heads_val;
    int head_val = blockIdx.y % num_heads_val;
    int embed_dim_val = num_heads_val * FLASH_HEAD_SIZE;
    size_t head_start_val = (size_t)blockIdx.y * seq_len_val * FLASH_HEAD_SIZE;
    int warp_first_token_val = query_block_val * FLASH_TILE + warp_index_val * FLASH_MMA_M;

    // 1. this warp's rows of query and output_grad into registers, and its rows' logsumexp and row_dot
    flash_load_tile(query_tile_list, query_ptr + head_start_val + (size_t)query_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                    FLASH_HEAD_SIZE);
    flash_load_tile(grad_tile_list,
                    output_grad_ptr + ((size_t)batch_val * seq_len_val + query_block_val * FLASH_TILE) * embed_dim_val +
                        head_val * FLASH_HEAD_SIZE,
                    embed_dim_val);
    __syncthreads();
    uint32_t query_a_list[FLASH_K_STEPS][FLASH_A_REGISTERS];
    uint32_t grad_a_list[FLASH_K_STEPS][FLASH_A_REGISTERS];
#pragma unroll
    for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
        flash_load_a(query_a_list[step_val], query_tile_list, warp_index_val * FLASH_MMA_M, step_val * FLASH_MMA_K);
        flash_load_a(grad_a_list[step_val], grad_tile_list, warp_index_val * FLASH_MMA_M, step_val * FLASH_MMA_K);
    }
    // rows group, group + FLASH_MMA_M / 2
    float row_logsumexp_list[FLASH_THREAD_ROWS];
    float row_dot_of_row_list[FLASH_THREAD_ROWS];
#pragma unroll
    for (int r_val = 0; r_val < FLASH_THREAD_ROWS; r_val++) {
        size_t row_index_val = (size_t)blockIdx.y * seq_len_val + warp_first_token_val + group_val + (FLASH_MMA_M / 2) * r_val;
        row_logsumexp_list[r_val] = logsumexp_ptr[row_index_val];
        row_dot_of_row_list[r_val] = row_dot_ptr[row_index_val];
    }

    // 2. walk the key tiles up to this block's own
    float query_grad_list[FLASH_HEAD_N_TILES][FLASH_C_VALUES] = {};
    for (int key_block_val = 0; key_block_val <= query_block_val; key_block_val++) {
        __syncthreads();
        flash_load_tile(key_tile_list, key_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                        FLASH_HEAD_SIZE);
        flash_load_tile(value_tile_list, value_ptr + head_start_val + (size_t)key_block_val * FLASH_TILE * FLASH_HEAD_SIZE,
                        FLASH_HEAD_SIZE);
        __syncthreads();

        // the weights {FLASH_MMA_M, FLASH_TILE}, and weights_grad, then scores_grad in its place
        float weight_list[FLASH_TILE_N_TILES][FLASH_C_VALUES];
        float score_grad_list[FLASH_TILE_N_TILES][FLASH_C_VALUES];
#pragma unroll
        for (int n_val = 0; n_val < FLASH_TILE_N_TILES; n_val++) {
            weight_list[n_val][0] = weight_list[n_val][1] = weight_list[n_val][2] = weight_list[n_val][3] = 0.0f;
            score_grad_list[n_val][0] = score_grad_list[n_val][1] = score_grad_list[n_val][2] = score_grad_list[n_val][3] = 0.0f;
#pragma unroll
            for (int step_val = 0; step_val < FLASH_K_STEPS; step_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b_turned(b_list, key_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(weight_list[n_val], query_a_list[step_val], b_list[0], b_list[1]);
                flash_load_b_turned(b_list, value_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(score_grad_list[n_val], grad_a_list[step_val], b_list[0], b_list[1]);
            }
#pragma unroll
            for (int i_val = 0; i_val < FLASH_C_VALUES; i_val++) {
                int r_val = i_val / 2;
                int token_val = warp_first_token_val + group_val + (FLASH_MMA_M / 2) * r_val;
                int key_token_val = key_block_val * FLASH_TILE + n_val * FLASH_MMA_N + pair_val + (i_val & 1);
                float w_val = key_token_val <= token_val
                                  ? __expf(weight_list[n_val][i_val] * scale_val - row_logsumexp_list[r_val])
                                  : 0.0f;
                score_grad_list[n_val][i_val] = w_val * (score_grad_list[n_val][i_val] - row_dot_of_row_list[r_val]);
            }
        }
        // B7: query_grad += scores_grad · key tile
#pragma unroll
        for (int step_val = 0; step_val < FLASH_TILE_K_STEPS; step_val++) {
            uint32_t a_list[FLASH_A_REGISTERS];
            flash_c_to_a(a_list, score_grad_list, step_val);
#pragma unroll
            for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++) {
                uint32_t b_list[FLASH_B_REGISTERS];
                flash_load_b(b_list, key_tile_list, n_val * FLASH_MMA_N, step_val * FLASH_MMA_K);
                flash_mma(query_grad_list[n_val], a_list, b_list[0], b_list[1]);
            }
        }
    }

    // 3. into qkv_grad's query part
#pragma unroll
    for (int r_val = 0; r_val < FLASH_THREAD_ROWS; r_val++) {
        int token_val = warp_first_token_val + group_val + (FLASH_MMA_M / 2) * r_val;
        nv_bfloat16* row_ptr = qkv_grad_ptr + ((size_t)batch_val * seq_len_val + token_val) * 3 * embed_dim_val +
                               head_val * FLASH_HEAD_SIZE;
#pragma unroll
        for (int n_val = 0; n_val < FLASH_HEAD_N_TILES; n_val++)
            *reinterpret_cast<nv_bfloat162*>(row_ptr + n_val * FLASH_MMA_N + pair_val) = __floats2bfloat162_rn(
                query_grad_list[n_val][2 * r_val] * scale_val, query_grad_list[n_val][2 * r_val + 1] * scale_val);
    }
#endif
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the cuBLAS handle every product uses, made on the first call and kept.
inline cublasHandle_t attention_cublas() {
    static cublasHandle_t handle_ptr = [] {
        cublasHandle_t made_ptr;
        attentionCublasCheck(cublasCreate(&made_ptr));
        return made_ptr;
    }();
    return handle_ptr;
}

/// Makes one product for every head at once: C = alpha · op(A) · op(B), written over C, with cuBLAS's strided
/// batched product (cublasGemmStridedBatchedEx for bfloat16, sums in float).
///
///   op_a_val, op_b_val                          turn A or B round, or not
///   m_val, n_val, k_val                         C is m x n as cuBLAS sees it (by columns); k the summed index
///   alpha_val                                   the scale
///   a_tensor, b_tensor, c_tensor                the tensors
///   lda_val, ldb_val, ldc_val                   values one column takes
///   stride_a_val, stride_b_val, stride_c_val    values from one head's matrix to the next
///   batch_val                                   how many heads
inline void attention_product(cublasOperation_t op_a_val, cublasOperation_t op_b_val, int m_val, int n_val, int k_val,
                              float alpha_val, const Tensor& a_tensor, int lda_val, long long stride_a_val,
                              const Tensor& b_tensor, int ldb_val, long long stride_b_val, const Tensor& c_tensor,
                              int ldc_val, long long stride_c_val, int batch_val) {
    float zero_val = 0.0f;
    if (c_tensor.dtype() == DType::BFloat16)
        attentionCublasCheck(cublasGemmStridedBatchedEx(
            attention_cublas(), op_a_val, op_b_val, m_val, n_val, k_val, &alpha_val, a_tensor.data_ptr(), CUDA_R_16BF,
            lda_val, stride_a_val, b_tensor.data_ptr(), CUDA_R_16BF, ldb_val, stride_b_val, &zero_val,
            c_tensor.data_ptr(), CUDA_R_16BF, ldc_val, stride_c_val, batch_val, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    else
        attentionCublasCheck(cublasSgemmStridedBatched(
            attention_cublas(), op_a_val, op_b_val, m_val, n_val, k_val, &alpha_val, a_tensor.data(), lda_val,
            stride_a_val, b_tensor.data(), ldb_val, stride_b_val, &zero_val, c_tensor.data(), ldc_val, stride_c_val,
            batch_val));
}

/// Gives where float4 `index_val` of a per-head tensor sits in a per-token tensor:
///     per head:    index = ((batch · num_heads + head) · seq_len + token) · head_float4s + column
///     per token:   (batch · seq_len + token) · row_float4s + head · head_float4s + column
///
///   index_val                    float4 of {batch_size, num_heads, seq_len, head_size}
///   num_heads_val, seq_len_val   the shape
///   head_float4s_val             head_size / VALUES_PER_FLOAT4
///   row_float4s_val              a per-token row: embed_dim (output) or 3 embed_dim (qkv), / VALUES_PER_FLOAT4
__device__ __forceinline__ size_t attention_merged_index(size_t index_val, size_t num_heads_val, size_t seq_len_val,
                                                         size_t head_float4s_val, size_t row_float4s_val) {
    size_t column_val = index_val % head_float4s_val;
    size_t token_val = (index_val / head_float4s_val) % seq_len_val;
    size_t head_val = (index_val / (head_float4s_val * seq_len_val)) % num_heads_val;
    size_t batch_val = index_val / (head_float4s_val * seq_len_val * num_heads_val);
    return (batch_val * seq_len_val + token_val) * row_float4s_val + head_val * head_float4s_val + column_val;
}

/// Gives the blocks for `total_val` pieces of work, rounded up.
///
///   total_val               pieces of work: float4s or rows
///   threads_per_block_val   the launch's
///   per_thread_val          threads per piece: 1 for a float4, THREADS_PER_WARP for a row (one warp)
inline int attention_blocks(size_t total_val, int threads_per_block_val, size_t per_thread_val) {
    size_t threads_val = total_val * per_thread_val;
    return (int)CEIL_DIV(threads_val, (size_t)threads_per_block_val);
}

/// Stops with a message unless the tensors fit: the shapes at the top of the file, all float or all bfloat16,
/// head_size a multiple of VALUES_PER_FLOAT4, each starting on a whole float4.
///
///   output_tensor, query_tensor, key_tensor, value_tensor, weights_tensor, head_outputs_tensor, qkv_tensor
///       forward: these; backward: the grads in their place
inline void attention_check_shapes(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                   const Tensor& value_tensor, const Tensor& weights_tensor,
                                   const Tensor& head_outputs_tensor, const Tensor& qkv_tensor) {
    // shapes: as at the top of the file
    bool fits_val = query_tensor.dim() == 4 && key_tensor.shape() == query_tensor.shape() &&
                    value_tensor.shape() == query_tensor.shape() && head_outputs_tensor.shape() == query_tensor.shape();
    if (fits_val) {
        size_t batch_size_val = query_tensor.size(0);
        size_t num_heads_val = query_tensor.size(1);
        size_t seq_len_val = query_tensor.size(2);
        size_t embed_dim_val = num_heads_val * query_tensor.size(3);
        fits_val = weights_tensor.shape() == Shape({batch_size_val, num_heads_val, seq_len_val, seq_len_val}) &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val});
    }
    if (!fits_val) {
        fprintf(stderr, "error: attention needs query, key, value and head_outputs {batch_size, num_heads, seq_len, "
                        "head_size}; weights {batch_size, num_heads, seq_len, seq_len}; output {batch_size, seq_len, "
                        "embed_dim} and qkv {batch_size, seq_len, 3 * embed_dim}, embed_dim = num_heads * head_size\n");
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    const Tensor* all_list[] = {&output_tensor, &query_tensor, &key_tensor, &value_tensor, &weights_tensor,
                                &head_outputs_tensor, &qkv_tensor};
    DType dtype_val = qkv_tensor.dtype();
    bool same_dtype_val = dtype_val == DType::Float32 || dtype_val == DType::BFloat16;
    for (const Tensor* tensor_ptr : all_list) same_dtype_val = same_dtype_val && tensor_ptr->dtype() == dtype_val;
    if (!same_dtype_val) {
        fprintf(stderr, "error: attention needs its tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: heads of whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * qkv_tensor.element_size();
    bool aligned_val = query_tensor.size(3) % VALUES_PER_FLOAT4 == 0;
    for (const Tensor* tensor_ptr : all_list) aligned_val = aligned_val && (size_t)tensor_ptr->data_ptr() % float4_bytes_val == 0;
    if (!aligned_val) {
        fprintf(stderr, "error: attention reads %d values at a time, so head_size must be a multiple of %d and every "
                        "tensor must start on %zu bytes; got head_size %zu\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, query_tensor.size(3));
        exit(EXIT_FAILURE);
    }
}

/// Stops with a message unless flash attention can run these tensors: the shapes, bfloat16, logsumexp float,
/// attention_flash_fits, each starting on a 16-byte load.
///
///   output_tensor, query_tensor, key_tensor, value_tensor, qkv_tensor
///       forward: these; backward: output_grad or output, and qkv_grad
///   logsumexp_tensor   {batch_size, num_heads, seq_len}; backward: also row_dot
inline void attention_flash_check_shapes(const Tensor& output_tensor, const Tensor& query_tensor, const Tensor& key_tensor,
                                         const Tensor& value_tensor, const Tensor& logsumexp_tensor,
                                         const Tensor& qkv_tensor) {
    // shapes, types and addresses together
    bool fits_val = query_tensor.dim() == 4 && key_tensor.shape() == query_tensor.shape() &&
                    value_tensor.shape() == query_tensor.shape() &&
                    attention_flash_fits(qkv_tensor.dtype(), query_tensor.size(2), query_tensor.size(3));
    if (fits_val) {
        size_t batch_size_val = query_tensor.size(0);
        size_t num_heads_val = query_tensor.size(1);
        size_t seq_len_val = query_tensor.size(2);
        size_t embed_dim_val = num_heads_val * query_tensor.size(3);
        fits_val = logsumexp_tensor.shape() == Shape({batch_size_val, num_heads_val, seq_len_val}) &&
                   logsumexp_tensor.dtype() == DType::Float32 &&
                   output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   qkv_tensor.shape() == Shape({batch_size_val, seq_len_val, 3 * embed_dim_val});
        for (const Tensor* tensor_ptr : {&output_tensor, &query_tensor, &key_tensor, &value_tensor, &qkv_tensor})
            fits_val = fits_val && tensor_ptr->dtype() == DType::BFloat16 && (size_t)tensor_ptr->data_ptr() % sizeof(uint4) == 0;
    }
    if (!fits_val) {
        fprintf(stderr, "error: flash attention needs bfloat16 query, key, value {batch_size, num_heads, seq_len, %d} "
                        "with seq_len a multiple of %d, output {batch_size, seq_len, embed_dim}, qkv {batch_size, "
                        "seq_len, 3 * embed_dim}, a float logsumexp {batch_size, num_heads, seq_len}, all starting on "
                        "%zu bytes, and a GPU of compute capability 8.0 or more\n",
                FLASH_HEAD_SIZE, FLASH_TILE, sizeof(uint4));
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_CUH
