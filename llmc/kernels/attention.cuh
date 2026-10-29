// attention.cuh — causal self-attention, GPT-2's way for each token to take in what the tokens before it
// say. The qkv linear gives every token a query (what it looks for), a key (what it offers) and a value
// (what it passes on), for each of num_heads heads. In each head, each token scores every earlier token
// (and itself), turns the scores into weights that add up to 1, and takes that mix of their values:
//
//     F1   split    query, key, value  = qkv's three parts, each cut into heads of head_size columns
//     F2   scores   score[t][j]        = query[t] · key[j] / sqrt(head_size)
//     F3   mask     only j <= t: a token cannot see the tokens after it
//     F4   softmax  weight[t][j]       = exp(score[t][j] - row_max) / row_sum, over j <= t
//     F5   mix      head_output[t]     = sum over j <= t of weight[t][j] * value[j]
//     F6   merge    output             = the heads side by side again
//
//     qkv, qkv_grad                {batch_size, seq_len, 3 * embed_dim}             {4, 1024, 2304}
//     output, output_grad          {batch_size, seq_len, embed_dim}                 {4, 1024, 768}
//     query, key, value            {batch_size, num_heads, seq_len, head_size}      {4, 12, 1024, 64}, kept for backward
//     weights                      {batch_size, num_heads, seq_len, seq_len}        {4, 12, 1024, 1024}, kept for backward
//     head_outputs, head_outputs_grad, query_grad, key_grad, value_grad   as query, scratch: used only inside a call
//     weights_grad                 as weights, scratch
//
// embed_dim = num_heads * head_size (768 = 12 * 64). The numbers of heads and head_size come from query's
// shape. weights is 201 MB for GPT-2 small: one float for every pair of tokens in every head.
//
// Backward, with output_grad the gradient arriving from the layer above (the attention's output linear):
//
//     B1   head_outputs_grad   = output_grad cut into heads again                       (F6 undone)
//     B2   weights_grad[t][j]  = head_outputs_grad[t] · value[j]                        (F5, for weights)
//     B3   value_grad[j]       = sum over t >= j of weight[t][j] * head_outputs_grad[t]  (F5, for value)
//     B4-B6 scores_grad[t][j]  = weight[t][j] * (weights_grad[t][j] - row_dot[t])       (F4, softmax: C1)
//                                row_dot[t] = sum over j <= t of weight[t][j] * weights_grad[t][j]; 0 above
//     B7   query_grad[t]       = sum over j <= t of scores_grad[t][j] * key[j] / sqrt(head_size)    (F2)
//     B8   key_grad[j]         = sum over t >= j of scores_grad[t][j] * query[t] / sqrt(head_size)  (F2)
//     B9   qkv_grad            = query_grad, key_grad, value_grad put back together     (F1 undone)
//
// The math is in the note https://ducbachsong.github.io/posts/attention-backward/. The comments name its
// steps (F1..F6 forward, B1..B9 backward, C1 and C2 rebuilt) and sections. Its letters for the names here:
//
//     q, k, v = query, key, value   s = scores   a = weights   y = head_outputs   d = head_size
//     dy = head_outputs_grad   da = weights_grad   ds = scores_grad   dq, dk, dv = query_grad, key_grad, value_grad
//
// The note's example, one head of 3 tokens with head_size 4, q = [2,0,0,0] [0,2,0,0] [2,2,0,0],
// k = [1,0,0,0] [0,1,0,0] [1,1,0,0] and v = [1,0,0,0] [0,1,0,0] [0,0,1,0], gives the weights
//
//     token 0:   1           0           0
//     token 1:   0.2689414   0.7310586   0
//     token 2:   0.2119416   0.2119416   0.5761169
//
// and, v being one 1 per row, head outputs that are those rows with a 0 after them.
//
// F2, F5, B2, B3, B7 and B8 are matrix products, one per (batch, head): 48 at once for GPT-2 small, made by
// cuBLAS's strided batched product (note 6). F4 and B4-B6 are a kernel each, one warp per row of weights;
// F1, F6, B1 and B9 only move values, one float4 per thread. A name ending in 4 is one float4 of that tensor.
//
// The tensors are all float or all bfloat16. The kernels are templates on that dtype, T, and the six products
// are told it (attention_product: CUDA_R_16BF for bfloat16). A float4 is 4 neighbouring values either way:
// for bfloat16, one 8-byte read that load4 (tensor.cuh) turns into 4 floats, and store4 rounds back. The
// softmax and every product sum in float; each result is rounded to the tensors' dtype once, as it is stored.
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <cublas_v2.h>
#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define ATTENTION_THREADS_PER_BLOCK 256           // F1, F6, B1, B9: one float4 per thread
#define ATTENTION_SOFTMAX_THREADS_PER_BLOCK 256   // F4, B4-B6: 8 warps, one row of weights each

// ── forward and backward ──────────────────────────────────────────────────────
void attention_forward(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                       const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv);
void attention_backward(const Tensor& qkv_grad, const Tensor& output_grad, const Tensor& query, const Tensor& key,
                        const Tensor& value, const Tensor& weights, const Tensor& head_outputs_grad,
                        const Tensor& weights_grad, const Tensor& query_grad, const Tensor& key_grad,
                        const Tensor& value_grad);

// ── for both ──────────────────────────────────────────────────────────────────
cublasHandle_t attention_cublas();
void attention_product(cublasOperation_t op_a, cublasOperation_t op_b, int m, int n, int k, float alpha,
                       const Tensor& a, int lda, long long stride_a, const Tensor& b, int ldb, long long stride_b,
                       const Tensor& c, int ldc, long long stride_c, int batch);
__device__ __forceinline__ size_t attention_merged_index(size_t index, size_t num_heads, size_t seq_len,
                                                         size_t head_float4s, size_t row_float4s);
int attention_blocks(size_t total, int threads_per_block, size_t per_thread);
void attention_check_shapes(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                            const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv);

/// Stops with a message if a cuBLAS call did not succeed, naming the file and line of the call.
static inline void attention_cublas_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %s\n", file, line, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}
#define attentionCublasCheck(call) attention_cublas_check((call), __FILE__, __LINE__)

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// What attention_forward calls, declared here so it can call them; their code follows it.
template <typename Value>
void attention_forward_in(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                          const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv);
template <typename T>
__global__ void attention_split_qkv_kernel(T* query, T* key, T* value, const T* qkv, size_t total_float4s,
                                           size_t num_heads, size_t seq_len, size_t head_float4s);
template <typename T>
__global__ void attention_softmax_forward_kernel(T* weights, size_t rows, size_t seq_len);
template <typename T>
__global__ void attention_merge_heads_kernel(T* output, const T* head_outputs, size_t total_float4s,
                                             size_t num_heads, size_t seq_len, size_t head_float4s);

/// Writes output = causal self-attention of qkv, and keeps query, key, value and weights for the backward.
/// head_outputs is scratch. The shapes are those of the head comment; num_heads and head_size are read from
/// query. The six steps F1..F6 run one after the other on the GPU's queue:
///
///     step       how                                   GPT-2 small, {4, 1024} tokens, 12 heads of 64
///     F1         attention_split_qkv_kernel            786,432 float4s of query (and key, value), 3072 blocks
///     F2         cuBLAS, 48 products {1024, 64} · {64, 1024}, scaled by 1 / sqrt(64) = 0.125
///     F3, F4     attention_softmax_forward_kernel      49,152 rows of weights, 6144 blocks of 8 warps
///     F5         cuBLAS, 48 products {1024, 1024} · {1024, 64}
///     F6         attention_merge_heads_kernel          786,432 float4s, 3072 blocks
inline void attention_forward(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                              const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv) {
    attention_check_shapes(output, query, key, value, weights, head_outputs, qkv);
    if (qkv.dtype() == DType::BFloat16)
        attention_forward_in<nv_bfloat16>(output, query, key, value, weights, head_outputs, qkv);
    else
        attention_forward_in<float>(output, query, key, value, weights, head_outputs, qkv);
}

/// attention_forward's six steps, the tensors' values being Value: float or nv_bfloat16. The tensors are
/// already checked. (Value, not T: T is seq_len here, as in the note.)
template <typename Value>
inline void attention_forward_in(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                                 const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv) {
    size_t num_heads = query.size(1), seq_len = query.size(2), head_size = query.size(3);
    size_t heads_in_batch = query.size(0) * num_heads, rows = heads_in_batch * seq_len;   // 48 heads, 49,152 rows
    size_t total_float4s = query.numel() / 4, head_float4s = head_size / 4;
    int T = (int)seq_len, d = (int)head_size;
    float scale = 1.0f / sqrtf((float)head_size);
    long long head_values = (long long)T * d, head_weights = (long long)T * T;   // the stride from head to head

    // F1: query, key, value, each head's rows next to each other, as the batched products read them
    attention_split_qkv_kernel<<<attention_blocks(total_float4s, ATTENTION_THREADS_PER_BLOCK, 1),
                                 ATTENTION_THREADS_PER_BLOCK>>>(
        query.data_ptr<Value>(), key.data_ptr<Value>(), value.data_ptr<Value>(), qkv.data_ptr<Value>(), total_float4s,
        num_heads, seq_len, head_float4s);
    cudaCheck(cudaGetLastError());

    // F2: scores = scale · query · keyᵀ, {T, T} per head, into weights. cuBLAS reads by columns (see
    // linear_cublas.cuh), so it is asked for scoresᵀ = key · queryᵀ: key's memory, read by columns, is keyᵀ
    // {d, T}, turned round (OP_T) it is key {T, d}; query's memory read by columns is queryᵀ {d, T}, as is.
    attention_product(CUBLAS_OP_T, CUBLAS_OP_N, T, T, d, scale, key, d, head_values, query, d, head_values, weights,
                      T, head_weights, (int)heads_in_batch);

    // F3, F4: the mask and the softmax, in place: weights holds the scores before, the weights after
    attention_softmax_forward_kernel<<<attention_blocks(rows, ATTENTION_SOFTMAX_THREADS_PER_BLOCK, 32),
                                       ATTENTION_SOFTMAX_THREADS_PER_BLOCK>>>(weights.data_ptr<Value>(), rows, seq_len);
    cudaCheck(cudaGetLastError());

    // F5: head_outputs = weights · value, {T, d} per head; asked for as head_outputsᵀ = valueᵀ · weightsᵀ,
    // which are value's and weights' memory read by columns, both as is.
    attention_product(CUBLAS_OP_N, CUBLAS_OP_N, d, T, T, 1.0f, value, d, head_values, weights, T, head_weights,
                      head_outputs, d, head_values, (int)heads_in_batch);

    // F6: the heads side by side again, one row of embed_dim per token
    attention_merge_heads_kernel<<<attention_blocks(total_float4s, ATTENTION_THREADS_PER_BLOCK, 1),
                                   ATTENTION_THREADS_PER_BLOCK>>>(
        output.data_ptr<Value>(), head_outputs.data_ptr<Value>(), total_float4s, num_heads, seq_len, head_float4s);
    cudaCheck(cudaGetLastError());
}

/// F1: copies qkv's three parts into query, key and value, one float4 of each per thread. Float4 n of query
/// is (batch, head, token, the n-th 4 columns of the head), and the same float4 of qkv's row (batch, token)
/// is in its query part; key's and value's are embed_dim and 2 embed_dim floats further along that row.
template <typename T>
__global__ void attention_split_qkv_kernel(T* query, T* key, T* value, const T* qkv, size_t total_float4s,
                                           size_t num_heads, size_t seq_len, size_t head_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s = num_heads * head_float4s;                       // 768 / 4 = 192
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        size_t query_part = attention_merged_index(index, num_heads, seq_len, head_float4s, 3 * embed_float4s);
        store4(query + 4 * index, load4(qkv + 4 * query_part));
        store4(key + 4 * index, load4(qkv + 4 * (query_part + embed_float4s)));
        store4(value + 4 * index, load4(qkv + 4 * (query_part + 2 * embed_float4s)));
    }
}

/// F3, F4: turns each row of scores into weights, in place, one warp per row. Row r is token r % seq_len's
/// row of its head: it may look at columns 0 .. token (F3), and the columns after are set to 0, so F5's
/// product, which reads the whole row, adds nothing from them.
///
/// row_max and row_sum come from one read of the row, the online softmax of classifier.cuh: each thread
/// keeps the largest score it has seen and the sum of exp(score - that largest), over columns
/// thread_in_warp, + 32, ...; then the 32 pairs are put together with shuffles. A second read writes
/// exp(score - row_max) / row_sum. Each thread writes only the columns it read.
template <typename T>
__global__ void attention_softmax_forward_kernel(T* weights, size_t rows, size_t seq_len) {
    int thread_in_warp = threadIdx.x % 32;
    size_t warp_number = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t warps_launched = (size_t)gridDim.x * blockDim.x / 32;
    for (size_t row = warp_number; row < rows; row += warps_launched) {
        size_t token = row % seq_len;                    // the query's token: it sees columns 0 .. token
        T* row_weights = weights + row * seq_len;

        // 1. this thread's (max, sum), over its columns up to token
        float thread_max = -FLT_MAX, thread_sum = 0.0f;
        for (size_t column = thread_in_warp; column <= token; column += 32) {
            float score = to_float(row_weights[column]);
            float new_max = fmaxf(thread_max, score);
            thread_sum = thread_sum * expf(thread_max - new_max) + expf(score - new_max);
            thread_max = new_max;
        }

        // 2. the row's: every thread adds the part of the thread `offset` away, 16, 8, 4, 2, 1; a thread with
        //    no columns (token < 31) brings (-FLT_MAX, 0), which adds 0
        float row_max = thread_max, row_sum = thread_sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, row_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, row_sum, offset);
            float new_max = fmaxf(row_max, other_max);   // each sum moved onto the larger max, then added
            row_sum = row_sum * expf(row_max - new_max) + other_sum * expf(other_max - new_max);
            row_max = new_max;
        }

        // 3. the weights: F4 for the columns it sees, 0 for the ones after (F3)
        float one_over_row_sum = 1.0f / row_sum;
        for (size_t column = thread_in_warp; column < seq_len; column += 32)
            row_weights[column] = from_float<T>(
                column <= token ? expf(to_float(row_weights[column]) - row_max) * one_over_row_sum : 0.0f);
    }
}

/// F6: copies head_outputs into output, one float4 per thread: float4 n of head_outputs (batch, head, token,
/// columns) goes to row (batch, token) of output, at the head's place in it.
template <typename T>
__global__ void attention_merge_heads_kernel(T* output, const T* head_outputs, size_t total_float4s,
                                             size_t num_heads, size_t seq_len, size_t head_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s = num_heads * head_float4s;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x)
        store4(output + 4 * attention_merged_index(index, num_heads, seq_len, head_float4s, embed_float4s),
               load4(head_outputs + 4 * index));
}

// ── backward ──────────────────────────────────────────────────────────────────

// What attention_backward calls, declared here so it can call them; their code follows it.
template <typename Value>
void attention_backward_in(const Tensor& qkv_grad, const Tensor& output_grad, const Tensor& query, const Tensor& key,
                           const Tensor& value, const Tensor& weights, const Tensor& head_outputs_grad,
                           const Tensor& weights_grad, const Tensor& query_grad, const Tensor& key_grad,
                           const Tensor& value_grad);
template <typename T>
__global__ void attention_split_heads_kernel(T* head_outputs_grad, const T* output_grad, size_t total_float4s,
                                             size_t num_heads, size_t seq_len, size_t head_float4s);
template <typename T>
__global__ void attention_softmax_backward_kernel(T* weights_grad, const T* weights, size_t rows, size_t seq_len);
template <typename T>
__global__ void attention_merge_qkv_grad_kernel(T* qkv_grad, const T* query_grad, const T* key_grad,
                                                const T* value_grad, size_t total_float4s, size_t num_heads,
                                                size_t seq_len, size_t head_float4s);

/// Writes qkv_grad, the gradient of the qkv linear's output, from output_grad and what the forward kept:
/// query, key, value and weights. qkv_grad is written over: qkv is read by the attention alone. The other
/// five are scratch, with the shapes of the head comment; weights_grad holds da after B2 and ds after B6.
/// The nine steps:
///
///     step       how                                   what
///     B1         attention_split_heads_kernel          head_outputs_grad, output_grad cut into heads
///     B2         cuBLAS, 48 x {T, d} · {d, T}           weights_grad = head_outputs_grad · valueᵀ
///     B3         cuBLAS, 48 x {T, T}ᵀ · {T, d}          value_grad = weightsᵀ · head_outputs_grad
///     B4-B6      attention_softmax_backward_kernel     weights_grad becomes scores_grad, in place: C1
///     B7         cuBLAS, 48 x {T, T} · {T, d}           query_grad = scores_grad · key / sqrt(d)
///     B8         cuBLAS, 48 x {T, T}ᵀ · {T, d}          key_grad = scores_gradᵀ · query / sqrt(d)
///     B9         attention_merge_qkv_grad_kernel       qkv_grad, the three put back together
inline void attention_backward(const Tensor& qkv_grad, const Tensor& output_grad, const Tensor& query, const Tensor& key,
                               const Tensor& value, const Tensor& weights, const Tensor& head_outputs_grad,
                               const Tensor& weights_grad, const Tensor& query_grad, const Tensor& key_grad,
                               const Tensor& value_grad) {
    attention_check_shapes(output_grad, query, key, value, weights, head_outputs_grad, qkv_grad);
    attention_check_shapes(output_grad, query_grad, key_grad, value_grad, weights_grad, head_outputs_grad, qkv_grad);
    if (qkv_grad.dtype() == DType::BFloat16)
        attention_backward_in<nv_bfloat16>(qkv_grad, output_grad, query, key, value, weights, head_outputs_grad,
                                           weights_grad, query_grad, key_grad, value_grad);
    else
        attention_backward_in<float>(qkv_grad, output_grad, query, key, value, weights, head_outputs_grad,
                                     weights_grad, query_grad, key_grad, value_grad);
}

/// attention_backward's nine steps, the tensors' values being Value: float or nv_bfloat16. The tensors are
/// already checked.
template <typename Value>
inline void attention_backward_in(const Tensor& qkv_grad, const Tensor& output_grad, const Tensor& query,
                                  const Tensor& key, const Tensor& value, const Tensor& weights,
                                  const Tensor& head_outputs_grad, const Tensor& weights_grad,
                                  const Tensor& query_grad, const Tensor& key_grad, const Tensor& value_grad) {
    size_t num_heads = query.size(1), seq_len = query.size(2), head_size = query.size(3);
    size_t heads_in_batch = query.size(0) * num_heads, rows = heads_in_batch * seq_len;
    size_t total_float4s = query.numel() / 4, head_float4s = head_size / 4;
    int T = (int)seq_len, d = (int)head_size;
    float scale = 1.0f / sqrtf((float)head_size);
    long long head_values = (long long)T * d, head_weights = (long long)T * T;   // the stride from head to head

    // B1: output_grad cut into heads, as F1 cut qkv
    attention_split_heads_kernel<<<attention_blocks(total_float4s, ATTENTION_THREADS_PER_BLOCK, 1),
                                   ATTENTION_THREADS_PER_BLOCK>>>(
        head_outputs_grad.data_ptr<Value>(), output_grad.data_ptr<Value>(), total_float4s, num_heads, seq_len,
        head_float4s);
    cudaCheck(cudaGetLastError());

    // B2: weights_grad = head_outputs_grad · valueᵀ, asked for as its turned-round, value · head_outputs_gradᵀ:
    // F2's form, value and head_outputs_grad in place of key and query
    attention_product(CUBLAS_OP_T, CUBLAS_OP_N, T, T, d, 1.0f, value, d, head_values, head_outputs_grad, d,
                      head_values, weights_grad, T, head_weights, (int)heads_in_batch);

    // B3: value_grad = weightsᵀ · head_outputs_grad, asked for as value_gradᵀ = head_outputs_gradᵀ · weights:
    // head_outputs_grad's memory as is, weights' turned round
    attention_product(CUBLAS_OP_N, CUBLAS_OP_T, d, T, T, 1.0f, head_outputs_grad, d, head_values, weights, T,
                      head_weights, value_grad, d, head_values, (int)heads_in_batch);

    // B4-B6: the softmax's backward, in place: weights_grad holds da before, ds after
    attention_softmax_backward_kernel<<<attention_blocks(rows, ATTENTION_SOFTMAX_THREADS_PER_BLOCK, 32),
                                        ATTENTION_SOFTMAX_THREADS_PER_BLOCK>>>(
        weights_grad.data_ptr<Value>(), weights.data_ptr<Value>(), rows, seq_len);
    cudaCheck(cudaGetLastError());

    // B7: query_grad = scale · scores_grad · key, asked for as query_gradᵀ = keyᵀ · scores_gradᵀ, both as is
    attention_product(CUBLAS_OP_N, CUBLAS_OP_N, d, T, T, scale, key, d, head_values, weights_grad, T, head_weights,
                      query_grad, d, head_values, (int)heads_in_batch);

    // B8: key_grad = scale · scores_gradᵀ · query, asked for as key_gradᵀ = queryᵀ · scores_grad: query's memory
    // as is, scores_grad's turned round
    attention_product(CUBLAS_OP_N, CUBLAS_OP_T, d, T, T, scale, query, d, head_values, weights_grad, T, head_weights,
                      key_grad, d, head_values, (int)heads_in_batch);

    // B9: the three gradients put back into qkv_grad's three parts, as F1 took them out
    attention_merge_qkv_grad_kernel<<<attention_blocks(total_float4s, ATTENTION_THREADS_PER_BLOCK, 1),
                                      ATTENTION_THREADS_PER_BLOCK>>>(
        qkv_grad.data_ptr<Value>(), query_grad.data_ptr<Value>(), key_grad.data_ptr<Value>(),
        value_grad.data_ptr<Value>(), total_float4s, num_heads, seq_len, head_float4s);
    cudaCheck(cudaGetLastError());
}

/// B1: copies output_grad into head_outputs_grad, one float4 per thread: F6 the other way round.
template <typename T>
__global__ void attention_split_heads_kernel(T* head_outputs_grad, const T* output_grad, size_t total_float4s,
                                             size_t num_heads, size_t seq_len, size_t head_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s = num_heads * head_float4s;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x)
        store4(head_outputs_grad + 4 * index,
               load4(output_grad + 4 * attention_merged_index(index, num_heads, seq_len, head_float4s, embed_float4s)));
}

/// B4-B6: turns each row of weights_grad (da) into scores_grad (ds), in place, one warp per row, as the
/// forward walks the rows (note C1):
///
///     ds[t][j] = a[t][j] * (da[t][j] - row_dot)      row_dot = sum over j <= t of a[t][j] * da[t][j]
///
/// for the columns token t sees, and 0 for the ones after: their weights were 0, so are their gradients.
/// One read for row_dot, added up across the warp with shuffles; a second read writes ds. Each thread writes
/// only the columns it read.
template <typename T>
__global__ void attention_softmax_backward_kernel(T* weights_grad, const T* weights, size_t rows, size_t seq_len) {
    int thread_in_warp = threadIdx.x % 32;
    size_t warp_number = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t warps_launched = (size_t)gridDim.x * blockDim.x / 32;
    for (size_t row = warp_number; row < rows; row += warps_launched) {
        size_t token = row % seq_len;
        T* row_grad = weights_grad + row * seq_len;
        const T* row_weights = weights + row * seq_len;

        // 1. row_dot: this thread's part, then the warp's
        float row_dot = 0.0f;
        for (size_t column = thread_in_warp; column <= token; column += 32)
            row_dot += to_float(row_weights[column]) * to_float(row_grad[column]);
        for (int offset = 16; offset > 0; offset /= 2)
            row_dot += __shfl_xor_sync(0xffffffff, row_dot, offset);

        // 2. ds: C1 for the columns token sees, 0 for the ones after
        for (size_t column = thread_in_warp; column < seq_len; column += 32)
            row_grad[column] = from_float<T>(
                column <= token ? to_float(row_weights[column]) * (to_float(row_grad[column]) - row_dot) : 0.0f);
    }
}

/// B9: copies query_grad, key_grad and value_grad into qkv_grad's three parts, one float4 of each per thread:
/// F1 the other way round.
template <typename T>
__global__ void attention_merge_qkv_grad_kernel(T* qkv_grad, const T* query_grad, const T* key_grad,
                                                const T* value_grad, size_t total_float4s, size_t num_heads,
                                                size_t seq_len, size_t head_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_float4s = num_heads * head_float4s;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        size_t query_part = attention_merged_index(index, num_heads, seq_len, head_float4s, 3 * embed_float4s);
        store4(qkv_grad + 4 * query_part, load4(query_grad + 4 * index));
        store4(qkv_grad + 4 * (query_part + embed_float4s), load4(key_grad + 4 * index));
        store4(qkv_grad + 4 * (query_part + 2 * embed_float4s), load4(value_grad + 4 * index));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the cuBLAS handle every product uses, making it on the first call (it takes milliseconds) and
/// keeping it until the program ends. A handle of its own, so this file does not need linear_cublas.cuh.
inline cublasHandle_t attention_cublas() {
    static cublasHandle_t handle = [] {
        cublasHandle_t made;
        attentionCublasCheck(cublasCreate(&made));
        return made;
    }();
    return handle;
}

/// Makes one of the six products for every head at once, with cuBLAS's strided batched product: C = alpha ·
/// op(A) · op(B), written over C (beta 0), for `batch` heads, each head's matrix stride_* values after the
/// last. m x n is C's size as cuBLAS sees it (by columns: see linear_cublas.cuh), k the summed index, ld* how
/// many values one column takes. Float tensors go to cublasSgemmStridedBatched; bfloat16 tensors to
/// cublasGemmStridedBatchedEx, told CUDA_R_16BF and to sum in float (CUBLAS_COMPUTE_32F).
inline void attention_product(cublasOperation_t op_a, cublasOperation_t op_b, int m, int n, int k, float alpha,
                              const Tensor& a, int lda, long long stride_a, const Tensor& b, int ldb,
                              long long stride_b, const Tensor& c, int ldc, long long stride_c, int batch) {
    float zero = 0.0f;
    if (c.dtype() == DType::BFloat16)
        attentionCublasCheck(cublasGemmStridedBatchedEx(
            attention_cublas(), op_a, op_b, m, n, k, &alpha, a.data_ptr(), CUDA_R_16BF, lda, stride_a, b.data_ptr(),
            CUDA_R_16BF, ldb, stride_b, &zero, c.data_ptr(), CUDA_R_16BF, ldc, stride_c, batch, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    else
        attentionCublasCheck(cublasSgemmStridedBatched(attention_cublas(), op_a, op_b, m, n, k, &alpha, a.data(), lda,
                                                       stride_a, b.data(), ldb, stride_b, &zero, c.data(), ldc,
                                                       stride_c, batch));
}

/// Gives where float4 `index` of a per-head tensor {batch_size, num_heads, seq_len, head_size} sits in a
/// per-token tensor whose rows are row_float4s long and start with the heads side by side: output
/// (row_float4s = embed_dim / 4) or qkv's query part (3 embed_dim / 4). The same batch, token, head and column.
///
///     per head:    index = ((batch * num_heads + head) * seq_len + token) * head_float4s + column
///     per token:   (batch * seq_len + token) * row_float4s + head * head_float4s + column
__device__ __forceinline__ size_t attention_merged_index(size_t index, size_t num_heads, size_t seq_len,
                                                         size_t head_float4s, size_t row_float4s) {
    size_t column = index % head_float4s;
    size_t token = (index / head_float4s) % seq_len;
    size_t head = (index / (head_float4s * seq_len)) % num_heads;
    size_t batch = index / (head_float4s * seq_len * num_heads);
    return (batch * seq_len + token) * row_float4s + head * head_float4s + column;
}

/// Gives the number of blocks of threads_per_block for `total` pieces of work, per_thread threads each:
/// 1 for the float4 kernels, 32 for the softmax kernels' one warp per row. Rounded up: a part-full last block.
inline int attention_blocks(size_t total, int threads_per_block, size_t per_thread) {
    size_t threads = total * per_thread;
    return (int)((threads + threads_per_block - 1) / threads_per_block);
}

/// Stops with a message if the tensors do not fit together, before a kernel reads past one: query, key,
/// value and head_outputs {batch_size, num_heads, seq_len, head_size}; weights {batch_size, num_heads,
/// seq_len, seq_len}; output {batch_size, seq_len, num_heads * head_size}; qkv the same with 3 times the
/// columns; all float or all bfloat16. Also head_size a multiple of 4 and every tensor starting on 4 values
/// (16 bytes for float, 8 for bfloat16), for the 4-value loads: off that, the GPU stops with "misaligned
/// address" at some later call, far from the cause. Backward passes output_grad, the gradients and qkv_grad in
/// the forward's places.
inline void attention_check_shapes(const Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
                                   const Tensor& weights, const Tensor& head_outputs, const Tensor& qkv) {
    bool fits = query.dim() == 4 && key.shape() == query.shape() && value.shape() == query.shape() &&
                head_outputs.shape() == query.shape();
    if (fits) {
        size_t batch_size = query.size(0), num_heads = query.size(1), seq_len = query.size(2);
        size_t embed_dim = num_heads * query.size(3);
        fits = weights.shape() == Shape({batch_size, num_heads, seq_len, seq_len}) &&
               output.shape() == Shape({batch_size, seq_len, embed_dim}) &&
               qkv.shape() == Shape({batch_size, seq_len, 3 * embed_dim});
    }
    if (!fits) {
        fprintf(stderr, "error: attention needs query, key, value and head_outputs {batch_size, num_heads, seq_len, "
                        "head_size}; weights {batch_size, num_heads, seq_len, seq_len}; output {batch_size, seq_len, "
                        "embed_dim} and qkv {batch_size, seq_len, 3 * embed_dim}, embed_dim = num_heads * head_size\n");
        exit(EXIT_FAILURE);
    }
    const Tensor* all[] = {&output, &query, &key, &value, &weights, &head_outputs, &qkv};
    DType dtype = qkv.dtype();
    bool same_dtype = dtype == DType::Float32 || dtype == DType::BFloat16;
    for (const Tensor* tensor : all) same_dtype = same_dtype && tensor->dtype() == dtype;
    if (!same_dtype) {
        fprintf(stderr, "error: attention needs its tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    size_t four_values = 4 * qkv.element_size();                        // 16 bytes for float, 8 for bfloat16
    bool aligned = query.size(3) % 4 == 0;
    for (const Tensor* tensor : all) aligned = aligned && (size_t)tensor->data_ptr() % four_values == 0;
    if (!aligned) {
        fprintf(stderr, "error: attention reads 4 values at a time, so head_size must be a multiple of 4 and every "
                        "tensor must start on %zu bytes; got head_size %zu\n", four_values, query.size(3));
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_CUH
