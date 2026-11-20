// qk_norm_8bit.cuh — QK-norm for the 8-bit model: each head row of query and key scaled to a fixed root mean square
// before attention.
//
//     tensor                    type in memory   shape
//     qkv, qkv_grad             bfloat16         {batch_size, seq_len, 3 · embed_dim}
//     qkv_rounded               bfloat16         {batch_size, seq_len, 3 · embed_dim}
//     query_rstd, key_rstd      float            {batch_size, num_heads, seq_len}
#ifndef QK_NORM_8BIT_CUH
#define QK_NORM_8BIT_CUH

#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum, CEIL_DIV
#include "../../common/tensor.cuh"      // read4_as_float, write4_from_float, VALUES_PER_FLOAT4

#define QK_NORM_8BIT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)   // the default block size: 8 warps, 8 head rows (not swept yet)
#define QK_NORM_8BIT_GAIN 1.2f            // y's root mean square: nanochat's (q, k = norm(q) x 1.2, norm(k) x 1.2)
#define QK_NORM_8BIT_EPS 1e-6f            // added to mean(x²): keeps a row of 0s from dividing by 0
#define QK_NORM_8BIT_PARTS 2              // the head rows' two kinds: query (0), key (1)

// ── forward and backward ──────────────────────────────────────────────────────
void qk_norm_8bit_forward(const Tensor& qkv_tensor, const Tensor& query_rstd_tensor, const Tensor& key_rstd_tensor,
                          int threads_per_block_val = QK_NORM_8BIT_THREADS_PER_BLOCK);
void qk_norm_8bit_backward(const Tensor& qkv_grad_tensor, const Tensor& qkv_rounded_tensor, const Tensor& query_rstd_tensor,
                           const Tensor& key_rstd_tensor, int threads_per_block_val = QK_NORM_8BIT_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
struct QkNorm8bitRow;
__device__ __forceinline__ QkNorm8bitRow qk_norm_8bit_row_of(size_t warp_row_val, size_t batch_size_val, size_t seq_len_val,
                                                             size_t num_heads_val, size_t head_size_val);
__global__ void qk_norm_8bit_forward_kernel(nv_bfloat16* qkv_ptr, float* query_rstd_ptr, float* key_rstd_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t num_heads_val, size_t head_size_val);
__global__ void qk_norm_8bit_backward_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* qkv_rounded_ptr,
                                             const float* query_rstd_ptr, const float* key_rstd_ptr, size_t batch_size_val,
                                             size_t seq_len_val, size_t num_heads_val, size_t head_size_val);
unsigned qk_norm_8bit_blocks(const Tensor& qkv_tensor, size_t num_heads_val, int threads_per_block_val);
void qk_norm_8bit_check_shapes(const Tensor& qkv_tensor, const Tensor& other_qkv_tensor, const Tensor& query_rstd_tensor,
                               const Tensor& key_rstd_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward and backward ──────────────────────────────────────────────────────

/// Scales each head row of query and key in qkv to a root mean square of QK_NORM_8BIT_GAIN, in place.
///
///   qkv_tensor              {batch_size, seq_len, 3 · embed_dim} bfloat16: the qkv linear's output; query and key written over
///   query_rstd_tensor,
///   key_rstd_tensor         {batch_size, num_heads, seq_len} float: written, 1 / sqrt(mean(x²) + eps) of each row
///   threads_per_block_val   whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given: QK_NORM_8BIT_THREADS_PER_BLOCK.
///                           Any size gives the same bits
inline void qk_norm_8bit_forward(const Tensor& qkv_tensor, const Tensor& query_rstd_tensor, const Tensor& key_rstd_tensor,
                                 int threads_per_block_val) {
    // Keeping each row's rstd for the backward. Value's rows are left as they are.
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "qk_norm_8bit_forward");
    qk_norm_8bit_check_shapes(qkv_tensor, qkv_tensor, query_rstd_tensor, key_rstd_tensor);
    // 2. launch: one warp a head row
    size_t num_heads_val = query_rstd_tensor.size(1);
    qk_norm_8bit_forward_kernel<<<qk_norm_8bit_blocks(qkv_tensor, num_heads_val, threads_per_block_val), threads_per_block_val>>>(
        qkv_tensor.data_ptr<nv_bfloat16>(), query_rstd_tensor.data(), key_rstd_tensor.data(), qkv_tensor.size(0),
        qkv_tensor.size(1), num_heads_val, qkv_tensor.size(2) / 3 / num_heads_val);
    cudaCheck(cudaGetLastError());
}

/// Turns the gradient of each query and key head row after the norm into the gradient before it, in place.
///
///   qkv_grad_tensor         {batch_size, seq_len, 3 · embed_dim} bfloat16: query's and key's parts written over
///   qkv_rounded_tensor      {batch_size, seq_len, 3 · embed_dim} bfloat16: query and key after the norm (y), as
///                           attention_8bit_backward turned them back from 8 bits
///   query_rstd_tensor,
///   key_rstd_tensor         {batch_size, num_heads, seq_len} float: the forward's
///   threads_per_block_val   as in qk_norm_8bit_forward
inline void qk_norm_8bit_backward(const Tensor& qkv_grad_tensor, const Tensor& qkv_rounded_tensor, const Tensor& query_rstd_tensor,
                                  const Tensor& key_rstd_tensor, int threads_per_block_val) {
    // Turns the gradient of each query and key head row after the norm (dy, as attention's backward wrote it into
    // qkv_grad) into the gradient before it (dx), in place. Value's rows are left as they are.
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "qk_norm_8bit_backward");
    qk_norm_8bit_check_shapes(qkv_grad_tensor, qkv_rounded_tensor, query_rstd_tensor, key_rstd_tensor);
    // 2. launch: one warp a head row
    size_t num_heads_val = query_rstd_tensor.size(1);
    qk_norm_8bit_backward_kernel<<<qk_norm_8bit_blocks(qkv_grad_tensor, num_heads_val, threads_per_block_val), threads_per_block_val>>>(
        qkv_grad_tensor.data_ptr<nv_bfloat16>(), qkv_rounded_tensor.data_ptr<nv_bfloat16>(), query_rstd_tensor.data(),
        key_rstd_tensor.data(), qkv_grad_tensor.size(0), qkv_grad_tensor.size(1), num_heads_val,
        qkv_grad_tensor.size(2) / 3 / num_heads_val);
    cudaCheck(cudaGetLastError());
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Where one warp's head row is: its first value in qkv (or qkv_grad, qkv_rounded) and its rstd's place.
struct QkNorm8bitRow {
    size_t first_value_index_val;   // in a {batch_size, seq_len, 3 · embed_dim} tensor
    size_t rstd_index_val;          // in a {batch_size, num_heads, seq_len} tensor
    bool key_val;                   // false: query's row, its rstd in query_rstd; true: key's, in key_rstd
};

/// Gives warp row warp_row_val's place.
///
///   warp_row_val                                          0 .. 2 x batch_size x seq_len x num_heads - 1
///   batch_size_val, seq_len_val, num_heads_val, head_size_val   the sizes
__device__ __forceinline__ QkNorm8bitRow qk_norm_8bit_row_of(size_t warp_row_val, size_t batch_size_val, size_t seq_len_val,
                                                             size_t num_heads_val, size_t head_size_val) {
    // The rows go token by token, query's heads then key's, so neighbouring warps read neighbouring memory:
    // warp_row = ((batch · seq_len + token) · 2 + part) · num_heads + head.
    size_t head_val = warp_row_val % num_heads_val;
    size_t part_val = warp_row_val / num_heads_val % QK_NORM_8BIT_PARTS;
    size_t batch_token_val = warp_row_val / num_heads_val / QK_NORM_8BIT_PARTS;    // batch · seq_len + token
    size_t batch_val = batch_token_val / seq_len_val;
    size_t token_val = batch_token_val % seq_len_val;
    size_t embed_dim_val = num_heads_val * head_size_val;
    QkNorm8bitRow row_val;
    row_val.first_value_index_val = batch_token_val * 3 * embed_dim_val + part_val * embed_dim_val + head_val * head_size_val;
    row_val.rstd_index_val = (batch_val * num_heads_val + head_val) * seq_len_val + token_val;
    row_val.key_val = part_val == 1;
    return row_val;
}

/// One warp a head row: 1. its sum of squares, 2. rstd, kept by lane 0, 3. each value times gain · rstd, written over.
///
///   qkv_ptr                       the qkv linear's output: query's and key's rows written over
///   query_rstd_ptr, key_rstd_ptr  one a head row: written
///   batch_size_val ... head_size_val   the sizes; head_size a multiple of VALUES_PER_FLOAT4
__global__ void qk_norm_8bit_forward_kernel(nv_bfloat16* qkv_ptr, float* query_rstd_ptr, float* key_rstd_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t num_heads_val, size_t head_size_val) {
    size_t lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_row_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;
    if (warp_row_val >= QK_NORM_8BIT_PARTS * batch_size_val * seq_len_val * num_heads_val) return;   // the whole warp
    QkNorm8bitRow row_val = qk_norm_8bit_row_of(warp_row_val, batch_size_val, seq_len_val, num_heads_val, head_size_val);
    nv_bfloat16* row_ptr = qkv_ptr + row_val.first_value_index_val;
    // 1. the row's sum of squares: each lane its float4s, then the warp's
    float thread_sum_val = 0.0f;
    for (size_t column_val = lane_index_val * VALUES_PER_FLOAT4; column_val < head_size_val;
         column_val += HARDWARE_THREADS_PER_WARP * VALUES_PER_FLOAT4) {
        float4 x_val = read4_as_float(row_ptr + column_val);
        thread_sum_val += x_val.x * x_val.x + x_val.y * x_val.y + x_val.z * x_val.z + x_val.w * x_val.w;
    }
    float row_sum_val = warp_reduce_sum(thread_sum_val);
    // 2. rstd = 1 / sqrt(mean(x²) + eps), kept for the backward
    float rstd_val = rsqrtf(row_sum_val / (float)head_size_val + QK_NORM_8BIT_EPS);
    if (lane_index_val == 0) (row_val.key_val ? key_rstd_ptr : query_rstd_ptr)[row_val.rstd_index_val] = rstd_val;
    // 3. y = gain · x · rstd, written over x (each lane its own float4s, read again: one row, still in L1)
    float factor_val = QK_NORM_8BIT_GAIN * rstd_val;
    for (size_t column_val = lane_index_val * VALUES_PER_FLOAT4; column_val < head_size_val;
         column_val += HARDWARE_THREADS_PER_WARP * VALUES_PER_FLOAT4) {
        float4 x_val = read4_as_float(row_ptr + column_val);
        write4_from_float(row_ptr + column_val,
                          make_float4(x_val.x * factor_val, x_val.y * factor_val, x_val.z * factor_val, x_val.w * factor_val));
    }
}

/// One warp a head row: 1. mean(dy · n), n = y / gain, 2. dx = gain · rstd · (dy − n · mean(dy · n)), written over dy.
///
///   qkv_grad_ptr                  dy in, dx out: query's and key's rows
///   qkv_rounded_ptr               y: query and key after the norm, turned back from 8 bits
///   query_rstd_ptr, key_rstd_ptr  the forward's
///   batch_size_val ... head_size_val   the sizes; head_size a multiple of VALUES_PER_FLOAT4
__global__ void qk_norm_8bit_backward_kernel(nv_bfloat16* qkv_grad_ptr, const nv_bfloat16* qkv_rounded_ptr,
                                             const float* query_rstd_ptr, const float* key_rstd_ptr, size_t batch_size_val,
                                             size_t seq_len_val, size_t num_heads_val, size_t head_size_val) {
    size_t lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_row_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;
    if (warp_row_val >= QK_NORM_8BIT_PARTS * batch_size_val * seq_len_val * num_heads_val) return;   // the whole warp
    QkNorm8bitRow row_val = qk_norm_8bit_row_of(warp_row_val, batch_size_val, seq_len_val, num_heads_val, head_size_val);
    nv_bfloat16* grad_row_ptr = qkv_grad_ptr + row_val.first_value_index_val;
    const nv_bfloat16* y_row_ptr = qkv_rounded_ptr + row_val.first_value_index_val;
    float n_factor_val = 1.0f / QK_NORM_8BIT_GAIN;                                  // n = y / gain
    // 1. mean(dy · n): each lane its float4s, then the warp's
    float thread_sum_val = 0.0f;
    for (size_t column_val = lane_index_val * VALUES_PER_FLOAT4; column_val < head_size_val;
         column_val += HARDWARE_THREADS_PER_WARP * VALUES_PER_FLOAT4) {
        float4 dy_val = read4_as_float(grad_row_ptr + column_val);
        float4 y_val = read4_as_float(y_row_ptr + column_val);
        thread_sum_val += dy_val.x * y_val.x + dy_val.y * y_val.y + dy_val.z * y_val.z + dy_val.w * y_val.w;
    }
    float mean_val = warp_reduce_sum(thread_sum_val) * n_factor_val / (float)head_size_val;
    // 2. dx = gain · rstd · (dy − n · mean), written over dy
    float rstd_val = (row_val.key_val ? key_rstd_ptr : query_rstd_ptr)[row_val.rstd_index_val];
    float factor_val = QK_NORM_8BIT_GAIN * rstd_val;
    float n_mean_val = n_factor_val * mean_val;                                     // n · mean = y · (mean / gain)
    for (size_t column_val = lane_index_val * VALUES_PER_FLOAT4; column_val < head_size_val;
         column_val += HARDWARE_THREADS_PER_WARP * VALUES_PER_FLOAT4) {
        float4 dy_val = read4_as_float(grad_row_ptr + column_val);
        float4 y_val = read4_as_float(y_row_ptr + column_val);
        write4_from_float(grad_row_ptr + column_val,
                          make_float4(factor_val * (dy_val.x - y_val.x * n_mean_val), factor_val * (dy_val.y - y_val.y * n_mean_val),
                                      factor_val * (dy_val.z - y_val.z * n_mean_val), factor_val * (dy_val.w - y_val.w * n_mean_val)));
    }
}

/// Gives the blocks for one warp a head row: 2 x batch_size x seq_len x num_heads warps, rounded up to whole blocks.
///
///   qkv_tensor              {batch_size, seq_len, 3 · embed_dim}
///   num_heads_val           heads
///   threads_per_block_val   the block size
inline unsigned qk_norm_8bit_blocks(const Tensor& qkv_tensor, size_t num_heads_val, int threads_per_block_val) {
    size_t warp_rows_val = QK_NORM_8BIT_PARTS * qkv_tensor.size(0) * qkv_tensor.size(1) * num_heads_val;
    size_t warps_per_block_val = (size_t)threads_per_block_val / HARDWARE_THREADS_PER_WARP;
    return (unsigned)CEIL_DIV(warp_rows_val, warps_per_block_val);
}

/// Stops with a message unless the tensors fit.
///
///   qkv_tensor          forward: qkv; backward: qkv_grad
///   other_qkv_tensor    forward: qkv again; backward: qkv_rounded
///   query_rstd_tensor, key_rstd_tensor   the rstds
inline void qk_norm_8bit_check_shapes(const Tensor& qkv_tensor, const Tensor& other_qkv_tensor, const Tensor& query_rstd_tensor,
                                      const Tensor& key_rstd_tensor) {
    // Qkv-shaped bfloat16 {batch_size, seq_len, 3 · embed_dim} (both the same shape), the rstds float {batch_size,
    // num_heads, seq_len}, embed_dim a multiple of num_heads and head_size of VALUES_PER_FLOAT4, and the qkv
    // tensors starting on an 8-byte load.
    size_t load_bytes_val = VALUES_PER_FLOAT4 * sizeof(nv_bfloat16);
    bool fits_val = qkv_tensor.dim() == 3 && qkv_tensor.dtype() == DType::BFloat16 && other_qkv_tensor.shape() == qkv_tensor.shape() &&
                    other_qkv_tensor.dtype() == DType::BFloat16 && query_rstd_tensor.dim() == 3 &&
                    query_rstd_tensor.dtype() == DType::Float32 && key_rstd_tensor.shape() == query_rstd_tensor.shape() &&
                    key_rstd_tensor.dtype() == DType::Float32 && qkv_tensor.size(2) % 3 == 0 &&
                    (size_t)qkv_tensor.data_ptr() % load_bytes_val == 0 && (size_t)other_qkv_tensor.data_ptr() % load_bytes_val == 0;
    if (fits_val) {
        size_t num_heads_val = query_rstd_tensor.size(1);
        size_t embed_dim_val = qkv_tensor.size(2) / 3;
        fits_val = num_heads_val > 0 && query_rstd_tensor.size(0) == qkv_tensor.size(0) && query_rstd_tensor.size(2) == qkv_tensor.size(1) &&
                   embed_dim_val % num_heads_val == 0 && (embed_dim_val / num_heads_val) % VALUES_PER_FLOAT4 == 0;
    }
    if (!fits_val) {
        fprintf(stderr, "error: qk_norm_8bit needs bfloat16 qkv (and qkv_grad, qkv_rounded) {batch_size, seq_len, 3 · embed_dim} "
                        "starting on %zu bytes, float query_rstd and key_rstd {batch_size, num_heads, seq_len}, and head_size = "
                        "embed_dim / num_heads a multiple of %d\n",
                load_bytes_val, VALUES_PER_FLOAT4);
        exit(EXIT_FAILURE);
    }
}

#endif // QK_NORM_8BIT_CUH
