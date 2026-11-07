// encoder.cuh — GPT-2's first layer: each token id becomes its token's row plus its position's row:
//     forward:    encoded[b][t] = wte[token_ids[b][t]] + wpe[t]
//     backward:   wte_grad[token_ids[b][t]] += encoded_grad[b][t]      wpe_grad[t] += encoded_grad[b][t]
//
//     wte       {vocab_size, embed_dim}             {50304, 768}
//     wpe       {max_seq_len, embed_dim}            {1024, 768}
//     encoded   {batch_size, seq_len, embed_dim}    {4, 1024, 768}
// All float or all bfloat16 (T); the adds are in float.
#ifndef ENCODER_CUH
#define ENCODER_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block
#include "../tensor.cuh"

#define ENCODER_BLOCKS 512                                  // blocks in every launch here
#define ENCODER_THREADS MAX_THREADS_PER_BLOCK               // the default block size of the two backwards
#define ENCODER_FORWARD_THREADS (4 * THREADS_PER_WARP)      // the forward's default: 4 warps, fastest in bfloat16 on a T4

// ── forward and backward ──────────────────────────────────────────────────────
void encoder_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                     const Tensor& wpe_tensor, int threads_per_block_val = ENCODER_FORWARD_THREADS);
void encoder_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                      const Tensor& token_ids_tensor, int threads_per_block_val = ENCODER_THREADS);

// ── for both ──────────────────────────────────────────────────────────────────
void encoder_check_shapes(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                          const Tensor& wpe_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void encoder_forward_kernel(T* encoded_ptr, const int* token_ids_ptr, const T* wte_ptr, const T* wpe_ptr,
                                       size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val);

/// Writes encoded = wte[token_ids] + wpe: checks, then launches encoder_forward_kernel. The ids are not
/// checked: an id >= vocab_size reads past wte.
///
///   encoded_tensor          {batch_size, seq_len, embed_dim}: the result, the first block's input
///   token_ids_tensor        {batch_size, seq_len} ints
///   wte_tensor              {vocab_size, embed_dim}: one row per token
///   wpe_tensor              {max_seq_len, embed_dim}: one row per position; rows seq_len and after are not read
///   threads_per_block_val   the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           ENCODER_FORWARD_THREADS. Any size gives the same bits; only the time changes.
inline void encoder_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                            const Tensor& wpe_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the encoder");
    encoder_check_shapes(encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    // 2. launch: ENCODER_BLOCKS blocks, each thread walking encoded a float4 at a time
    size_t batch_size_val = encoded_tensor.size(0);
    size_t seq_len_val = encoded_tensor.size(1);
    size_t embed_dim_val = encoded_tensor.size(2);
    if (encoded_tensor.dtype() == DType::BFloat16)
        encoder_forward_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            encoded_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(), wte_tensor.data_ptr<nv_bfloat16>(),
            wpe_tensor.data_ptr<nv_bfloat16>(), batch_size_val, seq_len_val, embed_dim_val);
    else
        encoder_forward_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            encoded_tensor.data(), token_ids_tensor.data_ptr<int>(), wte_tensor.data(), wpe_tensor.data(),
            batch_size_val, seq_len_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Writes encoded one float4 at a time, from a float4 of a wte row and a float4 of a wpe row.
///
///   encoded_ptr, token_ids_ptr, wte_ptr, wpe_ptr      as in encoder_forward, float or bfloat16 (T)
///   batch_size_val, seq_len_val, embed_dim_val        encoded's shape
template <typename T>
__global__ void encoder_forward_kernel(T* encoded_ptr, const int* token_ids_ptr, const T* wte_ptr, const T* wpe_ptr,
                                       size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_float4s_val = embed_dim_val / VALUES_PER_FLOAT4;                // float4s in one row: 768 / 4 = 192
    size_t total_float4s_val = batch_size_val * seq_len_val * row_float4s_val;  // all of encoded
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where this float4 is: which token (b x seq_len + t), which float4 of its row, which position t
        size_t token_index_val = index_val / row_float4s_val;
        size_t row_float4_index_val = index_val % row_float4s_val;
        size_t position_val = token_index_val % seq_len_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. read the token's and the position's float4
        float4 token_embed_val = read4_as_float(wte_ptr + (token_id_val * row_float4s_val + row_float4_index_val) * VALUES_PER_FLOAT4);
        float4 position_embed_val = read4_as_float(wpe_ptr + (position_val * row_float4s_val + row_float4_index_val) * VALUES_PER_FLOAT4);
        // 3. add and write
        write4_from_float(encoded_ptr + index_val * VALUES_PER_FLOAT4,
               make_float4(token_embed_val.x + position_embed_val.x, token_embed_val.y + position_embed_val.y,
                           token_embed_val.z + position_embed_val.z, token_embed_val.w + position_embed_val.w));
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void encoder_backward_wte_kernel(T* wte_grad_ptr, const T* encoded_grad_ptr, const int* token_ids_ptr,
                                            size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val);
template <typename T>
__global__ void encoder_backward_wpe_kernel(T* wpe_grad_ptr, const T* encoded_grad_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t embed_dim_val);

/// Adds encoded_grad back onto the rows the forward read: checks, then launches the wte and wpe kernels.
/// It adds, so the gradients must be 0 before a step's first backward.
///
///   wte_grad_tensor         wte's shape
///   wpe_grad_tensor         wpe's shape
///   encoded_grad_tensor     encoded's shape: the gradient from the first block
///   token_ids_tensor        the forward's
///   threads_per_block_val   the block size of both kernels: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           ENCODER_THREADS. wpe_grad's bits never change with it; wte_grad's atomic adds can
///                           land in another order, as they can from run to run.
inline void encoder_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                             const Tensor& token_ids_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the encoder");
    encoder_check_shapes(encoded_grad_tensor, token_ids_tensor, wte_grad_tensor, wpe_grad_tensor);
    // 2. launch the wte kernel, then the wpe kernel: ENCODER_BLOCKS blocks each
    size_t batch_size_val = encoded_grad_tensor.size(0);
    size_t seq_len_val = encoded_grad_tensor.size(1);
    size_t embed_dim_val = encoded_grad_tensor.size(2);
    if (encoded_grad_tensor.dtype() == DType::BFloat16) {
        encoder_backward_wte_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            wte_grad_tensor.data_ptr<nv_bfloat16>(), encoded_grad_tensor.data_ptr<nv_bfloat16>(),
            token_ids_tensor.data_ptr<int>(), batch_size_val, seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
        encoder_backward_wpe_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            wpe_grad_tensor.data_ptr<nv_bfloat16>(), encoded_grad_tensor.data_ptr<nv_bfloat16>(), batch_size_val,
            seq_len_val, embed_dim_val);
    } else {
        encoder_backward_wte_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            wte_grad_tensor.data(), encoded_grad_tensor.data(), token_ids_tensor.data_ptr<int>(), batch_size_val,
            seq_len_val, embed_dim_val);
        cudaCheck(cudaGetLastError());
        encoder_backward_wpe_kernel<<<ENCODER_BLOCKS, threads_per_block_val>>>(
            wpe_grad_tensor.data(), encoded_grad_tensor.data(), batch_size_val, seq_len_val, embed_dim_val);
    }
    cudaCheck(cudaGetLastError());
}

/// Adds encoded_grad onto wte_grad one float4 at a time, with atomic adds: a token seen at several
/// positions has several threads adding onto its row at once. The adds come in any order, so the last
/// bits can differ from run to run.
///
///   wte_grad_ptr, encoded_grad_ptr, token_ids_ptr   as in encoder_backward, float or bfloat16 (T)
///   batch_size_val, seq_len_val, embed_dim_val      encoded_grad's shape
template <typename T>
__global__ void encoder_backward_wte_kernel(T* wte_grad_ptr, const T* encoded_grad_ptr, const int* token_ids_ptr,
                                            size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_float4s_val = embed_dim_val / VALUES_PER_FLOAT4;                // float4s in one row: 768 / 4 = 192
    size_t total_float4s_val = batch_size_val * seq_len_val * row_float4s_val;  // all of encoded_grad
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where this float4 is: which token, which float4 of its row, which wte_grad row to add onto
        size_t token_index_val = index_val / row_float4s_val;
        size_t row_float4_index_val = index_val % row_float4s_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. read the grad, add it onto the token's row
        float4 grad_val = read4_as_float(encoded_grad_ptr + index_val * VALUES_PER_FLOAT4);
        atomic_add4_from_float(wte_grad_ptr + token_id_val * embed_dim_val + row_float4_index_val * VALUES_PER_FLOAT4,
                               grad_val);
    }
}

/// Adds the batch_size rows of encoded_grad at each position t onto row t of wpe_grad, one float4 at a
/// time. One thread per float4 of wpe_grad, adding b = 0, 1, ... in order: no atomics, the same result
/// every run.
///
///   wpe_grad_ptr, encoded_grad_ptr                  as in encoder_backward, float or bfloat16 (T)
///   batch_size_val, seq_len_val, embed_dim_val      encoded_grad's shape
template <typename T>
__global__ void encoder_backward_wpe_kernel(T* wpe_grad_ptr, const T* encoded_grad_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_float4s_val = embed_dim_val / VALUES_PER_FLOAT4;    // float4s in one row: 768 / 4 = 192
    size_t total_float4s_val = seq_len_val * row_float4s_val;      // wpe_grad's first seq_len rows
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. add up this float4 over the batch, b = 0, 1, ... in order
        float4 sum_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (size_t b_val = 0; b_val < batch_size_val; b_val++) {
            // row (b, t)
            float4 g_val = read4_as_float(encoded_grad_ptr + (b_val * seq_len_val * row_float4s_val + index_val) * VALUES_PER_FLOAT4);
            sum_val.x += g_val.x;
            sum_val.y += g_val.y;
            sum_val.z += g_val.z;
            sum_val.w += g_val.w;
        }
        // 2. add the sum onto wpe_grad
        float4 old_val = read4_as_float(wpe_grad_ptr + index_val * VALUES_PER_FLOAT4);
        write4_from_float(wpe_grad_ptr + index_val * VALUES_PER_FLOAT4,
                          make_float4(old_val.x + sum_val.x, old_val.y + sum_val.y, old_val.z + sum_val.z,
                                      old_val.w + sum_val.w));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit: the shapes in encoder_forward, seq_len <= max_seq_len,
/// encoded, wte and wpe all float or all bfloat16, embed_dim a multiple of VALUES_PER_FLOAT4, each starting on
/// a whole float4.
///
///   encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor   forward: these; backward: their gradients in their place
inline void encoder_check_shapes(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                                 const Tensor& wpe_tensor) {
    // shapes: as in encoder_forward
    bool fits_val = encoded_tensor.dim() == 3 && token_ids_tensor.dim() == 2 && wte_tensor.dim() == 2 &&
                    wpe_tensor.dim() == 2 && token_ids_tensor.dtype() == DType::Int32 &&
                    token_ids_tensor.size(0) == encoded_tensor.size(0) && token_ids_tensor.size(1) == encoded_tensor.size(1) &&
                    wte_tensor.size(1) == encoded_tensor.size(2) && wpe_tensor.size(1) == encoded_tensor.size(2) &&
                    encoded_tensor.size(1) <= wpe_tensor.size(0);
    if (!fits_val) {
        fprintf(stderr, "error: the encoder needs encoded {batch_size, seq_len, embed_dim}, token_ids {batch_size, "
                        "seq_len} ints, wte {vocab_size, embed_dim}, wpe {max_seq_len, embed_dim} with seq_len <= "
                        "max_seq_len\n");
        exit(EXIT_FAILURE);
    }
    // types: encoded, wte and wpe all float or all bfloat16
    DType dtype_val = encoded_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || wte_tensor.dtype() != dtype_val ||
        wpe_tensor.dtype() != dtype_val) {
        fprintf(stderr, "error: the encoder needs encoded, wte and wpe all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: rows of whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * encoded_tensor.element_size();
    if (encoded_tensor.size(2) % VALUES_PER_FLOAT4 != 0 || (size_t)encoded_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)wte_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)wpe_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: the encoder reads %d values at a time, so embed_dim must be a multiple of %d and "
                        "encoded, wte, wpe must start on %zu bytes; got embed_dim %zu, %p, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, encoded_tensor.size(2), encoded_tensor.data_ptr(),
                wte_tensor.data_ptr(), wpe_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // ENCODER_CUH
