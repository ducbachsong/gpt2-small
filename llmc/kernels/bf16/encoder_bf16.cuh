// encoder_bf16.cuh — GPT-2's first layer in bfloat16, 8 values a thread (one 16-byte load): fp32/encoder.cuh's
// math, whose bfloat16 path takes 4 values a thread (one 8-byte load):
//     forward:    encoded[b][t] = wte[token_ids[b][t]] + wpe[t]
//     backward:   wte_grad[token_ids[b][t]] += encoded_grad[b][t]      wpe_grad[t] += encoded_grad[b][t]
//
//     tensor                    type in memory   shape                               GPT-2 small
//     wte, wte_grad             bfloat16         {vocab_size, embed_dim}             {50304, 768}
//     wpe, wpe_grad             bfloat16         {max_seq_len, embed_dim}            {1024, 768}
//     encoded, encoded_grad     bfloat16         {batch_size, seq_len, embed_dim}    {4, 1024, 768}
//     token_ids                 int              {batch_size, seq_len}               {4, 1024}
//     in registers              float            -                                   every add
//
// The same bits as fp32/encoder.cuh on bfloat16: the same adds in float, each value rounded once as it is stored
// (wte_grad: after each atomic add onto a repeated token's row, as there). Only the loads change: a thread asks for
// 16 bytes at a time instead of 8, so it keeps twice the bytes in flight and the same bytes take half the loads.
//
//     GPT-2 small, one forward:   loads per row   loads in all   bytes a load
//     fp32/encoder.cuh, bf16      768 / 4 = 192   786,432        8
//     here                        768 / 8 = 96    393,216        16
//
// Each thread reads what it needs straight from GPU memory, no shared memory and no TMA: each value is read once,
// added once and written once, so a copy into shared memory would add a step and save nothing. Any GPU.
#ifndef ENCODER_BF16_CUH
#define ENCODER_BF16_CUH

#include "../cuda_common.h"          // cudaCheck, check_threads_per_block
#include "../../common/tensor.cuh"

#define ENCODER_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD            // bfloat16s in one 16-byte load (read8_as_float): 8
#define ENCODER_BF16_BLOCKS 512                                      // blocks in every launch here: fp32/encoder.cuh's, not swept for 8 values yet
#define ENCODER_BF16_THREADS HARDWARE_MAX_THREADS_PER_BLOCK          // the default block size of the two backwards, as fp32/encoder.cuh's
#define ENCODER_BF16_FORWARD_THREADS (8 * HARDWARE_THREADS_PER_WARP) // the forward's default: 8 warps, as fp32/encoder.cuh's
static_assert(ENCODER_BF16_VALUES_PER_LOAD == 2 * VALUES_PER_FLOAT4, "the wte backward adds a load as two atomic_add4_from_float");

// ── forward and backward ──────────────────────────────────────────────────────
void encoder_bf16_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                          const Tensor& wpe_tensor, int threads_per_block_val = ENCODER_BF16_FORWARD_THREADS);
void encoder_bf16_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                           const Tensor& token_ids_tensor, int threads_per_block_val = ENCODER_BF16_THREADS);

// ── for both ──────────────────────────────────────────────────────────────────
void encoder_bf16_check_shapes(const char* caller_ptr, const Tensor& encoded_tensor, const Tensor& token_ids_tensor,
                               const Tensor& wte_tensor, const Tensor& wpe_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void encoder_bf16_forward_kernel(nv_bfloat16* encoded_ptr, const int* token_ids_ptr, const nv_bfloat16* wte_ptr,
                                            const nv_bfloat16* wpe_ptr, size_t batch_size_val, size_t seq_len_val,
                                            size_t embed_dim_val);

/// Writes encoded = wte[token_ids] + wpe in bfloat16: checks, then launches encoder_bf16_forward_kernel. The ids are
/// not checked: an id >= vocab_size reads past wte.
///
///   encoded_tensor          {batch_size, seq_len, embed_dim} bfloat16: the result, the first block's input
///   token_ids_tensor        {batch_size, seq_len} ints
///   wte_tensor              {vocab_size, embed_dim} bfloat16: one row per token
///   wpe_tensor              {max_seq_len, embed_dim} bfloat16: one row per position; rows seq_len and after are not read
///   threads_per_block_val   the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                           ENCODER_BF16_FORWARD_THREADS. Any size gives the same bits; only the time changes.
inline void encoder_bf16_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                                 const Tensor& wpe_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "encoder_bf16_forward");
    encoder_bf16_check_shapes("encoder_bf16_forward", encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    // 2. launch: ENCODER_BF16_BLOCKS blocks, each thread walking encoded 8 values at a time
    encoder_bf16_forward_kernel<<<ENCODER_BF16_BLOCKS, threads_per_block_val>>>(
        encoded_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(), wte_tensor.data_ptr<nv_bfloat16>(),
        wpe_tensor.data_ptr<nv_bfloat16>(), encoded_tensor.size(0), encoded_tensor.size(1), encoded_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Writes encoded 8 values (one 16-byte load) at a time, from 8 values of a wte row and 8 of a wpe row, added in
/// float and rounded once.
///
///   encoded_ptr, token_ids_ptr, wte_ptr, wpe_ptr      as in encoder_bf16_forward
///   batch_size_val, seq_len_val, embed_dim_val        encoded's shape
__global__ void encoder_bf16_forward_kernel(nv_bfloat16* encoded_ptr, const int* token_ids_ptr, const nv_bfloat16* wte_ptr,
                                            const nv_bfloat16* wpe_ptr, size_t batch_size_val, size_t seq_len_val,
                                            size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_loads_val = embed_dim_val / ENCODER_BF16_VALUES_PER_LOAD;      // loads in one row: 768 / 8 = 96
    size_t total_loads_val = batch_size_val * seq_len_val * row_loads_val;    // all of encoded
    // Thread n takes loads n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where this load is: which token (b x seq_len + t), which load of its row, which position t
        size_t token_index_val = index_val / row_loads_val;
        size_t row_load_index_val = index_val % row_loads_val;
        size_t position_index_val = token_index_val % seq_len_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. read the token's and the position's 8 values
        float token_embed_list[ENCODER_BF16_VALUES_PER_LOAD];
        float position_embed_list[ENCODER_BF16_VALUES_PER_LOAD];
        read8_as_float(token_embed_list, wte_ptr + (token_id_val * row_loads_val + row_load_index_val) * ENCODER_BF16_VALUES_PER_LOAD);
        read8_as_float(position_embed_list, wpe_ptr + (position_index_val * row_loads_val + row_load_index_val) * ENCODER_BF16_VALUES_PER_LOAD);
        // 3. add and write
#pragma unroll
        for (int value_index_val = 0; value_index_val < ENCODER_BF16_VALUES_PER_LOAD; value_index_val++)
            token_embed_list[value_index_val] += position_embed_list[value_index_val];
        write8_from_float(encoded_ptr + index_val * ENCODER_BF16_VALUES_PER_LOAD, token_embed_list);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

__global__ void encoder_bf16_backward_wte_kernel(nv_bfloat16* wte_grad_ptr, const nv_bfloat16* encoded_grad_ptr,
                                                 const int* token_ids_ptr, size_t batch_size_val, size_t seq_len_val,
                                                 size_t embed_dim_val);
__global__ void encoder_bf16_backward_wpe_kernel(nv_bfloat16* wpe_grad_ptr, const nv_bfloat16* encoded_grad_ptr,
                                                 size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val);

/// Adds encoded_grad back onto the rows the forward read, in bfloat16: checks, then launches the wte and wpe
/// kernels. It adds, so the gradients must be 0 before a step's first backward.
///
///   wte_grad_tensor         wte's shape, bfloat16
///   wpe_grad_tensor         wpe's shape, bfloat16
///   encoded_grad_tensor     encoded's shape, bfloat16: the gradient from the first block
///   token_ids_tensor        the forward's
///   threads_per_block_val   the block size of both kernels: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                           given: ENCODER_BF16_THREADS. wpe_grad's bits never change with it; wte_grad's atomic adds
///                           can land in another order, as they can from run to run.
inline void encoder_bf16_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                                  const Tensor& token_ids_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "encoder_bf16_backward");
    encoder_bf16_check_shapes("encoder_bf16_backward", encoded_grad_tensor, token_ids_tensor, wte_grad_tensor, wpe_grad_tensor);
    // 2. launch the wte kernel, then the wpe kernel: ENCODER_BF16_BLOCKS blocks each
    size_t batch_size_val = encoded_grad_tensor.size(0);
    size_t seq_len_val = encoded_grad_tensor.size(1);
    size_t embed_dim_val = encoded_grad_tensor.size(2);
    encoder_bf16_backward_wte_kernel<<<ENCODER_BF16_BLOCKS, threads_per_block_val>>>(
        wte_grad_tensor.data_ptr<nv_bfloat16>(), encoded_grad_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(),
        batch_size_val, seq_len_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
    encoder_bf16_backward_wpe_kernel<<<ENCODER_BF16_BLOCKS, threads_per_block_val>>>(
        wpe_grad_tensor.data_ptr<nv_bfloat16>(), encoded_grad_tensor.data_ptr<nv_bfloat16>(), batch_size_val, seq_len_val,
        embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Adds encoded_grad onto wte_grad 8 values at a time: one 16-byte load, then two atomic_add4_from_float (an 8-byte
/// compare-and-swap each, so it runs on a T4 too, which has no bfloat16 atomicAdd). A token seen at several positions
/// has several threads adding onto its row at once; the adds come in any order, so the last bits can differ from run
/// to run.
///
///   wte_grad_ptr, encoded_grad_ptr, token_ids_ptr   as in encoder_bf16_backward
///   batch_size_val, seq_len_val, embed_dim_val      encoded_grad's shape
__global__ void encoder_bf16_backward_wte_kernel(nv_bfloat16* wte_grad_ptr, const nv_bfloat16* encoded_grad_ptr,
                                                 const int* token_ids_ptr, size_t batch_size_val, size_t seq_len_val,
                                                 size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_loads_val = embed_dim_val / ENCODER_BF16_VALUES_PER_LOAD;      // loads in one row: 768 / 8 = 96
    size_t total_loads_val = batch_size_val * seq_len_val * row_loads_val;    // all of encoded_grad
    // Thread n takes loads n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where this load is: which token, which load of its row, which wte_grad row to add onto
        size_t token_index_val = index_val / row_loads_val;
        size_t row_load_index_val = index_val % row_loads_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. read the grad, add it onto the token's row: the first 4 values, then the other 4
        float grad_list[ENCODER_BF16_VALUES_PER_LOAD];
        read8_as_float(grad_list, encoded_grad_ptr + index_val * ENCODER_BF16_VALUES_PER_LOAD);
        nv_bfloat16* row_grad_ptr = wte_grad_ptr + token_id_val * embed_dim_val + row_load_index_val * ENCODER_BF16_VALUES_PER_LOAD;
        atomic_add4_from_float(row_grad_ptr, make_float4(grad_list[0], grad_list[1], grad_list[2], grad_list[3]));
        atomic_add4_from_float(row_grad_ptr + VALUES_PER_FLOAT4, make_float4(grad_list[4], grad_list[5], grad_list[6], grad_list[7]));
    }
}

/// Adds the batch_size rows of encoded_grad at each position t onto row t of wpe_grad, 8 values at a time. One
/// thread per load of wpe_grad, adding b = 0, 1, ... in order in float and rounding once: no atomics, the same
/// result every run.
///
///   wpe_grad_ptr, encoded_grad_ptr                  as in encoder_bf16_backward
///   batch_size_val, seq_len_val, embed_dim_val      encoded_grad's shape
__global__ void encoder_bf16_backward_wpe_kernel(nv_bfloat16* wpe_grad_ptr, const nv_bfloat16* encoded_grad_ptr,
                                                 size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_loads_val = embed_dim_val / ENCODER_BF16_VALUES_PER_LOAD;      // loads in one row: 768 / 8 = 96
    size_t total_loads_val = seq_len_val * row_loads_val;                     // wpe_grad's first seq_len rows
    // Thread n takes loads n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. add up these 8 values over the batch, b = 0, 1, ... in order
        float sum_list[ENCODER_BF16_VALUES_PER_LOAD] = {};
        for (size_t batch_index_val = 0; batch_index_val < batch_size_val; batch_index_val++) {
            // row (b, t)
            float grad_list[ENCODER_BF16_VALUES_PER_LOAD];
            read8_as_float(grad_list, encoded_grad_ptr + (batch_index_val * total_loads_val + index_val) * ENCODER_BF16_VALUES_PER_LOAD);
#pragma unroll
            for (int value_index_val = 0; value_index_val < ENCODER_BF16_VALUES_PER_LOAD; value_index_val++)
                sum_list[value_index_val] += grad_list[value_index_val];
        }
        // 2. add the sums onto wpe_grad
        float old_list[ENCODER_BF16_VALUES_PER_LOAD];
        read8_as_float(old_list, wpe_grad_ptr + index_val * ENCODER_BF16_VALUES_PER_LOAD);
#pragma unroll
        for (int value_index_val = 0; value_index_val < ENCODER_BF16_VALUES_PER_LOAD; value_index_val++)
            old_list[value_index_val] += sum_list[value_index_val];
        write8_from_float(wpe_grad_ptr + index_val * ENCODER_BF16_VALUES_PER_LOAD, old_list);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit: the shapes in encoder_bf16_forward, seq_len <= max_seq_len, encoded,
/// wte and wpe all bfloat16, embed_dim a multiple of ENCODER_BF16_VALUES_PER_LOAD, each starting on 16 bytes.
///
///   caller_ptr                                                 the function checking, for the message
///   encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor   forward: these; backward: their gradients in their place
inline void encoder_bf16_check_shapes(const char* caller_ptr, const Tensor& encoded_tensor, const Tensor& token_ids_tensor,
                                      const Tensor& wte_tensor, const Tensor& wpe_tensor) {
    // 1. shapes: as in encoder_bf16_forward
    bool fits_val = encoded_tensor.dim() == 3 && token_ids_tensor.dim() == 2 && wte_tensor.dim() == 2 &&
                    wpe_tensor.dim() == 2 && token_ids_tensor.dtype() == DType::Int32 &&
                    token_ids_tensor.size(0) == encoded_tensor.size(0) && token_ids_tensor.size(1) == encoded_tensor.size(1) &&
                    wte_tensor.size(1) == encoded_tensor.size(2) && wpe_tensor.size(1) == encoded_tensor.size(2) &&
                    encoded_tensor.size(1) <= wpe_tensor.size(0);
    if (!fits_val) {
        fprintf(stderr, "error: %s needs encoded {batch_size, seq_len, embed_dim}, token_ids {batch_size, seq_len} ints, "
                        "wte {vocab_size, embed_dim}, wpe {max_seq_len, embed_dim} with seq_len <= max_seq_len\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. types: encoded, wte and wpe all bfloat16
    if (encoded_tensor.dtype() != DType::BFloat16 || wte_tensor.dtype() != DType::BFloat16 || wpe_tensor.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: %s needs encoded, wte and wpe all bfloat16 (float: fp32/encoder.cuh)\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte loads, each tensor starting on one
    size_t load_bytes_val = sizeof(uint4);
    if (encoded_tensor.size(2) % ENCODER_BF16_VALUES_PER_LOAD != 0 || (size_t)encoded_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)wte_tensor.data_ptr() % load_bytes_val != 0 || (size_t)wpe_tensor.data_ptr() % load_bytes_val != 0) {
        fprintf(stderr, "error: %s reads %d values at a time, so embed_dim must be a multiple of %d and encoded, wte, wpe "
                        "must start on %zu bytes; got embed_dim %zu, %p, %p, %p\n",
                caller_ptr, ENCODER_BF16_VALUES_PER_LOAD, ENCODER_BF16_VALUES_PER_LOAD, load_bytes_val, encoded_tensor.size(2),
                encoded_tensor.data_ptr(), wte_tensor.data_ptr(), wpe_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // ENCODER_BF16_CUH
