// residual.cuh — the residual add, around each block's attention and MLP:
//     forward:    output = residual + branch_output
//     backward:   residual_grad = branch_output_grad = output_grad
// All three tensors {batch_size, seq_len, embed_dim}, all float or all bfloat16 (T); the add is in float.
// residual_grad is only part 1: the branch's LayerNorm backward adds part 2 onto it, so this runs first.
// Math: https://ducbachsong.github.io/posts/residual-backward/
#ifndef RESIDUAL_CUH
#define RESIDUAL_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block
#include "../tensor.cuh"

#define RESIDUAL_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // the default block size: 8 warps, one float4 per thread

// ── forward and backward ──────────────────────────────────────────────────────
void residual_forward(const Tensor& output_tensor, const Tensor& residual_tensor, const Tensor& branch_output_tensor,
                      int threads_per_block_val = RESIDUAL_THREADS_PER_BLOCK);
void residual_backward(const Tensor& residual_grad_tensor, const Tensor& branch_output_grad_tensor,
                       const Tensor& output_grad_tensor, int threads_per_block_val = RESIDUAL_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
int residual_blocks(size_t total_float4s_val, int threads_per_block_val = RESIDUAL_THREADS_PER_BLOCK);
void residual_check_shapes(const Tensor& first_tensor, const Tensor& second_tensor, const Tensor& third_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void residual_forward_kernel(T* output_ptr, const T* residual_ptr, const T* branch_output_ptr,
                                        size_t total_float4s_val);

/// Writes output = residual + branch_output: checks, then launches residual_forward_kernel.
///
///   output_tensor           the sum; may be residual or branch_output itself
///   residual_tensor         the row the branch was given
///   branch_output_tensor    what the attention or MLP gave
///   threads_per_block_val   the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           RESIDUAL_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void residual_forward(const Tensor& output_tensor, const Tensor& residual_tensor, const Tensor& branch_output_tensor,
                             int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the residual add");
    residual_check_shapes(output_tensor, residual_tensor, branch_output_tensor);
    // 2. launch: one float4 per thread, the blocks rounded up
    size_t total_float4s_val = output_tensor.numel() / VALUES_PER_FLOAT4;
    int blocks_val = residual_blocks(total_float4s_val, threads_per_block_val);
    if (output_tensor.dtype() == DType::BFloat16)
        residual_forward_kernel<<<blocks_val, threads_per_block_val>>>(
            output_tensor.data_ptr<nv_bfloat16>(), residual_tensor.data_ptr<nv_bfloat16>(),
            branch_output_tensor.data_ptr<nv_bfloat16>(), total_float4s_val);
    else
        residual_forward_kernel<<<blocks_val, threads_per_block_val>>>(
            output_tensor.data(), residual_tensor.data(), branch_output_tensor.data(), total_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// Adds one float4 at a time: reads residual and branch_output, writes their sum.
///
///   output_ptr, residual_ptr, branch_output_ptr   as in residual_forward, float or bfloat16 (T)
///   total_float4s_val                             values / VALUES_PER_FLOAT4
template <typename T>
__global__ void residual_forward_kernel(T* output_ptr, const T* residual_ptr, const T* branch_output_ptr,
                                        size_t total_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        size_t first_val = VALUES_PER_FLOAT4 * index_val;    // this float4's first value
        // 1. read both, to float
        float4 residual_val = read4_as_float(residual_ptr + first_val);
        float4 branch_output_val = read4_as_float(branch_output_ptr + first_val);
        // 2. add and write
        write4_from_float(output_ptr + first_val,
                          make_float4(residual_val.x + branch_output_val.x, residual_val.y + branch_output_val.y,
                                      residual_val.z + branch_output_val.z, residual_val.w + branch_output_val.w));
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void residual_backward_kernel(T* residual_grad_ptr, T* branch_output_grad_ptr, const T* output_grad_ptr,
                                         size_t total_float4s_val);

/// Copies output_grad into residual_grad and branch_output_grad (written over, not added): checks, then
/// launches residual_backward_kernel.
///
///   residual_grad_tensor        part 1 of the residual's gradient
///   branch_output_grad_tensor   the branch's gradient
///   output_grad_tensor          the gradient from the layer above; may be either of the two itself
///   threads_per_block_val       the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                               RESIDUAL_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void residual_backward(const Tensor& residual_grad_tensor, const Tensor& branch_output_grad_tensor,
                              const Tensor& output_grad_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the residual add");
    residual_check_shapes(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    // 2. launch: one float4 per thread, the blocks rounded up
    size_t total_float4s_val = output_grad_tensor.numel() / VALUES_PER_FLOAT4;
    int blocks_val = residual_blocks(total_float4s_val, threads_per_block_val);
    if (output_grad_tensor.dtype() == DType::BFloat16)
        residual_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            residual_grad_tensor.data_ptr<nv_bfloat16>(), branch_output_grad_tensor.data_ptr<nv_bfloat16>(),
            output_grad_tensor.data_ptr<nv_bfloat16>(), total_float4s_val);
    else
        residual_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            residual_grad_tensor.data(), branch_output_grad_tensor.data(), output_grad_tensor.data(), total_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// Copies one float4 at a time, as the forward walks.
///
///   residual_grad_ptr, branch_output_grad_ptr, output_grad_ptr   as in residual_backward, float or bfloat16 (T)
///   total_float4s_val                                            values / VALUES_PER_FLOAT4
template <typename T>
__global__ void residual_backward_kernel(T* residual_grad_ptr, T* branch_output_grad_ptr, const T* output_grad_ptr,
                                         size_t total_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        size_t first_val = VALUES_PER_FLOAT4 * index_val;    // this float4's first value
        // 1. read output_grad
        float4 output_grad_val = read4_as_float(output_grad_ptr + first_val);
        // 2. write it to both
        write4_from_float(residual_grad_ptr + first_val, output_grad_val);
        write4_from_float(branch_output_grad_ptr + first_val, output_grad_val);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the blocks for one float4 per thread, rounded up: 786,432 float4s in blocks of 256 is 3072.
///
///   total_float4s_val       values / VALUES_PER_FLOAT4
///   threads_per_block_val   the block size; not given: RESIDUAL_THREADS_PER_BLOCK
inline int residual_blocks(size_t total_float4s_val, int threads_per_block_val) {
    return (int)CEIL_DIV(total_float4s_val, (size_t)threads_per_block_val);
}

/// Stops with a message unless the three tensors fit: same shape, all float or all bfloat16, values a
/// multiple of VALUES_PER_FLOAT4, each starting on a whole float4 (for the float4 loads).
///
///   first_tensor, second_tensor, third_tensor   forward: output, residual, branch_output; backward: the three grads
inline void residual_check_shapes(const Tensor& first_tensor, const Tensor& second_tensor, const Tensor& third_tensor) {
    // shapes: all the same
    if (first_tensor.shape() != second_tensor.shape() || first_tensor.shape() != third_tensor.shape()) {
        fprintf(stderr, "error: the residual add needs its three tensors the same shape\n");
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    DType dtype_val = first_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || second_tensor.dtype() != dtype_val ||
        third_tensor.dtype() != dtype_val) {
        fprintf(stderr, "error: the residual add needs its three tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * first_tensor.element_size();
    if (first_tensor.numel() % VALUES_PER_FLOAT4 != 0 || (size_t)first_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)second_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)third_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: the residual add reads %d values at a time, so the number of values must be a "
                        "multiple of %d and all three tensors must start on %zu bytes; got %zu values, %p, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, first_tensor.numel(), first_tensor.data_ptr(),
                second_tensor.data_ptr(), third_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // RESIDUAL_CUH
