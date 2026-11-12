// residual_bf16.cuh — the residual add in bfloat16, around each block's attention and MLP, 8 values a thread (one
// 16-byte load): fp32/residual.cuh's math, whose bfloat16 path takes 4 values a thread (one 8-byte load):
//     forward:    output = residual + branch_output
//     backward:   residual_grad = branch_output_grad = output_grad
//
//     tensor                                       type in memory   shape                               GPT-2 small
//     output, residual, branch_output, and grads   bfloat16         {batch_size, seq_len, embed_dim}    {4, 1024, 768}
//     in registers                                 float            -                                   the add
//
//     GPT-2 small, one forward     loads a tensor    bytes a load
//     fp32/residual.cuh, bf16      786,432           8
//     here                         393,216           16
//
// The same bits as fp32/residual.cuh on bfloat16: each value is one float add of the same two numbers, rounded once
// as it is stored (or, backward, one copy), whichever thread does it. residual_grad is only part 1: the branch's
// LayerNorm backward adds part 2 onto it, so this runs first. The 8-bit model runs this too: its residual stream stays
// bfloat16, and what reads it next is a LayerNorm, not an 8-bit product. Any GPU.
// Math: https://ducbachsong.github.io/posts/residual-backward/
#ifndef RESIDUAL_BF16_CUH
#define RESIDUAL_BF16_CUH

#include "../cuda_common.h"          // cudaCheck, check_threads_per_block
#include "../../common/tensor.cuh"          // read8_as_float, write8_from_float, VALUES_PER_BF16_LOAD

#define RESIDUAL_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD                     // bfloat16s in one 16-byte load: 8
#define RESIDUAL_BF16_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)        // the default block size: fp32/residual.cuh's, not swept for 8 values yet

// ── forward and backward ──────────────────────────────────────────────────────
void residual_bf16_forward(const Tensor& output_tensor, const Tensor& residual_tensor, const Tensor& branch_output_tensor,
                           int threads_per_block_val = RESIDUAL_BF16_THREADS_PER_BLOCK);
void residual_bf16_backward(const Tensor& residual_grad_tensor, const Tensor& branch_output_grad_tensor,
                            const Tensor& output_grad_tensor, int threads_per_block_val = RESIDUAL_BF16_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
int residual_bf16_blocks(size_t total_loads_val, int threads_per_block_val = RESIDUAL_BF16_THREADS_PER_BLOCK);
void residual_bf16_check_shapes(const char* caller_ptr, const Tensor& first_tensor, const Tensor& second_tensor,
                                const Tensor& third_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void residual_bf16_forward_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* residual_ptr,
                                             const nv_bfloat16* branch_output_ptr, size_t total_loads_val);

/// Writes output = residual + branch_output in bfloat16: checks, then launches residual_bf16_forward_kernel.
///
///   output_tensor           the sum; may be residual or branch_output itself
///   residual_tensor         the row the branch was given
///   branch_output_tensor    what the attention or MLP gave
///   threads_per_block_val   the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                           RESIDUAL_BF16_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void residual_bf16_forward(const Tensor& output_tensor, const Tensor& residual_tensor, const Tensor& branch_output_tensor,
                                  int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "residual_bf16_forward");
    residual_bf16_check_shapes("residual_bf16_forward", output_tensor, residual_tensor, branch_output_tensor);
    // 2. launch: one load (8 values) per thread, the blocks rounded up
    size_t total_loads_val = output_tensor.numel() / RESIDUAL_BF16_VALUES_PER_LOAD;
    residual_bf16_forward_kernel<<<residual_bf16_blocks(total_loads_val, threads_per_block_val), threads_per_block_val>>>(
        output_tensor.data_ptr<nv_bfloat16>(), residual_tensor.data_ptr<nv_bfloat16>(), branch_output_tensor.data_ptr<nv_bfloat16>(),
        total_loads_val);
    cudaCheck(cudaGetLastError());
}

/// Adds 8 values (one 16-byte load of each) at a time: reads residual and branch_output, writes their sum, added in
/// float and rounded once.
///
///   output_ptr, residual_ptr, branch_output_ptr   as in residual_bf16_forward
///   total_loads_val                               values / RESIDUAL_BF16_VALUES_PER_LOAD
__global__ void residual_bf16_forward_kernel(nv_bfloat16* output_ptr, const nv_bfloat16* residual_ptr,
                                             const nv_bfloat16* branch_output_ptr, size_t total_loads_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes loads n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        size_t first_value_index_val = RESIDUAL_BF16_VALUES_PER_LOAD * index_val;    // this load's first value
        // 1. read both, to float
        float residual_list[RESIDUAL_BF16_VALUES_PER_LOAD], branch_output_list[RESIDUAL_BF16_VALUES_PER_LOAD];
        read8_as_float(residual_list, residual_ptr + first_value_index_val);
        read8_as_float(branch_output_list, branch_output_ptr + first_value_index_val);
        // 2. add and write
#pragma unroll
        for (int value_index_val = 0; value_index_val < RESIDUAL_BF16_VALUES_PER_LOAD; value_index_val++)
            residual_list[value_index_val] += branch_output_list[value_index_val];
        write8_from_float(output_ptr + first_value_index_val, residual_list);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

__global__ void residual_bf16_backward_kernel(nv_bfloat16* residual_grad_ptr, nv_bfloat16* branch_output_grad_ptr,
                                              const nv_bfloat16* output_grad_ptr, size_t total_loads_val);

/// Copies output_grad into residual_grad and branch_output_grad (written over, not added), in bfloat16: checks, then
/// launches residual_bf16_backward_kernel.
///
///   residual_grad_tensor        part 1 of the residual's gradient
///   branch_output_grad_tensor   the branch's gradient
///   output_grad_tensor          the gradient from the layer above; may be either of the two itself
///   threads_per_block_val       the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                               RESIDUAL_BF16_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void residual_bf16_backward(const Tensor& residual_grad_tensor, const Tensor& branch_output_grad_tensor,
                                   const Tensor& output_grad_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "residual_bf16_backward");
    residual_bf16_check_shapes("residual_bf16_backward", residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    // 2. launch: one load (8 values) per thread, the blocks rounded up
    size_t total_loads_val = output_grad_tensor.numel() / RESIDUAL_BF16_VALUES_PER_LOAD;
    residual_bf16_backward_kernel<<<residual_bf16_blocks(total_loads_val, threads_per_block_val), threads_per_block_val>>>(
        residual_grad_tensor.data_ptr<nv_bfloat16>(), branch_output_grad_tensor.data_ptr<nv_bfloat16>(),
        output_grad_tensor.data_ptr<nv_bfloat16>(), total_loads_val);
    cudaCheck(cudaGetLastError());
}

/// Copies 8 values (one 16-byte load) at a time, as the forward walks: the bytes as they are, no conversion.
///
///   residual_grad_ptr, branch_output_grad_ptr, output_grad_ptr   as in residual_bf16_backward
///   total_loads_val                                              values / RESIDUAL_BF16_VALUES_PER_LOAD
__global__ void residual_bf16_backward_kernel(nv_bfloat16* residual_grad_ptr, nv_bfloat16* branch_output_grad_ptr,
                                              const nv_bfloat16* output_grad_ptr, size_t total_loads_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes loads n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. read output_grad's 16 bytes
        uint4 output_grad_val = reinterpret_cast<const uint4*>(output_grad_ptr)[index_val];
        // 2. write them to both
        reinterpret_cast<uint4*>(residual_grad_ptr)[index_val] = output_grad_val;
        reinterpret_cast<uint4*>(branch_output_grad_ptr)[index_val] = output_grad_val;
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the blocks for one load per thread, rounded up: 393,216 loads in blocks of 256 is 1536.
///
///   total_loads_val         values / RESIDUAL_BF16_VALUES_PER_LOAD
///   threads_per_block_val   the block size; not given: RESIDUAL_BF16_THREADS_PER_BLOCK
inline int residual_bf16_blocks(size_t total_loads_val, int threads_per_block_val) {
    return (int)CEIL_DIV(total_loads_val, (size_t)threads_per_block_val);
}

/// Stops with a message unless the three tensors fit: same shape, all bfloat16, values a multiple of
/// RESIDUAL_BF16_VALUES_PER_LOAD, each starting on 16 bytes (for the 16-byte loads).
///
///   caller_ptr                                  the function checking, for the message
///   first_tensor, second_tensor, third_tensor   forward: output, residual, branch_output; backward: the three grads
inline void residual_bf16_check_shapes(const char* caller_ptr, const Tensor& first_tensor, const Tensor& second_tensor,
                                       const Tensor& third_tensor) {
    // 1. shapes: all the same
    if (first_tensor.shape() != second_tensor.shape() || first_tensor.shape() != third_tensor.shape()) {
        fprintf(stderr, "error: %s needs its three tensors the same shape\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16
    if (first_tensor.dtype() != DType::BFloat16 || second_tensor.dtype() != DType::BFloat16 ||
        third_tensor.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: %s needs its three tensors all bfloat16 (float: fp32/residual.cuh)\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: whole 16-byte loads, each tensor starting on one
    size_t load_bytes_val = sizeof(uint4);
    if (first_tensor.numel() % RESIDUAL_BF16_VALUES_PER_LOAD != 0 || (size_t)first_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)second_tensor.data_ptr() % load_bytes_val != 0 || (size_t)third_tensor.data_ptr() % load_bytes_val != 0) {
        fprintf(stderr, "error: %s reads %d values at a time, so the number of values must be a multiple of %d and all "
                        "three tensors must start on %zu bytes; got %zu values, %p, %p, %p\n",
                caller_ptr, RESIDUAL_BF16_VALUES_PER_LOAD, RESIDUAL_BF16_VALUES_PER_LOAD, load_bytes_val, first_tensor.numel(),
                first_tensor.data_ptr(), second_tensor.data_ptr(), third_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // RESIDUAL_BF16_CUH
