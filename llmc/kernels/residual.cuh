// residual.cuh — the residual add: what a block's attention or MLP gives is added back onto the row it
// was given, value by value, so the branch only has to learn a change to the row, not the whole row:
//
//     output = residual + branch_output
//
//     residual, branch_output, output   {batch_size, seq_len, embed_dim}, the same for all three:   {4, 1024, 768}
//
// GPT-2 runs two in each of its 12 blocks, 24 in all, one around the attention and one around the MLP:
//
//     residual ──┬── LayerNorm ── attention ── linear ──► branch_output ──┐
//                └────────────────────────────────────────────────────── + ──► output, the next residual
//
// Backward, with output_grad the gradient arriving from the layer above:
//
//     residual_grad      = output_grad      the slope of residual + branch_output is 1 for each
//     branch_output_grad = output_grad
//
// residual_grad is only part 1 of the residual's gradient: the residual also goes into the branch's
// LayerNorm, and layer_norm_backward adds that part 2 onto residual_grad (input_grad +=). So the residual
// backward writes over (=) and runs first, before the branch's backward.
//
// The math is in the note https://ducbachsong.github.io/posts/residual-backward/. The comments below
// name its steps (F1 forward, B1..B4 backward, C1 and C2 rebuilt) and sections. Its letters for the
// names here:
//
//     r = residual   b = branch_output   y = output   dy = output_grad   db = branch_output_grad
//     dr(1) = residual_grad, before the branch's LayerNorm adds dr(2)
//
// The note's example, r = [1, 2, 3, 4] and b = [0.5, -2, 0, 2], gives y = [1.5, 0, 3, 6]; with
// dy = [1, -2, 0.5, 3], residual_grad and branch_output_grad are both [1, -2, 0.5, 3].
//
// Every value is worked out on its own, as in gelu.cuh. Its time is memory: the forward reads 8 bytes
// and writes 4 per value, the backward reads 4 and writes 8 (note 6).
// A name ending in 4 (residual4, output_grad4, ...) is one float4 of that tensor: 4 values a thread has.
//
// The three tensors are all float or all bfloat16; the kernels are templates on that dtype, T. A float4 is
// 4 neighbouring values either way: for bfloat16, one 8-byte read that load4 (tensor.cuh) turns into 4
// floats, and store4 rounds back. The add itself is always in float.
#ifndef RESIDUAL_CUH
#define RESIDUAL_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define RESIDUAL_THREADS_PER_BLOCK 256   // one float4 per thread: {4, 1024, 768} is 786,432 float4s, 3072 blocks

// ── forward and backward ──────────────────────────────────────────────────────
void residual_forward(const Tensor& output, const Tensor& residual, const Tensor& branch_output);
void residual_backward(const Tensor& residual_grad, const Tensor& branch_output_grad, const Tensor& output_grad);

// ── for both ──────────────────────────────────────────────────────────────────
int residual_blocks(size_t total_float4s);
void residual_check_shapes(const Tensor& first, const Tensor& second, const Tensor& third);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel residual_forward launches, declared here so it can call it; its code follows it.
template <typename T>
__global__ void residual_forward_kernel(T* output, const T* residual, const T* branch_output, size_t total_float4s);

/// Writes output = residual + branch_output for every value: F1. All three have the same shape, any shape
/// whose number of values is a multiple of 4. output may be residual or branch_output itself (each value
/// is read before it is written, by the same thread). One thread per float4: {4, 1024, 768} is 786,432
/// float4s, 3072 blocks of 256 threads.
inline void residual_forward(const Tensor& output, const Tensor& residual, const Tensor& branch_output) {
    residual_check_shapes(output, residual, branch_output);
    size_t total_float4s = output.numel() / 4;
    if (output.dtype() == DType::BFloat16)
        residual_forward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
            output.data_ptr<nv_bfloat16>(), residual.data_ptr<nv_bfloat16>(), branch_output.data_ptr<nv_bfloat16>(),
            total_float4s);
    else
        residual_forward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
            output.data(), residual.data(), branch_output.data(), total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes output, one float4 at a time. Thread n takes float4s n, n + every thread launched, ... (with the
/// usual launch, one each). Threads next to each other take float4s next to each other, so a warp reads
/// 32 x 16 = 512 neighbouring bytes of each input at once.
template <typename T>
__global__ void residual_forward_kernel(T* output, const T* residual, const T* branch_output, size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 residual4 = load4(residual + 4 * index), branch_output4 = load4(branch_output + 4 * index);
        store4(output + 4 * index,
               make_float4(residual4.x + branch_output4.x, residual4.y + branch_output4.y,   // F1: y = r + b
                           residual4.z + branch_output4.z, residual4.w + branch_output4.w));
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernel residual_backward launches, declared here so it can call it; its code follows it.
template <typename T>
__global__ void residual_backward_kernel(T* residual_grad, T* branch_output_grad, const T* output_grad,
                                         size_t total_float4s);

/// Writes output_grad into residual_grad and into branch_output_grad: B2 and B1. Both are written over, not
/// added onto: branch_output is read by the add alone, and residual_grad is part 1 of the residual's
/// gradient, which the branch's LayerNorm backward then adds part 2 onto (B3, B4). Either may be
/// output_grad itself (each value is read before it is written, by the same thread). The backward needs
/// nothing from the forward: the slope of an add is 1 whatever was added. The shapes are the forward's.
///
/// The trainer can skip this kernel by giving residual_grad and branch_output_grad output_grad's own
/// memory, as llm.c does (note 6). It is here so the backward reads like every other layer's.
inline void residual_backward(const Tensor& residual_grad, const Tensor& branch_output_grad, const Tensor& output_grad) {
    residual_check_shapes(residual_grad, branch_output_grad, output_grad);
    size_t total_float4s = output_grad.numel() / 4;
    if (output_grad.dtype() == DType::BFloat16)
        residual_backward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
            residual_grad.data_ptr<nv_bfloat16>(), branch_output_grad.data_ptr<nv_bfloat16>(),
            output_grad.data_ptr<nv_bfloat16>(), total_float4s);
    else
        residual_backward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
            residual_grad.data(), branch_output_grad.data(), output_grad.data(), total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes residual_grad and branch_output_grad, one float4 at a time, as the forward walks output: thread n
/// takes float4s n, n + every thread launched, ... Each value reads output_grad once and writes it twice:
/// 12 bytes, as the forward's.
template <typename T>
__global__ void residual_backward_kernel(T* residual_grad, T* branch_output_grad, const T* output_grad,
                                         size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 output_grad4 = load4(output_grad + 4 * index);
        store4(residual_grad + 4 * index, output_grad4);         // B2: dr(1) = dy · 1
        store4(branch_output_grad + 4 * index, output_grad4);    // B1: db = dy · 1
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the number of blocks for one float4 per thread, rounded up: a part-full last block. 45 float4s
/// is 1 block, 786,432 is 3072.
inline int residual_blocks(size_t total_float4s) {
    return (int)((total_float4s + RESIDUAL_THREADS_PER_BLOCK - 1) / RESIDUAL_THREADS_PER_BLOCK);
}

/// Stops with a message if the three tensors do not fit together, before a kernel reads past one: all
/// the same shape, and all float or all bfloat16. Also the number of values a multiple of 4 and all three
/// starting on 4 values (16 bytes for float, 8 for bfloat16), for the 4-value loads: off that, the GPU
/// stops with "misaligned address" at some later call, far from the cause. Forward passes output,
/// residual, branch_output; backward residual_grad, branch_output_grad, output_grad.
inline void residual_check_shapes(const Tensor& first, const Tensor& second, const Tensor& third) {
    if (first.shape() != second.shape() || first.shape() != third.shape()) {
        fprintf(stderr, "error: the residual add needs its three tensors the same shape\n");
        exit(EXIT_FAILURE);
    }
    DType dtype = first.dtype();
    if ((dtype != DType::Float32 && dtype != DType::BFloat16) || second.dtype() != dtype || third.dtype() != dtype) {
        fprintf(stderr, "error: the residual add needs its three tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    size_t four_values = 4 * first.element_size();                      // 16 bytes for float, 8 for bfloat16
    if (first.numel() % 4 != 0 || (size_t)first.data_ptr() % four_values != 0 ||
        (size_t)second.data_ptr() % four_values != 0 || (size_t)third.data_ptr() % four_values != 0) {
        fprintf(stderr, "error: the residual add reads 4 values at a time, so the number of values must be a "
                        "multiple of 4 and all three tensors must start on %zu bytes; got %zu values, %p, %p, %p\n",
                four_values, first.numel(), first.data_ptr(), second.data_ptr(), third.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // RESIDUAL_CUH
