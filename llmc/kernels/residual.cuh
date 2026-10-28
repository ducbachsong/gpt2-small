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
__global__ void residual_forward_kernel(float4* output, const float4* residual, const float4* branch_output,
                                        size_t total_float4s);

/// Writes output = residual + branch_output for every value: F1. All three have the same shape, any shape
/// whose number of values is a multiple of 4. output may be residual or branch_output itself (each value
/// is read before it is written, by the same thread). One thread per float4: {4, 1024, 768} is 786,432
/// float4s, 3072 blocks of 256 threads.
inline void residual_forward(const Tensor& output, const Tensor& residual, const Tensor& branch_output) {
    residual_check_shapes(output, residual, branch_output);
    size_t total_float4s = output.numel() / 4;
    residual_forward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
        (float4*)output.data(), (const float4*)residual.data(), (const float4*)branch_output.data(), total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes output, one float4 at a time. Thread n takes float4s n, n + every thread launched, ... (with the
/// usual launch, one each). Threads next to each other take float4s next to each other, so a warp reads
/// 32 x 16 = 512 neighbouring bytes of each input at once.
__global__ void residual_forward_kernel(float4* output, const float4* residual, const float4* branch_output,
                                        size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 residual4 = residual[index], branch_output4 = branch_output[index];
        output[index] = make_float4(residual4.x + branch_output4.x, residual4.y + branch_output4.y,   // F1: y = r + b
                                    residual4.z + branch_output4.z, residual4.w + branch_output4.w);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernel residual_backward launches, declared here so it can call it; its code follows it.
__global__ void residual_backward_kernel(float4* residual_grad, float4* branch_output_grad, const float4* output_grad,
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
    residual_backward_kernel<<<residual_blocks(total_float4s), RESIDUAL_THREADS_PER_BLOCK>>>(
        (float4*)residual_grad.data(), (float4*)branch_output_grad.data(), (const float4*)output_grad.data(),
        total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes residual_grad and branch_output_grad, one float4 at a time, as the forward walks output: thread n
/// takes float4s n, n + every thread launched, ... Each value reads output_grad once and writes it twice:
/// 12 bytes, as the forward's.
__global__ void residual_backward_kernel(float4* residual_grad, float4* branch_output_grad, const float4* output_grad,
                                         size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 output_grad4 = output_grad[index];
        residual_grad[index] = output_grad4;         // B2: dr(1) = dy · 1
        branch_output_grad[index] = output_grad4;    // B1: db = dy · 1
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives the number of blocks for one float4 per thread, rounded up: a part-full last block. 45 float4s
/// is 1 block, 786,432 is 3072.
inline int residual_blocks(size_t total_float4s) {
    return (int)((total_float4s + RESIDUAL_THREADS_PER_BLOCK - 1) / RESIDUAL_THREADS_PER_BLOCK);
}

/// Stops with a message if the three tensors do not fit together, before a kernel reads past one: all
/// the same shape. Also the number of values a multiple of 4 and all three starting on 16 bytes, for the
/// float4 loads: off 16 bytes, the GPU stops with "misaligned address" at some later call, far from the
/// cause. Forward passes output, residual, branch_output; backward residual_grad, branch_output_grad,
/// output_grad.
inline void residual_check_shapes(const Tensor& first, const Tensor& second, const Tensor& third) {
    if (first.shape() != second.shape() || first.shape() != third.shape()) {
        fprintf(stderr, "error: the residual add needs its three tensors the same shape\n");
        exit(EXIT_FAILURE);
    }
    if (first.numel() % 4 != 0 || (size_t)first.data() % 16 != 0 || (size_t)second.data() % 16 != 0 ||
        (size_t)third.data() % 16 != 0) {
        fprintf(stderr, "error: the residual add reads 4 floats at a time (float4), so the number of values must "
                        "be a multiple of 4 and all three tensors must start on 16 bytes; got %zu values, %p, %p, %p\n",
                first.numel(), (void*)first.data(), (void*)second.data(), (void*)third.data());
        exit(EXIT_FAILURE);
    }
}

#endif // RESIDUAL_CUH
