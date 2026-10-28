// gelu.cuh — GELU, the activation in GPT-2's MLP: each value on its own goes through a smooth curve
// that lets positive values through and pushes negative ones toward 0.
//
// The math is in the note https://ducbachsong.github.io/posts/gelu-backward/. The comments below name
// its steps (F1..F5 forward, B1..B7 backward, C1 and C2 rebuilt) and sections. Its letters for the
// names here:
//
//     x = input   y = output   s = inner   h = tanh(inner)   p = 0.5 (1 + h), the gate
//     β = GELU_SCALE   κ = GELU_CUBE_FACTOR   dy = output_grad   dx = input_grad
//
//     inner  = sqrt(2 / pi) * (x + 0.044715 x³)                                     F1: u = x + κx³, F2: s = βu
//     gelu   = 0.5 x (1 + tanh(inner))         the "tanh" GELU, as GPT-2 and llm.c use
//                                                                                   F3: h = tanh s, F4: p, F5: y = x · p
//
//     x        -10      -3       -1       0      1       3      10
//     gelu      0       -0.004   -0.159   0      0.841   2.996  10
//
//     input, output   any shape, the same for both   in GPT-2, the MLP's middle:   {4, 1024, 3072}
//
// GPT-2 runs one in the MLP of each of its 12 blocks, between the two linears:
//
//     {4, 1024, 768} ── MLP up ──► {4, 1024, 3072} ── GELU ──► {4, 1024, 3072} ── MLP down ──► {4, 1024, 768}
//
// Backward, with output_grad the gradient arriving from the layer above (MLP down's input_grad):
//
//     input_grad = output_grad * gelu'(x)                                           C2: dx = dy · gelu'(x)
//
// gelu'(x), the slope, by the product rule on 0.5 x times (1 + tanh(inner)), with tanh'(u) = 1 - tanh²(u):
//
//     gelu'(x) = 0.5 (1 + tanh(inner))  +  0.5 x (1 - tanh²(inner)) * sqrt(2 / pi) * (1 + 3 * 0.044715 x²)
//                └ d(0.5 x) times the rest ┘  └ 0.5 x times d(1 + tanh(inner)): tanh' times d(inner)/dx ┘
//              = p                       +  x · p'(x)                                        C2
//                B2's part, dx(1)           C1: B1 x, B3 ½, B4 1 - h², B5 β, B6 1 + 3κx²: dx(2)
//
// The note's example, x = [1, -1, 2, -2], gives gelu [0.8411920, -0.1588080, 1.9545977, -0.0454023] and
// gelu' [1.0829641, -0.0829641, 1.0860993, -0.0860993].
//
// Every value is worked out on its own, with nothing shared: the simplest kind of kernel. Its time is
// memory: the forward reads 4 bytes and writes 4 per value, and the tanh is quick next to that (note 7).
// A name ending in 4 (input4, output_grad4, ...) is one float4 of that tensor: 4 values a thread has.
#ifndef GELU_CUH
#define GELU_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define GELU_THREADS_PER_BLOCK 256   // one float4 per thread: {4, 1024, 3072} is 3,145,728 float4s, 12,288 blocks
#define GELU_SCALE 0.7978845608f     // sqrt(2 / pi)
#define GELU_CUBE_FACTOR 0.044715f   // how much of x³ goes into inner

// ── forward and backward ──────────────────────────────────────────────────────
void gelu_forward(const Tensor& output, const Tensor& input);
void gelu_backward(const Tensor& input_grad, const Tensor& output_grad, const Tensor& input);

// ── for both ──────────────────────────────────────────────────────────────────
__device__ __forceinline__ float gelu_value(float x);
__device__ __forceinline__ float gelu_slope(float x);
int gelu_blocks(size_t total_float4s);
void gelu_check_shapes(const Tensor& output, const Tensor& input);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel gelu_forward launches, declared here so it can call it; its code follows it.
__global__ void gelu_forward_kernel(float4* output, const float4* input, size_t total_float4s);

/// Writes output = gelu(input) for every value. output and input have the same shape, any shape whose
/// number of values is a multiple of 4. One thread per float4: {4, 1024, 3072} is 3,145,728 float4s,
/// 12,288 blocks of 256 threads.
inline void gelu_forward(const Tensor& output, const Tensor& input) {
    gelu_check_shapes(output, input);
    size_t total_float4s = input.numel() / 4;
    gelu_forward_kernel<<<gelu_blocks(total_float4s), GELU_THREADS_PER_BLOCK>>>(
        (float4*)output.data(), (const float4*)input.data(), total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes output, one float4 at a time. Thread n takes float4s n, n + every thread launched, ... (with the
/// usual launch, one each). Threads next to each other take float4s next to each other, so a warp reads
/// 32 x 16 = 512 neighbouring bytes at once.
__global__ void gelu_forward_kernel(float4* output, const float4* input, size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 input4 = input[index];
        output[index] = make_float4(gelu_value(input4.x), gelu_value(input4.y), gelu_value(input4.z),
                                    gelu_value(input4.w));
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernel gelu_backward launches, declared here so it can call it; its code follows it.
__global__ void gelu_backward_kernel(float4* input_grad, const float4* output_grad, const float4* input,
                                     size_t total_float4s);

/// Writes input_grad = output_grad * gelu'(input) for every value, from output_grad and what the forward
/// read: input. input_grad is written over, not added onto: in GPT-2 it is the gradient of MLP up's
/// output, which only GELU reads. input_grad may be output_grad itself (each value is read before it is
/// written, by the same thread), so the trainer can save a {4, 1024, 3072} tensor, 50 MB (note 7). The
/// shapes are the forward's: input_grad and output_grad as input.
inline void gelu_backward(const Tensor& input_grad, const Tensor& output_grad, const Tensor& input) {
    gelu_check_shapes(input_grad, input);
    gelu_check_shapes(output_grad, input);
    size_t total_float4s = input.numel() / 4;
    gelu_backward_kernel<<<gelu_blocks(total_float4s), GELU_THREADS_PER_BLOCK>>>(
        (float4*)input_grad.data(), (const float4*)output_grad.data(), (const float4*)input.data(), total_float4s);
    cudaCheck(cudaGetLastError());
}

/// Writes input_grad, one float4 at a time, as the forward walks output: thread n takes float4s n,
/// n + every thread launched, ... Each value reads input and output_grad and writes input_grad: 12 bytes,
/// against the forward's 8. Each value is the note's C2, dx = dy · gelu'(x); the seven steps B1..B7 of
/// its Section 3 are never run one by one.
__global__ void gelu_backward_kernel(float4* input_grad, const float4* output_grad, const float4* input,
                                     size_t total_float4s) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 input4 = input[index], output_grad4 = output_grad[index];
        input_grad[index] = make_float4(output_grad4.x * gelu_slope(input4.x), output_grad4.y * gelu_slope(input4.y),
                                        output_grad4.z * gelu_slope(input4.z), output_grad4.w * gelu_slope(input4.w));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives gelu(x) for one value: 0.5 x (1 + tanh(inner)). For a large x, tanh is 1 and gelu(x) = x; for a
/// large negative x, tanh is -1 and gelu(x) = 0.
__device__ __forceinline__ float gelu_value(float x) {
    float inner = GELU_SCALE * (x + GELU_CUBE_FACTOR * x * x * x);   // F1, F2: s = β (x + κx³)
    return 0.5f * x * (1.0f + tanhf(inner));                          // F3, F4, F5: y = x · ½ (1 + tanh s)
}

/// Gives gelu'(x) for one value: how much gelu's output moves for each unit its input moves, which the
/// backward multiplies output_grad by. 1 - tanh² reuses the tanh already worked out; llm.c writes the same
/// as 1 / cosh², which costs a cosh more (note 7). This is the note's C2 boxed formula, gelu'(x).
__device__ __forceinline__ float gelu_slope(float x) {
    float inner = GELU_SCALE * (x + GELU_CUBE_FACTOR * x * x * x);                // F1, F2: s
    float tanh_inner = tanhf(inner);                                               // F3: h, worked out again, not kept
    float inner_slope = GELU_SCALE * (1.0f + 3.0f * GELU_CUBE_FACTOR * x * x);    // d(inner) / dx: B5's β times B6's 1 + 3κx²
    return 0.5f * (1.0f + tanh_inner) +                                            // p: B2's part, dx(1)
           0.5f * x * (1.0f - tanh_inner * tanh_inner) * inner_slope;              // x · p'(x): C1, dx(2); ½ is B3, 1 - h² B4
}

/// Gives the number of blocks for one float4 per thread, rounded up: a part-full last block. 45 float4s
/// is 1 block, 3,145,728 is 12,288.
inline int gelu_blocks(size_t total_float4s) {
    return (int)((total_float4s + GELU_THREADS_PER_BLOCK - 1) / GELU_THREADS_PER_BLOCK);
}

/// Stops with a message if the tensors do not fit together, before a kernel reads past one: output the
/// same shape as input. Also the number of values a multiple of 4 and both starting on 16 bytes, for the
/// float4 loads: off 16 bytes, the GPU stops with "misaligned address" at some later call, far from the
/// cause. Backward passes input_grad, then output_grad, in output's place.
inline void gelu_check_shapes(const Tensor& output, const Tensor& input) {
    if (output.shape() != input.shape()) {
        fprintf(stderr, "error: GELU needs output the same shape as input\n");
        exit(EXIT_FAILURE);
    }
    if (input.numel() % 4 != 0 || (size_t)input.data() % 16 != 0 || (size_t)output.data() % 16 != 0) {
        fprintf(stderr, "error: GELU reads 4 floats at a time (float4), so the number of values must be a multiple "
                        "of 4 and input and output must start on 16 bytes; got %zu values, %p, %p\n",
                input.numel(), (void*)input.data(), (void*)output.data());
        exit(EXIT_FAILURE);
    }
}

#endif // GELU_CUH
