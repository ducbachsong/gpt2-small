// gelu.cuh — GELU, the activation in GPT-2's MLP, the "tanh" form GPT-2 and llm.c use:
//     forward:    y = 0.5 x (1 + tanh(s))       s = sqrt(2 / pi) (x + 0.044715 x³)
//     backward:   dx = dy · gelu'(x)
//                 gelu'(x) = 0.5 (1 + tanh(s)) + 0.5 x (1 - tanh²(s)) sqrt(2 / pi) (1 + 3 · 0.044715 x²)
// input and output any shape, the same for both: {4, 1024, 3072} in GPT-2. All float or all bfloat16 (T);
// the math is in float. Math: https://ducbachsong.github.io/posts/gelu-backward/
#ifndef GELU_CUH
#define GELU_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block
#include "../tensor.cuh"

#define GELU_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // the default block size: 8 warps, one float4 per thread
#define GELU_SCALE 0.7978845608f     // sqrt(2 / pi)
#define GELU_CUBE_FACTOR 0.044715f   // how much of x³ goes into s

// ── forward and backward ──────────────────────────────────────────────────────
void gelu_forward(const Tensor& output_tensor, const Tensor& input_tensor, int threads_per_block_val = GELU_THREADS_PER_BLOCK);
void gelu_backward(const Tensor& input_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor,
                   int threads_per_block_val = GELU_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
__device__ __forceinline__ float gelu_value(float x_val);
__device__ __forceinline__ float gelu_slope(float x_val);
int gelu_blocks(size_t total_float4s_val, int threads_per_block_val = GELU_THREADS_PER_BLOCK);
void gelu_check_shapes(const Tensor& output_tensor, const Tensor& input_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void gelu_forward_kernel(T* output_ptr, const T* input_ptr, size_t total_float4s_val);

/// Writes output = gelu(input): checks, then launches gelu_forward_kernel.
///
///   output_tensor           the result, input's shape
///   input_tensor            any shape, values a multiple of VALUES_PER_FLOAT4
///   threads_per_block_val   the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           GELU_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void gelu_forward(const Tensor& output_tensor, const Tensor& input_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "GELU");
    gelu_check_shapes(output_tensor, input_tensor);
    // 2. launch: one float4 per thread, the blocks rounded up
    size_t total_float4s_val = input_tensor.numel() / VALUES_PER_FLOAT4;
    int blocks_val = gelu_blocks(total_float4s_val, threads_per_block_val);
    if (input_tensor.dtype() == DType::BFloat16)
        gelu_forward_kernel<<<blocks_val, threads_per_block_val>>>(
            output_tensor.data_ptr<nv_bfloat16>(), input_tensor.data_ptr<nv_bfloat16>(), total_float4s_val);
    else
        gelu_forward_kernel<<<blocks_val, threads_per_block_val>>>(output_tensor.data(), input_tensor.data(),
                                                                    total_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// Works out gelu one float4 at a time: reads input, writes gelu of each value.
///
///   output_ptr, input_ptr   as in gelu_forward, float or bfloat16 (T)
///   total_float4s_val       values / VALUES_PER_FLOAT4
template <typename T>
__global__ void gelu_forward_kernel(T* output_ptr, const T* input_ptr, size_t total_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        size_t first_val = VALUES_PER_FLOAT4 * index_val;    // this float4's first value
        // 1. read input, to float
        float4 input_val = read4_as_float(input_ptr + first_val);
        // 2. gelu of each, written
        write4_from_float(output_ptr + first_val, make_float4(gelu_value(input_val.x), gelu_value(input_val.y),
                                                              gelu_value(input_val.z), gelu_value(input_val.w)));
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void gelu_backward_kernel(T* input_grad_ptr, const T* output_grad_ptr, const T* input_ptr,
                                     size_t total_float4s_val);

/// Writes input_grad = output_grad · gelu'(input) (written over, not added): checks, then launches
/// gelu_backward_kernel.
///
///   input_grad_tensor       the result; may be output_grad itself
///   output_grad_tensor      the gradient from the layer above
///   input_tensor            what the forward read
///   threads_per_block_val   the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           GELU_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void gelu_backward(const Tensor& input_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor,
                          int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "GELU");
    gelu_check_shapes(input_grad_tensor, input_tensor);
    gelu_check_shapes(output_grad_tensor, input_tensor);
    // 2. launch: one float4 per thread, the blocks rounded up
    size_t total_float4s_val = input_tensor.numel() / VALUES_PER_FLOAT4;
    int blocks_val = gelu_blocks(total_float4s_val, threads_per_block_val);
    if (input_tensor.dtype() == DType::BFloat16)
        gelu_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            input_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
            input_tensor.data_ptr<nv_bfloat16>(), total_float4s_val);
    else
        gelu_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            input_grad_tensor.data(), output_grad_tensor.data(), input_tensor.data(), total_float4s_val);
    cudaCheck(cudaGetLastError());
}

/// Works out input_grad one float4 at a time, as the forward walks.
///
///   input_grad_ptr, output_grad_ptr, input_ptr   as in gelu_backward, float or bfloat16 (T)
///   total_float4s_val                            values / VALUES_PER_FLOAT4
template <typename T>
__global__ void gelu_backward_kernel(T* input_grad_ptr, const T* output_grad_ptr, const T* input_ptr,
                                     size_t total_float4s_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += (size_t)gridDim.x * blockDim.x) {
        size_t first_val = VALUES_PER_FLOAT4 * index_val;    // this float4's first value
        // 1. read input and output_grad, to float
        float4 input_val = read4_as_float(input_ptr + first_val);
        float4 output_grad_val = read4_as_float(output_grad_ptr + first_val);
        // 2. output_grad x gelu'(input), written
        write4_from_float(input_grad_ptr + first_val,
               make_float4(output_grad_val.x * gelu_slope(input_val.x), output_grad_val.y * gelu_slope(input_val.y),
                           output_grad_val.z * gelu_slope(input_val.z), output_grad_val.w * gelu_slope(input_val.w)));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives gelu(x) = 0.5 x (1 + tanh(s)): about x for a large x, about 0 for a large negative x.
///
///   x_val   one input value
__device__ __forceinline__ float gelu_value(float x_val) {
    float inner_val = GELU_SCALE * (x_val + GELU_CUBE_FACTOR * x_val * x_val * x_val);   // s
    return 0.5f * x_val * (1.0f + tanhf(inner_val));
}

/// Gives gelu'(x), the slope the backward multiplies output_grad by. 1 - tanh² reuses the tanh.
///
///   x_val   one input value
__device__ __forceinline__ float gelu_slope(float x_val) {
    float inner_val = GELU_SCALE * (x_val + GELU_CUBE_FACTOR * x_val * x_val * x_val);              // s
    float tanh_inner_val = tanhf(inner_val);
    float inner_slope_val = GELU_SCALE * (1.0f + 3.0f * GELU_CUBE_FACTOR * x_val * x_val);           // ds / dx
    return 0.5f * (1.0f + tanh_inner_val) +
           0.5f * x_val * (1.0f - tanh_inner_val * tanh_inner_val) * inner_slope_val;
}

/// Gives the blocks for one float4 per thread, rounded up: 3,145,728 float4s in blocks of 256 is 12,288.
///
///   total_float4s_val       values / VALUES_PER_FLOAT4
///   threads_per_block_val   the block size; not given: GELU_THREADS_PER_BLOCK
inline int gelu_blocks(size_t total_float4s_val, int threads_per_block_val) {
    return (int)CEIL_DIV(total_float4s_val, (size_t)threads_per_block_val);
}

/// Stops with a message unless the tensors fit: same shape, both float or both bfloat16, values a multiple
/// of VALUES_PER_FLOAT4, both starting on a whole float4 (for the float4 loads).
///
///   output_tensor   forward: output; backward: input_grad, then output_grad
///   input_tensor    input
inline void gelu_check_shapes(const Tensor& output_tensor, const Tensor& input_tensor) {
    // shapes: the same
    if (output_tensor.shape() != input_tensor.shape()) {
        fprintf(stderr, "error: GELU needs output the same shape as input\n");
        exit(EXIT_FAILURE);
    }
    // types: both float or both bfloat16
    if ((input_tensor.dtype() != DType::Float32 && input_tensor.dtype() != DType::BFloat16) ||
        output_tensor.dtype() != input_tensor.dtype()) {
        fprintf(stderr, "error: GELU needs its tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: whole float4s, both tensors starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * input_tensor.element_size();
    if (input_tensor.numel() % VALUES_PER_FLOAT4 != 0 || (size_t)input_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)output_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: GELU reads %d values at a time, so the number of values must be a multiple of %d and "
                        "input and output must start on %zu bytes; got %zu values, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, input_tensor.numel(), input_tensor.data_ptr(),
                output_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // GELU_CUH
