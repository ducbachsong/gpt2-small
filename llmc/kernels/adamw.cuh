// adamw.cuh — gradient clipping and one AdamW step, in one pass over all the parameters:
//     g = g * clip                clip = min(1, max_norm / (norm + 1e-6))
//     p = p * (1 - lr wd)         weight decay, matrices only
//     m = b1 m + (1-b1) g         v = b2 v + (1-b2) g²
//     p = p - (lr / (1-b1ᵗ)) m / (sqrt(v) / sqrt(1-b2ᵗ) + eps)
// One launch for every tensor: the matrices come first, so the first decayed_count
// values get weight decay and the rest (biases, LayerNorm weights) do not:
//
//     params   [ wte | wpe | qkvw | ... matrices ... | ln1w | ln1b | ... the rest ... ]
//                                                     ↑ decayed_count
//
// Each thread takes 4 values with float4 loads, so every tensor's size must be a
// multiple of 4. The norm stays on the GPU (from grad_norm.cuh).
//
// params and grads may be float or bfloat16 (T, the kernel's template); m and v are always
// float. Each value is read as a float, updated in float, and the new parameter rounded to
// T as it is stored (load4 and store4, tensor.cuh). A bfloat16 keeps 8 significant bits:
// an update smaller than half the gap to the next bfloat16 (up to 2^-8 of the value) is lost.
#ifndef ADAMW_CUH
#define ADAMW_CUH

#include "cuda_common.h"
#include "../tensor.cuh"             // load4, store4: 4 values of float or bfloat16 as a float4

__device__ __forceinline__ float adamw_value(float p, float g, float* m, float* v, float clip, float decay_factor,
                                             float beta1, float beta2, float eps, float step_size,
                                             float inverse_sqrt_bias_correction2) {
    g = g * clip;
    p = p * decay_factor;
    *m = beta1 * *m + (1.0f - beta1) * g;
    *v = beta2 * *v + (1.0f - beta2) * g * g;
    float denominator = sqrtf(*v) * inverse_sqrt_bias_correction2 + eps;
    return p - step_size * *m / denominator;
}

template <typename T>
__global__ void adamw_kernel(T* params, const T* grads, float* m_memory, float* v_memory, size_t n,
                             size_t decayed_count, const float* grad_norm, float max_norm, float weight_decay_factor,
                             float beta1, float beta2, float eps, float step_size,
                             float inverse_sqrt_bias_correction2) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // values 4 group .. 4 group + 3
    if (group * 4 >= n) return;
    float clip = 1.0f;
    if (max_norm > 0.0f) {
        clip = max_norm / (*grad_norm + 1e-6f);
        if (clip > 1.0f) clip = 1.0f;
    }
    // Before the border: a matrix, weight decay. decayed_count is a multiple of 4, so
    // all 4 values of this thread are on the same side.
    float decay_factor = group * 4 < decayed_count ? weight_decay_factor : 1.0f;
    float4 p = load4(params + 4 * group);
    float4 g = load4(grads + 4 * group);
    float4 m = reinterpret_cast<float4*>(m_memory)[group];
    float4 v = reinterpret_cast<float4*>(v_memory)[group];
    p.x = adamw_value(p.x, g.x, &m.x, &v.x, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.y = adamw_value(p.y, g.y, &m.y, &v.y, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.z = adamw_value(p.z, g.z, &m.z, &v.z, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.w = adamw_value(p.w, g.w, &m.w, &v.w, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    store4(params + 4 * group, p);
    reinterpret_cast<float4*>(m_memory)[group] = m;
    reinterpret_cast<float4*>(v_memory)[group] = v;
}

/// One AdamW step t on n values in a row: all the parameters, matrices first. The first
/// decayed_count values get weight decay, the rest do not. max_norm 0: no clipping. params and
/// grads are float* or nv_bfloat16*, both the same; m, v and grad_norm float*.
template <typename T>
void adamw_update(T* params, const T* grads, float* m, float* v, size_t n, size_t decayed_count,
                  const float* grad_norm, float max_norm, float learning_rate, float beta1, float beta2, float eps,
                  float weight_decay, int t) {
    if (n % 4 != 0 || decayed_count % 4 != 0 || decayed_count > n) {
        fprintf(stderr, "error: AdamW needs n and decayed_count to be multiples of 4 (float4 loads), with "
                        "decayed_count <= n; got n = %zu, decayed_count = %zu\n", n, decayed_count);
        exit(EXIT_FAILURE);
    }
    // 4-value loads need each block to start on 4 values: 16 bytes for float (m and v always),
    // 8 for bfloat16; off it, the GPU stops with "misaligned address" at some later call, far
    // from the cause.
    size_t four_values = 4 * sizeof(T);
    if ((size_t)params % four_values != 0 || (size_t)grads % four_values != 0 || (size_t)m % 16 != 0 ||
        (size_t)v % 16 != 0) {
        fprintf(stderr, "error: AdamW needs params and grads to start on %zu bytes and m and v on 16 (4-value "
                        "loads); got %p, %p, %p, %p\n", four_values, (void*)params, (const void*)grads, (void*)m,
                (void*)v);
        exit(EXIT_FAILURE);
    }
    float bias_correction1 = 1.0f - powf(beta1, (float)t);
    float bias_correction2 = 1.0f - powf(beta2, (float)t);
    float step_size = learning_rate / bias_correction1;
    float inverse_sqrt_bias_correction2 = 1.0f / sqrtf(bias_correction2);
    float weight_decay_factor = 1.0f - learning_rate * weight_decay;
    size_t groups = n / 4;
    adamw_kernel<<<(unsigned)CEIL_DIV(groups, 256), 256>>>(params, grads, m, v, n, decayed_count, grad_norm, max_norm,
                                                           weight_decay_factor, beta1, beta2, eps, step_size,
                                                           inverse_sqrt_bias_correction2);
    cudaCheck(cudaGetLastError());
}

#endif // ADAMW_CUH
