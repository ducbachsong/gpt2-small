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
// multiple of 4. The norm stays on the GPU (from global_norm.cuh).
#ifndef ADAMW_CUH
#define ADAMW_CUH

#include "cuda_common.h"

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

__global__ void adamw_kernel(float* params, const float* grads, float* m_memory, float* v_memory, size_t n,
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
    float4 p = reinterpret_cast<float4*>(params)[group];
    float4 g = reinterpret_cast<const float4*>(grads)[group];
    float4 m = reinterpret_cast<float4*>(m_memory)[group];
    float4 v = reinterpret_cast<float4*>(v_memory)[group];
    p.x = adamw_value(p.x, g.x, &m.x, &v.x, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.y = adamw_value(p.y, g.y, &m.y, &v.y, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.z = adamw_value(p.z, g.z, &m.z, &v.z, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    p.w = adamw_value(p.w, g.w, &m.w, &v.w, clip, decay_factor, beta1, beta2, eps, step_size, inverse_sqrt_bias_correction2);
    reinterpret_cast<float4*>(params)[group] = p;
    reinterpret_cast<float4*>(m_memory)[group] = m;
    reinterpret_cast<float4*>(v_memory)[group] = v;
}

/// One AdamW step t on n values in a row: all the parameters, matrices first. The first
/// decayed_count values get weight decay, the rest do not. max_norm 0: no clipping.
void adamw_update(float* params, const float* grads, float* m, float* v, size_t n, size_t decayed_count,
                  const float* grad_norm, float max_norm, float learning_rate, float beta1, float beta2, float eps,
                  float weight_decay, int t) {
    if (n % 4 != 0 || decayed_count % 4 != 0 || decayed_count > n) {
        fprintf(stderr, "error: AdamW needs n and decayed_count to be multiples of 4 (float4 loads), with "
                        "decayed_count <= n; got n = %zu, decayed_count = %zu\n", n, decayed_count);
        exit(EXIT_FAILURE);
    }
    // float4 loads need each block to start on 16 bytes (a multiple of 4 floats); off it,
    // the GPU stops with "misaligned address" at some later call, far from the cause.
    if ((size_t)params % 16 != 0 || (size_t)grads % 16 != 0 || (size_t)m % 16 != 0 || (size_t)v % 16 != 0) {
        fprintf(stderr, "error: AdamW needs params, grads, m and v to start on 16 bytes (float4 loads); got "
                        "%p, %p, %p, %p\n", (void*)params, (const void*)grads, (void*)m, (void*)v);
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
