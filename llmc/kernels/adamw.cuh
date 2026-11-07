// adamw.cuh — gradient clipping and one AdamW step, in one pass over all the parameters, for a float model:
//     g = g * clip                clip = min(1, max_norm / (norm + 1e-6))
//     p = p * (1 - lr wd)         weight decay, matrices only
//     m = b1 m + (1-b1) g         v = b2 v + (1-b2) g²
//     p = p - (lr / (1-b1ᵗ)) m / (sqrt(v) / sqrt(1-b2ᵗ) + eps)
//
//     tensor          type in memory   read with
//     params          float            reinterpret_cast
//     grads           float            reinterpret_cast
//     m, v            float            reinterpret_cast
//     in registers    float4           all the math is in float
// 28 bytes moved per value: reads 4 + 4 + 4 + 4, writes 4 + 4 + 4. A bfloat16 model: adamw_bf16.cuh.

#ifndef ADAMW_CUH
#define ADAMW_CUH

#include "cuda_common.h"
#include "../tensor.cuh"

#define ADAMW_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // the default block size: 8 warps

// CUDA's built-in values. A launch <<<blocks, threads>>> starts many threads in groups
// called blocks; every thread runs the same kernel and reads these to know which one it is:
//     gridDim.x     how many blocks there are
//     blockDim.x    how many threads are in each block
//     blockIdx.x    which block this thread is in: 0, 1, 2, ...
//     threadIdx.x   which thread it is inside its block: 0, 1, 2, ...

/// Moves one parameter one AdamW step: clips g, decays p, updates m and v, then steps p
/// against m, scaled by sqrt(v). Returns the new p; m and v are written back through their pointers.
///
///   p_val                              the parameter
///   g_val                              its gradient
///   m_ptr                              its first moment m: read, then written back
///   v_ptr                              its second moment v: read, then written back
///   clip_val                           g is multiplied by it: min(1, max_norm / norm)
///   decay_factor_val                   p is multiplied by it: weight_decay_factor for a matrix, 1 otherwise
///   beta1_val, beta2_val               how much of the old m and v stays: 0.9, 0.95
///   eps_val                            keeps the division from dividing by 0: 1e-8
///   step_size_val                      learning_rate / (1 - beta1^t)
///   inverse_sqrt_bias_correction2_val  1 / sqrt(1 - beta2^t)
__device__ __forceinline__ float adamw_update_param(float p_val, float g_val, float* m_ptr, float* v_ptr, float clip_val,
                                                    float decay_factor_val, float beta1_val, float beta2_val, float eps_val,
                                                    float step_size_val, float inverse_sqrt_bias_correction2_val) {
    g_val = g_val * clip_val;
    p_val = p_val * decay_factor_val;
    *m_ptr = beta1_val * *m_ptr + (1.0f - beta1_val) * g_val;
    *v_ptr = beta2_val * *v_ptr + (1.0f - beta2_val) * g_val * g_val;
    float denominator_val = sqrtf(*v_ptr) * inverse_sqrt_bias_correction2_val + eps_val;
    return p_val - step_size_val * *m_ptr / denominator_val;
}

/// Updates every value, one float4 at a time: reads p, g, m and v, does the AdamW math, writes p, m and v back.
///
///   params_ptr                         the parameters
///   grads_ptr                          their gradients
///   m_ptr                              first moment m: running average of g
///   v_ptr                              second moment v: running average of g²
///   params_in_slice_val                values to update: this GPU's slice (all of them on one GPU)
///   decayed_count_val                  the first this many values get weight decay (the matrices)
///   grad_norm_ptr                      norm of all the gradients
///   max_norm_val                       clip to this norm; 0: no clipping
///   weight_decay_factor_val            1 - learning_rate x weight_decay
///   beta1_val, beta2_val               how much of the old m and v stays: 0.9, 0.95
///   eps_val                            keeps the division from dividing by 0: 1e-8
///   step_size_val                      learning_rate / (1 - beta1^t)
///   inverse_sqrt_bias_correction2_val  1 / sqrt(1 - beta2^t)
__global__ void adamw_kernel(float* params_ptr, const float* grads_ptr, float* m_ptr, float* v_ptr,
                             size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr,
                             float max_norm_val, float weight_decay_factor_val, float beta1_val, float beta2_val,
                             float eps_val, float step_size_val, float inverse_sqrt_bias_correction2_val) {
    // Gradient clipping: scale g down so its norm is at most max_norm; 1 if already below it.
    float clip_val = 1.0f;
    if (max_norm_val > 0.0f) {
        clip_val = max_norm_val / (*grad_norm_ptr + 1e-6f);
        if (clip_val > 1.0f) clip_val = 1.0f;
    }
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < params_in_slice_val / VALUES_PER_FLOAT4;
         index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. read p, g, m and v
        // decay: matrices only
        float decay_factor_val = index_val * VALUES_PER_FLOAT4 < decayed_count_val ? weight_decay_factor_val : 1.0f;
        float4 p_val = reinterpret_cast<float4*>(params_ptr)[index_val];
        float4 g_val = reinterpret_cast<const float4*>(grads_ptr)[index_val];
        float4 m_val = reinterpret_cast<float4*>(m_ptr)[index_val];
        float4 v_val = reinterpret_cast<float4*>(v_ptr)[index_val];
        // 2. AdamW
        p_val.x = adamw_update_param(p_val.x, g_val.x, &m_val.x, &v_val.x, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.y = adamw_update_param(p_val.y, g_val.y, &m_val.y, &v_val.y, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.z = adamw_update_param(p_val.z, g_val.z, &m_val.z, &v_val.z, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.w = adamw_update_param(p_val.w, g_val.w, &m_val.w, &v_val.w, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        // 3. write p, m and v back
        reinterpret_cast<float4*>(params_ptr)[index_val] = p_val;
        reinterpret_cast<float4*>(m_ptr)[index_val] = m_val;
        reinterpret_cast<float4*>(v_ptr)[index_val] = v_val;
    }
}

/// One AdamW step on all the parameters: checks the launch, counts and addresses, then launches adamw_kernel.
///
///   params_ptr, grads_ptr           the parameters (matrices first) and their gradients
///   m_ptr, v_ptr                    first and second moments
///   params_in_slice_val             values to update: this GPU's slice
///   decayed_count_val               the first this many values get weight decay (the matrices)
///   grad_norm_ptr                   norm of all the gradients
///   max_norm_val                    clip to this norm; 0: no clipping
///   learning_rate_val               the learning rate of this step
///   beta1_val, beta2_val, eps_val   0.9, 0.95, 1e-8
///   weight_decay_val                0.1
///   t_val                           the step number: 1, 2, 3, ...
///   threads_per_block_val           the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                                   ADAMW_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void adamw_update(float* params_ptr, const float* grads_ptr, float* m_ptr, float* v_ptr, size_t params_in_slice_val,
                         size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val, float learning_rate_val,
                         float beta1_val, float beta2_val, float eps_val, float weight_decay_val, int t_val,
                         int threads_per_block_val = ADAMW_THREADS_PER_BLOCK) {
    // 1. check the launch, counts and addresses; any of them wrong: stop with an error
    // block size: whole warps, at most MAX_THREADS_PER_BLOCK
    check_threads_per_block(threads_per_block_val, "AdamW");
    // counts: whole float4s, the decayed ones inside the slice
    if (params_in_slice_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val % VALUES_PER_FLOAT4 != 0 || decayed_count_val > params_in_slice_val) {
        fprintf(stderr, "error: AdamW needs params_in_slice and decayed_count to be multiples of %d (float4 loads), with "
                        "decayed_count <= params_in_slice; got params_in_slice = %zu, decayed_count = %zu\n",
                VALUES_PER_FLOAT4, params_in_slice_val, decayed_count_val);
        exit(EXIT_FAILURE);
    }
    // addresses: a float4 load needs a start on a whole float4
    size_t float4_bytes_val = sizeof(float4);
    if ((size_t)params_ptr % float4_bytes_val != 0 || (size_t)grads_ptr % float4_bytes_val != 0 ||
        (size_t)m_ptr % float4_bytes_val != 0 || (size_t)v_ptr % float4_bytes_val != 0) {
        fprintf(stderr, "error: AdamW needs params, grads, m and v to start on %zu bytes (float4 loads); got %p, %p, %p, %p\n",
                float4_bytes_val, (void*)params_ptr, (const void*)grads_ptr, (void*)m_ptr, (void*)v_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. this step's numbers, the same for every value: worked out once here, not per thread
    // bias corrections: m and v start at 0, so early m and v are too small; dividing by 1 - beta^t undoes it
    float bias_correction1_val = 1.0f - powf(beta1_val, (float)t_val);
    float bias_correction2_val = 1.0f - powf(beta2_val, (float)t_val);
    // step size = learning_rate / (1 - beta1^t)
    float step_size_val = learning_rate_val / bias_correction1_val;
    // 1 / sqrt(1 - beta2^t)
    float inverse_sqrt_bias_correction2_val = 1.0f / sqrtf(bias_correction2_val);
    // decay: p x (1 - learning_rate x weight_decay)
    float weight_decay_factor_val = 1.0f - learning_rate_val * weight_decay_val;
    // 3. launch: one float4 per thread, the blocks rounded up
    int blocks_val = (int)CEIL_DIV(params_in_slice_val / VALUES_PER_FLOAT4, threads_per_block_val);
    adamw_kernel<<<blocks_val, threads_per_block_val>>>(
        params_ptr, grads_ptr, m_ptr, v_ptr, params_in_slice_val, decayed_count_val, grad_norm_ptr, max_norm_val,
        weight_decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
    cudaCheck(cudaGetLastError());
}

#endif // ADAMW_CUH
