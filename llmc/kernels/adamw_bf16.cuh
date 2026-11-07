// adamw_bf16.cuh — gradient clipping and one AdamW step for a bfloat16 model, with m and v in bfloat16 too
// (DeepSeek-V3's way). The math is adamw.cuh's, in float; each value is rounded to bfloat16 only as it is stored.
//
//     tensor          type in memory   read with          why
//     params          bfloat16         -                  the forward reads them: the master, rounded
//     grads           bfloat16         read4_as_float     made fresh each step, used once: rounding does not add up
//     master params   float            reinterpret_cast   adds up small updates over thousands of steps
//     m, v            bfloat16         read4_as_float     running averages: they forget, so rounding stays bounded
//     in registers    float4           -                  all the math is in float
// 20 bytes moved per value: reads 4 + 2 + 2 + 2, writes 4 + 2 + 2 + 2 (adamw.cuh with float m, v: 28).

#ifndef ADAMW_BF16_CUH
#define ADAMW_BF16_CUH

#include "cuda_common.h"
#include "../tensor.cuh"

#define ADAMW_BF16_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // the default block size: 8 warps

/// Moves one parameter one AdamW step: clips g, decays p, updates m and v, then steps p against m, scaled by
/// sqrt(v). Returns the new p; m and v are written back through their pointers.
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
__device__ __forceinline__ float adamw_bf16_update_param(float p_val, float g_val, float* m_ptr, float* v_ptr, float clip_val,
                                                    float decay_factor_val, float beta1_val, float beta2_val, float eps_val,
                                                    float step_size_val, float inverse_sqrt_bias_correction2_val) {
    g_val = g_val * clip_val;
    p_val = p_val * decay_factor_val;
    *m_ptr = beta1_val * *m_ptr + (1.0f - beta1_val) * g_val;
    *v_ptr = beta2_val * *v_ptr + (1.0f - beta2_val) * g_val * g_val;
    float denominator_val = sqrtf(*v_ptr) * inverse_sqrt_bias_correction2_val + eps_val;
    return p_val - step_size_val * *m_ptr / denominator_val;
}

/// Updates every value, one float4 at a time: reads m and v to floats, does the AdamW math, stores them back
/// rounded.
///
///   params_ptr                         bfloat16: written, the master rounded
///   grads_ptr                          bfloat16
///   m_ptr, v_ptr                       bfloat16: read, then written back
///   params_in_slice_val                values to update: this GPU's slice (all of them on one GPU)
///   decayed_count_val                  the first this many values get weight decay (the matrices)
///   grad_norm_ptr                      norm of all the gradients
///   max_norm_val                       clip to this norm; 0: no clipping
///   weight_decay_factor_val            1 - learning_rate x weight_decay
///   beta1_val, beta2_val               how much of the old m and v stays: 0.9, 0.95
///   eps_val                            keeps the division from dividing by 0: 1e-8
///   step_size_val                      learning_rate / (1 - beta1^t)
///   inverse_sqrt_bias_correction2_val  1 / sqrt(1 - beta2^t)
///   master_params_ptr                  float copy of the params; nullptr: read the bfloat16 params (updates get lost)
__global__ void adamw_bf16_kernel(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr,
                                  nv_bfloat16* v_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                                  const float* grad_norm_ptr, float max_norm_val, float weight_decay_factor_val,
                                  float beta1_val, float beta2_val, float eps_val, float step_size_val,
                                  float inverse_sqrt_bias_correction2_val, float* master_params_ptr) {
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
        size_t first_val = VALUES_PER_FLOAT4 * index_val;    // this float4's first value
        // 1. read: params from the float copy (else bfloat16), g, m and v, all to float
        // decay: matrices only
        float decay_factor_val = first_val < decayed_count_val ? weight_decay_factor_val : 1.0f;
        float4 p_val = master_params_ptr != nullptr ? reinterpret_cast<float4*>(master_params_ptr)[index_val]
                                                    : read4_as_float(params_ptr + first_val);
        float4 g_val = read4_as_float(grads_ptr + first_val);
        float4 m_val = read4_as_float(m_ptr + first_val);
        float4 v_val = read4_as_float(v_ptr + first_val);
        // 2. AdamW, in float
        p_val.x = adamw_bf16_update_param(p_val.x, g_val.x, &m_val.x, &v_val.x, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.y = adamw_bf16_update_param(p_val.y, g_val.y, &m_val.y, &v_val.y, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.z = adamw_bf16_update_param(p_val.z, g_val.z, &m_val.z, &v_val.z, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        p_val.w = adamw_bf16_update_param(p_val.w, g_val.w, &m_val.w, &v_val.w, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
        // 3. write: the float copy exact, the rest rounded to bfloat16
        if (master_params_ptr != nullptr) reinterpret_cast<float4*>(master_params_ptr)[index_val] = p_val;
        write4_from_float(params_ptr + first_val, p_val);
        write4_from_float(m_ptr + first_val, m_val);
        write4_from_float(v_ptr + first_val, v_val);
    }
}

/// One AdamW step on all the parameters: checks the launch, counts and addresses, then launches adamw_bf16_kernel.
///
///   params_ptr, grads_ptr           bfloat16: the parameters (matrices first) and their gradients
///   m_ptr, v_ptr                    bfloat16; all 0 at the start
///   params_in_slice_val             values to update: this GPU's slice
///   decayed_count_val               the first this many values get weight decay (the matrices)
///   grad_norm_ptr                   norm of all the gradients
///   max_norm_val                    clip to this norm; 0: no clipping
///   learning_rate_val               the learning rate of this step
///   beta1_val, beta2_val, eps_val   0.9, 0.95, 1e-8
///   weight_decay_val                0.1
///   t_val                           the step number: 1, 2, 3, ...
///   master_params_ptr               float copy of the params; nullptr: none
///   threads_per_block_val           the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                                   ADAMW_BF16_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void adamw_update_bf16(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, nv_bfloat16* m_ptr,
                              nv_bfloat16* v_ptr, size_t params_in_slice_val, size_t decayed_count_val,
                              const float* grad_norm_ptr, float max_norm_val, float learning_rate_val, float beta1_val,
                              float beta2_val, float eps_val, float weight_decay_val, int t_val,
                              float* master_params_ptr, int threads_per_block_val = ADAMW_BF16_THREADS_PER_BLOCK) {
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
    // addresses: a float4 load needs a start on a whole float4 of its type
    size_t bfloat16_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(nv_bfloat16);
    size_t float_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(float);
    if ((size_t)params_ptr % bfloat16_float4_bytes_val != 0 || (size_t)grads_ptr % bfloat16_float4_bytes_val != 0 ||
        (size_t)m_ptr % bfloat16_float4_bytes_val != 0 || (size_t)v_ptr % bfloat16_float4_bytes_val != 0 ||
        (size_t)master_params_ptr % float_float4_bytes_val != 0) {
        fprintf(stderr, "error: AdamW needs params, grads, m and v to start on %zu bytes and master_params on %zu "
                        "(float4 loads); got %p, %p, %p, %p, %p\n", bfloat16_float4_bytes_val, float_float4_bytes_val,
                (void*)params_ptr, (const void*)grads_ptr, (void*)m_ptr, (void*)v_ptr, (void*)master_params_ptr);
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
    adamw_bf16_kernel<<<blocks_val, threads_per_block_val>>>(
        params_ptr, grads_ptr, m_ptr, v_ptr, params_in_slice_val, decayed_count_val, grad_norm_ptr, max_norm_val,
        weight_decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val,
        master_params_ptr);
    cudaCheck(cudaGetLastError());
}

#endif // ADAMW_BF16_CUH
