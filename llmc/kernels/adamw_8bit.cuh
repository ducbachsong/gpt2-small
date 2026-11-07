// adamw_8bit.cuh — gradient clipping and one AdamW step for a bfloat16 model, with m and v in 8 bits (FP8 E4M3):
//
//     tensor          type in memory                    read with
//     params          bfloat16                          -: written, the master rounded
//     grads           bfloat16                          read4_as_float
//     master params   float                             reinterpret_cast
//     m, sqrt(v)      FP8 E4M3, a float scale per warp  reinterpret_cast<__nv_fp8x4_e4m3*>, x scale
//     in registers    float4                            all the math is in float

#ifndef ADAMW_8BIT_CUH
#define ADAMW_8BIT_CUH

#include <cuda_fp8.h>             // FP8 E4M3
#include "cuda_common.h"             // cudaCheck, warp_reduce_max
#include "../tensor.cuh"

#define ADAMW_8BIT_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)   // the default block size: 8 warps, so a multiple of THREADS_PER_WARP
#define ADAMW_VALUES_PER_SCALE (THREADS_PER_WARP * VALUES_PER_FLOAT4)   // one scale per warp, a float4 per thread
#define ADAMW_E4M3_LARGEST 448.0f           // largest E4M3 value
#define ADAMW_E4M3_SMALLEST 0.001953125f    // smallest E4M3 value above 0: 2^-9
#define FP8_PER_INT (sizeof(int) / sizeof(__nv_fp8_e4m3))   // 8-bit values one int holds, when an Int32 tensor holds them: 4

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
__device__ __forceinline__ float adamw_8bit_update_param(float p_val, float g_val, float* m_ptr, float* v_ptr, float clip_val,
                                                    float decay_factor_val, float beta1_val, float beta2_val, float eps_val,
                                                    float step_size_val, float inverse_sqrt_bias_correction2_val) {
    g_val = g_val * clip_val;
    p_val = p_val * decay_factor_val;
    *m_ptr = beta1_val * *m_ptr + (1.0f - beta1_val) * g_val;
    *v_ptr = beta2_val * *v_ptr + (1.0f - beta2_val) * g_val * g_val;
    float denominator_val = sqrtf(*v_ptr) * inverse_sqrt_bias_correction2_val + eps_val;
    return p_val - step_size_val * *m_ptr / denominator_val;
}

/// Gives sqrt(v) / scale, but never 0 when sqrt(v) is not: v = 0 would make the step m / eps, far too big.
///
///   sqrt_v_val     the new sqrt(v)
///   to_stored_val  1 / scale
__device__ __forceinline__ float adamw_sqrt_v_to_stored(float sqrt_v_val, float to_stored_val) {
    float stored_val = sqrt_v_val * to_stored_val;
    return sqrt_v_val > 0.0f ? fmaxf(stored_val, ADAMW_E4M3_SMALLEST) : stored_val;
}

/// Updates every value, one float4 at a time: reads m and sqrt(v) to floats, does the AdamW math, stores them
/// back with new scales. One warp = ADAMW_VALUES_PER_SCALE values = one scale, so blockDim.x must be a
/// multiple of THREADS_PER_WARP.
///
///   params_ptr                         bfloat16: written, the master rounded
///   grads_ptr                          bfloat16
///   m_ptr, m_scales_ptr                m / scale, 1 byte per value; one scale per ADAMW_VALUES_PER_SCALE values
///   sqrt_v_ptr, sqrt_v_scales_ptr      sqrt(v) / scale, 1 byte per value; one scale per ADAMW_VALUES_PER_SCALE values
///   params_in_slice_val                values to update: this GPU's slice (all of them on one GPU)
///   decayed_count_val                  the first this many values get weight decay (the matrices)
///   grad_norm_ptr                      norm of all the gradients
///   max_norm_val                       clip to this norm; 0: no clipping
///   weight_decay_factor_val            1 - learning_rate x weight_decay
///   beta1_val, beta2_val               how much of the old m and v stays: 0.9, 0.95
///   eps_val                            keeps the division from dividing by 0: 1e-8
///   step_size_val                      learning_rate / (1 - beta1^t)
///   inverse_sqrt_bias_correction2_val  1 / sqrt(1 - beta2^t)
///   master_params_ptr                  float copy of the params; nullptr: read the bfloat16 params
__global__ void adamw_8bit_kernel(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, __nv_fp8_e4m3* m_ptr, float* m_scales_ptr,
                                  __nv_fp8_e4m3* sqrt_v_ptr, float* sqrt_v_scales_ptr, size_t params_in_slice_val,
                                  size_t decayed_count_val, const float* grad_norm_ptr, float max_norm_val,
                                  float weight_decay_factor_val, float beta1_val, float beta2_val, float eps_val,
                                  float step_size_val, float inverse_sqrt_bias_correction2_val, float* master_params_ptr) {
    // Gradient clipping: scale g down so its norm is at most max_norm; 1 if already below it.
    float clip_val = 1.0f;
    if (max_norm_val > 0.0f) {
        clip_val = max_norm_val / (*grad_norm_ptr + 1e-6f);
        if (clip_val > 1.0f) clip_val = 1.0f;
    }
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t lane_index_val = threadIdx.x % THREADS_PER_WARP;   // this thread's place in its warp: 0 .. THREADS_PER_WARP - 1
    size_t float4_count_val = params_in_slice_val / VALUES_PER_FLOAT4;
    // Loops while the warp's first float4 (lane 0's) is in the slice, the same test for all its threads, so the
    // warp stays whole: the swaps below need all THREADS_PER_WARP threads. Each round jumps over the whole grid.
    for (size_t warp_first_index_val = thread_index_val - lane_index_val; warp_first_index_val < float4_count_val;
         warp_first_index_val += (size_t)gridDim.x * blockDim.x) {
        size_t index_val = warp_first_index_val + lane_index_val;   // this thread's float4
        bool in_slice_val = index_val < float4_count_val;           // false: no reads or writes, only the swaps
        size_t scale_index_val = warp_first_index_val * VALUES_PER_FLOAT4 / ADAMW_VALUES_PER_SCALE;   // one per warp
        float4 p_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 m_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // 1. un-zoom and update: last step's bytes x last step's scale = m and sqrt(v) in float, then AdamW
        if (in_slice_val) {
            // decay: matrices only
            float decay_factor_val = index_val * VALUES_PER_FLOAT4 < decayed_count_val ? weight_decay_factor_val : 1.0f;
            // params: float copy, else bfloat16
            p_val = master_params_ptr != nullptr ? reinterpret_cast<float4*>(master_params_ptr)[index_val]
                                                 : read4_as_float(params_ptr + VALUES_PER_FLOAT4 * index_val);
            // grads
            float4 g_val = read4_as_float(grads_ptr + VALUES_PER_FLOAT4 * index_val);
            // stored bytes, still zoomed
            float4 m_stored_val = (float4)reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val];
            float4 sqrt_v_stored_val = (float4)reinterpret_cast<__nv_fp8x4_e4m3*>(sqrt_v_ptr)[index_val];
            // the group's scales
            float m_scale_val = m_scales_ptr[scale_index_val];
            float sqrt_v_scale_val = sqrt_v_scales_ptr[scale_index_val];
            // m = stored x scale
            m_val = make_float4(m_stored_val.x * m_scale_val, m_stored_val.y * m_scale_val,
                                m_stored_val.z * m_scale_val, m_stored_val.w * m_scale_val);
            // sqrt(v) = stored x scale
            float4 old_sqrt_v_val = make_float4(sqrt_v_stored_val.x * sqrt_v_scale_val, sqrt_v_stored_val.y * sqrt_v_scale_val,
                                                sqrt_v_stored_val.z * sqrt_v_scale_val, sqrt_v_stored_val.w * sqrt_v_scale_val);
            // v = sqrt(v)²
            v_val = make_float4(old_sqrt_v_val.x * old_sqrt_v_val.x, old_sqrt_v_val.y * old_sqrt_v_val.y,
                                old_sqrt_v_val.z * old_sqrt_v_val.z, old_sqrt_v_val.w * old_sqrt_v_val.w);
            // AdamW
            p_val.x = adamw_8bit_update_param(p_val.x, g_val.x, &m_val.x, &v_val.x, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
            p_val.y = adamw_8bit_update_param(p_val.y, g_val.y, &m_val.y, &v_val.y, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
            p_val.z = adamw_8bit_update_param(p_val.z, g_val.z, &m_val.z, &v_val.z, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
            p_val.w = adamw_8bit_update_param(p_val.w, g_val.w, &m_val.w, &v_val.w, clip_val, decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val, inverse_sqrt_bias_correction2_val);
            // write params: float copy, bfloat16 rounded
            if (master_params_ptr != nullptr) reinterpret_cast<float4*>(master_params_ptr)[index_val] = p_val;
            write4_from_float(params_ptr + VALUES_PER_FLOAT4 * index_val, p_val);
        }
        // 2. the largest |m| and sqrt(v) of the warp's ADAMW_VALUES_PER_SCALE values: each thread's own 4, then the warp's
        // sqrt(v): what gets stored
        float4 sqrt_v_val = make_float4(sqrtf(v_val.x), sqrtf(v_val.y), sqrtf(v_val.z), sqrtf(v_val.w));
        // this thread's largest of 4
        float m_largest_val = fmaxf(fmaxf(fabsf(m_val.x), fabsf(m_val.y)), fmaxf(fabsf(m_val.z), fabsf(m_val.w)));
        float sqrt_v_largest_val = fmaxf(fmaxf(sqrt_v_val.x, sqrt_v_val.y), fmaxf(sqrt_v_val.z, sqrt_v_val.w));
        // the warp's largest of ADAMW_VALUES_PER_SCALE
        m_largest_val = warp_reduce_max(m_largest_val);
        sqrt_v_largest_val = warp_reduce_max(sqrt_v_largest_val);
        // 3. zoom and store: x 448 / largest, so the largest becomes 448, rounded to the nearest E4M3
        if (in_slice_val) {
            // zoom = 448 / largest; 0 if all 0
            float m_to_stored_val = m_largest_val > 0.0f ? ADAMW_E4M3_LARGEST / m_largest_val : 0.0f;
            float sqrt_v_to_stored_val = sqrt_v_largest_val > 0.0f ? ADAMW_E4M3_LARGEST / sqrt_v_largest_val : 0.0f;
            // m x zoom, to E4M3
            reinterpret_cast<__nv_fp8x4_e4m3*>(m_ptr)[index_val] = __nv_fp8x4_e4m3(
                make_float4(m_val.x * m_to_stored_val, m_val.y * m_to_stored_val, m_val.z * m_to_stored_val,
                            m_val.w * m_to_stored_val));
            // sqrt(v) x zoom, never rounded to 0
            reinterpret_cast<__nv_fp8x4_e4m3*>(sqrt_v_ptr)[index_val] = __nv_fp8x4_e4m3(
                make_float4(adamw_sqrt_v_to_stored(sqrt_v_val.x, sqrt_v_to_stored_val),
                            adamw_sqrt_v_to_stored(sqrt_v_val.y, sqrt_v_to_stored_val),
                            adamw_sqrt_v_to_stored(sqrt_v_val.z, sqrt_v_to_stored_val),
                            adamw_sqrt_v_to_stored(sqrt_v_val.w, sqrt_v_to_stored_val)));
        }
        // 4. the new scale, largest / 448, for the next step's un-zoom: one thread writes it
        if (lane_index_val == 0) {
            m_scales_ptr[scale_index_val] = m_largest_val / ADAMW_E4M3_LARGEST;
            sqrt_v_scales_ptr[scale_index_val] = sqrt_v_largest_val / ADAMW_E4M3_LARGEST;
        }
    }
}

/// One AdamW step on all the parameters: checks the launch, counts and addresses, then launches adamw_8bit_kernel.
///
///   params_ptr, grads_ptr           bfloat16: the parameters (matrices first) and their gradients
///   m_ptr, m_scales_ptr             m / scale, 1 byte per value; one scale per ADAMW_VALUES_PER_SCALE values; all 0 at the start
///   sqrt_v_ptr, sqrt_v_scales_ptr   sqrt(v) / scale, the same
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
///                                   ADAMW_8BIT_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void adamw_update_8bit(nv_bfloat16* params_ptr, const nv_bfloat16* grads_ptr, __nv_fp8_e4m3* m_ptr,
                              float* m_scales_ptr, __nv_fp8_e4m3* sqrt_v_ptr, float* sqrt_v_scales_ptr,
                              size_t params_in_slice_val, size_t decayed_count_val, const float* grad_norm_ptr,
                              float max_norm_val, float learning_rate_val, float beta1_val, float beta2_val,
                              float eps_val, float weight_decay_val, int t_val, float* master_params_ptr,
                              int threads_per_block_val = ADAMW_8BIT_THREADS_PER_BLOCK) {
    // 1. check the launch, counts and addresses; any of them wrong: stop with an error
    // block size: whole warps, as a warp is one scale group; at most MAX_THREADS_PER_BLOCK
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
    size_t fp8_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(__nv_fp8_e4m3);
    size_t float_float4_bytes_val = VALUES_PER_FLOAT4 * sizeof(float);
    if ((size_t)params_ptr % bfloat16_float4_bytes_val != 0 || (size_t)grads_ptr % bfloat16_float4_bytes_val != 0 ||
        (size_t)m_ptr % fp8_float4_bytes_val != 0 || (size_t)sqrt_v_ptr % fp8_float4_bytes_val != 0 ||
        (size_t)master_params_ptr % float_float4_bytes_val != 0) {
        fprintf(stderr, "error: AdamW needs params and grads to start on %zu bytes, 8-bit m and sqrt_v on %zu and "
                        "master_params on %zu (float4 loads); got %p, %p, %p, %p, %p\n", bfloat16_float4_bytes_val,
                fp8_float4_bytes_val, float_float4_bytes_val, (void*)params_ptr, (const void*)grads_ptr, (void*)m_ptr,
                (void*)sqrt_v_ptr, (void*)master_params_ptr);
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
    adamw_8bit_kernel<<<blocks_val, threads_per_block_val>>>(
        params_ptr, grads_ptr, m_ptr, m_scales_ptr, sqrt_v_ptr, sqrt_v_scales_ptr, params_in_slice_val, decayed_count_val,
        grad_norm_ptr, max_norm_val, weight_decay_factor_val, beta1_val, beta2_val, eps_val, step_size_val,
        inverse_sqrt_bias_correction2_val, master_params_ptr);
    cudaCheck(cudaGetLastError());
}

#endif // ADAMW_8BIT_CUH
