// attention_8bit.cuh — query, key and value of a bfloat16 model kept in 8 bits (FP8 E4M3, one float scale per
// head row): the rounding (attention_8bit_round_qkv) and the turning back to bfloat16 (attention_8bit_turn_back).
// The attention that uses them, on an H100's tensor cores: attention_flash_8bit.cuh.
//
//     tensor                      type in memory                  shape                                           why
//     qkv                         bfloat16                        {batch_size, seq_len, 3 · embed_dim}            the model's
//     query_8bit, key_8bit,       E4M3, 4 in each int of an       {batch_size, num_heads, seq_len,                half of bfloat16's bytes
//     value_8bit                  Int32 tensor                     head_size / ATTENTION_8BIT_VALUES_PER_INT}
//     query_scales, key_scales,   float                           {batch_size, num_heads, seq_len}                one per head row
//     value_scales
//     qkv_rounded                 bfloat16                        {batch_size, seq_len, 3 · embed_dim}            the 8-bit ones turned back,
//                                                                                                                 where qkv has them
//
// Rounding: each head row (head_size values of one token) is divided by its scale, the smallest power of 2 with
//     largest |value| / scale <= HARDWARE_E4M3_LARGEST (448)
// and rounded to the nearest E4M3: by up to 2^-4 (6%) of a value, against bfloat16's 2^-8. The scale is a power of 2
// so that turning back is exact: E4M3 x 2^k has 4 significant bits, which bfloat16's 8 hold.
// Needs head_size ATTENTION_8BIT_HEAD_SIZE; any GPU (only the attention's products need E4M3 tensor cores).
#ifndef ATTENTION_8BIT_CUH
#define ATTENTION_8BIT_CUH

#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_max
#include "../../common/tensor.cuh"

#define ATTENTION_8BIT_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)            // the rounding's and the turn-back's default: one warp per row, one int per thread
#define ATTENTION_8BIT_HEAD_SIZE 128                                                // the head_size the kernels take: 64 (GPT-2 small's) or 128 (the 32 x 2,304 model's, 18 heads)
#define ATTENTION_8BIT_VALUES_PER_INT (sizeof(int) / sizeof(__nv_fp8_e4m3))         // E4M3s in one int: an Int32 tensor's
#define ATTENTION_8BIT_PAIRS_PER_LANE (ATTENTION_8BIT_HEAD_SIZE / (2 * HARDWARE_THREADS_PER_WARP))   // the rounding: pairs of a head row each lane takes, 1 or 2
static_assert(ATTENTION_8BIT_HEAD_SIZE % (2 * HARDWARE_THREADS_PER_WARP) == 0, "the rounding: each lane takes whole pairs of a head row");
static_assert(ATTENTION_8BIT_VALUES_PER_INT == VALUES_PER_FLOAT4, "the turn-back takes an int of E4M3s as one float4");

// ── the rounding and the turning back ────────────────────────────────────────────────────────────
void attention_8bit_round_qkv(const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                              const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                              const Tensor& value_scales_tensor, const Tensor& qkv_tensor,
                              int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK);
void attention_8bit_turn_back(const Tensor& qkv_rounded_tensor, const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor,
                              const Tensor& value_8bit_tensor, const Tensor& query_scales_tensor,
                              const Tensor& key_scales_tensor, const Tensor& value_scales_tensor,
                              int threads_per_block_val = ATTENTION_8BIT_THREADS_PER_BLOCK);

// ── for both ────────────────────────────────────────────────────────────
void attention_8bit_check_copies(const Tensor& qkv_tensor, const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor,
                                 const Tensor& value_8bit_tensor, const Tensor& query_scales_tensor,
                                 const Tensor& key_scales_tensor, const Tensor& value_scales_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the rounding ────────────────────────────────────────────────────────────

__global__ void attention_8bit_round_qkv_kernel(__nv_fp8_e4m3* query_8bit_ptr, __nv_fp8_e4m3* key_8bit_ptr,
                                                __nv_fp8_e4m3* value_8bit_ptr, float* query_scales_ptr,
                                                float* key_scales_ptr, float* value_scales_ptr,
                                                const nv_bfloat16* qkv_ptr, int num_heads_val, int seq_len_val,
                                                size_t rows_val);

/// Rounds query, key and value to E4M3, each head row with its own scale: checks, then launches
/// attention_8bit_round_qkv_kernel, one warp per head row. F1 of attention_flash_8bit.cuh's forward.
///
///   query_8bit_tensor, key_8bit_tensor,
///   value_8bit_tensor                            {batch_size, num_heads, seq_len, head_size / ATTENTION_8BIT_VALUES_PER_INT}
///                                                Int32, ATTENTION_8BIT_VALUES_PER_INT E4M3s in each int: written
///   query_scales_tensor, key_scales_tensor,
///   value_scales_tensor                          {batch_size, num_heads, seq_len} float: written
///   qkv_tensor                                   {batch_size, seq_len, 3 · embed_dim} bfloat16: the qkv linear's output
///   threads_per_block_val                        the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                                                given: ATTENTION_8BIT_THREADS_PER_BLOCK. Any size gives the same bits;
///                                                only the time changes.
inline void attention_8bit_round_qkv(const Tensor& query_8bit_tensor, const Tensor& key_8bit_tensor,
                                     const Tensor& value_8bit_tensor, const Tensor& query_scales_tensor,
                                     const Tensor& key_scales_tensor, const Tensor& value_scales_tensor,
                                     const Tensor& qkv_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "8-bit attention's rounding");
    attention_8bit_check_copies(qkv_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor,
                                key_scales_tensor, value_scales_tensor);
    // 2. launch: one warp per head row
    size_t num_heads_val = query_8bit_tensor.size(1);
    size_t seq_len_val = query_8bit_tensor.size(2);
    size_t rows_val = query_8bit_tensor.size(0) * num_heads_val * seq_len_val;          // one head row per (batch, head, token)
    attention_8bit_round_qkv_kernel<<<(int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val),
                                      threads_per_block_val>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(query_8bit_tensor.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(key_8bit_tensor.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(value_8bit_tensor.data_ptr()), query_scales_tensor.data(),
        key_scales_tensor.data(), value_scales_tensor.data(), qkv_tensor.data_ptr<nv_bfloat16>(), (int)num_heads_val,
        (int)seq_len_val, rows_val);
    cudaCheck(cudaGetLastError());
}

/// Gives a head row's scale: the smallest power of 2 with row_largest / scale <= HARDWARE_E4M3_LARGEST; 1 for a
/// row of 0s. The row's largest |value| / scale then lands in (HARDWARE_E4M3_LARGEST / 2, HARDWARE_E4M3_LARGEST].
///
///   row_largest_val   the row's largest |value|
__device__ __forceinline__ float attention_8bit_scale(float row_largest_val) {
    // 1. a row of 0s: any scale; 1
    if (row_largest_val == 0.0f) return 1.0f;
    // 2. ratio = mantissa x 2^exponent, mantissa in [0.5, 1): the power of 2 at or above it is 2^(exponent - 1)
    //    when mantissa is exactly 0.5 (ratio a power of 2 itself), else 2^exponent
    int exponent_val = 0;
    float mantissa_val = frexpf(row_largest_val / HARDWARE_E4M3_LARGEST, &exponent_val);
    return ldexpf(1.0f, mantissa_val == 0.5f ? exponent_val - 1 : exponent_val);
}

/// F1: splits qkv into heads and rounds each head row of query, key and value to E4M3, one warp per head row:
/// the row's largest |value| (warp_reduce_max), its scale, then value / scale rounded. Each lane takes
/// ATTENTION_8BIT_PAIRS_PER_LANE pairs of neighbouring columns, the pairs a warp's width apart.
///
///   query_8bit_ptr, key_8bit_ptr, value_8bit_ptr        {batch_size, num_heads, seq_len, ATTENTION_8BIT_HEAD_SIZE} E4M3: written
///   query_scales_ptr, key_scales_ptr, value_scales_ptr  one float per head row: written
///   qkv_ptr                                             {batch_size, seq_len, 3 · embed_dim} bfloat16
///   num_heads_val, seq_len_val                          the shape
///   rows_val                                            batch_size x num_heads x seq_len
__global__ void attention_8bit_round_qkv_kernel(__nv_fp8_e4m3* query_8bit_ptr, __nv_fp8_e4m3* key_8bit_ptr,
                                                __nv_fp8_e4m3* value_8bit_ptr, float* query_scales_ptr,
                                                float* key_scales_ptr, float* value_scales_ptr,
                                                const nv_bfloat16* qkv_ptr, int num_heads_val, int seq_len_val,
                                                size_t rows_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t embed_dim_val = (size_t)num_heads_val * ATTENTION_8BIT_HEAD_SIZE;
    __nv_fp8_e4m3* rounded_list[3] = {query_8bit_ptr, key_8bit_ptr, value_8bit_ptr};
    float* scales_list[3] = {query_scales_ptr, key_scales_ptr, value_scales_ptr};
    // Warp n takes head rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        // 1. where the row is: which (batch, head), which token; where its query part sits in qkv
        size_t batch_head_index_val = row_index_val / seq_len_val;                    // batch x num_heads + head
        size_t token_index_val = row_index_val % seq_len_val;
        size_t batch_index_val = batch_head_index_val / num_heads_val;
        size_t head_index_val = batch_head_index_val % num_heads_val;
        const nv_bfloat16* query_part_ptr =
            qkv_ptr + (batch_index_val * seq_len_val + token_index_val) * 3 * embed_dim_val + head_index_val * ATTENTION_8BIT_HEAD_SIZE;
        size_t row_first_index_val = row_index_val * ATTENTION_8BIT_HEAD_SIZE;           // the row's first value, per head
        // 2. query, key, value in turn, embed_dim apart in qkv: the row's largest |value|, its scale, the values rounded
#pragma unroll
        for (int part_index_val = 0; part_index_val < 3; part_index_val++) {
            float2 values_list[ATTENTION_8BIT_PAIRS_PER_LANE];
            float lane_largest_val = 0.0f;
#pragma unroll
            for (int pair_index_val = 0; pair_index_val < ATTENTION_8BIT_PAIRS_PER_LANE; pair_index_val++) {
                int column_index_val = 2 * (pair_index_val * HARDWARE_THREADS_PER_WARP + lane_index_val);
                values_list[pair_index_val] = __bfloat1622float2(
                    *reinterpret_cast<const nv_bfloat162*>(query_part_ptr + part_index_val * embed_dim_val + column_index_val));
                lane_largest_val = fmaxf(lane_largest_val, fmaxf(fabsf(values_list[pair_index_val].x), fabsf(values_list[pair_index_val].y)));
            }
            float row_largest_val = warp_reduce_max(lane_largest_val);
            float scale_val = attention_8bit_scale(row_largest_val);
            float to_stored_val = 1.0f / scale_val;                             // exact: a power of 2
#pragma unroll
            for (int pair_index_val = 0; pair_index_val < ATTENTION_8BIT_PAIRS_PER_LANE; pair_index_val++) {
                int column_index_val = 2 * (pair_index_val * HARDWARE_THREADS_PER_WARP + lane_index_val);
                *reinterpret_cast<__nv_fp8x2_e4m3*>(rounded_list[part_index_val] + row_first_index_val + column_index_val) =
                    __nv_fp8x2_e4m3(make_float2(values_list[pair_index_val].x * to_stored_val, values_list[pair_index_val].y * to_stored_val));
            }
            if (lane_index_val == 0) scales_list[part_index_val][row_index_val] = scale_val;
        }
    }
}

// ── the turning back ────────────────────────────────────────────────────────────

__global__ void attention_8bit_turn_back_kernel(nv_bfloat16* qkv_rounded_ptr, const __nv_fp8_e4m3* query_8bit_ptr,
                                                const __nv_fp8_e4m3* key_8bit_ptr, const __nv_fp8_e4m3* value_8bit_ptr,
                                                const float* query_scales_ptr, const float* key_scales_ptr,
                                                const float* value_scales_ptr, int num_heads_val, int seq_len_val,
                                                size_t total_ints_val);

/// Writes query, key and value in bfloat16 from their 8-bit copies, where qkv has them: each E4M3 times its
/// row's scale, exact. Used by attention_flash_8bit.cuh's backward; the tests call it to see the forward's rounded values.
///
///   qkv_rounded_tensor                                 {batch_size, seq_len, 3 · embed_dim} bfloat16: written
///   query_8bit_tensor, key_8bit_tensor,
///   value_8bit_tensor                                  {batch_size, num_heads, seq_len, head_size / ATTENTION_8BIT_VALUES_PER_INT} Int32
///   query_scales_tensor, key_scales_tensor,
///   value_scales_tensor                                {batch_size, num_heads, seq_len} float
///   threads_per_block_val                              the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                                                      given: ATTENTION_8BIT_THREADS_PER_BLOCK
inline void attention_8bit_turn_back(const Tensor& qkv_rounded_tensor, const Tensor& query_8bit_tensor,
                                     const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                     const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                     const Tensor& value_scales_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "8-bit attention's turn-back");
    attention_8bit_check_copies(qkv_rounded_tensor, query_8bit_tensor, key_8bit_tensor, value_8bit_tensor,
                                query_scales_tensor, key_scales_tensor, value_scales_tensor);
    // 2. launch: one int (ATTENTION_8BIT_VALUES_PER_INT values) of each per thread
    size_t total_ints_val = query_8bit_tensor.numel();
    attention_8bit_turn_back_kernel<<<(int)CEIL_DIV(total_ints_val, (size_t)threads_per_block_val), threads_per_block_val>>>(
        qkv_rounded_tensor.data_ptr<nv_bfloat16>(), reinterpret_cast<const __nv_fp8_e4m3*>(query_8bit_tensor.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(key_8bit_tensor.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(value_8bit_tensor.data_ptr()), query_scales_tensor.data(),
        key_scales_tensor.data(), value_scales_tensor.data(), (int)query_8bit_tensor.size(1),
        (int)query_8bit_tensor.size(2), total_ints_val);
    cudaCheck(cudaGetLastError());
}

/// Turns ATTENTION_8BIT_VALUES_PER_INT E4M3s of query, key and value back into bfloat16 per thread, each times
/// its head row's scale, and writes them where qkv has them.
///
///   qkv_rounded_ptr                                       {batch_size, seq_len, 3 · embed_dim} bfloat16: written
///   query_8bit_ptr, key_8bit_ptr, value_8bit_ptr          {batch_size, num_heads, seq_len, ATTENTION_8BIT_HEAD_SIZE} E4M3
///   query_scales_ptr, key_scales_ptr, value_scales_ptr    one float per head row
///   num_heads_val, seq_len_val                            the shape
///   total_ints_val                                        an 8-bit tensor's values / ATTENTION_8BIT_VALUES_PER_INT
__global__ void attention_8bit_turn_back_kernel(nv_bfloat16* qkv_rounded_ptr, const __nv_fp8_e4m3* query_8bit_ptr,
                                                const __nv_fp8_e4m3* key_8bit_ptr, const __nv_fp8_e4m3* value_8bit_ptr,
                                                const float* query_scales_ptr, const float* key_scales_ptr,
                                                const float* value_scales_ptr, int num_heads_val, int seq_len_val,
                                                size_t total_ints_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t embed_dim_val = (size_t)num_heads_val * ATTENTION_8BIT_HEAD_SIZE;
    const __nv_fp8_e4m3* rounded_list[3] = {query_8bit_ptr, key_8bit_ptr, value_8bit_ptr};
    const float* scales_list[3] = {query_scales_ptr, key_scales_ptr, value_scales_ptr};
    // Thread n takes ints n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_ints_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. which head row the int is in, and where its query part sits in qkv; key and value follow, embed_dim apart
        size_t value_index_val = index_val * ATTENTION_8BIT_VALUES_PER_INT;          // its first value, per head
        size_t row_index_val = value_index_val / ATTENTION_8BIT_HEAD_SIZE;                    // (batch x num_heads + head) x seq_len + token
        size_t column_index_val = value_index_val % ATTENTION_8BIT_HEAD_SIZE;
        size_t batch_head_index_val = row_index_val / seq_len_val;
        size_t token_index_val = row_index_val % seq_len_val;
        size_t batch_index_val = batch_head_index_val / num_heads_val;
        size_t head_index_val = batch_head_index_val % num_heads_val;
        nv_bfloat16* query_part_ptr = qkv_rounded_ptr + (batch_index_val * seq_len_val + token_index_val) * 3 * embed_dim_val +
                                      head_index_val * ATTENTION_8BIT_HEAD_SIZE + column_index_val;
        // 2. each of the three: ATTENTION_8BIT_VALUES_PER_INT E4M3s as a float4, times the row's scale, written as bfloat16s
#pragma unroll
        for (int part_index_val = 0; part_index_val < 3; part_index_val++) {
            float4 stored_val = (float4)reinterpret_cast<const __nv_fp8x4_e4m3*>(rounded_list[part_index_val])[index_val];
            float scale_val = scales_list[part_index_val][row_index_val];
            write4_from_float(query_part_ptr + part_index_val * embed_dim_val,
                              make_float4(stored_val.x * scale_val, stored_val.y * scale_val, stored_val.z * scale_val,
                                          stored_val.w * scale_val));
        }
    }
}

// ── for both ────────────────────────────────────────────────────────────

/// Stops with a message unless the rounding or the turn-back can run these tensors: qkv's shape from the 8-bit
/// copies', the types, head_size ATTENTION_8BIT_HEAD_SIZE, qkv starting on a float4 of bfloat16s.
///
///   qkv_tensor                                   the rounding: qkv; the turn-back: qkv_rounded
///   query_8bit_tensor, key_8bit_tensor, value_8bit_tensor, query_scales_tensor, key_scales_tensor,
///   value_scales_tensor                          as in attention_8bit_round_qkv
inline void attention_8bit_check_copies(const Tensor& qkv_tensor, const Tensor& query_8bit_tensor,
                                        const Tensor& key_8bit_tensor, const Tensor& value_8bit_tensor,
                                        const Tensor& query_scales_tensor, const Tensor& key_scales_tensor,
                                        const Tensor& value_scales_tensor) {
    // 1. shapes, types and addresses together
    bool fits_val = query_8bit_tensor.dim() == 4;
    if (fits_val) {
        size_t head_size_val = query_8bit_tensor.size(3) * ATTENTION_8BIT_VALUES_PER_INT;
        Shape per_token_list = {query_8bit_tensor.size(0), query_8bit_tensor.size(2),
                                3 * query_8bit_tensor.size(1) * head_size_val};          // qkv's
        Shape per_row_list = {query_8bit_tensor.size(0), query_8bit_tensor.size(1), query_8bit_tensor.size(2)};
        fits_val = qkv_tensor.shape() == per_token_list && qkv_tensor.dtype() == DType::BFloat16 &&
                   (size_t)qkv_tensor.data_ptr() % (VALUES_PER_FLOAT4 * sizeof(nv_bfloat16)) == 0;
        for (const Tensor* tensor_ptr : {&key_8bit_tensor, &value_8bit_tensor})
            fits_val = fits_val && tensor_ptr->shape() == query_8bit_tensor.shape();
        for (const Tensor* tensor_ptr : {&query_8bit_tensor, &key_8bit_tensor, &value_8bit_tensor})
            fits_val = fits_val && tensor_ptr->dtype() == DType::Int32;
        for (const Tensor* tensor_ptr : {&query_scales_tensor, &key_scales_tensor, &value_scales_tensor})
            fits_val = fits_val && tensor_ptr->shape() == per_row_list && tensor_ptr->dtype() == DType::Float32;
        fits_val = fits_val && head_size_val == ATTENTION_8BIT_HEAD_SIZE;
    }
    // 2. any of them wrong: stop with an error
    if (!fits_val) {
        fprintf(stderr, "error: 8-bit attention's rounding and turn-back need a bfloat16 qkv {batch_size, seq_len, 3 * "
                        "num_heads * %d}, the 8-bit copies Int32 {batch_size, num_heads, seq_len, %d} and float scales "
                        "{batch_size, num_heads, seq_len}\n",
                ATTENTION_8BIT_HEAD_SIZE, (int)(ATTENTION_8BIT_HEAD_SIZE / ATTENTION_8BIT_VALUES_PER_INT));
        exit(EXIT_FAILURE);
    }
}

#endif // ATTENTION_8BIT_CUH
