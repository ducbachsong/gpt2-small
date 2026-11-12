// encoder_8bit.cuh — GPT-2's first layer for a bfloat16 model whose products are in 8 bits: wte read from its 8-bit
// copy (FP8 E4M3, one power-of-2 scale per row, made by round_8bit), the copy the head (classifier_8bit.cuh) needs
// for its own product:
//     forward:    wte_8bit, wte_scales = round_8bit(wte)                      all of wte, once a step
//                 encoded[b][t] = wte_scale[id] · wte_8bit[id] + wpe[t]       id = token_ids[b][t]
//     backward:   bf16/encoder_bf16.cuh's: wte_grad[id] += encoded_grad[b][t], wpe_grad[t] += encoded_grad[b][t]
//                 (the rounding's gradient taken as 1, the straight-through way the 8-bit products' backwards take it)
//
//     tensor                    type in memory                   shape                                 GPT-2 small
//     wte, wte_grad             bfloat16                         {vocab_size, embed_dim}               {50304, 768}
//     wte_8bit                  E4M3, 4 in each int of an        {vocab_size, embed_dim / 4}           {50304, 192}
//                               Int32 tensor
//     wte_scales                float                            {vocab_size}                          {50304}
//     wpe, wpe_grad             bfloat16                         {max_seq_len, embed_dim}              {1024, 768}
//     encoded, encoded_grad     bfloat16                         {batch_size, seq_len, embed_dim}      {4, 1024, 768}
//     token_ids                 int                              {batch_size, seq_len}                 {4, 1024}
//     in registers              float                            -                                     every add
//
// Why: the head's product rounds all of wte to E4M3 each step anyway. Made here, at the start of the step, the one
// copy serves both: the caller hands wte_8bit and wte_scales to the head (Linear8bitScratch's output_copies
// right_8bit_tensor and right_scales_tensor have this shape), so wte is rounded once a step, not twice.
// What it costs: each token's row is off by up to 2^-4 (6%) of a value, against bfloat16's 2^-8 (0.4%); wpe stays
// bfloat16. The 8 values a thread are encoder_bf16.cuh's: the token's 8 E4M3s (8 bytes), turned back exactly
// (E4M3 x a power of 2), the position's 8 bfloat16s (16 bytes), added in float and rounded once to bfloat16.
//
//     GPT-2 small, one forward     reads                                  writes
//     the rounding                 wte: 50,304 x 768 x 2 = 77.3 MB        wte_8bit 38.6 MB, wte_scales 0.2 MB
//     the gather                   4096 rows of wte_8bit: 3.1 MB          encoded 6.3 MB
//                                  4096 rows of wpe: 6.3 MB
// The gather reads half encoder_bf16's wte bytes, but the rounding moves far more than the whole bfloat16 encoder
// (about 19 MB): it pays only as the head's rounding it stands in for. Any GPU (no tensor cores here).
#ifndef ENCODER_8BIT_CUH
#define ENCODER_8BIT_CUH

#include <cuda_fp8.h>                    // FP8 E4M3
#include "../cuda_common.h"              // cudaCheck, check_threads_per_block
#include "../../tensor.cuh"
#include "round_8bit.cuh"                // the 8-bit copy: round_8bit, ROUND_8BIT_VALUES_PER_INT
#include "../bf16/encoder_bf16.cuh"      // the 8 values a thread, the checks, the backward

#define ENCODER_8BIT_VALUES_PER_THREAD ENCODER_BF16_VALUES_PER_LOAD     // values a thread takes at a time: 8, encoder_bf16's
#define ENCODER_8BIT_FOURS_PER_THREAD (ENCODER_8BIT_VALUES_PER_THREAD / ROUND_8BIT_VALUES_PER_INT)   // E4M3 fours (ints) in them: 2
#define ENCODER_8BIT_BLOCKS ENCODER_BF16_BLOCKS                          // blocks in the gather's launch
#define ENCODER_8BIT_FORWARD_THREADS ENCODER_BF16_FORWARD_THREADS        // the gather's default block size
#define ENCODER_8BIT_THREADS ENCODER_BF16_THREADS                        // the backward's default block size: encoder_bf16's
static_assert(sizeof(uint2) == ENCODER_8BIT_VALUES_PER_THREAD * sizeof(__nv_fp8_e4m3), "a thread's E4M3s are one 8-byte load");

// ── forward and backward ──────────────────────────────────────────────────────
void encoder_8bit_forward(const Tensor& encoded_tensor, const Tensor& wte_8bit_tensor, const Tensor& wte_scales_tensor,
                          const Tensor& token_ids_tensor, const Tensor& wte_tensor, const Tensor& wpe_tensor,
                          int threads_per_block_val = ENCODER_8BIT_FORWARD_THREADS);
void encoder_8bit_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                           const Tensor& token_ids_tensor, int threads_per_block_val = ENCODER_8BIT_THREADS);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void encoder_8bit_forward_kernel(nv_bfloat16* encoded_ptr, const int* token_ids_ptr, const __nv_fp8_e4m3* wte_8bit_ptr,
                                            const float* wte_scales_ptr, const nv_bfloat16* wpe_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t embed_dim_val);

/// Writes wte's 8-bit copy, then encoded = wte_scale[id] · wte_8bit[id] + wpe: checks, rounds wte (round_8bit), then
/// launches encoder_8bit_forward_kernel. The ids are not checked: an id >= vocab_size reads past the copy.
///
///   encoded_tensor          {batch_size, seq_len, embed_dim} bfloat16: the result, the first block's input
///   wte_8bit_tensor         {vocab_size, embed_dim / ROUND_8BIT_VALUES_PER_INT} Int32: written, wte in E4M3, for the
///                           head too
///   wte_scales_tensor       {vocab_size} float: written, one scale per row of wte
///   token_ids_tensor        {batch_size, seq_len} ints
///   wte_tensor              {vocab_size, embed_dim} bfloat16; embed_dim a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE
///   wpe_tensor              {max_seq_len, embed_dim} bfloat16; rows seq_len and after are not read
///   threads_per_block_val   the gather's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                           ENCODER_8BIT_FORWARD_THREADS. Any size gives the same bits; only the time changes. The
///                           rounding takes round_8bit's own.
inline void encoder_8bit_forward(const Tensor& encoded_tensor, const Tensor& wte_8bit_tensor, const Tensor& wte_scales_tensor,
                                 const Tensor& token_ids_tensor, const Tensor& wte_tensor, const Tensor& wpe_tensor,
                                 int threads_per_block_val) {
    // 1. check the launch, the tensors and the copy's; any of them wrong: stop with an error (round_8bit checks the
    //    copy against wte too)
    check_threads_per_block(threads_per_block_val, "encoder_8bit_forward");
    encoder_bf16_check_shapes("encoder_8bit_forward", encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    size_t vocab_size_val = wte_tensor.size(0);
    size_t embed_dim_val = wte_tensor.size(1);
    bool fits_val = wte_8bit_tensor.dim() == 2 && wte_8bit_tensor.dtype() == DType::Int32 &&
                    wte_8bit_tensor.size(0) == vocab_size_val &&
                    wte_8bit_tensor.size(1) * ROUND_8BIT_VALUES_PER_INT == embed_dim_val &&
                    wte_scales_tensor.dtype() == DType::Float32 && wte_scales_tensor.numel() == vocab_size_val &&
                    (size_t)wte_8bit_tensor.data_ptr() % sizeof(uint2) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: encoder_8bit_forward needs wte_8bit {vocab_size, embed_dim / %d} Int32 starting on %zu bytes "
                        "and wte_scales {vocab_size} float\n", (int)ROUND_8BIT_VALUES_PER_INT, sizeof(uint2));
        exit(EXIT_FAILURE);
    }
    // 2. the 8-bit copy of wte: all its rows, one scale each
    round_8bit(wte_8bit_tensor, wte_scales_tensor, wte_tensor);
    // 3. launch the gather: ENCODER_8BIT_BLOCKS blocks, each thread walking encoded 8 values at a time
    encoder_8bit_forward_kernel<<<ENCODER_8BIT_BLOCKS, threads_per_block_val>>>(
        encoded_tensor.data_ptr<nv_bfloat16>(), token_ids_tensor.data_ptr<int>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(wte_8bit_tensor.data_ptr()), wte_scales_tensor.data(),
        wpe_tensor.data_ptr<nv_bfloat16>(), encoded_tensor.size(0), encoded_tensor.size(1), embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Writes encoded 8 values at a time: 8 E4M3s of the token's row (one 8-byte load) times its row's scale, plus 8
/// bfloat16s of the position's row (one 16-byte load), added in float and rounded once.
///
///   encoded_ptr, token_ids_ptr, wpe_ptr               as in encoder_8bit_forward
///   wte_8bit_ptr                                      {vocab_size, embed_dim} E4M3
///   wte_scales_ptr                                    one float per row of wte_8bit
///   batch_size_val, seq_len_val, embed_dim_val        encoded's shape
__global__ void encoder_8bit_forward_kernel(nv_bfloat16* encoded_ptr, const int* token_ids_ptr, const __nv_fp8_e4m3* wte_8bit_ptr,
                                            const float* wte_scales_ptr, const nv_bfloat16* wpe_ptr, size_t batch_size_val,
                                            size_t seq_len_val, size_t embed_dim_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_loads_val = embed_dim_val / ENCODER_8BIT_VALUES_PER_THREAD;    // a thread's pieces in one row: 768 / 8 = 96
    size_t total_loads_val = batch_size_val * seq_len_val * row_loads_val;    // all of encoded
    // Thread n takes pieces n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        // 1. where this piece is: which token (b x seq_len + t), which piece of its row, which position t
        size_t token_index_val = index_val / row_loads_val;
        size_t row_load_index_val = index_val % row_loads_val;
        size_t position_index_val = token_index_val % seq_len_val;
        size_t token_id_val = token_ids_ptr[token_index_val];
        // 2. the token's 8 E4M3s, turned back: each four to a float4, times the row's scale (exact: a power of 2)
        uint2 e4m3_load_val = *reinterpret_cast<const uint2*>(wte_8bit_ptr + token_id_val * embed_dim_val +
                                                              row_load_index_val * ENCODER_8BIT_VALUES_PER_THREAD);
        unsigned int fours_list[ENCODER_8BIT_FOURS_PER_THREAD] = {e4m3_load_val.x, e4m3_load_val.y};
        float scale_val = wte_scales_ptr[token_id_val];
        float token_embed_list[ENCODER_8BIT_VALUES_PER_THREAD];
#pragma unroll
        for (int four_index_val = 0; four_index_val < ENCODER_8BIT_FOURS_PER_THREAD; four_index_val++) {
            __nv_fp8x4_e4m3 four_val;
            four_val.__x = fours_list[four_index_val];
            float4 values_val = static_cast<float4>(four_val);
            float* first_ptr = token_embed_list + four_index_val * ROUND_8BIT_VALUES_PER_INT;
            first_ptr[0] = values_val.x * scale_val;
            first_ptr[1] = values_val.y * scale_val;
            first_ptr[2] = values_val.z * scale_val;
            first_ptr[3] = values_val.w * scale_val;
        }
        // 3. the position's 8 bfloat16s; add and write
        float position_embed_list[ENCODER_8BIT_VALUES_PER_THREAD];
        read8_as_float(position_embed_list, wpe_ptr + (position_index_val * row_loads_val + row_load_index_val) * ENCODER_8BIT_VALUES_PER_THREAD);
#pragma unroll
        for (int value_index_val = 0; value_index_val < ENCODER_8BIT_VALUES_PER_THREAD; value_index_val++)
            token_embed_list[value_index_val] += position_embed_list[value_index_val];
        write8_from_float(encoded_ptr + index_val * ENCODER_8BIT_VALUES_PER_THREAD, token_embed_list);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Adds encoded_grad back onto the rows of wte and wpe the forward read: encoder_bf16_backward, in bfloat16. The
/// forward's rounding of wte passes the gradient through as it is (straight-through), so wte_grad is the bfloat16
/// encoder's. It adds, so the gradients must be 0 before a step's first backward.
///
///   wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor   as in encoder_bf16_backward
///   threads_per_block_val                                                     as in encoder_bf16_backward
inline void encoder_8bit_backward(const Tensor& wte_grad_tensor, const Tensor& wpe_grad_tensor, const Tensor& encoded_grad_tensor,
                                  const Tensor& token_ids_tensor, int threads_per_block_val) {
    encoder_bf16_backward(wte_grad_tensor, wpe_grad_tensor, encoded_grad_tensor, token_ids_tensor, threads_per_block_val);
}

#endif // ENCODER_8BIT_CUH
