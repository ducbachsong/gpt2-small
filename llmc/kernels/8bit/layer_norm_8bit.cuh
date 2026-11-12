// layer_norm_8bit.cuh — LayerNorm for a bfloat16 model whose products are in 8 bits: bf16/layer_norm_bf16.cuh's
// LayerNorm, whose forward also writes the output's 8-bit copy (FP8 E4M3, one power-of-2 scale per row), the copy
// the 8-bit product after it reads (qkv after ln1, mlp_up after ln2, the head after lnf):
//     forward:    output                 = LayerNorm(input) · weight + bias      bfloat16, as layer_norm_bf16
//                 output_8bit, scales    = round_8bit(output)                    the same bits, in the same kernel
//     backward:   layer_norm_bf16_backward's (the rounding's gradient taken as 1, straight-through, as the 8-bit
//                 products' backwards take it)
//
//     tensor                    type in memory                   shape                                    GPT-2 small
//     input, output             bfloat16                         {batch_size, seq_len, embed_dim}         {4, 1024, 768}
//     output_8bit               E4M3, 4 in each int of an        {batch_size x seq_len, embed_dim / 4}    {4096, 192}
//                               Int32 tensor
//     output_scales             float                            {batch_size x seq_len}                   {4096}
//     weight, bias              bfloat16                         {embed_dim}                              {768}
//     mean, rstd                float                            {batch_size, seq_len}                    kept for the backward
//
// Why: an 8-bit product rounds its input to E4M3 first, with round_8bit: a pass of its own that reads the whole
// output back from GPU memory. Here the row's warp does it while the row is fresh: after writing the bfloat16
// output it takes the row's largest |value| (warp_reduce_max), its scale (round_8bit_scale), and writes the E4M3s,
// reading back the values it has just written (most likely still in the L2 cache: 50 MB on an H100, the row a
// microsecond old). The copy is round_8bit's to the last bit, so the product can take it in place of rounding again
// (Linear8bitScratch's left_8bit_tensor and left_scales_tensor have its shape).
//
//     GPT-2 small, one LayerNorm      the output read again                       launches
//     layer_norm_bf16 + round_8bit    6.3 MB, a pass of its own, after            2
//     here                            the same 6.3 MB, row by row, just written   1
// Any GPU (no tensor cores here).
// Math: https://ducbachsong.github.io/posts/layer-norm-backward
#ifndef LAYER_NORM_8BIT_CUH
#define LAYER_NORM_8BIT_CUH

#include <cuda_fp8.h>                    // FP8 E4M3
#include "../cuda_common.h"              // cudaCheck, check_threads_per_block, warp_reduce_max
#include "../../tensor.cuh"              // read8_as_float, write8_from_float
#include "round_8bit.cuh"                // round_8bit_scale, ROUND_8BIT_VALUES_PER_INT, ROUND_8BIT_ROW_LENGTH_MULTIPLE
#include "../bf16/layer_norm_bf16.cuh"   // the 8 values a lane, the row's mean and rstd, the checks, the backward

#define LAYER_NORM_8BIT_VALUES_PER_LOAD LAYER_NORM_BF16_VALUES_PER_LOAD                       // values a lane takes at a time: 8
#define LAYER_NORM_8BIT_FOURS_PER_LOAD (LAYER_NORM_8BIT_VALUES_PER_LOAD / ROUND_8BIT_VALUES_PER_INT)   // E4M3 fours (ints) in them: 2
#define LAYER_NORM_8BIT_THREADS_PER_BLOCK LAYER_NORM_BF16_THREADS_PER_BLOCK                   // the forward's default block size
#define LAYER_NORM_8BIT_BACKWARD_THREADS_PER_BLOCK LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK // the backward's: layer_norm_bf16's
static_assert(sizeof(uint2) == LAYER_NORM_8BIT_VALUES_PER_LOAD * sizeof(__nv_fp8_e4m3), "a load's E4M3s are one 8-byte store");

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_8bit_forward(const Tensor& output_tensor, const Tensor& output_8bit_tensor, const Tensor& output_scales_tensor,
                             const Tensor& mean_tensor, const Tensor& rstd_tensor, const Tensor& input_tensor,
                             const Tensor& weight_tensor, const Tensor& bias_tensor,
                             int threads_per_block_val = LAYER_NORM_8BIT_THREADS_PER_BLOCK);
void layer_norm_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                              const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                              const Tensor& mean_tensor, const Tensor& rstd_tensor,
                              int threads_per_block_val = LAYER_NORM_8BIT_BACKWARD_THREADS_PER_BLOCK);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void layer_norm_8bit_forward_kernel(nv_bfloat16* output_ptr, __nv_fp8_e4m3* output_8bit_ptr, float* output_scales_ptr,
                                               float* mean_ptr, float* rstd_ptr, const nv_bfloat16* input_ptr,
                                               const nv_bfloat16* weight_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                                               size_t embed_dim_val);

/// Writes output = LayerNorm(input) · weight + bias in bfloat16, its E4M3 copy with a scale per row, and each row's
/// mean and rstd: checks, then launches layer_norm_8bit_forward_kernel, one warp per row.
///
///   output_tensor, input_tensor   {batch_size, seq_len, embed_dim} bfloat16; embed_dim a multiple of
///                                 ROUND_8BIT_ROW_LENGTH_MULTIPLE (the product reads rows of whole 16-byte pieces)
///   output_8bit_tensor            {batch_size x seq_len, embed_dim / ROUND_8BIT_VALUES_PER_INT} Int32 (or any shape
///                                 of that many ints, embed_dim / 4 last): written, round_8bit(output)'s E4M3s
///   output_scales_tensor          {batch_size x seq_len} float (any shape of that many): written, one scale per row
///   mean_tensor, rstd_tensor      {batch_size, seq_len} float: for the backward
///   weight_tensor, bias_tensor    {embed_dim} bfloat16
///   threads_per_block_val         the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                 LAYER_NORM_8BIT_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void layer_norm_8bit_forward(const Tensor& output_tensor, const Tensor& output_8bit_tensor, const Tensor& output_scales_tensor,
                                    const Tensor& mean_tensor, const Tensor& rstd_tensor, const Tensor& input_tensor,
                                    const Tensor& weight_tensor, const Tensor& bias_tensor, int threads_per_block_val) {
    // 1. check the launch, the tensors and the copy's; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "layer_norm_8bit_forward");
    layer_norm_bf16_check_shapes("layer_norm_8bit_forward", output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                                 bias_tensor);
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    bool fits_val = embed_dim_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && output_8bit_tensor.dtype() == DType::Int32 &&
                    output_8bit_tensor.dim() >= 1 &&
                    output_8bit_tensor.size(output_8bit_tensor.dim() - 1) * ROUND_8BIT_VALUES_PER_INT == embed_dim_val &&
                    output_8bit_tensor.numel() * ROUND_8BIT_VALUES_PER_INT == rows_val * embed_dim_val &&
                    output_scales_tensor.dtype() == DType::Float32 && output_scales_tensor.numel() == rows_val &&
                    (size_t)output_8bit_tensor.data_ptr() % sizeof(uint2) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: layer_norm_8bit_forward needs embed_dim a multiple of %d, output_8bit {rows, embed_dim / %d} "
                        "Int32 starting on %zu bytes and output_scales {rows} float; got embed_dim %zu\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE, (int)ROUND_8BIT_VALUES_PER_INT, sizeof(uint2), embed_dim_val);
        exit(EXIT_FAILURE);
    }
    // 2. launch: one warp per row, the blocks rounded up
    size_t warps_per_block_val = threads_per_block_val / HARDWARE_THREADS_PER_WARP;
    int blocks_val = (int)CEIL_DIV(rows_val, warps_per_block_val);
    layer_norm_8bit_forward_kernel<<<blocks_val, threads_per_block_val>>>(
        output_tensor.data_ptr<nv_bfloat16>(), reinterpret_cast<__nv_fp8_e4m3*>(output_8bit_tensor.data_ptr()),
        output_scales_tensor.data(), mean_tensor.data(), rstd_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(),
        weight_tensor.data_ptr<nv_bfloat16>(), bias_tensor.data_ptr<nv_bfloat16>(), rows_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Normalizes the rows as layer_norm_bf16_forward_kernel does, one warp per row, 8 values a lane, then rounds each
/// row as round_8bit_kernel does: the largest |value| of the bfloat16 output as stored, its scale, each value over
/// the scale to the nearest E4M3 (saturating, as __nv_fp8x4_e4m3 does).
///
///   output_ptr, mean_ptr, rstd_ptr, input_ptr, weight_ptr, bias_ptr   as in layer_norm_8bit_forward
///   output_8bit_ptr                                                   {rows, embed_dim} E4M3: written
///   output_scales_ptr                                                 one float per row: written
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
__global__ void layer_norm_8bit_forward_kernel(nv_bfloat16* output_ptr, __nv_fp8_e4m3* output_8bit_ptr, float* output_scales_ptr,
                                               float* mean_ptr, float* rstd_ptr, const nv_bfloat16* input_ptr,
                                               const nv_bfloat16* weight_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                                               size_t embed_dim_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t row_loads_val = embed_dim_val / LAYER_NORM_8BIT_VALUES_PER_LOAD;                          // loads in one row: 768 / 8 = 96
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const nv_bfloat16* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        nv_bfloat16* output_row_ptr = output_ptr + row_index_val * embed_dim_val;
        __nv_fp8_e4m3* output_8bit_row_ptr = output_8bit_ptr + row_index_val * embed_dim_val;

        // 1. the mean, then the variance around it and rstd
        float row_mean_val, row_rstd_val;
        layer_norm_bf16_row_mean_rstd(&row_mean_val, &row_rstd_val, input_row_ptr, lane_index_val, embed_dim_val);

        // 2. norm, then the output, rounded to bfloat16 and stored; this lane's largest |value| of it as stored
        float thread_largest_val = 0.0f;
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_8BIT_VALUES_PER_LOAD * load_index_val;
            float input_list[LAYER_NORM_8BIT_VALUES_PER_LOAD], weight_list[LAYER_NORM_8BIT_VALUES_PER_LOAD];
            float bias_list[LAYER_NORM_8BIT_VALUES_PER_LOAD], output_list[LAYER_NORM_8BIT_VALUES_PER_LOAD];
            read8_as_float(input_list, input_row_ptr + first_index_val);
            read8_as_float(weight_list, weight_ptr + first_index_val);
            read8_as_float(bias_list, bias_ptr + first_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < LAYER_NORM_8BIT_VALUES_PER_LOAD; value_index_val++) {
                float output_val = (input_list[value_index_val] - row_mean_val) * row_rstd_val * weight_list[value_index_val] +
                                   bias_list[value_index_val];
                output_list[value_index_val] = to_float(from_float<nv_bfloat16>(output_val));   // as stored
                thread_largest_val = fmaxf(thread_largest_val, fabsf(output_list[value_index_val]));
            }
            write8_from_float(output_row_ptr + first_index_val, output_list);
        }

        // 3. the row's largest |value| (the warp's), and its scale: round_8bit's
        float row_scale_val = round_8bit_scale(warp_reduce_max(thread_largest_val));
        float to_stored_val = 1.0f / row_scale_val;                       // exact: a power of 2

        // 4. the output over the scale, to E4M3: each lane reads back the values it stored in step 2
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_8BIT_VALUES_PER_LOAD * load_index_val;
            float output_list[LAYER_NORM_8BIT_VALUES_PER_LOAD];
            read8_as_float(output_list, output_row_ptr + first_index_val);
            unsigned int fours_list[LAYER_NORM_8BIT_FOURS_PER_LOAD];
#pragma unroll
            for (int four_index_val = 0; four_index_val < LAYER_NORM_8BIT_FOURS_PER_LOAD; four_index_val++) {
                const float* four_ptr = output_list + four_index_val * ROUND_8BIT_VALUES_PER_INT;
                fours_list[four_index_val] = __nv_fp8x4_e4m3(make_float4(four_ptr[0] * to_stored_val, four_ptr[1] * to_stored_val,
                                                                         four_ptr[2] * to_stored_val, four_ptr[3] * to_stored_val)).__x;
            }
            *reinterpret_cast<uint2*>(output_8bit_row_ptr + first_index_val) = make_uint2(fours_list[0], fours_list[1]);
        }

        // 5. lane 0 keeps the row's mean, rstd and scale
        if (lane_index_val == 0) {
            mean_ptr[row_index_val] = row_mean_val;
            rstd_ptr[row_index_val] = row_rstd_val;
            output_scales_ptr[row_index_val] = row_scale_val;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Adds the gradients of input, weight and bias onto what is there: layer_norm_bf16_backward, in bfloat16, from the
/// bfloat16 output's math. The forward's rounding to E4M3 passes the gradient through as it is (straight-through).
/// weight_grad and bias_grad must be 0 before a step's first backward.
///
///   the tensors and threads_per_block_val   as in layer_norm_bf16_backward
inline void layer_norm_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                     const Tensor& mean_tensor, const Tensor& rstd_tensor, int threads_per_block_val) {
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                             weight_tensor, mean_tensor, rstd_tensor, threads_per_block_val);
}

#endif // LAYER_NORM_8BIT_CUH
