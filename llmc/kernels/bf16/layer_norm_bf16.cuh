// layer_norm_bf16.cuh — LayerNorm in bfloat16, 8 values a lane (one 16-byte load): fp32/layer_norm.cuh's math,
// whose bfloat16 path takes 4 values a lane (one 8-byte load):
//     mean     = sum(x) / embed_dim
//     variance = sum((x - mean)²) / embed_dim
//     rstd     = 1 / sqrt(variance + 1e-5)
//     norm     = (x - mean) · rstd
//     output   = norm · weight + bias
//     backward:   bias_grad   += sum over rows of output_grad
//                 weight_grad += sum over rows of output_grad · norm
//                 norm_grad    = output_grad · weight
//                 input_grad  += rstd · (norm_grad - mean(norm_grad) - norm · mean(norm_grad · norm))
//
//     tensor                                type in memory   shape                               GPT-2 small
//     input, output, input_grad, output_grad bfloat16         {batch_size, seq_len, embed_dim}    {4, 1024, 768}
//     weight, bias, and their grads          bfloat16         {embed_dim}                         {768}
//     mean, rstd                             float            {batch_size, seq_len}               kept for the backward
//     in registers                           float            -                                   every sum
//
// One warp per row, as there; each lane walks the row 8 values at a time instead of 4:
//
//     GPT-2 small, one row      loads a row     loads a lane   bytes a load
//     fp32/layer_norm.cuh, bf16  768 / 4 = 192   6              8
//     here                       768 / 8 = 96    3              16
//
// A lane holds other values than there, so each row's sums are added in another order: mean, rstd, output and
// input_grad land within a bfloat16 rounding of the 4-value path's, not to the last bit. Any GPU.
// Math: https://ducbachsong.github.io/posts/layer-norm-backward
#ifndef LAYER_NORM_BF16_CUH
#define LAYER_NORM_BF16_CUH

#include "../cuda_common.h"          // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../../common/tensor.cuh"          // read8_as_float, write8_from_float, VALUES_PER_BF16_LOAD

#define LAYER_NORM_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD                          // bfloat16s in one 16-byte load: 8
#define LAYER_NORM_BF16_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)             // the forward's default: fp32/layer_norm.cuh's, not swept for 8 values yet
#define LAYER_NORM_BF16_BACKWARD_BLOCKS 80                                            // backward: blocks in every launch, as fp32/layer_norm.cuh's
#define LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK (16 * HARDWARE_THREADS_PER_WARP)   // the backward's default, as fp32/layer_norm.cuh's
#define LAYER_NORM_BF16_EPS 1e-5f                                                     // added to the variance, as GPT-2 does
static_assert(LAYER_NORM_BF16_VALUES_PER_LOAD == 2 * VALUES_PER_FLOAT4, "weight_grad and bias_grad take a load as two atomic_add4_from_float");

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_bf16_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                             const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                             int threads_per_block_val = LAYER_NORM_BF16_THREADS_PER_BLOCK);
void layer_norm_bf16_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                              const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                              const Tensor& mean_tensor, const Tensor& rstd_tensor,
                              int threads_per_block_val = LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
__device__ __forceinline__ void layer_norm_bf16_row_mean_rstd(float* row_mean_ptr, float* row_rstd_ptr, const nv_bfloat16* input_row_ptr,
                                                              int lane_index_val, size_t embed_dim_val);
void layer_norm_bf16_check_shapes(const char* caller_ptr, const Tensor& output_tensor, const Tensor& mean_tensor,
                                  const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                  const Tensor& bias_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void layer_norm_bf16_forward_kernel(nv_bfloat16* output_ptr, float* mean_ptr, float* rstd_ptr, const nv_bfloat16* input_ptr,
                                               const nv_bfloat16* weight_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                                               size_t embed_dim_val);

/// Writes output = LayerNorm(input) · weight + bias in bfloat16, and each row's mean and rstd: checks, then launches
/// layer_norm_bf16_forward_kernel, one warp per row.
///
///   output_tensor, input_tensor   {batch_size, seq_len, embed_dim} bfloat16
///   mean_tensor, rstd_tensor      {batch_size, seq_len} float: for the backward
///   weight_tensor, bias_tensor    {embed_dim} bfloat16
///   threads_per_block_val         the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                 LAYER_NORM_BF16_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void layer_norm_bf16_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                                    const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                                    int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "layer_norm_bf16_forward");
    layer_norm_bf16_check_shapes("layer_norm_bf16_forward", output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor,
                                 bias_tensor);
    // 2. launch: one warp per row, the blocks rounded up
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t warps_per_block_val = threads_per_block_val / HARDWARE_THREADS_PER_WARP;
    int blocks_val = (int)CEIL_DIV(rows_val, warps_per_block_val);
    layer_norm_bf16_forward_kernel<<<blocks_val, threads_per_block_val>>>(
        output_tensor.data_ptr<nv_bfloat16>(), mean_tensor.data(), rstd_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(),
        weight_tensor.data_ptr<nv_bfloat16>(), bias_tensor.data_ptr<nv_bfloat16>(), rows_val, input_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Normalizes the rows, one warp per row; the warp's lanes walk the row 8 values (one 16-byte load) at a time. Three
/// passes: the mean and the variance around it (layer_norm_bf16_row_mean_rstd), then the output.
///
///   output_ptr, mean_ptr, rstd_ptr, input_ptr, weight_ptr, bias_ptr   as in layer_norm_bf16_forward
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
__global__ void layer_norm_bf16_forward_kernel(nv_bfloat16* output_ptr, float* mean_ptr, float* rstd_ptr, const nv_bfloat16* input_ptr,
                                               const nv_bfloat16* weight_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                                               size_t embed_dim_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t row_loads_val = embed_dim_val / LAYER_NORM_BF16_VALUES_PER_LOAD;                          // loads in one row: 768 / 8 = 96
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const nv_bfloat16* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        nv_bfloat16* output_row_ptr = output_ptr + row_index_val * embed_dim_val;

        // 1. the mean, then the variance around it and rstd
        float row_mean_val, row_rstd_val;
        layer_norm_bf16_row_mean_rstd(&row_mean_val, &row_rstd_val, input_row_ptr, lane_index_val, embed_dim_val);

        // 2. norm, then the output
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val;
            float input_list[LAYER_NORM_BF16_VALUES_PER_LOAD], weight_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            float bias_list[LAYER_NORM_BF16_VALUES_PER_LOAD], output_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            read8_as_float(input_list, input_row_ptr + first_index_val);
            read8_as_float(weight_list, weight_ptr + first_index_val);
            read8_as_float(bias_list, bias_ptr + first_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++)
                output_list[value_index_val] = (input_list[value_index_val] - row_mean_val) * row_rstd_val * weight_list[value_index_val] +
                                               bias_list[value_index_val];
            write8_from_float(output_row_ptr + first_index_val, output_list);
        }
        // 3. lane 0 keeps the row's mean and rstd for the backward
        if (lane_index_val == 0) {
            mean_ptr[row_index_val] = row_mean_val;
            rstd_ptr[row_index_val] = row_rstd_val;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

__global__ void layer_norm_bf16_backward_kernel(nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                                const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                                size_t rows_val, size_t embed_dim_val);

/// Adds the gradients of input, weight and bias onto what is there, in bfloat16: checks, then launches
/// layer_norm_bf16_backward_kernel. weight_grad and bias_grad must be 0 before a step's first backward.
///
///   input_grad_tensor                                       input's shape; added onto: the residual stream's gradient
///   weight_grad_tensor, bias_grad_tensor                    {embed_dim} bfloat16
///   output_grad_tensor                                      the gradient from the layer above
///   input_tensor, weight_tensor, mean_tensor, rstd_tensor   the forward's
///   threads_per_block_val                                   the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK;
///                                                           not given: LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK. input_grad
///                                                           never changes with it; weight_grad's and bias_grad's adds can
///                                                           land in another order, as they can from run to run.
inline void layer_norm_bf16_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                     const Tensor& mean_tensor, const Tensor& rstd_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "layer_norm_bf16_backward");
    layer_norm_bf16_check_shapes("layer_norm_bf16_backward", output_grad_tensor, mean_tensor, rstd_tensor, input_tensor,
                                 weight_tensor, bias_grad_tensor);
    layer_norm_bf16_check_shapes("layer_norm_bf16_backward", input_grad_tensor, mean_tensor, rstd_tensor, input_tensor,
                                 weight_grad_tensor, bias_grad_tensor);
    // 2. launch: LAYER_NORM_BF16_BACKWARD_BLOCKS blocks, one warp per row, the block's bias_grad and weight_grad in
    //    shared memory
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    size_t shared_bytes_val = 2 * embed_dim_val * sizeof(float);
    layer_norm_bf16_backward_kernel<<<LAYER_NORM_BF16_BACKWARD_BLOCKS, threads_per_block_val, shared_bytes_val>>>(
        input_grad_tensor.data_ptr<nv_bfloat16>(), weight_grad_tensor.data_ptr<nv_bfloat16>(),
        bias_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(), input_tensor.data_ptr<nv_bfloat16>(),
        weight_tensor.data_ptr<nv_bfloat16>(), mean_tensor.data(), rstd_tensor.data(), rows_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Works out the three gradients, one warp per row, 8 values a lane at a time. Two passes per row: the two means,
/// then the gradients. weight_grad and bias_grad are sums over all rows: each block adds its rows into shared memory
/// first (in float), then once onto GPU memory at the end, two atomic_add4_from_float a load (rounded to bfloat16
/// once per block). The adds come in any order, so the last bits can differ run to run.
///
///   input_grad_ptr, weight_grad_ptr, bias_grad_ptr, output_grad_ptr   as in layer_norm_bf16_backward
///   input_ptr, weight_ptr, mean_ptr, rstd_ptr                         the forward's
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
__global__ void layer_norm_bf16_backward_kernel(nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                                const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                                size_t rows_val, size_t embed_dim_val) {
    // 1. the block's bias_grad and weight_grad in shared memory, 2 x embed_dim, all 0
    extern __shared__ float block_sums_list[];
    float* block_bias_grad_ptr = block_sums_list;
    float* block_weight_grad_ptr = block_sums_list + embed_dim_val;
    for (size_t value_index_val = threadIdx.x; value_index_val < 2 * embed_dim_val; value_index_val += blockDim.x)
        block_sums_list[value_index_val] = 0.0f;
    __syncthreads();

    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t row_loads_val = embed_dim_val / LAYER_NORM_BF16_VALUES_PER_LOAD;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const nv_bfloat16* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        const nv_bfloat16* output_grad_row_ptr = output_grad_ptr + row_index_val * embed_dim_val;
        nv_bfloat16* input_grad_row_ptr = input_grad_ptr + row_index_val * embed_dim_val;
        float row_mean_val = mean_ptr[row_index_val];
        float row_rstd_val = rstd_ptr[row_index_val];

        // 2. mean(norm_grad) and mean(norm_grad · norm)
        float norm_grad_sum_val = 0.0f;
        float norm_grad_norm_sum_val = 0.0f;
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val;
            float input_list[LAYER_NORM_BF16_VALUES_PER_LOAD], output_grad_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            float weight_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            read8_as_float(input_list, input_row_ptr + first_index_val);
            read8_as_float(output_grad_list, output_grad_row_ptr + first_index_val);
            read8_as_float(weight_list, weight_ptr + first_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++) {
                float norm_val = (input_list[value_index_val] - row_mean_val) * row_rstd_val;
                float norm_grad_val = output_grad_list[value_index_val] * weight_list[value_index_val];
                norm_grad_sum_val += norm_grad_val;
                norm_grad_norm_sum_val += norm_grad_val * norm_val;
            }
        }
        float norm_grad_mean_val = warp_reduce_sum(norm_grad_sum_val) / (float)embed_dim_val;
        float norm_grad_norm_mean_val = warp_reduce_sum(norm_grad_norm_sum_val) / (float)embed_dim_val;

        // 3. the three gradients
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val;
            float input_list[LAYER_NORM_BF16_VALUES_PER_LOAD], output_grad_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            float weight_list[LAYER_NORM_BF16_VALUES_PER_LOAD], input_grad_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
            read8_as_float(input_list, input_row_ptr + first_index_val);
            read8_as_float(output_grad_list, output_grad_row_ptr + first_index_val);
            read8_as_float(weight_list, weight_ptr + first_index_val);
            read8_as_float(input_grad_list, input_grad_row_ptr + first_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++) {
                float norm_val = (input_list[value_index_val] - row_mean_val) * row_rstd_val;
                float norm_grad_val = output_grad_list[value_index_val] * weight_list[value_index_val];
                // bias_grad += output_grad and weight_grad += output_grad · norm, into the block's copy
                atomicAdd(block_bias_grad_ptr + first_index_val + value_index_val, output_grad_list[value_index_val]);
                atomicAdd(block_weight_grad_ptr + first_index_val + value_index_val, output_grad_list[value_index_val] * norm_val);
                // input_grad += rstd · (norm_grad - mean(norm_grad) - norm · mean(norm_grad · norm)); one warp per row
                input_grad_list[value_index_val] += row_rstd_val * (norm_grad_val - norm_grad_mean_val - norm_val * norm_grad_norm_mean_val);
            }
            write8_from_float(input_grad_row_ptr + first_index_val, input_grad_list);
        }
    }

    // 4. the block's bias_grad and weight_grad, added onto GPU memory: a load's 8 values as two adds of 4
    __syncthreads();
    for (size_t load_index_val = threadIdx.x; load_index_val < row_loads_val; load_index_val += blockDim.x) {
        size_t first_index_val = LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val;
        for (size_t four_index_val = first_index_val; four_index_val < first_index_val + LAYER_NORM_BF16_VALUES_PER_LOAD;
             four_index_val += VALUES_PER_FLOAT4) {
            const float* bias_sums_ptr = block_bias_grad_ptr + four_index_val;
            const float* weight_sums_ptr = block_weight_grad_ptr + four_index_val;
            atomic_add4_from_float(bias_grad_ptr + four_index_val,
                                   make_float4(bias_sums_ptr[0], bias_sums_ptr[1], bias_sums_ptr[2], bias_sums_ptr[3]));
            atomic_add4_from_float(weight_grad_ptr + four_index_val,
                                   make_float4(weight_sums_ptr[0], weight_sums_ptr[1], weight_sums_ptr[2], weight_sums_ptr[3]));
        }
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Works out one row's mean and rstd with its warp, 8 values a lane at a time: the mean, then the variance around it
/// (two passes, as fp32/layer_norm.cuh). Every lane gets both. Also layer_norm_8bit.cuh's.
///
///   row_mean_ptr, row_rstd_ptr   written: the row's mean and 1 / sqrt(variance + LAYER_NORM_BF16_EPS)
///   input_row_ptr                the row, embed_dim bfloat16s on 16 bytes
///   lane_index_val               this lane of the row's warp
///   embed_dim_val                the row length: 768
__device__ __forceinline__ void layer_norm_bf16_row_mean_rstd(float* row_mean_ptr, float* row_rstd_ptr, const nv_bfloat16* input_row_ptr,
                                                              int lane_index_val, size_t embed_dim_val) {
    size_t row_loads_val = embed_dim_val / LAYER_NORM_BF16_VALUES_PER_LOAD;
    // 1. the mean
    float input_sum_val = 0.0f;
    for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
        float input_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
        read8_as_float(input_list, input_row_ptr + LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val);
#pragma unroll
        for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++)
            input_sum_val += input_list[value_index_val];
    }
    float row_mean_val = warp_reduce_sum(input_sum_val) / (float)embed_dim_val;
    // 2. the variance around it, then rstd
    float squares_sum_val = 0.0f;
    for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
        float input_list[LAYER_NORM_BF16_VALUES_PER_LOAD];
        read8_as_float(input_list, input_row_ptr + LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val);
#pragma unroll
        for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++) {
            float centered_val = input_list[value_index_val] - row_mean_val;
            squares_sum_val += centered_val * centered_val;
        }
    }
    float row_variance_val = warp_reduce_sum(squares_sum_val) / (float)embed_dim_val;
    *row_mean_ptr = row_mean_val;
    *row_rstd_ptr = rsqrtf(row_variance_val + LAYER_NORM_BF16_EPS);
}

/// Stops with a message unless the tensors fit: the shapes at the top of the file, input, output, weight and bias
/// bfloat16, mean and rstd float, embed_dim a multiple of LAYER_NORM_BF16_VALUES_PER_LOAD and small enough for the
/// backward's shared memory, each bfloat16 tensor starting on 16 bytes.
///
///   caller_ptr                                                                the function checking, for the message
///   output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor
///       forward: these; backward: the gradients in their place
inline void layer_norm_bf16_check_shapes(const char* caller_ptr, const Tensor& output_tensor, const Tensor& mean_tensor,
                                         const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                         const Tensor& bias_tensor) {
    // 1. shapes: as at the top of the file
    bool fits_val = input_tensor.dim() == 3 && output_tensor.shape() == input_tensor.shape() &&
                    mean_tensor.shape() == Shape({input_tensor.size(0), input_tensor.size(1)}) &&
                    rstd_tensor.shape() == mean_tensor.shape() && weight_tensor.shape() == Shape({input_tensor.size(2)}) &&
                    bias_tensor.shape() == weight_tensor.shape();
    if (!fits_val) {
        fprintf(stderr, "error: %s needs input and output {batch_size, seq_len, embed_dim}, mean and rstd {batch_size, "
                        "seq_len}, weight and bias {embed_dim}\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 2. types: input, output, weight, bias bfloat16; mean and rstd float
    if (input_tensor.dtype() != DType::BFloat16 || output_tensor.dtype() != DType::BFloat16 ||
        weight_tensor.dtype() != DType::BFloat16 || bias_tensor.dtype() != DType::BFloat16 ||
        mean_tensor.dtype() != DType::Float32 || rstd_tensor.dtype() != DType::Float32) {
        fprintf(stderr, "error: %s needs input, output, weight and bias bfloat16 (float: fp32/layer_norm.cuh), and mean "
                        "and rstd float\n", caller_ptr);
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte loads, each bfloat16 tensor starting on one
    size_t load_bytes_val = sizeof(uint4);
    if (input_tensor.size(2) % LAYER_NORM_BF16_VALUES_PER_LOAD != 0 || (size_t)input_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)output_tensor.data_ptr() % load_bytes_val != 0 || (size_t)weight_tensor.data_ptr() % load_bytes_val != 0 ||
        (size_t)bias_tensor.data_ptr() % load_bytes_val != 0) {
        fprintf(stderr, "error: %s reads %d values at a time, so embed_dim must be a multiple of %d and input, output, "
                        "weight, bias must start on %zu bytes; got embed_dim %zu, %p, %p, %p, %p\n",
                caller_ptr, LAYER_NORM_BF16_VALUES_PER_LOAD, LAYER_NORM_BF16_VALUES_PER_LOAD, load_bytes_val, input_tensor.size(2),
                input_tensor.data_ptr(), output_tensor.data_ptr(), weight_tensor.data_ptr(), bias_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
    // 4. shared memory: the backward keeps 2 x embed_dim floats per block
    size_t largest_embed_dim_val = HARDWARE_DEFAULT_MAX_SHARED_BYTES_PER_BLOCK / (2 * sizeof(float));
    if (input_tensor.size(2) > largest_embed_dim_val) {
        fprintf(stderr, "error: %s's backward keeps 2 x embed_dim floats per block in shared memory, at most %d bytes, so "
                        "embed_dim must be at most %zu; got %zu\n",
                caller_ptr, HARDWARE_DEFAULT_MAX_SHARED_BYTES_PER_BLOCK, largest_embed_dim_val, input_tensor.size(2));
        exit(EXIT_FAILURE);
    }
}

#endif // LAYER_NORM_BF16_CUH
