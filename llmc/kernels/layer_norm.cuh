// layer_norm.cuh — LayerNorm: each row (one token) moved to mean 0 and std 1, then scaled and shifted:
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
//     input, output   {batch_size, seq_len, embed_dim}   {4, 1024, 768}
//     weight, bias    {embed_dim}                        {768}
//     mean, rstd      {batch_size, seq_len}: kept for the backward
// input, output, weight, bias and their gradients float or bfloat16 (T); mean and rstd float; the math in float.
// Math: https://ducbachsong.github.io/posts/layer-norm-backward
#ifndef LAYER_NORM_CUH
#define LAYER_NORM_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../tensor.cuh"

#define LAYER_NORM_THREADS_PER_BLOCK (16 * THREADS_PER_WARP)            // forward's default: 16 warps, one row each
#define LAYER_NORM_BACKWARD_BLOCKS 80                                   // backward: blocks in every launch
#define LAYER_NORM_BACKWARD_THREADS_PER_BLOCK (16 * THREADS_PER_WARP)   // backward's default: 16 warps; fastest in bfloat16 on an H100 PCIe
#define LAYER_NORM_EPS 1e-5f                                            // added to the variance, as GPT-2 does
#define LAYER_NORM_MAX_SHARED_BYTES (48 * 1024)                         // shared memory a block gets without asking for more

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                        const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                        int threads_per_block_val = LAYER_NORM_THREADS_PER_BLOCK);
void layer_norm_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                         const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                         const Tensor& mean_tensor, const Tensor& rstd_tensor,
                         int threads_per_block_val = LAYER_NORM_BACKWARD_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
void layer_norm_check_shapes(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                             const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void layer_norm_forward_kernel(T* output_ptr, float* mean_ptr, float* rstd_ptr, const T* input_ptr,
                                          const T* weight_ptr, const T* bias_ptr, size_t rows_val, size_t embed_dim_val);

/// Writes output = LayerNorm(input) · weight + bias, and each row's mean and rstd: checks, then launches
/// layer_norm_forward_kernel, one warp per row.
///
///   output_tensor, input_tensor   {batch_size, seq_len, embed_dim}
///   mean_tensor, rstd_tensor      {batch_size, seq_len}: for the backward
///   weight_tensor, bias_tensor    {embed_dim}
///   threads_per_block_val         the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                                 LAYER_NORM_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void layer_norm_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                               const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                               int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    // block size: whole warps, as each warp takes a row; at most MAX_THREADS_PER_BLOCK
    check_threads_per_block(threads_per_block_val, "LayerNorm");
    layer_norm_check_shapes(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor);
    // 2. launch: one warp per row, the blocks rounded up
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    size_t warps_per_block_val = threads_per_block_val / THREADS_PER_WARP;
    int blocks_val = (int)CEIL_DIV(rows_val, warps_per_block_val);
    if (input_tensor.dtype() == DType::BFloat16)
        layer_norm_forward_kernel<<<blocks_val, threads_per_block_val>>>(
            output_tensor.data_ptr<nv_bfloat16>(), mean_tensor.data(), rstd_tensor.data(),
            input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(),
            bias_tensor.data_ptr<nv_bfloat16>(), rows_val, embed_dim_val);
    else
        layer_norm_forward_kernel<<<blocks_val, threads_per_block_val>>>(
            output_tensor.data(), mean_tensor.data(), rstd_tensor.data(), input_tensor.data(), weight_tensor.data(),
            bias_tensor.data(), rows_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Normalizes the rows, one warp per row; the warp's THREADS_PER_WARP threads walk the row a float4 at a time.
/// Three passes: the mean, the variance around it, the output. Each sum: per thread, then across the warp.
///
///   output_ptr, mean_ptr, rstd_ptr, input_ptr, weight_ptr, bias_ptr   as in layer_norm_forward, float or bfloat16 (T)
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
template <typename T>
__global__ void layer_norm_forward_kernel(T* output_ptr, float* mean_ptr, float* rstd_ptr, const T* input_ptr,
                                          const T* weight_ptr, const T* bias_ptr, size_t rows_val, size_t embed_dim_val) {
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / THREADS_PER_WARP;
    size_t row_float4s_val = embed_dim_val / VALUES_PER_FLOAT4;                                     // float4s in one row: 768 / 4 = 192
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const T* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        T* output_row_ptr = output_ptr + row_index_val * embed_dim_val;

        // 1. the mean
        float input_sum_val = 0.0f;
        for (size_t i_val = lane_index_val; i_val < row_float4s_val; i_val += THREADS_PER_WARP) {
            float4 input_val = read4_as_float(input_row_ptr + VALUES_PER_FLOAT4 * i_val);
            input_sum_val += input_val.x + input_val.y + input_val.z + input_val.w;
        }
        float row_mean_val = warp_reduce_sum(input_sum_val) / (float)embed_dim_val;

        // 2. the variance, then rstd
        float squares_sum_val = 0.0f;
        for (size_t i_val = lane_index_val; i_val < row_float4s_val; i_val += THREADS_PER_WARP) {
            float4 input_val = read4_as_float(input_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 centered_val = make_float4(input_val.x - row_mean_val, input_val.y - row_mean_val,
                                              input_val.z - row_mean_val, input_val.w - row_mean_val);
            squares_sum_val += centered_val.x * centered_val.x + centered_val.y * centered_val.y +
                               centered_val.z * centered_val.z + centered_val.w * centered_val.w;
        }
        float row_variance_val = warp_reduce_sum(squares_sum_val) / (float)embed_dim_val;
        float row_rstd_val = rsqrtf(row_variance_val + LAYER_NORM_EPS);

        // 3. norm, then the output
        for (size_t i_val = lane_index_val; i_val < row_float4s_val; i_val += THREADS_PER_WARP) {
            float4 input_val = read4_as_float(input_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 weight_val = read4_as_float(weight_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 bias_val = read4_as_float(bias_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 norm_val = make_float4((input_val.x - row_mean_val) * row_rstd_val, (input_val.y - row_mean_val) * row_rstd_val,
                                          (input_val.z - row_mean_val) * row_rstd_val, (input_val.w - row_mean_val) * row_rstd_val);
            write4_from_float(output_row_ptr + VALUES_PER_FLOAT4 * i_val,
                              make_float4(norm_val.x * weight_val.x + bias_val.x, norm_val.y * weight_val.y + bias_val.y,
                                          norm_val.z * weight_val.z + bias_val.z, norm_val.w * weight_val.w + bias_val.w));
        }
        // 4. lane 0 keeps the row's mean and rstd for the backward
        if (lane_index_val == 0) {
            mean_ptr[row_index_val] = row_mean_val;
            rstd_ptr[row_index_val] = row_rstd_val;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void layer_norm_backward_kernel(T* input_grad_ptr, T* weight_grad_ptr, T* bias_grad_ptr,
                                           const T* output_grad_ptr, const T* input_ptr, const T* weight_ptr,
                                           const float* mean_ptr, const float* rstd_ptr, size_t rows_val,
                                           size_t embed_dim_val);

/// Adds the gradients of input, weight and bias onto what is there: checks, then launches
/// layer_norm_backward_kernel. weight_grad and bias_grad must be 0 before a step's first backward.
///
///   input_grad_tensor                                    input's shape; added onto: the residual stream's gradient
///   weight_grad_tensor, bias_grad_tensor                 {embed_dim}
///   output_grad_tensor                                   the gradient from the layer above
///   input_tensor, weight_tensor, mean_tensor, rstd_tensor   the forward's
///   threads_per_block_val                                the block size: whole warps, at most MAX_THREADS_PER_BLOCK;
///                                                        not given: LAYER_NORM_BACKWARD_THREADS_PER_BLOCK. input_grad
///                                                        never changes with it; weight_grad's and bias_grad's adds
///                                                        can land in another order, as they can from run to run.
inline void layer_norm_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& mean_tensor,
                                const Tensor& rstd_tensor, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    // block size: whole warps, as each warp takes a row; at most MAX_THREADS_PER_BLOCK
    check_threads_per_block(threads_per_block_val, "LayerNorm");
    layer_norm_check_shapes(output_grad_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_grad_tensor);
    layer_norm_check_shapes(input_grad_tensor, mean_tensor, rstd_tensor, input_tensor, weight_grad_tensor, bias_grad_tensor);
    // 2. launch: LAYER_NORM_BACKWARD_BLOCKS blocks, one warp per row, the block's bias_grad and weight_grad in
    //    shared memory
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    int blocks_val = LAYER_NORM_BACKWARD_BLOCKS;
    size_t shared_bytes_val = 2 * embed_dim_val * sizeof(float);
    if (input_tensor.dtype() == DType::BFloat16)
        layer_norm_backward_kernel<<<blocks_val, threads_per_block_val, shared_bytes_val>>>(
            input_grad_tensor.data_ptr<nv_bfloat16>(), weight_grad_tensor.data_ptr<nv_bfloat16>(),
            bias_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
            input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(), mean_tensor.data(),
            rstd_tensor.data(), rows_val, embed_dim_val);
    else
        layer_norm_backward_kernel<<<blocks_val, threads_per_block_val, shared_bytes_val>>>(
            input_grad_tensor.data(), weight_grad_tensor.data(), bias_grad_tensor.data(), output_grad_tensor.data(),
            input_tensor.data(), weight_tensor.data(), mean_tensor.data(), rstd_tensor.data(), rows_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Works out the three gradients, one warp per row. Two passes per row: the two means, then the gradients.
/// weight_grad and bias_grad are sums over all rows: each block adds its rows into shared memory first, then
/// once onto GPU memory at the end. The adds come in any order, so the last bits can differ run to run.
///
///   input_grad_ptr, weight_grad_ptr, bias_grad_ptr, output_grad_ptr   as in layer_norm_backward, float or bfloat16 (T)
///   input_ptr, weight_ptr, mean_ptr, rstd_ptr                         the forward's
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
template <typename T>
__global__ void layer_norm_backward_kernel(T* input_grad_ptr, T* weight_grad_ptr, T* bias_grad_ptr,
                                           const T* output_grad_ptr, const T* input_ptr, const T* weight_ptr,
                                           const float* mean_ptr, const float* rstd_ptr, size_t rows_val,
                                           size_t embed_dim_val) {
    // 1. the block's bias_grad and weight_grad in shared memory, 2 x embed_dim, all 0
    extern __shared__ float block_sums_list[];
    float* block_bias_grad_ptr = block_sums_list;
    float* block_weight_grad_ptr = block_sums_list + embed_dim_val;
    for (size_t i_val = threadIdx.x; i_val < 2 * embed_dim_val; i_val += blockDim.x) block_sums_list[i_val] = 0.0f;
    __syncthreads();

    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / THREADS_PER_WARP;
    size_t row_float4s_val = embed_dim_val / VALUES_PER_FLOAT4;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const T* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        const T* output_grad_row_ptr = output_grad_ptr + row_index_val * embed_dim_val;
        T* input_grad_row_ptr = input_grad_ptr + row_index_val * embed_dim_val;
        float row_mean_val = mean_ptr[row_index_val];
        float row_rstd_val = rstd_ptr[row_index_val];

        // 2. mean(norm_grad) and mean(norm_grad · norm)
        float norm_grad_sum_val = 0.0f;
        float norm_grad_norm_sum_val = 0.0f;
        for (size_t i_val = lane_index_val; i_val < row_float4s_val; i_val += THREADS_PER_WARP) {
            float4 input_val = read4_as_float(input_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 output_grad_val = read4_as_float(output_grad_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 weight_val = read4_as_float(weight_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 norm_val = make_float4((input_val.x - row_mean_val) * row_rstd_val, (input_val.y - row_mean_val) * row_rstd_val,
                                          (input_val.z - row_mean_val) * row_rstd_val, (input_val.w - row_mean_val) * row_rstd_val);
            float4 norm_grad_val = make_float4(output_grad_val.x * weight_val.x, output_grad_val.y * weight_val.y,
                                               output_grad_val.z * weight_val.z, output_grad_val.w * weight_val.w);
            norm_grad_sum_val += norm_grad_val.x + norm_grad_val.y + norm_grad_val.z + norm_grad_val.w;
            norm_grad_norm_sum_val += norm_grad_val.x * norm_val.x + norm_grad_val.y * norm_val.y +
                                      norm_grad_val.z * norm_val.z + norm_grad_val.w * norm_val.w;
        }
        float norm_grad_mean_val = warp_reduce_sum(norm_grad_sum_val) / (float)embed_dim_val;
        float norm_grad_norm_mean_val = warp_reduce_sum(norm_grad_norm_sum_val) / (float)embed_dim_val;

        // 3. the three gradients
        for (size_t i_val = lane_index_val; i_val < row_float4s_val; i_val += THREADS_PER_WARP) {
            float4 input_val = read4_as_float(input_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 output_grad_val = read4_as_float(output_grad_row_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 weight_val = read4_as_float(weight_ptr + VALUES_PER_FLOAT4 * i_val);
            float4 norm_val = make_float4((input_val.x - row_mean_val) * row_rstd_val, (input_val.y - row_mean_val) * row_rstd_val,
                                          (input_val.z - row_mean_val) * row_rstd_val, (input_val.w - row_mean_val) * row_rstd_val);
            float4 norm_grad_val = make_float4(output_grad_val.x * weight_val.x, output_grad_val.y * weight_val.y,
                                               output_grad_val.z * weight_val.z, output_grad_val.w * weight_val.w);
            float* bias_grad_columns_ptr = block_bias_grad_ptr + i_val * VALUES_PER_FLOAT4;
            float* weight_grad_columns_ptr = block_weight_grad_ptr + i_val * VALUES_PER_FLOAT4;
            // bias_grad += output_grad, into the block's copy
            atomicAdd(bias_grad_columns_ptr + 0, output_grad_val.x);
            atomicAdd(bias_grad_columns_ptr + 1, output_grad_val.y);
            atomicAdd(bias_grad_columns_ptr + 2, output_grad_val.z);
            atomicAdd(bias_grad_columns_ptr + 3, output_grad_val.w);
            // weight_grad += output_grad · norm, into the block's copy
            atomicAdd(weight_grad_columns_ptr + 0, output_grad_val.x * norm_val.x);
            atomicAdd(weight_grad_columns_ptr + 1, output_grad_val.y * norm_val.y);
            atomicAdd(weight_grad_columns_ptr + 2, output_grad_val.z * norm_val.z);
            atomicAdd(weight_grad_columns_ptr + 3, output_grad_val.w * norm_val.w);
            // input_grad += rstd · (norm_grad - mean(norm_grad) - norm · mean(norm_grad · norm)); one warp per row
            float4 old_val = read4_as_float(input_grad_row_ptr + VALUES_PER_FLOAT4 * i_val);
            write4_from_float(input_grad_row_ptr + VALUES_PER_FLOAT4 * i_val,
                   make_float4(old_val.x + row_rstd_val * (norm_grad_val.x - norm_grad_mean_val - norm_val.x * norm_grad_norm_mean_val),
                               old_val.y + row_rstd_val * (norm_grad_val.y - norm_grad_mean_val - norm_val.y * norm_grad_norm_mean_val),
                               old_val.z + row_rstd_val * (norm_grad_val.z - norm_grad_mean_val - norm_val.z * norm_grad_norm_mean_val),
                               old_val.w + row_rstd_val * (norm_grad_val.w - norm_grad_mean_val - norm_val.w * norm_grad_norm_mean_val)));
        }
    }

    // 4. the block's bias_grad and weight_grad, added onto GPU memory
    __syncthreads();
    for (size_t i_val = threadIdx.x; i_val < row_float4s_val; i_val += blockDim.x) {
        const float* bias_sums_ptr = block_bias_grad_ptr + VALUES_PER_FLOAT4 * i_val;
        const float* weight_sums_ptr = block_weight_grad_ptr + VALUES_PER_FLOAT4 * i_val;
        atomic_add4_from_float(bias_grad_ptr + VALUES_PER_FLOAT4 * i_val,
                               make_float4(bias_sums_ptr[0], bias_sums_ptr[1], bias_sums_ptr[2], bias_sums_ptr[3]));
        atomic_add4_from_float(weight_grad_ptr + VALUES_PER_FLOAT4 * i_val,
                               make_float4(weight_sums_ptr[0], weight_sums_ptr[1], weight_sums_ptr[2], weight_sums_ptr[3]));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit: the shapes at the top of the file, the dtypes, embed_dim a
/// multiple of VALUES_PER_FLOAT4 and small enough for the backward's shared memory, each starting on a whole
/// float4.
///
///   output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor
///       forward: these; backward: the gradients in their place
inline void layer_norm_check_shapes(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                                    const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor) {
    // shapes: as at the top of the file
    bool fits_val = input_tensor.dim() == 3 && output_tensor.shape() == input_tensor.shape() &&
                    mean_tensor.shape() == Shape({input_tensor.size(0), input_tensor.size(1)}) &&
                    rstd_tensor.shape() == mean_tensor.shape() && weight_tensor.shape() == Shape({input_tensor.size(2)}) &&
                    bias_tensor.shape() == weight_tensor.shape();
    if (!fits_val) {
        fprintf(stderr, "error: LayerNorm needs input and output {batch_size, seq_len, embed_dim}, mean and rstd "
                        "{batch_size, seq_len}, weight and bias {embed_dim}\n");
        exit(EXIT_FAILURE);
    }
    // types: input, output, weight, bias all float or all bfloat16; mean and rstd float
    DType dtype_val = input_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || output_tensor.dtype() != dtype_val ||
        weight_tensor.dtype() != dtype_val || bias_tensor.dtype() != dtype_val || mean_tensor.dtype() != DType::Float32 ||
        rstd_tensor.dtype() != DType::Float32) {
        fprintf(stderr, "error: LayerNorm needs input, output, weight and bias all float or all bfloat16, and mean "
                        "and rstd float\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: rows of whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * input_tensor.element_size();
    if (input_tensor.size(2) % VALUES_PER_FLOAT4 != 0 || (size_t)input_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)output_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)weight_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)bias_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: LayerNorm reads %d values at a time, so embed_dim must be a multiple of %d and input, "
                        "output, weight, bias must start on %zu bytes; got embed_dim %zu, %p, %p, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, input_tensor.size(2), input_tensor.data_ptr(),
                output_tensor.data_ptr(), weight_tensor.data_ptr(), bias_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
    // shared memory: the backward keeps 2 x embed_dim floats per block
    size_t largest_embed_dim_val = LAYER_NORM_MAX_SHARED_BYTES / (2 * sizeof(float));
    if (input_tensor.size(2) > largest_embed_dim_val) {
        fprintf(stderr, "error: LayerNorm's backward keeps 2 x embed_dim floats per block in shared memory, at most "
                        "%d bytes, so embed_dim must be at most %zu; got %zu\n",
                LAYER_NORM_MAX_SHARED_BYTES, largest_embed_dim_val, input_tensor.size(2));
        exit(EXIT_FAILURE);
    }
}

#endif // LAYER_NORM_CUH
