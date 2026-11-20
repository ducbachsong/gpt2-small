// layer_norm_bf16.cuh — LayerNorm in bfloat16.
//
//     tensor                                type in memory   shape
//     input, output, input_grad, output_grad bfloat16         {batch_size, seq_len, embed_dim}
//     weight, bias, and their grads          bfloat16         {embed_dim}
//     mean, rstd                             float            {batch_size, seq_len}
#ifndef LAYER_NORM_BF16_CUH
#define LAYER_NORM_BF16_CUH

#include "../cuda_common.h"          // cudaCheck, check_threads_per_block, warp_reduce_sum
#include "../../common/tensor.cuh"          // read8_as_float, write8_from_float, VALUES_PER_BF16_LOAD

#define LAYER_NORM_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD                          // bfloat16s in one 16-byte load: 8
#define LAYER_NORM_BF16_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)             // the forward's default: fp32/layer_norm.cuh's, not swept for 8 values yet
#define LAYER_NORM_BF16_BACKWARD_BLOCKS 80                                            // backward: blocks in every launch, as fp32/layer_norm.cuh's
#define LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK (16 * HARDWARE_THREADS_PER_WARP)   // the backward's default, as fp32/layer_norm.cuh's
#define LAYER_NORM_BF16_MAX_LOADS_PER_LANE 9                                       // backward: loads a lane takes per row, at most: embed_dim up to 9 x 32 x 8 = 2304 (the 32 x 2,304 model's); its sums stay in registers, one kernel made for each count (768: 3)
#define LAYER_NORM_BF16_EPS 1e-5f                                                     // added to the variance, as GPT-2 does
#define LAYER_NORM_BF16_SUMS_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)     // the backward's second kernel: one thread per value of bias_grad and weight_grad

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_bf16_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                             const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                             int threads_per_block_val = LAYER_NORM_BF16_THREADS_PER_BLOCK);
void layer_norm_bf16_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                              const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                              const Tensor& mean_tensor, const Tensor& rstd_tensor,
                              int threads_per_block_val = LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK);
void layer_norm_bf16_backward_launch(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                     const Tensor& mean_tensor, const Tensor& rstd_tensor, int blocks_val, int threads_per_block_val);

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

/// Writes output = LayerNorm(input) · weight + bias in bfloat16, and each row's mean and rstd.
///
///   output_tensor, input_tensor   {batch_size, seq_len, embed_dim} bfloat16
///   mean_tensor, rstd_tensor      {batch_size, seq_len} float: for the backward
///   weight_tensor, bias_tensor    {embed_dim} bfloat16
///   threads_per_block_val         the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                 LAYER_NORM_BF16_THREADS_PER_BLOCK. Any size gives the same bits; only the time changes.
inline void layer_norm_bf16_forward(const Tensor& output_tensor, const Tensor& mean_tensor, const Tensor& rstd_tensor,
                                    const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor,
                                    int threads_per_block_val) {
    // Checks, then launches layer_norm_bf16_forward_kernel, one warp per row.
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

/// Normalizes the rows, one warp per row.
///
///   output_ptr, mean_ptr, rstd_ptr, input_ptr, weight_ptr, bias_ptr   as in layer_norm_bf16_forward
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
__global__ void layer_norm_bf16_forward_kernel(nv_bfloat16* output_ptr, float* mean_ptr, float* rstd_ptr, const nv_bfloat16* input_ptr,
                                               const nv_bfloat16* weight_ptr, const nv_bfloat16* bias_ptr, size_t rows_val,
                                               size_t embed_dim_val) {
    // The warp's lanes walk the row 8 values (one 16-byte load) at a time. Three passes: the mean and the variance
    // around it (layer_norm_bf16_row_mean_rstd), then the output.
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

template <int LOADS_PER_LANE>
__global__ void layer_norm_bf16_backward_kernel(nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                                const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                                float* block_sums_ptr, size_t rows_val, size_t embed_dim_val);
template <int LOADS_PER_LANE>
void layer_norm_bf16_backward_kernel_launch(int blocks_val, int threads_per_block_val, size_t shared_bytes_val,
                                            nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                            const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                            const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                            float* block_sums_ptr, size_t rows_val, size_t embed_dim_val);
__global__ void layer_norm_bf16_backward_sums_kernel(nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                     const float* block_sums_ptr, int blocks_val, size_t embed_dim_val);

/// Adds the gradients of input, weight and bias onto what is there, in bfloat16.
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
    // Checks, then layer_norm_bf16_backward_launch with LAYER_NORM_BF16_BACKWARD_BLOCKS blocks. weight_grad and
    // bias_grad must be 0 before a step's first backward.
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "layer_norm_bf16_backward");
    layer_norm_bf16_check_shapes("layer_norm_bf16_backward", output_grad_tensor, mean_tensor, rstd_tensor, input_tensor,
                                 weight_tensor, bias_grad_tensor);
    layer_norm_bf16_check_shapes("layer_norm_bf16_backward", input_grad_tensor, mean_tensor, rstd_tensor, input_tensor,
                                 weight_grad_tensor, bias_grad_tensor);
    if (input_tensor.size(2) > (size_t)LAYER_NORM_BF16_MAX_LOADS_PER_LANE * HARDWARE_THREADS_PER_WARP * LAYER_NORM_BF16_VALUES_PER_LOAD) {
        fprintf(stderr, "error: layer_norm_bf16_backward keeps a lane's sums in registers, %d loads a row at most: embed_dim "
                        "up to %d; got %zu\n", LAYER_NORM_BF16_MAX_LOADS_PER_LANE,
                LAYER_NORM_BF16_MAX_LOADS_PER_LANE * HARDWARE_THREADS_PER_WARP * LAYER_NORM_BF16_VALUES_PER_LOAD, input_tensor.size(2));
        exit(EXIT_FAILURE);
    }
    // 2. launch: LAYER_NORM_BF16_BACKWARD_BLOCKS blocks
    layer_norm_bf16_backward_launch(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                                    weight_tensor, mean_tensor, rstd_tensor, LAYER_NORM_BF16_BACKWARD_BLOCKS, threads_per_block_val);
}

/// Launches the backward's two kernels, the tensors already checked.
///
///   blocks_val               the first kernel's blocks: LAYER_NORM_BF16_BACKWARD_BLOCKS, or a tune's
///   threads_per_block_val    its block size
inline void layer_norm_bf16_backward_launch(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor,
                                            const Tensor& bias_grad_tensor, const Tensor& output_grad_tensor,
                                            const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& mean_tensor,
                                            const Tensor& rstd_tensor, int blocks_val, int threads_per_block_val) {
    // Layer_norm_bf16_backward_kernel (blocks_val blocks, one warp per row, each block's bias_grad and weight_grad
    // sums into its own row of a scratch in GPU memory), then layer_norm_bf16_backward_sums_kernel (the rows added
    // up in float, then onto bias_grad and weight_grad, rounded once). The scratch, blocks x 2 x embed_dim floats,
    // is made the first time and grows with the largest launch.
    //
    //   the tensors              as in layer_norm_bf16_backward
    // 1. the scratch: one row of 2 x embed_dim float sums per block
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    static Tensor block_sums_tensor;
    if ((size_t)block_sums_tensor.numel() < (size_t)blocks_val * 2 * embed_dim_val)
        block_sums_tensor = Tensor::zeros({(size_t)blocks_val * 2 * embed_dim_val});
    // 2. the gradients: input_grad written, each block's sums into its row, by the kernel made for the row's loads
    //    a lane (768: 3, 2,304: 9)
    using KernelLaunch = void (*)(int, int, size_t, nv_bfloat16*, nv_bfloat16*, nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*,
                                  const nv_bfloat16*, const float*, const float*, float*, size_t, size_t);
    static const KernelLaunch launches_list[LAYER_NORM_BF16_MAX_LOADS_PER_LANE] = {
        layer_norm_bf16_backward_kernel_launch<1>, layer_norm_bf16_backward_kernel_launch<2>, layer_norm_bf16_backward_kernel_launch<3>,
        layer_norm_bf16_backward_kernel_launch<4>, layer_norm_bf16_backward_kernel_launch<5>, layer_norm_bf16_backward_kernel_launch<6>,
        layer_norm_bf16_backward_kernel_launch<7>, layer_norm_bf16_backward_kernel_launch<8>, layer_norm_bf16_backward_kernel_launch<9>};
    static_assert(LAYER_NORM_BF16_MAX_LOADS_PER_LANE == 9, "one launch above for each count up to the most");
    int loads_per_lane_val = (int)CEIL_DIV(embed_dim_val / LAYER_NORM_BF16_VALUES_PER_LOAD, (size_t)HARDWARE_THREADS_PER_WARP);
    size_t shared_bytes_val = 2 * embed_dim_val * sizeof(float);
    launches_list[loads_per_lane_val - 1](
        blocks_val, threads_per_block_val, shared_bytes_val, input_grad_tensor.data_ptr<nv_bfloat16>(),
        weight_grad_tensor.data_ptr<nv_bfloat16>(), bias_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
        input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(), mean_tensor.data(), rstd_tensor.data(),
        block_sums_tensor.data(), rows_val, embed_dim_val);
    // 3. the blocks' rows added up, onto bias_grad and weight_grad: one thread per value
    layer_norm_bf16_backward_sums_kernel<<<(unsigned)CEIL_DIV(2 * embed_dim_val, (size_t)LAYER_NORM_BF16_SUMS_THREADS_PER_BLOCK),
                                           LAYER_NORM_BF16_SUMS_THREADS_PER_BLOCK>>>(
        weight_grad_tensor.data_ptr<nv_bfloat16>(), bias_grad_tensor.data_ptr<nv_bfloat16>(), block_sums_tensor.data(), blocks_val,
        embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Works out the three gradients, one warp per row, 8 values a lane at a time.
///
///   input_grad_ptr, weight_grad_ptr, bias_grad_ptr, output_grad_ptr   as in layer_norm_bf16_backward
///   input_ptr, weight_ptr, mean_ptr, rstd_ptr                         the forward's
///   block_sums_ptr                                                    {blocks, 2, embed_dim}: this block's row written
///   rows_val                                                          batch_size x seq_len
///   embed_dim_val                                                     the row length: 768
template <int LOADS_PER_LANE>
__global__ void layer_norm_bf16_backward_kernel(nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                                const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                                float* block_sums_ptr, size_t rows_val, size_t embed_dim_val) {
    // Two passes per row: the two means, then the gradients. weight_grad and bias_grad are sums over all rows. A
    // lane takes the same columns in every row (loads lane, lane + 32, ...), so it keeps its columns' sums in
    // registers over all the rows its warp takes; then each warp adds them into the block's copy in shared memory
    // (once a warp, not twice a value), and each block writes its copy to its own row of block_sums, in float: no
    // atomics in GPU memory (blocks adding bfloat16s onto the same 2 x embed_dim values took most of the time,
    // more the more blocks). layer_norm_bf16_backward_sums_kernel then adds the rows up. Within a block the warps'
    // adds come in any order, so the last bits can differ run to run.
    //
    // LOADS_PER_LANE: the loads a lane takes in a row, at least embed_dim / (8 x 32): the registers its sums take.
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
    // this lane's columns' bias_grad and weight_grad over the rows its warp takes, in registers: its loads are
    // lane, lane + HARDWARE_THREADS_PER_WARP, ... in every row
    float lane_bias_grad_list[LOADS_PER_LANE][LAYER_NORM_BF16_VALUES_PER_LOAD] = {};
    float lane_weight_grad_list[LOADS_PER_LANE][LAYER_NORM_BF16_VALUES_PER_LOAD] = {};
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

        // 3. the three gradients: input_grad written; bias_grad and weight_grad into the lane's registers
#pragma unroll
        for (int lane_load_index_val = 0; lane_load_index_val < LOADS_PER_LANE; lane_load_index_val++) {
            size_t load_index_val = lane_index_val + (size_t)lane_load_index_val * HARDWARE_THREADS_PER_WARP;
            if (load_index_val >= row_loads_val) continue;
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
                // bias_grad += output_grad and weight_grad += output_grad · norm, into the lane's registers
                lane_bias_grad_list[lane_load_index_val][value_index_val] += output_grad_list[value_index_val];
                lane_weight_grad_list[lane_load_index_val][value_index_val] += output_grad_list[value_index_val] * norm_val;
                // input_grad += rstd · (norm_grad - mean(norm_grad) - norm · mean(norm_grad · norm)); one warp per row
                input_grad_list[value_index_val] += row_rstd_val * (norm_grad_val - norm_grad_mean_val - norm_val * norm_grad_norm_mean_val);
            }
            write8_from_float(input_grad_row_ptr + first_index_val, input_grad_list);
        }
    }

    // 4. each lane's sums into the block's copy: once a warp
#pragma unroll
    for (int lane_load_index_val = 0; lane_load_index_val < LOADS_PER_LANE; lane_load_index_val++) {
        size_t load_index_val = lane_index_val + (size_t)lane_load_index_val * HARDWARE_THREADS_PER_WARP;
        if (load_index_val >= row_loads_val) continue;
        size_t first_index_val = LAYER_NORM_BF16_VALUES_PER_LOAD * load_index_val;
#pragma unroll
        for (int value_index_val = 0; value_index_val < LAYER_NORM_BF16_VALUES_PER_LOAD; value_index_val++) {
            atomicAdd(block_bias_grad_ptr + first_index_val + value_index_val, lane_bias_grad_list[lane_load_index_val][value_index_val]);
            atomicAdd(block_weight_grad_ptr + first_index_val + value_index_val, lane_weight_grad_list[lane_load_index_val][value_index_val]);
        }
    }

    // 5. the block's bias_grad and weight_grad, into its own row of block_sums
    __syncthreads();
    float* block_row_ptr = block_sums_ptr + (size_t)blockIdx.x * 2 * embed_dim_val;
    for (size_t value_index_val = threadIdx.x; value_index_val < 2 * embed_dim_val; value_index_val += blockDim.x)
        block_row_ptr[value_index_val] = block_sums_list[value_index_val];
}

/// Launches layer_norm_bf16_backward_kernel<LOADS_PER_LANE>, its block no larger than its registers allow.
///
///   blocks_val, threads_per_block_val, shared_bytes_val   the launch asked for
template <int LOADS_PER_LANE>
inline void layer_norm_bf16_backward_kernel_launch(int blocks_val, int threads_per_block_val, size_t shared_bytes_val,
                                                   nv_bfloat16* input_grad_ptr, nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                   const nv_bfloat16* output_grad_ptr, const nv_bfloat16* input_ptr,
                                                   const nv_bfloat16* weight_ptr, const float* mean_ptr, const float* rstd_ptr,
                                                   float* block_sums_ptr, size_t rows_val, size_t embed_dim_val) {
    // A lane's sums take 2 x LOADS_PER_LANE x 8 registers (144 at 9 loads), so the 9-load kernel runs at most 256
    // or so threads a block where the 3-load one runs 512. The most is asked of CUDA once (cudaFuncGetAttributes),
    // and the asked block size is lowered to it, whole warps; input_grad's bits do not change with the block size.
    //
    //   the pointers and sizes                               as layer_norm_bf16_backward_kernel takes them
    // 1. the most threads a block of this kernel can have, whole warps: asked once
    static int most_threads_val = [] {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, layer_norm_bf16_backward_kernel<LOADS_PER_LANE>));
        return attributes_val.maxThreadsPerBlock / HARDWARE_THREADS_PER_WARP * HARDWARE_THREADS_PER_WARP;
    }();
    // 2. launch at the asked size, or the most
    int threads_val = threads_per_block_val < most_threads_val ? threads_per_block_val : most_threads_val;
    layer_norm_bf16_backward_kernel<LOADS_PER_LANE><<<blocks_val, threads_val, shared_bytes_val>>>(
        input_grad_ptr, weight_grad_ptr, bias_grad_ptr, output_grad_ptr, input_ptr, weight_ptr, mean_ptr, rstd_ptr,
        block_sums_ptr, rows_val, embed_dim_val);
    cudaCheck(cudaGetLastError());
}

/// Adds the blocks' rows of block_sums up, in float, then onto bias_grad and weight_grad.
///
///   weight_grad_ptr, bias_grad_ptr   {embed_dim} bfloat16: added onto
///   block_sums_ptr                   {blocks, 2, embed_dim}: each block's bias_grad, then weight_grad
///   blocks_val                       the rows
///   embed_dim_val                    the row length
__global__ void layer_norm_bf16_backward_sums_kernel(nv_bfloat16* weight_grad_ptr, nv_bfloat16* bias_grad_ptr,
                                                     const float* block_sums_ptr, int blocks_val, size_t embed_dim_val) {
    // Each value rounded to bfloat16 once: one thread per value, the blocks in order, so the sum is the same from
    // run to run.
    // 1. this thread's value: bias_grad's first, then weight_grad's
    size_t value_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (value_index_val >= 2 * embed_dim_val) return;
    // 2. the blocks' sums of it, in float
    float sum_val = 0.0f;
    for (int block_index_val = 0; block_index_val < blocks_val; block_index_val++)
        sum_val += block_sums_ptr[(size_t)block_index_val * 2 * embed_dim_val + value_index_val];
    // 3. onto the gradient, rounded once
    nv_bfloat16* grad_ptr = value_index_val < embed_dim_val ? bias_grad_ptr + value_index_val
                                                          : weight_grad_ptr + (value_index_val - embed_dim_val);
    *grad_ptr = __float2bfloat16(__bfloat162float(*grad_ptr) + sum_val);
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Works out one row's mean and rstd with its warp, 8 values a lane at a time.
///
///   row_mean_ptr, row_rstd_ptr   written: the row's mean and 1 / sqrt(variance + LAYER_NORM_BF16_EPS)
///   input_row_ptr                the row, embed_dim bfloat16s on 16 bytes
///   lane_index_val               this lane of the row's warp
///   embed_dim_val                the row length: 768
__device__ __forceinline__ void layer_norm_bf16_row_mean_rstd(float* row_mean_ptr, float* row_rstd_ptr, const nv_bfloat16* input_row_ptr,
                                                              int lane_index_val, size_t embed_dim_val) {
    // The mean, then the variance around it (two passes, as fp32/layer_norm.cuh). Every lane gets both. Also
    // layer_norm_8bit.cuh's.
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

/// Stops with a message unless the tensors fit.
///
///   caller_ptr                                                                the function checking, for the message
///   output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor
///       forward: these; backward: the gradients in their place
inline void layer_norm_bf16_check_shapes(const char* caller_ptr, const Tensor& output_tensor, const Tensor& mean_tensor,
                                         const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                         const Tensor& bias_tensor) {
    // The shapes at the top of the file, input, output, weight and bias bfloat16, mean and rstd float, embed_dim a
    // multiple of LAYER_NORM_BF16_VALUES_PER_LOAD and small enough for the backward's shared memory, each bfloat16
    // tensor starting on 16 bytes.
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
