// grad_norm_bf16.cuh — the norm of all the bfloat16 gradients together, as one long list.
//
//     tensor          type in memory   shape
//     grads           bfloat16         flat
//     block_sums      float            {2, GRAD_NORM_BF16_BLOCKS}
//     grad_norm       float            {1}
#ifndef GRAD_NORM_BF16_CUH
#define GRAD_NORM_BF16_CUH

#include "../cuda_common.h"          // cudaCheck, check_threads_per_block, block_reduce_sum
#include "../../common/tensor.cuh"          // read8_as_float, VALUES_PER_BF16_LOAD

#define GRAD_NORM_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD       // bfloat16s in one 16-byte load: 8
#define GRAD_NORM_BF16_BLOCKS 512                                 // blocks in the launch, each writes one block sum: fp32/grad_norm.cuh's, not swept for 8 values yet
#define GRAD_NORM_BF16_THREADS HARDWARE_MAX_THREADS_PER_BLOCK     // the default block size, as fp32/grad_norm.cuh's
#define GRAD_NORM_BF16_LOAD_BYTES sizeof(uint4)                   // bytes in one load: 16

class GradNormBf16 {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    explicit GradNormBf16(const Tensor& block_sums_tensor);             // {2, GRAD_NORM_BF16_BLOCKS}: the block sums, then the counter

    // ── the norm ────────────────────────────────────────────────────────────────
    // grad_norm (1 value) = sqrt(sum of every grad²), grads bfloat16
    void compute(const Tensor& grads_tensor, const Tensor& grad_norm_tensor,
                 int threads_per_block_val = GRAD_NORM_BF16_THREADS) const;

private:
    Tensor block_sums_tensor_;           // row 0: each block's sum of squares
    Tensor blocks_done_tensor_;          // row 1, value 0: blocks that have written their sum; 0 between computes
};

// ══ the code, in the order of the list above ══════════════════════════════════

/// Makes a GradNormBf16 on a {2, GRAD_NORM_BF16_BLOCKS} float view of a buffer; it allocates nothing.
///
///   block_sums_tensor   row 0: the GRAD_NORM_BF16_BLOCKS block sums; row 1, value 0: the counter, 0 at the start
inline GradNormBf16::GradNormBf16(const Tensor& block_sums_tensor) {
    if (block_sums_tensor.shape() != Shape({2, GRAD_NORM_BF16_BLOCKS}) || block_sums_tensor.dtype() != DType::Float32) {
        fprintf(stderr, "error: GradNormBf16 needs float block sums of shape {2, %d}, but got %zu values\n", GRAD_NORM_BF16_BLOCKS,
                block_sums_tensor.numel());
        exit(EXIT_FAILURE);
    }
    block_sums_tensor_ = block_sums_tensor[0];
    blocks_done_tensor_ = block_sums_tensor[1][0];
}

// ── the norm ──────────────────────────────────────────────────────────────────

__global__ void grad_norm_bf16_kernel(float* grad_norm_ptr, float* block_sums_ptr, float* blocks_done_ptr, const nv_bfloat16* grads_ptr,
                                      size_t grads_numel_val, size_t head_values_val);

/// Writes the norm of all of grads into grad_norm.
///
///   grads_tensor            every gradient, bfloat16, starting anywhere
///   grad_norm_tensor        1 float: the result
///   threads_per_block_val   the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                           GRAD_NORM_BF16_THREADS. Another size adds the squares in another order, so the last bits
///                           can change; the same size gives the same bits every run.
inline void GradNormBf16::compute(const Tensor& grads_tensor, const Tensor& grad_norm_tensor, int threads_per_block_val) const {
    // Checks, works out the head (the values before the first 16-byte boundary), then launches
    // grad_norm_bf16_kernel.
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "GradNormBf16");
    if (grad_norm_tensor.numel() != 1 || grad_norm_tensor.dtype() != DType::Float32) {
        fprintf(stderr, "error: GradNormBf16 writes 1 float, but grad_norm has %zu values\n", grad_norm_tensor.numel());
        exit(EXIT_FAILURE);
    }
    if (grads_tensor.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: GradNormBf16 needs grads bfloat16 (float: fp32/grad_norm.cuh's GradNorm)\n");
        exit(EXIT_FAILURE);
    }
    // 2. the head: values up to the first 16-byte boundary (0 .. 7), at most all of them
    size_t past_boundary_bytes_val = (size_t)grads_tensor.data_ptr() % GRAD_NORM_BF16_LOAD_BYTES;
    size_t head_values_val = (GRAD_NORM_BF16_LOAD_BYTES - past_boundary_bytes_val) % GRAD_NORM_BF16_LOAD_BYTES / sizeof(nv_bfloat16);
    if (head_values_val > grads_tensor.numel()) head_values_val = grads_tensor.numel();
    // 3. launch: GRAD_NORM_BF16_BLOCKS blocks, one block sum each
    grad_norm_bf16_kernel<<<GRAD_NORM_BF16_BLOCKS, threads_per_block_val>>>(
        grad_norm_tensor.data(), block_sums_tensor_.data(), blocks_done_tensor_.data(), grads_tensor.data_ptr<nv_bfloat16>(),
        grads_tensor.numel(), head_values_val);
    cudaCheck(cudaGetLastError());
}

/// Works out the norm in one kernel, the same bits every run.
///
///   grad_norm_ptr     1 float: the result
///   block_sums_ptr    one per block
///   blocks_done_ptr   the counter of blocks done: 0 at the start, put back to 0 at the end
///   grads_ptr         the gradients, bfloat16
///   grads_numel_val   how many
///   head_values_val   how many come before the first 16-byte boundary: 0 .. 7
__global__ void grad_norm_bf16_kernel(float* grad_norm_ptr, float* block_sums_ptr, float* blocks_done_ptr, const nv_bfloat16* grads_ptr,
                                      size_t grads_numel_val, size_t head_values_val) {
    // Works out the norm in one kernel, as fp32/grad_norm.cuh's: each block adds its share of squares, 8 values (one
    // 16-byte load) a thread at a time, and the last block to finish adds the block sums in a fixed order and writes
    // the square root, so the same launch gives the same norm every run, to the last bit.
    __shared__ bool is_last_block_val;          // did this block finish last?

    // 1. this thread's squares: whole loads thread_index, + every thread launched, ... from the first 16-byte boundary
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const nv_bfloat16* loads_ptr = grads_ptr + head_values_val;                                    // on 16 bytes
    size_t loads_val = (grads_numel_val - head_values_val) / GRAD_NORM_BF16_VALUES_PER_LOAD;       // whole loads
    float sum_val = 0.0f;
    for (size_t index_val = thread_index_val; index_val < loads_val; index_val += (size_t)gridDim.x * blockDim.x) {
        float grads_list[GRAD_NORM_BF16_VALUES_PER_LOAD];
        read8_as_float(grads_list, loads_ptr + GRAD_NORM_BF16_VALUES_PER_LOAD * index_val);
#pragma unroll
        for (int value_index_val = 0; value_index_val < GRAD_NORM_BF16_VALUES_PER_LOAD; value_index_val++)
            sum_val += grads_list[value_index_val] * grads_list[value_index_val];
    }
    //    the head (before the first load) and the tail (after the last one), 0 .. 7 values each: one each for the
    //    first threads
    size_t tail_start_val = head_values_val + loads_val * GRAD_NORM_BF16_VALUES_PER_LOAD;
    if (thread_index_val < head_values_val) {
        float g_val = to_float(grads_ptr[thread_index_val]);
        sum_val += g_val * g_val;
    }
    if (tail_start_val + thread_index_val < grads_numel_val) {
        float g_val = to_float(grads_ptr[tail_start_val + thread_index_val]);
        sum_val += g_val * g_val;
    }

    // 2. this block's sum; thread 0 writes it, then counts the block done
    float block_total_val = block_reduce_sum(sum_val);
    if (threadIdx.x == 0) {
        block_sums_ptr[blockIdx.x] = block_total_val;
        __threadfence();                        // the block sum reaches GPU memory before the count goes up
        float done_before_val = atomicAdd(blocks_done_ptr, 1.0f);
        is_last_block_val = done_before_val == (float)(gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last_block_val) return;

    // 3. last block only: add every block sum the same way. volatile: read GPU memory, not an old cached copy.
    float total_val = 0.0f;
    for (size_t block_index_val = threadIdx.x; block_index_val < gridDim.x; block_index_val += blockDim.x)
        total_val += reinterpret_cast<volatile float*>(block_sums_ptr)[block_index_val];
    float all_blocks_total_val = block_reduce_sum(total_val);
    // 4. the square root; the counter back to 0, ready for the next compute
    if (threadIdx.x == 0) {
        *grad_norm_ptr = sqrtf(all_blocks_total_val);
        *blocks_done_ptr = 0.0f;
    }
}

#endif // GRAD_NORM_BF16_CUH
