// grad_norm.cuh — the norm of all the gradients together, as one long list:
//     grad_norm = sqrt(sum of every g²)
// The gradients are float or bfloat16 (T); the squares are added in float.
#ifndef GRAD_NORM_CUH
#define GRAD_NORM_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block, block_reduce_sum
#include "../tensor.cuh"

#define GRAD_NORM_BLOCKS 512                     // blocks in the launch: each writes one block sum
#define GRAD_NORM_THREADS MAX_THREADS_PER_BLOCK  // the default block size

class GradNorm {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    explicit GradNorm(const Tensor& block_sums_tensor);                 // {2, GRAD_NORM_BLOCKS}: the block sums, then the counter

    // ── the norm ────────────────────────────────────────────────────────────────
    // grad_norm (1 value) = sqrt(sum of every grad²)
    void compute(const Tensor& grads_tensor, const Tensor& grad_norm_tensor,
                 int threads_per_block_val = GRAD_NORM_THREADS) const;

private:
    Tensor block_sums_tensor_;           // row 0: each block's sum of squares
    Tensor blocks_done_tensor_;          // row 1, value 0: blocks that have written their sum; 0 between computes
};

// ══ the code, in the order of the list above ══════════════════════════════════

/// Makes a GradNorm on a {2, GRAD_NORM_BLOCKS} view of the flat buffer; it allocates nothing.
///
///   block_sums_tensor   row 0: the GRAD_NORM_BLOCKS block sums; row 1, value 0: the counter, 0 at the start
inline GradNorm::GradNorm(const Tensor& block_sums_tensor) {
    if (block_sums_tensor.shape() != Shape({2, GRAD_NORM_BLOCKS})) {
        fprintf(stderr, "error: GradNorm needs block sums of shape {2, %d}, but got %zu values\n", GRAD_NORM_BLOCKS,
                block_sums_tensor.numel());
        exit(EXIT_FAILURE);
    }
    block_sums_tensor_ = block_sums_tensor[0];
    blocks_done_tensor_ = block_sums_tensor[1][0];
}

// ── the norm ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void compute_grad_norm_kernel(float* grad_norm_ptr, float* block_sums_ptr, float* blocks_done_ptr,
                                         const T* grads_ptr, size_t grads_numel_val);

/// Writes the norm of all of grads into grad_norm: checks, then launches compute_grad_norm_kernel.
///
///   grads_tensor            every gradient, float or bfloat16, starting on a whole float4
///   grad_norm_tensor        1 float: the result
///   threads_per_block_val   the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                           GRAD_NORM_THREADS. Another size adds the squares in another order, so the last bits
///                           can change; the same size gives the same bits every run.
inline void GradNorm::compute(const Tensor& grads_tensor, const Tensor& grad_norm_tensor, int threads_per_block_val) const {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    // block size: whole warps, for block_reduce_sum; at most MAX_THREADS_PER_BLOCK
    check_threads_per_block(threads_per_block_val, "GradNorm");
    // grad_norm: one value
    if (grad_norm_tensor.numel() != 1) {
        fprintf(stderr, "error: GradNorm writes 1 value, but grad_norm has %zu\n", grad_norm_tensor.numel());
        exit(EXIT_FAILURE);
    }
    // types: float or bfloat16
    if (grads_tensor.dtype() != DType::Float32 && grads_tensor.dtype() != DType::BFloat16) {
        fprintf(stderr, "error: GradNorm needs grads float or bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // addresses: a float4 load needs a start on a whole float4
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * grads_tensor.element_size();
    if ((size_t)grads_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: GradNorm reads grads %d values at a time, so they must start on %zu bytes; got %p\n",
                VALUES_PER_FLOAT4, float4_bytes_val, grads_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
    // 2. launch: GRAD_NORM_BLOCKS blocks, one block sum each
    if (grads_tensor.dtype() == DType::BFloat16)
        compute_grad_norm_kernel<<<GRAD_NORM_BLOCKS, threads_per_block_val>>>(
            grad_norm_tensor.data(), block_sums_tensor_.data(), blocks_done_tensor_.data(),
            grads_tensor.data_ptr<nv_bfloat16>(), grads_tensor.numel());
    else
        compute_grad_norm_kernel<<<GRAD_NORM_BLOCKS, threads_per_block_val>>>(
            grad_norm_tensor.data(), block_sums_tensor_.data(), blocks_done_tensor_.data(), grads_tensor.data(),
            grads_tensor.numel());
    cudaCheck(cudaGetLastError());
}

/// Works out the norm in one kernel: each block adds its share of squares, and the last block to finish adds
/// the block sums in a fixed order and writes the square root, so the same launch gives the same norm every
/// run, to the last bit.
///
///   grad_norm_ptr     1 float: the result
///   block_sums_ptr    one per block
///   blocks_done_ptr   the counter of blocks done: 0 at the start, put back to 0 at the end
///   grads_ptr         the gradients, float or bfloat16 (T)
///   grads_numel_val   how many
template <typename T>
__global__ void compute_grad_norm_kernel(float* grad_norm_ptr, float* block_sums_ptr, float* blocks_done_ptr,
                                         const T* grads_ptr, size_t grads_numel_val) {
    __shared__ bool is_last_block_val;          // did this block finish last?

    // 1. this thread's squares: float4s thread_index, + every thread launched, ...
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t float4_count_val = grads_numel_val / VALUES_PER_FLOAT4;      // whole float4s
    float sum_val = 0.0f;
    for (size_t index_val = thread_index_val; index_val < float4_count_val; index_val += (size_t)gridDim.x * blockDim.x) {
        float4 g_val = read4_as_float(grads_ptr + VALUES_PER_FLOAT4 * index_val);
        sum_val += g_val.x * g_val.x + g_val.y * g_val.y + g_val.z * g_val.z + g_val.w * g_val.w;
    }
    //    the 0 .. VALUES_PER_FLOAT4 - 1 values left after the last float4: one each for the first threads
    size_t leftover_index_val = float4_count_val * VALUES_PER_FLOAT4 + thread_index_val;
    if (leftover_index_val < grads_numel_val)
        sum_val += to_float(grads_ptr[leftover_index_val]) * to_float(grads_ptr[leftover_index_val]);

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

#endif // GRAD_NORM_CUH
