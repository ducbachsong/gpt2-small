// grad_norm.cuh — the norm of all the gradients together, as if they were one long list:
#ifndef GRAD_NORM_CUH
#define GRAD_NORM_CUH

#include "cuda_common.h"             // cudaCheck only: the adding up is written out in the kernel, below
#include "../tensor.cuh"             // compute takes views: each knows its address and its numel

#define GRAD_NORM_BLOCKS 512        // blocks in the launch: each writes one block sum
#define GRAD_NORM_THREADS 1024      // threads per block

class GradNorm {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    explicit GradNorm(const Tensor& block_sums);                        // {2, 512}: the block sums, then the counter

    // ── the norm ────────────────────────────────────────────────────────────────
    void compute(const Tensor& grads, const Tensor& grad_norm) const;  // grad_norm (1 value) = sqrt(sum of every grad²)

private:
    Tensor block_sums_;                  // row 0 of the {2, 512}: each block's sum of squares
    Tensor blocks_done_;                 // row 1, value 0: how many blocks have written their sum; 0 between computes
};

// ══ the code, in the order of the list above ══════════════════════════════════

/// Makes a GradNorm that keeps its block sums and its counter in one {2, 512} view of the flat buffer: it allocates
/// nothing. Row 0 holds the 512 block sums; value 0 of row 1 is the counter of blocks done (the other 511 of row 1
/// are not used: 2 x 512 = 1024 values keeps whatever follows in the buffer on 16 bytes). The counter must start at
/// 0 (a new buffer is all 0); each compute puts it back to 0. Stops if the view is not {2, 512}.
inline GradNorm::GradNorm(const Tensor& block_sums) {
    if (block_sums.shape() != Shape({2, GRAD_NORM_BLOCKS})) {
        fprintf(stderr, "error: GradNorm needs block sums of shape {2, %d}, but got %zu values\n", GRAD_NORM_BLOCKS,
                block_sums.numel());
        exit(EXIT_FAILURE);
    }
    block_sums_ = block_sums[0];         // the 512 block sums
    blocks_done_ = block_sums[1][0];     // the counter: one value
}

// ── the norm ──────────────────────────────────────────────────────────────────

// The kernel compute launches, declared here so compute can call it; its code follows it.
__global__ void compute_grad_norm_kernel(float* grad_norm, float* block_sums, float* blocks_done, const float* grads,
                                         size_t grads_numel);

/// Works out the norm of every value in `grads` and writes it into `grad_norm`, both views of the flat buffer, with
inline void GradNorm::compute(const Tensor& grads, const Tensor& grad_norm) const {
    if (grad_norm.numel() != 1) {
        fprintf(stderr, "error: GradNorm writes 1 value, but grad_norm has %zu\n", grad_norm.numel());
        exit(EXIT_FAILURE);
    }
    // Off 16 bytes, the GPU stops with "misaligned address" at some later call, far from the cause.
    if ((size_t)grads.data() % 16 != 0) {
        fprintf(stderr, "error: GradNorm reads grads 4 floats at a time (float4), so they must start on 16 bytes; got %p\n",
                (void*)grads.data());
        exit(EXIT_FAILURE);
    }
    compute_grad_norm_kernel<<<GRAD_NORM_BLOCKS, GRAD_NORM_THREADS>>>(grad_norm.data(), block_sums_.data(),
                                                                      blocks_done_.data(), grads.data(), grads.numel());
    cudaCheck(cudaGetLastError());
}

/// Works out the norm of the grads_numel gradients, the whole process in one kernel:
///
///     1. every thread adds the squares of its share, 4 floats per load (float4, a grid stride)
///     2. every block adds its 1024 sums; thread 0 writes the block sum and counts its block as done
///     3. the block that finishes last (the count reaches 512) adds the 512 block sums and writes the square root
///
/// A block adds 1024 numbers in two rounds of 32 (steps 2 and 3 both): each warp adds its 32 with shuffles, the 32
/// warp totals go to shared memory, then each warp adds those 32 the same way. Blocks cannot wait for each other, so
/// the counter tells the last one that every block sum is written. The last block adds block_sums[0 .. 511] in that
/// order whichever block it is: the same norm every run, to the last bit.
__global__ void compute_grad_norm_kernel(float* grad_norm, float* block_sums, float* blocks_done, const float* grads,
                                         size_t grads_numel) {
    __shared__ float warp_totals[32];           // one per warp, shared by the whole block: used in steps 2 and 3
    __shared__ bool is_last_block;              // one per block: did this block finish last?
    int thread_in_warp = threadIdx.x % 32, warp = threadIdx.x / 32;

    // 1. this thread's squares. The grads are read as float4s: group g is values 4g .. 4g + 3, one 16-byte load, so
    //    each request brings 4 floats instead of 1 and memory stays busier. Thread t takes groups t, t + 524,288,
    //    t + 2 x 524,288, ... (the jump: every thread launched). GPT-2 small's 124,439,808 values are 31,109,952
    //    groups: 59.3 rounds, so each thread reads 59 or 60 float4s.
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t groups = grads_numel / 4;                                    // whole float4s; 0 .. 3 values are left over
    const float4* grads4 = reinterpret_cast<const float4*>(grads);
    float sum = 0.0f;
    for (size_t g = thread_number; g < groups; g += (size_t)gridDim.x * blockDim.x) {
        float4 v = grads4[g];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    //    The 0 .. 3 values after the last whole float4 (grads_numel not a multiple of 4): one each for threads 0 .. 2.
    size_t leftover = groups * 4 + thread_number;
    if (leftover < grads_numel) sum += grads[leftover] * grads[leftover];

    // 2. this block's 1024 sums, added.
    //    Round 1: each warp adds its 32 sums: every thread takes the value of the thread `offset` away,
    //    32 → 16 → 8 → 4 → 2 → 1; then each thread holds its warp's total.
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (thread_in_warp == 0) warp_totals[warp] = sum;
    __syncthreads();                            // wait until all 32 warps have written theirs
    //    Round 2: each thread reads one of the 32 warp totals, and each warp adds them the same way: the block sum.
    float block_total = warp_totals[thread_in_warp];
    for (int offset = 16; offset > 0; offset /= 2) block_total += __shfl_xor_sync(0xffffffff, block_total, offset);
    //    Thread 0 writes the block sum, then counts the block as done.
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_total;
        __threadfence();                        // the block sum reaches GPU memory before the count goes up
        float done_before = atomicAdd(blocks_done, 1.0f);   // +1, one block at a time; gives the count before it
        is_last_block = done_before == GRAD_NORM_BLOCKS - 1;     // 511 before this one: every other block is done
    }
    __syncthreads();                            // every thread sees is_last_block, and is done reading warp_totals
    if (!is_last_block) return;

    // 3. only the last block: all 512 block sums are in GPU memory (the blocks that wrote them have returned; what
    //    they wrote stays). threadIdx.x is not tied to any block here: it is just a number 0 .. 1023, used as an index
    //    to pick up the sums. Threads 0 .. 511 each take block_sums[threadIdx.x]; threads 512 .. 1023 have no slot and
    //    take 0, which adds nothing (one read per thread covers all 512, as 1024 threads >= 512 sums). Then the block
    //    adds the 1024 values in the same two rounds. volatile: read the sums from GPU memory, not from an old copy
    //    in this SM's cache.
    float total = threadIdx.x < GRAD_NORM_BLOCKS ? ((volatile float*)block_sums)[threadIdx.x] : 0.0f;
    //    Round 1: each warp adds its 32.
    for (int offset = 16; offset > 0; offset /= 2) total += __shfl_xor_sync(0xffffffff, total, offset);
    if (thread_in_warp == 0) warp_totals[warp] = total;
    __syncthreads();                            // wait until all 32 warps have written theirs
    //    Round 2: each warp adds the 32 warp totals: the sum of every square.
    float all_blocks_total = warp_totals[thread_in_warp];
    for (int offset = 16; offset > 0; offset /= 2)
        all_blocks_total += __shfl_xor_sync(0xffffffff, all_blocks_total, offset);
    //    Thread 0 writes the square root, the norm, and sets the counter back to 0.
    if (threadIdx.x == 0) {
        *grad_norm = sqrtf(all_blocks_total);
        *blocks_done = 0.0f;                    // ready for the next compute
    }
}

#endif // GRAD_NORM_CUH
