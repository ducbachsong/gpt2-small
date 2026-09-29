// cuda_common.h — what every CUDA file here needs: error checks, and adding up
// numbers across a warp and across a block.
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Every CUDA call returns an error code; this stops with the file and line.
static inline void cuda_check(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(call) cuda_check((call), __FILE__, __LINE__)

/// a / b rounded up: how many blocks of b threads cover a items. CEIL_DIV(1000, 256) = 4.
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ── combining across a warp (a "reduction") ──────────────────────────────────
// What for: adding up N values with a warp (32 threads that run together) instead
// of one thread. It goes in two phases:
//
//     phase 1: each lane adds up its own share of N / 32 values, all lanes at once
//              → 32 partial sums, one per lane                     N / 32 steps
//     phase 2: warp_reduce_sum folds the 32 partial sums into 1    5 steps, always
//
//     N = 768 (a LayerNorm row):   1 thread: 768 steps   warp: 24 + 5 = 29 steps
//
//     float part = ...;                          // phase 1: this lane's 24 values added
//     float total = warp_reduce_sum(part);       // phase 2: every lane gets the sum of all 768
//     float mean = total / 768;
//
// Phase 2, the fold: each step, every lane adds the value of a partner lane, read
// straight from its registers (a "shuffle", no memory). The partner is `offset`
// lanes away (lane i ^ offset). Each step halves the number of different sums:
//
//     32 ──(offset 16)──► 16 ──(8)──► 8 ──(4)──► 4 ──(2)──► 2 ──(1)──► 1: the total, in every lane
//
// It starts at offset 16 because a warp's lanes are 0..31: 16 apart is half the warp,
// so the first step adds the right half onto the left half, like folding paper.
// With 4 lanes holding 1, 2, 3, 4 (offsets 2, then 1):
//
//     offset 2:   1+3=4    2+4=6    3+1=4    4+2=6
//     offset 1:   4+6=10   6+4=10   4+6=10   6+4=10      every lane: 10

/// The sum of `value` over the 32 threads of the warp, in every thread.
__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

// ── the largest value across a warp ──────────────────────────────────────────
// What for: softmax, exp(x) / sum of exp, used by attention and at the output.
// exp of a big score overflows a float (exp(1000) = inf, and then NaN), so softmax
// first subtracts the row's largest score. The answer is the same, since the factor
// cancels in the division, but now the biggest exp is exp(0) = 1:
//
//     scores 1000, 1001, 1002   max 1002   exp(-2), exp(-1), exp(0) = 0.14, 0.37, 1.0
//
// Finding that largest score with a warp, instead of one thread, goes in two phases:
//
//     phase 1: each lane finds the largest of its own share of N / 32 values,
//              all lanes at once → 32 partial maxes, one per lane   N / 32 steps
//     phase 2: warp_reduce_max folds the 32 partial maxes into 1    5 steps, always
//
//     N = 1024 (an attention row):   1 thread: 1024 steps   warp: 32 + 5 = 37 steps
//
//     float part = ...;                          // phase 1: the largest of this lane's 32 scores
//     float row_max = warp_reduce_max(part);     // phase 2: every lane gets the largest of all 1024
//     float e = expf(score - row_max);           // never more than exp(0) = 1
//
// Phase 2, the fold: the same as warp_reduce_sum's, but each step keeps the larger of
// the two values (fmaxf) instead of adding them. Each step, every lane compares its
// value with a partner lane's, read straight from its registers (a "shuffle", no
// memory). The partner is `offset` lanes away (lane i ^ offset). Each step halves the
// number of different maxes:
//
//     32 ──(offset 16)──► 16 ──(8)──► 8 ──(4)──► 4 ──(2)──► 2 ──(1)──► 1: the largest, in every lane
//
// It starts at offset 16 because a warp's lanes are 0..31: 16 apart is half the warp,
// so the first step compares the right half with the left half, like folding paper.
// With 4 lanes holding 3, 9, 1, 5 (offsets 2, then 1):
//
//     offset 2:   max(3,1)=3   max(9,5)=9   max(1,3)=3   max(5,9)=9
//     offset 1:   max(3,9)=9   max(9,3)=9   max(3,9)=9   max(9,3)=9      every lane: 9

/// The largest `value` over the 32 threads of the warp, in every thread.
__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

// ── adding up across a block ─────────────────────────────────────────────────
// Up to 32 warps: each warp adds its own 32 values, its lane 0 writes the
// warp's total to shared memory, and the first warp adds those. Every thread of
// the block must call it (it has __syncthreads() inside); every thread gets the
// total back. blockDim.x must be a multiple of 32.
__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_totals[32];
    __shared__ float block_total;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, num_warps = blockDim.x / 32;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        float total = warp_reduce_sum(lane < num_warps ? warp_totals[lane] : 0.0f);
        if (lane == 0) block_total = total;
    }
    __syncthreads();
    float result = block_total;
    __syncthreads();          // before anyone calls it again and overwrites warp_totals
    return result;
}

__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float warp_maxima[32];
    __shared__ float block_max;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, num_warps = blockDim.x / 32;
    value = warp_reduce_max(value);
    if (lane == 0) warp_maxima[warp] = value;
    __syncthreads();
    if (warp == 0) {
        float maximum = warp_reduce_max(lane < num_warps ? warp_maxima[lane] : -FLT_MAX);
        if (lane == 0) block_max = maximum;
    }
    __syncthreads();
    float result = block_max;
    __syncthreads();
    return result;
}

#endif // CUDA_COMMON_H
