// cuda_common.h — what every CUDA file here needs: error checks, and sums and maxes across a warp and a block.
// A warp's sum folds its THREADS_PER_WARP values in log2(THREADS_PER_WARP) steps: each thread adds the value of
// the thread `offset` away (a shuffle, no memory), offset THREADS_PER_WARP / 2, halved each step down to 1;
// then every thread holds the total.
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/// Stops with the file and line if a CUDA call failed.
///
///   error_val             what the call returned
///   file_ptr, line_val    where the call is
static inline void cuda_check(cudaError_t error_val, const char* file_ptr, int line_val) {
    if (error_val != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file_ptr, line_val, cudaGetErrorString(error_val));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(call) cuda_check((call), __FILE__, __LINE__)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))   // a / b rounded up: CEIL_DIV(1000, 256) = 4
#define THREADS_PER_WARP 32                      // threads that run together: CUDA's warpSize
#define WARP_ALL_LANES 0xffffffff                // a shuffle's mask: one bit per thread of the warp, all taking part
#define MAX_THREADS_PER_BLOCK 1024               // CUDA's limit on one block
#define MAX_WARPS_PER_BLOCK (MAX_THREADS_PER_BLOCK / THREADS_PER_WARP)   // one shared slot per warp holds them all

/// Stops with a message unless a launch's block size is whole warps and within CUDA's limit: every launcher
/// checks its threads_per_block with it.
///
///   threads_per_block_val   the block size asked for
///   kernel_name_ptr         the launcher's name, for the message
static inline void check_threads_per_block(int threads_per_block_val, const char* kernel_name_ptr) {
    if (threads_per_block_val <= 0 || threads_per_block_val > MAX_THREADS_PER_BLOCK ||
        threads_per_block_val % THREADS_PER_WARP != 0) {
        fprintf(stderr, "error: %s needs threads_per_block to be a multiple of %d, at most %d; got %d\n",
                kernel_name_ptr, THREADS_PER_WARP, MAX_THREADS_PER_BLOCK, threads_per_block_val);
        exit(EXIT_FAILURE);
    }
}

/// Gives the sum of `value_val` over the warp's THREADS_PER_WARP threads, in every thread.
///
///   value_val   this thread's part
__device__ __forceinline__ float warp_reduce_sum(float value_val) {
    for (int offset_val = THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2)
        value_val += __shfl_xor_sync(WARP_ALL_LANES, value_val, offset_val);
    return value_val;
}

/// Gives the largest `value_val` over the warp's THREADS_PER_WARP threads, in every thread: the same fold, with
/// fmaxf.
///
///   value_val   this thread's part
__device__ __forceinline__ float warp_reduce_max(float value_val) {
    for (int offset_val = THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2)
        value_val = fmaxf(value_val, __shfl_xor_sync(WARP_ALL_LANES, value_val, offset_val));
    return value_val;
}

/// Gives the sum of `value_val` over the block, in every thread: each warp's total to shared memory, then warp 0
/// adds those. Every thread must call it; blockDim.x a multiple of THREADS_PER_WARP.
///
///   value_val   this thread's part
__device__ __forceinline__ float block_reduce_sum(float value_val) {
    // one slot per warp; warp 0's THREADS_PER_WARP lanes must cover them all
    static_assert(MAX_WARPS_PER_BLOCK <= THREADS_PER_WARP, "warp 0 must hold one total per warp");
    __shared__ float warp_totals_list[MAX_WARPS_PER_BLOCK];
    __shared__ float block_total_val;
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / THREADS_PER_WARP;
    // 1. each warp's total, by its lane 0
    value_val = warp_reduce_sum(value_val);
    if (lane_index_val == 0) warp_totals_list[warp_index_val] = value_val;
    __syncthreads();
    // 2. warp 0 adds the warps' totals
    if (warp_index_val == 0) {
        float total_val = warp_reduce_sum(lane_index_val < warps_in_block_val ? warp_totals_list[lane_index_val] : 0.0f);
        if (lane_index_val == 0) block_total_val = total_val;
    }
    __syncthreads();
    // 3. every thread reads it
    float result_val = block_total_val;
    __syncthreads();          // before a next call writes warp_totals_list again
    return result_val;
}

/// Gives the largest `value_val` over the block, in every thread, as block_reduce_sum.
///
///   value_val   this thread's part
__device__ __forceinline__ float block_reduce_max(float value_val) {
    // one slot per warp; warp 0's THREADS_PER_WARP lanes must cover them all
    static_assert(MAX_WARPS_PER_BLOCK <= THREADS_PER_WARP, "warp 0 must hold one maximum per warp");
    __shared__ float warp_maxima_list[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_val;
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / THREADS_PER_WARP;
    // 1. each warp's largest, by its lane 0
    value_val = warp_reduce_max(value_val);
    if (lane_index_val == 0) warp_maxima_list[warp_index_val] = value_val;
    __syncthreads();
    // 2. warp 0 takes the largest of the warps'
    if (warp_index_val == 0) {
        float maximum_val = warp_reduce_max(lane_index_val < warps_in_block_val ? warp_maxima_list[lane_index_val] : -FLT_MAX);
        if (lane_index_val == 0) block_max_val = maximum_val;
    }
    __syncthreads();
    // 3. every thread reads it
    float result_val = block_max_val;
    __syncthreads();
    return result_val;
}

#endif // CUDA_COMMON_H
