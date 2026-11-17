// classifier_8bit.cuh — GPT-2's head and last step for a bfloat16 model with the head's products in 8 bits (FP8
// E4M3) on an H100's tensor cores: bf16/classifier_bf16.cuh's math, with lnf_output and wte rounded to E4M3 first
// (round_8bit, one power-of-2 scale per row):
//     logits[r][v] = lnf_output_scale[r] · wte_scale[v] · sum over k of lnf_output_8bit[r][k] · wte_8bit[v][k]
//     then row_max, row_sum, loss as in classifier_bf16.cuh
//     backward:   logits_grad as in classifier_bf16.cuh, then linear_8bit.cuh's backward (no bias):
//                 lnf_output_grad = logits_grad · wte, wte_grad += logits_gradᵀ · lnf_output, both in 8 bits
//
//     tensor                             type in memory                   shape
//     lnf_output, lnf_output_grad        bfloat16                         {batch_size, seq_len, embed_dim}
//     wte, wte_grad                      bfloat16                         {padded_vocab_size, embed_dim}
//     logits, logits_grad                bfloat16                         {batch_size, seq_len, padded_vocab_size}
//     targets, losses, losses_grad, max_of_each_row, sum_of_each_row,
//     tile_max, tile_sum                 as in classifier_bf16.cuh
//     the 8-bit copies                   E4M3, a float scale per row      Linear8bitScratch(rows, embed_dim,
//                                                                         padded_vocab_size)'s: the head is a linear
//                                                                         layer from embed_dim to padded_vocab_size
//     sums                               float                            in registers
//
// The forward: round_8bit makes the copies, then classifier_8bit_logits_kernel is wgmma_matmul_8bit.cuh's product
// (WGMMA m64n64k32, the tiles brought by TMA) whose sums, times both scales, go to classifier_bf16_store_tile:
// stored as bfloat16 logits with each tile's rows' (max, sum); then classifier_bf16.cuh's loss kernel. Each E4M3
// is off by up to 2^-4 (6%) of its value, so the logits move by a few percent; the softmax and the loss are worked
// out from the logits as stored, in float. Needs an H100 (WGMMA and TMA), built with -arch=sm_90a, and rows,
// embed_dim and padded_vocab_size multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE (each is the k of one of the head's
// products). The same in bfloat16: bf16/classifier_bf16.cuh; the softmax alone: fp32/classifier.cuh.
// Math: https://ducbachsong.github.io/posts/classifier-backward/
#ifndef CLASSIFIER_8BIT_CUH
#define CLASSIFIER_8BIT_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include <cuda_fp8.h>                // FP8 E4M3
#include "../cuda_common.h"          // cudaCheck, check_threads_per_block, HARDWARE_WGMMA_*, HARDWARE_TMA_*
#include "../../common/tensor.cuh"
#include "round_8bit.cuh"            // the 8-bit copies: round_8bit
#include "wgmma_matmul_8bit.cuh"     // the TMA tile description and the tile x tile E4M3 WGMMA
#include "linear_8bit.cuh"           // Linear8bitScratch, linear_8bit_check_sizes, the head's backward: linear_8bit_backward
#include "../bf16/classifier_bf16.cuh"   // the tiles, classifier_bf16_store_tile, the loss and logits_grad kernels, the checks

static_assert(WGMMA_MATMUL_8BIT_BLOCK_COLUMNS == CLASSIFIER_BF16_TILE_COLUMNS && WGMMA_MATMUL_8BIT_COLUMN_GROUPS == CLASSIFIER_BF16_TILE_GROUPS,
              "the 8-bit product's squares are classifier_bf16.cuh's tiles: classifier_bf16_store_tile stores them");

// ── forward and backward ──────────────────────────────────────────────────────
void classifier_8bit_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                             const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                             const Tensor& tile_sum_tensor, const Tensor& logits_tensor, const Tensor& lnf_output_tensor,
                             const Tensor& wte_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                             const Linear8bitScratch& scratch_val,
                             int threads_per_block_val = CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK);
void classifier_8bit_backward(const Tensor& lnf_output_grad_tensor, const Tensor& wte_grad_tensor,
                              const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                              const Tensor& logits_tensor, const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                              const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                              const Tensor& sum_of_each_row_tensor, size_t vocab_size_val,
                              const Linear8bitScratch& scratch_val,
                              int threads_per_block_val = CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
classifier_8bit_logits_kernel(const __grid_constant__ CUtensorMap lnf_output_map_val, const __grid_constant__ CUtensorMap wte_map_val,
                              nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr,
                              const float* lnf_output_scales_ptr, const float* wte_scales_ptr, int rows_val,
                              int padded_vocab_size_val, int vocab_size_val, int embed_dim_val);

/// Writes logits = lnf_output · wteᵀ with the product in 8 bits, and each row's loss, row_max and row_sum: checks,
/// rounds lnf_output and wte to E4M3 (both sit embed_dim next to each other already), describes the copies' tiles
/// for TMA, launches classifier_8bit_logits_kernel (one block per tile of logits, as one line of blocks), then
/// classifier_bf16.cuh's loss kernel. The targets are not checked: a target >= vocab_size reads a padding column.
///
///   losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, tile_max_tensor, tile_sum_tensor, logits_tensor,
///   lnf_output_tensor, wte_tensor, targets_tensor, vocab_size_val     as in classifier_bf16_forward
///   scratch_val              Linear8bitScratch(batch_size x seq_len, embed_dim, padded_vocab_size): its
///                            output_copies are written
///   threads_per_block_val    the loss kernel's block size, as in classifier_bf16_forward
inline void classifier_8bit_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                    const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                                    const Tensor& tile_sum_tensor, const Tensor& logits_tensor,
                                    const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                    const Tensor& targets_tensor, size_t vocab_size_val,
                                    const Linear8bitScratch& scratch_val, int threads_per_block_val) {
    // 1. check the launch, the tensors and the scratch; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "classifier_8bit_forward");
    classifier_bf16_check_forward("classifier_8bit_forward", losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor,
                                  tile_max_tensor, tile_sum_tensor, logits_tensor, lnf_output_tensor, wte_tensor,
                                  targets_tensor, vocab_size_val);
    linear_8bit_check_sizes("classifier_8bit_forward", lnf_output_tensor, wte_tensor, scratch_val);
    size_t rows_val = targets_tensor.numel();
    size_t embed_dim_val = wte_tensor.size(1);
    size_t padded_vocab_size_val = wte_tensor.size(0);

    // 2. the 8-bit copies: lnf_output {rows, embed_dim} and wte {padded_vocab_size, embed_dim}
    const Linear8bitProductCopies& copies_val = scratch_val.output_copies;
    round_8bit(copies_val.left_8bit_tensor, copies_val.left_scales_tensor, lnf_output_tensor);
    round_8bit(copies_val.right_8bit_tensor, copies_val.right_scales_tensor, wte_tensor);

    // 3. TMA's description of the copies' tiles: {lines, embed_dim} bytes in boxes of {64 lines, WGMMA_MATMUL_8BIT_BLOCK_K}
    CUtensorMap lnf_output_map_val = wgmma_matmul_8bit_tensor_map(copies_val.left_8bit_tensor.data_ptr(), rows_val, embed_dim_val,
                                                                  CLASSIFIER_BF16_TILE_ROWS);
    CUtensorMap wte_map_val = wgmma_matmul_8bit_tensor_map(copies_val.right_8bit_tensor.data_ptr(), padded_vocab_size_val,
                                                           embed_dim_val, CLASSIFIER_BF16_TILE_COLUMNS);

    // 4. the logits and each tile's (max, sum): tiles across and down, rounded up, as one line of blocks
    size_t total_block_columns_val = classifier_bf16_tile_columns(padded_vocab_size_val);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)CLASSIFIER_BF16_TILE_ROWS);
    classifier_8bit_logits_kernel<<<(unsigned)(total_block_columns_val * total_block_rows_val), HARDWARE_WARPGROUP_THREADS>>>(
        lnf_output_map_val, wte_map_val, logits_tensor.data_ptr<nv_bfloat16>(), tile_max_tensor.data(),
        tile_sum_tensor.data(), copies_val.left_scales_tensor.data(), copies_val.right_scales_tensor.data(), (int)rows_val,
        (int)padded_vocab_size_val, (int)vocab_size_val, (int)embed_dim_val);
    cudaCheck(cudaGetLastError());

    // 5. each row's tiles put together: row_max, row_sum, the loss
    classifier_bf16_losses(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, tile_max_tensor, tile_sum_tensor,
                           logits_tensor, targets_tensor, threads_per_block_val);
}

/// Works out one tile of logits with one warpgroup, as wgmma_matmul_8bit_kernel does: step by step, TMA brings a
/// tile of lnf_output's copy and of wte's into shared memory, then the E4M3 tensor cores add their product into the
/// sums. Then each sum is times its row's and column's scales (in wgmma_matmul_8bit_kernel's order: sum · row scale
/// · column scale), and classifier_bf16_store_tile stores the tile and its rows' (max, sum).
///
///   lnf_output_map_val, wte_map_val          TMA's descriptions of the copies' tiles
///   logits_ptr, tile_max_ptr, tile_sum_ptr   as in classifier_bf16_logits_kernel
///   lnf_output_scales_ptr, wte_scales_ptr    one float per row of each copy
///   rows_val, padded_vocab_size_val, vocab_size_val, embed_dim_val   as in classifier_bf16_logits_kernel
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
classifier_8bit_logits_kernel(const __grid_constant__ CUtensorMap lnf_output_map_val, const __grid_constant__ CUtensorMap wte_map_val,
                              nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr,
                              const float* lnf_output_scales_ptr, const float* wte_scales_ptr, int rows_val,
                              int padded_vocab_size_val, int vocab_size_val, int embed_dim_val) {
    namespace cde = cuda::device::experimental;
    // this step's tiles, in TMA's swizzled layout (on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom), and the barrier
    // that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t lnf_output_tile_list[CLASSIFIER_BF16_TILE_ROWS * WGMMA_MATMUL_8BIT_BLOCK_K];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) uint8_t wte_tile_list[CLASSIFIER_BF16_TILE_COLUMNS * WGMMA_MATMUL_8BIT_BLOCK_K];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ WgmmaMatmul8bitBarrier tiles_barrier_val;

    // 1. the block's tile, from its number: tiles row after row; and the barrier, by one thread
    int total_block_columns_val = CEIL_DIV(padded_vocab_size_val, CLASSIFIER_BF16_TILE_COLUMNS);
    int block_row_index_val = blockIdx.x / total_block_columns_val;
    int block_column_index_val = blockIdx.x % total_block_columns_val;
    int block_first_row_index_val = block_row_index_val * CLASSIFIER_BF16_TILE_ROWS;
    int block_first_column_index_val = block_column_index_val * CLASSIFIER_BF16_TILE_COLUMNS;
    if (threadIdx.x == 0) {
        init(&tiles_barrier_val, blockDim.x);                 // every thread arrives once a step
        cde::fence_proxy_async_shared_cta();                  // the barrier visible to TMA
    }
    __syncthreads();

    // 2. walk the sum over embed_dim: TMA brings the step's tiles, the tensor cores add their product
    float sums_list[CLASSIFIER_BF16_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of the tile
    int total_steps_val = CEIL_DIV(embed_dim_val, WGMMA_MATMUL_8BIT_BLOCK_K);
    for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. one thread asks for both tiles (past the edges TMA gives 0s) and says how many bytes to wait for; the
        //    others arrive; all wait until the bytes are in
        WgmmaMatmul8bitBarrier::arrival_token token_val;
        int step_first_k_index_val = step_index_val * WGMMA_MATMUL_8BIT_BLOCK_K;
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(lnf_output_tile_list, &lnf_output_map_val, step_first_k_index_val,
                                                          block_first_row_index_val, tiles_barrier_val);
            cde::cp_async_bulk_tensor_2d_global_to_shared(wte_tile_list, &wte_map_val, step_first_k_index_val,
                                                          block_first_column_index_val, tiles_barrier_val);
            token_val = cuda::device::barrier_arrive_tx(tiles_barrier_val, 1, sizeof(lnf_output_tile_list) + sizeof(wte_tile_list));
        } else {
            token_val = tiles_barrier_val.arrive();
        }
        tiles_barrier_val.wait(std::move(token_val));

        // b. the tensor cores: WGMMA_MATMUL_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3 WGMMAs, each HARDWARE_WGMMA_K_E4M3
        //    more values (bytes) of k
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < WGMMA_MATMUL_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3; k_step_index_val++)
            wgmma_matmul_8bit_multiply_add(sums_list, lnf_output_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                                           wte_tile_list + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                                           1);   // always add onto: the sum is only embed_dim long, short enough for the tensor cores
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // ... done

        // c. every thread done with the tiles before TMA fills them again
        __syncthreads();
    }

    // 3. each sum times its row's and its column's scale; past the edges (TMA's 0s) the scales are left at 0
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int thread_first_row_index_val = block_first_row_index_val + warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int thread_first_column_index_val = block_first_column_index_val + HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
    float row_scales_list[HARDWARE_WGMMA_ROWS_PER_THREAD];
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        int row_index_val = thread_first_row_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        row_scales_list[thread_row_index_val] = row_index_val < rows_val ? lnf_output_scales_ptr[row_index_val] : 0.0f;
    }
#pragma unroll
    for (int group_index_val = 0; group_index_val < CLASSIFIER_BF16_TILE_GROUPS; group_index_val++) {
        int column_index_val = thread_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS;
        float2 column_scales_val = column_index_val < padded_vocab_size_val
                                       ? *reinterpret_cast<const float2*>(wte_scales_ptr + column_index_val)
                                       : make_float2(0.0f, 0.0f);
#pragma unroll
        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
            float* pair_ptr = &sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val];
            pair_ptr[0] = pair_ptr[0] * row_scales_list[thread_row_index_val] * column_scales_val.x;
            pair_ptr[1] = pair_ptr[1] * row_scales_list[thread_row_index_val] * column_scales_val.y;
        }
    }

    // 4. store the tile, and its rows' (max, sum)
    classifier_bf16_store_tile(sums_list, logits_ptr, tile_max_ptr, tile_sum_ptr, block_first_row_index_val,
                               block_column_index_val, total_block_columns_val, rows_val, padded_vocab_size_val,
                               vocab_size_val);
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of lnf_output and wte with the head's products in 8 bits: writes logits_grad
/// (classifier_bf16_logits_grad, from the stored logits, in float), then the head's backward
/// (linear_8bit_backward, no bias): logits_grad, wte and lnf_output rounded to E4M3 (turned while rounding where
/// they sit turned round), lnf_output_grad = logits_grad · wte, wte_grad += logits_gradᵀ · lnf_output.
/// lnf_output_grad and logits_grad are written over; wte_grad is added onto (the encoder adds its part too).
///
///   the tensors and vocab_size_val   as in classifier_bf16_backward
///   scratch_val                      Linear8bitScratch(batch_size x seq_len, embed_dim, padded_vocab_size): its
///                                    input_grad_copies and weight_grad_copies are written
///   threads_per_block_val            logits_grad's block size, as in classifier_bf16_backward
inline void classifier_8bit_backward(const Tensor& lnf_output_grad_tensor, const Tensor& wte_grad_tensor,
                                     const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                                     const Tensor& logits_tensor, const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                     const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                                     const Tensor& sum_of_each_row_tensor, size_t vocab_size_val,
                                     const Linear8bitScratch& scratch_val, int threads_per_block_val) {
    // 1. logits_grad, written over (it checks its tensors)
    classifier_bf16_logits_grad("classifier_8bit_backward", logits_grad_tensor, losses_grad_tensor, logits_tensor,
                                targets_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, vocab_size_val,
                                threads_per_block_val);

    // 2. the head's two products in 8 bits (linear_8bit_backward checks the rest, and the scratch)
    linear_8bit_backward(lnf_output_grad_tensor, wte_grad_tensor, Tensor(), logits_grad_tensor, lnf_output_tensor, wte_tensor,
                         scratch_val);
}

#endif // CLASSIFIER_8BIT_CUH
