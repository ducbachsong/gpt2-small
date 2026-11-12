// classifier_bf16.cuh — GPT-2's head and last step in bfloat16, the head's product on an H100's tensor cores
// (WGMMA, fed by TMA) with the softmax's row sums worked out as each tile of logits is stored:
//     logits  = lnf_output · wteᵀ                       the head (wte is the token embedding, tied)
//     row_max = max of the row's logits
//     row_sum = sum of exp(logit - row_max)
//     loss    = -log(prob[target]) = log(row_sum) - (logit[target] - row_max)
//     backward:   logits_grad      = (prob - 1 at the target, prob elsewhere) · losses_grad[row]
//                 lnf_output_grad  = logits_grad · wte
//                 wte_grad        += logits_gradᵀ · lnf_output
//
//     tensor                             type in memory   shape
//     lnf_output, lnf_output_grad        bfloat16         {batch_size, seq_len, embed_dim}
//     wte, wte_grad                      bfloat16         {padded_vocab_size, embed_dim}
//     logits, logits_grad                bfloat16         {batch_size, seq_len, padded_vocab_size}: kept for the backward
//     targets                            int              {batch_size, seq_len}
//     losses, losses_grad                float            {batch_size, seq_len}
//     max_of_each_row, sum_of_each_row   float            {batch_size, seq_len}: kept for the backward
//     tile_max, tile_sum                 float            {batch_size, seq_len, tile columns}: scratch
//     sums                               float            in registers
//
// The forward is two kernels:
//     1. classifier_bf16_logits_kernel: wgmma_matmul_bf16.cuh's product, one warpgroup per CLASSIFIER_BF16_TILE_ROWS
//        x CLASSIFIER_BF16_TILE_COLUMNS tile of logits. As the tile is stored (rounded to bfloat16), each of its
//        rows' (max, sum of exp) over the tile's columns is worked out from the registers and written to
//        tile_max and tile_sum: the tile is never read again for the softmax.
//     2. classifier_bf16_loss_kernel: one warp per row puts the row's tiles' (max, sum) together, the online
//        softmax's way, and writes row_max, row_sum and the loss.
// The softmax uses each logit as it is stored, rounded to bfloat16, as fp32/classifier.cuh does on bfloat16
// logits: the backward works the probs out again from the stored logits, so they add up to 1 there too.
// The backward: logits_grad (one float4 a thread, written over logits_grad, which may be logits itself), then
// linear_bf16.cuh's backward for the head's two products. Columns from vocab_size on are padding: left out of
// the softmax, gradient 0. Needs an H100 (WGMMA and TMA exist only on Hopper), built with -arch=sm_90a, and
// embed_dim and padded_vocab_size multiples of WGMMA_MATMUL_BF16_VALUE_MULTIPLE. The softmax alone on float or
// bfloat16 logits, any GPU: fp32/classifier.cuh; with the head's products in 8 bits: 8bit/classifier_8bit.cuh.
// Math: https://ducbachsong.github.io/posts/classifier-backward/
#ifndef CLASSIFIER_BF16_CUH
#define CLASSIFIER_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cuda/barrier>              // cuda::barrier, and the TMA copy into shared memory
#include "../cuda_common.h"          // cudaCheck, check_threads_per_block, HARDWARE_WGMMA_*, HARDWARE_TMA_*
#include "../../common/tensor.cuh"
#include "wgmma_matmul_bf16.cuh"     // the TMA tile description, the WGMMA descriptor and the tile x tile WGMMA
#include "linear_bf16.cuh"           // the head's backward: linear_bf16_backward

// The tiles of logits: wgmma_matmul_bf16.cuh's squares. The index math is written for these numbers: change them
// together with it.
#define CLASSIFIER_BF16_TILE_ROWS HARDWARE_WGMMA_M                              // rows of a tile: one WGMMA's M
#define CLASSIFIER_BF16_TILE_COLUMNS WGMMA_MATMUL_BF16_BLOCK_COLUMNS            // columns of a tile: one WGMMA's N
#define CLASSIFIER_BF16_TILE_GROUPS WGMMA_MATMUL_BF16_COLUMN_GROUPS             // groups of C across a tile
#define CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK (8 * HARDWARE_THREADS_PER_WARP)      // the loss kernel's default: one warp per row
#define CLASSIFIER_BF16_BACKWARD_BLOCKS 40960                                       // logits_grad's blocks, at most
#define CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK (4 * HARDWARE_THREADS_PER_WARP)  // logits_grad's default: fp32/classifier.cuh's bfloat16 pick on an H100
static_assert(CLASSIFIER_BF16_TILE_COLUMNS == CLASSIFIER_BF16_TILE_GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS,
              "a tile's columns are whole groups of C");
static_assert(WGMMA_MATMUL_BF16_VALUE_MULTIPLE % HARDWARE_WGMMA_COLUMNS_PER_THREAD == 0,
              "padded_vocab_size is even: a thread's column pair is all in or all out");

// ── forward and backward ──────────────────────────────────────────────────────
void classifier_bf16_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                             const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                             const Tensor& tile_sum_tensor, const Tensor& logits_tensor, const Tensor& lnf_output_tensor,
                             const Tensor& wte_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                             int threads_per_block_val = CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK);
void classifier_bf16_backward(const Tensor& lnf_output_grad_tensor, const Tensor& wte_grad_tensor,
                              const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                              const Tensor& logits_tensor, const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                              const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                              const Tensor& sum_of_each_row_tensor, size_t vocab_size_val,
                              int threads_per_block_val = CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK);

// ── for both, and for 8bit/classifier_8bit.cuh ────────────────────────────────
size_t classifier_bf16_tile_columns(size_t padded_vocab_size_val);
void classifier_bf16_check_forward(const char* name_ptr, const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                   const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                                   const Tensor& tile_sum_tensor, const Tensor& logits_tensor,
                                   const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                   const Tensor& targets_tensor, size_t vocab_size_val);
void classifier_bf16_losses(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                            const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                            const Tensor& tile_sum_tensor, const Tensor& logits_tensor, const Tensor& targets_tensor,
                            int threads_per_block_val);
void classifier_bf16_logits_grad(const char* name_ptr, const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                                 const Tensor& logits_tensor, const Tensor& targets_tensor,
                                 const Tensor& max_of_each_row_tensor, const Tensor& sum_of_each_row_tensor,
                                 size_t vocab_size_val, int threads_per_block_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
classifier_bf16_logits_kernel(const __grid_constant__ CUtensorMap lnf_output_map_val, const __grid_constant__ CUtensorMap wte_map_val,
                              nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr, int rows_val,
                              int padded_vocab_size_val, int vocab_size_val, int embed_dim_val);
__device__ __forceinline__ void classifier_bf16_store_tile(
        const float (&logits_list)[CLASSIFIER_BF16_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
        nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr, int block_first_row_index_val,
        int block_column_index_val, int total_block_columns_val, int rows_val, int padded_vocab_size_val,
        int vocab_size_val);

/// Writes logits = lnf_output · wteᵀ, and each row's loss, row_max and row_sum: checks, describes lnf_output's
/// and wte's tiles for TMA, launches classifier_bf16_logits_kernel (one block per tile of logits, as one line of
/// blocks), then classifier_bf16_loss_kernel. The targets are not checked: a target >= vocab_size reads a padding
/// column.
///
///   losses_tensor            {batch_size, seq_len} float: one loss per row
///   max_of_each_row_tensor   {batch_size, seq_len} float: row_max, for the backward
///   sum_of_each_row_tensor   {batch_size, seq_len} float: row_sum, for the backward
///   tile_max_tensor,
///   tile_sum_tensor          {batch_size, seq_len, classifier_bf16_tile_columns(padded_vocab_size)} float: scratch
///   logits_tensor            {batch_size, seq_len, padded_vocab_size} bfloat16: written, kept for the backward
///   lnf_output_tensor        {batch_size, seq_len, embed_dim} bfloat16: the last LayerNorm's output
///   wte_tensor               {padded_vocab_size, embed_dim} bfloat16: the token embedding
///   targets_tensor           {batch_size, seq_len} ints: the right next word of each row
///   vocab_size_val           the real words
///   threads_per_block_val    the loss kernel's block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not
///                            given: CLASSIFIER_BF16_LOSS_THREADS_PER_BLOCK. Any size gives the same bits (each row is
///                            one warp's); the logits kernel's block is one warpgroup, fixed by its tiles.
inline void classifier_bf16_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                    const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                                    const Tensor& tile_sum_tensor, const Tensor& logits_tensor,
                                    const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                    const Tensor& targets_tensor, size_t vocab_size_val, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "classifier_bf16_forward");
    classifier_bf16_check_forward("classifier_bf16_forward", losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor,
                                  tile_max_tensor, tile_sum_tensor, logits_tensor, lnf_output_tensor, wte_tensor,
                                  targets_tensor, vocab_size_val);
    size_t rows_val = targets_tensor.numel();
    size_t embed_dim_val = wte_tensor.size(1);
    size_t padded_vocab_size_val = wte_tensor.size(0);

    // 2. TMA's description of lnf_output's and wte's tiles: {lines, embed_dim} in boxes of {64 lines, 64 k}
    CUtensorMap lnf_output_map_val = wgmma_matmul_bf16_tensor_map(lnf_output_tensor.data_ptr<nv_bfloat16>(), rows_val,
                                                                  embed_dim_val, CLASSIFIER_BF16_TILE_ROWS);
    CUtensorMap wte_map_val = wgmma_matmul_bf16_tensor_map(wte_tensor.data_ptr<nv_bfloat16>(), padded_vocab_size_val,
                                                           embed_dim_val, CLASSIFIER_BF16_TILE_COLUMNS);

    // 3. the logits and each tile's (max, sum): tiles across and down, rounded up, as one line of blocks
    size_t total_block_columns_val = classifier_bf16_tile_columns(padded_vocab_size_val);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)CLASSIFIER_BF16_TILE_ROWS);
    classifier_bf16_logits_kernel<<<(unsigned)(total_block_columns_val * total_block_rows_val), HARDWARE_WARPGROUP_THREADS>>>(
        lnf_output_map_val, wte_map_val, logits_tensor.data_ptr<nv_bfloat16>(), tile_max_tensor.data(),
        tile_sum_tensor.data(), (int)rows_val, (int)padded_vocab_size_val, (int)vocab_size_val, (int)embed_dim_val);
    cudaCheck(cudaGetLastError());

    // 4. each row's tiles put together: row_max, row_sum, the loss
    classifier_bf16_losses(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, tile_max_tensor, tile_sum_tensor,
                           logits_tensor, targets_tensor, threads_per_block_val);
}

/// Works out one CLASSIFIER_BF16_TILE_ROWS x CLASSIFIER_BF16_TILE_COLUMNS tile of logits with one warpgroup, as
/// wgmma_matmul_bf16_kernel does (neither side turned: lnf_output's rows and wte's rows both hold embed_dim next to
/// each other): step by step, TMA brings a tile of each into shared memory, then the tensor cores add their product
/// into the sums. Then classifier_bf16_store_tile stores the tile and its rows' (max, sum).
///
///   lnf_output_map_val, wte_map_val          TMA's descriptions of lnf_output's and wte's tiles
///   logits_ptr                               {rows, padded_vocab_size}: written
///   tile_max_ptr, tile_sum_ptr               {rows, tile columns}: this tile's column written
///   rows_val, padded_vocab_size_val          logits' shape
///   vocab_size_val                           the real words: the columns the softmax takes
///   embed_dim_val                            the length summed over
__global__ void __launch_bounds__(HARDWARE_WARPGROUP_THREADS)
classifier_bf16_logits_kernel(const __grid_constant__ CUtensorMap lnf_output_map_val, const __grid_constant__ CUtensorMap wte_map_val,
                              nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr, int rows_val,
                              int padded_vocab_size_val, int vocab_size_val, int embed_dim_val) {
    namespace cde = cuda::device::experimental;
    // this step's tiles, in TMA's swizzled layout (on a HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom), and the barrier
    // that says they arrived
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 lnf_output_tile_list[CLASSIFIER_BF16_TILE_ROWS * WGMMA_MATMUL_BF16_BLOCK_K];
    __shared__ alignas(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) nv_bfloat16 wte_tile_list[CLASSIFIER_BF16_TILE_COLUMNS * WGMMA_MATMUL_BF16_BLOCK_K];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ WgmmaMatmulBf16Barrier tiles_barrier_val;

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
    int total_steps_val = CEIL_DIV(embed_dim_val, WGMMA_MATMUL_BF16_BLOCK_K);
    for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        // a. one thread asks for both tiles (past the edges TMA gives 0s) and says how many bytes to wait for; the
        //    others arrive; all wait until the bytes are in. A box's corner is (k, line)
        WgmmaMatmulBf16Barrier::arrival_token token_val;
        int step_first_k_index_val = step_index_val * WGMMA_MATMUL_BF16_BLOCK_K;
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

        // b. the tensor cores: WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs, each HARDWARE_WGMMA_K_BF16
        //    more values of k along both tiles' rows
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
#pragma unroll
        for (int k_step_index_val = 0; k_step_index_val < WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16; k_step_index_val++)
            wgmma_matmul_bf16_multiply_add<false, false>(sums_list, lnf_output_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16,
                                                         wte_tile_list + k_step_index_val * HARDWARE_WGMMA_K_BF16);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // ... done

        // c. every thread done with the tiles before TMA fills them again
        __syncthreads();
    }

    // 3. store the tile, and its rows' (max, sum)
    classifier_bf16_store_tile(sums_list, logits_ptr, tile_max_ptr, tile_sum_ptr, block_first_row_index_val,
                               block_column_index_val, total_block_columns_val, rows_val, padded_vocab_size_val,
                               vocab_size_val);
}

/// Stores this thread's part of a tile of logits, rounded to bfloat16, and writes each of the tile's rows'
/// (max, sum of exp(logit - max)) over the tile's real columns, from the rounded values. A thread holds 2 rows
/// (group and group + HARDWARE_WGMMA_ROWS_PER_WARP) x HARDWARE_WGMMA_COLUMNS_PER_THREAD columns of each of the
/// CLASSIFIER_BF16_TILE_GROUPS groups (cuda_common.h's WGMMA layout); a row's values are spread over
/// HARDWARE_WGMMA_THREADS_PER_ROW neighbouring lanes, put together by shuffles. A row with no real column in the
/// tile writes (-FLT_MAX, 0), which adds nothing when the tiles are put together. Every thread of the warpgroup
/// must call it (the shuffles). Also used by 8bit/classifier_8bit.cuh, its sums already times the scales.
///
///   logits_list                                   this thread's part of the tile, in float
///   logits_ptr                                    {rows, padded_vocab_size}: written
///   tile_max_ptr, tile_sum_ptr                    {rows, total_block_columns}: this tile's column written
///   block_first_row_index_val                     the tile's first row
///   block_column_index_val, total_block_columns_val   which tile across, of how many
///   rows_val, padded_vocab_size_val, vocab_size_val   logits' shape, and the real words
__device__ __forceinline__ void classifier_bf16_store_tile(
        const float (&logits_list)[CLASSIFIER_BF16_TILE_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
        nv_bfloat16* logits_ptr, float* tile_max_ptr, float* tile_sum_ptr, int block_first_row_index_val,
        int block_column_index_val, int total_block_columns_val, int rows_val, int padded_vocab_size_val,
        int vocab_size_val) {
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int thread_first_row_index_val = block_first_row_index_val + warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
    int thread_first_column_index_val = block_column_index_val * CLASSIFIER_BF16_TILE_COLUMNS +
                                        HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
#pragma unroll
    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {
        int row_index_val = thread_first_row_index_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
        bool row_in_val = row_index_val < rows_val;

        // 1. round each pair to bfloat16 and store it; keep the rounded values, and their largest over the real
        //    columns. padded_vocab_size is even, so a pair is all in or all out
        float rounded_list[CLASSIFIER_BF16_TILE_GROUPS][HARDWARE_WGMMA_COLUMNS_PER_THREAD];
        float thread_max_val = -FLT_MAX;
#pragma unroll
        for (int group_index_val = 0; group_index_val < CLASSIFIER_BF16_TILE_GROUPS; group_index_val++) {
            int column_index_val = thread_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS;
            nv_bfloat162 pair_val = __floats2bfloat162_rn(
                logits_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val],
                logits_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1]);
            if (row_in_val && column_index_val < padded_vocab_size_val)
                *reinterpret_cast<nv_bfloat162*>(logits_ptr + (size_t)row_index_val * padded_vocab_size_val + column_index_val) = pair_val;
            float2 rounded_val = __bfloat1622float2(pair_val);
            rounded_list[group_index_val][0] = rounded_val.x;
            rounded_list[group_index_val][1] = rounded_val.y;
#pragma unroll
            for (int pair_value_index_val = 0; pair_value_index_val < HARDWARE_WGMMA_COLUMNS_PER_THREAD; pair_value_index_val++)
                if (column_index_val + pair_value_index_val < vocab_size_val)
                    thread_max_val = fmaxf(thread_max_val, rounded_list[group_index_val][pair_value_index_val]);
        }

        // 2. this thread's sum of exp(logit - its max), over the real columns
        float thread_sum_val = 0.0f;
#pragma unroll
        for (int group_index_val = 0; group_index_val < CLASSIFIER_BF16_TILE_GROUPS; group_index_val++) {
            int column_index_val = thread_first_column_index_val + group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS;
#pragma unroll
            for (int pair_value_index_val = 0; pair_value_index_val < HARDWARE_WGMMA_COLUMNS_PER_THREAD; pair_value_index_val++)
                if (column_index_val + pair_value_index_val < vocab_size_val)
                    thread_sum_val += expf(rounded_list[group_index_val][pair_value_index_val] - thread_max_val);
        }

        // 3. the row's, over its HARDWARE_WGMMA_THREADS_PER_ROW lanes: m = max(m1, m2), s = s1 · exp(m1 - m) +
        //    s2 · exp(m2 - m); (-FLT_MAX, 0) adds nothing
        for (int offset_val = 1; offset_val < HARDWARE_WGMMA_THREADS_PER_ROW; offset_val *= 2) {
            float other_max_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, thread_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, thread_sum_val, offset_val);
            float new_max_val = fmaxf(thread_max_val, other_max_val);
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
            thread_max_val = new_max_val;
        }

        // 4. the row's first lane writes the tile's (max, sum)
        if (row_in_val && lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW == 0) {
            size_t tile_index_val = (size_t)row_index_val * total_block_columns_val + block_column_index_val;
            tile_max_ptr[tile_index_val] = thread_max_val;
            tile_sum_ptr[tile_index_val] = thread_sum_val;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of lnf_output and wte: checks, writes logits_grad (classifier_bf16_logits_grad: the
/// probs worked out again from logits, row_max and row_sum, not stored), then the head's backward
/// (linear_bf16_backward, no bias): lnf_output_grad = logits_grad · wte, wte_grad += logits_gradᵀ · lnf_output.
/// lnf_output_grad and logits_grad are written over; wte_grad is added onto (the encoder adds its part too).
///
///   lnf_output_grad_tensor                           lnf_output's shape, bfloat16
///   wte_grad_tensor                                  wte's shape, bfloat16: added onto
///   logits_grad_tensor                               logits' shape, bfloat16; may be logits itself
///   losses_grad_tensor                               {batch_size, seq_len} float: d(loss) / d(losses[row])
///   logits_tensor, lnf_output_tensor, wte_tensor,
///   targets_tensor                                   the forward's
///   max_of_each_row_tensor, sum_of_each_row_tensor   the forward's results
///   vocab_size_val                                   the real words
///   threads_per_block_val                            logits_grad's block size: whole warps, at most
///                                                    HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                                                    CLASSIFIER_BF16_BACKWARD_THREADS_PER_BLOCK. Any size gives the
///                                                    same bits; only the time changes.
inline void classifier_bf16_backward(const Tensor& lnf_output_grad_tensor, const Tensor& wte_grad_tensor,
                                     const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                                     const Tensor& logits_tensor, const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                     const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                                     const Tensor& sum_of_each_row_tensor, size_t vocab_size_val,
                                     int threads_per_block_val) {
    // 1. logits_grad, written over (it checks its tensors)
    classifier_bf16_logits_grad("classifier_bf16_backward", logits_grad_tensor, losses_grad_tensor, logits_tensor,
                                targets_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, vocab_size_val,
                                threads_per_block_val);

    // 2. the head's two products (linear_bf16_backward checks the rest)
    linear_bf16_backward(lnf_output_grad_tensor, wte_grad_tensor, Tensor(), logits_grad_tensor, lnf_output_tensor, wte_tensor);
}

// ── for both, and for 8bit/classifier_8bit.cuh ────────────────────────────────

/// Gives the tiles of logits across a row: padded_vocab_size / CLASSIFIER_BF16_TILE_COLUMNS, rounded up. tile_max
/// and tile_sum hold one value per row per tile.
///
///   padded_vocab_size_val   logits' row length
inline size_t classifier_bf16_tile_columns(size_t padded_vocab_size_val) {
    return CEIL_DIV(padded_vocab_size_val, (size_t)CLASSIFIER_BF16_TILE_COLUMNS);
}

/// Stops with a message unless a forward's tensors fit: the shapes at the top of the file, 0 < vocab_size <=
/// padded_vocab_size, the types, embed_dim and padded_vocab_size multiples of WGMMA_MATMUL_BF16_VALUE_MULTIPLE
/// (TMA reads rows of whole 16-byte pieces), each bfloat16 tensor starting on one.
///
///   name_ptr     the caller's name, for the message
///   the others   as in classifier_bf16_forward
inline void classifier_bf16_check_forward(const char* name_ptr, const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                          const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                                          const Tensor& tile_sum_tensor, const Tensor& logits_tensor,
                                          const Tensor& lnf_output_tensor, const Tensor& wte_tensor,
                                          const Tensor& targets_tensor, size_t vocab_size_val) {
    // 1. shapes: as at the top of the file
    bool fits_val = targets_tensor.dim() == 2 && wte_tensor.dim() == 2;
    if (fits_val) {
        size_t batch_size_val = targets_tensor.size(0);
        size_t seq_len_val = targets_tensor.size(1);
        size_t padded_vocab_size_val = wte_tensor.size(0);
        size_t embed_dim_val = wte_tensor.size(1);
        Shape per_row_list = {batch_size_val, seq_len_val};
        fits_val = lnf_output_tensor.shape() == Shape({batch_size_val, seq_len_val, embed_dim_val}) &&
                   logits_tensor.shape() == Shape({batch_size_val, seq_len_val, padded_vocab_size_val}) &&
                   losses_tensor.shape() == per_row_list && max_of_each_row_tensor.shape() == per_row_list &&
                   sum_of_each_row_tensor.shape() == per_row_list &&
                   tile_max_tensor.shape() == Shape({batch_size_val, seq_len_val, classifier_bf16_tile_columns(padded_vocab_size_val)}) &&
                   tile_sum_tensor.shape() == tile_max_tensor.shape() && vocab_size_val > 0 &&
                   vocab_size_val <= padded_vocab_size_val;
    }
    if (!fits_val) {
        fprintf(stderr, "error: %s needs lnf_output {batch_size, seq_len, embed_dim}, wte {padded_vocab_size, "
                        "embed_dim}, logits {batch_size, seq_len, padded_vocab_size}, targets, losses, "
                        "max_of_each_row and sum_of_each_row {batch_size, seq_len}, tile_max and tile_sum "
                        "{batch_size, seq_len, padded_vocab_size / %d rounded up}, and 0 < vocab_size <= "
                        "padded_vocab_size\n", name_ptr, CLASSIFIER_BF16_TILE_COLUMNS);
        exit(EXIT_FAILURE);
    }
    // 2. types: lnf_output, wte and logits bfloat16, targets ints, the rest float
    bool types_fit_val = lnf_output_tensor.dtype() == DType::BFloat16 && wte_tensor.dtype() == DType::BFloat16 &&
                         logits_tensor.dtype() == DType::BFloat16 && targets_tensor.dtype() == DType::Int32;
    for (const Tensor* tensor_ptr : {&losses_tensor, &max_of_each_row_tensor, &sum_of_each_row_tensor, &tile_max_tensor, &tile_sum_tensor})
        types_fit_val = types_fit_val && tensor_ptr->dtype() == DType::Float32;
    if (!types_fit_val) {
        fprintf(stderr, "error: %s needs lnf_output, wte and logits bfloat16, targets Int32, and losses, "
                        "max_of_each_row, sum_of_each_row, tile_max and tile_sum float\n", name_ptr);
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte pieces, each bfloat16 tensor starting on one
    size_t embed_dim_val = wte_tensor.size(1);
    size_t padded_vocab_size_val = wte_tensor.size(0);
    bool counts_fit_val = embed_dim_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE == 0 &&
                          padded_vocab_size_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE == 0;
    for (const Tensor* tensor_ptr : {&lnf_output_tensor, &wte_tensor, &logits_tensor})
        counts_fit_val = counts_fit_val && (size_t)tensor_ptr->data_ptr() % HARDWARE_TMA_BYTES_PER_CELL == 0;
    if (!counts_fit_val) {
        fprintf(stderr, "error: %s's TMA reads rows of whole %d-byte pieces, so embed_dim and padded_vocab_size must "
                        "be multiples of %d and lnf_output, wte and logits must start on %d bytes; got embed_dim %zu, "
                        "padded_vocab_size %zu\n", name_ptr, HARDWARE_TMA_BYTES_PER_CELL, WGMMA_MATMUL_BF16_VALUE_MULTIPLE,
                HARDWARE_TMA_BYTES_PER_CELL, embed_dim_val, padded_vocab_size_val);
        exit(EXIT_FAILURE);
    }
}

__global__ void classifier_bf16_loss_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                            const float* tile_max_ptr, const float* tile_sum_ptr,
                                            const nv_bfloat16* logits_ptr, const int* targets_ptr, size_t rows_val,
                                            int total_block_columns_val, size_t padded_vocab_size_val);

/// Writes each row's row_max, row_sum and loss from its tiles' (max, sum): launches classifier_bf16_loss_kernel,
/// one warp per row. The callers have checked the tensors.
///
///   losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor   written
///   tile_max_tensor, tile_sum_tensor                                the logits kernel's
///   logits_tensor, targets_tensor                                   the target's logit is read from logits
///   threads_per_block_val                                           as in classifier_bf16_forward, checked there
inline void classifier_bf16_losses(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                   const Tensor& sum_of_each_row_tensor, const Tensor& tile_max_tensor,
                                   const Tensor& tile_sum_tensor, const Tensor& logits_tensor, const Tensor& targets_tensor,
                                   int threads_per_block_val) {
    size_t rows_val = targets_tensor.numel();
    int blocks_val = (int)CEIL_DIV(rows_val * HARDWARE_THREADS_PER_WARP, (size_t)threads_per_block_val);
    classifier_bf16_loss_kernel<<<blocks_val, threads_per_block_val>>>(
        losses_tensor.data(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), tile_max_tensor.data(),
        tile_sum_tensor.data(), logits_tensor.data_ptr<nv_bfloat16>(), targets_tensor.data_ptr<int>(), rows_val,
        (int)tile_max_tensor.size(2), logits_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Puts each row's tiles' (max, sum) together, one warp per row, the online softmax's way: each lane takes tiles
/// lane, lane + HARDWARE_THREADS_PER_WARP, ..., then the lanes' are put together by shuffles; lane 0 writes
/// row_max, row_sum and loss = log(row_sum) - (logit[target] - row_max), the target's logit read as stored.
///
///   losses_ptr, max_of_each_row_ptr, sum_of_each_row_ptr   one float per row: written
///   tile_max_ptr, tile_sum_ptr                             {rows, total_block_columns}
///   logits_ptr, targets_ptr                                {rows, padded_vocab_size} bfloat16, one int per row
///   rows_val, total_block_columns_val, padded_vocab_size_val   the shapes
__global__ void classifier_bf16_loss_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                            const float* tile_max_ptr, const float* tile_sum_ptr,
                                            const nv_bfloat16* logits_ptr, const int* targets_ptr, size_t rows_val,
                                            int total_block_columns_val, size_t padded_vocab_size_val) {
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    size_t warp_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / HARDWARE_THREADS_PER_WARP;   // which warp of all launched
    size_t warps_launched_val = (size_t)gridDim.x * blockDim.x / HARDWARE_THREADS_PER_WARP;
    // Warp n takes rows n, n + every warp launched, ...: the same for all its lanes, so the warp stays whole
    for (size_t row_index_val = warp_index_val; row_index_val < rows_val; row_index_val += warps_launched_val) {
        const float* row_tile_max_ptr = tile_max_ptr + row_index_val * total_block_columns_val;
        const float* row_tile_sum_ptr = tile_sum_ptr + row_index_val * total_block_columns_val;

        // 1. this lane's (max, sum) over its tiles
        float row_max_val = -FLT_MAX;
        float row_sum_val = 0.0f;
        for (int tile_index_val = lane_index_val; tile_index_val < total_block_columns_val; tile_index_val += HARDWARE_THREADS_PER_WARP) {
            float tile_max_val = row_tile_max_ptr[tile_index_val];
            float new_max_val = fmaxf(row_max_val, tile_max_val);
            row_sum_val = row_sum_val * expf(row_max_val - new_max_val) + row_tile_sum_ptr[tile_index_val] * expf(tile_max_val - new_max_val);
            row_max_val = new_max_val;
        }

        // 2. the warp's: log2(HARDWARE_THREADS_PER_WARP) swaps; m = max(m1, m2), s = s1 · exp(m1 - m) + s2 · exp(m2 - m)
        for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
            float other_max_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, row_sum_val, offset_val);
            float new_max_val = fmaxf(row_max_val, other_max_val);
            row_sum_val = row_sum_val * expf(row_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
            row_max_val = new_max_val;
        }

        // 3. lane 0 writes row_max, row_sum and the loss
        if (lane_index_val == 0) {
            float target_logit_val = __bfloat162float(logits_ptr[row_index_val * padded_vocab_size_val + targets_ptr[row_index_val]]);
            max_of_each_row_ptr[row_index_val] = row_max_val;
            sum_of_each_row_ptr[row_index_val] = row_sum_val;
            losses_ptr[row_index_val] = logf(row_sum_val) - (target_logit_val - row_max_val);
        }
    }
}

__global__ void classifier_bf16_logits_grad_kernel(nv_bfloat16* logits_grad_ptr, const float* losses_grad_ptr,
                                                   const nv_bfloat16* logits_ptr, const int* targets_ptr,
                                                   const float* max_of_each_row_ptr, const float* sum_of_each_row_ptr,
                                                   size_t rows_val, size_t vocab_size_val, size_t padded_vocab_size_val);

/// Writes logits_grad (written over, not added): checks, then launches classifier_bf16_logits_grad_kernel, one
/// float4 per thread but at most CLASSIFIER_BF16_BACKWARD_BLOCKS blocks.
///
///   name_ptr                 the caller's name, for the messages
///   threads_per_block_val    the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK
///   the others               as in classifier_bf16_backward
inline void classifier_bf16_logits_grad(const char* name_ptr, const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                                        const Tensor& logits_tensor, const Tensor& targets_tensor,
                                        const Tensor& max_of_each_row_tensor, const Tensor& sum_of_each_row_tensor,
                                        size_t vocab_size_val, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, name_ptr);
    bool fits_val = logits_tensor.dim() == 3 && logits_tensor.dtype() == DType::BFloat16 &&
                    logits_grad_tensor.shape() == logits_tensor.shape() && logits_grad_tensor.dtype() == DType::BFloat16 &&
                    targets_tensor.shape() == Shape({logits_tensor.size(0), logits_tensor.size(1)}) &&
                    targets_tensor.dtype() == DType::Int32 && vocab_size_val > 0 && vocab_size_val <= logits_tensor.size(2) &&
                    logits_tensor.size(2) % VALUES_PER_FLOAT4 == 0;
    for (const Tensor* tensor_ptr : {&losses_grad_tensor, &max_of_each_row_tensor, &sum_of_each_row_tensor})
        fits_val = fits_val && tensor_ptr->shape() == targets_tensor.shape() && tensor_ptr->dtype() == DType::Float32;
    for (const Tensor* tensor_ptr : {&logits_tensor, &logits_grad_tensor})
        fits_val = fits_val && (size_t)tensor_ptr->data_ptr() % (VALUES_PER_FLOAT4 * sizeof(nv_bfloat16)) == 0;
    if (!fits_val) {
        fprintf(stderr, "error: %s needs logits and logits_grad bfloat16 {batch_size, seq_len, padded_vocab_size} "
                        "(a multiple of %d, starting on a whole float4 of bfloat16s), targets Int32 and losses_grad, "
                        "max_of_each_row and sum_of_each_row float {batch_size, seq_len}, and 0 < vocab_size <= "
                        "padded_vocab_size\n", name_ptr, VALUES_PER_FLOAT4);
        exit(EXIT_FAILURE);
    }

    // 2. launch: one float4 per thread, but at most CLASSIFIER_BF16_BACKWARD_BLOCKS blocks
    size_t rows_val = targets_tensor.numel();
    size_t total_float4s_val = logits_tensor.numel() / VALUES_PER_FLOAT4;
    size_t one_float4_per_thread_blocks_val = CEIL_DIV(total_float4s_val, (size_t)threads_per_block_val);
    int blocks_val = (int)(one_float4_per_thread_blocks_val < CLASSIFIER_BF16_BACKWARD_BLOCKS ? one_float4_per_thread_blocks_val
                                                                                              : CLASSIFIER_BF16_BACKWARD_BLOCKS);
    classifier_bf16_logits_grad_kernel<<<blocks_val, threads_per_block_val>>>(
        logits_grad_tensor.data_ptr<nv_bfloat16>(), losses_grad_tensor.data(), logits_tensor.data_ptr<nv_bfloat16>(),
        targets_tensor.data_ptr<int>(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), rows_val,
        vocab_size_val, logits_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Writes logits_grad one float4 at a time: works out each value's prob again, takes 1 off at the target, and
/// scales by the row's loss grad; padding gets 0. Reads each float4 of logits before writing that of logits_grad,
/// so the two may be the same tensor.
///
///   logits_grad_ptr, losses_grad_ptr, logits_ptr, targets_ptr   as in classifier_bf16_backward
///   max_of_each_row_ptr, sum_of_each_row_ptr                    the forward's results
///   rows_val                                                    batch_size x seq_len
///   vocab_size_val, padded_vocab_size_val                       real words, and the row length
__global__ void classifier_bf16_logits_grad_kernel(nv_bfloat16* logits_grad_ptr, const float* losses_grad_ptr,
                                                   const nv_bfloat16* logits_ptr, const int* targets_ptr,
                                                   const float* max_of_each_row_ptr, const float* sum_of_each_row_ptr,
                                                   size_t rows_val, size_t vocab_size_val, size_t padded_vocab_size_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_float4s_val = padded_vocab_size_val / VALUES_PER_FLOAT4;
    size_t total_float4s_val = rows_val * row_float4s_val;
    size_t threads_launched_val = (size_t)gridDim.x * blockDim.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t float4_index_val = thread_index_val; float4_index_val < total_float4s_val; float4_index_val += threads_launched_val) {
        // 1. where this float4 is, and its row's numbers from the forward
        size_t row_index_val = float4_index_val / row_float4s_val;
        size_t first_column_index_val = (float4_index_val % row_float4s_val) * VALUES_PER_FLOAT4;
        size_t target_index_val = (size_t)targets_ptr[row_index_val];
        float row_max_val = max_of_each_row_ptr[row_index_val];
        float one_over_row_sum_val = 1.0f / sum_of_each_row_ptr[row_index_val];
        float row_loss_grad_val = losses_grad_ptr[row_index_val];

        // 2. each value: (prob - 1 at the target) x the row's loss grad; padding: 0
        float4 logits_val = read4_as_float(logits_ptr + VALUES_PER_FLOAT4 * float4_index_val);
        float logit_list[VALUES_PER_FLOAT4] = {logits_val.x, logits_val.y, logits_val.z, logits_val.w};
        float grad_list[VALUES_PER_FLOAT4];
#pragma unroll
        for (int value_index_val = 0; value_index_val < VALUES_PER_FLOAT4; value_index_val++) {
            size_t column_index_val = first_column_index_val + value_index_val;
            float prob_val = expf(logit_list[value_index_val] - row_max_val) * one_over_row_sum_val;
            float one_at_target_val = column_index_val == target_index_val ? 1.0f : 0.0f;
            grad_list[value_index_val] = column_index_val < vocab_size_val ? (prob_val - one_at_target_val) * row_loss_grad_val : 0.0f;
        }
        // 3. write
        write4_from_float(logits_grad_ptr + VALUES_PER_FLOAT4 * float4_index_val,
                          make_float4(grad_list[0], grad_list[1], grad_list[2], grad_list[3]));
    }
}

#endif // CLASSIFIER_BF16_CUH
