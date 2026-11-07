// classifier.cuh — GPT-2's last step: softmax over each row of logits, and the cross-entropy loss:
//     row_max = max of the row's logits
//     row_sum = sum of exp(logit - row_max)
//     prob    = exp(logit - row_max) / row_sum
//     loss    = -log(prob[target]) = log(row_sum) - (logit[target] - row_max)
//     backward:   logits_grad = (prob - 1 at the target, prob elsewhere) · losses_grad[row]
//
//     logits, logits_grad                {batch_size, seq_len, padded_vocab_size}   {4, 1024, 50304}
//     targets                            {batch_size, seq_len} ints                 {4, 1024}
//     losses, losses_grad                {batch_size, seq_len}
//     max_of_each_row, sum_of_each_row   {batch_size, seq_len}: kept for the backward
// Columns from vocab_size (50257) on are padding: left out of the softmax, gradient 0. logits and logits_grad
// are float or bfloat16 (T); the rest float; the math is in float.
// Math: https://ducbachsong.github.io/posts/classifier-backward/
#ifndef CLASSIFIER_CUH
#define CLASSIFIER_CUH

#include "cuda_common.h"             // cudaCheck, check_threads_per_block, warp and block defines
#include "../tensor.cuh"

#define CLASSIFIER_FORWARD_THREADS_PER_BLOCK (8 * THREADS_PER_WARP)    // the forward's default: 8 warps, one block per row
#define CLASSIFIER_BACKWARD_BLOCKS 40960                               // at most
#define CLASSIFIER_BACKWARD_THREADS_PER_BLOCK (4 * THREADS_PER_WARP)   // the backward's default: 4 warps, fastest in bfloat16 on an H100 PCIe

// ── forward and backward ──────────────────────────────────────────────────────
void classifier_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                        const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor, const Tensor& targets_tensor,
                        size_t vocab_size_val, int threads_per_block_val = CLASSIFIER_FORWARD_THREADS_PER_BLOCK);
void classifier_backward(const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor, const Tensor& logits_tensor,
                         const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                         const Tensor& sum_of_each_row_tensor, size_t vocab_size_val,
                         int threads_per_block_val = CLASSIFIER_BACKWARD_THREADS_PER_BLOCK);

// ── for both ──────────────────────────────────────────────────────────────────
void classifier_check_shapes(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                             const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor,
                             const Tensor& targets_tensor, size_t vocab_size_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void classifier_forward_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                          const T* logits_ptr, const int* targets_ptr, size_t rows_val,
                                          size_t vocab_size_val, size_t padded_vocab_size_val);

/// Writes each row's loss, row_max and row_sum: checks, then launches classifier_forward_kernel, one block
/// per row. The targets are not checked: a target >= vocab_size reads a padding column.
///
///   losses_tensor            one loss per row
///   max_of_each_row_tensor   row_max, for the backward
///   sum_of_each_row_tensor   row_sum, for the backward
///   logits_tensor            the scores, left as they are
///   targets_tensor           the right next word of each row
///   vocab_size_val           the real words: 50257
///   threads_per_block_val    the block size: whole warps, at most MAX_THREADS_PER_BLOCK; not given:
///                            CLASSIFIER_FORWARD_THREADS_PER_BLOCK. Another size adds each row in another order,
///                            so the last bits can change.
inline void classifier_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                               const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor,
                               const Tensor& targets_tensor, size_t vocab_size_val, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the classifier");
    classifier_check_shapes(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor,
                            targets_tensor, vocab_size_val);
    // 2. launch: one block per row
    size_t rows_val = logits_tensor.size(0) * logits_tensor.size(1);
    if (logits_tensor.dtype() == DType::BFloat16)
        classifier_forward_kernel<<<(int)rows_val, threads_per_block_val>>>(
            losses_tensor.data(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(),
            logits_tensor.data_ptr<nv_bfloat16>(), targets_tensor.data_ptr<int>(), rows_val, vocab_size_val,
            logits_tensor.size(2));
    else
        classifier_forward_kernel<<<(int)rows_val, threads_per_block_val>>>(
            losses_tensor.data(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), logits_tensor.data(),
            targets_tensor.data_ptr<int>(), rows_val, vocab_size_val, logits_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Works out each row's row_max, row_sum and loss in one read of the row (the online softmax). Each thread
/// keeps its largest logit and its sum of exp(logit - largest); when a larger logit comes, the sum is moved
/// onto it: sum · exp(old - new). Then the threads' (max, sum) are put together the same way, across each
/// warp, then across the warps.
///
///   losses_ptr, max_of_each_row_ptr, sum_of_each_row_ptr,
///   logits_ptr, targets_ptr                  as in classifier_forward, logits float or bfloat16 (T)
///   rows_val                                 batch_size x seq_len
///   vocab_size_val, padded_vocab_size_val    real words, and the row length
template <typename T>
__global__ void classifier_forward_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                          const T* logits_ptr, const int* targets_ptr, size_t rows_val,
                                          size_t vocab_size_val, size_t padded_vocab_size_val) {
    // one slot per warp; warp 0's THREADS_PER_WARP lanes must cover them all
    static_assert(MAX_WARPS_PER_BLOCK <= THREADS_PER_WARP, "warp 0 must hold one (max, sum) per warp");
    __shared__ float max_of_each_warp_list[MAX_WARPS_PER_BLOCK];
    __shared__ float sum_of_each_warp_list[MAX_WARPS_PER_BLOCK];
    int lane_index_val = threadIdx.x % THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / THREADS_PER_WARP;
    size_t float4s_with_words_val = CEIL_DIV(vocab_size_val, (size_t)VALUES_PER_FLOAT4);   // the padding-only ones left out

    // Block n takes rows n, n + every block launched, ...
    for (size_t row_index_val = blockIdx.x; row_index_val < rows_val; row_index_val += gridDim.x) {
        const T* row_logits_ptr = logits_ptr + row_index_val * padded_vocab_size_val;

        // 1. this thread's (max, sum) over its float4s; padding counts as -FLT_MAX, exp 0
        float thread_max_val = -FLT_MAX;
        float thread_sum_val = 0.0f;
        for (size_t row_float4_index_val = threadIdx.x; row_float4_index_val < float4s_with_words_val;
             row_float4_index_val += blockDim.x) {
            float4 logits_val = read4_as_float(row_logits_ptr + VALUES_PER_FLOAT4 * row_float4_index_val);
            size_t first_column_val = row_float4_index_val * VALUES_PER_FLOAT4;
            float logit0_val = logits_val.x;                                         // always a word
            float logit1_val = first_column_val + 1 < vocab_size_val ? logits_val.y : -FLT_MAX;
            float logit2_val = first_column_val + 2 < vocab_size_val ? logits_val.z : -FLT_MAX;
            float logit3_val = first_column_val + 3 < vocab_size_val ? logits_val.w : -FLT_MAX;
            float new_max_val = fmaxf(thread_max_val, fmaxf(fmaxf(logit0_val, logit1_val), fmaxf(logit2_val, logit3_val)));
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_max_val) +   // the sum so far, moved onto the new max
                             expf(logit0_val - new_max_val) + expf(logit1_val - new_max_val) +
                             expf(logit2_val - new_max_val) + expf(logit3_val - new_max_val);
            thread_max_val = new_max_val;
        }

        // 2. the warp's: log2(THREADS_PER_WARP) swaps; m = max(m1, m2), s = s1 · exp(m1 - m) + s2 · exp(m2 - m)
        float warp_max_val = thread_max_val;
        float warp_sum_val = thread_sum_val;
        for (int offset_val = THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
            float other_max_val = __shfl_xor_sync(WARP_ALL_LANES, warp_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(WARP_ALL_LANES, warp_sum_val, offset_val);
            float new_max_val = fmaxf(warp_max_val, other_max_val);
            warp_sum_val = warp_sum_val * expf(warp_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
            warp_max_val = new_max_val;
        }

        // 3. the row's: warp 0 puts the warps' parts together; its lane 0 writes the results
        if (lane_index_val == 0) {
            max_of_each_warp_list[warp_index_val] = warp_max_val;
            sum_of_each_warp_list[warp_index_val] = warp_sum_val;
        }
        __syncthreads();
        if (warp_index_val == 0) {
            float row_max_val = lane_index_val < warps_in_block_val ? max_of_each_warp_list[lane_index_val] : -FLT_MAX;
            float row_sum_val = lane_index_val < warps_in_block_val ? sum_of_each_warp_list[lane_index_val] : 0.0f;
            for (int offset_val = THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
                float other_max_val = __shfl_xor_sync(WARP_ALL_LANES, row_max_val, offset_val);
                float other_sum_val = __shfl_xor_sync(WARP_ALL_LANES, row_sum_val, offset_val);
                float new_max_val = fmaxf(row_max_val, other_max_val);
                row_sum_val = row_sum_val * expf(row_max_val - new_max_val) + other_sum_val * expf(other_max_val - new_max_val);
                row_max_val = new_max_val;
            }
            // 4. lane 0 writes row_max, row_sum and the loss
            if (lane_index_val == 0) {
                float target_logit_val = to_float(row_logits_ptr[targets_ptr[row_index_val]]);
                max_of_each_row_ptr[row_index_val] = row_max_val;
                sum_of_each_row_ptr[row_index_val] = row_sum_val;
                losses_ptr[row_index_val] = logf(row_sum_val) - (target_logit_val - row_max_val);
            }
        }
        __syncthreads();                       // before the next row writes the two lists again
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void classifier_backward_kernel(T* logits_grad_ptr, const float* losses_grad_ptr, const T* logits_ptr,
                                           const int* targets_ptr, const float* max_of_each_row_ptr,
                                           const float* sum_of_each_row_ptr, size_t rows_val, size_t vocab_size_val,
                                           size_t padded_vocab_size_val);

/// Writes logits_grad (written over, not added): checks, then launches classifier_backward_kernel. The probs
/// are worked out again from logits, row_max and row_sum, not stored.
///
///   logits_grad_tensor                               the result; may be logits itself
///   losses_grad_tensor                               d(loss) / d(losses[row]): 1 / rows for the mean
///   logits_tensor, targets_tensor                    the forward's
///   max_of_each_row_tensor, sum_of_each_row_tensor   the forward's results
///   vocab_size_val                                   the real words: 50257
///   threads_per_block_val                            the block size: whole warps, at most MAX_THREADS_PER_BLOCK;
///                                                    not given: CLASSIFIER_BACKWARD_THREADS_PER_BLOCK. Any size
///                                                    gives the same bits; only the time changes.
inline void classifier_backward(const Tensor& logits_grad_tensor, const Tensor& losses_grad_tensor,
                                const Tensor& logits_tensor, const Tensor& targets_tensor,
                                const Tensor& max_of_each_row_tensor, const Tensor& sum_of_each_row_tensor,
                                size_t vocab_size_val, int threads_per_block_val) {
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "the classifier");
    classifier_check_shapes(losses_grad_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_tensor,
                            targets_tensor, vocab_size_val);
    // logits_grad: logits' shape and type, starting on a whole float4
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * logits_tensor.element_size();
    if (logits_grad_tensor.shape() != logits_tensor.shape() || logits_grad_tensor.dtype() != logits_tensor.dtype() ||
        (size_t)logits_grad_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: the classifier needs logits_grad the same shape and dtype as logits, starting on %zu "
                        "bytes\n", float4_bytes_val);
        exit(EXIT_FAILURE);
    }
    // 2. launch: one float4 per thread, but at most CLASSIFIER_BACKWARD_BLOCKS blocks
    size_t rows_val = logits_tensor.size(0) * logits_tensor.size(1);
    size_t total_float4s_val = logits_tensor.numel() / VALUES_PER_FLOAT4;
    size_t one_float4_per_thread_blocks_val = CEIL_DIV(total_float4s_val, (size_t)threads_per_block_val);
    int blocks_val = (int)(one_float4_per_thread_blocks_val < CLASSIFIER_BACKWARD_BLOCKS ? one_float4_per_thread_blocks_val
                                                                                         : CLASSIFIER_BACKWARD_BLOCKS);
    if (logits_tensor.dtype() == DType::BFloat16)
        classifier_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            logits_grad_tensor.data_ptr<nv_bfloat16>(), losses_grad_tensor.data(), logits_tensor.data_ptr<nv_bfloat16>(),
            targets_tensor.data_ptr<int>(), max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), rows_val,
            vocab_size_val, logits_tensor.size(2));
    else
        classifier_backward_kernel<<<blocks_val, threads_per_block_val>>>(
            logits_grad_tensor.data(), losses_grad_tensor.data(), logits_tensor.data(), targets_tensor.data_ptr<int>(),
            max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(), rows_val, vocab_size_val,
            logits_tensor.size(2));
    cudaCheck(cudaGetLastError());
}

/// Writes logits_grad one float4 at a time: works out each value's prob again, takes 1 off at the target, and
/// scales by the row's loss grad; padding gets 0.
///
///   logits_grad_ptr, losses_grad_ptr, logits_ptr, targets_ptr   as in classifier_backward, logits float or bfloat16 (T)
///   max_of_each_row_ptr, sum_of_each_row_ptr                    the forward's results
///   rows_val                                                    batch_size x seq_len
///   vocab_size_val, padded_vocab_size_val                       real words, and the row length
template <typename T>
__global__ void classifier_backward_kernel(T* logits_grad_ptr, const float* losses_grad_ptr, const T* logits_ptr,
                                           const int* targets_ptr, const float* max_of_each_row_ptr,
                                           const float* sum_of_each_row_ptr, size_t rows_val, size_t vocab_size_val,
                                           size_t padded_vocab_size_val) {
    size_t thread_index_val = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_float4s_val = padded_vocab_size_val / VALUES_PER_FLOAT4;
    size_t total_float4s_val = rows_val * row_float4s_val;
    size_t threads_launched_val = (size_t)gridDim.x * blockDim.x;
    // Thread n takes float4s n, n + every thread launched, ...
    for (size_t index_val = thread_index_val; index_val < total_float4s_val; index_val += threads_launched_val) {
        // 1. where this float4 is, and its row's numbers from the forward
        size_t row_index_val = index_val / row_float4s_val;
        size_t first_column_val = (index_val % row_float4s_val) * VALUES_PER_FLOAT4;
        size_t target_val = (size_t)targets_ptr[row_index_val];
        float row_max_val = max_of_each_row_ptr[row_index_val];
        float one_over_row_sum_val = 1.0f / sum_of_each_row_ptr[row_index_val];
        float row_loss_grad_val = losses_grad_ptr[row_index_val];

        // 2. each value: (prob - 1 at the target) x the row's loss grad; padding: 0
        float4 logits_val = read4_as_float(logits_ptr + VALUES_PER_FLOAT4 * index_val);
        float logit_list[VALUES_PER_FLOAT4] = {logits_val.x, logits_val.y, logits_val.z, logits_val.w};
        float grad_list[VALUES_PER_FLOAT4];
        for (int value_index_val = 0; value_index_val < VALUES_PER_FLOAT4; value_index_val++) {
            size_t column_val = first_column_val + value_index_val;
            float prob_val = expf(logit_list[value_index_val] - row_max_val) * one_over_row_sum_val;
            float one_at_target_val = column_val == target_val ? 1.0f : 0.0f;
            grad_list[value_index_val] = column_val < vocab_size_val ? (prob_val - one_at_target_val) * row_loss_grad_val : 0.0f;
        }
        // 3. write
        write4_from_float(logits_grad_ptr + VALUES_PER_FLOAT4 * index_val,
                          make_float4(grad_list[0], grad_list[1], grad_list[2], grad_list[3]));
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit: the shapes at the top of the file, 0 < vocab_size <=
/// padded_vocab_size, logits float or bfloat16 and the others float, padded_vocab_size a multiple of
/// VALUES_PER_FLOAT4 and logits starting on a whole float4.
///
///   losses_tensor                                    forward: losses; backward: losses_grad
///   max_of_each_row_tensor, sum_of_each_row_tensor   row_max and row_sum
///   logits_tensor, targets_tensor                    logits and the right next words
///   vocab_size_val                                   the real words
inline void classifier_check_shapes(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                    const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor,
                                    const Tensor& targets_tensor, size_t vocab_size_val) {
    // shapes: as at the top of the file
    bool fits_val = logits_tensor.dim() == 3 &&
                    targets_tensor.shape() == Shape({logits_tensor.size(0), logits_tensor.size(1)}) &&
                    targets_tensor.dtype() == DType::Int32 && losses_tensor.shape() == targets_tensor.shape() &&
                    max_of_each_row_tensor.shape() == targets_tensor.shape() &&
                    sum_of_each_row_tensor.shape() == targets_tensor.shape() && vocab_size_val > 0 &&
                    vocab_size_val <= logits_tensor.size(2);
    if (!fits_val) {
        fprintf(stderr, "error: the classifier needs logits {batch_size, seq_len, padded_vocab_size}; targets (ints), "
                        "losses, max_of_each_row and sum_of_each_row {batch_size, seq_len}; and 0 < vocab_size <= "
                        "padded_vocab_size\n");
        exit(EXIT_FAILURE);
    }
    // types: logits float or bfloat16, the rest float
    if ((logits_tensor.dtype() != DType::Float32 && logits_tensor.dtype() != DType::BFloat16) ||
        losses_tensor.dtype() != DType::Float32 || max_of_each_row_tensor.dtype() != DType::Float32 ||
        sum_of_each_row_tensor.dtype() != DType::Float32) {
        fprintf(stderr, "error: the classifier needs logits float or bfloat16, and losses, max_of_each_row and "
                        "sum_of_each_row float\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: rows of whole float4s, logits starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * logits_tensor.element_size();
    if (logits_tensor.size(2) % VALUES_PER_FLOAT4 != 0 || (size_t)logits_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: the classifier reads %d values at a time, so padded_vocab_size must be a multiple of "
                        "%d and logits must start on %zu bytes; got %zu, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, logits_tensor.size(2), logits_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }
}

#endif // CLASSIFIER_CUH
