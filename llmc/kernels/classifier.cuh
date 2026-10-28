// classifier.cuh — GPT-2's last step: each row of logits (one token's score for every word of the
// vocabulary) becomes probabilities with softmax, and the loss is how unlikely the right next word was.
//
// The math is in the note https://ducbachsong.github.io/posts/classifier-backward/. The comments below
// name its steps (F1..F5 forward, B1..B7 backward, C1 and C2 rebuilt) and sections. Its letters for the
// names here:
//
//     z = logit   t = target   m = row_max   S = row_sum   p = prob   ℓ = loss   dℓ = losses_grad   dz = logits_grad
//
//     row_max  = the largest logit of the row                                       F1
//     row_sum  = sum over the vocabulary of exp(logit - row_max)                    F2, F3, F4
//     prob     = exp(logit - row_max) / row_sum        softmax: every prob is in 0..1, and they add up to 1
//     loss     = -log(prob[target])                    cross-entropy: 0 if the right word got prob 1
//              = log(row_sum) - (logit[target] - row_max)                           F5
//
//     logits, logits_grad      {batch_size, seq_len, padded_vocab_size}   one score per word:    {4, 1024, 50304}
//     targets                  {batch_size, seq_len} ints                 the right next word:   {4, 1024}
//     losses, losses_grad      {batch_size, seq_len}                      one loss per token:    {4, 1024}
//     max_of_each_row          {batch_size, seq_len}                      row_max, kept for backward
//     sum_of_each_row          {batch_size, seq_len}                      row_sum, kept for backward
//
// Only the first vocab_size columns are words (50257). The other 47 are padding, so the head's matrix
// product works on a multiple of 128 columns; they are left out of the softmax and get gradient 0
// (note 6.3). Subtracting row_max changes nothing in prob (it cancels in the division) but keeps exp
// from overflowing: exp(1003) is infinity in a float, exp(1003 - 1003) is 1 (note 1.1).
//
// Backward, with losses_grad the gradient arriving for each row's loss (1 / 4096 in every row, for the
// mean over 4096 rows: note 1.3):
//
//     logits_grad = (prob - 1 at the target column, prob elsewhere) * losses_grad[row]
//                 = dℓ · (p_j - [j = t])                                            C2: B1..B7 folded into one line
//
// probs [0.7, 0.2, 0.1] with target 1 give [0.7, -0.8, 0.1] * losses_grad[row]: push the right word up,
// every other word down by as much as it got. The note's example, logits [1, 2, 3, -1] with target 1,
// gives probs [0.0889468, 0.2417825, 0.6572330, 0.0120376], loss 1.4197166 and
// logits_grad [0.0889468, -0.7582175, 0.6572330, 0.0120376] * losses_grad[row].
//
// The logits are 206M floats, 824 MB, so the time is GPU memory: the forward reads them once, the
// backward reads them once and writes logits_grad. The probs are never stored: the backward works each
// out again from logits, max_of_each_row and sum_of_each_row, as LayerNorm's backward does with mean and rstd
// (note 6.1).
// logits4 is one float4 of the logits: the 4 columns a thread has.
#ifndef CLASSIFIER_CUH
#define CLASSIFIER_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define CLASSIFIER_FORWARD_THREADS_PER_BLOCK 1024   // one block per row: about 12 of its 12,576 float4s each
#define CLASSIFIER_BACKWARD_THREADS_PER_BLOCK 256   // one float4 per thread

// ── forward and backward ──────────────────────────────────────────────────────
void classifier_forward(const Tensor& losses, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                        const Tensor& logits, const Tensor& targets, size_t vocab_size);
void classifier_backward(const Tensor& logits_grad, const Tensor& losses_grad, const Tensor& logits,
                         const Tensor& targets, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                         size_t vocab_size);

// ── for both ──────────────────────────────────────────────────────────────────
void classifier_check_shapes(const Tensor& losses, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                             const Tensor& logits, const Tensor& targets, size_t vocab_size);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel classifier_forward launches, declared here so it can call it; its code follows it.
__global__ void classifier_forward_kernel(float* losses, float* max_of_each_row, float* sum_of_each_row,
                                          const float4* logits, const int* targets, size_t rows, size_t vocab_size,
                                          size_t padded_vocab_size);

/// Writes each row's loss into losses, and its row_max and row_sum into max_of_each_row and sum_of_each_row for the
/// backward. The logits are left as they are. logits is {batch_size, seq_len, padded_vocab_size},
/// targets {batch_size, seq_len} ints, losses, max_of_each_row and sum_of_each_row {batch_size, seq_len}; the columns
/// from vocab_size on are padding. The targets are not checked: a target >= vocab_size reads a padding
/// column. One block of 1024 threads per row: 4096 blocks for batch_size = 4, seq_len = 1024.
inline void classifier_forward(const Tensor& losses, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                               const Tensor& logits, const Tensor& targets, size_t vocab_size) {
    classifier_check_shapes(losses, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    size_t rows = logits.size(0) * logits.size(1);
    classifier_forward_kernel<<<(int)rows, CLASSIFIER_FORWARD_THREADS_PER_BLOCK>>>(
        losses.data(), max_of_each_row.data(), sum_of_each_row.data(), (const float4*)logits.data(),
        targets.data_ptr<int>(), rows, vocab_size, logits.size(2));
    cudaCheck(cudaGetLastError());
}

/// Works out each row's row_max, row_sum and loss, one block per row. Block b takes rows b, b + blocks
/// launched, ... (with the usual launch, one row each). Its threads walk the row together, a float4 at a
/// time: thread t takes float4s t, t + 1024, t + 2048, ..., so a warp reads 32 x 16 = 512 neighbouring
/// bytes at once.
///
/// row_max and row_sum come from one read of the row: the online softmax. Each thread keeps the largest
/// logit it has seen and the sum of exp(logit - that largest). When a larger logit comes, the sum so far
/// was worked out against the old largest, so it is multiplied by exp(old - new) to put it against the new:
///
///     logits 1, 3, 2      seen 1:   max 1   sum e^0                          = 1
///                         seen 3:   max 3   sum 1 x e^(1-3) + e^0            = 1.1353353
///                         seen 2:   max 3   sum 1.1353353 + e^(2-3)          = 1.5032147
///                         the same as e^(1-3) + e^(3-3) + e^(2-3), worked out knowing the max first
///
/// Then the 1024 threads' (max, sum) parts are put together the same way: across each warp with
/// shuffles, then across the 32 warps in shared memory. Two reads, one for the max and one for the sum,
/// would read the 196 KB row twice. This is note 6.2 and its Figure 3: step 1 below is the update rule,
/// steps 2 and 3 the merge rule; the results are F1's m, F4's S and F5's loss.
__global__ void classifier_forward_kernel(float* losses, float* max_of_each_row, float* sum_of_each_row,
                                          const float4* logits, const int* targets, size_t rows, size_t vocab_size,
                                          size_t padded_vocab_size) {
    // warp_max and warp_sum of every warp of the block, slot w for warp w, so warp 0 can put them together:
    // shuffles only pass values inside one warp. 32 slots: a block has at most 1024 / 32 = 32 warps.
    __shared__ float max_of_each_warp[32], sum_of_each_warp[32];
    int thread_in_warp = threadIdx.x % 32, warp_in_block = threadIdx.x / 32, warps_in_block = blockDim.x / 32;
    size_t float4s_per_row = padded_vocab_size / 4;          // 50304 / 4 = 12,576
    size_t float4s_with_words = (vocab_size + 3) / 4;        // 50257 words: 12,565, the last part padding

    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float4* row_logits = logits + row * float4s_per_row;

        // 1. this thread's (max, sum), over its float4s: note 6.2's update rule. Padding columns count as
        //    -FLT_MAX: exp of them is 0 (note 6.3). Only float4s with at least one word, so the max of the 4
        //    is never -FLT_MAX itself.
        float thread_max = -FLT_MAX, thread_sum = 0.0f;
        for (size_t float4_in_row = threadIdx.x; float4_in_row < float4s_with_words; float4_in_row += blockDim.x) {
            float4 logits4 = row_logits[float4_in_row];
            size_t first_column = float4_in_row * 4;
            float logit0 = logits4.x;                                     // first_column is always a word here
            float logit1 = first_column + 1 < vocab_size ? logits4.y : -FLT_MAX;
            float logit2 = first_column + 2 < vocab_size ? logits4.z : -FLT_MAX;
            float logit3 = first_column + 3 < vocab_size ? logits4.w : -FLT_MAX;
            float new_max = fmaxf(thread_max, fmaxf(fmaxf(logit0, logit1), fmaxf(logit2, logit3)));   // m' = max(m, x)
            thread_sum = thread_sum * expf(thread_max - new_max) +        // s · exp(m - m'): the sum so far, against the new max
                         expf(logit0 - new_max) + expf(logit1 - new_max) + expf(logit2 - new_max) +
                         expf(logit3 - new_max);
            thread_max = new_max;
        }

        // 2. the warp's: every thread adds the part of the thread `offset` away, 16, 8, 4, 2, 1 (note 6.2's
        //    merge rule: m = max(m1, m2), s = s1 · exp(m1 - m) + s2 · exp(m2 - m))
        float warp_max = thread_max, warp_sum = thread_sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, warp_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, warp_sum, offset);
            float new_max = fmaxf(warp_max, other_max);      // each sum moved onto the larger max, then added
            warp_sum = warp_sum * expf(warp_max - new_max) + other_sum * expf(other_max - new_max);
            warp_max = new_max;
        }

        // 3. the row's: warp 0 puts together the warps' parts; its thread 0 writes the results
        if (thread_in_warp == 0) {
            max_of_each_warp[warp_in_block] = warp_max;
            sum_of_each_warp[warp_in_block] = warp_sum;
        }
        __syncthreads();                       // every warp's part is in shared memory
        if (warp_in_block == 0) {
            float row_max = thread_in_warp < warps_in_block ? max_of_each_warp[thread_in_warp] : -FLT_MAX;
            float row_sum = thread_in_warp < warps_in_block ? sum_of_each_warp[thread_in_warp] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_max = __shfl_xor_sync(0xffffffff, row_max, offset);
                float other_sum = __shfl_xor_sync(0xffffffff, row_sum, offset);
                float new_max = fmaxf(row_max, other_max);   // as in step 2; an empty slot (-FLT_MAX, 0) adds 0
                row_sum = row_sum * expf(row_max - new_max) + other_sum * expf(other_max - new_max);
                row_max = new_max;
            }
            if (thread_in_warp == 0) {
                float target_logit = ((const float*)row_logits)[targets[row]];
                max_of_each_row[row] = row_max;                            // F1: m
                sum_of_each_row[row] = row_sum;                            // F4: S
                losses[row] = logf(row_sum) - (target_logit - row_max);   // F5: ℓ = log S - a_t, -log(prob of the target)
            }
        }
        __syncthreads();                       // before the next row writes the two arrays again
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernel classifier_backward launches, declared here so it can call it; its code follows it.
__global__ void classifier_backward_kernel(float4* logits_grad, const float* losses_grad, const float4* logits,
                                           const int* targets, const float* max_of_each_row,
                                           const float* sum_of_each_row, size_t rows, size_t vocab_size,
                                           size_t padded_vocab_size);

/// Writes the gradient of the logits into logits_grad, from losses_grad and what the forward read and
/// kept: logits, targets, max_of_each_row and sum_of_each_row. logits_grad is written over: the logits are read by
/// the classifier alone. It may be logits itself (each value is read before it is written, by the same
/// thread), so the trainer can turn the 824 MB of logits into their gradient without a second 824 MB.
/// losses_grad[row] is d(loss)/d(losses[row]): 1 / rows in every row for the mean over the rows (note 1.3),
/// divided again by the micro-batches of a step. The shapes are the forward's: logits_grad as logits,
/// losses_grad as losses. One float4 per thread: 51,511,296 float4s, 201,216 blocks of 256 threads.
/// Written over, and may be in place: note 6.4.
inline void classifier_backward(const Tensor& logits_grad, const Tensor& losses_grad, const Tensor& logits,
                                const Tensor& targets, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                                size_t vocab_size) {
    classifier_check_shapes(losses_grad, max_of_each_row, sum_of_each_row, logits, targets, vocab_size);
    if (logits_grad.shape() != logits.shape() || (size_t)logits_grad.data() % 16 != 0) {
        fprintf(stderr, "error: the classifier needs logits_grad the same shape as logits, starting on 16 bytes\n");
        exit(EXIT_FAILURE);
    }
    size_t rows = logits.size(0) * logits.size(1);
    size_t total_float4s = logits.numel() / 4;
    int blocks = (int)((total_float4s + CLASSIFIER_BACKWARD_THREADS_PER_BLOCK - 1) /
                       CLASSIFIER_BACKWARD_THREADS_PER_BLOCK);    // rounded up: a part-full last block
    classifier_backward_kernel<<<blocks, CLASSIFIER_BACKWARD_THREADS_PER_BLOCK>>>(
        (float4*)logits_grad.data(), losses_grad.data(), (const float4*)logits.data(), targets.data_ptr<int>(),
        max_of_each_row.data(), sum_of_each_row.data(), rows, vocab_size, logits.size(2));
    cudaCheck(cudaGetLastError());
}

/// Writes logits_grad, one float4 at a time: thread n takes float4s n, n + every thread launched, ...
/// Each value needs only its own logit and four numbers of its row: max_of_each_row, sum_of_each_row, the target and
/// losses_grad. A row is 12,576 float4s, so the threads of a block all read the same row's four, which
/// come from the cache after the first. The loop body is the note's C2, written out as in its 6.4; the
/// seven steps B1..B7 of its Section 3 are never run one by one.
__global__ void classifier_backward_kernel(float4* logits_grad, const float* losses_grad, const float4* logits,
                                           const int* targets, const float* max_of_each_row,
                                           const float* sum_of_each_row, size_t rows, size_t vocab_size,
                                           size_t padded_vocab_size) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t float4s_per_row = padded_vocab_size / 4;
    size_t total_float4s = rows * float4s_per_row;
    size_t threads_launched = (size_t)gridDim.x * blockDim.x;
    for (size_t float4_index = thread_number; float4_index < total_float4s; float4_index += threads_launched) {
        size_t row = float4_index / float4s_per_row;
        size_t first_column = (float4_index % float4s_per_row) * 4;
        size_t target = (size_t)targets[row];
        float row_max = max_of_each_row[row], one_over_row_sum = 1.0f / sum_of_each_row[row];
        float row_loss_grad = losses_grad[row];

        float4 logits4 = logits[float4_index];
        float logit_values[4] = {logits4.x, logits4.y, logits4.z, logits4.w};
        float grad_values[4];
        for (int value_in_float4 = 0; value_in_float4 < 4; value_in_float4++) {
            size_t column = first_column + value_in_float4;
            float prob = expf(logit_values[value_in_float4] - row_max) * one_over_row_sum;   // C1: p_j = exp(z_j - m) / S
            float one_at_target = column == target ? 1.0f : 0.0f;                            // [j = t]: B4's part
            grad_values[value_in_float4] =                                                    // C2: dz_j = dℓ · (p_j - [j = t]);
                column < vocab_size ? (prob - one_at_target) * row_loss_grad : 0.0f;          // padding gets 0 (note 6.3)
        }
        logits_grad[float4_index] = make_float4(grad_values[0], grad_values[1], grad_values[2], grad_values[3]);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message if the tensors do not fit together, before a kernel reads past one: logits
/// {batch_size, seq_len, padded_vocab_size}; targets, losses, max_of_each_row and sum_of_each_row {batch_size,
/// seq_len}, targets ints; 0 < vocab_size <= padded_vocab_size. Also padded_vocab_size a multiple of 4
/// and logits starting on 16 bytes, for the float4 loads: off 16 bytes, the GPU stops with "misaligned
/// address" at some later call, far from the cause. Backward passes losses_grad in losses' place.
inline void classifier_check_shapes(const Tensor& losses, const Tensor& max_of_each_row, const Tensor& sum_of_each_row,
                                    const Tensor& logits, const Tensor& targets, size_t vocab_size) {
    bool fits = logits.dim() == 3 && targets.shape() == Shape({logits.size(0), logits.size(1)}) &&
                targets.dtype() == DType::Int32 && losses.shape() == targets.shape() &&
                max_of_each_row.shape() == targets.shape() && sum_of_each_row.shape() == targets.shape() &&
                vocab_size > 0 &&
                vocab_size <= logits.size(2);
    if (!fits) {
        fprintf(stderr, "error: the classifier needs logits {batch_size, seq_len, padded_vocab_size}; targets (ints), "
                        "losses, max_of_each_row and sum_of_each_row {batch_size, seq_len}; and 0 < vocab_size <= "
                        "padded_vocab_size\n");
        exit(EXIT_FAILURE);
    }
    if (logits.size(2) % 4 != 0 || (size_t)logits.data() % 16 != 0) {
        fprintf(stderr, "error: the classifier reads 4 floats at a time (float4), so padded_vocab_size must be a "
                        "multiple of 4 and logits must start on 16 bytes; got %zu, %p\n",
                logits.size(2), (void*)logits.data());
        exit(EXIT_FAILURE);
    }
}

#endif // CLASSIFIER_CUH
