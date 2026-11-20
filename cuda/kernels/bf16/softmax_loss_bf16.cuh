// softmax_loss_bf16.cuh — GPT-2's softmax, cross-entropy loss and its gradient on bfloat16 logits, in one pass over
// each row.
//
//     tensor                                     type in memory   shape
//     logits, logits_grad                        bfloat16         {batch_size, seq_len, padded_vocab_size}
//     targets                                    int              {batch_size, seq_len}
//     losses, max_of_each_row, sum_of_each_row   float            {batch_size, seq_len}
#ifndef SOFTMAX_LOSS_BF16_CUH
#define SOFTMAX_LOSS_BF16_CUH

#include "../cuda_common.h"             // cudaCheck, check_threads_per_block, warp and block defines
#include "../../common/tensor.cuh"      // read8_as_float, write8_from_float, VALUES_PER_BF16_LOAD

#define SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK (32 * HARDWARE_THREADS_PER_WARP)   // the default block size: 32 warps (tune_softmax_loss_bf16 sweeps it)
#define SOFTMAX_LOSS_BF16_VALUES_PER_LOAD VALUES_PER_BF16_LOAD                    // bfloat16s in one 16-byte load: 8
#define SOFTMAX_LOSS_BF16_LOG2_E 1.4426950408889634f                              // log2(e): exp(x) = 2^(x log2 e)
#define SOFTMAX_LOSS_BF16_LN_2 0.6931471805599453f                                // ln(2): 2^x = e^(x ln 2)

// ── forward and backward ──────────────────────────────────────────────────────
void softmax_loss_bf16_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                               const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor,
                               const Tensor& targets_tensor, size_t vocab_size_val,
                               int threads_per_block_val = SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, float softcap_val = 0.0f);
void softmax_loss_bf16_backward(const Tensor& logits_grad_tensor, const Tensor& logits_tensor, const Tensor& targets_tensor,
                                const Tensor& max_of_each_row_tensor, const Tensor& sum_of_each_row_tensor,
                                size_t vocab_size_val, float loss_grad_val,
                                int threads_per_block_val = SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK, float softcap_val = 0.0f);
void softmax_loss_bf16_forward_backward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                        const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                                        const Tensor& logits_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                                        float loss_grad_val, int threads_per_block_val = SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK,
                                        float softcap_val = 0.0f);

// ── for all three ─────────────────────────────────────────────────────────────
template <bool FORWARD, bool BACKWARD>
__global__ void softmax_loss_bf16_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                         nv_bfloat16* logits_grad_ptr, const nv_bfloat16* logits_ptr, const int* targets_ptr,
                                         size_t vocab_size_val, size_t padded_vocab_size_val, float loss_grad_val,
                                         float softcap_val);
template <bool FORWARD, bool BACKWARD>
void softmax_loss_bf16_launch(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                              const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                              const Tensor& logits_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                              float loss_grad_val, int threads_per_block_val, float softcap_val);
void softmax_loss_bf16_check_shapes(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                    const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                                    const Tensor& logits_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                                    bool backward_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward and backward ──────────────────────────────────────────────────────

/// Writes each row's loss, row_max (times log2 e) and row_sum.
///
///   losses_tensor            one loss per row
///   max_of_each_row_tensor   row_max · log2 e, for the backward
///   sum_of_each_row_tensor   row_sum, for the backward
///   logits_tensor            the scores, left as they are
///   targets_tensor           the right next word of each row
///   vocab_size_val           the real words: 50257
///   threads_per_block_val    the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK; not given:
///                            SOFTMAX_LOSS_BF16_THREADS_PER_BLOCK. Another size adds each row in another order, so
///                            the last bits can change.
///   softcap_val              c > 0: each logit capped to c · tanh(logit / c) before the softmax; 0 (not given): none
inline void softmax_loss_bf16_forward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                      const Tensor& sum_of_each_row_tensor, const Tensor& logits_tensor,
                                      const Tensor& targets_tensor, size_t vocab_size_val, int threads_per_block_val,
                                      float softcap_val) {
    // Pass 1 alone, for evaluation, or for a backward later. The targets are not checked: a target >= vocab_size
    // reads a padding column.
    softmax_loss_bf16_launch<true, false>(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, Tensor(),
                                          logits_tensor, targets_tensor, vocab_size_val, 0.0f, threads_per_block_val,
                                          softcap_val);
}

/// Writes logits_grad (written over, not added) from the forward's row_max and row_sum: pass 2 alone.
///
///   logits_grad_tensor                               the result; may be logits itself
///   logits_tensor, targets_tensor                    the forward's
///   max_of_each_row_tensor, sum_of_each_row_tensor   the forward's results
///   vocab_size_val                                   the real words: 50257
///   loss_grad_val                                    d(loss) / d(each row's loss): 1 / rows for the mean
///   threads_per_block_val                            as in softmax_loss_bf16_forward; here any size gives the same bits
///   softcap_val                                      the forward's
inline void softmax_loss_bf16_backward(const Tensor& logits_grad_tensor, const Tensor& logits_tensor,
                                       const Tensor& targets_tensor, const Tensor& max_of_each_row_tensor,
                                       const Tensor& sum_of_each_row_tensor, size_t vocab_size_val, float loss_grad_val,
                                       int threads_per_block_val, float softcap_val) {
    softmax_loss_bf16_launch<false, true>(Tensor(), max_of_each_row_tensor, sum_of_each_row_tensor, logits_grad_tensor,
                                          logits_tensor, targets_tensor, vocab_size_val, loss_grad_val, threads_per_block_val,
                                          softcap_val);
}

/// Writes each row's loss, row_max and row_sum, and logits_grad, in one kernel.
///
///   losses_tensor ... targets_tensor, vocab_size_val, loss_grad_val   as in the two above
///   threads_per_block_val, softcap_val                                 as in softmax_loss_bf16_forward
inline void softmax_loss_bf16_forward_backward(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                               const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                                               const Tensor& logits_tensor, const Tensor& targets_tensor,
                                               size_t vocab_size_val, float loss_grad_val, int threads_per_block_val,
                                               float softcap_val) {
    // Pass 1 and pass 2 together, so the logits come from memory once. The same bits as softmax_loss_bf16_forward
    // then softmax_loss_bf16_backward at the same block size.
    softmax_loss_bf16_launch<true, true>(losses_tensor, max_of_each_row_tensor, sum_of_each_row_tensor, logits_grad_tensor,
                                         logits_tensor, targets_tensor, vocab_size_val, loss_grad_val, threads_per_block_val,
                                         softcap_val);
}

// ── for all three ─────────────────────────────────────────────────────────────

/// Gives 2^x by the special function unit's one instruction.
///
///   x_val   the power
__device__ __forceinline__ float softmax_loss_bf16_exp2(float x_val) {
    // (ex2.approx.ftz: about 2 ulp, under 2^-126 to 0; 2^-inf is 0).
    float result_val;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result_val) : "f"(x_val));
    return result_val;
}

/// Gives a sum of 2^(value - max) moved onto a new, larger max.
///
///   sum_val, max_val   the part: its sum, and the max it is taken from
///   new_max_val        the max it moves onto
__device__ __forceinline__ float softmax_loss_bf16_moved(float sum_val, float max_val, float new_max_val) {
    // Sum · 2^(max - new_max); an empty part (max -inf) gives 0, not -inf - -inf.
    return max_val == -INFINITY ? 0.0f : sum_val * softmax_loss_bf16_exp2(max_val - new_max_val);
}

/// Gives a logit after the softcap, c · tanh(logit / c); no softcap (c 0): the logit itself.
///
///   logit_val                            the raw logit
///   softcap_val, one_over_softcap_val    c and 1 / c; 0 and 0: none
__device__ __forceinline__ float softmax_loss_bf16_capped(float logit_val, float softcap_val, float one_over_softcap_val) {
    // Gives a logit after the softcap, c · tanh(logit / c), by the special function unit's one tanh.approx.f32 (about
    // 2^-11 relative); no softcap (c 0): the logit itself, untouched.
    if (softcap_val <= 0.0f) return logit_val;
    float tanh_val;
    asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_val) : "f"(logit_val * one_over_softcap_val));
    return softcap_val * tanh_val;
}

/// One row per block: pass 1 (max, sum, loss) for FORWARD, pass 2 (each value's gradient) for BACKWARD.
///
///   losses_ptr, max_of_each_row_ptr, sum_of_each_row_ptr   one per row: written by FORWARD, read by BACKWARD alone
///   logits_grad_ptr                                        BACKWARD: the row's gradient written; may be logits_ptr
///   logits_ptr, targets_ptr                                the scores, the right next words
///   vocab_size_val, padded_vocab_size_val                  real words, and the row length (a multiple of 8)
///   loss_grad_val                                          BACKWARD: d(loss) / d(each row's loss)
///   softcap_val                                            c > 0: every logit read is capped first; 0: none
template <bool FORWARD, bool BACKWARD>
__global__ void softmax_loss_bf16_kernel(float* losses_ptr, float* max_of_each_row_ptr, float* sum_of_each_row_ptr,
                                         nv_bfloat16* logits_grad_ptr, const nv_bfloat16* logits_ptr, const int* targets_ptr,
                                         size_t vocab_size_val, size_t padded_vocab_size_val, float loss_grad_val,
                                         float softcap_val) {
    // One row per block. FORWARD: pass 1, the row's max and sum in powers of 2 (each thread over its 16-byte loads,
    // then the warp, then the block), and the loss; BACKWARD: pass 2, each value's gradient, 8 a thread at a time.
    // Both: pass 2 takes the max and sum pass 1 just worked out; BACKWARD alone: the ones in memory.
    // one slot per warp; warp 0's HARDWARE_THREADS_PER_WARP lanes must cover them all; then the row's two numbers
    static_assert(HARDWARE_MAX_WARPS_PER_BLOCK <= HARDWARE_THREADS_PER_WARP, "warp 0 must hold one (max, sum) per warp");
    __shared__ float max_of_each_warp_list[HARDWARE_MAX_WARPS_PER_BLOCK];
    __shared__ float sum_of_each_warp_list[HARDWARE_MAX_WARPS_PER_BLOCK];
    __shared__ float row_max_and_sum_list[2];
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t row_index_val = blockIdx.x;
    const nv_bfloat16* row_logits_ptr = logits_ptr + row_index_val * padded_vocab_size_val;
    float row_max_val = 0.0f;                                   // row_max · log2 e
    float row_sum_val = 0.0f;
    float one_over_softcap_val = softcap_val > 0.0f ? 1.0f / softcap_val : 0.0f;

    if (FORWARD) {
        // 1. this thread's (max, sum) over its loads, in powers of 2; padding counts as -inf, 2^-inf 0. Every load
        //    it takes starts on a word, so its max is finite once it has one
        size_t loads_with_words_val = CEIL_DIV(vocab_size_val, (size_t)SOFTMAX_LOSS_BF16_VALUES_PER_LOAD);
        float thread_max_val = -INFINITY;
        float thread_sum_val = 0.0f;
        for (size_t load_index_val = threadIdx.x; load_index_val < loads_with_words_val; load_index_val += blockDim.x) {
            float values_list[SOFTMAX_LOSS_BF16_VALUES_PER_LOAD];
            size_t first_column_index_val = load_index_val * SOFTMAX_LOSS_BF16_VALUES_PER_LOAD;
            read8_as_float(values_list, row_logits_ptr + first_column_index_val);
            float load_max_val = -INFINITY;
#pragma unroll
            for (int value_index_val = 0; value_index_val < SOFTMAX_LOSS_BF16_VALUES_PER_LOAD; value_index_val++) {
                values_list[value_index_val] = first_column_index_val + value_index_val < vocab_size_val
                                                   ? softmax_loss_bf16_capped(values_list[value_index_val], softcap_val,
                                                                              one_over_softcap_val) * SOFTMAX_LOSS_BF16_LOG2_E
                                                   : -INFINITY;
                load_max_val = fmaxf(load_max_val, values_list[value_index_val]);
            }
            float new_max_val = fmaxf(thread_max_val, load_max_val);
            float new_sum_val = softmax_loss_bf16_moved(thread_sum_val, thread_max_val, new_max_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < SOFTMAX_LOSS_BF16_VALUES_PER_LOAD; value_index_val++)
                new_sum_val += softmax_loss_bf16_exp2(values_list[value_index_val] - new_max_val);
            thread_max_val = new_max_val;
            thread_sum_val = new_sum_val;
        }

        // 2. the warp's: log2(HARDWARE_THREADS_PER_WARP) swaps; m = max(m1, m2), s = s1 · 2^(m1 - m) + s2 · 2^(m2 - m)
        for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
            float other_max_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, thread_max_val, offset_val);
            float other_sum_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, thread_sum_val, offset_val);
            float new_max_val = fmaxf(thread_max_val, other_max_val);
            thread_sum_val = softmax_loss_bf16_moved(thread_sum_val, thread_max_val, new_max_val) +
                             softmax_loss_bf16_moved(other_sum_val, other_max_val, new_max_val);
            thread_max_val = new_max_val;
        }

        // 3. the row's: warp 0 puts the warps' parts together; its lane 0 writes the results, and leaves the max
        //    and sum in shared memory for pass 2
        if (lane_index_val == 0) {
            max_of_each_warp_list[warp_index_val] = thread_max_val;
            sum_of_each_warp_list[warp_index_val] = thread_sum_val;
        }
        __syncthreads();
        if (warp_index_val == 0) {
            float warp_max_val = lane_index_val < warps_in_block_val ? max_of_each_warp_list[lane_index_val] : -INFINITY;
            float warp_sum_val = lane_index_val < warps_in_block_val ? sum_of_each_warp_list[lane_index_val] : 0.0f;
            for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2) {
                float other_max_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, warp_max_val, offset_val);
                float other_sum_val = __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, warp_sum_val, offset_val);
                float new_max_val = fmaxf(warp_max_val, other_max_val);
                warp_sum_val = softmax_loss_bf16_moved(warp_sum_val, warp_max_val, new_max_val) +
                               softmax_loss_bf16_moved(other_sum_val, other_max_val, new_max_val);
                warp_max_val = new_max_val;
            }
            // 4. lane 0: loss = log(row_sum) - (logit[target] - row_max), row_max back in natural units
            if (lane_index_val == 0) {
                float target_logit_val = softmax_loss_bf16_capped(to_float(row_logits_ptr[targets_ptr[row_index_val]]),
                                                                  softcap_val, one_over_softcap_val);
                losses_ptr[row_index_val] = logf(warp_sum_val) - (target_logit_val - warp_max_val * SOFTMAX_LOSS_BF16_LN_2);
                max_of_each_row_ptr[row_index_val] = warp_max_val;
                sum_of_each_row_ptr[row_index_val] = warp_sum_val;
                row_max_and_sum_list[0] = warp_max_val;
                row_max_and_sum_list[1] = warp_sum_val;
            }
        }
        if (BACKWARD) {
            __syncthreads();                                    // the row's max and sum in, for every thread
            row_max_val = row_max_and_sum_list[0];
            row_sum_val = row_max_and_sum_list[1];
        }
    } else {
        // 1. backward alone: the forward's max and sum
        row_max_val = max_of_each_row_ptr[row_index_val];
        row_sum_val = sum_of_each_row_ptr[row_index_val];
    }

    if (BACKWARD) {
        // 5. each value: (prob - 1 at the target) · loss_grad, padding 0, with a softcap times 1 - (capped / c)²; 8 a
        //    thread, read then written by the same thread (so logits_grad may be logits)
        nv_bfloat16* row_grad_ptr = logits_grad_ptr + row_index_val * padded_vocab_size_val;
        float one_over_sum_val = 1.0f / row_sum_val;
        size_t target_val = (size_t)targets_ptr[row_index_val];
        size_t row_loads_val = padded_vocab_size_val / SOFTMAX_LOSS_BF16_VALUES_PER_LOAD;
        for (size_t load_index_val = threadIdx.x; load_index_val < row_loads_val; load_index_val += blockDim.x) {
            float values_list[SOFTMAX_LOSS_BF16_VALUES_PER_LOAD];
            size_t first_column_index_val = load_index_val * SOFTMAX_LOSS_BF16_VALUES_PER_LOAD;
            read8_as_float(values_list, row_logits_ptr + first_column_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < SOFTMAX_LOSS_BF16_VALUES_PER_LOAD; value_index_val++) {
                size_t column_index_val = first_column_index_val + value_index_val;
                float capped_val = softmax_loss_bf16_capped(values_list[value_index_val], softcap_val, one_over_softcap_val);
                float prob_val = softmax_loss_bf16_exp2(fmaf(capped_val, SOFTMAX_LOSS_BF16_LOG2_E, -row_max_val)) * one_over_sum_val;
                float one_at_target_val = column_index_val == target_val ? 1.0f : 0.0f;
                float grad_val = (prob_val - one_at_target_val) * loss_grad_val;
                if (softcap_val > 0.0f) {                       // back through the cap: tanh' = 1 - tanh²
                    float tanh_val = capped_val * one_over_softcap_val;
                    grad_val *= 1.0f - tanh_val * tanh_val;
                }
                values_list[value_index_val] = column_index_val < vocab_size_val ? grad_val : 0.0f;
            }
            write8_from_float(row_grad_ptr + first_column_index_val, values_list);
        }
    }
}

/// Launches softmax_loss_bf16_kernel for FORWARD and BACKWARD, one block per row.
///
///   losses_tensor ... loss_grad_val   as in softmax_loss_bf16_forward_backward; the parts a launch does not use
///                                     may be Tensor()
///   threads_per_block_val             the block size
///   softcap_val                       c > 0: the logits capped as they are read; 0: none
template <bool FORWARD, bool BACKWARD>
inline void softmax_loss_bf16_launch(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                     const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                                     const Tensor& logits_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                                     float loss_grad_val, int threads_per_block_val, float softcap_val) {
    // Checks the block size and the tensors first.
    // 1. check the launch and the tensors; any of them wrong: stop with an error
    check_threads_per_block(threads_per_block_val, "softmax_loss_bf16");
    softmax_loss_bf16_check_shapes(FORWARD ? losses_tensor : max_of_each_row_tensor, max_of_each_row_tensor,
                                   sum_of_each_row_tensor, logits_grad_tensor, logits_tensor, targets_tensor, vocab_size_val,
                                   BACKWARD);
    // 2. launch: one block per row
    size_t rows_val = logits_tensor.size(0) * logits_tensor.size(1);
    softmax_loss_bf16_kernel<FORWARD, BACKWARD><<<(unsigned)rows_val, threads_per_block_val>>>(
        FORWARD ? losses_tensor.data() : nullptr, max_of_each_row_tensor.data(), sum_of_each_row_tensor.data(),
        BACKWARD ? logits_grad_tensor.data_ptr<nv_bfloat16>() : nullptr, logits_tensor.data_ptr<nv_bfloat16>(),
        targets_tensor.data_ptr<int>(), vocab_size_val, logits_tensor.size(2), loss_grad_val, softcap_val);
    cudaCheck(cudaGetLastError());
}

/// Stops with a message unless the tensors fit.
///
///   losses_tensor                                    the forward's losses (the backward alone: max_of_each_row again)
///   max_of_each_row_tensor, sum_of_each_row_tensor   row_max and row_sum
///   logits_grad_tensor                               checked when backward_val
///   logits_tensor, targets_tensor                    logits and the right next words
///   vocab_size_val                                   the real words
///   backward_val                                     the launch writes logits_grad
inline void softmax_loss_bf16_check_shapes(const Tensor& losses_tensor, const Tensor& max_of_each_row_tensor,
                                           const Tensor& sum_of_each_row_tensor, const Tensor& logits_grad_tensor,
                                           const Tensor& logits_tensor, const Tensor& targets_tensor, size_t vocab_size_val,
                                           bool backward_val) {
    // The shapes at the top of the file, 0 < vocab_size <= padded_vocab_size, logits (and logits_grad) bfloat16
    // with rows of whole 16-byte loads and starting on one, the per-row tensors float, targets ints.
    size_t load_bytes_val = SOFTMAX_LOSS_BF16_VALUES_PER_LOAD * sizeof(nv_bfloat16);
    bool fits_val = logits_tensor.dim() == 3 && logits_tensor.dtype() == DType::BFloat16 &&
                    targets_tensor.shape() == Shape({logits_tensor.size(0), logits_tensor.size(1)}) &&
                    targets_tensor.dtype() == DType::Int32 && vocab_size_val > 0 && vocab_size_val <= logits_tensor.size(2) &&
                    logits_tensor.size(2) % SOFTMAX_LOSS_BF16_VALUES_PER_LOAD == 0 &&
                    (size_t)logits_tensor.data_ptr() % load_bytes_val == 0;
    for (const Tensor* tensor_ptr : {&losses_tensor, &max_of_each_row_tensor, &sum_of_each_row_tensor})
        fits_val = fits_val && tensor_ptr->shape() == targets_tensor.shape() && tensor_ptr->dtype() == DType::Float32;
    if (backward_val)
        fits_val = fits_val && logits_grad_tensor.shape() == logits_tensor.shape() &&
                   logits_grad_tensor.dtype() == DType::BFloat16 && (size_t)logits_grad_tensor.data_ptr() % load_bytes_val == 0;
    if (!fits_val) {
        fprintf(stderr, "error: softmax_loss_bf16 needs bfloat16 logits (and logits_grad) {batch_size, seq_len, "
                        "padded_vocab_size} with padded_vocab_size a multiple of %d, starting on %zu bytes; targets "
                        "(ints), losses, max_of_each_row and sum_of_each_row (float) {batch_size, seq_len}; and 0 < "
                        "vocab_size <= padded_vocab_size\n",
                SOFTMAX_LOSS_BF16_VALUES_PER_LOAD, load_bytes_val);
        exit(EXIT_FAILURE);
    }
}

#endif // SOFTMAX_LOSS_BF16_CUH
