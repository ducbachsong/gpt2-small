// layer_norm.cuh — LayerNorm: each row of embed_dim numbers (one token) is moved to mean 0 and
// scaled to standard deviation 1, then every column gets its own scale and shift:
//
//     mean     = sum(x) / embed_dim
//     variance = sum((x - mean)²) / embed_dim
//     rstd     = 1 / sqrt(variance + 1e-5)   1 / the standard deviation; the 1e-5 keeps a flat row from / 0
//     norm     = (x - mean) * rstd           the row with mean 0 and standard deviation 1
//     output   = norm * weight + bias        weight and bias: one per column, learned
//
//     input, output   {batch_size, seq_len, embed_dim}   one row per token:          {4, 1024, 768}
//     weight, bias    {embed_dim}                        one per column:             {768}
//     mean, rstd      {batch_size, seq_len}              one per row, for backward:  {4, 1024}
//
// GPT-2 runs one before the attention and one before the MLP in each of its 12 blocks, and one
// after the last block: 25 in all. Each row is normalized on its own; rows never mix.
//
// Backward, with output_grad the gradient arriving from the layer above (means over the row):
//
//     bias_grad   += sum over all rows of output_grad
//     weight_grad += sum over all rows of output_grad * norm
//     norm_grad    = output_grad * weight
//     input_grad  += rstd * (norm_grad - mean(norm_grad) - norm * mean(norm_grad * norm))
//
// The steps in the comments (F1 to F6, B1 to B9, C7) are those of the post that works all this out,
// ducbachsong.github.io/posts/layer-norm-backward. The same values, by their names here and there:
//
//     in the code      in the post     in the code           in the post
//     input            x               output_grad           dy
//     mean, row_mean   μ               norm_grad             dx̂
//     centered         d               norm_grad_mean        m1
//     row_variance     v               norm_grad_norm_mean   m2
//     rstd, row_rstd   r               bias_grad             dβ
//     norm             x̂               weight_grad           dγ
//     output           y               input_grad            dx
//     weight, bias     γ, β
//
// A name ending in 4 (input4, weight4, ...) is one float4 of that tensor: the 4 columns a thread has.
#ifndef LAYER_NORM_CUH
#define LAYER_NORM_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define LAYER_NORM_THREADS_PER_BLOCK 256            // forward: 8 warps, one row each
#define LAYER_NORM_BACKWARD_THREADS_PER_BLOCK 512   // backward: 16 warps, two rows each
#define LAYER_NORM_EPS 1e-5f                        // added to the variance, as GPT-2 and PyTorch's default do

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_forward(const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
                        const Tensor& weight, const Tensor& bias);
void layer_norm_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                         const Tensor& output_grad, const Tensor& input, const Tensor& weight, const Tensor& mean,
                         const Tensor& rstd);

// ── for both ──────────────────────────────────────────────────────────────────
void layer_norm_check_shapes(const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
                             const Tensor& weight, const Tensor& bias);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel layer_norm_forward launches, declared here so it can call it; its code follows it.
__global__ void layer_norm_forward_kernel(float4* output, float* mean, float* rstd, const float4* input,
                                          const float4* weight, const float4* bias, size_t rows, size_t embed_dim);

/// Writes output = LayerNorm(input) * weight + bias for every row, and each row's mean and rstd for the
/// backward. output and input are {batch_size, seq_len, embed_dim}, mean and rstd {batch_size, seq_len},
/// weight and bias {embed_dim}. One warp per row: with batch_size = 4, seq_len = 1024 that is 4096 rows,
/// 4096 / 8 = 512 blocks of 256 threads.
inline void layer_norm_forward(const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
                               const Tensor& weight, const Tensor& bias) {
    layer_norm_check_shapes(output, mean, rstd, input, weight, bias);
    size_t rows = input.size(0) * input.size(1), embed_dim = input.size(2);
    size_t warps_per_block = LAYER_NORM_THREADS_PER_BLOCK / 32;
    int blocks = (int)((rows + warps_per_block - 1) / warps_per_block);  // rounded up: a part-full last block
    layer_norm_forward_kernel<<<blocks, LAYER_NORM_THREADS_PER_BLOCK>>>(
        (float4*)output.data(), mean.data(), rstd.data(), (const float4*)input.data(), (const float4*)weight.data(),
        (const float4*)bias.data(), rows, embed_dim);
    cudaCheck(cudaGetLastError());
}

/// Normalizes the rows, one warp per row. Warp w of all launched takes rows w, w + warps launched, ...
/// (with the usual launch, one row each). The 32 threads of the warp walk the row together, a float4
/// at a time: thread_in_warp t takes float4s t, t + 32, t + 64, ... A row of 768 is 192 float4s,
/// so each thread takes 6, and a warp reads 32 x 16 = 512 neighbouring bytes per step.
///
/// Three passes over the row: 1. the mean, 2. the variance around it, 3. the output. Each sum is first
/// added up per thread (its 6 float4s), then across the warp with shuffles: every thread adds the value
/// of the thread `offset` away, 16, 8, 4, 2, 1, and then every thread holds the row's sum. The variance
/// is the mean of (x - mean)², not mean(x²) - mean², which loses digits when the mean is large. Passes 2
/// and 3 read the row again; the warp has just read those 3 KB, so most likely they come from L1.
__global__ void layer_norm_forward_kernel(float4* output, float* mean, float* rstd, const float4* input,
                                          const float4* weight, const float4* bias, size_t rows, size_t embed_dim) {
    int thread_in_warp = threadIdx.x % 32;
    size_t warp_number = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;   // which warp of all launched
    size_t warps_launched = (size_t)gridDim.x * blockDim.x / 32;
    size_t float4s_embed_dim = embed_dim / 4;                                    // 4 floats per float4: 768 / 4 = 192
    for (size_t row = warp_number; row < rows; row += warps_launched) {
        const float4* input_row = input + row * float4s_embed_dim;
        float4* output_row = output + row * float4s_embed_dim;

        // 1. the mean, F1
        float input_sum = 0.0f;
        for (size_t i = thread_in_warp; i < float4s_embed_dim; i += 32) {
            float4 input4 = input_row[i];
            input_sum += input4.x + input4.y + input4.z + input4.w;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            input_sum += __shfl_xor_sync(0xffffffff, input_sum, offset);
        }
        float row_mean = input_sum / (float)embed_dim;

        // 2. centered (F2), the variance (F3), and from it rstd (F4)
        float squares_sum = 0.0f;
        for (size_t i = thread_in_warp; i < float4s_embed_dim; i += 32) {
            float4 input4 = input_row[i];
            float4 centered = make_float4(input4.x - row_mean, input4.y - row_mean, input4.z - row_mean,
                                          input4.w - row_mean);
            squares_sum += centered.x * centered.x + centered.y * centered.y + centered.z * centered.z +
                           centered.w * centered.w;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            squares_sum += __shfl_xor_sync(0xffffffff, squares_sum, offset);
        }
        float row_variance = squares_sum / (float)embed_dim;
        float row_rstd = rsqrtf(row_variance + LAYER_NORM_EPS);                    // 1 / sqrt(...) in one instruction

        // 3. norm (F5), then the output with each column's weight and bias (F6)
        for (size_t i = thread_in_warp; i < float4s_embed_dim; i += 32) {
            float4 input4 = input_row[i], weight4 = weight[i], bias4 = bias[i];
            float4 norm = make_float4((input4.x - row_mean) * row_rstd, (input4.y - row_mean) * row_rstd,
                                      (input4.z - row_mean) * row_rstd, (input4.w - row_mean) * row_rstd);
            output_row[i] = make_float4(norm.x * weight4.x + bias4.x, norm.y * weight4.y + bias4.y,
                                        norm.z * weight4.z + bias4.z, norm.w * weight4.w + bias4.w);
        }
        if (thread_in_warp == 0) {
            mean[row] = row_mean;
            rstd[row] = row_rstd;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernel layer_norm_backward launches, declared here so it can call it; its code follows it.
__global__ void layer_norm_backward_kernel(float4* input_grad, float* weight_grad, float* bias_grad,
                                           const float4* output_grad, const float4* input, const float4* weight,
                                           const float* mean, const float* rstd, size_t rows, size_t embed_dim);

/// Adds the gradients of input, weight and bias onto what is there, from output_grad and what the
/// forward kept: input, weight, mean and rstd. It adds, so weight_grad and bias_grad must be 0 before
/// the first backward of a step; input_grad adds because in the trainer it is the residual stream's
/// gradient, which also gets the gradient of the layer it skips. The shapes are the forward's: input_grad
/// and output_grad as input, weight_grad and bias_grad as weight.
///
/// 16 warps per block, two rows per warp: with 4096 rows, 4096 / 32 = 128 blocks of 512 threads.
inline void layer_norm_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                                const Tensor& output_grad, const Tensor& input, const Tensor& weight,
                                const Tensor& mean, const Tensor& rstd) {
    layer_norm_check_shapes(output_grad, mean, rstd, input, weight, bias_grad);
    layer_norm_check_shapes(input_grad, mean, rstd, input, weight_grad, bias_grad);
    size_t rows = input.size(0) * input.size(1), embed_dim = input.size(2);
    size_t rows_per_block = (LAYER_NORM_BACKWARD_THREADS_PER_BLOCK / 32) * 2;
    int blocks = (int)((rows + rows_per_block - 1) / rows_per_block);   // rounded up: a part-full last block
    size_t shared_bytes = 2 * embed_dim * sizeof(float);                 // the block's bias_grad and weight_grad
    layer_norm_backward_kernel<<<blocks, LAYER_NORM_BACKWARD_THREADS_PER_BLOCK, shared_bytes>>>(
        (float4*)input_grad.data(), weight_grad.data(), bias_grad.data(), (const float4*)output_grad.data(),
        (const float4*)input.data(), (const float4*)weight.data(), mean.data(), rstd.data(), rows, embed_dim);
    cudaCheck(cudaGetLastError());
}

/// Works out the three gradients, one warp per row, as the forward walks it. input_grad is per row: the
/// warp of that row is the only one writing it. weight_grad and bias_grad are sums over every row of the
/// batch, so many warps add onto the same 768 values at once. Each block first adds its rows into a copy
/// of its own in shared memory (atomicAdd there stays on the SM), and at the end adds that copy onto
/// weight_grad and bias_grad in GPU memory, one atomicAdd per value per block: with 128 blocks,
/// 128 x 2 x 768 = 196,608 adds in GPU memory instead of 4096 x 2 x 768 = 6,291,456. The order of the
/// adds is whatever order the threads get there, so the last bits of weight_grad and bias_grad can
/// differ from run to run.
///
/// Per row, two passes: 1. the two means over the row, mean(norm_grad) and mean(norm_grad * norm),
/// added up across the warp with shuffles as in the forward; 2. the three gradients. norm is worked out
/// again from input, mean and rstd instead of being kept from the forward: one more read of input,
/// against a whole {batch_size, seq_len, embed_dim} tensor kept in GPU memory.
///
/// Where each gradient is worked out, with its step in the post (see the top of the file). A name_grad
/// is the gradient of the loss with respect to name, written d(name) in the post:
///
///     gradient      in the post   step   one per                        written in
///     norm_grad     dx̂            B3     value, {4, 1024, 768}          registers only, both passes
///     bias_grad     dβ            B1     column, {768}: sum over rows   pass 2, then the end of the kernel
///     weight_grad   dγ            B2     column, {768}: sum over rows   pass 2, then the end of the kernel
///     input_grad    dx            C7     value, {4, 1024, 768}          pass 2
///
/// C7 is steps B4 to B9 folded into one line; the post's dr, dv, dμ and dd are never worked out here.
__global__ void layer_norm_backward_kernel(float4* input_grad, float* weight_grad, float* bias_grad,
                                           const float4* output_grad, const float4* input, const float4* weight,
                                           const float* mean, const float* rstd, size_t rows, size_t embed_dim) {
    extern __shared__ float block_sums[];      // 2 x embed_dim, sized at launch: bias_grad, then weight_grad
    float* block_bias_grad = block_sums;
    float* block_weight_grad = block_sums + embed_dim;
    for (size_t i = threadIdx.x; i < 2 * embed_dim; i += blockDim.x) block_sums[i] = 0.0f;
    __syncthreads();                           // all 0 before any warp adds onto them

    int thread_in_warp = threadIdx.x % 32;
    size_t warp_number = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t warps_launched = (size_t)gridDim.x * blockDim.x / 32;
    size_t float4s_embed_dim = embed_dim / 4;
    for (size_t row = warp_number; row < rows; row += warps_launched) {
        const float4* input_row = input + row * float4s_embed_dim;
        const float4* output_grad_row = output_grad + row * float4s_embed_dim;
        float4* input_grad_row = input_grad + row * float4s_embed_dim;
        float row_mean = mean[row], row_rstd = rstd[row];          // kept by the forward

        // 1. the two means over the row that input_grad needs, m1 and m2 in C7. No gradient is written yet.
        float norm_grad_sum = 0.0f, norm_grad_norm_sum = 0.0f;
        for (size_t i = thread_in_warp; i < float4s_embed_dim; i += 32) {
            float4 input4 = input_row[i], output_grad4 = output_grad_row[i], weight4 = weight[i];
            // norm (x̂): the forward's normalized value, worked out again from input, mean and rstd
            float4 norm = make_float4((input4.x - row_mean) * row_rstd, (input4.y - row_mean) * row_rstd,
                                      (input4.z - row_mean) * row_rstd, (input4.w - row_mean) * row_rstd);
            // norm_grad (dx̂, B3) = output_grad * weight: the gradient at norm, one per value
            float4 norm_grad = make_float4(output_grad4.x * weight4.x, output_grad4.y * weight4.y,
                                           output_grad4.z * weight4.z, output_grad4.w * weight4.w);
            // this thread's part of the two sums: of norm_grad for m1, of norm_grad * norm for m2
            norm_grad_sum += norm_grad.x + norm_grad.y + norm_grad.z + norm_grad.w;
            norm_grad_norm_sum += norm_grad.x * norm.x + norm_grad.y * norm.y + norm_grad.z * norm.z +
                                  norm_grad.w * norm.w;
        }
        for (int offset = 16; offset > 0; offset /= 2) {   // the warp's 32 parts added: every thread has both sums
            norm_grad_sum += __shfl_xor_sync(0xffffffff, norm_grad_sum, offset);
            norm_grad_norm_sum += __shfl_xor_sync(0xffffffff, norm_grad_norm_sum, offset);
        }
        float norm_grad_mean = norm_grad_sum / (float)embed_dim;             // m1 = mean(norm_grad)
        float norm_grad_norm_mean = norm_grad_norm_sum / (float)embed_dim;   // m2 = mean(norm_grad * norm)

        // 2. the three gradients, 4 columns (one float4) at a time
        for (size_t i = thread_in_warp; i < float4s_embed_dim; i += 32) {
            float4 input4 = input_row[i], output_grad4 = output_grad_row[i], weight4 = weight[i];
            float4 norm = make_float4((input4.x - row_mean) * row_rstd, (input4.y - row_mean) * row_rstd,
                                      (input4.z - row_mean) * row_rstd, (input4.w - row_mean) * row_rstd);
            float4 norm_grad = make_float4(output_grad4.x * weight4.x, output_grad4.y * weight4.y,
                                           output_grad4.z * weight4.z, output_grad4.w * weight4.w);
            float* bias_grad_columns = block_bias_grad + i * 4;          // the 4 columns this float4 covers
            float* weight_grad_columns = block_weight_grad + i * 4;
            // bias_grad (dβ, B1) += output_grad: this row's part of the sum over all rows. Added into the
            // block's copy in shared memory here, and onto bias_grad itself at the end of the kernel.
            atomicAdd(bias_grad_columns + 0, output_grad4.x);
            atomicAdd(bias_grad_columns + 1, output_grad4.y);
            atomicAdd(bias_grad_columns + 2, output_grad4.z);
            atomicAdd(bias_grad_columns + 3, output_grad4.w);
            // weight_grad (dγ, B2) += output_grad * norm: this row's part, the same way as bias_grad
            atomicAdd(weight_grad_columns + 0, output_grad4.x * norm.x);
            atomicAdd(weight_grad_columns + 1, output_grad4.y * norm.y);
            atomicAdd(weight_grad_columns + 2, output_grad4.z * norm.z);
            atomicAdd(weight_grad_columns + 3, output_grad4.w * norm.w);
            // input_grad (dx, C7) += rstd * (norm_grad - m1 - norm * m2). Only this warp writes this row, so a
            // plain read and write, no atomicAdd. It adds onto what is there: see layer_norm_backward.
            float4 old = input_grad_row[i];
            input_grad_row[i] =
                make_float4(old.x + row_rstd * (norm_grad.x - norm_grad_mean - norm.x * norm_grad_norm_mean),
                            old.y + row_rstd * (norm_grad.y - norm_grad_mean - norm.y * norm_grad_norm_mean),
                            old.z + row_rstd * (norm_grad.z - norm_grad_mean - norm.z * norm_grad_norm_mean),
                            old.w + row_rstd * (norm_grad.w - norm_grad_mean - norm.w * norm_grad_norm_mean));
        }
    }

    // bias_grad (dβ, B1) and weight_grad (dγ, B2): this block's sums, added onto the gradients in GPU
    // memory. Once every block has added its own, each is the sum over all rows.
    __syncthreads();                           // every warp of the block has added its rows
    for (size_t i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        atomicAdd(bias_grad + i, block_bias_grad[i]);
        atomicAdd(weight_grad + i, block_weight_grad[i]);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message if the tensors do not fit together, before a kernel reads past one: input and
/// output {batch_size, seq_len, embed_dim}, mean and rstd {batch_size, seq_len}, weight and bias
/// {embed_dim}. Also embed_dim a multiple of 4 and input, output, weight, bias starting on 16 bytes, for
/// the float4 loads: off 16 bytes, the GPU stops with "misaligned address" at some later call, far from
/// the cause. And embed_dim at most 6144, so the backward's two rows of sums fit the 48 KB of shared
/// memory a block can have. Backward passes the gradients in their place.
inline void layer_norm_check_shapes(const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
                                    const Tensor& weight, const Tensor& bias) {
    bool fits = input.dim() == 3 && output.shape() == input.shape() &&
                mean.shape() == Shape({input.size(0), input.size(1)}) && rstd.shape() == mean.shape() &&
                weight.shape() == Shape({input.size(2)}) && bias.shape() == weight.shape();
    if (!fits) {
        fprintf(stderr, "error: LayerNorm needs input and output {batch_size, seq_len, embed_dim}, mean and rstd "
                        "{batch_size, seq_len}, weight and bias {embed_dim}\n");
        exit(EXIT_FAILURE);
    }
    if (input.size(2) % 4 != 0 || (size_t)input.data() % 16 != 0 || (size_t)output.data() % 16 != 0 ||
        (size_t)weight.data() % 16 != 0 || (size_t)bias.data() % 16 != 0) {
        fprintf(stderr, "error: LayerNorm reads 4 floats at a time (float4), so embed_dim must be a multiple of 4 "
                        "and input, output, weight, bias must start on 16 bytes; got embed_dim %zu, %p, %p, %p, %p\n",
                input.size(2), (void*)input.data(), (void*)output.data(), (void*)weight.data(), (void*)bias.data());
        exit(EXIT_FAILURE);
    }
    if (2 * input.size(2) * sizeof(float) > 48 * 1024) {
        fprintf(stderr, "error: LayerNorm's backward keeps 2 x embed_dim floats per block in shared memory, at most "
                        "48 KB, so embed_dim must be at most 6144; got %zu\n", input.size(2));
        exit(EXIT_FAILURE);
    }
}

#endif // LAYER_NORM_CUH
