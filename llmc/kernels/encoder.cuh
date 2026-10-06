// encoder.cuh — GPT-2's first layer: each token id becomes a row of embed_dim numbers, its
// token's row plus its position's row:
//
//     encoded[b][t] = wte[token_ids[b][t]] + wpe[t]
//
//     wte       {vocab_size, embed_dim}             one row per token:      {50304, 768}
//     wpe       {max_seq_len, embed_dim}            one row per position:   {1024, 768}
//     encoded   {batch_size, seq_len, embed_dim}    one row per token read: {4, 1024, 768}
//
// encoded is the encoder's output, the tokens as numbers: the input of the first block.
// batch_size is the rows of text read at once, seq_len the tokens per row (llm.c calls them
// B and T). seq_len can be less than max_seq_len: then wpe's last rows are not used.
//
// Backward sends encoded's gradient back to the rows that made it, adding onto what is there:
//
//     wte_grad[token_ids[b][t]] += encoded_grad[b][t]    a token seen 5 times gets 5 rows added
//     wpe_grad[t]               += encoded_grad[b][t]    each position t gets batch_size rows
#ifndef ENCODER_CUH
#define ENCODER_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

#define ENCODER_BLOCKS 512          // blocks in every launch here
#define ENCODER_THREADS 1024        // threads per block: 512 x 1024 = 524,288 threads in all

// ── forward and backward ──────────────────────────────────────────────────────
void encoder_forward(const Tensor& encoded, const Tensor& token_ids, const Tensor& wte, const Tensor& wpe);
void encoder_backward(const Tensor& wte_grad, const Tensor& wpe_grad, const Tensor& encoded_grad,
                      const Tensor& token_ids);

// ── for both ──────────────────────────────────────────────────────────────────
void encoder_check_shapes(const Tensor& encoded, const Tensor& token_ids, const Tensor& wte, const Tensor& wpe);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel encoder_forward launches, declared here so it can call it; its code follows it.
__global__ void encoder_forward_kernel(float4* encoded, const int* token_ids, const float4* wte, const float4* wpe,
                                       size_t batch_size, size_t seq_len, size_t embed_dim);

/// Writes encoded[b][t] = wte[token_ids[b][t]] + wpe[t] for every b < batch_size and
/// t < seq_len: the input of the first block. encoded is {batch_size, seq_len, embed_dim},
/// token_ids {batch_size, seq_len} ints, wte {vocab_size, embed_dim}, wpe {max_seq_len,
/// embed_dim} with seq_len <= max_seq_len (rows seq_len and after are not read). The ids are
/// not checked: an id >= vocab_size reads past wte.
inline void encoder_forward(const Tensor& encoded, const Tensor& token_ids, const Tensor& wte, const Tensor& wpe) {
    encoder_check_shapes(encoded, token_ids, wte, wpe);
    size_t batch_size = encoded.size(0), seq_len = encoded.size(1), embed_dim = encoded.size(2);
    encoder_forward_kernel<<<ENCODER_BLOCKS, ENCODER_THREADS>>>((float4*)encoded.data(), token_ids.data_ptr<int>(),
                                                                (const float4*)wte.data(), (const float4*)wpe.data(),
                                                                batch_size, seq_len, embed_dim);
    cudaCheck(cudaGetLastError());
}

/// Writes encoded, one float4 at a time: 4 values of one row, from 4 values of a wte row and 4 of
/// a wpe row. Thread n takes float4s n, n + 524,288, n + 2 x 524,288, ... (the jump: every
/// thread launched). A row of 768 is 192 float4s, so with batch_size = 4, seq_len = 1024
/// encoded has 4 x 1024 x 192 = 786,432 float4s: 1.5 rounds, so threads 0 .. 262,143 write 2
/// and the rest 1.
/// Threads next to each other take float4s next to each other in the row, so a warp reads
/// 32 x 16 = 512 bytes of wte in a row, in a few big requests.
__global__ void encoder_forward_kernel(float4* encoded, const int* token_ids, const float4* wte, const float4* wpe,
                                       size_t batch_size, size_t seq_len, size_t embed_dim) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t float4s_embed_dim = embed_dim / 4;                           // 4 floats per float4: 768 / 4 = 192
    size_t total_float4s = batch_size * seq_len * float4s_embed_dim;    // all of encoded: 786,432
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        size_t token_index = index / float4s_embed_dim;                 // which token of the batch: b x seq_len + t
        size_t float4_embed_index = index % float4s_embed_dim;          // which of its 192 float4s
        size_t position = token_index % seq_len;                        // t: its place in its row of text
        size_t token_id = token_ids[token_index];                       // which wte row to read
        float4 token_embed = wte[token_id * float4s_embed_dim + float4_embed_index];
        float4 position_embed = wpe[position * float4s_embed_dim + float4_embed_index];
        encoded[index] = make_float4(token_embed.x + position_embed.x, token_embed.y + position_embed.y,
                                     token_embed.z + position_embed.z, token_embed.w + position_embed.w);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernels encoder_backward launches, declared here so it can call them; their code follows it.
__global__ void encoder_backward_wte_kernel(float* wte_grad, const float* encoded_grad, const int* token_ids,
                                            size_t batch_size, size_t seq_len, size_t embed_dim);
__global__ void encoder_backward_wpe_kernel(float4* wpe_grad, const float4* encoded_grad, size_t batch_size,
                                            size_t seq_len, size_t embed_dim);

/// Adds encoded_grad back into the rows the forward read: row (b, t) of encoded_grad onto row
/// token_ids[b][t] of wte_grad and onto row t of wpe_grad. It adds, so the gradients must be
/// 0 before the first backward of a step; each micro-batch then adds its share. The shapes are
/// the forward's: wte_grad as wte, wpe_grad as wpe, encoded_grad as encoded.
inline void encoder_backward(const Tensor& wte_grad, const Tensor& wpe_grad, const Tensor& encoded_grad,
                             const Tensor& token_ids) {
    encoder_check_shapes(encoded_grad, token_ids, wte_grad, wpe_grad);
    size_t batch_size = encoded_grad.size(0), seq_len = encoded_grad.size(1), embed_dim = encoded_grad.size(2);
    encoder_backward_wte_kernel<<<ENCODER_BLOCKS, ENCODER_THREADS>>>(wte_grad.data(), encoded_grad.data(),
                                                                     token_ids.data_ptr<int>(), batch_size, seq_len,
                                                                     embed_dim);
    cudaCheck(cudaGetLastError());
    encoder_backward_wpe_kernel<<<ENCODER_BLOCKS, ENCODER_THREADS>>>((float4*)wpe_grad.data(),
                                                                     (const float4*)encoded_grad.data(), batch_size,
                                                                     seq_len, embed_dim);
    cudaCheck(cudaGetLastError());
}

/// Adds encoded_grad onto wte_grad, one value at a time. Thread n takes values n, n + 524,288,
/// n + 2 x 524,288, ...: with batch_size = 4, seq_len = 1024 encoded_grad has 4 x 1024 x 768 =
/// 3,145,728 values, exactly 6 rounds, so every thread adds 6. A token that is read at several
/// positions has several threads adding onto the same row, maybe at the same moment. Two plain
/// += on one address at once can lose one:
///
///     thread A reads 5, thread B reads 5, A writes 5 + 1 = 6, B writes 5 + 2 = 7   (1 is lost)
///
/// atomicAdd makes each read-add-write whole, one thread at a time, so nothing is lost. But
/// the order of the adds is whatever order the threads get there, so the last bits of a row
/// can differ from run to run (float adds give slightly different sums in another order).
__global__ void encoder_backward_wte_kernel(float* wte_grad, const float* encoded_grad, const int* token_ids,
                                            size_t batch_size, size_t seq_len, size_t embed_dim) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_values = batch_size * seq_len * embed_dim;             // all of encoded_grad: 3,145,728
    for (size_t index = thread_number; index < total_values; index += (size_t)gridDim.x * blockDim.x) {
        size_t token_index = index / embed_dim;                         // which token of the batch: b x seq_len + t
        size_t embed_index = index % embed_dim;                         // which of its 768 values
        size_t token_id = token_ids[token_index];                       // which wte_grad row to add onto
        atomicAdd(&wte_grad[token_id * embed_dim + embed_index], encoded_grad[index]);
    }
}

/// Adds the batch_size rows of encoded_grad at position t onto row t of wpe_grad, one float4 at
/// a time. Thread n takes float4s n, n + 524,288, ...: with seq_len = 1024 the first seq_len
/// rows of wpe_grad are 1024 x 192 = 196,608 float4s, less than one round, so threads
/// 0 .. 196,607 add 1 and the rest (blocks 192 .. 511) have nothing to do. Each float4 of
/// wpe_grad has one thread, so no other thread writes it and no atomics are needed. It adds
/// b = 0, 1, ..., batch_size - 1 in that order, every run: the same wpe_grad every time, to the
/// last bit.
__global__ void encoder_backward_wpe_kernel(float4* wpe_grad, const float4* encoded_grad, size_t batch_size,
                                            size_t seq_len, size_t embed_dim) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t float4s_embed_dim = embed_dim / 4;                           // 4 floats per float4: 768 / 4 = 192
    size_t total_float4s = seq_len * float4s_embed_dim;                 // wpe_grad's first seq_len rows: 196,608
    for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (size_t b = 0; b < batch_size; b++) {
            float4 g = encoded_grad[b * seq_len * float4s_embed_dim + index]; // row (b, t), the same float4 of it
            sum.x += g.x;
            sum.y += g.y;
            sum.z += g.z;
            sum.w += g.w;
        }
        float4 old = wpe_grad[index];
        wpe_grad[index] = make_float4(old.x + sum.x, old.y + sum.y, old.z + sum.z, old.w + sum.w);
    }
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Stops with a message if the tensors do not fit together, before a kernel reads past one:
/// encoded {batch_size, seq_len, embed_dim}, token_ids {batch_size, seq_len} ints, wte
/// {vocab_size, embed_dim}, wpe {max_seq_len, embed_dim} with seq_len <= max_seq_len. Also
/// embed_dim a multiple of 4 and every float tensor starting on 16 bytes, for the float4 loads:
/// off 16 bytes, the GPU stops with "misaligned address" at some later call, far from the
/// cause. Backward passes the gradients in their place.
inline void encoder_check_shapes(const Tensor& encoded, const Tensor& token_ids, const Tensor& wte, const Tensor& wpe) {
    bool fits = encoded.dim() == 3 && token_ids.dim() == 2 && wte.dim() == 2 && wpe.dim() == 2 &&
                token_ids.dtype() == DType::Int32 && token_ids.size(0) == encoded.size(0) &&
                token_ids.size(1) == encoded.size(1) && wte.size(1) == encoded.size(2) &&
                wpe.size(1) == encoded.size(2) && encoded.size(1) <= wpe.size(0);
    if (!fits) {
        fprintf(stderr, "error: the encoder needs encoded {batch_size, seq_len, embed_dim}, token_ids {batch_size, "
                        "seq_len} ints, wte {vocab_size, embed_dim}, wpe {max_seq_len, embed_dim} with seq_len <= "
                        "max_seq_len\n");
        exit(EXIT_FAILURE);
    }
    if (encoded.size(2) % 4 != 0 || (size_t)encoded.data() % 16 != 0 || (size_t)wte.data() % 16 != 0 ||
        (size_t)wpe.data() % 16 != 0) {
        fprintf(stderr, "error: the encoder reads 4 floats at a time (float4), so embed_dim must be a multiple of 4 "
                        "and encoded, wte, wpe must start on 16 bytes; got embed_dim %zu, %p, %p, %p\n",
                encoded.size(2), (void*)encoded.data(), (void*)wte.data(), (void*)wpe.data());
        exit(EXIT_FAILURE);
    }
}

#endif // ENCODER_CUH
