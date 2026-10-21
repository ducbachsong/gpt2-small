// linear.cuh — a linear layer, as PyTorch's nn.Linear: each row of input (one token) times the weight
// matrix, plus the bias:
//
//     output = input · weightᵀ + bias        F1:  y[t][j] = sum over i of x[t][i] · W[j][i] + b[j]
//
//     input    {batch_size, seq_len, in_features}    one row per token:             {4, 1024, 768}
//     weight   {out_features, in_features}           one row per output column:     {2304, 768}
//     bias     {out_features}                        one per output column:         {2304}
//     output   {batch_size, seq_len, out_features}                                  {4, 1024, 2304}
//
// Each output value is the dot product of two rows that sit in memory with their values next to each other:
// a row of input and a row of weight (that is why PyTorch keeps weight as {out, in}, and the formula has
// weightᵀ). GPT-2 small has 4 in each of its 12 blocks, and the head after the last block, whose weight is
// wte. With batch_size = 4, seq_len = 1024 (4096 rows):
//
//     linear             in_features   out_features   forward work    blocks of the forward
//     qkv                768           2304           14.5 GFLOP      18 x 32 = 576
//     attention output   768           768             4.8 GFLOP       6 x 32 = 192
//     MLP up             768           3072           19.3 GFLOP      24 x 32 = 768
//     MLP down           3072          768            19.3 GFLOP       6 x 32 = 192
//     head (no bias)     768           50304 (wte)   316.5 GFLOP     393 x 32 = 12,576
//
// Backward, with output_grad the gradient arriving from the layer above (the post's C2):
//
//     B1   bias_grad   += sum over all rows of output_grad    db = dyᵀ · 1
//     B2   weight_grad += output_gradᵀ · input                dW = dyᵀ x    {out, rows} · {rows, in} = {out, in}
//     B3   input_grad   = output_grad · weight                dx = dy W     {rows, out} · {out, in}  = {rows, in}
//
// Two more products the size of the forward's: the backward is about twice the forward's work. Each is one
// kernel: the forward's blocks make the squares of output; the backward's make those of weight_grad and
// input_grad, and bias_grad on the way.
//
// The steps in the comments (F1, B1 to B3, C1, C2) are those of the post that works all this out,
// ducbachsong.github.io/posts/linear-backward. The same values, by their names here and there:
//
//     in the code      in the post     in the code             in the post
//     input            x               output_grad             dy
//     weight           W               input_grad              dx
//     bias             b               weight_grad             dW
//     output           y               bias_grad               db
//     rows             T               row                     t: which token
//     in_features      n_in            input column            i
//     out_features     n_out           output column           j
//
// The forward kernel's loops are F1's t, i, j. The two backward kernels' are C1's r, c, k, as they make two
// products with one code: product[r][c] = sum over k of left[r][k] · right[k][c], k the summed index.
#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

// The kernels' index math is written for these numbers: change them together with it.
#define LINEAR_VALUES_PER_BLOCK_ROWS 128   // each block makes 128 rows x 128 columns of a product;
#define LINEAR_VALUES_PER_BLOCK_COLUMNS 128 // both must stay 128: 256 threads copy 128 lines of each piece
#define LINEAR_VALUES_PER_THREAD_ROWS 8    // each of its threads makes 8 rows x 8 columns of them;
#define LINEAR_VALUES_PER_THREAD_COLUMNS 8 // both must stay 8: the 256-thread copy, the two-float4 reads and
                                           // the float4 store are all written for 8 x 8
#define LINEAR_THREADS_PER_BLOCK 256       // (128 / 8 rows) x (128 / 8 columns) = 16 x 16 threads
#define LINEAR_VALUES_PER_THREAD 8         // values of the sum (k) each thread adds per step, not output values;
                                           // the block copies these 8 into its pieces
// linear_store_piece fills the left piece (128 rows) in dW and the right piece (128 columns) everywhere else
static_assert(LINEAR_VALUES_PER_BLOCK_ROWS == LINEAR_VALUES_PER_BLOCK_COLUMNS, "the copy is written for a square");
// A row of a piece holds 128 values but is 132 floats long: 4 unused floats at its end. The turned copies have
// one warp write rows k and k + 4 at once. 128 long, those sit 4 x 128 = 512 floats apart, a multiple of the
// 32 banks of shared memory, so both halves of the warp hit banks 0..15 and take turns. 132 long, they sit 528
// apart, 16 banks on: one half takes banks 0..15, the other 16..31, together. 132 floats is 528 bytes, a
// multiple of 16, so every row still starts where a float4 can be read.
#define LINEAR_BLOCK_PIECE_ROW_LENGTH 132
static_assert(LINEAR_BLOCK_PIECE_ROW_LENGTH % 4 == 0, "each piece row must start on 16 bytes, for the float4s");
// A thread's 8 columns are two halves of 4, the second 64 columns after the first: thread 0 makes columns 0..3
// and 64..67, thread 1 4..7 and 68..71, .., thread 15 60..63 and 124..127. At each k, a warp then reads its
// first halves of the right piece as 64 neighbouring floats (threads 16..31 read the same as 0..15), each of
// the 32 banks twice: the fewest reads 64 floats can take. As 8 neighbours (0..7, 8..15, ..), their float4s
// started 8 floats apart, and columns 0, 32, 64 and 96 all fell on banks 0..3: 4 reads where 2 do (ncu: 40%
// of the shared loads were such repeats). The stores of the sums get the same 64 neighbours.
#define LINEAR_SECOND_HALF_COLUMN_OFFSET (LINEAR_VALUES_PER_BLOCK_COLUMNS / 2)
static_assert((LINEAR_VALUES_PER_BLOCK_COLUMNS / LINEAR_VALUES_PER_THREAD_COLUMNS) *
                  (LINEAR_VALUES_PER_THREAD_COLUMNS / 2) == LINEAR_SECOND_HALF_COLUMN_OFFSET,
              "16 threads x 4 columns must fill the first half exactly");
// The forward numbers its blocks in groups of 8 rows of squares, down a group first, then across, then the
// next group. The 80 blocks a T4 runs at once (2 on each of 40 SMs) then make about 8 rows x 10 columns of
// squares: they share 8 bands of x and 10 bands of W, read from GPU memory once and then from L2. Numbered
// across first, they would make 1 row x 80 columns: one band of x, but 80 different bands of W. For the head
// (W 50304 x 768, 154 MB, the T4's L2 is 4 MB), that reads all of W once per row of squares: 32 x 154 MB =
// 4.9 GB per forward. In groups of 8 it is 4 x 154 MB.
#define LINEAR_BLOCK_ROWS_PER_GROUP 8

// ── forward and backward ──────────────────────────────────────────────────────
void linear_forward(const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias);
void linear_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                     const Tensor& output_grad, const Tensor& input, const Tensor& weight);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

// The kernel linear_forward launches, declared here so it can call it; its code follows it.
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_forward_kernel(float* output, const float* input, const float* weight, const float* bias,
                      size_t input_rows, size_t in_features, size_t out_features);
// The copies all three kernels use, declared here so the forward can call them; their code is near the end
__device__ __forceinline__ float4 linear_load_piece(const float* matrix, size_t matrix_rows, size_t matrix_columns,
                                                    size_t step_first_k, size_t block_first_column);
__device__ __forceinline__ void linear_store_piece(float (*block_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH], float4 values);
__device__ __forceinline__ float4 linear_load_piece_turned(const float* matrix, size_t matrix_rows,
                                                           size_t matrix_columns, size_t block_first_row,
                                                           size_t step_first_k);
__device__ __forceinline__ void linear_store_piece_turned(float (*block_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
                                                          float4 values);

/// Writes output = input · weightᵀ + bias (F1, and C1's y = x Wᵀ + b): each row of input dotted with each
/// row of weight, plus that column's bias. input is {..., in_features} ({batch_size, seq_len, in_features}
/// in GPT-2), output the same with out_features last, weight {out_features, in_features}, bias
/// {out_features} or Tensor() for none (the head). One block per 128 x 128 square of output, rounded up:
/// for the qkv, 4096 rows and 2304 output columns, 18 x 32 = 576 blocks of 256 threads.
///
/// First it stops with a message if the tensors do not fit together, before the kernel reads past one;
/// or if in_features or out_features is not a multiple of 4, or a tensor does not start on 16 bytes: the
/// kernel reads 4 floats at a time (float4), and off 16 bytes the GPU stops with "misaligned address" at
/// some later call, far from the cause.
inline void linear_forward(const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias) {
    bool has_bias = bias.data() != nullptr;                     // Tensor() has no values: no bias
    bool fits = input.dim() >= 1 && weight.dim() == 2 && output.dim() == input.dim() &&
                input.size(input.dim() - 1) == weight.size(1) && output.size(output.dim() - 1) == weight.size(0) &&
                (!has_bias || bias.shape() == Shape({weight.size(0)}));
    for (int d = 0; fits && d < input.dim() - 1; d++) fits = output.size(d) == input.size(d);   // the same rows
    if (!fits) {
        fprintf(stderr, "error: a linear layer needs input {..., in_features}, output {..., out_features} with the "
                        "same rows, weight {out_features, in_features}, bias {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    if (weight.size(0) % 4 != 0 || weight.size(1) % 4 != 0 || (size_t)output.data() % 16 != 0 ||
        (size_t)input.data() % 16 != 0 || (size_t)weight.data() % 16 != 0 || (size_t)bias.data() % 16 != 0) {
        fprintf(stderr, "error: a linear layer reads 4 floats at a time (float4), so in_features and out_features "
                        "must be multiples of 4 and output, input, weight, bias must start on 16 bytes; got "
                        "in_features %zu, out_features %zu, %p, %p, %p, %p\n",
                weight.size(1), weight.size(0), (void*)output.data(), (void*)input.data(), (void*)weight.data(),
                (void*)bias.data());
        exit(EXIT_FAILURE);
    }

    size_t out_features = weight.size(0), in_features = weight.size(1), input_rows = input.numel() / in_features;
    // One block per square, launched as one line of blocks; the kernel turns its number back into a square.
    // Squares across (output columns j) and down (rows t), rounded up: for the qkv, 18 x 32 = 576 blocks
    size_t total_block_columns = (out_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    size_t total_block_rows = (input_rows + LINEAR_VALUES_PER_BLOCK_ROWS - 1) / LINEAR_VALUES_PER_BLOCK_ROWS;
    unsigned total_blocks = (unsigned)(total_block_columns * total_block_rows);
    linear_forward_kernel<<<total_blocks, LINEAR_THREADS_PER_BLOCK>>>(output.data(), input.data(), weight.data(),
                                                                      bias.data(), input_rows, in_features,
                                                                      out_features);
    cudaCheck(cudaGetLastError());
}

/// Works out F1 for one 128 x 128 square of output: y[t][j] = sum over i of x[t][i] · W[j][i] + b[j], for
/// rows t = 128 block_row .. + 127 and output columns j = 128 block_column .. + 127, the square blockIdx.x
/// gives in groups of 8 rows of squares (see LINEAR_BLOCK_ROWS_PER_GROUP). Each thread makes 8 x 8 of
/// them: rows 8 (threadIdx.x / 16) .. + 7 and output columns 4 (threadIdx.x % 16) .. + 3 and 64 on from
/// those, of the block's (see LINEAR_SECOND_HALF_COLUMN_OFFSET).
///
/// The sum over i is in_features values long. The block walks it 8 input columns at a time. Each step, its
/// 256 threads copy those 8 values of the block's 128 rows of x and 128 rows of W into shared memory, one
/// float4 each; then each thread adds 8 x 8 x 8 = 512 products into its 64 sums, which stay in registers. A
/// step reads 2 x 1024 values from GPU memory and does 128 x 128 x 8 = 131,072 multiply-adds: 64 per value
/// read. One thread per output value, reading its two rows from GPU memory, would do 1 per 2 values read.
///
/// The copy is double-buffered. There are two sets of pieces, and each step's math reads one. Before that
/// math, each thread has already asked GPU memory for its float4s of the next step, which it writes into the
/// other set once the math is done. The ~300 cycles a read from GPU memory takes pass during the math, not
/// before it, and one __syncthreads() per step is enough (see the loop). __launch_bounds__(256, 2) keeps it
/// at 128 registers or fewer, so 2 blocks fit an SM.
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_forward_kernel(float* output, const float* input, const float* weight, const float* bias,
                      size_t input_rows, size_t in_features, size_t out_features) {
    // This step's 8 input columns i of x and W, with i as the rows: block_input_pieces[set][i][t] = x[t][i] and
    // block_weight_pieces[set][i][j] = W[j][i] (a piece of Wᵀ), for the block's 128 rows t and 128 output
    // columns j. The 8 values a thread needs at one i sit next to each other: two float4 reads. Two sets: one
    // step's math reads one while the next step's values go into the other (16.5 KB).
    __shared__ __align__(16) float block_input_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_weight_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    // Its square. The blocks are numbered in groups of 8 rows of squares (LINEAR_BLOCK_ROWS_PER_GROUP), down a
    // group first: / and % find the group, then its row and column inside the group. For the head, 32 rows
    // x 393 columns of squares: 4 groups of 8 x 393 = 3144 blocks; block 3145 is group 1, row 9, column 0.
    size_t total_block_columns = (out_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    size_t total_block_rows = (input_rows + LINEAR_VALUES_PER_BLOCK_ROWS - 1) / LINEAR_VALUES_PER_BLOCK_ROWS;
    size_t blocks_per_group = LINEAR_BLOCK_ROWS_PER_GROUP * total_block_columns;
    size_t group = blockIdx.x / blocks_per_group;
    size_t group_first_block_row = group * LINEAR_BLOCK_ROWS_PER_GROUP;
    size_t group_block_rows = total_block_rows - group_first_block_row;   // 8, or fewer in the last group
    if (group_block_rows > LINEAR_BLOCK_ROWS_PER_GROUP) group_block_rows = LINEAR_BLOCK_ROWS_PER_GROUP;
    size_t block_number_in_group = blockIdx.x % blocks_per_group;
    size_t block_row = group_first_block_row + block_number_in_group % group_block_rows;   // which 128 rows t
    size_t block_column = block_number_in_group / group_block_rows;              // which 128 output columns j
    size_t block_first_row = block_row * LINEAR_VALUES_PER_BLOCK_ROWS;                    // the block's first t
    size_t block_first_output_column = block_column * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // the block's first j
    // Its 256 threads stand 16 x 16, numbered across first, each making 8 x 8 of the block's 128 x 128
    const int threads_across = LINEAR_VALUES_PER_BLOCK_COLUMNS / LINEAR_VALUES_PER_THREAD_COLUMNS;   // 128 / 8 = 16
    // its first t inside the block's 128 x 128 (0, 8, .. 120), and its first j: its 8 columns j are two
    // halves of 4, at 0, 4, .. 60 and 64 on from there (see LINEAR_SECOND_HALF_COLUMN_OFFSET)
    int thread_first_row = (threadIdx.x / threads_across) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_output_column = (threadIdx.x % threads_across) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);
    const int max_floats_per_thread_read = 4;   // a float4: 16 bytes, the widest read one thread can do

    // sums[t][j]: its 64 values of y, in registers (all indexes known when compiling)
    float sums[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};
    // i is the summed index: each step sums over 8 input columns. x and W both sit with i last ({T, n_in},
    // {n_out, n_in}), so both are copied turned round; a row of W is an output column j.
    size_t total_steps = (in_features + LINEAR_VALUES_PER_THREAD - 1) / LINEAR_VALUES_PER_THREAD;
    // its float4s of x and W for the step being copied, read from GPU memory into registers: step 0's first
    float4 input4 = linear_load_piece_turned(input, input_rows, in_features, block_first_row, 0);
    float4 weight4 = linear_load_piece_turned(weight, out_features, in_features, block_first_output_column, 0);
    for (size_t step = 0; step < total_steps; step++) {
        int piece_set = (int)(step % 2);    // the set of pieces this step reads; the next step fills the other
        linear_store_piece_turned(block_input_pieces[piece_set], input4);
        linear_store_piece_turned(block_weight_pieces[piece_set], weight4);
        // Both pieces whole before any thread reads them. Every thread is also past the last step's math, so
        // no one reads the other set any more: the next step may write over it.
        __syncthreads();
        // The next step's float4s, asked for now: the ~300 cycles GPU memory takes pass during the math below,
        // and they are not needed until the next step's store
        size_t next_step_first_input_column = (step + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_input_column < in_features) {
            input4 = linear_load_piece_turned(input, input_rows, in_features, block_first_row,
                                              next_step_first_input_column);
            weight4 = linear_load_piece_turned(weight, out_features, in_features, block_first_output_column,
                                               next_step_first_input_column);
        }

        // F1's sum over i, for this step's 8 values of i: sums[t][j] += x[t][i] · W[j][i]
        const float (*block_input_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH] = block_input_pieces[piece_set];
        const float (*block_weight_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH] = block_weight_pieces[piece_set];
#pragma unroll
        for (int i = 0; i < LINEAR_VALUES_PER_THREAD; i++) {
            // x[t][i] of its 8 rows t and W[j][i] of its 8 columns j: a float4 holds 4, so two halves each,
            // its rows 0..3 and 4..7, and its columns 0..3 and the 4 that are 64 on
            float4 input_first_half = *(const float4*)&block_input_piece[i][thread_first_row];
            float4 input_second_half =
                *(const float4*)&block_input_piece[i][thread_first_row + max_floats_per_thread_read];
            float4 weight_first_half = *(const float4*)&block_weight_piece[i][thread_first_output_column];
            float4 weight_second_half =
                *(const float4*)&block_weight_piece[i][thread_first_output_column + LINEAR_SECOND_HALF_COLUMN_OFFSET];
            float input_values[LINEAR_VALUES_PER_THREAD_ROWS] = {
                input_first_half.x,  input_first_half.y,  input_first_half.z,  input_first_half.w,
                input_second_half.x, input_second_half.y, input_second_half.z, input_second_half.w};
            float weight_values[LINEAR_VALUES_PER_THREAD_COLUMNS] = {
                weight_first_half.x,  weight_first_half.y,  weight_first_half.z,  weight_first_half.w,
                weight_second_half.x, weight_second_half.y, weight_second_half.z, weight_second_half.w};
#pragma unroll
            for (int t = 0; t < LINEAR_VALUES_PER_THREAD_ROWS;t++) {
#pragma unroll
                for (int j = 0; j < LINEAR_VALUES_PER_THREAD_COLUMNS;j++)
                    sums[t][j] += input_values[t] * weight_values[j];
            }
        }
    }

    // y[t][j] = sums[t][j] + b[j], F1's last term; one float4 (4 output columns j) at a time
    const int max_floats_per_thread_write = 4;   // a float4: 16 bytes, the widest write one thread can do
#pragma unroll
    for (int t = 0; t < LINEAR_VALUES_PER_THREAD_ROWS;t++) {
        size_t row = block_first_row + thread_first_row + t;
#pragma unroll
        for (int j = 0; j < LINEAR_VALUES_PER_THREAD_COLUMNS;j += max_floats_per_thread_write) {
            // sums j .. j + 3: its first half (j = 0) or its second, 64 columns on (j = 4)
            int half = j / max_floats_per_thread_write;
            size_t output_column =
                block_first_output_column + thread_first_output_column + half * LINEAR_SECOND_HALF_COLUMN_OFFSET;
            if (row >= input_rows || output_column >= out_features) continue;   // past the edge of a part-full square
            float4 result = make_float4(sums[t][j], sums[t][j + 1], sums[t][j + 2], sums[t][j + 3]);
            if (bias != nullptr) {
                float4 bias4 = *(const float4*)(bias + output_column);
                result = make_float4(result.x + bias4.x, result.y + bias4.y, result.z + bias4.z, result.w + bias4.w);
            }
            *(float4*)(output + row * out_features + output_column) = result;
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

// The kernels linear_backward launches, declared here so it can call them; their code follows it.
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_weight_grad_kernel(float* weight_grad, float* bias_grad, const float* output_grad,
                          const float* input, size_t input_rows, size_t in_features,
                          size_t out_features);
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_input_grad_kernel(float* input_grad, const float* output_grad, const float* weight,
                         size_t input_rows, size_t in_features, size_t out_features);

/// Works out the gradients of input, weight and bias (B3, B2, B1, as C2's products) from output_grad and
/// what the forward read: input and weight. input_grad is written over: in GPT-2 it is the gradient of a
/// LayerNorm's, GELU's or attention's output, which nothing else adds to. weight_grad and bias_grad are
/// added onto, so they must be 0 before the first backward of a step; each micro-batch then adds its share,
/// and the head's weight_grad (wte's) also gets the encoder's (the post's Table 16). The shapes are the
/// forward's: input_grad as input, weight_grad as weight, bias_grad as bias (Tensor() for none),
/// output_grad as output.
///
/// Two launches, one after the other: one block per 128 x 128 square of weight_grad, then one per square of
/// input_grad, rounded up. For the qkv: weight_grad 18 down x 6 across = 108 blocks, input_grad 32 x 6 = 192,
/// of 256 threads each. Two kernels, not one, so each gets the registers its own product needs: one kernel
/// making both took 158, the larger product's, and fit only 1 block per SM. First it stops with a message,
/// as the forward does, if the tensors do not fit together or are not ready for float4 reads.
inline void linear_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                            const Tensor& output_grad, const Tensor& input, const Tensor& weight) {
    bool has_bias = bias_grad.data() != nullptr;                // Tensor() has no values: no bias
    bool fits = input.dim() >= 1 && weight.dim() == 2 && output_grad.dim() == input.dim() &&
                input.size(input.dim() - 1) == weight.size(1) &&
                output_grad.size(output_grad.dim() - 1) == weight.size(0) && input_grad.shape() == input.shape() &&
                weight_grad.shape() == weight.shape() && (!has_bias || bias_grad.shape() == Shape({weight.size(0)}));
    for (int d = 0; fits && d < input.dim() - 1; d++) fits = output_grad.size(d) == input.size(d);   // the same rows
    if (!fits) {
        fprintf(stderr, "error: a linear layer's backward needs input and input_grad {..., in_features}, output_grad "
                        "{..., out_features} with the same rows, weight and weight_grad {out_features, in_features}, "
                        "bias_grad {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    if (weight.size(0) % 4 != 0 || weight.size(1) % 4 != 0 || (size_t)input_grad.data() % 16 != 0 ||
        (size_t)weight_grad.data() % 16 != 0 || (size_t)bias_grad.data() % 16 != 0 ||
        (size_t)output_grad.data() % 16 != 0 || (size_t)input.data() % 16 != 0 || (size_t)weight.data() % 16 != 0) {
        fprintf(stderr, "error: a linear layer reads 4 floats at a time (float4), so in_features and out_features "
                        "must be multiples of 4 and input_grad, weight_grad, bias_grad, output_grad, input, weight "
                        "must start on 16 bytes; got in_features %zu, out_features %zu, %p, %p, %p, %p, %p, %p\n",
                weight.size(1), weight.size(0), (void*)input_grad.data(), (void*)weight_grad.data(),
                (void*)bias_grad.data(), (void*)output_grad.data(), (void*)input.data(), (void*)weight.data());
        exit(EXIT_FAILURE);
    }

    size_t out_features = weight.size(0), in_features = weight.size(1), input_rows = input.numel() / in_features;
    // One block per square of each product, launched as one line of blocks; each kernel turns a block's
    // number back into its square. Rounded up; the numbers are the qkv's.
    size_t total_weight_grad_block_rows =                                                     // rows j: 18
        (out_features + LINEAR_VALUES_PER_BLOCK_ROWS - 1) / LINEAR_VALUES_PER_BLOCK_ROWS;
    size_t total_weight_grad_block_columns =                                                  // columns i: 6
        (in_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    size_t total_input_grad_block_rows =                                                      // rows t: 32
        (input_rows + LINEAR_VALUES_PER_BLOCK_ROWS - 1) / LINEAR_VALUES_PER_BLOCK_ROWS;
    size_t total_input_grad_block_columns =                                                   // columns i: 6
        (in_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    unsigned total_weight_grad_blocks = (unsigned)(total_weight_grad_block_rows * total_weight_grad_block_columns);
    unsigned total_input_grad_blocks = (unsigned)(total_input_grad_block_rows * total_input_grad_block_columns);
    linear_weight_grad_kernel<<<total_weight_grad_blocks, LINEAR_THREADS_PER_BLOCK>>>(   // 108 blocks
        weight_grad.data(), bias_grad.data(), output_grad.data(), input.data(), input_rows, in_features, out_features);
    cudaCheck(cudaGetLastError());
    linear_input_grad_kernel<<<total_input_grad_blocks, LINEAR_THREADS_PER_BLOCK>>>(     // 192 blocks
        input_grad.data(), output_grad.data(), weight.data(), input_rows, in_features, out_features);
    cudaCheck(cudaGetLastError());
}

// The pieces the two backward kernels are made of, declared here so they can call them; their code follows
// them, in this order. All are inlined into the kernels, so the sums they are handed stay in registers.
__device__ __forceinline__ void linear_weight_grad_block(
        float (*block_left_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], float* weight_grad,
        float* bias_grad, const float* output_grad, const float* input, size_t input_rows, size_t in_features,
        size_t out_features, size_t block_row, size_t block_column);
__device__ __forceinline__ void linear_input_grad_block(
        float (*block_left_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], float* input_grad,
        const float* output_grad, const float* weight, size_t input_rows, size_t in_features, size_t out_features,
        size_t block_row, size_t block_column);
__device__ __forceinline__ void linear_add_outer_products(
        float (&sums)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS],
        const float (*block_left_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        const float (*block_right_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        int thread_first_row, int thread_first_column,
        float (&bias_sums)[LINEAR_VALUES_PER_THREAD_ROWS], bool adds_bias_grad);
__device__ __forceinline__ void linear_store_sums(
        float* product, size_t product_rows, size_t product_columns,
        const float (&sums)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS], size_t first_row,
        size_t first_column, bool add_onto);

// The two backward products are C2's, read as C1 reads them: product[r][c] = sum over k of left[r][k] ·
// right[k][c], with k the summed index and dy as the left factor:
//
//     step   product                    r    c    k (summed)   left: dy sits as            right sits as
//     B2     dW = dyᵀ x   {out, in}     j    i    t: rows      {T, n_out}: turned round    x {T, n_in}
//     B3     dx = dy W    {rows, in}    t    i    j: n_out     {T, n_out}                  W {n_out, n_in}
//
// Each kernel works out one 128 x 128 square of its product per block: / and % turn the block's number into
// a row and a column of squares, across first, and its sub-function (linear_weight_grad_block,
// linear_input_grad_block) does the rest.
//
// __launch_bounds__(256, 2) asks the compiler to fit 2 blocks of 256 threads on an SM: at most 65,536 / 512
// = 128 registers per thread. With the double-buffered copy, left to itself it took 133 (dW) and 130 (dx),
// and only 1 block fit an SM.

/// Works out one 128 x 128 square of weight_grad (B2), and of bias_grad with it (B1): see
/// linear_weight_grad_block. For the qkv, 108 blocks: 18 squares down (output columns j), 6 across (input
/// columns i).
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_weight_grad_kernel(float* weight_grad, float* bias_grad, const float* output_grad,
                          const float* input, size_t input_rows, size_t in_features,
                          size_t out_features) {
    // This step's 8 values of k of left and right, with k as the rows: block_left_pieces[set][k][r] = left[r][k]
    // and block_right_pieces[set][k][c] = right[k][c], for the block's 128 rows r and 128 columns c. Two sets
    // (double buffering, see the forward): one step's math reads one while the next step's values go into the
    // other (16.5 KB).
    __shared__ __align__(16) float block_left_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_right_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    size_t total_block_columns =                                                              // columns i: 6
        (in_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    size_t block_row = blockIdx.x / total_block_columns;           // which 128 rows r of dW: output columns j
    size_t block_column = blockIdx.x % total_block_columns;        // which 128 columns c: input columns i
    linear_weight_grad_block(block_left_pieces, block_right_pieces, weight_grad, bias_grad, output_grad, input,
                             input_rows, in_features, out_features, block_row, block_column);
}

/// Works out one 128 x 128 square of input_grad (B3): see linear_input_grad_block. For the qkv, 192 blocks:
/// 32 squares down (rows t), 6 across (input columns i).
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_input_grad_kernel(float* input_grad, const float* output_grad, const float* weight,
                         size_t input_rows, size_t in_features, size_t out_features) {
    // the same two sets of pieces as linear_weight_grad_kernel's (16.5 KB)
    __shared__ __align__(16) float block_left_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_right_pieces[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    size_t total_block_columns =                                                              // columns i: 6
        (in_features + LINEAR_VALUES_PER_BLOCK_COLUMNS - 1) / LINEAR_VALUES_PER_BLOCK_COLUMNS;
    size_t block_row = blockIdx.x / total_block_columns;           // which 128 rows r of dx: rows t
    size_t block_column = blockIdx.x % total_block_columns;        // which 128 columns c: input columns i
    linear_input_grad_block(block_left_pieces, block_right_pieces, input_grad, output_grad, weight, input_rows,
                            in_features, out_features, block_row, block_column);
}

/// Makes one 128 x 128 square of weight_grad, B2: dW += dyᵀ x, added onto (see linear_backward). r is an
/// output column j, c an input column i, and the sum runs over all T rows t. dy sits as {T, n_out}, which is
/// {k, r}: dyᵀ is read turned round by copying it as it is.
///
/// B1 rides along. db = dyᵀ · 1 is B2 with a right factor of all 1s (the post's check "the bias is a weight on an
/// input of 1"), and block_left_piece already holds dy. So the blocks of the first column of squares (block_column
/// = 0) also add up the rows r of block_left_piece: with a bias, each thread adds its 8 rows' values at every k
/// (inside linear_add_outer_products, from the values it reads anyway), and the 16 threads with thread_first_column
/// 0 add them onto bias_grad at the end. One thread per bias_grad value, adding in the same order every run:
/// bias_grad is the same to the last bit (no atomicAdd).
__device__ __forceinline__ void linear_weight_grad_block(
        float (*block_left_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], float* weight_grad,
        float* bias_grad, const float* output_grad, const float* input, size_t input_rows, size_t in_features,
        size_t out_features, size_t block_row, size_t block_column) {
    size_t block_first_row = block_row * LINEAR_VALUES_PER_BLOCK_ROWS;             // its first r: an output column j
    size_t block_first_column = block_column * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // its first c: an input column i
    // Its threads as the forward's: 16 x 16, numbered across first, each making 8 x 8 of the 128 x 128
    const int threads_across = LINEAR_VALUES_PER_BLOCK_COLUMNS / LINEAR_VALUES_PER_THREAD_COLUMNS;   // 128 / 8 = 16
    // its first r inside the block's 128 x 128 (0, 8, .. 120), and its first c: its 8 columns c are two
    // halves of 4, at 0, 4, .. 60 and 64 on from there (see LINEAR_SECOND_HALF_COLUMN_OFFSET)
    int thread_first_row = (threadIdx.x / threads_across) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_column = (threadIdx.x % threads_across) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);
    bool adds_bias_grad = bias_grad != nullptr && block_column == 0;                  // B1, see above

    // its 64 values of dW, in registers
    float sums[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};
    float bias_sums[LINEAR_VALUES_PER_THREAD_ROWS] = {};                   // with adds_bias_grad: db of its 8 rows r
    // k is t: each step sums over 8 input rows. dy as {k, r}, and x as {k, c}: both copied as they sit
    size_t total_steps = (input_rows + LINEAR_VALUES_PER_THREAD - 1) / LINEAR_VALUES_PER_THREAD;
    // its float4s of dy and x for the step being copied, read from GPU memory into registers: step 0's first
    float4 left4 = linear_load_piece(output_grad, input_rows, out_features, 0, block_first_row);
    float4 right4 = linear_load_piece(input, input_rows, in_features, 0, block_first_column);
    for (size_t step = 0; step < total_steps; step++) {
        int piece_set = (int)(step % 2);    // the set of pieces this step reads; the next step fills the other
        linear_store_piece(block_left_pieces[piece_set], left4);
        linear_store_piece(block_right_pieces[piece_set], right4);
        __syncthreads();            // both pieces whole, and the other set free: see the forward's loop
        // the next step's float4s, asked for now: GPU memory's wait passes during the math below
        size_t next_step_first_input_row = (step + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_input_row < input_rows) {
            left4 = linear_load_piece(output_grad, input_rows, out_features, next_step_first_input_row,
                                      block_first_row);
            right4 = linear_load_piece(input, input_rows, in_features, next_step_first_input_row,
                                       block_first_column);
        }
        // and B1: with adds_bias_grad, bias_sums gets left's values as they are read
        linear_add_outer_products(sums, block_left_pieces[piece_set], block_right_pieces[piece_set],
                                  thread_first_row, thread_first_column, bias_sums, adds_bias_grad);
    }

    linear_store_sums(weight_grad, out_features, in_features, sums, block_first_row + thread_first_row,
                      block_first_column + thread_first_column, true);
    // B1: db += bias_sums, one thread per 8 output columns j: those with thread_first_column 0
    if (adds_bias_grad && thread_first_column == 0) {
#pragma unroll
        for (int r = 0; r < LINEAR_VALUES_PER_THREAD_ROWS; r++) {
            size_t row = block_first_row + thread_first_row + r;                  // a row r of dW: output column j
            if (row < out_features) bias_grad[row] += bias_sums[r];
        }
    }
}

/// Makes one 128 x 128 square of input_grad, B3: dx = dy W, written over (see linear_backward). r is a row t,
/// c an input column i, and the sum runs over the n_out output columns j. dy sits as {T, n_out}, which is
/// {r, k}: it is copied turned round, as the forward copies x.
__device__ __forceinline__ void linear_input_grad_block(
        float (*block_left_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], float* input_grad,
        const float* output_grad, const float* weight, size_t input_rows, size_t in_features, size_t out_features,
        size_t block_row, size_t block_column) {
    size_t block_first_row = block_row * LINEAR_VALUES_PER_BLOCK_ROWS;             // its first r: a row t
    size_t block_first_column = block_column * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // its first c: an input column i
    // Its threads as the forward's: 16 x 16, numbered across first, each making 8 x 8 of the 128 x 128
    const int threads_across = LINEAR_VALUES_PER_BLOCK_COLUMNS / LINEAR_VALUES_PER_THREAD_COLUMNS;   // 128 / 8 = 16
    // its first r inside the block's 128 x 128 (0, 8, .. 120), and its first c: its 8 columns c are two
    // halves of 4, at 0, 4, .. 60 and 64 on from there (see LINEAR_SECOND_HALF_COLUMN_OFFSET)
    int thread_first_row = (threadIdx.x / threads_across) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_column = (threadIdx.x % threads_across) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);

    // its 64 values of dx, in registers
    float sums[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};
    float no_bias_sums[LINEAR_VALUES_PER_THREAD_ROWS];                   // never written: dx has no bias part
    // k is j: each step sums over 8 output columns. dy as {r, k}, copied turned round; W as {k, c}, as it sits
    size_t total_steps = (out_features + LINEAR_VALUES_PER_THREAD - 1) / LINEAR_VALUES_PER_THREAD;
    // its float4s of dy and W for the step being copied, read from GPU memory into registers: step 0's first
    float4 left4 = linear_load_piece_turned(output_grad, input_rows, out_features, block_first_row, 0);
    float4 right4 = linear_load_piece(weight, out_features, in_features, 0, block_first_column);
    for (size_t step = 0; step < total_steps; step++) {
        int piece_set = (int)(step % 2);    // the set of pieces this step reads; the next step fills the other
        linear_store_piece_turned(block_left_pieces[piece_set], left4);
        linear_store_piece(block_right_pieces[piece_set], right4);
        __syncthreads();            // both pieces whole, and the other set free: see the forward's loop
        // the next step's float4s, asked for now: GPU memory's wait passes during the math below
        size_t next_step_first_output_column = (step + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_output_column < out_features) {
            left4 = linear_load_piece_turned(output_grad, input_rows, out_features, block_first_row,
                                             next_step_first_output_column);
            right4 = linear_load_piece(weight, out_features, in_features, next_step_first_output_column,
                                       block_first_column);
        }
        linear_add_outer_products(sums, block_left_pieces[piece_set], block_right_pieces[piece_set],
                                  thread_first_row, thread_first_column, no_bias_sums, false);
    }

    linear_store_sums(input_grad, input_rows, in_features, sums, block_first_row + thread_first_row,
                      block_first_column + thread_first_column, false);
}

/// Reads, for linear_store_piece, this thread's float4 of one step's piece of a matrix that sits with k first,
/// {k, c}: matrix[step_first_k + k][block_first_column + c .. c + 3], for the step's 8 values of k and the block's
/// 128 columns c; 0 past the matrix's edges, so a part-full square adds nothing. matrix is {matrix_rows,
/// matrix_columns}. The 256 threads read one float4 each, 4 neighbouring c at one k: 32 threads per k, a warp
/// reading 512 bytes in a row. It only reads, into a register: the value is not needed until linear_store_piece,
/// so the thread can do other work while GPU memory answers.
__device__ __forceinline__ float4 linear_load_piece(const float* matrix, size_t matrix_rows, size_t matrix_columns,
                                                    size_t step_first_k, size_t block_first_column) {
    const int max_floats_per_thread_read = 4;   // a float4: 16 bytes, the widest read one thread can do
    // a row of block_piece is 128 columns c: 128 / 4 = 32 threads per row, one warp, 8 rows
    const int threads_per_block_piece_row = LINEAR_VALUES_PER_BLOCK_COLUMNS / max_floats_per_thread_read;
    // its float4: matrix[step_first_k + k][block_first_column + c .. c + 3]
    int k = threadIdx.x / threads_per_block_piece_row;
    int c = (threadIdx.x % threads_per_block_piece_row) * max_floats_per_thread_read;
    size_t matrix_row = step_first_k + k, matrix_column = block_first_column + c;
    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (matrix_row < matrix_rows && matrix_column < matrix_columns)
        values = *(const float4*)(matrix + matrix_row * matrix_columns + matrix_column);
    return values;
}

/// Writes the float4 linear_load_piece read into block_piece as it sat: block_piece[k][c .. c + 3], the same k and
/// c the thread read, one float4 write.
__device__ __forceinline__ void linear_store_piece(float (*block_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH], float4 values) {
    const int max_floats_per_thread_write = 4;   // a float4: 16 bytes, the widest write one thread can do
    const int threads_per_block_piece_row = LINEAR_VALUES_PER_BLOCK_COLUMNS / max_floats_per_thread_write;   // 32
    int k = threadIdx.x / threads_per_block_piece_row;
    int c = (threadIdx.x % threads_per_block_piece_row) * max_floats_per_thread_write;
    *(float4*)&block_piece[k][c] = values;
}

/// Reads, for linear_store_piece_turned, this thread's float4 of one step's piece of a matrix that sits with k
/// last, {r, k}: matrix[block_first_row + r][step_first_k + k .. k + 3], for the block's 128 rows r and the step's
/// 8 values of k; 0 past the matrix's edges. matrix is {matrix_rows, matrix_columns}. Each thread reads 4
/// neighbouring k of one r (2 threads per r), into a register, as linear_load_piece does.
__device__ __forceinline__ float4 linear_load_piece_turned(const float* matrix, size_t matrix_rows,
                                                           size_t matrix_columns, size_t block_first_row,
                                                           size_t step_first_k) {
    const int max_floats_per_thread_read = 4;   // a float4: 16 bytes, the widest read one thread can do
    const int threads_per_line = LINEAR_VALUES_PER_THREAD / max_floats_per_thread_read;   // 8 values of k / 4 = 2
    // its float4: matrix row r, k .. k + 3
    int r = threadIdx.x / threads_per_line;
    int k = (threadIdx.x % threads_per_line) * max_floats_per_thread_read;
    size_t matrix_row = block_first_row + r, matrix_column = step_first_k + k;
    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (matrix_row < matrix_rows && matrix_column < matrix_columns)
        values = *(const float4*)(matrix + matrix_row * matrix_columns + matrix_column);
    return values;
}

/// Writes the float4 linear_load_piece_turned read into block_piece turned round: block_piece[k .. k + 3][r],
/// its 4 values down a column of the piece, one at a time (4 neighbouring k are 4 rows of the piece).
__device__ __forceinline__ void linear_store_piece_turned(float (*block_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
                                                          float4 values) {
    const int max_floats_per_thread_read = 4;   // the float4 it read: 4 values of k
    const int threads_per_line = LINEAR_VALUES_PER_THREAD / max_floats_per_thread_read;   // 2
    int r = threadIdx.x / threads_per_line;
    int k = (threadIdx.x % threads_per_line) * max_floats_per_thread_read;
    block_piece[k + 0][r] = values.x;
    block_piece[k + 1][r] = values.y;
    block_piece[k + 2][r] = values.z;
    block_piece[k + 3][r] = values.w;
}
/// Adds one step's 8 outer products into a thread's 64 sums, C1's sum over k for the step's 8 values of k:
/// sums[r][c] += left[r][k] · right[k][c], for its 8 rows r (thread_first_row .. + 7) and 8 columns c
/// (thread_first_column .. + 3 and the 4 that are 64 on) of the block's square. At each k it reads its 8
/// values of each piece, two float4 each, and multiplies every one by every one: 64 multiply-adds for 16
/// values read.
///
/// With adds_bias_grad (dW's blocks of square column 0), it also adds the 8 left values to bias_sums[r]:
/// B1's db, left's values alone, as if right were all 1s. They are already in registers here, so this costs
/// 8 adds per k and no reads. dx passes false, and as the function is inlined the compiler drops the adds.
__device__ __forceinline__ void linear_add_outer_products(
        float (&sums)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS],
        const float (*block_left_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        const float (*block_right_piece)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        int thread_first_row, int thread_first_column,
        float (&bias_sums)[LINEAR_VALUES_PER_THREAD_ROWS], bool adds_bias_grad) {
    const int max_floats_per_thread_read = 4;   // a float4: 16 bytes, the widest read one thread can do
#pragma unroll
    for (int k = 0; k < LINEAR_VALUES_PER_THREAD; k++) {
        // left[r][k] of its 8 rows r and right[k][c] of its 8 columns c: a float4 holds 4, so two halves each,
        // its rows 0..3 and 4..7, and its columns 0..3 and the 4 that are 64 on
        float4 left_first_half = *(const float4*)&block_left_piece[k][thread_first_row];
        float4 left_second_half = *(const float4*)&block_left_piece[k][thread_first_row + max_floats_per_thread_read];
        float4 right_first_half = *(const float4*)&block_right_piece[k][thread_first_column];
        float4 right_second_half =
            *(const float4*)&block_right_piece[k][thread_first_column + LINEAR_SECOND_HALF_COLUMN_OFFSET];
        float left_values[LINEAR_VALUES_PER_THREAD_ROWS] = {
            left_first_half.x,  left_first_half.y,  left_first_half.z,  left_first_half.w,
            left_second_half.x, left_second_half.y, left_second_half.z, left_second_half.w};
        float right_values[LINEAR_VALUES_PER_THREAD_COLUMNS] = {
            right_first_half.x,  right_first_half.y,  right_first_half.z,  right_first_half.w,
            right_second_half.x, right_second_half.y, right_second_half.z, right_second_half.w};
#pragma unroll
        for (int r = 0; r < LINEAR_VALUES_PER_THREAD_ROWS; r++) {
#pragma unroll
            for (int c = 0; c < LINEAR_VALUES_PER_THREAD_COLUMNS; c++) sums[r][c] += left_values[r] * right_values[c];
            if (adds_bias_grad) bias_sums[r] += left_values[r];
        }
    }
}

/// Writes a thread's 64 sums into product, {product_rows, product_columns}, at rows first_row .. + 7 and
/// columns first_column .. + 3 and the 4 that are 64 on: added onto what is there with add_onto (dW), written
/// over without (dx). One float4 (4 columns) at a time; past the edges of a part-full square, nothing.
__device__ __forceinline__ void linear_store_sums(
        float* product, size_t product_rows, size_t product_columns,
        const float (&sums)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS], size_t first_row,
        size_t first_column, bool add_onto) {
    const int max_floats_per_thread_write = 4;   // a float4: 16 bytes, the widest write one thread can do
#pragma unroll
    for (int r = 0; r < LINEAR_VALUES_PER_THREAD_ROWS; r++) {
        size_t row = first_row + r;
#pragma unroll
        for (int c = 0; c < LINEAR_VALUES_PER_THREAD_COLUMNS; c += max_floats_per_thread_write) {
            // sums c .. c + 3: its first half (c = 0) or its second, 64 columns on (c = 4)
            int half = c / max_floats_per_thread_write;
            size_t column = first_column + half * LINEAR_SECOND_HALF_COLUMN_OFFSET;
            if (row >= product_rows || column >= product_columns) continue;   // past the edge of a part-full square
            float4 result = make_float4(sums[r][c], sums[r][c + 1], sums[r][c + 2], sums[r][c + 3]);
            float4* product4 = (float4*)(product + row * product_columns + column);
            if (add_onto) {
                float4 old = *product4;
                result = make_float4(old.x + result.x, old.y + result.y, old.z + result.z, old.w + result.w);
            }
            *product4 = result;
        }
    }
}

#endif // LINEAR_CUH
