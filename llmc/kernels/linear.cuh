// linear.cuh — a linear layer, as PyTorch's nn.Linear:
//     forward:    output = input · weightᵀ + bias
//     backward:   bias_grad   += sum over rows of output_grad      db = dyᵀ · 1
//                 weight_grad += output_gradᵀ · input              dW = dyᵀ x
//                 input_grad   = output_grad · weight              dx = dy W
//
//     input    {batch_size, seq_len, in_features}    {4, 1024, 768}
//     weight   {out_features, in_features}           {2304, 768}
//     bias     {out_features}                        {2304}
//     output   {batch_size, seq_len, out_features}   {4, 1024, 2304}
// Each block makes one LINEAR_VALUES_PER_BLOCK_ROWS x LINEAR_VALUES_PER_BLOCK_COLUMNS square of a product; each
// thread LINEAR_VALUES_PER_THREAD_ROWS x LINEAR_VALUES_PER_THREAD_COLUMNS of it. The backward kernels make
// product[r][c] = sum over k of left[r][k] · right[k][c]. All float or all bfloat16 (T); the math in float.
// The block size is fixed by the square (one thread per LINEAR_VALUES_PER_THREAD_ROWS x ..._COLUMNS of it), so
// these launchers take no block size.
// Math: https://ducbachsong.github.io/posts/linear-backward
#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "cuda_common.h"             // cudaCheck only
#include "../tensor.cuh"

// The kernels' index math is written for these numbers: change them together with it.
#define LINEAR_VALUES_PER_BLOCK_ROWS 128   // each block makes 128 rows x 128 columns of a product
#define LINEAR_VALUES_PER_BLOCK_COLUMNS 128
#define LINEAR_VALUES_PER_THREAD_ROWS 8    // each thread makes 8 rows x 8 columns of them
#define LINEAR_VALUES_PER_THREAD_COLUMNS 8
#define LINEAR_THREADS_ACROSS (LINEAR_VALUES_PER_BLOCK_COLUMNS / LINEAR_VALUES_PER_THREAD_COLUMNS)   // threads across a square
#define LINEAR_THREADS_DOWN (LINEAR_VALUES_PER_BLOCK_ROWS / LINEAR_VALUES_PER_THREAD_ROWS)           // threads down a square
#define LINEAR_THREADS_PER_BLOCK (LINEAR_THREADS_ACROSS * LINEAR_THREADS_DOWN)                      // one per thread's part
#define LINEAR_VALUES_PER_THREAD 8         // values of the sum (k) each step copies and adds
static_assert(LINEAR_VALUES_PER_BLOCK_ROWS == LINEAR_VALUES_PER_BLOCK_COLUMNS, "the copy is written for a square");
static_assert(LINEAR_THREADS_PER_BLOCK % THREADS_PER_WARP == 0, "a block must be whole warps");
// A piece row has one float4 more than the square's columns: the spare floats move each row that many banks on,
// so the turned copies' two half-warps hit different banks of shared memory.
#define LINEAR_BLOCK_PIECE_ROW_LENGTH (LINEAR_VALUES_PER_BLOCK_COLUMNS + VALUES_PER_FLOAT4)
static_assert(LINEAR_BLOCK_PIECE_ROW_LENGTH % VALUES_PER_FLOAT4 == 0, "each piece row must start on a float4, for the float4s");
// A thread's columns are two halves of a float4 each, half a square apart (thread 0: 0..3 and 64..67): a warp
// then reads neighbouring floats of shared memory, with no repeated banks.
#define LINEAR_SECOND_HALF_COLUMN_OFFSET (LINEAR_VALUES_PER_BLOCK_COLUMNS / 2)
static_assert(LINEAR_VALUES_PER_THREAD_COLUMNS == 2 * VALUES_PER_FLOAT4, "a thread's columns are two float4s");
static_assert(LINEAR_THREADS_ACROSS * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2) == LINEAR_SECOND_HALF_COLUMN_OFFSET,
              "the threads across x half their columns must fill the first half exactly");
// The forward numbers its blocks in groups of 8 rows of squares, down first: blocks running at once share
// bands of input and weight in L2, instead of reading all of weight once per row of squares.
#define LINEAR_BLOCK_ROWS_PER_GROUP 8

// ── forward and backward ──────────────────────────────────────────────────────
void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                    const Tensor& bias_tensor);
void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_forward_kernel(T* output_ptr, const T* input_ptr, const T* weight_ptr, const T* bias_ptr, size_t input_rows_val,
                      size_t in_features_val, size_t out_features_val);
template <typename T>
__device__ __forceinline__ float4 linear_load_piece(const T* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                                    size_t step_first_k_val, size_t block_first_column_val);
__device__ __forceinline__ void linear_store_piece(float (*block_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH], float4 values_val);
template <typename T>
__device__ __forceinline__ float4 linear_load_piece_turned(const T* matrix_ptr, size_t matrix_rows_val,
                                                           size_t matrix_columns_val, size_t block_first_row_val,
                                                           size_t step_first_k_val);
__device__ __forceinline__ void linear_store_piece_turned(float (*block_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
                                                          float4 values_val);

/// Writes output = input · weightᵀ + bias: checks, then launches linear_forward_kernel, one block per square
/// of output. in_features and out_features must be multiples of VALUES_PER_FLOAT4.
///
///   output_tensor   {..., out_features}
///   input_tensor    {..., in_features}: {batch_size, seq_len, in_features} in GPT-2
///   weight_tensor   {out_features, in_features}
///   bias_tensor     {out_features}, or Tensor() for none (the head)
inline void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                           const Tensor& bias_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == weight_tensor.size(0) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer needs input {..., in_features}, output {..., out_features} with the "
                        "same rows, weight {out_features, in_features}, bias {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    DType dtype_val = input_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || output_tensor.dtype() != dtype_val ||
        weight_tensor.dtype() != dtype_val || (has_bias_val && bias_tensor.dtype() != dtype_val)) {
        fprintf(stderr, "error: a linear layer needs output, input, weight and bias all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * input_tensor.element_size();
    if (weight_tensor.size(0) % VALUES_PER_FLOAT4 != 0 || weight_tensor.size(1) % VALUES_PER_FLOAT4 != 0 ||
        (size_t)output_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)input_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)weight_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)bias_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: a linear layer reads %d values at a time, so in_features and out_features must be "
                        "multiples of %d and output, input, weight, bias must start on %zu bytes; got in_features "
                        "%zu, out_features %zu, %p, %p, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, weight_tensor.size(1), weight_tensor.size(0),
                output_tensor.data_ptr(), input_tensor.data_ptr(), weight_tensor.data_ptr(), bias_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }

    // 2. launch: squares across and down, rounded up, as one line of blocks
    size_t out_features_val = weight_tensor.size(0);
    size_t in_features_val = weight_tensor.size(1);
    size_t input_rows_val = input_tensor.numel() / in_features_val;
    size_t total_block_columns_val = CEIL_DIV(out_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(input_rows_val, (size_t)LINEAR_VALUES_PER_BLOCK_ROWS);
    unsigned total_blocks_val = (unsigned)(total_block_columns_val * total_block_rows_val);
    if (dtype_val == DType::BFloat16)
        linear_forward_kernel<<<total_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            output_tensor.data_ptr<nv_bfloat16>(), input_tensor.data_ptr<nv_bfloat16>(),
            weight_tensor.data_ptr<nv_bfloat16>(), has_bias_val ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr,
            input_rows_val, in_features_val, out_features_val);
    else
        linear_forward_kernel<<<total_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            output_tensor.data(), input_tensor.data(), weight_tensor.data(), has_bias_val ? bias_tensor.data() : nullptr,
            input_rows_val, in_features_val, out_features_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output. The block walks the sum over in_features LINEAR_VALUES_PER_THREAD at a time:
/// it copies those columns of its rows of input and weight into shared memory, then each thread adds their
/// outer products into its sums in registers. Double-buffered: the next step's values are read from GPU memory
/// during this step's math. __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2): at most 128 registers, so 2 blocks
/// fit an SM.
///
///   output_ptr, input_ptr, weight_ptr, bias_ptr   as in linear_forward, float or bfloat16 (T); bias nullptr for none
///   input_rows_val                                batch_size x seq_len
///   in_features_val, out_features_val             weight's shape
template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_forward_kernel(T* output_ptr, const T* input_ptr, const T* weight_ptr, const T* bias_ptr, size_t input_rows_val,
                      size_t in_features_val, size_t out_features_val) {
    // this step's columns i of input and weight, i as the rows: [set][i][t] = x[t][i], [set][i][j] = W[j][i]
    __shared__ __align__(16) float block_input_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_weight_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    // 1. the block's square, from its number: groups of LINEAR_BLOCK_ROWS_PER_GROUP rows of squares, down a group first
    size_t total_block_columns_val = CEIL_DIV(out_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(input_rows_val, (size_t)LINEAR_VALUES_PER_BLOCK_ROWS);
    size_t blocks_per_group_val = LINEAR_BLOCK_ROWS_PER_GROUP * total_block_columns_val;
    size_t group_index_val = blockIdx.x / blocks_per_group_val;
    size_t group_first_block_row_val = group_index_val * LINEAR_BLOCK_ROWS_PER_GROUP;
    size_t group_block_rows_val = total_block_rows_val - group_first_block_row_val;   // fewer in the last group
    if (group_block_rows_val > LINEAR_BLOCK_ROWS_PER_GROUP) group_block_rows_val = LINEAR_BLOCK_ROWS_PER_GROUP;
    size_t block_index_in_group_val = blockIdx.x % blocks_per_group_val;
    size_t block_row_val = group_first_block_row_val + block_index_in_group_val % group_block_rows_val;
    size_t block_column_val = block_index_in_group_val / group_block_rows_val;
    size_t block_first_row_val = block_row_val * LINEAR_VALUES_PER_BLOCK_ROWS;                    // the block's first t
    size_t block_first_output_column_val = block_column_val * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // the block's first j
    // the thread's part inside the square: LINEAR_THREADS_ACROSS x LINEAR_THREADS_DOWN threads, across first
    int thread_first_row_val = (threadIdx.x / LINEAR_THREADS_ACROSS) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_output_column_val = (threadIdx.x % LINEAR_THREADS_ACROSS) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);

    // 2. walk the sum: copy each step's pieces, read the next step's, add the outer products
    float sums_list[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};   // its values of y
    size_t total_steps_val = CEIL_DIV(in_features_val, (size_t)LINEAR_VALUES_PER_THREAD);
    float4 input_val = linear_load_piece_turned(input_ptr, input_rows_val, in_features_val, block_first_row_val, 0);
    float4 weight_val = linear_load_piece_turned(weight_ptr, out_features_val, in_features_val, block_first_output_column_val, 0);
    for (size_t step_val = 0; step_val < total_steps_val; step_val++) {
        int piece_set_val = (int)(step_val % 2);    // this step reads one set; the next step fills the other
        linear_store_piece_turned(block_input_pieces_list[piece_set_val], input_val);
        linear_store_piece_turned(block_weight_pieces_list[piece_set_val], weight_val);
        __syncthreads();                            // the pieces whole, and the other set free
        // the next step's values, read now: they arrive during the math below
        size_t next_step_first_input_column_val = (step_val + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_input_column_val < in_features_val) {
            input_val = linear_load_piece_turned(input_ptr, input_rows_val, in_features_val, block_first_row_val,
                                                 next_step_first_input_column_val);
            weight_val = linear_load_piece_turned(weight_ptr, out_features_val, in_features_val,
                                                  block_first_output_column_val, next_step_first_input_column_val);
        }

        // sums[t][j] += x[t][i] · W[j][i] for this step's values of i
        const float (*block_input_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH] = block_input_pieces_list[piece_set_val];
        const float (*block_weight_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH] = block_weight_pieces_list[piece_set_val];
#pragma unroll
        for (int i_val = 0; i_val < LINEAR_VALUES_PER_THREAD; i_val++) {
            float4 input_first_half_val = *reinterpret_cast<const float4*>(&block_input_piece_ptr[i_val][thread_first_row_val]);
            float4 input_second_half_val = *reinterpret_cast<const float4*>(
                &block_input_piece_ptr[i_val][thread_first_row_val + VALUES_PER_FLOAT4]);
            float4 weight_first_half_val =
                *reinterpret_cast<const float4*>(&block_weight_piece_ptr[i_val][thread_first_output_column_val]);
            float4 weight_second_half_val = *reinterpret_cast<const float4*>(
                &block_weight_piece_ptr[i_val][thread_first_output_column_val + LINEAR_SECOND_HALF_COLUMN_OFFSET]);
            float input_list[LINEAR_VALUES_PER_THREAD_ROWS] = {
                input_first_half_val.x,  input_first_half_val.y,  input_first_half_val.z,  input_first_half_val.w,
                input_second_half_val.x, input_second_half_val.y, input_second_half_val.z, input_second_half_val.w};
            float weight_list[LINEAR_VALUES_PER_THREAD_COLUMNS] = {
                weight_first_half_val.x,  weight_first_half_val.y,  weight_first_half_val.z,  weight_first_half_val.w,
                weight_second_half_val.x, weight_second_half_val.y, weight_second_half_val.z, weight_second_half_val.w};
#pragma unroll
            for (int t_val = 0; t_val < LINEAR_VALUES_PER_THREAD_ROWS; t_val++) {
#pragma unroll
                for (int j_val = 0; j_val < LINEAR_VALUES_PER_THREAD_COLUMNS; j_val++)
                    sums_list[t_val][j_val] += input_list[t_val] * weight_list[j_val];
            }
        }
    }

    // 3. y = sums + bias, one float4 at a time; nothing past the edges
#pragma unroll
    for (int t_val = 0; t_val < LINEAR_VALUES_PER_THREAD_ROWS; t_val++) {
        size_t row_val = block_first_row_val + thread_first_row_val + t_val;
#pragma unroll
        for (int j_val = 0; j_val < LINEAR_VALUES_PER_THREAD_COLUMNS; j_val += VALUES_PER_FLOAT4) {
            int half_val = j_val / VALUES_PER_FLOAT4;   // its first float4 of columns, or the one half a square on
            size_t output_column_val =
                block_first_output_column_val + thread_first_output_column_val + half_val * LINEAR_SECOND_HALF_COLUMN_OFFSET;
            if (row_val >= input_rows_val || output_column_val >= out_features_val) continue;
            float4 result_val = make_float4(sums_list[t_val][j_val], sums_list[t_val][j_val + 1],
                                            sums_list[t_val][j_val + 2], sums_list[t_val][j_val + 3]);
            if (bias_ptr != nullptr) {
                float4 bias_val = read4_as_float(bias_ptr + output_column_val);
                result_val = make_float4(result_val.x + bias_val.x, result_val.y + bias_val.y,
                                         result_val.z + bias_val.z, result_val.w + bias_val.w);
            }
            write4_from_float(output_ptr + row_val * out_features_val + output_column_val, result_val);
        }
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_weight_grad_kernel(T* weight_grad_ptr, T* bias_grad_ptr, const T* output_grad_ptr, const T* input_ptr,
                          size_t input_rows_val, size_t in_features_val, size_t out_features_val);
template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_input_grad_kernel(T* input_grad_ptr, const T* output_grad_ptr, const T* weight_ptr, size_t input_rows_val,
                         size_t in_features_val, size_t out_features_val);

/// Works out the gradients of input, weight and bias: checks, then launches linear_weight_grad_kernel and
/// linear_input_grad_kernel. Two kernels so each gets the registers its own product needs. input_grad is
/// written over; weight_grad and bias_grad are added onto, so they must be 0 before a step's first backward.
///
///   input_grad_tensor              input's shape
///   weight_grad_tensor             weight's shape
///   bias_grad_tensor               {out_features}, or Tensor() for none
///   output_grad_tensor             the gradient from the layer above
///   input_tensor, weight_tensor    the forward's
inline void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                            const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight, each grad its tensor's shape
    bool has_bias_val = bias_grad_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_grad_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_grad_tensor.size(output_grad_tensor.dim() - 1) == weight_tensor.size(0) &&
                    input_grad_tensor.shape() == input_tensor.shape() && weight_grad_tensor.shape() == weight_tensor.shape() &&
                    (!has_bias_val || bias_grad_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_grad_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer's backward needs input and input_grad {..., in_features}, output_grad "
                        "{..., out_features} with the same rows, weight and weight_grad {out_features, in_features}, "
                        "bias_grad {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    DType dtype_val = input_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || input_grad_tensor.dtype() != dtype_val ||
        weight_grad_tensor.dtype() != dtype_val || output_grad_tensor.dtype() != dtype_val ||
        weight_tensor.dtype() != dtype_val || (has_bias_val && bias_grad_tensor.dtype() != dtype_val)) {
        fprintf(stderr, "error: a linear layer's backward needs its tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // counts and addresses: whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * input_tensor.element_size();
    if (weight_tensor.size(0) % VALUES_PER_FLOAT4 != 0 || weight_tensor.size(1) % VALUES_PER_FLOAT4 != 0 ||
        (size_t)input_grad_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)weight_grad_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)bias_grad_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)output_grad_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)input_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)weight_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: a linear layer reads %d values at a time, so in_features and out_features must be "
                        "multiples of %d and input_grad, weight_grad, bias_grad, output_grad, input, weight must start "
                        "on %zu bytes; got in_features %zu, out_features %zu, %p, %p, %p, %p, %p, %p\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, weight_tensor.size(1), weight_tensor.size(0),
                input_grad_tensor.data_ptr(), weight_grad_tensor.data_ptr(), bias_grad_tensor.data_ptr(),
                output_grad_tensor.data_ptr(), input_tensor.data_ptr(), weight_tensor.data_ptr());
        exit(EXIT_FAILURE);
    }

    // 2. launch: one block per square of each product, rounded up; the numbers are the qkv's
    size_t out_features_val = weight_tensor.size(0);
    size_t in_features_val = weight_tensor.size(1);
    size_t input_rows_val = input_tensor.numel() / in_features_val;
    size_t total_weight_grad_block_rows_val = CEIL_DIV(out_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_ROWS);       // rows j: 18
    size_t total_weight_grad_block_columns_val = CEIL_DIV(in_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);  // columns i: 6
    size_t total_input_grad_block_rows_val = CEIL_DIV(input_rows_val, (size_t)LINEAR_VALUES_PER_BLOCK_ROWS);          // rows t: 32
    size_t total_input_grad_block_columns_val = CEIL_DIV(in_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);   // columns i: 6
    unsigned total_weight_grad_blocks_val = (unsigned)(total_weight_grad_block_rows_val * total_weight_grad_block_columns_val);
    unsigned total_input_grad_blocks_val = (unsigned)(total_input_grad_block_rows_val * total_input_grad_block_columns_val);
    if (dtype_val == DType::BFloat16) {
        linear_weight_grad_kernel<<<total_weight_grad_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            weight_grad_tensor.data_ptr<nv_bfloat16>(), has_bias_val ? bias_grad_tensor.data_ptr<nv_bfloat16>() : nullptr,
            output_grad_tensor.data_ptr<nv_bfloat16>(), input_tensor.data_ptr<nv_bfloat16>(), input_rows_val,
            in_features_val, out_features_val);
        cudaCheck(cudaGetLastError());
        linear_input_grad_kernel<<<total_input_grad_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            input_grad_tensor.data_ptr<nv_bfloat16>(), output_grad_tensor.data_ptr<nv_bfloat16>(),
            weight_tensor.data_ptr<nv_bfloat16>(), input_rows_val, in_features_val, out_features_val);
    } else {
        linear_weight_grad_kernel<<<total_weight_grad_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            weight_grad_tensor.data(), has_bias_val ? bias_grad_tensor.data() : nullptr, output_grad_tensor.data(),
            input_tensor.data(), input_rows_val, in_features_val, out_features_val);
        cudaCheck(cudaGetLastError());
        linear_input_grad_kernel<<<total_input_grad_blocks_val, LINEAR_THREADS_PER_BLOCK>>>(
            input_grad_tensor.data(), output_grad_tensor.data(), weight_tensor.data(), input_rows_val, in_features_val,
            out_features_val);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ void linear_weight_grad_block(
        float (*block_left_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], T* weight_grad_ptr,
        T* bias_grad_ptr, const T* output_grad_ptr, const T* input_ptr, size_t input_rows_val, size_t in_features_val,
        size_t out_features_val, size_t block_row_val, size_t block_column_val);
template <typename T>
__device__ __forceinline__ void linear_input_grad_block(
        float (*block_left_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], T* input_grad_ptr,
        const T* output_grad_ptr, const T* weight_ptr, size_t input_rows_val, size_t in_features_val,
        size_t out_features_val, size_t block_row_val, size_t block_column_val);
__device__ __forceinline__ void linear_add_outer_products(
        float (&sums_list)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS],
        const float (*block_left_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        const float (*block_right_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        int thread_first_row_val, int thread_first_column_val,
        float (&bias_sums_list)[LINEAR_VALUES_PER_THREAD_ROWS], bool adds_bias_grad_val);
template <typename T>
__device__ __forceinline__ void linear_store_sums(
        T* product_ptr, size_t product_rows_val, size_t product_columns_val,
        const float (&sums_list)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS], size_t first_row_val,
        size_t first_column_val, bool add_onto_val);

// The two backward products, as product[r][c] = sum over k of left[r][k] · right[k][c]:
//     product                  r    c    k (summed)   left              right
//     dW = dyᵀ x   {out, in}   j    i    t: rows      dy, turned round  x {rows, in}
//     dx = dy W    {rows, in}  t    i    j: out       dy {rows, out}    W {out, in}

/// Works out one square of weight_grad, and of bias_grad with it (linear_weight_grad_block).
///
///   weight_grad_ptr, bias_grad_ptr, output_grad_ptr, input_ptr   as in linear_backward, float or bfloat16 (T)
///   input_rows_val                                               batch_size x seq_len
///   in_features_val, out_features_val                            weight's shape
template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_weight_grad_kernel(T* weight_grad_ptr, T* bias_grad_ptr, const T* output_grad_ptr, const T* input_ptr,
                          size_t input_rows_val, size_t in_features_val, size_t out_features_val) {
    // this step's values of k, as the rows: [set][k][r] = left[r][k], [set][k][c] = right[k][c]
    __shared__ __align__(16) float block_left_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_right_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    // the block's square, from its number
    size_t total_block_columns_val = CEIL_DIV(in_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);   // columns i: 6
    size_t block_row_val = blockIdx.x / total_block_columns_val;           // which output columns j
    size_t block_column_val = blockIdx.x % total_block_columns_val;        // which input columns i
    linear_weight_grad_block(block_left_pieces_list, block_right_pieces_list, weight_grad_ptr, bias_grad_ptr,
                             output_grad_ptr, input_ptr, input_rows_val, in_features_val, out_features_val, block_row_val,
                             block_column_val);
}

/// Works out one square of input_grad (linear_input_grad_block).
///
///   input_grad_ptr, output_grad_ptr, weight_ptr   as in linear_backward, float or bfloat16 (T)
///   input_rows_val                                batch_size x seq_len
///   in_features_val, out_features_val             weight's shape
template <typename T>
__global__ void __launch_bounds__(LINEAR_THREADS_PER_BLOCK, 2)
linear_input_grad_kernel(T* input_grad_ptr, const T* output_grad_ptr, const T* weight_ptr, size_t input_rows_val,
                         size_t in_features_val, size_t out_features_val) {
    __shared__ __align__(16) float block_left_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float block_right_pieces_list[2][LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH];
    // the block's square, from its number
    size_t total_block_columns_val = CEIL_DIV(in_features_val, (size_t)LINEAR_VALUES_PER_BLOCK_COLUMNS);   // columns i: 6
    size_t block_row_val = blockIdx.x / total_block_columns_val;           // which rows t
    size_t block_column_val = blockIdx.x % total_block_columns_val;        // which input columns i
    linear_input_grad_block(block_left_pieces_list, block_right_pieces_list, input_grad_ptr, output_grad_ptr, weight_ptr,
                            input_rows_val, in_features_val, out_features_val, block_row_val, block_column_val);
}

/// Adds one square of dW = dyᵀ x onto weight_grad. The blocks of square column 0 also add up dy's columns onto
/// bias_grad (db = dyᵀ · 1), from values they read anyway: one thread per value, in order, so no atomics and
/// the same result every run.
///
///   block_left_pieces_ptr, block_right_pieces_ptr   the kernel's shared memory
///   weight_grad_ptr, bias_grad_ptr, output_grad_ptr, input_ptr,
///   input_rows_val, in_features_val, out_features_val   as in the kernel
///   block_row_val, block_column_val                 the square: which output columns j, which input columns i
template <typename T>
__device__ __forceinline__ void linear_weight_grad_block(
        float (*block_left_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], T* weight_grad_ptr,
        T* bias_grad_ptr, const T* output_grad_ptr, const T* input_ptr, size_t input_rows_val, size_t in_features_val,
        size_t out_features_val, size_t block_row_val, size_t block_column_val) {
    // 1. where the block and this thread are
    size_t block_first_row_val = block_row_val * LINEAR_VALUES_PER_BLOCK_ROWS;             // its first output column j
    size_t block_first_column_val = block_column_val * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // its first input column i
    int thread_first_row_val = (threadIdx.x / LINEAR_THREADS_ACROSS) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_column_val = (threadIdx.x % LINEAR_THREADS_ACROSS) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);
    bool adds_bias_grad_val = bias_grad_ptr != nullptr && block_column_val == 0;

    // 2. walk the sum over k = t: dy and x both sit as {k, ...}, copied as they are
    float sums_list[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};   // its values of dW
    float bias_sums_list[LINEAR_VALUES_PER_THREAD_ROWS] = {};                                // db of its rows
    size_t total_steps_val = CEIL_DIV(input_rows_val, (size_t)LINEAR_VALUES_PER_THREAD);
    float4 left_val = linear_load_piece(output_grad_ptr, input_rows_val, out_features_val, 0, block_first_row_val);
    float4 right_val = linear_load_piece(input_ptr, input_rows_val, in_features_val, 0, block_first_column_val);
    for (size_t step_val = 0; step_val < total_steps_val; step_val++) {
        int piece_set_val = (int)(step_val % 2);
        linear_store_piece(block_left_pieces_ptr[piece_set_val], left_val);
        linear_store_piece(block_right_pieces_ptr[piece_set_val], right_val);
        __syncthreads();
        size_t next_step_first_input_row_val = (step_val + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_input_row_val < input_rows_val) {
            left_val = linear_load_piece(output_grad_ptr, input_rows_val, out_features_val, next_step_first_input_row_val,
                                         block_first_row_val);
            right_val = linear_load_piece(input_ptr, input_rows_val, in_features_val, next_step_first_input_row_val,
                                          block_first_column_val);
        }
        linear_add_outer_products(sums_list, block_left_pieces_ptr[piece_set_val], block_right_pieces_ptr[piece_set_val],
                                  thread_first_row_val, thread_first_column_val, bias_sums_list, adds_bias_grad_val);
    }

    // 3. dW += sums
    linear_store_sums(weight_grad_ptr, out_features_val, in_features_val, sums_list, block_first_row_val + thread_first_row_val,
                      block_first_column_val + thread_first_column_val, true);
    // 4. db += bias_sums: the threads of the first float4 of columns
    if (adds_bias_grad_val && thread_first_column_val == 0) {
#pragma unroll
        for (int r_val = 0; r_val < LINEAR_VALUES_PER_THREAD_ROWS; r_val++) {
            size_t row_val = block_first_row_val + thread_first_row_val + r_val;          // an output column j
            if (row_val < out_features_val)
                bias_grad_ptr[row_val] = from_float<T>(to_float(bias_grad_ptr[row_val]) + bias_sums_list[r_val]);
        }
    }
}

/// Writes one square of dx = dy W into input_grad (written over).
///
///   block_left_pieces_ptr, block_right_pieces_ptr   the kernel's shared memory
///   input_grad_ptr, output_grad_ptr, weight_ptr,
///   input_rows_val, in_features_val, out_features_val   as in the kernel
///   block_row_val, block_column_val                 the square: which rows t, which input columns i
template <typename T>
__device__ __forceinline__ void linear_input_grad_block(
        float (*block_left_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH],
        float (*block_right_pieces_ptr)[LINEAR_VALUES_PER_THREAD][LINEAR_BLOCK_PIECE_ROW_LENGTH], T* input_grad_ptr,
        const T* output_grad_ptr, const T* weight_ptr, size_t input_rows_val, size_t in_features_val,
        size_t out_features_val, size_t block_row_val, size_t block_column_val) {
    // 1. where the block and this thread are
    size_t block_first_row_val = block_row_val * LINEAR_VALUES_PER_BLOCK_ROWS;             // its first row t
    size_t block_first_column_val = block_column_val * LINEAR_VALUES_PER_BLOCK_COLUMNS;    // its first input column i
    int thread_first_row_val = (threadIdx.x / LINEAR_THREADS_ACROSS) * LINEAR_VALUES_PER_THREAD_ROWS;
    int thread_first_column_val = (threadIdx.x % LINEAR_THREADS_ACROSS) * (LINEAR_VALUES_PER_THREAD_COLUMNS / 2);

    // 2. walk the sum over k = j: dy sits as {r, k}, copied turned round; W as {k, c}, as it sits
    float sums_list[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS] = {};   // its values of dx
    float no_bias_sums_list[LINEAR_VALUES_PER_THREAD_ROWS];                                  // never written: dx has no bias part
    size_t total_steps_val = CEIL_DIV(out_features_val, (size_t)LINEAR_VALUES_PER_THREAD);
    float4 left_val = linear_load_piece_turned(output_grad_ptr, input_rows_val, out_features_val, block_first_row_val, 0);
    float4 right_val = linear_load_piece(weight_ptr, out_features_val, in_features_val, 0, block_first_column_val);
    for (size_t step_val = 0; step_val < total_steps_val; step_val++) {
        int piece_set_val = (int)(step_val % 2);
        linear_store_piece_turned(block_left_pieces_ptr[piece_set_val], left_val);
        linear_store_piece(block_right_pieces_ptr[piece_set_val], right_val);
        __syncthreads();
        size_t next_step_first_output_column_val = (step_val + 1) * LINEAR_VALUES_PER_THREAD;
        if (next_step_first_output_column_val < out_features_val) {
            left_val = linear_load_piece_turned(output_grad_ptr, input_rows_val, out_features_val, block_first_row_val,
                                                next_step_first_output_column_val);
            right_val = linear_load_piece(weight_ptr, out_features_val, in_features_val, next_step_first_output_column_val,
                                          block_first_column_val);
        }
        linear_add_outer_products(sums_list, block_left_pieces_ptr[piece_set_val], block_right_pieces_ptr[piece_set_val],
                                  thread_first_row_val, thread_first_column_val, no_bias_sums_list, false);
    }

    // 3. dx = sums
    linear_store_sums(input_grad_ptr, input_rows_val, in_features_val, sums_list, block_first_row_val + thread_first_row_val,
                      block_first_column_val + thread_first_column_val, false);
}

/// Gives this thread's float4 of one step's piece of a matrix sitting as {k, c}: VALUES_PER_FLOAT4 neighbouring
/// c at one k; 0 past the matrix's edges.
///
///   matrix_ptr                          {matrix_rows, matrix_columns}
///   matrix_rows_val, matrix_columns_val its shape
///   step_first_k_val                    the step's first k
///   block_first_column_val              the block's first c
template <typename T>
__device__ __forceinline__ float4 linear_load_piece(const T* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                                    size_t step_first_k_val, size_t block_first_column_val) {
    const int threads_per_block_piece_row_val = LINEAR_VALUES_PER_BLOCK_COLUMNS / VALUES_PER_FLOAT4;
    int k_val = threadIdx.x / threads_per_block_piece_row_val;
    int c_val = (threadIdx.x % threads_per_block_piece_row_val) * VALUES_PER_FLOAT4;
    size_t matrix_row_val = step_first_k_val + k_val;
    size_t matrix_column_val = block_first_column_val + c_val;
    float4 values_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (matrix_row_val < matrix_rows_val && matrix_column_val < matrix_columns_val)
        values_val = read4_as_float(matrix_ptr + matrix_row_val * matrix_columns_val + matrix_column_val);
    return values_val;
}

/// Writes linear_load_piece's float4 into the piece as it sat: block_piece[k][c .. c + 3].
///
///   block_piece_ptr   one set of the shared memory pieces
///   values_val        what linear_load_piece gave
__device__ __forceinline__ void linear_store_piece(float (*block_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH], float4 values_val) {
    const int threads_per_block_piece_row_val = LINEAR_VALUES_PER_BLOCK_COLUMNS / VALUES_PER_FLOAT4;
    int k_val = threadIdx.x / threads_per_block_piece_row_val;
    int c_val = (threadIdx.x % threads_per_block_piece_row_val) * VALUES_PER_FLOAT4;
    *reinterpret_cast<float4*>(&block_piece_ptr[k_val][c_val]) = values_val;
}

/// Gives this thread's float4 of one step's piece of a matrix sitting as {r, k}: VALUES_PER_FLOAT4 neighbouring
/// k of one r; 0 past the matrix's edges.
///
///   matrix_ptr                          {matrix_rows, matrix_columns}
///   matrix_rows_val, matrix_columns_val its shape
///   block_first_row_val                 the block's first r
///   step_first_k_val                    the step's first k
template <typename T>
__device__ __forceinline__ float4 linear_load_piece_turned(const T* matrix_ptr, size_t matrix_rows_val,
                                                           size_t matrix_columns_val, size_t block_first_row_val,
                                                           size_t step_first_k_val) {
    const int threads_per_line_val = LINEAR_VALUES_PER_THREAD / VALUES_PER_FLOAT4;   // threads per r
    int r_val = threadIdx.x / threads_per_line_val;
    int k_val = (threadIdx.x % threads_per_line_val) * VALUES_PER_FLOAT4;
    size_t matrix_row_val = block_first_row_val + r_val;
    size_t matrix_column_val = step_first_k_val + k_val;
    float4 values_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (matrix_row_val < matrix_rows_val && matrix_column_val < matrix_columns_val)
        values_val = read4_as_float(matrix_ptr + matrix_row_val * matrix_columns_val + matrix_column_val);
    return values_val;
}

/// Writes linear_load_piece_turned's float4 into the piece turned round: block_piece[k .. k + 3][r].
///
///   block_piece_ptr   one set of the shared memory pieces
///   values_val        what linear_load_piece_turned gave
__device__ __forceinline__ void linear_store_piece_turned(float (*block_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
                                                          float4 values_val) {
    const int threads_per_line_val = LINEAR_VALUES_PER_THREAD / VALUES_PER_FLOAT4;   // threads per r
    int r_val = threadIdx.x / threads_per_line_val;
    int k_val = (threadIdx.x % threads_per_line_val) * VALUES_PER_FLOAT4;
    block_piece_ptr[k_val + 0][r_val] = values_val.x;
    block_piece_ptr[k_val + 1][r_val] = values_val.y;
    block_piece_ptr[k_val + 2][r_val] = values_val.z;
    block_piece_ptr[k_val + 3][r_val] = values_val.w;
}

/// Adds one step's outer products into the thread's sums: sums[r][c] += left[r][k] · right[k][c].
/// LINEAR_VALUES_PER_THREAD_ROWS x ..._COLUMNS multiply-adds for ..._ROWS + ..._COLUMNS values read at each k.
///
///   sums_list                                       the thread's part
///   block_left_piece_ptr, block_right_piece_ptr     this step's pieces
///   thread_first_row_val, thread_first_column_val   the thread's place in the square
///   bias_sums_list, adds_bias_grad_val              with adds_bias_grad, also bias_sums[r] += left[r][k]
__device__ __forceinline__ void linear_add_outer_products(
        float (&sums_list)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS],
        const float (*block_left_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        const float (*block_right_piece_ptr)[LINEAR_BLOCK_PIECE_ROW_LENGTH],
        int thread_first_row_val, int thread_first_column_val,
        float (&bias_sums_list)[LINEAR_VALUES_PER_THREAD_ROWS], bool adds_bias_grad_val) {
#pragma unroll
    for (int k_val = 0; k_val < LINEAR_VALUES_PER_THREAD; k_val++) {
        // 1. this k's rows of left and columns of right, two float4s each
        float4 left_first_half_val = *reinterpret_cast<const float4*>(&block_left_piece_ptr[k_val][thread_first_row_val]);
        float4 left_second_half_val = *reinterpret_cast<const float4*>(
            &block_left_piece_ptr[k_val][thread_first_row_val + VALUES_PER_FLOAT4]);
        float4 right_first_half_val = *reinterpret_cast<const float4*>(&block_right_piece_ptr[k_val][thread_first_column_val]);
        float4 right_second_half_val = *reinterpret_cast<const float4*>(
            &block_right_piece_ptr[k_val][thread_first_column_val + LINEAR_SECOND_HALF_COLUMN_OFFSET]);
        float left_list[LINEAR_VALUES_PER_THREAD_ROWS] = {
            left_first_half_val.x,  left_first_half_val.y,  left_first_half_val.z,  left_first_half_val.w,
            left_second_half_val.x, left_second_half_val.y, left_second_half_val.z, left_second_half_val.w};
        float right_list[LINEAR_VALUES_PER_THREAD_COLUMNS] = {
            right_first_half_val.x,  right_first_half_val.y,  right_first_half_val.z,  right_first_half_val.w,
            right_second_half_val.x, right_second_half_val.y, right_second_half_val.z, right_second_half_val.w};
        // 2. their outer product into the sums
#pragma unroll
        for (int r_val = 0; r_val < LINEAR_VALUES_PER_THREAD_ROWS; r_val++) {
#pragma unroll
            for (int c_val = 0; c_val < LINEAR_VALUES_PER_THREAD_COLUMNS; c_val++)
                sums_list[r_val][c_val] += left_list[r_val] * right_list[c_val];
            if (adds_bias_grad_val) bias_sums_list[r_val] += left_list[r_val];
        }
    }
}

/// Writes the thread's sums into product, one float4 at a time; nothing past the edges.
///
///   product_ptr                              {product_rows, product_columns}: weight_grad or input_grad
///   product_rows_val, product_columns_val    its shape
///   sums_list                                the thread's part
///   first_row_val, first_column_val          where they go
///   add_onto_val                             true: added onto what is there (dW); false: written over (dx)
template <typename T>
__device__ __forceinline__ void linear_store_sums(
        T* product_ptr, size_t product_rows_val, size_t product_columns_val,
        const float (&sums_list)[LINEAR_VALUES_PER_THREAD_ROWS][LINEAR_VALUES_PER_THREAD_COLUMNS], size_t first_row_val,
        size_t first_column_val, bool add_onto_val) {
#pragma unroll
    for (int r_val = 0; r_val < LINEAR_VALUES_PER_THREAD_ROWS; r_val++) {
        size_t row_val = first_row_val + r_val;
#pragma unroll
        for (int c_val = 0; c_val < LINEAR_VALUES_PER_THREAD_COLUMNS; c_val += VALUES_PER_FLOAT4) {
            int half_val = c_val / VALUES_PER_FLOAT4;   // its first float4 of columns, or the one half a square on
            size_t column_val = first_column_val + half_val * LINEAR_SECOND_HALF_COLUMN_OFFSET;
            if (row_val >= product_rows_val || column_val >= product_columns_val) continue;   // past the edge
            float4 result_val = make_float4(sums_list[r_val][c_val], sums_list[r_val][c_val + 1],
                                            sums_list[r_val][c_val + 2], sums_list[r_val][c_val + 3]);
            T* product_float4_ptr = product_ptr + row_val * product_columns_val + column_val;
            if (add_onto_val) {
                float4 old_val = read4_as_float(product_float4_ptr);
                result_val = make_float4(old_val.x + result_val.x, old_val.y + result_val.y, old_val.z + result_val.z,
                                         old_val.w + result_val.w);
            }
            write4_from_float(product_float4_ptr, result_val);
        }
    }
}

#endif // LINEAR_CUH
