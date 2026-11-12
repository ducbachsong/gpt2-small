// fma_matmul_bf16.cuh — matrix products on the CUDA cores, one FMA (multiply-add) per thread per instruction:
//     output = left · rightᵀ (+ bias) (+ output)
// bfloat16, or float (the exact reference, linear.cuh's float path); the sums in float either way. Any GPU.
//
//     tensor   type in memory        shape                                       note
//     left     T (float or bf16)     {rows, sum_length}, turned {sum_length, rows}
//     right    T                     {columns, sum_length}, turned {sum_length, columns}
//     bias     T                     {columns}, or none
//     output   T                     {rows, columns}                            written over, or added onto
//
// turned: the tensor holds its matrix turned round. A linear layer needs all three ways:
//     forward       output      = input · weightᵀ          left input,              right weight
//     backward      input_grad  = output_grad · weight     left output_grad,        right weight turned
//                   weight_grad += output_gradᵀ · input    left output_grad turned, right input turned
//
// Each block makes one FMA_MATMUL_BF16_BLOCK_ROWS x FMA_MATMUL_BF16_BLOCK_COLUMNS square of output, each thread
// FMA_MATMUL_BF16_THREAD_ROWS x FMA_MATMUL_BF16_THREAD_COLUMNS of it. The block walks the sum
// FMA_MATMUL_BF16_STEP_K values at a time: it copies those values of its rows of left and of right into shared
// memory, k as the rows (left[r][k] at piece[k][r]), then each thread adds their outer products into its sums in
// registers: 8 x 8 multiply-adds for 8 + 8 values read at each k. Double-buffered: the next step's values are
// read from GPU memory during this step's math. The block size is fixed by the square, so this takes none.
// The same with tensor cores: mma_matmul_bf16.cuh (mma.sync), wgmma_matmul_bf16.cuh (WGMMA, H100).
#ifndef FMA_MATMUL_BF16_CUH
#define FMA_MATMUL_BF16_CUH

#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"             // read4_as_float, write4_from_float, VALUES_PER_FLOAT4

// The squares and steps: our choices. The kernels' index math is written for these numbers: change them together with it.
#define FMA_MATMUL_BF16_BLOCK_ROWS 128     // each block makes 128 rows x 128 columns of output
#define FMA_MATMUL_BF16_BLOCK_COLUMNS 128
#define FMA_MATMUL_BF16_THREAD_ROWS 8      // each thread makes 8 rows x 8 columns of them
#define FMA_MATMUL_BF16_THREAD_COLUMNS 8
#define FMA_MATMUL_BF16_THREADS_ACROSS (FMA_MATMUL_BF16_BLOCK_COLUMNS / FMA_MATMUL_BF16_THREAD_COLUMNS)   // threads across a square: 16
#define FMA_MATMUL_BF16_THREADS_DOWN (FMA_MATMUL_BF16_BLOCK_ROWS / FMA_MATMUL_BF16_THREAD_ROWS)           // threads down a square: 16
#define FMA_MATMUL_BF16_THREADS_PER_BLOCK (FMA_MATMUL_BF16_THREADS_ACROSS * FMA_MATMUL_BF16_THREADS_DOWN) // one per thread's part: 256
#define FMA_MATMUL_BF16_STEP_K 8           // values of the sum (k) each step copies and adds
static_assert(FMA_MATMUL_BF16_BLOCK_ROWS == FMA_MATMUL_BF16_BLOCK_COLUMNS, "the copies are written for a square");
static_assert(FMA_MATMUL_BF16_THREADS_PER_BLOCK % HARDWARE_THREADS_PER_WARP == 0, "a block must be whole warps");
static_assert(FMA_MATMUL_BF16_STEP_K * FMA_MATMUL_BF16_BLOCK_ROWS == FMA_MATMUL_BF16_THREADS_PER_BLOCK * VALUES_PER_FLOAT4,
              "each thread copies one float4 of each step's piece");
// A piece row has one float4 more than the square's rows: the spare floats move each row that many banks on, so
// the turned copies' writes spread over the banks of shared memory.
#define FMA_MATMUL_BF16_PIECE_ROW_LENGTH (FMA_MATMUL_BF16_BLOCK_ROWS + VALUES_PER_FLOAT4)
static_assert(FMA_MATMUL_BF16_PIECE_ROW_LENGTH % VALUES_PER_FLOAT4 == 0, "each piece row must start on a float4, for the float4s");
// A thread's columns are two halves of a float4 each, half a square apart (thread 0: 0..3 and 64..67): a warp then
// reads neighbouring floats of shared memory, with no repeated banks.
#define FMA_MATMUL_BF16_SECOND_HALF_OFFSET (FMA_MATMUL_BF16_BLOCK_COLUMNS / 2)
static_assert(FMA_MATMUL_BF16_THREAD_COLUMNS == 2 * VALUES_PER_FLOAT4, "a thread's columns are two float4s");
static_assert(FMA_MATMUL_BF16_THREADS_ACROSS * (FMA_MATMUL_BF16_THREAD_COLUMNS / 2) == FMA_MATMUL_BF16_SECOND_HALF_OFFSET,
              "the threads across x half their columns must fill the first half exactly");
// The blocks are numbered in groups of 8 rows of squares, down first: blocks running at once share bands of left
// and right in L2, instead of reading all of right once per row of squares.
#define FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP 8

// ── the product ───────────────────────────────────────────────────────────────
void fma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                     const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                     bool right_turned_val = false);

// ── for it ────────────────────────────────────────────────────────────────────
void fma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                  const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                  size_t* columns_ptr, size_t* sum_length_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

template <typename T, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(FMA_MATMUL_BF16_THREADS_PER_BLOCK, 2)
fma_matmul_bf16_kernel(T* output_ptr, const T* left_ptr, const T* right_ptr, const T* bias_ptr, size_t rows_val,
                       size_t columns_val, size_t sum_length_val, bool add_onto_val);
template <typename T>
void fma_matmul_bf16_launch(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                            const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                            size_t rows_val, size_t columns_val, size_t sum_length_val);
__device__ __forceinline__ void fma_matmul_bf16_block_square(size_t rows_val, size_t columns_val, size_t* block_first_row_index_ptr,
                                                             size_t* block_first_column_index_ptr);
template <typename T, bool TURNED>
__device__ __forceinline__ float4 fma_matmul_bf16_load_piece(const T* matrix_ptr, size_t lines_val, size_t sum_length_val,
                                                             size_t block_first_line_index_val, size_t step_first_k_index_val);
template <bool TURNED>
__device__ __forceinline__ void fma_matmul_bf16_store_piece(float (*piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH], float4 values_val);
__device__ __forceinline__ void fma_matmul_bf16_add_outer_products(
        float (&sums_list)[FMA_MATMUL_BF16_THREAD_ROWS][FMA_MATMUL_BF16_THREAD_COLUMNS],
        const float (*left_piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH], const float (*right_piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH],
        int thread_first_row_index_val, int thread_first_column_index_val);

/// Writes output = left · rightᵀ (+ bias) (+ output): checks, then launches fma_matmul_bf16_kernel, one block per
/// square of output, as one line of blocks.
///
///   output_tensor      {..., columns}, rows x columns values: written over, or added onto
///   left_tensor        {..., sum_length} (rows = the other dims), or turned {..., rows} (sum_length = the other dims)
///   right_tensor       {..., sum_length} (columns = the other dims), or turned {..., columns}
///   bias_tensor        {columns}, or Tensor() for none
///   add_onto_val       true: added onto what output holds; false: written over
///   left_turned_val, right_turned_val   true: that tensor holds its matrix turned round
/// All float or all bfloat16; every tensor's last dim a multiple of VALUES_PER_FLOAT4 and its start on 4 values.
inline void fma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                            const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val) {
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    fma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                 &rows_val, &columns_val, &sum_length_val);
    // 2. launch, in the tensors' type
    if (left_tensor.dtype() == DType::BFloat16)
        fma_matmul_bf16_launch<nv_bfloat16>(output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val,
                                            right_turned_val, rows_val, columns_val, sum_length_val);
    else
        fma_matmul_bf16_launch<float>(output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val,
                                      right_turned_val, rows_val, columns_val, sum_length_val);
}

/// Launches the kernel made for the two turned flags: squares across and down, rounded up, as one line of blocks.
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val, right_turned_val
///                                              as in fma_matmul_bf16
///   rows_val, columns_val, sum_length_val      the product's sizes
template <typename T>
void fma_matmul_bf16_launch(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                            const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                            size_t rows_val, size_t columns_val, size_t sum_length_val) {
    unsigned blocks_val = (unsigned)(CEIL_DIV(rows_val, (size_t)FMA_MATMUL_BF16_BLOCK_ROWS) *
                                     CEIL_DIV(columns_val, (size_t)FMA_MATMUL_BF16_BLOCK_COLUMNS));
    T* output_ptr = output_tensor.data_ptr<T>();
    const T* left_ptr = left_tensor.data_ptr<T>();
    const T* right_ptr = right_tensor.data_ptr<T>();
    const T* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<T>() : nullptr;
    if (!left_turned_val && !right_turned_val)
        fma_matmul_bf16_kernel<T, false, false><<<blocks_val, FMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else if (!left_turned_val && right_turned_val)
        fma_matmul_bf16_kernel<T, false, true><<<blocks_val, FMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else if (left_turned_val && !right_turned_val)
        fma_matmul_bf16_kernel<T, true, false><<<blocks_val, FMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    else
        fma_matmul_bf16_kernel<T, true, true><<<blocks_val, FMA_MATMUL_BF16_THREADS_PER_BLOCK>>>(
            output_ptr, left_ptr, right_ptr, bias_ptr, rows_val, columns_val, sum_length_val, add_onto_val);
    cudaCheck(cudaGetLastError());
}

/// Works out one square of output. __launch_bounds__(FMA_MATMUL_BF16_THREADS_PER_BLOCK, 2): at most 128 registers,
/// so 2 blocks fit an SM.
///
///   output_ptr, left_ptr, right_ptr, bias_ptr    as in fma_matmul_bf16, float or bfloat16 (T); bias nullptr for none
///   rows_val, columns_val, sum_length_val        the product's sizes
///   add_onto_val                                 as in fma_matmul_bf16
/// LEFT_TURNED, RIGHT_TURNED: how left and right sit, so each copy is made for its own layout.
template <typename T, bool LEFT_TURNED, bool RIGHT_TURNED>
__global__ void __launch_bounds__(FMA_MATMUL_BF16_THREADS_PER_BLOCK, 2)
fma_matmul_bf16_kernel(T* output_ptr, const T* left_ptr, const T* right_ptr, const T* bias_ptr, size_t rows_val,
                       size_t columns_val, size_t sum_length_val, bool add_onto_val) {
    // this step's values of k, as the rows: [set][k][r] = left[r][k], [set][k][c] = right[c][k]
    __shared__ __align__(16) float left_pieces_list[2][FMA_MATMUL_BF16_STEP_K][FMA_MATMUL_BF16_PIECE_ROW_LENGTH];
    __shared__ __align__(16) float right_pieces_list[2][FMA_MATMUL_BF16_STEP_K][FMA_MATMUL_BF16_PIECE_ROW_LENGTH];
    // 1. the block's square and the thread's part of it: FMA_MATMUL_BF16_THREADS_ACROSS x _DOWN threads, across first
    size_t block_first_row_index_val = 0, block_first_column_index_val = 0;
    fma_matmul_bf16_block_square(rows_val, columns_val, &block_first_row_index_val, &block_first_column_index_val);
    int thread_first_row_index_val = (threadIdx.x / FMA_MATMUL_BF16_THREADS_ACROSS) * FMA_MATMUL_BF16_THREAD_ROWS;
    int thread_first_column_index_val = (threadIdx.x % FMA_MATMUL_BF16_THREADS_ACROSS) * (FMA_MATMUL_BF16_THREAD_COLUMNS / 2);

    // 2. walk the sum: copy each step's pieces, read the next step's, add the outer products
    float sums_list[FMA_MATMUL_BF16_THREAD_ROWS][FMA_MATMUL_BF16_THREAD_COLUMNS] = {};
    size_t total_steps_val = CEIL_DIV(sum_length_val, (size_t)FMA_MATMUL_BF16_STEP_K);
    float4 left_val = fma_matmul_bf16_load_piece<T, LEFT_TURNED>(left_ptr, rows_val, sum_length_val, block_first_row_index_val, 0);
    float4 right_val = fma_matmul_bf16_load_piece<T, RIGHT_TURNED>(right_ptr, columns_val, sum_length_val, block_first_column_index_val, 0);
    for (size_t step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
        int piece_set_index_val = (int)(step_index_val % 2);    // this step reads one set; the next step fills the other
        fma_matmul_bf16_store_piece<LEFT_TURNED>(left_pieces_list[piece_set_index_val], left_val);
        fma_matmul_bf16_store_piece<RIGHT_TURNED>(right_pieces_list[piece_set_index_val], right_val);
        __syncthreads();                                        // the pieces whole, and the other set free
        // the next step's values, read now: they arrive during the math below
        size_t next_step_first_k_index_val = (step_index_val + 1) * FMA_MATMUL_BF16_STEP_K;
        if (next_step_first_k_index_val < sum_length_val) {
            left_val = fma_matmul_bf16_load_piece<T, LEFT_TURNED>(left_ptr, rows_val, sum_length_val, block_first_row_index_val,
                                                                  next_step_first_k_index_val);
            right_val = fma_matmul_bf16_load_piece<T, RIGHT_TURNED>(right_ptr, columns_val, sum_length_val,
                                                                    block_first_column_index_val, next_step_first_k_index_val);
        }
        fma_matmul_bf16_add_outer_products(sums_list, left_pieces_list[piece_set_index_val], right_pieces_list[piece_set_index_val],
                                           thread_first_row_index_val, thread_first_column_index_val);
    }

    // 3. output = sums (+ bias) (+ output), one float4 at a time; nothing past the edges
#pragma unroll
    for (int r_index_val = 0; r_index_val < FMA_MATMUL_BF16_THREAD_ROWS; r_index_val++) {
        size_t row_index_val = block_first_row_index_val + thread_first_row_index_val + r_index_val;
#pragma unroll
        for (int c_index_val = 0; c_index_val < FMA_MATMUL_BF16_THREAD_COLUMNS; c_index_val += VALUES_PER_FLOAT4) {
            int half_index_val = c_index_val / VALUES_PER_FLOAT4;   // its first float4 of columns, or the one half a square on
            size_t column_index_val = block_first_column_index_val + thread_first_column_index_val + half_index_val * FMA_MATMUL_BF16_SECOND_HALF_OFFSET;
            if (row_index_val >= rows_val || column_index_val >= columns_val) continue;
            float4 result_val = make_float4(sums_list[r_index_val][c_index_val], sums_list[r_index_val][c_index_val + 1],
                                            sums_list[r_index_val][c_index_val + 2], sums_list[r_index_val][c_index_val + 3]);
            if (bias_ptr != nullptr) {
                float4 bias_val = read4_as_float(bias_ptr + column_index_val);
                result_val = make_float4(result_val.x + bias_val.x, result_val.y + bias_val.y, result_val.z + bias_val.z,
                                         result_val.w + bias_val.w);
            }
            T* output_float4_ptr = output_ptr + row_index_val * columns_val + column_index_val;
            if (add_onto_val) {
                float4 old_val = read4_as_float(output_float4_ptr);
                result_val = make_float4(old_val.x + result_val.x, old_val.y + result_val.y, old_val.z + result_val.z,
                                         old_val.w + result_val.w);
            }
            write4_from_float(output_float4_ptr, result_val);
        }
    }
}

/// Gives the block's square from its number: groups of FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP rows of squares, down
/// a group first, so the blocks running at once share bands of left and right in L2.
///
///   rows_val, columns_val                                     the product's sizes
///   block_first_row_index_ptr, block_first_column_index_ptr   written: the square's first row and column
__device__ __forceinline__ void fma_matmul_bf16_block_square(size_t rows_val, size_t columns_val, size_t* block_first_row_index_ptr,
                                                             size_t* block_first_column_index_ptr) {
    size_t total_block_columns_val = CEIL_DIV(columns_val, (size_t)FMA_MATMUL_BF16_BLOCK_COLUMNS);
    size_t total_block_rows_val = CEIL_DIV(rows_val, (size_t)FMA_MATMUL_BF16_BLOCK_ROWS);
    size_t blocks_per_group_val = FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP * total_block_columns_val;
    size_t group_index_val = blockIdx.x / blocks_per_group_val;
    size_t group_first_block_row_index_val = group_index_val * FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP;
    size_t group_block_rows_val = total_block_rows_val - group_first_block_row_index_val;   // fewer in the last group
    if (group_block_rows_val > FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP) group_block_rows_val = FMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP;
    size_t block_index_in_group_val = blockIdx.x % blocks_per_group_val;
    *block_first_row_index_ptr = (group_first_block_row_index_val + block_index_in_group_val % group_block_rows_val) * FMA_MATMUL_BF16_BLOCK_ROWS;
    *block_first_column_index_ptr = (block_index_in_group_val / group_block_rows_val) * FMA_MATMUL_BF16_BLOCK_COLUMNS;
}

/// Gives this thread's float4 of one step's piece of left or right; 0 past the matrix's edges. Each line (a row
/// of left, a column of output for right) has sum_length values of k.
///     not TURNED: the matrix sits as {lines, sum_length}: the float4 is 4 neighbouring k of one line (2 threads a line)
///     TURNED:     it sits as {sum_length, lines}: the float4 is 4 neighbouring lines at one k (32 threads a k)
///
///   matrix_ptr                     left or right
///   lines_val, sum_length_val      its matrix's sizes: rows or columns, and sum_length
///   block_first_line_index_val     the block's first line
///   step_first_k_index_val         the step's first k
template <typename T, bool TURNED>
__device__ __forceinline__ float4 fma_matmul_bf16_load_piece(const T* matrix_ptr, size_t lines_val, size_t sum_length_val,
                                                             size_t block_first_line_index_val, size_t step_first_k_index_val) {
    float4 values_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (TURNED) {
        const int threads_per_k_val = FMA_MATMUL_BF16_BLOCK_ROWS / VALUES_PER_FLOAT4;
        size_t k_index_val = step_first_k_index_val + threadIdx.x / threads_per_k_val;
        size_t line_index_val = block_first_line_index_val + (threadIdx.x % threads_per_k_val) * VALUES_PER_FLOAT4;
        if (k_index_val < sum_length_val && line_index_val < lines_val)
            values_val = read4_as_float(matrix_ptr + k_index_val * lines_val + line_index_val);
    } else {
        const int threads_per_line_val = FMA_MATMUL_BF16_STEP_K / VALUES_PER_FLOAT4;
        size_t line_index_val = block_first_line_index_val + threadIdx.x / threads_per_line_val;
        size_t k_index_val = step_first_k_index_val + (threadIdx.x % threads_per_line_val) * VALUES_PER_FLOAT4;
        if (line_index_val < lines_val && k_index_val < sum_length_val)
            values_val = read4_as_float(matrix_ptr + line_index_val * sum_length_val + k_index_val);
    }
    return values_val;
}

/// Writes fma_matmul_bf16_load_piece's float4 into the piece, k as the rows: piece[k][line].
///     not TURNED: 4 neighbouring k of one line, turned round into 4 piece rows
///     TURNED:     4 neighbouring lines at one k, as one float4
///
///   piece_ptr    one set of the shared memory pieces
///   values_val   what fma_matmul_bf16_load_piece gave
template <bool TURNED>
__device__ __forceinline__ void fma_matmul_bf16_store_piece(float (*piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH], float4 values_val) {
    if constexpr (TURNED) {
        const int threads_per_k_val = FMA_MATMUL_BF16_BLOCK_ROWS / VALUES_PER_FLOAT4;
        int k_index_val = threadIdx.x / threads_per_k_val;
        int line_index_val = (threadIdx.x % threads_per_k_val) * VALUES_PER_FLOAT4;
        *reinterpret_cast<float4*>(&piece_ptr[k_index_val][line_index_val]) = values_val;
    } else {
        const int threads_per_line_val = FMA_MATMUL_BF16_STEP_K / VALUES_PER_FLOAT4;
        int line_index_val = threadIdx.x / threads_per_line_val;
        int k_index_val = (threadIdx.x % threads_per_line_val) * VALUES_PER_FLOAT4;
        piece_ptr[k_index_val + 0][line_index_val] = values_val.x;
        piece_ptr[k_index_val + 1][line_index_val] = values_val.y;
        piece_ptr[k_index_val + 2][line_index_val] = values_val.z;
        piece_ptr[k_index_val + 3][line_index_val] = values_val.w;
    }
}

/// Adds one step's outer products into the thread's sums: sums[r][c] += left[r][k] · right[c][k].
/// FMA_MATMUL_BF16_THREAD_ROWS x _COLUMNS multiply-adds for _ROWS + _COLUMNS values read at each k.
///
///   sums_list                                                   the thread's part
///   left_piece_ptr, right_piece_ptr                             this step's pieces
///   thread_first_row_index_val, thread_first_column_index_val   the thread's place in the square
__device__ __forceinline__ void fma_matmul_bf16_add_outer_products(
        float (&sums_list)[FMA_MATMUL_BF16_THREAD_ROWS][FMA_MATMUL_BF16_THREAD_COLUMNS],
        const float (*left_piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH], const float (*right_piece_ptr)[FMA_MATMUL_BF16_PIECE_ROW_LENGTH],
        int thread_first_row_index_val, int thread_first_column_index_val) {
#pragma unroll
    for (int k_index_val = 0; k_index_val < FMA_MATMUL_BF16_STEP_K; k_index_val++) {
        // 1. this k's rows of left and columns of right, two float4s each
        float4 left_first_half_val = *reinterpret_cast<const float4*>(&left_piece_ptr[k_index_val][thread_first_row_index_val]);
        float4 left_second_half_val = *reinterpret_cast<const float4*>(&left_piece_ptr[k_index_val][thread_first_row_index_val + VALUES_PER_FLOAT4]);
        float4 right_first_half_val = *reinterpret_cast<const float4*>(&right_piece_ptr[k_index_val][thread_first_column_index_val]);
        float4 right_second_half_val = *reinterpret_cast<const float4*>(
            &right_piece_ptr[k_index_val][thread_first_column_index_val + FMA_MATMUL_BF16_SECOND_HALF_OFFSET]);
        float left_list[FMA_MATMUL_BF16_THREAD_ROWS] = {
            left_first_half_val.x,  left_first_half_val.y,  left_first_half_val.z,  left_first_half_val.w,
            left_second_half_val.x, left_second_half_val.y, left_second_half_val.z, left_second_half_val.w};
        float right_list[FMA_MATMUL_BF16_THREAD_COLUMNS] = {
            right_first_half_val.x,  right_first_half_val.y,  right_first_half_val.z,  right_first_half_val.w,
            right_second_half_val.x, right_second_half_val.y, right_second_half_val.z, right_second_half_val.w};
        // 2. their outer product into the sums: one FMA each
#pragma unroll
        for (int r_index_val = 0; r_index_val < FMA_MATMUL_BF16_THREAD_ROWS; r_index_val++) {
#pragma unroll
            for (int c_index_val = 0; c_index_val < FMA_MATMUL_BF16_THREAD_COLUMNS; c_index_val++)
                sums_list[r_index_val][c_index_val] += left_list[r_index_val] * right_list[c_index_val];
        }
    }
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Stops with a message unless the tensors fit, and gives the product's sizes: left's and right's sum_length the
/// same, output rows x columns with columns last, bias {columns}; all float or all bfloat16; every last dim a
/// multiple of VALUES_PER_FLOAT4 and every tensor starting on 4 values (the float4 reads and writes).
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val   as in fma_matmul_bf16
///   rows_ptr, columns_ptr, sum_length_ptr                                                       written: the sizes
inline void fma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                         const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                         size_t* columns_ptr, size_t* sum_length_ptr) {
    // 1. shapes: each matrix's last dim and the rest, by how it sits
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = left_tensor.dim() >= 1 && right_tensor.dim() >= 1 && output_tensor.dim() >= 1;
    size_t left_last_val = fits_val ? left_tensor.size(left_tensor.dim() - 1) : 1;
    size_t right_last_val = fits_val ? right_tensor.size(right_tensor.dim() - 1) : 1;
    size_t rows_val = left_turned_val ? left_last_val : left_tensor.numel() / left_last_val;
    size_t sum_length_val = left_turned_val ? left_tensor.numel() / left_last_val : left_last_val;
    size_t columns_val = right_turned_val ? right_last_val : right_tensor.numel() / right_last_val;
    size_t right_sum_length_val = right_turned_val ? right_tensor.numel() / right_last_val : right_last_val;
    fits_val = fits_val && sum_length_val == right_sum_length_val && output_tensor.size(output_tensor.dim() - 1) == columns_val &&
               output_tensor.numel() == rows_val * columns_val && (!has_bias_val || bias_tensor.shape() == Shape({columns_val}));
    if (!fits_val) {
        fprintf(stderr, "error: fma_matmul_bf16 needs left {rows, sum_length} (turned {sum_length, rows}), right {columns, "
                        "sum_length} (turned {sum_length, columns}), output {rows, columns}, bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all float or all bfloat16
    DType dtype_val = left_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || right_tensor.dtype() != dtype_val ||
        output_tensor.dtype() != dtype_val || (has_bias_val && bias_tensor.dtype() != dtype_val)) {
        fprintf(stderr, "error: fma_matmul_bf16 needs output, left, right and bias all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: whole float4s, each tensor starting on one
    size_t float4_bytes_val = VALUES_PER_FLOAT4 * left_tensor.element_size();
    if (left_last_val % VALUES_PER_FLOAT4 != 0 || right_last_val % VALUES_PER_FLOAT4 != 0 || columns_val % VALUES_PER_FLOAT4 != 0 ||
        (size_t)output_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)left_tensor.data_ptr() % float4_bytes_val != 0 ||
        (size_t)right_tensor.data_ptr() % float4_bytes_val != 0 || (size_t)bias_tensor.data_ptr() % float4_bytes_val != 0) {
        fprintf(stderr, "error: fma_matmul_bf16 reads %d values at a time, so every tensor's last dim must be a multiple of %d "
                        "and every tensor must start on %zu bytes; got last dims %zu (left), %zu (right), %zu (output)\n",
                VALUES_PER_FLOAT4, VALUES_PER_FLOAT4, float4_bytes_val, left_last_val, right_last_val, columns_val);
        exit(EXIT_FAILURE);
    }
    *rows_ptr = rows_val;
    *columns_ptr = columns_val;
    *sum_length_ptr = sum_length_val;
}

#endif // FMA_MATMUL_BF16_CUH
