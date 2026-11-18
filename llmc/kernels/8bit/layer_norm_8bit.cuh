// layer_norm_8bit.cuh — LayerNorm for a bfloat16 model whose products are in 8 bits, its output written straight in
// 8 bits: both FP8 E4M3 copies the products read, one scale for the whole tensor, as mlp_up_8bit_gelu_8bit writes
// gelu_output:
//     forward:    rows copy, turned copy   = LayerNorm(input) · weight + bias, rounded as round_8bit_per_tensor does
//                 output (if given)        = the same in bfloat16, as layer_norm_bf16
//     backward:   layer_norm_bf16_backward's (the rounding's gradient taken as 1, straight-through, as the 8-bit
//                 products' backwards take it)
//
//     tensor          type in memory   shape                                 note
//     input, output   bfloat16         {batch_size, seq_len, embed_dim}      output: Tensor() when not wanted
//     rows_8bit       E4M3             {rows, embed_dim}                     rows = batch_size x seq_len: the forward
//                                                                            product's left side (qkv, mlp_up, head)
//     turned_8bit     E4M3             {embed_dim, rows}                     the weight_grad product's right side
//     scale           float            {1}                                   both copies' one scale
//     weight, bias    bfloat16         {embed_dim}
//     mean, rstd      float            {batch_size, seq_len}                 kept for the backward
//
// Why: a linear's input is rounded to E4M3 twice, as it sits for the forward's product and turned for the backward's
// weight_grad. Done apart, LayerNorm writes its bfloat16 output, the forward's rounding reads it back, and the
// bfloat16 output is kept until the backward, whose rounding reads it again. Here both copies come out of LayerNorm
// itself, and no bfloat16 output is written or kept:
//
//     a value of the output                    written             read back    kept until the backward
//     layer_norm_bf16, then 2 rounding passes  2 + 1 + 1 bytes     2 + 2 bytes  2 bytes (bfloat16)
//     here                                     1 + 1 bytes         none         1 byte (the turned copy)
//
// One scale for the tensor has to be known before the first value is rounded, but the tensor's largest |value| is
// known only after the last: so the scale is a delayed one (NVIDIA Transformer Engine's delayed scaling), from the
// largest of the tensor this place held the call before (round_8bit_delayed_largest) times ROUND_8BIT_DELAYED_MARGIN,
// and this kernel maxes this tensor's largest onto the next call's slot. With one scale, a value's E4M3 byte is the
// same in both copies: each value is rounded once, written to the row copy straight away and kept in shared memory;
// when a block's TILE_ROWS rows are done, it writes them turned, LAYER_NORM_8BIT_TURNED_STORE_VALUES rows a store.
// A history's first call has no largest yet: the caller gives one (model_8bit.cuh starts these histories at scale 1,
// as Transformer Engine does).
// Any GPU (no tensor cores here).
// Math: https://ducbachsong.github.io/posts/layer-norm-backward
#ifndef LAYER_NORM_8BIT_CUH
#define LAYER_NORM_8BIT_CUH

#include <climits>                       // CHAR_BIT
#include <cuda_fp8.h>                    // FP8 E4M3
#include "../cuda_common.h"              // cudaCheck, check_threads_per_block, warp_reduce_max, HARDWARE_MAX_SHARED_BYTES_PER_BLOCK
#include "../../common/tensor.cuh"       // read8_as_float, write8_from_float
#include "round_8bit.cuh"                // round_8bit_scale, ROUND_8BIT_ROW_LENGTH_MULTIPLE, ROUND_8BIT_STORE_VALUES
#include "../bf16/layer_norm_bf16.cuh"   // the 8 values a lane, the row's mean and rstd, the checks, the backward

#define LAYER_NORM_8BIT_VALUES_PER_LOAD LAYER_NORM_BF16_VALUES_PER_LOAD                       // values a lane takes at a time
#define LAYER_NORM_8BIT_FOURS_PER_LOAD (LAYER_NORM_8BIT_VALUES_PER_LOAD / VALUES_PER_FLOAT4)   // E4M3 fours (4 bytes each) in them
#define LAYER_NORM_8BIT_THREADS_PER_BLOCK LAYER_NORM_BF16_THREADS_PER_BLOCK                   // the forward's default block size: an untuned guess (tune_layer_norm_8bit sweeps it)
#define LAYER_NORM_8BIT_TILE_ROWS 32                                                          // rows a block normalizes, then writes turned: an untuned guess (tune_layer_norm_8bit sweeps 16, 32, 64)
#define LAYER_NORM_8BIT_TURNED_STORE_VALUES ROUND_8BIT_STORE_VALUES                           // E4M3s in one store of the turned copy: one 16-byte store
#define LAYER_NORM_8BIT_E4M3S_PER_WORD ((int)(sizeof(unsigned int) / sizeof(__nv_fp8_e4m3)))  // E4M3s in one 4-byte word
#define LAYER_NORM_8BIT_BITS_PER_E4M3 (CHAR_BIT * (int)sizeof(__nv_fp8_e4m3))                 // an E4M3's bits: its shift in a word
#define LAYER_NORM_8BIT_BACKWARD_THREADS_PER_BLOCK LAYER_NORM_BF16_BACKWARD_THREADS_PER_BLOCK // the backward's: layer_norm_bf16's
static_assert(sizeof(uint2) == LAYER_NORM_8BIT_VALUES_PER_LOAD * sizeof(__nv_fp8_e4m3), "a load's E4M3s are one 8-byte store");
static_assert(LAYER_NORM_8BIT_FOURS_PER_LOAD == 2, "a load's E4M3 fours are the two words of a uint2");
static_assert(sizeof(uint4) == LAYER_NORM_8BIT_TURNED_STORE_VALUES * sizeof(__nv_fp8_e4m3) &&
              LAYER_NORM_8BIT_TURNED_STORE_VALUES / LAYER_NORM_8BIT_E4M3S_PER_WORD == 4,
              "a turned store's E4M3s are the four words of a uint4");
static_assert(LAYER_NORM_8BIT_TILE_ROWS % LAYER_NORM_8BIT_TURNED_STORE_VALUES == 0, "a tile is whole turned stores");

/// The output in 8 bits, as layer_norm_8bit_forward writes it: both copies the products read, with one scale for the
/// tensor (one float, written here) made from a largest |value| in GPU memory, and where this tensor's
/// largest goes for the next call.
struct LayerNorm8bitCopies {
    Tensor rows_8bit_tensor;                       // {rows, embed_dim} E4M3: as it sits
    Tensor turned_8bit_tensor;                     // {embed_dim, rows} E4M3: turned round
    Tensor scale_tensor;                           // {1} float: both copies' one scale
    const unsigned* scale_largest_ptr = nullptr;   // the largest |value| the scale is made from (a float's bits, in GPU memory)
    float margin_val = 1.0f;                       // ... times this (ROUND_8BIT_DELAYED_MARGIN for a delayed one)
    unsigned* next_largest_ptr = nullptr;          // this tensor's largest |value| maxed onto it (0 before), or nullptr
};

// ── forward and backward ──────────────────────────────────────────────────────
void layer_norm_8bit_forward(const Tensor& output_tensor, const LayerNorm8bitCopies& copies_val, const Tensor& mean_tensor,
                             const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                             const Tensor& bias_tensor, int threads_per_block_val = LAYER_NORM_8BIT_THREADS_PER_BLOCK);
template <int TILE_ROWS>
void layer_norm_8bit_forward_at(const Tensor& output_tensor, const LayerNorm8bitCopies& copies_val, const Tensor& mean_tensor,
                                const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                const Tensor& bias_tensor, int threads_per_block_val);
void layer_norm_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                              const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                              const Tensor& mean_tensor, const Tensor& rstd_tensor,
                              int threads_per_block_val = LAYER_NORM_8BIT_BACKWARD_THREADS_PER_BLOCK);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

template <int TILE_ROWS>
__global__ void layer_norm_8bit_forward_kernel(nv_bfloat16* output_ptr, __nv_fp8_e4m3* rows_8bit_ptr,
                                               __nv_fp8_e4m3* turned_8bit_ptr, float* scale_ptr, float* mean_ptr,
                                               float* rstd_ptr, const nv_bfloat16* input_ptr, const nv_bfloat16* weight_ptr,
                                               const nv_bfloat16* bias_ptr, size_t rows_val, size_t embed_dim_val,
                                               const unsigned* scale_largest_ptr, float margin_val, unsigned* next_largest_ptr);

/// Writes LayerNorm(input) · weight + bias in 8 bits, both copies with one scale for the tensor, each row's mean and
/// rstd, and (if given) the bfloat16 output, at this file's tile: layer_norm_8bit_forward_at<LAYER_NORM_8BIT_TILE_ROWS>.
///
///   the tensors and threads_per_block_val   as in layer_norm_8bit_forward_at
inline void layer_norm_8bit_forward(const Tensor& output_tensor, const LayerNorm8bitCopies& copies_val, const Tensor& mean_tensor,
                                    const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                    const Tensor& bias_tensor, int threads_per_block_val) {
    layer_norm_8bit_forward_at<LAYER_NORM_8BIT_TILE_ROWS>(output_tensor, copies_val, mean_tensor, rstd_tensor, input_tensor,
                                                          weight_tensor, bias_tensor, threads_per_block_val);
}

/// Writes LayerNorm(input) · weight + bias in 8 bits at a tile given: checks, asks CUDA for the tile's shared memory
/// past the default, then launches layer_norm_8bit_forward_kernel<TILE_ROWS>, a block per TILE_ROWS rows, one warp a
/// row.
///
///   output_tensor                 {batch_size, seq_len, embed_dim} bfloat16: written; or Tensor() (not written)
///   copies_val                    the two copies, their one scale {1}, the largest the scale comes from (not nullptr),
///                                 its margin and where this tensor's largest goes; rows = batch_size x seq_len and
///                                 embed_dim multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE (a copy's rows are whole
///                                 16-byte pieces)
///   mean_tensor, rstd_tensor      {batch_size, seq_len} float: for the backward
///   input_tensor                  {batch_size, seq_len, embed_dim} bfloat16
///   weight_tensor, bias_tensor    {embed_dim} bfloat16
///   threads_per_block_val         the block size: whole warps, at most HARDWARE_MAX_THREADS_PER_BLOCK. Any size gives
///                                 the same bits; only the time changes.
/// TILE_ROWS: rows a block takes, a multiple of LAYER_NORM_8BIT_TURNED_STORE_VALUES; its E4M3s, TILE_ROWS x embed_dim
/// bytes, are its shared memory. Any tile gives the same bits.
template <int TILE_ROWS>
inline void layer_norm_8bit_forward_at(const Tensor& output_tensor, const LayerNorm8bitCopies& copies_val, const Tensor& mean_tensor,
                                       const Tensor& rstd_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                       const Tensor& bias_tensor, int threads_per_block_val) {
    static_assert(TILE_ROWS > 0 && TILE_ROWS % LAYER_NORM_8BIT_TURNED_STORE_VALUES == 0, "a tile is whole turned stores");
    // 1. check the launch, the tensors and the copies; any of them wrong: stop with an error (no output: input's shape
    //    stands in for it in the shared checks)
    check_threads_per_block(threads_per_block_val, "layer_norm_8bit_forward");
    bool has_output_val = output_tensor.numel() > 0;
    layer_norm_bf16_check_shapes("layer_norm_8bit_forward", has_output_val ? output_tensor : input_tensor, mean_tensor, rstd_tensor,
                                 input_tensor, weight_tensor, bias_tensor);
    size_t rows_val = input_tensor.size(0) * input_tensor.size(1);
    size_t embed_dim_val = input_tensor.size(2);
    size_t shared_bytes_val = (size_t)TILE_ROWS * embed_dim_val * sizeof(__nv_fp8_e4m3);
    const LayerNorm8bitCopies& c = copies_val;
    bool fits_val = rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && embed_dim_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 &&
                    c.rows_8bit_tensor.dtype() == DType::E4M3 && c.rows_8bit_tensor.dim() >= 1 &&
                    c.rows_8bit_tensor.numel() == rows_val * embed_dim_val &&
                    c.rows_8bit_tensor.size(c.rows_8bit_tensor.dim() - 1) == embed_dim_val &&
                    c.turned_8bit_tensor.dtype() == DType::E4M3 && c.turned_8bit_tensor.dim() >= 1 &&
                    c.turned_8bit_tensor.numel() == rows_val * embed_dim_val &&
                    c.turned_8bit_tensor.size(c.turned_8bit_tensor.dim() - 1) == rows_val &&
                    c.scale_tensor.dtype() == DType::Float32 && c.scale_tensor.numel() == 1 &&
                    (size_t)c.rows_8bit_tensor.data_ptr() % sizeof(uint2) == 0 && (size_t)c.turned_8bit_tensor.data_ptr() % sizeof(uint4) == 0 &&
                    c.scale_largest_ptr != nullptr && shared_bytes_val <= HARDWARE_MAX_SHARED_BYTES_PER_BLOCK;
    if (!fits_val) {
        fprintf(stderr, "error: layer_norm_8bit_forward needs rows and embed_dim multiples of %d, E4M3 copies {rows, embed_dim} "
                        "and {embed_dim, rows} on 8 and 16 bytes with a float scale {1}, a largest for the "
                        "scale, and %d x embed_dim bytes of shared memory at most %d; got rows %zu, embed_dim %zu\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE, TILE_ROWS, HARDWARE_MAX_SHARED_BYTES_PER_BLOCK, rows_val, embed_dim_val);
        exit(EXIT_FAILURE);
    }
    // 2. the tile's shared memory: past the default, asked of CUDA (once for the most asked so far)
    static size_t asked_bytes_val = HARDWARE_DEFAULT_MAX_SHARED_BYTES_PER_BLOCK;
    if (shared_bytes_val > asked_bytes_val) {
        cudaCheck(cudaFuncSetAttribute(layer_norm_8bit_forward_kernel<TILE_ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)shared_bytes_val));
        asked_bytes_val = shared_bytes_val;
    }
    // 3. launch: a block per TILE_ROWS rows, the last one maybe short
    int blocks_val = (int)CEIL_DIV(rows_val, (size_t)TILE_ROWS);
    layer_norm_8bit_forward_kernel<TILE_ROWS><<<blocks_val, threads_per_block_val, shared_bytes_val>>>(
        has_output_val ? output_tensor.data_ptr<nv_bfloat16>() : nullptr, reinterpret_cast<__nv_fp8_e4m3*>(c.rows_8bit_tensor.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(c.turned_8bit_tensor.data_ptr()), c.scale_tensor.data(),
        mean_tensor.data(), rstd_tensor.data(), input_tensor.data_ptr<nv_bfloat16>(), weight_tensor.data_ptr<nv_bfloat16>(),
        bias_tensor.data_ptr<nv_bfloat16>(), rows_val, embed_dim_val, c.scale_largest_ptr, c.margin_val, c.next_largest_ptr);
    cudaCheck(cudaGetLastError());
}

/// Normalizes a block's TILE_ROWS rows as layer_norm_bf16_forward_kernel does, one warp a row, 8 values a lane, and
/// rounds each value of the bfloat16 output as stored over the tensor's scale to E4M3 (saturating, as
/// __nv_fp8x4_e4m3 does, as round_8bit_per_tensor does): written to the row copy, and kept in shared memory; then
/// writes the kept E4M3s turned round, LAYER_NORM_8BIT_TURNED_STORE_VALUES rows of one column a store.
///
///   output_ptr                                    {rows, embed_dim} bfloat16: written; or nullptr
///   rows_8bit_ptr                                 {rows, embed_dim} E4M3: written
///   turned_8bit_ptr                               {embed_dim, rows} E4M3: written
///   scale_ptr                                     both copies' one scale {1}: written (by the first block)
///   mean_ptr, rstd_ptr, input_ptr, weight_ptr, bias_ptr   as in layer_norm_8bit_forward_at
///   rows_val                                      batch_size x seq_len: a multiple of LAYER_NORM_8BIT_TURNED_STORE_VALUES
///   embed_dim_val                                 the row length
///   scale_largest_ptr, margin_val                 the scale is round_8bit_scale(largest x margin)
///   next_largest_ptr                              this tensor's largest |value| maxed onto it, or nullptr
/// TILE_ROWS: as in layer_norm_8bit_forward_at; the dynamic shared memory holds TILE_ROWS x embed_dim E4M3s.
template <int TILE_ROWS>
__global__ void layer_norm_8bit_forward_kernel(nv_bfloat16* output_ptr, __nv_fp8_e4m3* rows_8bit_ptr,
                                               __nv_fp8_e4m3* turned_8bit_ptr, float* scale_ptr, float* mean_ptr,
                                               float* rstd_ptr, const nv_bfloat16* input_ptr, const nv_bfloat16* weight_ptr,
                                               const nv_bfloat16* bias_ptr, size_t rows_val, size_t embed_dim_val,
                                               const unsigned* scale_largest_ptr, float margin_val, unsigned* next_largest_ptr) {
    extern __shared__ __align__(16) unsigned char tile_bytes_list[];   // {TILE_ROWS, embed_dim}: the tile's E4M3s
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int warps_val = blockDim.x / HARDWARE_THREADS_PER_WARP;
    size_t first_row_index_val = (size_t)blockIdx.x * TILE_ROWS;
    size_t row_loads_val = embed_dim_val / LAYER_NORM_8BIT_VALUES_PER_LOAD;

    // 1. the tensor's scale, known before any value: from the largest given, with its margin
    float scale_val = round_8bit_scale(__uint_as_float(*scale_largest_ptr) * margin_val);
    float to_stored_val = 1.0f / scale_val;                       // exact: a power of 2
    float thread_largest_val = 0.0f;                              // this lane's largest |value| of the output as stored
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale_ptr = scale_val;   // both copies' (every block works out the same)

    // 2. the tile's rows: warp w takes tile rows w, w + warps, ...: the same for all its lanes, so the warp stays whole
    for (int tile_row_index_val = warp_index_val; tile_row_index_val < TILE_ROWS; tile_row_index_val += warps_val) {
        size_t row_index_val = first_row_index_val + tile_row_index_val;
        if (row_index_val >= rows_val) break;                     // the last block's short tile: the rest are past too
        const nv_bfloat16* input_row_ptr = input_ptr + row_index_val * embed_dim_val;
        unsigned char* tile_row_ptr = tile_bytes_list + (size_t)tile_row_index_val * embed_dim_val;

        // 2a. the mean, then the variance around it and rstd
        float row_mean_val, row_rstd_val;
        layer_norm_bf16_row_mean_rstd(&row_mean_val, &row_rstd_val, input_row_ptr, lane_index_val, embed_dim_val);

        // 2b. norm, then the output rounded to bfloat16 as layer_norm_bf16 stores it; over the scale to E4M3, into the
        //     row copy and the tile
        for (size_t load_index_val = lane_index_val; load_index_val < row_loads_val; load_index_val += HARDWARE_THREADS_PER_WARP) {
            size_t first_index_val = LAYER_NORM_8BIT_VALUES_PER_LOAD * load_index_val;
            float input_list[LAYER_NORM_8BIT_VALUES_PER_LOAD], weight_list[LAYER_NORM_8BIT_VALUES_PER_LOAD];
            float bias_list[LAYER_NORM_8BIT_VALUES_PER_LOAD], output_list[LAYER_NORM_8BIT_VALUES_PER_LOAD];
            read8_as_float(input_list, input_row_ptr + first_index_val);
            read8_as_float(weight_list, weight_ptr + first_index_val);
            read8_as_float(bias_list, bias_ptr + first_index_val);
#pragma unroll
            for (int value_index_val = 0; value_index_val < LAYER_NORM_8BIT_VALUES_PER_LOAD; value_index_val++) {
                float output_val = (input_list[value_index_val] - row_mean_val) * row_rstd_val * weight_list[value_index_val] +
                                   bias_list[value_index_val];
                output_list[value_index_val] = to_float(from_float<nv_bfloat16>(output_val));   // as stored
                thread_largest_val = fmaxf(thread_largest_val, fabsf(output_list[value_index_val]));
            }
            if (output_ptr != nullptr) write8_from_float(output_ptr + row_index_val * embed_dim_val + first_index_val, output_list);
            unsigned int fours_list[LAYER_NORM_8BIT_FOURS_PER_LOAD];
#pragma unroll
            for (int four_index_val = 0; four_index_val < LAYER_NORM_8BIT_FOURS_PER_LOAD; four_index_val++) {
                const float* four_ptr = output_list + four_index_val * VALUES_PER_FLOAT4;
                fours_list[four_index_val] = __nv_fp8x4_e4m3(make_float4(four_ptr[0] * to_stored_val, four_ptr[1] * to_stored_val,
                                                                         four_ptr[2] * to_stored_val, four_ptr[3] * to_stored_val)).__x;
            }
            uint2 stored_val = make_uint2(fours_list[0], fours_list[1]);
            *reinterpret_cast<uint2*>(rows_8bit_ptr + row_index_val * embed_dim_val + first_index_val) = stored_val;
            *reinterpret_cast<uint2*>(tile_row_ptr + first_index_val) = stored_val;
        }

        // 2c. lane 0 keeps the row's mean and rstd
        if (lane_index_val == 0) {
            mean_ptr[row_index_val] = row_mean_val;
            rstd_ptr[row_index_val] = row_rstd_val;
        }
    }

    // 3. this tensor's largest |value| onto the next call's, a warp at a time
    if (next_largest_ptr != nullptr) {
        float warp_largest_val = warp_reduce_max(thread_largest_val);
        if (lane_index_val == 0 && warp_largest_val > 0.0f) atomicMax(next_largest_ptr, __float_as_uint(warp_largest_val));
    }
    __syncthreads();

    // 4. the turned copy: piece p is column p % embed_dim (the turned copy's row), tile rows
    //    LAYER_NORM_8BIT_TURNED_STORE_VALUES x (p / embed_dim) on; a warp's lanes on neighbouring columns, so each of
    //    its reads of the tile is one row's neighbouring bytes
    size_t pieces_val = embed_dim_val * (TILE_ROWS / LAYER_NORM_8BIT_TURNED_STORE_VALUES);
    for (size_t piece_index_val = threadIdx.x; piece_index_val < pieces_val; piece_index_val += blockDim.x) {
        size_t column_index_val = piece_index_val % embed_dim_val;
        int first_tile_row_index_val = (int)(piece_index_val / embed_dim_val) * LAYER_NORM_8BIT_TURNED_STORE_VALUES;
        size_t piece_first_row_index_val = first_row_index_val + first_tile_row_index_val;
        if (piece_first_row_index_val >= rows_val) break;         // pieces go down the tile: the rest are past too
        unsigned int words_list[LAYER_NORM_8BIT_TURNED_STORE_VALUES / LAYER_NORM_8BIT_E4M3S_PER_WORD];
#pragma unroll
        for (int word_index_val = 0; word_index_val < LAYER_NORM_8BIT_TURNED_STORE_VALUES / LAYER_NORM_8BIT_E4M3S_PER_WORD; word_index_val++) {
            unsigned int word_val = 0;                            // the next E4M3S_PER_WORD rows' bytes, the first lowest
#pragma unroll
            for (int byte_index_val = 0; byte_index_val < LAYER_NORM_8BIT_E4M3S_PER_WORD; byte_index_val++) {
                int tile_row_index_val = first_tile_row_index_val + word_index_val * LAYER_NORM_8BIT_E4M3S_PER_WORD + byte_index_val;
                word_val |= (unsigned int)tile_bytes_list[(size_t)tile_row_index_val * embed_dim_val + column_index_val]
                            << (LAYER_NORM_8BIT_BITS_PER_E4M3 * byte_index_val);
            }
            words_list[word_index_val] = word_val;
        }
        *reinterpret_cast<uint4*>(turned_8bit_ptr + column_index_val * rows_val + piece_first_row_index_val) =
            make_uint4(words_list[0], words_list[1], words_list[2], words_list[3]);
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Adds the gradients of input, weight and bias onto what is there: layer_norm_bf16_backward, in bfloat16, from the
/// bfloat16 output's math. The forward's rounding to E4M3 passes the gradient through as it is (straight-through).
/// weight_grad and bias_grad must be 0 before a step's first backward.
///
///   the tensors and threads_per_block_val   as in layer_norm_bf16_backward
inline void layer_norm_8bit_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                                     const Tensor& mean_tensor, const Tensor& rstd_tensor, int threads_per_block_val) {
    layer_norm_bf16_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                             weight_tensor, mean_tensor, rstd_tensor, threads_per_block_val);
}

#endif // LAYER_NORM_8BIT_CUH
