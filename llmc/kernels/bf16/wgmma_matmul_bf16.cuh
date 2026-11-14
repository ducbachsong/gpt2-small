// wgmma_matmul_bf16.cuh — matrix products in bfloat16 on an H100's tensor cores, fed by TMA:
//     output = left · rightᵀ (+ bias) (+ output)
// Sums in float; each value rounded to bfloat16 only as it is stored. Built after Pranjal Shankhdhar's
// "Outperforming cuBLAS on H100: a Worklog", one of his steps at a time.
//
//     tensor   type in memory   shape                                               read with
//     left     bfloat16         {rows, sum_length}, turned {sum_length, rows}       TMA, a tile at a time, into shared memory
//     right    bfloat16         {columns, sum_length}, turned {sum_length, columns} TMA, the same
//     bias     bfloat16         {columns}, or none                                  plain loads, as the sums are stored
//     output   bfloat16         {rows, columns}                                     written over, or added onto
//     sums     float            in registers                                        the tensor cores' C
//
// turned: the tensor holds its matrix turned round (a linear layer's backward; see fma_matmul_bf16.cuh's table).
// A tile is then {64 k, lines} in shared memory, lines next to each other ("MN-major") where it is usually
// {lines, 64 k} ("K-major"); the WGMMA's transpose bit for that side says so, and the tensor cores read it
// turned. A turned tile's row holds at most WGMMA_MATMUL_BF16_ROW_VALUES (64) lines, one 128-byte swizzled row, so
// a turned right tile of more columns is COLUMNS / 64 boxes side by side, and the WGMMA descriptor's leading
// offset says how far apart they are.
//
// Each block makes one (ROW_GROUPS x HARDWARE_WGMMA_M) x COLUMNS square of output with ROW_GROUPS warpgroups (4
// warps, 128 threads each), walking the sum WGMMA_MATMUL_BF16_BLOCK_K at a time. Three of its numbers are template
// parameters, swept by tune_wgmma_matmul_bf16 (defaults WGMMA_MATMUL_BF16_COLUMNS, WGMMA_MATMUL_BF16_STAGES and
// WGMMA_MATMUL_BF16_ROW_GROUPS):
//     COLUMNS   64, 128 or 256: the square's columns, and its WGMMA's N (m64nCOLUMNSk16). A wider square reads
//               each left tile once for more columns of output
//     STAGES    1 to WGMMA_MATMUL_BF16_MAX_STAGES: sets of tiles the block keeps. TMA brings the next steps' tiles
//               while the tensor cores work on this one's (STAGES 1: ask, wait, work, as the worklog's kernel 2)
//     ROW_GROUPS   1 or 2: warpgroups a block, each 64 rows of the square with its own left tile, all reading
//               the one right tile. Per value of k a block reads (64 x ROW_GROUPS + COLUMNS) lines for
//               64 x ROW_GROUPS x COLUMNS products: at 64 x 128, 3 bytes of tiles a 128 flops, 2x what 128 x 256
//               reads, and more than L2 gives the tensor cores at full speed
// A block is one producer warpgroup and ROW_GROUPS consumer warpgroups (the worklog's kernels 4 and 7). Each set of
// tiles has two barriers (mbarrier, in PTX, the phase worked out from the step: no arrival tokens):
//     full    the set's bytes are in: the producer says how many are coming, TMA completes it
//     empty   every consumer is done with the set: each consumer warpgroup's first thread arrives once
// The producer's thread 0 walks the steps: from the second lap of the sets on it waits for the set's empty, then
// asks TMA for the step's tiles. A consumer's step:
//     1. wait for the set's full
//     2. its WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs: its sums {64, COLUMNS} += its left tile ·
//        right tileᵀ, read straight from shared memory
//     3. wait for the step before's WGMMAs, not this one's (WGMMA_MATMUL_BF16_IN_FLIGHT; STAGES 1: this step's),
//        and arrive at that step's set's empty
// So the loads never wait for the maths nor the maths for a block-wide sync: the tiles of the next steps come while
// the tensor cores work, and no step waits for the whole block.
// The tiles live in dynamic shared memory: STAGES x (64 x ROW_GROUPS + COLUMNS) rows of 128 bytes, each set on a
// 1024-byte atom.
// The blocks are persistent (the worklog's kernel 6): a launch has as many as the GPU runs at once (or blocks, a
// launch parameter), and each walks many squares, square n, n + blocks, ... So no block starts cold a square: while
// the consumers store one square, the producer already asks for the next one's tiles.
// CLUSTER (a template parameter, 1 or 2; the worklog's kernel 8): blocks of a cluster run on neighbouring SMs and take
// squares one under the other, so they need the same right tile: block 0's producer asks TMA for it once, into
// every block's shared memory (multicast), half the reads of right from L2. The cluster's blocks walk in step: a
// set's empty barrier in each block takes an arrival from every consumer warpgroup of the cluster. A cluster's
// last square may lie past the rows: TMA gives it 0s and nothing is stored.
// The store, the plain epilogue with no add_onto and no splits (the worklog's kernel 10): each consumer warpgroup
// writes its 64 rows (+ bias) as bfloat16 into its room in TMA's 128-byte swizzled layout (a box of 64 columns at a
// time, no bank conflicts), one thread asks TMA to store the boxes, and the warpgroup goes straight on to the next
// unit's WGMMAs; it writes the room again only once TMA has read it. Whole rows to memory, past the edges clipped
// by TMA, and nobody waits for them: measured on an H100 PCIe, the staged store below held the tensor cores up as
// long as the walk itself on a sum of 768 (qkv 0.157 ms, 0.081 with no store at all); stored straight from the
// registers, each row's 16 bytes of a 32-byte sector made it slower still (0.197).
// An epilogue with a tensor of its own beside output says so by WGMMA_SIDE, and gives side_ptr() (the tensor) and
// side_of (WGMMA_MATMUL_BF16_SIDE_WRITTEN: what it writes there for each pair) or with_side (_READ: output from each
// pair and the side's pair). Written: after output, the side's values go through the room and TMA too. Read: before
// the walk TMA brings the side's 64 rows into the room, and each pair's output is worked out there. GELU's two do
// (gelu_bf16.cuh); an epilogue with no WGMMA_SIDE is stored as below.
// The store, the others (add_onto, splits, an epilogue with no WGMMA_SIDE): each consumer warpgroup puts its sums (+ bias) into
// its own room of shared memory as floats,
// WGMMA_MATMUL_BF16_STAGED_ROWS rows a round, and stores them row by row, WGMMA_MATMUL_BF16_STORE_VALUES neighbouring
// values a thread: the epilogue is called on each pair there, so its own reads and writes of other tensors are
// neighbours too, and a warp's stores are one stretch of a row. Stored straight from WGMMA's layout, a thread's
// pairs are 8 rows apart: half of every 32-byte sector wasted, and GELU's extra tensor the same. The room is apart
// from the tiles', which the producer is filling again.
// SPLITS (a launch parameter, default WGMMA_MATMUL_BF16_SPLITS, swept by the tune): blocks a square's sum is split
// over. A product of few squares and a long sum (768 x 768 out of 4096 values of k: 72 squares of 64 x 128 on 132
// SMs) leaves SMs idle; split in S, each split's block walks 1 / S of the steps and writes its float sums into its
// slice of a scratch {S, rows, columns}, and wgmma_matmul_bf16_splits_sum_kernel adds the S slices in order (+ bias),
// then the epilogue (+ output). Fixed for a launch, so the bits never change between runs; they differ from 1
// split's, whose sums are added in another order.
// A product's setting is one type, WgmmaMatmulBf16Setting<ROW_GROUPS, COLUMNS, STAGES, SPLITS, BLOCK_ROWS_PER_GROUP>,
// so each caller picks its own (model_bf16.cuh: one per GPT-2 product, from tune_wgmma_matmul_bf16's sweep); not
// given: the defaults. wgmma_matmul_bf16_at takes the last two at run time, for the tune and the tests.
// The blocks are numbered in groups of block_rows_per_group rows of squares, down a group first (a launch
// parameter, default WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, swept by the tune): the blocks running at once then
// share a few rows of left and a few columns of right, which stay in L2. Row after row (1), a product whose right
// is larger than L2 (the head's wte, 77 MB) reads all of right again for every row of squares.
// The epilogue: what is done to each pair of results (two neighbouring columns of a row) as they are stored, after
// the bias and before adding onto output. A functor, a template parameter of the kernel, so it costs nothing when it
// does nothing: WgmmaMatmulBf16Store (the default) gives the pair back as it is; bf16/gelu_bf16.cuh's apply GELU
// there, so the MLP's GELU needs no pass of its own over memory. An epilogue is
//     struct { ...its pointers...; __device__ float2 operator()(float2 pair, size_t index) const; }
// called with the pair in float and the index of its first value in output (row x columns + column); what it gives
// back is stored (rounded to bfloat16), and it may read or write other tensors of output's shape at that index.
// WGMMA and TMA exist only on Hopper: run it on an H100 (compute capability 9.0), built with -arch=sm_90a; nothing
// here checks. Other GPUs: mma_matmul_bf16.cuh (sm_80 or later), fma_matmul_bf16.cuh (any). The block size is
// fixed by the tiles, so this takes none.
#ifndef WGMMA_MATMUL_BF16_CUH
#define WGMMA_MATMUL_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <type_traits>               // std::is_same: the plain epilogue is stored by TMA
#include <cuda/barrier>              // fence_proxy_async_shared_cta; cuda::barrier for the kernels built on this file
#include "../cuda_common.h"             // cudaCheck, HARDWARE_THREADS_PER_WARP
#include "../../common/tensor.cuh"

// The squares and steps: our choices. The hardware's numbers (WARPGROUP_*, WGMMA_*, TMA_*) are in cuda_common.h.
// The index math is written for these numbers: change them together with it.
#define WGMMA_MATMUL_BF16_COLUMNS 128                                                             // the product's default columns per block, its WGMMA's N: 64, 128 or 256 (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_STAGES 3                                                                // the product's default sets of tiles: 1 to WGMMA_MATMUL_BF16_MAX_STAGES. 128 x 3: a micro-batch's GPT-2 products 10.03 ms against 12.40 at 64 x 1 (H100 NVL)
#define WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP 64                                                 // the default block order: rows of squares a group of blocks runs down before the next column (1: row after row). 64: a micro-batch's products 9.93 ms against 10.96 row after row, the head 1.04 against 1.71 (H100 NVL)
#define WGMMA_MATMUL_BF16_ROW_GROUPS 1                                                             // the product's default warpgroups a block, 64 rows each: 1 or 2 (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_MAX_ROW_GROUPS 2                                                         // the most: 2 x 128 threads, each warpgroup's sums up to 128 floats (COLUMNS 256), inside a block's registers
#define WGMMA_MATMUL_BF16_SPLITS 1                                                                // the product's default splits of the sum: 1, none (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK 256                                        // the splits' sum's default block size (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_CLUSTER 1                                                               // the product's default blocks a cluster, sharing each right tile: 1 (none) or 2 (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_MAX_CLUSTER 2                                                           // the most: the worklog found larger clusters slower
#define WGMMA_MATMUL_BF16_BLOCKS 0                                                                // the product's default blocks a launch: 0, as many as the GPU runs at once; more than the squares: one a square (tune_wgmma_matmul_bf16 sweeps it)
#define WGMMA_MATMUL_BF16_STAGED_ROWS 32                                                          // rows a consumer warpgroup stages and stores a round: 16, 32 or 64 (64: 135 KB of room at 2 x 256 columns, no room for 3 sets of tiles)
#define WGMMA_MATMUL_BF16_PRODUCER_REGISTERS 40                                                    // with 2 consumer warpgroups: the producer's registers a thread (setmaxnreg); the consumers take the rest: 232 each. 24 (the least) spilled the producer's unit math
#define WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS 1                                                    // warpgroups of a block that only load tiles: the first one (its thread 0 asks TMA)
#define WGMMA_MATMUL_BF16_IN_FLIGHT 1                                                              // steps of WGMMAs left running while the threads go on (STAGES - 1 at most: the set must be free)
#define WGMMA_MATMUL_BF16_MAX_STAGES 4                                                            // the most; with the store's room a square must fit HARDWARE_MAX_SHARED_BYTES_PER_BLOCK (wgmma_matmul_bf16_shared_bytes): all do but 2 warpgroups x 256 columns x 4 sets
#define WGMMA_MATMUL_BF16_BLOCK_COLUMNS 64                                                        // a 64-column tile: the one attention_bf16, attention_flash_bf16 and classifier_bf16 use
#define WGMMA_MATMUL_BF16_BLOCK_K ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))       // values of k a step loads: one swizzled row, 64
#define WGMMA_MATMUL_BF16_COLUMN_GROUPS (WGMMA_MATMUL_BF16_BLOCK_COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS)   // groups of 8 across a 64-column tile: 8, 32 sums a thread
#define WGMMA_MATMUL_BF16_ROW_VALUES ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))    // a TMA box row: one swizzled row, 64 values
#define WGMMA_MATMUL_BF16_VALUE_MULTIPLE ((int)(HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16)))       // a matrix row's multiple: whole 16-byte cells, 8
#define WGMMA_MATMUL_BF16_LEFT_TILE_VALUES (HARDWARE_WGMMA_M * WGMMA_MATMUL_BF16_BLOCK_K)                  // a left tile: 64 rows of 64 k
#define WGMMA_MATMUL_BF16_STAGED_ROW_PADDING 8                                                    // floats past COLUMNS in a staged row: the 8 rows a warp writes start in other banks, rows stay on 16 bytes
#define WGMMA_MATMUL_BF16_SIDE_STAGED -1                                                           // an epilogue's WGMMA_SIDE: none given, stored by the staged store
#define WGMMA_MATMUL_BF16_SIDE_NONE 0                                                             // ... no tensor of its own (the plain store): stored by TMA
#define WGMMA_MATMUL_BF16_SIDE_WRITTEN 1                                                          // ... writes one beside output (side_of each value): both stored by TMA
#define WGMMA_MATMUL_BF16_SIDE_READ 2                                                             // ... reads one beside output (with_side each value): brought by TMA, then output stored by TMA
#define WGMMA_MATMUL_BF16_STORE_VALUES 4                                                          // neighbouring values a thread stores at once: two pairs, 8 bytes
static_assert(WGMMA_MATMUL_BF16_BLOCK_COLUMNS == WGMMA_MATMUL_BF16_ROW_VALUES, "a 64-column tile turned round is one swizzled row");
static_assert(WGMMA_MATMUL_BF16_BLOCK_K % HARDWARE_WGMMA_K_BF16 == 0, "each step is whole WGMMAs");
static_assert(HARDWARE_WGMMA_M % WGMMA_MATMUL_BF16_STAGED_ROWS == 0, "a warpgroup's 64 rows are staged in whole rounds");
static_assert(HARDWARE_WGMMA_M == WGMMA_MATMUL_BF16_ROW_VALUES, "a turned left tile's row (64 rows) is one swizzled row, as a tile's row of 64 k is");

/// The plain epilogue: each pair of results is stored as it is.
struct WgmmaMatmulBf16Store {
    static constexpr int WGMMA_SIDE = WGMMA_MATMUL_BF16_SIDE_NONE;   // no tensor of its own: stored by TMA

    /// Gives the pair back unchanged.
    ///
    ///   pair_val    two neighbouring results, in float
    ///   index_val   the first one's index in output (unused)
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const { return pair_val; }
};

/// Gives an epilogue's WGMMA_SIDE (WGMMA_MATMUL_BF16_SIDE_*): how its product is stored. An epilogue that gives none:
/// WGMMA_MATMUL_BF16_SIDE_STAGED, the staged store calling its operator() on each pair.
template <typename EPILOGUE, typename = void>
struct WgmmaMatmulBf16SideOf {
    static constexpr int value = WGMMA_MATMUL_BF16_SIDE_STAGED;
};
template <typename EPILOGUE>
struct WgmmaMatmulBf16SideOf<EPILOGUE, std::void_t<decltype(EPILOGUE::WGMMA_SIDE)>> {
    static constexpr int value = EPILOGUE::WGMMA_SIDE;
};

/// A product's setting, as one type a caller names once (model_bf16.cuh: one per GPT-2 product): the square's
/// warpgroups (64 rows each) and columns, the sets of tiles and the cluster, fixed when built (each makes its own
/// kernel); the splits of the sum, the block order and the blocks, given to the kernel at run time. Not given: the
/// defaults.
template <int ROW_GROUPS_ARG = WGMMA_MATMUL_BF16_ROW_GROUPS, int COLUMNS_ARG = WGMMA_MATMUL_BF16_COLUMNS,
          int STAGES_ARG = WGMMA_MATMUL_BF16_STAGES, int SPLITS_ARG = WGMMA_MATMUL_BF16_SPLITS,
          int BLOCK_ROWS_PER_GROUP_ARG = WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, int CLUSTER_ARG = WGMMA_MATMUL_BF16_CLUSTER,
          int BLOCKS_ARG = WGMMA_MATMUL_BF16_BLOCKS>
struct WgmmaMatmulBf16Setting {
    static constexpr int ROW_GROUPS = ROW_GROUPS_ARG;
    static constexpr int COLUMNS = COLUMNS_ARG;
    static constexpr int STAGES = STAGES_ARG;
    static constexpr int SPLITS = SPLITS_ARG;
    static constexpr int BLOCK_ROWS_PER_GROUP = BLOCK_ROWS_PER_GROUP_ARG;
    static constexpr int CLUSTER = CLUSTER_ARG;
    static constexpr int BLOCKS = BLOCKS_ARG;
};

/// Gives a consumer warpgroup's room for the store: a round of the staged store's floats, or its 64 rows as bfloat16
/// for TMA, whichever is larger, rounded up to whole HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atoms (TMA's swizzle needs
/// each box on one).
template <int COLUMNS>
__host__ __device__ constexpr size_t wgmma_matmul_bf16_room_bytes() {
    return CEIL_DIV((WGMMA_MATMUL_BF16_STAGED_ROWS * (COLUMNS + WGMMA_MATMUL_BF16_STAGED_ROW_PADDING) * sizeof(float) >
                             HARDWARE_WGMMA_M * COLUMNS * sizeof(nv_bfloat16)
                         ? WGMMA_MATMUL_BF16_STAGED_ROWS * (COLUMNS + WGMMA_MATMUL_BF16_STAGED_ROW_PADDING) * sizeof(float)
                         : HARDWARE_WGMMA_M * COLUMNS * sizeof(nv_bfloat16)),
                    (size_t)HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) * HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
}

/// Gives the dynamic shared memory a block takes: STAGES sets of tiles (a left tile a warpgroup and the right tile),
/// each consumer warpgroup's room for a round of the store, and one atom of room to start the tiles on one.
template <int COLUMNS, int STAGES, int ROW_GROUPS>
constexpr size_t wgmma_matmul_bf16_shared_bytes() {
    return (size_t)STAGES * (ROW_GROUPS * WGMMA_MATMUL_BF16_LEFT_TILE_VALUES + COLUMNS * WGMMA_MATMUL_BF16_BLOCK_K) * sizeof(nv_bfloat16) +
           (size_t)ROW_GROUPS * wgmma_matmul_bf16_room_bytes<COLUMNS>() + HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
}

// ── the product ───────────────────────────────────────────────────────────────
template <typename SETTING = WgmmaMatmulBf16Setting<>, typename EPILOGUE = WgmmaMatmulBf16Store>
void wgmma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                       const Tensor& bias_tensor = Tensor(), bool add_onto_val = false, bool left_turned_val = false,
                       bool right_turned_val = false, const EPILOGUE& epilogue_val = EPILOGUE());
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER = WGMMA_MATMUL_BF16_CLUSTER, typename EPILOGUE = WgmmaMatmulBf16Store>
void wgmma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                          const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                          const EPILOGUE& epilogue_val, int splits_val, int block_rows_per_group_val,
                          int sum_threads_per_block_val = WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK,
                          int blocks_val = WGMMA_MATMUL_BF16_BLOCKS);

// ── for it ────────────────────────────────────────────────────────────────────
CUtensorMap wgmma_matmul_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                         unsigned box_rows_val);
void wgmma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                    const Tensor& bias_tensor, bool left_turned_val, bool right_turned_val, size_t* rows_ptr,
                                    size_t* columns_ptr, size_t* sum_length_ptr);


// ── the product ───────────────────────────────────────────────────────────────

using WgmmaMatmulBf16Barrier = cuda::barrier<cuda::thread_scope_block>;

template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE = WgmmaMatmulBf16Store, int COLUMNS = WGMMA_MATMUL_BF16_COLUMNS,
          int STAGES = WGMMA_MATMUL_BF16_STAGES, int ROW_GROUPS = WGMMA_MATMUL_BF16_ROW_GROUPS, int CLUSTER = WGMMA_MATMUL_BF16_CLUSTER>
__global__ void __launch_bounds__((WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         const __grid_constant__ CUtensorMap output_map_val,
                         const __grid_constant__ CUtensorMap side_map_val, nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr,
                         int rows_val, int columns_val, int sum_length_val, bool add_onto_val, EPILOGUE epilogue_val,
                         int block_rows_per_group_val, int splits_val, float* split_sums_ptr);
template <typename EPILOGUE>
__global__ void wgmma_matmul_bf16_splits_sum_kernel(nv_bfloat16* output_ptr, const float* split_sums_ptr,
                                                    const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val,
                                                    int splits_val, bool add_onto_val, EPILOGUE epilogue_val);
template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE, int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER>
void wgmma_matmul_bf16_launch(size_t total_units_val, int blocks_val, const CUtensorMap& left_map_val, const CUtensorMap& right_map_val,
                              const CUtensorMap& output_map_val, const CUtensorMap& side_map_val,
                              nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, int rows_val, int columns_val,
                              int sum_length_val, bool add_onto_val, const EPILOGUE& epilogue_val, int block_rows_per_group_val,
                              int splits_val, float* split_sums_ptr);
__device__ __forceinline__ uint64_t wgmma_matmul_bf16_descriptor(const nv_bfloat16* tile_ptr,
                                                                 uint32_t leading_cells_val = HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);
template <bool LEFT_TURNED, bool RIGHT_TURNED, int GROUPS>
__device__ __forceinline__ void wgmma_matmul_bf16_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                               const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr,
                                                               uint32_t right_leading_cells_val = HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);

/// Writes output = left · rightᵀ (+ bias) (+ output) at SETTING (WgmmaMatmulBf16Setting: the square, the sets of
/// tiles, the cluster, the splits, the block order and the blocks; not given: the defaults): wgmma_matmul_bf16_at
/// with them.
///
///   output_tensor      {..., columns}, rows x columns values: written over, or added onto
///   left_tensor        {..., sum_length} (rows = the other dims): x in a linear layer; or turned {..., rows}
///                      (sum_length = the other dims)
///   right_tensor       {..., sum_length} (columns = the other dims): W; or turned {..., columns}
///   bias_tensor        {columns}, or Tensor() for none
///   add_onto_val       true: added onto what output holds; false: written over
///   left_turned_val, right_turned_val   true: that tensor holds its matrix turned round
///   epilogue_val       what is done to each pair of results as it is stored (see the top of the file); not given:
///                      WgmmaMatmulBf16Store, nothing
/// Every tensor's last dim a multiple of 8 (TMA reads rows of whole 16-byte pieces); every tensor bfloat16.
template <typename SETTING, typename EPILOGUE>
inline void wgmma_matmul_bf16(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                              const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                              const EPILOGUE& epilogue_val) {
    wgmma_matmul_bf16_at<SETTING::ROW_GROUPS, SETTING::COLUMNS, SETTING::STAGES, SETTING::CLUSTER>(
        output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val, right_turned_val, epilogue_val,
        SETTING::SPLITS, SETTING::BLOCK_ROWS_PER_GROUP, WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK, SETTING::BLOCKS);
}

/// Writes output = left · rightᵀ (+ bias) (+ output): checks, describes left's and right's tiles for TMA, then
/// launches wgmma_matmul_bf16_kernel made for the two turned flags, COLUMNS, STAGES, ROW_GROUPS and CLUSTER, its
/// blocks walking the squares of output and the splits of the sum; split, the splits' sums added by
/// wgmma_matmul_bf16_splits_sum_kernel.
///
///   output_tensor ... epilogue_val   as in wgmma_matmul_bf16
///   splits_val                 blocks a square's sum is split over (at least 1; more than the steps of
///                              WGMMA_MATMUL_BF16_BLOCK_K: one a step)
///   block_rows_per_group_val   the block order: rows of squares a group runs down (at least 1)
///   sum_threads_per_block_val  the splits' sum's block size; not given: WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK
///   blocks_val                 blocks launched: 0, as many as the GPU runs at once; more than the squares: one a
///                              square; not given: WGMMA_MATMUL_BF16_BLOCKS
/// ROW_GROUPS, COLUMNS, STAGES, CLUSTER: the square's warpgroups and columns, the sets of tiles and the blocks of a
/// cluster (see the top of the file). Every tensor's last dim a multiple of 8 (TMA reads rows of whole 16-byte
/// pieces); every tensor bfloat16.
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, typename EPILOGUE>
inline void wgmma_matmul_bf16_at(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
                                 const Tensor& bias_tensor, bool add_onto_val, bool left_turned_val, bool right_turned_val,
                                 const EPILOGUE& epilogue_val, int splits_val, int block_rows_per_group_val,
                                 int sum_threads_per_block_val, int blocks_val) {
    static_assert(COLUMNS == 64 || COLUMNS == 128 || COLUMNS == 256, "COLUMNS is one WGMMA's N this file has: 64, 128 or 256");
    static_assert(STAGES >= 1 && STAGES <= WGMMA_MATMUL_BF16_MAX_STAGES, "STAGES is 1 to WGMMA_MATMUL_BF16_MAX_STAGES");
    static_assert(ROW_GROUPS >= 1 && ROW_GROUPS <= WGMMA_MATMUL_BF16_MAX_ROW_GROUPS, "ROW_GROUPS is 1 to WGMMA_MATMUL_BF16_MAX_ROW_GROUPS");
    static_assert(CLUSTER >= 1 && CLUSTER <= WGMMA_MATMUL_BF16_MAX_CLUSTER, "CLUSTER is 1 to WGMMA_MATMUL_BF16_MAX_CLUSTER");
    // 1. check the tensors and get the sizes; any of them wrong: stop with an error
    size_t rows_val = 0, columns_val = 0, sum_length_val = 0;
    wgmma_matmul_bf16_check_shapes(output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val,
                                   &rows_val, &columns_val, &sum_length_val);
    if (block_rows_per_group_val < 1 || splits_val < 1 || blocks_val < 0) {
        fprintf(stderr, "error: wgmma_matmul_bf16's block_rows_per_group and splits must be 1 or more and blocks 0 or more; "
                        "got %d, %d and %d\n", block_rows_per_group_val, splits_val, blocks_val);
        exit(EXIT_FAILURE);
    }
    int total_steps_val = (int)CEIL_DIV(sum_length_val, (size_t)WGMMA_MATMUL_BF16_BLOCK_K);
    if (splits_val > total_steps_val) splits_val = total_steps_val > 0 ? total_steps_val : 1;   // at most one a step

    // 2. TMA's description of left's and right's tiles: {lines, sum_length} in boxes of {lines, 64 k} (64 rows of
    //    left, COLUMNS of right), or turned {sum_length, lines} in boxes of {64 k, 64 lines} (a turned right tile is
    //    COLUMNS / 64 of them); a block's left tile is ROW_GROUPS boxes, one a warpgroup
    const nv_bfloat16* left_ptr = left_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* right_ptr = right_tensor.data_ptr<nv_bfloat16>();
    CUtensorMap left_map_val = left_turned_val ? wgmma_matmul_bf16_tensor_map(left_ptr, sum_length_val, rows_val, WGMMA_MATMUL_BF16_BLOCK_K)
                                               : wgmma_matmul_bf16_tensor_map(left_ptr, rows_val, sum_length_val, HARDWARE_WGMMA_M);
    CUtensorMap right_map_val = right_turned_val ? wgmma_matmul_bf16_tensor_map(right_ptr, sum_length_val, columns_val, WGMMA_MATMUL_BF16_BLOCK_K)
                                                 : wgmma_matmul_bf16_tensor_map(right_ptr, columns_val, sum_length_val, COLUMNS);

    //    and the output's, for the TMA store: boxes of {64 rows, 64 columns}, the same swizzle
    CUtensorMap output_map_val = wgmma_matmul_bf16_tensor_map(output_tensor.data_ptr<nv_bfloat16>(), rows_val, columns_val, HARDWARE_WGMMA_M);
    //    and the epilogue's side tensor's, if it has one (output's shape): the same boxes
    CUtensorMap side_map_val = output_map_val;
    if constexpr (WgmmaMatmulBf16SideOf<EPILOGUE>::value == WGMMA_MATMUL_BF16_SIDE_WRITTEN ||
                  WgmmaMatmulBf16SideOf<EPILOGUE>::value == WGMMA_MATMUL_BF16_SIDE_READ)
        side_map_val = wgmma_matmul_bf16_tensor_map(epilogue_val.side_ptr(), rows_val, columns_val, HARDWARE_WGMMA_M);

    // 3. split: the scratch, a {rows, columns} slice of float sums per split, made the first time and grown when a
    //    product needs more; the bias goes in the splits' sum, not the blocks
    nv_bfloat16* output_ptr = output_tensor.data_ptr<nv_bfloat16>();
    const nv_bfloat16* bias_ptr = bias_tensor.data_ptr() != nullptr ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;
    static Tensor split_sums_tensor;
    float* split_sums_ptr = nullptr;
    if (splits_val > 1) {
        if ((size_t)split_sums_tensor.numel() < (size_t)splits_val * rows_val * columns_val)
            split_sums_tensor = Tensor::empty({(size_t)splits_val * rows_val * columns_val});
        split_sums_ptr = split_sums_tensor.data();
    }
    const nv_bfloat16* block_bias_ptr = splits_val > 1 ? nullptr : bias_ptr;

    // 4. launch: a cluster's unit of work is CLUSTER squares one under the other, for one split; the units across
    //    and down, rounded up, times the splits
    size_t total_unit_columns_val = CEIL_DIV(columns_val, (size_t)COLUMNS);
    size_t total_unit_rows_val = CEIL_DIV(rows_val, (size_t)(CLUSTER * ROW_GROUPS * HARDWARE_WGMMA_M));
    size_t total_units_val = total_unit_columns_val * total_unit_rows_val * (size_t)splits_val;
    if (!left_turned_val && !right_turned_val)
        wgmma_matmul_bf16_launch<false, false, EPILOGUE, COLUMNS, STAGES, ROW_GROUPS, CLUSTER>(
            total_units_val, blocks_val, left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, block_bias_ptr, (int)rows_val,
            (int)columns_val,
            (int)sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val, splits_val, split_sums_ptr);
    else if (!left_turned_val && right_turned_val)
        wgmma_matmul_bf16_launch<false, true, EPILOGUE, COLUMNS, STAGES, ROW_GROUPS, CLUSTER>(
            total_units_val, blocks_val, left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, block_bias_ptr, (int)rows_val,
            (int)columns_val,
            (int)sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val, splits_val, split_sums_ptr);
    else if (left_turned_val && !right_turned_val)
        wgmma_matmul_bf16_launch<true, false, EPILOGUE, COLUMNS, STAGES, ROW_GROUPS, CLUSTER>(
            total_units_val, blocks_val, left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, block_bias_ptr, (int)rows_val,
            (int)columns_val,
            (int)sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val, splits_val, split_sums_ptr);
    else
        wgmma_matmul_bf16_launch<true, true, EPILOGUE, COLUMNS, STAGES, ROW_GROUPS, CLUSTER>(
            total_units_val, blocks_val, left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, block_bias_ptr, (int)rows_val,
            (int)columns_val,
            (int)sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val, splits_val, split_sums_ptr);

    // 5. split: the splits' sums in order (+ bias), the epilogue (+ output), WGMMA_MATMUL_BF16_STORE_VALUES values a thread
    if (splits_val > 1) {
        check_threads_per_block(sum_threads_per_block_val, "wgmma_matmul_bf16_splits_sum_kernel");
        size_t total_stores_val = rows_val * columns_val / WGMMA_MATMUL_BF16_STORE_VALUES;
        wgmma_matmul_bf16_splits_sum_kernel<EPILOGUE><<<(unsigned)CEIL_DIV(total_stores_val, (size_t)sum_threads_per_block_val),
                                                        sum_threads_per_block_val>>>(
            output_ptr, split_sums_ptr, bias_ptr, rows_val, columns_val, splits_val, add_onto_val, epilogue_val);
        cudaCheck(cudaGetLastError());
    }
}

/// Launches one wgmma_matmul_bf16_kernel: clusters of CLUSTER blocks, each block a producer and ROW_GROUPS consumer
/// warpgroups, with the dynamic shared memory its sets of tiles and its store take. The first time for each
/// kernel: asks CUDA to allow that much (past the 48 KB a block gets without asking), and asks how many of its
/// blocks the GPU runs at once (SMs x blocks an SM holds).
///
///   total_units_val               units of work: a cluster's CLUSTER squares, for one split
///   blocks_val                    blocks to launch: 0, as many as run at once; never more than a cluster a unit
///   the others                    as in wgmma_matmul_bf16_kernel
template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE, int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER>
inline void wgmma_matmul_bf16_launch(size_t total_units_val, int blocks_val, const CUtensorMap& left_map_val, const CUtensorMap& right_map_val,
                              const CUtensorMap& output_map_val, const CUtensorMap& side_map_val,
                                     nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr, int rows_val, int columns_val,
                                     int sum_length_val, bool add_onto_val, const EPILOGUE& epilogue_val, int block_rows_per_group_val,
                                     int splits_val, float* split_sums_ptr) {
    constexpr size_t SHARED_BYTES = wgmma_matmul_bf16_shared_bytes<COLUMNS, STAGES, ROW_GROUPS>();
    constexpr int THREADS = (WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS;
    static_assert(SHARED_BYTES <= HARDWARE_MAX_SHARED_BYTES_PER_BLOCK, "the sets of tiles and the store's room fit a block's shared memory");
    // 1. the first time: the shared memory allowed, and the blocks the GPU runs at once
    auto kernel_ptr = wgmma_matmul_bf16_kernel<LEFT_TURNED, RIGHT_TURNED, EPILOGUE, COLUMNS, STAGES, ROW_GROUPS, CLUSTER>;
    static int blocks_at_once_val = 0;                       // one for each kernel this template makes
    if (blocks_at_once_val == 0) {
        int device_index_val = 0, total_sms_val = 0, blocks_per_sm_val = 0;
        cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SHARED_BYTES));
        cudaCheck(cudaGetDevice(&device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&total_sms_val, cudaDevAttrMultiProcessorCount, device_index_val));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_val, kernel_ptr, THREADS, SHARED_BYTES));
        blocks_at_once_val = total_sms_val * (blocks_per_sm_val > 0 ? blocks_per_sm_val : 1);
    }
    // 2. the clusters: the blocks asked for in whole clusters, at least one, at most one a unit
    size_t wanted_blocks_val = blocks_val > 0 ? (size_t)blocks_val : (size_t)blocks_at_once_val;
    size_t total_clusters_val = wanted_blocks_val / CLUSTER;
    if (total_clusters_val < 1) total_clusters_val = 1;
    if (total_clusters_val > total_units_val) total_clusters_val = total_units_val;
    unsigned total_blocks_val = (unsigned)(total_clusters_val * CLUSTER);
    // 3. launch; with clusters, by cudaLaunchKernelEx, which takes the cluster's size
    if (CLUSTER == 1) {
        kernel_ptr<<<total_blocks_val, THREADS, SHARED_BYTES>>>(left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, bias_ptr, rows_val, columns_val,
                                                                sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val,
                                                                splits_val, split_sums_ptr);
        cudaCheck(cudaGetLastError());
    } else {
        cudaLaunchConfig_t config_val = {};
        cudaLaunchAttribute attributes_list[1];
        attributes_list[0].id = cudaLaunchAttributeClusterDimension;
        attributes_list[0].val.clusterDim.x = CLUSTER;
        attributes_list[0].val.clusterDim.y = 1;
        attributes_list[0].val.clusterDim.z = 1;
        config_val.gridDim = dim3(total_blocks_val);
        config_val.blockDim = dim3(THREADS);
        config_val.dynamicSmemBytes = SHARED_BYTES;
        config_val.attrs = attributes_list;
        config_val.numAttrs = 1;
        cudaCheck(cudaLaunchKernelEx(&config_val, kernel_ptr, left_map_val, right_map_val, output_map_val, side_map_val, output_ptr, bias_ptr, rows_val, columns_val,
                                     sum_length_val, add_onto_val, epilogue_val, block_rows_per_group_val, splits_val, split_sums_ptr));
    }
}

/// Gives a shared-memory object's address as PTX's shared::cta instructions take it.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t wgmma_matmul_bf16_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void wgmma_matmul_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(wgmma_matmul_bf16_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming: its phase completes when they
/// are in.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void wgmma_matmul_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(wgmma_matmul_bf16_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Arrives once at a barrier of this block.
///
///   barrier_ptr   the barrier
__device__ __forceinline__ void wgmma_matmul_bf16_barrier_arrive(uint64_t* barrier_ptr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(wgmma_matmul_bf16_shared_address(barrier_ptr)), "r"(1) : "memory");
}

/// Arrives once at the same barrier in another block of the cluster (or this one): the barrier at this address of
/// block cluster_block_val's shared memory.
///
///   barrier_ptr         the barrier, as it sits in this block
///   cluster_block_val   which block of the cluster: 0 .. CLUSTER - 1
__device__ __forceinline__ void wgmma_matmul_bf16_barrier_arrive_in_cluster(uint64_t* barrier_ptr, uint32_t cluster_block_val) {
    asm volatile("{\n"
                 ".reg .b32 remote_address;\n"
                 "mapa.shared::cluster.u32 remote_address, %0, %1;\n"
                 "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote_address], %2;\n"
                 "}\n"
                 ::"r"(wgmma_matmul_bf16_shared_address(barrier_ptr)), "r"(cluster_block_val), "r"(1) : "memory");
}

/// Waits until a barrier's phase of this parity is complete: the caller keeps count of the phases (the first is
/// 0, the next 1, then 0 again), so no token is passed round.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void wgmma_matmul_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    uint32_t barrier_address_val = wgmma_matmul_bf16_shared_address(barrier_ptr);
    uint32_t done_val;
    do {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n"
                     "}\n"
                     : "=r"(done_val) : "r"(barrier_address_val), "r"(parity_val) : "memory");
    } while (done_val == 0);
}

/// Asks TMA for one box of a matrix into this block's shared memory, its bytes counted at a barrier.
///
///   tile_ptr                       where the box goes, in shared memory
///   map_ptr                        TMA's description of the matrix
///   column_val, row_val            the box's corner, as the matrix sits
///   barrier_ptr                    the barrier that expects the bytes
__device__ __forceinline__ void wgmma_matmul_bf16_copy_tile(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                            int row_val, uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(wgmma_matmul_bf16_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(wgmma_matmul_bf16_shared_address(barrier_ptr))
                 : "memory");
}

/// Asks TMA for one box of a matrix into the shared memory of several blocks of the cluster at once (multicast):
/// read from L2 once, written at the same place in each block, its bytes counted at each block's barrier there.
///
///   tile_ptr, map_ptr, column_val, row_val, barrier_ptr   as in wgmma_matmul_bf16_copy_tile, as they sit in this block
///   cluster_blocks_mask_val                               bit b set: block b of the cluster gets the box
__device__ __forceinline__ void wgmma_matmul_bf16_copy_tile_to_cluster(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                                       int row_val, uint64_t* barrier_ptr,
                                                                       uint16_t cluster_blocks_mask_val) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                 "[%0], [%1, {%2, %3}], [%4], %5;\n"
                 ::"r"(wgmma_matmul_bf16_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(wgmma_matmul_bf16_shared_address(barrier_ptr)), "h"(cluster_blocks_mask_val)
                 : "memory");
}

/// Walks this block's squares of output, square after square (persistent), with a producer warpgroup and ROW_GROUPS
/// consumer warpgroups, 64 rows each: the producer's thread 0 asks TMA for each step's tiles of left and of right
/// into one of STAGES sets in shared memory, as soon as the consumers free the set; the consumers' tensor cores add
/// the tiles' product into the sums in registers, then each consumer warpgroup stores its rows while the producer
/// goes on to the next square.
///
///   left_map_val, right_map_val                  TMA's descriptions of left's and right's tiles
///   output_ptr, bias_ptr                         as in wgmma_matmul_bf16; bias nullptr for none
///   rows_val, columns_val, sum_length_val        the shapes
///   add_onto_val                                 as in wgmma_matmul_bf16
///   epilogue_val                                 as in wgmma_matmul_bf16
///   block_rows_per_group_val                     the order of the units: rows of units a group runs down
///   splits_val, split_sums_ptr                   the splits of the sum and their scratch {splits, rows, columns};
///                                                nullptr when 1: output written here
/// LEFT_TURNED, RIGHT_TURNED: how left and right sit: which corner TMA's box starts at, how far apart two WGMMAs'
/// 16 values of k are in a tile, and the WGMMA's transpose bits. EPILOGUE: the epilogue's type. COLUMNS, STAGES,
/// ROW_GROUPS, CLUSTER: as at the top of the file.
template <bool LEFT_TURNED, bool RIGHT_TURNED, typename EPILOGUE, int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER>
__global__ void __launch_bounds__((WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS)
wgmma_matmul_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                         const __grid_constant__ CUtensorMap output_map_val,
                         const __grid_constant__ CUtensorMap side_map_val, nv_bfloat16* output_ptr, const nv_bfloat16* bias_ptr,
                         int rows_val, int columns_val, int sum_length_val, bool add_onto_val, EPILOGUE epilogue_val,
                         int block_rows_per_group_val, int splits_val, float* split_sums_ptr) {
    namespace cde = cuda::device::experimental;
    constexpr int COLUMN_GROUPS = COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS;                                   // groups of 8 across the square
    constexpr int RIGHT_TILE_VALUES = COLUMNS * WGMMA_MATMUL_BF16_BLOCK_K;                                  // a right tile: COLUMNS lines of 64 k
    constexpr int BLOCK_ROWS = ROW_GROUPS * HARDWARE_WGMMA_M;                                              // a square's rows
    constexpr int UNIT_ROWS = CLUSTER * BLOCK_ROWS;                                                        // a cluster's unit: its squares one under the other
    constexpr int LEFT_VALUES = ROW_GROUPS * WGMMA_MATMUL_BF16_LEFT_TILE_VALUES;                            // a set's left tiles, one a warpgroup
    constexpr int SET_VALUES = LEFT_VALUES + RIGHT_TILE_VALUES;                                             // one set: left tiles, then right tile
    constexpr int IN_FLIGHT = STAGES > WGMMA_MATMUL_BF16_IN_FLIGHT ? WGMMA_MATMUL_BF16_IN_FLIGHT : STAGES - 1;   // steps of WGMMAs left running
    constexpr int RIGHT_BOXES = RIGHT_TURNED ? COLUMNS / WGMMA_MATMUL_BF16_ROW_VALUES : 1;                  // TMA boxes in a right tile
    constexpr int RIGHT_BOX_VALUES = RIGHT_TURNED ? WGMMA_MATMUL_BF16_ROW_VALUES * WGMMA_MATMUL_BF16_BLOCK_K : RIGHT_TILE_VALUES;
    constexpr uint32_t SET_BYTES = SET_VALUES * sizeof(nv_bfloat16);                                         // what TMA brings a block a step
    constexpr int STAGED_ROW_FLOATS = COLUMNS + WGMMA_MATMUL_BF16_STAGED_ROW_PADDING;                       // a staged row
    constexpr int STAGED_FLOATS = WGMMA_MATMUL_BF16_STAGED_ROWS * STAGED_ROW_FLOATS;                        // a consumer warpgroup's room for a round
    constexpr int STORE_ROUNDS = HARDWARE_WGMMA_M / WGMMA_MATMUL_BF16_STAGED_ROWS;                          // rounds to store a warpgroup's 64 rows
    constexpr int STORES_PER_ROW = COLUMNS / WGMMA_MATMUL_BF16_STORE_VALUES;                                // a row's 8-byte stores
    constexpr int PRODUCER_THREADS = WGMMA_MATMUL_BF16_PRODUCER_WARPGROUPS * HARDWARE_WARPGROUP_THREADS;
    // the registers (the worklog's kernel 5b): a block of THREADS gets LAUNCH_REGISTERS a thread (168 at 384 threads),
    // too few for 2 consumer warpgroups' 128 sums each; the producer needs few, so with 2 it gives back all but
    // WGMMA_MATMUL_BF16_PRODUCER_REGISTERS and the consumers take them: 40 x 128 + 232 x 256 = 168 x 384
    constexpr int THREADS = PRODUCER_THREADS + ROW_GROUPS * HARDWARE_WARPGROUP_THREADS;
    constexpr int LAUNCH_REGISTERS = HARDWARE_REGISTERS_PER_SM / THREADS / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    constexpr int PRODUCER_REGISTERS = WGMMA_MATMUL_BF16_PRODUCER_REGISTERS;
    static_assert(PRODUCER_REGISTERS >= HARDWARE_SETMAXNREG_MIN && PRODUCER_REGISTERS % HARDWARE_REGISTER_ALLOCATION_UNIT == 0,
                  "setmaxnreg takes at least HARDWARE_SETMAXNREG_MIN, in whole allocation units");
    constexpr int CONSUMER_REGISTERS_FIT = (LAUNCH_REGISTERS * THREADS - PRODUCER_REGISTERS * PRODUCER_THREADS) /
                                           (ROW_GROUPS * HARDWARE_WARPGROUP_THREADS) / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    constexpr int CONSUMER_REGISTERS = CONSUMER_REGISTERS_FIT < HARDWARE_SETMAXNREG_MAX ? CONSUMER_REGISTERS_FIT : HARDWARE_SETMAXNREG_MAX;
    constexpr bool MOVE_REGISTERS = ROW_GROUPS > 1;           // 1 consumer warpgroup: 255 a thread already, plenty
    // dynamic shared memory, which starts wherever the static ends, from its next HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM
    // atom on: the sets of tiles in TMA's swizzled layout, then each consumer warpgroup's room for the store. Static:
    // each set's two barriers, and one a consumer warpgroup for its store
    extern __shared__ __align__(16) uint8_t wgmma_matmul_bf16_shared_list[];
    __shared__ uint64_t full_barriers_list[STAGES];          // set s's bytes are in
    __shared__ uint64_t empty_barriers_list[STAGES];         // every consumer of the cluster is done with set s
    __shared__ uint64_t store_barriers_list[ROW_GROUPS];     // consumer warpgroup g's threads are all at the same point of its store
    __shared__ uint64_t side_barriers_list[ROW_GROUPS];      // a read side: consumer warpgroup g's 64 rows of it are in its room
    uint32_t shared_address_val = (uint32_t)__cvta_generic_to_shared(wgmma_matmul_bf16_shared_list);
    nv_bfloat16* tiles_ptr = reinterpret_cast<nv_bfloat16*>(
        wgmma_matmul_bf16_shared_list + (HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM - shared_address_val % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM);
    float* staged_rooms_ptr = reinterpret_cast<float*>(tiles_ptr + STAGES * SET_VALUES);
    // where a WGMMA's 16 values of k start in a tile: 16 values along a row, or turned 16 rows on (16 x 64 values)
    const int left_k_step_values_val = LEFT_TURNED ? HARDWARE_WGMMA_K_BF16 * HARDWARE_WGMMA_M : HARDWARE_WGMMA_K_BF16;
    const int right_k_step_values_val = RIGHT_TURNED ? HARDWARE_WGMMA_K_BF16 * WGMMA_MATMUL_BF16_ROW_VALUES : HARDWARE_WGMMA_K_BF16;
    // the right descriptor's leading offset, in cells: a turned right tile of more than 64 columns is boxes side by
    // side, one box apart; otherwise not used (the 128-byte swizzle's own step), the usual 1
    const uint32_t right_leading_cells_val = RIGHT_TURNED && COLUMNS > WGMMA_MATMUL_BF16_ROW_VALUES
                                                 ? (uint32_t)(RIGHT_BOX_VALUES * sizeof(nv_bfloat16) / HARDWARE_WGMMA_DESCRIPTOR_UNIT)
                                                 : (uint32_t)(HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);

    // 1. the barriers, by one thread; with clusters, every block of the cluster waits until all have theirs
    if (threadIdx.x == 0) {
#pragma unroll
        for (int set_index_val = 0; set_index_val < STAGES; set_index_val++) {
            wgmma_matmul_bf16_barrier_init(&full_barriers_list[set_index_val], 1);                       // the producer's one arrival, and the bytes
            wgmma_matmul_bf16_barrier_init(&empty_barriers_list[set_index_val], ROW_GROUPS * CLUSTER);   // one arrival a consumer warpgroup of the cluster
        }
#pragma unroll
        for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++)
            wgmma_matmul_bf16_barrier_init(&store_barriers_list[row_group_index_val], HARDWARE_WARPGROUP_THREADS);   // every thread of the warpgroup
#pragma unroll
        for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++)
            wgmma_matmul_bf16_barrier_init(&side_barriers_list[row_group_index_val], 1);   // the warpgroup's first thread, and the bytes
        cde::fence_proxy_async_shared_cta();                          // the barriers visible to TMA
    }
    __syncthreads();
    if (CLUSTER > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }

    // 2. the units of work, and which are this block's. A unit: the cluster's UNIT_ROWS x COLUMNS of output, for one
    //    split; block b of the cluster takes its square b. The units of a split are numbered in groups of
    //    block_rows_per_group rows of units, down a group first (the last group has the rows left). Cluster c takes
    //    units c, c + clusters, ...
    int cluster_block_index_val = blockIdx.x % CLUSTER;
    int cluster_index_val = blockIdx.x / CLUSTER;
    int total_clusters_val = gridDim.x / CLUSTER;
    int total_unit_columns_val = CEIL_DIV(columns_val, COLUMNS);
    int total_unit_rows_val = CEIL_DIV(rows_val, UNIT_ROWS);
    int units_per_split_val = total_unit_columns_val * total_unit_rows_val;
    int total_units_val = units_per_split_val * splits_val;
    int all_steps_val = CEIL_DIV(sum_length_val, WGMMA_MATMUL_BF16_BLOCK_K);
    int units_per_group_val = block_rows_per_group_val * total_unit_columns_val;
    // gives a unit's place for this block: its square's first row and column, its split, and the split's steps
    auto place_unit_val = [&](int unit_index_val, int& block_first_row_index_val, int& block_first_column_index_val, int& split_index_val,
                              int& first_step_index_val, int& total_steps_val) {
        split_index_val = unit_index_val / units_per_split_val;
        int unit_in_split_val = unit_index_val % units_per_split_val;
        int group_first_unit_row_index_val = (unit_in_split_val / units_per_group_val) * block_rows_per_group_val;
        int group_unit_rows_val = min(block_rows_per_group_val, total_unit_rows_val - group_first_unit_row_index_val);
        int unit_in_group_val = unit_in_split_val % units_per_group_val;
        int unit_row_index_val = group_first_unit_row_index_val + unit_in_group_val % group_unit_rows_val;
        int unit_column_index_val = unit_in_group_val / group_unit_rows_val;
        block_first_row_index_val = unit_row_index_val * UNIT_ROWS + cluster_block_index_val * BLOCK_ROWS;
        block_first_column_index_val = unit_column_index_val * COLUMNS;
        first_step_index_val = (int)((long long)split_index_val * all_steps_val / splits_val);
        total_steps_val = (int)((long long)(split_index_val + 1) * all_steps_val / splits_val) - first_step_index_val;
    };

    if (threadIdx.x < PRODUCER_THREADS) {
        if (MOVE_REGISTERS)                                   // the whole warpgroup: give back all but PRODUCER_REGISTERS
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PRODUCER_REGISTERS));
        // 3. the producer: its thread 0 asks for every step's tiles of every unit, set after set (walk_step counts
        //    the steps since the launch: its set is walk_step % STAGES, the set's lap walk_step / STAGES). From the
        //    second lap on, it first waits until the cluster's consumers are done with the set (its empty barrier's
        //    phase lap - 1). A step's ask: tell the set's full barrier how many bytes are coming, then the left tiles
        //    and the right tile's boxes: with clusters, block 0 asks for the right tile for every block. A box's
        //    corner is (column, row) of the matrix as it sits: (k, line), or turned (line, k); past the edges TMA
        //    gives 0s
        if (threadIdx.x == 0) {
            int walk_step_index_val = 0;
            for (int unit_index_val = cluster_index_val; unit_index_val < total_units_val; unit_index_val += total_clusters_val) {
                int block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val, total_steps_val;
                place_unit_val(unit_index_val, block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val,
                               total_steps_val);
                for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++, walk_step_index_val++) {
                    int set_index_val = walk_step_index_val % STAGES;
                    int lap_index_val = walk_step_index_val / STAGES;
                    if (lap_index_val > 0)
                        wgmma_matmul_bf16_barrier_wait(&empty_barriers_list[set_index_val], (uint32_t)((lap_index_val - 1) % 2));
                    nv_bfloat16* left_tile_ptr = tiles_ptr + set_index_val * SET_VALUES;
                    nv_bfloat16* right_tile_ptr = left_tile_ptr + LEFT_VALUES;
                    int step_first_k_index_val = (first_step_index_val + step_index_val) * WGMMA_MATMUL_BF16_BLOCK_K;
                    wgmma_matmul_bf16_barrier_expect(&full_barriers_list[set_index_val], SET_BYTES);
    #pragma unroll
                    for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++) {
                        int group_first_row_index_val = block_first_row_index_val + row_group_index_val * HARDWARE_WGMMA_M;
                        wgmma_matmul_bf16_copy_tile(left_tile_ptr + row_group_index_val * WGMMA_MATMUL_BF16_LEFT_TILE_VALUES, &left_map_val,
                                                    LEFT_TURNED ? group_first_row_index_val : step_first_k_index_val,
                                                    LEFT_TURNED ? step_first_k_index_val : group_first_row_index_val,
                                                    &full_barriers_list[set_index_val]);
                    }
    #pragma unroll
                    for (int box_index_val = 0; box_index_val < RIGHT_BOXES; box_index_val++) {
                        int box_column_val = RIGHT_TURNED ? block_first_column_index_val + box_index_val * WGMMA_MATMUL_BF16_ROW_VALUES
                                                          : step_first_k_index_val;
                        int box_row_val = RIGHT_TURNED ? step_first_k_index_val : block_first_column_index_val;
                        if (CLUSTER == 1)
                            wgmma_matmul_bf16_copy_tile(right_tile_ptr + box_index_val * RIGHT_BOX_VALUES, &right_map_val, box_column_val,
                                                        box_row_val, &full_barriers_list[set_index_val]);
                        else if (cluster_block_index_val == 0)
                            wgmma_matmul_bf16_copy_tile_to_cluster(right_tile_ptr + box_index_val * RIGHT_BOX_VALUES, &right_map_val,
                                                                   box_column_val, box_row_val, &full_barriers_list[set_index_val],
                                                                   (uint16_t)((1u << CLUSTER) - 1));
                    }
                }
            }
        }
    } else {
        if (MOVE_REGISTERS)                                   // the whole warpgroup: take what the producer gave back
            asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CONSUMER_REGISTERS));
        // 4. a consumer: walk each unit's sum on its own left tile and the shared right tile, then store its rows
        int consumer_thread_index_val = (int)threadIdx.x - PRODUCER_THREADS;       // 0 on: warpgroup g's threads 128 g on
        int row_group_index_val = consumer_thread_index_val / HARDWARE_WARPGROUP_THREADS;
        int warpgroup_thread_index_val = consumer_thread_index_val % HARDWARE_WARPGROUP_THREADS;
        bool warpgroup_first_thread_val = warpgroup_thread_index_val == 0;
        int warp_index_val = warpgroup_thread_index_val / HARDWARE_THREADS_PER_WARP;   // in the warpgroup: its rows 16 w on
        int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
        int thread_first_row_in_group_val = warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
        int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
        uint8_t* room_ptr = reinterpret_cast<uint8_t*>(staged_rooms_ptr) + row_group_index_val * wgmma_matmul_bf16_room_bytes<COLUMNS>();
    float* staged_ptr = reinterpret_cast<float*>(room_ptr);
        uint32_t store_phase_val = 0;                                              // the store barrier's phase to wait for next
        uint32_t side_phase_val = 0;                                               // ... and the side barrier's
        constexpr int SIDE = WgmmaMatmulBf16SideOf<EPILOGUE>::value;               // how the epilogue's product is stored
        constexpr int BOX_COLUMNS = WGMMA_MATMUL_BF16_ROW_VALUES;                  // the TMA store's box: 64 columns, one swizzled row of 128 bytes ...
        constexpr int BOX_BYTES = HARDWARE_WGMMA_M * BOX_COLUMNS * sizeof(nv_bfloat16);   // ... by 64 rows: 8 KB
        // this warpgroup is done with a step's set: its first thread arrives at the set's empty barrier in every block
        // of the cluster
        auto free_set_val = [&](int set_index_val) {
            if (!warpgroup_first_thread_val) return;
            if (CLUSTER == 1) {
                wgmma_matmul_bf16_barrier_arrive(&empty_barriers_list[set_index_val]);
            } else {
    #pragma unroll
                for (int cluster_block_val = 0; cluster_block_val < CLUSTER; cluster_block_val++)
                    wgmma_matmul_bf16_barrier_arrive_in_cluster(&empty_barriers_list[set_index_val], (uint32_t)cluster_block_val);
            }
        };
        // every thread of this warpgroup at the same point: all arrive at its store barrier and wait for the phase
        auto meet_warpgroup_val = [&]() {
            wgmma_matmul_bf16_barrier_arrive(&store_barriers_list[row_group_index_val]);
            wgmma_matmul_bf16_barrier_wait(&store_barriers_list[row_group_index_val], store_phase_val);
            store_phase_val ^= 1;
        };

        int walk_step_index_val = 0;                                               // steps since the launch, as the producer counts them
        for (int unit_index_val = cluster_index_val; unit_index_val < total_units_val; unit_index_val += total_clusters_val) {
            int block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val, total_steps_val;
            place_unit_val(unit_index_val, block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val,
                           total_steps_val);
            int group_first_row_index_val = block_first_row_index_val + row_group_index_val * HARDWARE_WGMMA_M;
            bool tma_store_val = SIDE != WGMMA_MATMUL_BF16_SIDE_STAGED && !add_onto_val && split_sums_ptr == nullptr;
            // a read side: before the walk, the warpgroup's first thread asks TMA for its 64 rows of it, into its room
            // (once TMA has read the room for the unit before), so they come while the tensor cores work
            if constexpr (SIDE == WGMMA_MATMUL_BF16_SIDE_READ)
                if (tma_store_val && warpgroup_first_thread_val) {
                    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
                    wgmma_matmul_bf16_barrier_expect(&side_barriers_list[row_group_index_val], (uint32_t)(COLUMNS / BOX_COLUMNS * BOX_BYTES));
#pragma unroll
                    for (int box_index_val = 0; box_index_val < COLUMNS / BOX_COLUMNS; box_index_val++)
                        wgmma_matmul_bf16_copy_tile(reinterpret_cast<nv_bfloat16*>(room_ptr + box_index_val * BOX_BYTES), &side_map_val,
                                                    block_first_column_index_val + box_index_val * BOX_COLUMNS, group_first_row_index_val,
                                                    &side_barriers_list[row_group_index_val]);
                }
            float sums_list[COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of its warpgroup's 64 rows
            for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
                // a. wait until this step's set is full: its barrier's phase is the set's lap
                int set_index_val = (walk_step_index_val + step_index_val) % STAGES;
                wgmma_matmul_bf16_barrier_wait(&full_barriers_list[set_index_val], (uint32_t)(((walk_step_index_val + step_index_val) / STAGES) % 2));

                // b. the tensor cores: WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs, each 16 more values of k
                const nv_bfloat16* left_tile_ptr = tiles_ptr + set_index_val * SET_VALUES + row_group_index_val * WGMMA_MATMUL_BF16_LEFT_TILE_VALUES;
                const nv_bfloat16* right_tile_ptr = tiles_ptr + set_index_val * SET_VALUES + LEFT_VALUES;
                asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // the sums ready for the tensor cores
    #pragma unroll
                for (int k_step_index_val = 0; k_step_index_val < WGMMA_MATMUL_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16; k_step_index_val++)
                    wgmma_matmul_bf16_multiply_add<LEFT_TURNED, RIGHT_TURNED>(sums_list, left_tile_ptr + k_step_index_val * left_k_step_values_val,
                                                                              right_tile_ptr + k_step_index_val * right_k_step_values_val,
                                                                              right_leading_cells_val);
                asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // the step's WGMMAs as one group

                // c. wait for the WGMMAs of IN_FLIGHT steps back (this step's own still running); this warpgroup is then
                //    done with that step's set
                asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(IN_FLIGHT) : "memory");
                if (step_index_val >= IN_FLIGHT)
                    free_set_val((walk_step_index_val + step_index_val - IN_FLIGHT) % STAGES);
            }
            // d. the last steps' WGMMAs done, and their sets free: the producer may be filling them for the next unit
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            for (int step_index_val = max(total_steps_val - IN_FLIGHT, 0); step_index_val < total_steps_val; step_index_val++)
                free_set_val((walk_step_index_val + step_index_val) % STAGES);
            walk_step_index_val += total_steps_val;

            // e. the epilogues stored by TMA (no add_onto, no splits): TMA stores the warpgroup's 64 rows. The room is
            //    the warpgroup's first thread's to wait for: until TMA has read it for the unit before, or with a read
            //    side, until the side's rows are in it (every thread waits). Then every thread writes its pairs (+
            //    bias), or with a read side with_side(pair, the side's pair it reads at the same place), rounded once, as
            //    bfloat16 at their place in the room's boxes (box b: columns 64 b on, rows of 128 bytes, the 16-byte
            //    cell c of row r at c XOR r % 8, as TMA's 128-byte swizzle has it), makes them visible to TMA, and the
            //    first thread asks TMA to store the boxes. A written side: then the same again, side_of(each pair) into
            //    the room once TMA has read output's, and TMA stores the side. Past the edges TMA reads 0s and writes
            //    nothing
            if (tma_store_val) {
                // the room's pair of each of this thread's pairs, and the pair (+ bias), handed to what_val, whose
                // result is written there
                auto write_room_val = [&](auto what_val) {
#pragma unroll
                    for (int group_index_val = 0; group_index_val < COLUMN_GROUPS; group_index_val++) {
                        int column_in_square_val = group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
                        int column_index_val = block_first_column_index_val + column_in_square_val;
                        float2 bias_val = bias_ptr != nullptr && column_index_val < columns_val
                                              ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                              : make_float2(0.0f, 0.0f);
                        int box_index_val = column_in_square_val / BOX_COLUMNS;
                        int column_in_box_val = column_in_square_val % BOX_COLUMNS;
                        int cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) / HARDWARE_TMA_BYTES_PER_CELL;
                        int byte_in_cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) % HARDWARE_TMA_BYTES_PER_CELL;
#pragma unroll
                        for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
                            int row_in_group_val = thread_first_row_in_group_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                            int swizzled_cell_val = cell_val ^ (row_in_group_val % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN);
                            nv_bfloat162* room_pair_ptr = reinterpret_cast<nv_bfloat162*>(
                                room_ptr + box_index_val * BOX_BYTES + row_in_group_val * HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW +
                                swizzled_cell_val * HARDWARE_TMA_BYTES_PER_CELL + byte_in_cell_val);
                            float2 pair_val = make_float2(sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] + bias_val.x,
                                                          sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] + bias_val.y);
                            float2 result_val = what_val(pair_val, room_pair_ptr);
                            *room_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
                        }
                    }
                };
                // the room's boxes stored by TMA into map_ptr's tensor, once every thread's writes are visible to it
                auto store_room_val = [&](const CUtensorMap* map_ptr) {
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // this thread's writes visible to TMA
                    meet_warpgroup_val();                                          // every thread's
                    if (warpgroup_first_thread_val) {
#pragma unroll
                        for (int box_index_val = 0; box_index_val < COLUMNS / BOX_COLUMNS; box_index_val++)
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                                         ::"l"(map_ptr), "r"(block_first_column_index_val + box_index_val * BOX_COLUMNS),
                                           "r"(group_first_row_index_val), "r"(wgmma_matmul_bf16_shared_address(room_ptr + box_index_val * BOX_BYTES))
                                         : "memory");
                        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
                    }
                };
                // the room free (TMA has read it): the first thread waits, then the warpgroup meets
                auto wait_room_read_val = [&]() {
                    if (warpgroup_first_thread_val)
                        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
                    meet_warpgroup_val();
                };
                if constexpr (SIDE == WGMMA_MATMUL_BF16_SIDE_READ) {
                    wgmma_matmul_bf16_barrier_wait(&side_barriers_list[row_group_index_val], side_phase_val);   // the side's rows are in
                    side_phase_val ^= 1;
                    write_room_val([&](float2 pair_val, const nv_bfloat162* room_pair_ptr) {
                        return epilogue_val.with_side(pair_val, __bfloat1622float2(*room_pair_ptr));
                    });
                } else {
                    wait_room_read_val();
                    write_room_val([](float2 pair_val, const nv_bfloat162*) { return pair_val; });
                }
                store_room_val(&output_map_val);
                if constexpr (SIDE == WGMMA_MATMUL_BF16_SIDE_WRITTEN) {
                    wait_room_read_val();
                    write_room_val([&](float2 pair_val, const nv_bfloat162*) { return epilogue_val.side_of(pair_val); });
                    store_room_val(&side_map_val);
                }
                continue;
            }

            // e. the others: the store, WGMMA_MATMUL_BF16_STAGED_ROWS of this warpgroup's rows a round
            if (warpgroup_first_thread_val)                                    // TMA done reading the room, if it stored the unit before
                asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
            meet_warpgroup_val();
            for (int round_index_val = 0; round_index_val < STORE_ROUNDS; round_index_val++) {
                // the round's sums (+ bias) into the warpgroup's room as floats, each thread's pairs at their row and
                // column: the threads whose rows are the round's
    #pragma unroll
                for (int group_index_val = 0; group_index_val < COLUMN_GROUPS; group_index_val++) {
                    int column_in_square_val = group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
                    int column_index_val = block_first_column_index_val + column_in_square_val;
                    float2 bias_val = bias_ptr != nullptr && column_index_val < columns_val
                                          ? __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val))
                                          : make_float2(0.0f, 0.0f);
    #pragma unroll
                    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
                        int row_in_group_val = thread_first_row_in_group_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                        if (row_in_group_val / WGMMA_MATMUL_BF16_STAGED_ROWS != round_index_val) continue;
                        *reinterpret_cast<float2*>(staged_ptr + (row_in_group_val % WGMMA_MATMUL_BF16_STAGED_ROWS) * STAGED_ROW_FLOATS +
                                                   column_in_square_val) =
                            make_float2(sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] + bias_val.x,
                                        sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] + bias_val.y);
                    }
                }
                meet_warpgroup_val();                                              // the round's rows are in the room

                // the rows out, row by row, WGMMA_MATMUL_BF16_STORE_VALUES neighbouring values a thread: epilogue(each
                // pair) (+ output), rounded once, one 8-byte store; nothing past the edges (columns is a multiple of 8,
                // so a thread's 4 values are all in or all out). Split: the 4 float sums into the split's slice of the
                // scratch as they are
                for (int store_index_val = warpgroup_thread_index_val; store_index_val < WGMMA_MATMUL_BF16_STAGED_ROWS * STORES_PER_ROW;
                     store_index_val += HARDWARE_WARPGROUP_THREADS) {
                    int row_in_round_val = store_index_val / STORES_PER_ROW;
                    int column_in_square_val = WGMMA_MATMUL_BF16_STORE_VALUES * (store_index_val % STORES_PER_ROW);
                    int row_index_val = group_first_row_index_val + round_index_val * WGMMA_MATMUL_BF16_STAGED_ROWS + row_in_round_val;
                    int column_index_val = block_first_column_index_val + column_in_square_val;
                    if (row_index_val >= rows_val || column_index_val >= columns_val) continue;
                    float4 values_val = *reinterpret_cast<const float4*>(staged_ptr + row_in_round_val * STAGED_ROW_FLOATS + column_in_square_val);
                    size_t output_index_val = (size_t)row_index_val * columns_val + column_index_val;
                    if (split_sums_ptr != nullptr) {
                        *reinterpret_cast<float4*>(split_sums_ptr + (size_t)split_index_val * rows_val * columns_val + output_index_val) = values_val;
                        continue;
                    }
                    float2 first_val = epilogue_val(make_float2(values_val.x, values_val.y), output_index_val);
                    float2 second_val = epilogue_val(make_float2(values_val.z, values_val.w), output_index_val + HARDWARE_WGMMA_COLUMNS_PER_THREAD);
                    nv_bfloat162* output_pairs_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + output_index_val);
                    if (add_onto_val) {
                        float2 first_old_val = __bfloat1622float2(output_pairs_ptr[0]);
                        float2 second_old_val = __bfloat1622float2(output_pairs_ptr[1]);
                        first_val = make_float2(first_val.x + first_old_val.x, first_val.y + first_old_val.y);
                        second_val = make_float2(second_val.x + second_old_val.x, second_val.y + second_old_val.y);
                    }
                    nv_bfloat162 pairs_list[2] = {__floats2bfloat162_rn(first_val.x, first_val.y), __floats2bfloat162_rn(second_val.x, second_val.y)};
                    *reinterpret_cast<uint2*>(output_pairs_ptr) = *reinterpret_cast<const uint2*>(pairs_list);
                }
                meet_warpgroup_val();                                              // every thread has read the room: the next round may write it
            }
        }
    }

    // 5. the consumers' TMA stores all written before the block leaves
    if (threadIdx.x >= PRODUCER_THREADS && (threadIdx.x - PRODUCER_THREADS) % HARDWARE_WARPGROUP_THREADS == 0)
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");

    // 6. with clusters: every thread of the cluster meets before any block leaves, so no block's shared memory
    //    goes while another block may still write into it (the right tile's multicast, the arrivals at its
    //    empty barriers). Not .aligned: the producer's threads come from a branch only its thread 0 took
    if (CLUSTER > 1) {
        asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
    }
}

/// Adds the splits' float sums of WGMMA_MATMUL_BF16_STORE_VALUES neighbouring values of output, in order of the
/// splits (+ bias), then the epilogue on each pair (+ output), rounded once, one 8-byte store: what a split product's
/// blocks leave undone. One thread per 4 values, the 4 a thread's neighbours and a warp's 32 neighbours too.
///
///   output_ptr                   {rows, columns}: written over, or added onto
///   split_sums_ptr               {splits, rows, columns} float: each split's sums
///   bias_ptr                     {columns}, or nullptr for none
///   rows_val, columns_val        output's shape (columns a multiple of WGMMA_MATMUL_BF16_STORE_VALUES)
///   splits_val                   the slices
///   add_onto_val, epilogue_val   as in wgmma_matmul_bf16
template <typename EPILOGUE>
__global__ void wgmma_matmul_bf16_splits_sum_kernel(nv_bfloat16* output_ptr, const float* split_sums_ptr,
                                                    const nv_bfloat16* bias_ptr, size_t rows_val, size_t columns_val,
                                                    int splits_val, bool add_onto_val, EPILOGUE epilogue_val) {
    size_t output_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * WGMMA_MATMUL_BF16_STORE_VALUES;
    if (output_index_val >= rows_val * columns_val) return;
    // 1. the splits' sums, in order
    size_t slice_values_val = rows_val * columns_val;
    float4 sum_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int split_index_val = 0; split_index_val < splits_val; split_index_val++) {
        float4 split_val = *reinterpret_cast<const float4*>(split_sums_ptr + split_index_val * slice_values_val + output_index_val);
        sum_val = make_float4(sum_val.x + split_val.x, sum_val.y + split_val.y, sum_val.z + split_val.z, sum_val.w + split_val.w);
    }
    // 2. + bias
    if (bias_ptr != nullptr) {
        size_t column_index_val = output_index_val % columns_val;
        float2 first_bias_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val));
        float2 second_bias_val = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(bias_ptr + column_index_val + HARDWARE_WGMMA_COLUMNS_PER_THREAD));
        sum_val = make_float4(sum_val.x + first_bias_val.x, sum_val.y + first_bias_val.y, sum_val.z + second_bias_val.x,
                              sum_val.w + second_bias_val.y);
    }
    // 3. the epilogue on each pair (+ output), rounded once, one 8-byte store
    float2 first_val = epilogue_val(make_float2(sum_val.x, sum_val.y), output_index_val);
    float2 second_val = epilogue_val(make_float2(sum_val.z, sum_val.w), output_index_val + HARDWARE_WGMMA_COLUMNS_PER_THREAD);
    nv_bfloat162* output_pairs_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + output_index_val);
    if (add_onto_val) {
        float2 first_old_val = __bfloat1622float2(output_pairs_ptr[0]);
        float2 second_old_val = __bfloat1622float2(output_pairs_ptr[1]);
        first_val = make_float2(first_val.x + first_old_val.x, first_val.y + first_old_val.y);
        second_val = make_float2(second_val.x + second_old_val.x, second_val.y + second_old_val.y);
    }
    nv_bfloat162 pairs_list[2] = {__floats2bfloat162_rn(first_val.x, first_val.y), __floats2bfloat162_rn(second_val.x, second_val.y)};
    *reinterpret_cast<uint2*>(output_pairs_ptr) = *reinterpret_cast<const uint2*>(pairs_list);
}

/// Gives the WGMMA descriptor of a tile in shared memory: where it starts and how its rows sit (rows of 128 bytes
/// in TMA's 128-byte swizzle, 8-row atoms HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM apart).
///     K-major (rows of k): the stride offset is from 8 lines to the next 8; moving 16 values of k along a row is
///                          32 bytes more on the start address (the swizzle is undone from the address); the leading
///                          offset is not used
///     MN-major (turned, rows of lines): the stride offset is from 8 values of k to the next 8; moving 16 values
///                          of k is 16 rows (2 atoms) more on the start address; the leading offset is from 64 lines
///                          to the next 64 (a tile of more than 64 lines is boxes side by side)
///
///   tile_ptr            the tile's first value to read, in shared memory
///   leading_cells_val   the leading offset, in 16-byte cells; not given: 1 (not used)
__device__ __forceinline__ uint64_t wgmma_matmul_bf16_descriptor(const nv_bfloat16* tile_ptr, uint32_t leading_cells_val) {
    uint64_t address_val = (uint64_t)__cvta_generic_to_shared(tile_ptr);
    uint64_t descriptor_val = ((address_val & HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS) / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT;
    descriptor_val |= (uint64_t)leading_cells_val << HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT;
    descriptor_val |= HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 << HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT;
    return descriptor_val;
}

/// sums {64, N} += left tile {64, 16} · right tile {N, 16}ᵀ on the tensor cores, for the whole warpgroup, N =
/// GROUPS x 8 (64, 128 or 256): wgmma.mma_async m64nNk16, bfloat16 in, float sums, read straight from shared memory;
/// asynchronous: done only after wgmma.wait_group. LEFT_TURNED, RIGHT_TURNED: the transpose bits (0: K-major, 1:
/// MN-major), which must be numbers in the instruction itself ("n": written in as constants). The instruction and
/// its 4 x GROUPS sum registers are written out for each N (generated, not typed).
///
///   sums_list                       this thread's sums, in WGMMA's layout: GROUPS groups of 4
///   left_tile_ptr, right_tile_ptr   the 16 values of k to use, in the shared memory tiles
///   right_leading_cells_val         the right descriptor's leading offset (wgmma_matmul_bf16_descriptor); not given: 1
template <bool LEFT_TURNED, bool RIGHT_TURNED, int GROUPS>
__device__ __forceinline__ void wgmma_matmul_bf16_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                               const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr,
                                                               uint32_t right_leading_cells_val) {
    static_assert(GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 64 || GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 128 ||
                      GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 256,
                  "N is 64, 128 or 256: the instructions written out below");
    uint64_t left_descriptor_val = wgmma_matmul_bf16_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = wgmma_matmul_bf16_descriptor(right_tile_ptr, right_leading_cells_val);
    // scale-d 1: add onto the sums; scale-a, scale-b 1; transpose-a, transpose-b: the turned flags
    if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 64) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
            "%32, %33, p, 1, 1, %35, %36;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
              "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
              "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
              "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(1), "n"(LEFT_TURNED ? 1 : 0), "n"(RIGHT_TURNED ? 1 : 0));
    } else if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 128) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
            "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
            "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
            "%64, %65, p, 1, 1, %67, %68;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
              "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
              "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
              "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3]),
              "+f"(sums_list[8][0]), "+f"(sums_list[8][1]), "+f"(sums_list[8][2]), "+f"(sums_list[8][3]), "+f"(sums_list[9][0]), "+f"(sums_list[9][1]), "+f"(sums_list[9][2]), "+f"(sums_list[9][3]),
              "+f"(sums_list[10][0]), "+f"(sums_list[10][1]), "+f"(sums_list[10][2]), "+f"(sums_list[10][3]), "+f"(sums_list[11][0]), "+f"(sums_list[11][1]), "+f"(sums_list[11][2]), "+f"(sums_list[11][3]),
              "+f"(sums_list[12][0]), "+f"(sums_list[12][1]), "+f"(sums_list[12][2]), "+f"(sums_list[12][3]), "+f"(sums_list[13][0]), "+f"(sums_list[13][1]), "+f"(sums_list[13][2]), "+f"(sums_list[13][3]),
              "+f"(sums_list[14][0]), "+f"(sums_list[14][1]), "+f"(sums_list[14][2]), "+f"(sums_list[14][3]), "+f"(sums_list[15][0]), "+f"(sums_list[15][1]), "+f"(sums_list[15][2]), "+f"(sums_list[15][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(1), "n"(LEFT_TURNED ? 1 : 0), "n"(RIGHT_TURNED ? 1 : 0));
    } else if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 256) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %130, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
            "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
            "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
            "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
            "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
            "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
            "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
            "%128, %129, p, 1, 1, %131, %132;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]),
              "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]),
              "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]),
              "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3]),
              "+f"(sums_list[8][0]), "+f"(sums_list[8][1]), "+f"(sums_list[8][2]), "+f"(sums_list[8][3]), "+f"(sums_list[9][0]), "+f"(sums_list[9][1]), "+f"(sums_list[9][2]), "+f"(sums_list[9][3]),
              "+f"(sums_list[10][0]), "+f"(sums_list[10][1]), "+f"(sums_list[10][2]), "+f"(sums_list[10][3]), "+f"(sums_list[11][0]), "+f"(sums_list[11][1]), "+f"(sums_list[11][2]), "+f"(sums_list[11][3]),
              "+f"(sums_list[12][0]), "+f"(sums_list[12][1]), "+f"(sums_list[12][2]), "+f"(sums_list[12][3]), "+f"(sums_list[13][0]), "+f"(sums_list[13][1]), "+f"(sums_list[13][2]), "+f"(sums_list[13][3]),
              "+f"(sums_list[14][0]), "+f"(sums_list[14][1]), "+f"(sums_list[14][2]), "+f"(sums_list[14][3]), "+f"(sums_list[15][0]), "+f"(sums_list[15][1]), "+f"(sums_list[15][2]), "+f"(sums_list[15][3]),
              "+f"(sums_list[16][0]), "+f"(sums_list[16][1]), "+f"(sums_list[16][2]), "+f"(sums_list[16][3]), "+f"(sums_list[17][0]), "+f"(sums_list[17][1]), "+f"(sums_list[17][2]), "+f"(sums_list[17][3]),
              "+f"(sums_list[18][0]), "+f"(sums_list[18][1]), "+f"(sums_list[18][2]), "+f"(sums_list[18][3]), "+f"(sums_list[19][0]), "+f"(sums_list[19][1]), "+f"(sums_list[19][2]), "+f"(sums_list[19][3]),
              "+f"(sums_list[20][0]), "+f"(sums_list[20][1]), "+f"(sums_list[20][2]), "+f"(sums_list[20][3]), "+f"(sums_list[21][0]), "+f"(sums_list[21][1]), "+f"(sums_list[21][2]), "+f"(sums_list[21][3]),
              "+f"(sums_list[22][0]), "+f"(sums_list[22][1]), "+f"(sums_list[22][2]), "+f"(sums_list[22][3]), "+f"(sums_list[23][0]), "+f"(sums_list[23][1]), "+f"(sums_list[23][2]), "+f"(sums_list[23][3]),
              "+f"(sums_list[24][0]), "+f"(sums_list[24][1]), "+f"(sums_list[24][2]), "+f"(sums_list[24][3]), "+f"(sums_list[25][0]), "+f"(sums_list[25][1]), "+f"(sums_list[25][2]), "+f"(sums_list[25][3]),
              "+f"(sums_list[26][0]), "+f"(sums_list[26][1]), "+f"(sums_list[26][2]), "+f"(sums_list[26][3]), "+f"(sums_list[27][0]), "+f"(sums_list[27][1]), "+f"(sums_list[27][2]), "+f"(sums_list[27][3]),
              "+f"(sums_list[28][0]), "+f"(sums_list[28][1]), "+f"(sums_list[28][2]), "+f"(sums_list[28][3]), "+f"(sums_list[29][0]), "+f"(sums_list[29][1]), "+f"(sums_list[29][2]), "+f"(sums_list[29][3]),
              "+f"(sums_list[30][0]), "+f"(sums_list[30][1]), "+f"(sums_list[30][2]), "+f"(sums_list[30][3]), "+f"(sums_list[31][0]), "+f"(sums_list[31][1]), "+f"(sums_list[31][2]), "+f"(sums_list[31][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(1), "n"(LEFT_TURNED ? 1 : 0), "n"(RIGHT_TURNED ? 1 : 0));
    }
}

/// Gives TMA's description of a matrix's tiles: a bfloat16 matrix {matrix_rows, matrix_columns} sitting row after
/// row, read in boxes of {box_rows, WGMMA_MATMUL_BF16_ROW_VALUES} written into shared memory in the 128-byte
/// swizzle; past the edges, 0s. Made by the driver's cuTensorMapEncodeTiled, found at run time, so the program
/// needs no -lcuda.
///
///   matrix_ptr                            its first value, on a 16-byte piece
///   matrix_rows_val, matrix_columns_val   its shape as it sits; matrix_columns a multiple of 8 (rows of whole 16-byte pieces)
///   box_rows_val                          rows of a tile: HARDWARE_WGMMA_M or WGMMA_MATMUL_BF16_BLOCK_COLUMNS (lines), or
///                                         turned WGMMA_MATMUL_BF16_BLOCK_K (values of k)
inline CUtensorMap wgmma_matmul_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                                unsigned box_rows_val) {
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function_ptr, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: wgmma_matmul_bf16 could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: columns, then rows
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {matrix_columns_val, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {matrix_columns_val * sizeof(nv_bfloat16)};            // from one row to the next
    cuuint32_t box_list[2] = {WGMMA_MATMUL_BF16_ROW_VALUES, box_rows_val};
    cuuint32_t element_steps_list[2] = {1, 1};                                            // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bfloat16*>(matrix_ptr),
                                     sizes_list, row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %zu} matrix at %p\n", (int)result_val,
                matrix_rows_val, matrix_columns_val, (const void*)matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Stops with a message unless the tensors fit, and gives the product's sizes: left's and right's sum_length the
/// same, output rows x columns with columns last, bias {columns}; all bfloat16; every last dim a multiple of 8 and
/// every tensor starting on a 16-byte piece.
///
///   output_tensor, left_tensor, right_tensor, bias_tensor, left_turned_val, right_turned_val   as in wgmma_matmul_bf16
///   rows_ptr, columns_ptr, sum_length_ptr                                                       written: the sizes
inline void wgmma_matmul_bf16_check_shapes(const Tensor& output_tensor, const Tensor& left_tensor, const Tensor& right_tensor,
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
        fprintf(stderr, "error: wgmma_matmul_bf16 needs left {rows, sum_length} (turned {sum_length, rows}), right {columns, "
                        "sum_length} (turned {sum_length, columns}), output {rows, columns}, bias {columns} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // 2. types: all bfloat16
    if (left_tensor.dtype() != DType::BFloat16 || right_tensor.dtype() != DType::BFloat16 ||
        output_tensor.dtype() != DType::BFloat16 || (has_bias_val && bias_tensor.dtype() != DType::BFloat16)) {
        fprintf(stderr, "error: wgmma_matmul_bf16 needs output, left, right and bias all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // 3. counts and addresses: rows of whole 16-byte pieces, each tensor starting on one
    if (left_last_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 || right_last_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 ||
        columns_val % WGMMA_MATMUL_BF16_VALUE_MULTIPLE != 0 ||
        (size_t)output_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)left_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 ||
        (size_t)right_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0 || (size_t)bias_tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0) {
        fprintf(stderr, "error: wgmma_matmul_bf16's TMA reads rows of whole %d-byte pieces, so every tensor's last dim must be a "
                        "multiple of %d and every tensor must start on %d bytes; got last dims %zu (left), %zu (right), %zu (output)\n",
                HARDWARE_TMA_BYTES_PER_CELL, WGMMA_MATMUL_BF16_VALUE_MULTIPLE, HARDWARE_TMA_BYTES_PER_CELL, left_last_val, right_last_val,
                columns_val);
        exit(EXIT_FAILURE);
    }
    *rows_ptr = rows_val;
    *columns_ptr = columns_val;
    *sum_length_ptr = sum_length_val;
}

#endif // WGMMA_MATMUL_BF16_CUH
