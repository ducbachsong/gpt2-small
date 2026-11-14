// attention_output_weight_grad_bf16.cuh — attention_out_weight_grad += branch_output_gradᵀ · attention_output: the projection's weight gradient,
// in bfloat16 on an H100's tensor cores (WGMMA, fed by TMA). GPT-2's product of its own: one kernel for this product
// alone, so its code and settings can change without touching the 14 others.
// Everything it uses is in this file, its hardware calls (TMA, barriers, WGMMA) included, save cuda_common.h's
// HARDWARE_ numbers and the Tensor class.
//
//     tensor        type in memory   shape                                  read / written
//     output_grad   bfloat16         {rows, out_features}                   TMA, turned: 64 x 64 boxes (left)
//     input         bfloat16         {rows, in_features}                    TMA, turned: 64 x 64 boxes (right)
//     weight_grad   bfloat16         {out_features, in_features}            added onto
//     sums          float            in registers                           the tensor cores' C
//     split sums    float            {splits, out_features, in_features}    scratch, only when split
//
// As a product: output {rows, columns} = left {rows, sum_length} · right {columns, sum_length}ᵀ (+ output), with
// output = weight_grad (rows = out_features, columns = in_features), the sum over the micro-batch's rows
// (sum_length = rows of output_grad), left = output_grad and right = input both sitting turned round: their tiles
// are {64 k, lines} in shared memory, and the WGMMA reads them turned.
//
// A block works out squares of (ROW_GROUPS x 64) x COLUMNS of weight_grad, one after another (persistent), with one
// producer warpgroup and ROW_GROUPS consumer warpgroups:
//     the producer's thread 0 asks TMA for each step's left tiles and right tile (64 values of the sum a step)
//         into one of STAGES sets of shared memory, once the consumers have freed it (the set's empty barrier)
//     each consumer warpgroup waits for the set (its full barrier), adds its 64 rows' tile products into its sums
//         with WGMMA, and frees the set a step later (one step's WGMMAs left running)
//     before a square's walk, each consumer warpgroup's first thread asks TMA for its 64 rows of weight_grad as
//         they are (the old values) into its room of shared memory, in TMA's swizzled layout: they come while the
//         tensor cores work
//     at the square's end, each thread adds its sums onto the old values in the room, rounded once to bfloat16,
//         and one thread asks TMA to store the room, while the producer already brings the next square's tiles.
//         (Stored row by row from a staged float copy, reading the old values then, the store held the tensor
//         cores up: cuBLAS adds onto its output for nothing.)
// SPLITS > 1: the sum (the micro-batch's rows) is split in SPLITS, each split a unit of its own, which puts its
// float sums into its room STAGED_ROWS rows a round and writes them row by row, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES neighbouring
// values a thread, into its slice of a scratch; then attention_output_weight_grad_bf16_splits_sum_kernel adds the slices in order of the splits and
// onto weight_grad: a product of few squares and a long sum keeps every SM busy. Fixed for a launch, so the bits
// never change between runs; they differ from 1 split's, whose sums are added in another order. The splits' code
// is in the split kernel only (SPLIT), so the unsplit one carries none of it.
// SPLITS 0: the count is chosen at each launch (attention_output_weight_grad_bf16_auto_splits) for this GPU: the blocks walk the pieces in
// rounds, and a last round half empty leaves SMs idle, so the best count is the one whose pieces fill the
// rounds: the least rounds x (a piece's steps + ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLIT_COST_STEPS, what a piece's float store and its share
// of the splits' sum cost). Measured on an H100 PCIe (114 SMs), mlp up's weight_grad at 2 x 256 x 3 (72 squares
// of 256 steps): 1 split 0.216 ms, 2 0.238, 3 0.183, 4 0.216, 5 0.240, 6 0.223; the rule gives 3. With 132 SMs
// another count wins, so a fixed count is right on one GPU only. The count changes the order sums are added in:
// the bits are the same from run to run on one GPU, not from one GPU model to another.
// Clusters of CLUSTER blocks take squares one under the other and share each right tile: block 0's producer asks
// TMA for it once, into every block's shared memory (multicast); every consumer warpgroup of the cluster frees a set
// in every block. Its settings (the defines below, from tests/bf16/prod/tune_attention_output_weight_grad_bf16.cu):
//     ROW_GROUPS            consumer warpgroups a block, 64 rows each: 1 or 2
//     COLUMNS               a square's columns, its WGMMA's N: 64, 128 or 256
//     STAGES                sets of tiles in shared memory: 1 to ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_STAGES
//     SPLITS                splits of the sum: 1 (none) or more; 0: chosen at each launch for this GPU
//     BLOCK_ROWS_PER_GROUP  the squares' order: rows of squares the blocks run down before the next column
//     CLUSTER               blocks a cluster: 1 or 2
//     BLOCKS                blocks a launch: 0, as many as the GPU runs at once; ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ONE_A_SQUARE, one a square
//     PRODUCER_REGISTERS    with 2 consumer warpgroups, the producer's registers a thread; the consumers take the rest
//     IN_FLIGHT             steps of WGMMAs left running while a consumer goes on (at most STAGES - 1)
//     STAGED_ROWS           rows a consumer warpgroup stages and stores a round: 16, 32 or 64
//     STAGED_ROW_PADDING    floats past COLUMNS in a staged row, a multiple of 4
// WGMMA and TMA exist only on Hopper: run it on an H100 (compute capability 9.0), built with -arch=sm_90a.
#ifndef ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_CUH
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda_bf16.h>
#include "../../cuda_common.h"       // cudaCheck, check_threads_per_block, the HARDWARE_ numbers
#include "../../../common/tensor.cuh"

// the setting: tune_attention_output_weight_grad_bf16 sweeps each and prints the fastest
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_GROUPS 2                       // consumer warpgroups a block, 64 rows each: 1 or 2
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_COLUMNS 256                        // a square's columns, its WGMMA's N: 64, 128 or 256
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGES 3                           // sets of tiles in shared memory: 1 to ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_STAGES
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLITS 0                           // splits of the sum: 1, none; 0: chosen at each launch (attention_output_weight_grad_bf16_auto_splits)
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP 1            // rows of squares a group of blocks runs down before the next column (1: row after row)
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_CLUSTER 1                          // blocks a cluster, sharing each right tile: 1 (none) or 2
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCKS 0                           // blocks a launch: 0, as many as the GPU runs at once (persistent); ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ONE_A_SQUARE: one a square
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK 256     // the splits' sum's block size
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLIT_COST_STEPS 15                  // SPLITS 0: what a split piece's float store and its share of the splits' sum cost, in steps of the walk (tune_attention_output_weight_grad_bf16 prints each count's time beside the rule's)
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_AUTO_SPLITS 16                   // SPLITS 0: the most it picks
// the walk's and the store's own numbers
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ONE_A_SQUARE (1 << 30)              // BLOCKS: more than any product's squares, so one block a square
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_ROW_GROUPS 2                     // the most: 2 x 128 threads, each warpgroup's sums up to 128 floats (COLUMNS 256), inside a block's registers
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_STAGES 4                         // the most sets of tiles: with the room, a square must fit HARDWARE_MAX_SHARED_BYTES_PER_BLOCK
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_CLUSTER 2                        // the most blocks a cluster
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_WARPGROUPS 1                // warpgroups of a block that only load tiles: the first one (its thread 0 asks TMA)
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS 40                // with 2 consumer warpgroups: the producer's registers a thread (setmaxnreg), the consumers take the rest; tune_attention_output_weight_grad_bf16 sweeps it
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_IN_FLIGHT 1                          // steps of WGMMAs left running while the threads go on (STAGES - 1 at most: the set must be free); tune_attention_output_weight_grad_bf16 sweeps it
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROWS 32                       // rows a consumer warpgroup stages and stores a round: 16, 32 or 64; tune_attention_output_weight_grad_bf16 sweeps it
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING 8                 // floats past COLUMNS in a staged row, a multiple of 4 (rows stay on 16 bytes): the rows a warp writes start in other banks; tune_attention_output_weight_grad_bf16 sweeps it
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES 4                       // neighbouring values a thread stores at once: two pairs, 8 bytes

// ── the hardware calls: this kernel's own, one or a few PTX instructions each ──────
//     barrier_init, _expect, _arrive, _wait  an mbarrier in shared memory: "this set is full" / "this set is free"
//     barrier_arrive_in_cluster              the same, at the barrier of another block of the cluster
//     copy_tile, copy_tile_to_cluster        TMA: one box of a matrix into shared memory (or every block's of the cluster)
//     store_box, store_commit, store_wait_*  TMA: one box of shared memory out to a matrix, and waiting for it
//     fence_for_tma                          this thread's writes to shared memory made visible to TMA
//     descriptor, multiply_add               WGMMA: where a tile sits, and sums {64, N} += left tile · right tileᵀ
//     wgmma_fence, wgmma_commit, wgmma_wait  WGMMA's groups: the sums ready, a step's group closed, waiting for it
//     cluster_sync                           every thread of the cluster at the same point
//     give_registers, take_registers         setmaxnreg: a warpgroup gives back or takes registers
//     tensor_map                             (host) TMA's description of a bfloat16 matrix in 64-value boxes
//     check_tensor                           (host) a tensor TMA can read: bfloat16, rows of whole 16-byte cells

// What the hardware's 128-byte swizzle fixes for every product: a tile's row is one swizzled row of 128 bytes.
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))       // values of k a step loads: one swizzled row
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))    // a TMA box's row: one swizzled row of bfloat16
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_LEFT_TILE_VALUES (HARDWARE_WGMMA_M * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K)                  // a left tile: 64 rows of a step's k
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_VALUE_MULTIPLE ((int)(HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16)))       // a matrix row's multiple: whole 16-byte cells
#define ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_BOX_BYTES (HARDWARE_WGMMA_M * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES * (int)sizeof(nv_bfloat16))   // a TMA store's box: 64 rows of one swizzled row
static_assert(ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K % HARDWARE_WGMMA_K_BF16 == 0, "each step is whole WGMMAs");
static_assert(HARDWARE_WGMMA_M == ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES, "a turned left tile's row (64 rows) is one swizzled row, as a tile's row of k is");

__device__ __forceinline__ uint64_t attention_output_weight_grad_bf16_descriptor(const nv_bfloat16* tile_ptr,
                                                                uint32_t leading_cells_val = HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);
template <bool LEFT_TURNED, bool RIGHT_TURNED, int GROUPS>
__device__ __forceinline__ void attention_output_weight_grad_bf16_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                              const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr,
                                                              uint32_t right_leading_cells_val = HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);

// ── barriers and TMA loads ────────────────────────────────────────────────────

/// Gives a shared-memory object's address as PTX's shared::cta instructions take it.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t attention_output_weight_grad_bf16_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void attention_output_weight_grad_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming: its phase completes when they
/// are in.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void attention_output_weight_grad_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Arrives once at a barrier of this block.
///
///   barrier_ptr   the barrier
__device__ __forceinline__ void attention_output_weight_grad_bf16_barrier_arrive(uint64_t* barrier_ptr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr)), "r"(1) : "memory");
}

/// Arrives once at the same barrier in another block of the cluster (or this one): the barrier at this address of
/// block cluster_block_val's shared memory. Released at the block's scope (.cta, as CUTLASS does), not the
/// cluster's: what it orders is this block's own reads of the set (the WGMMAs, already waited for); .release.cluster
/// made every arrival wait on a cluster-wide fence, and clusters of 2 took twice as long (H100 PCIe, 2 x 256: qkv
/// 0.250 ms against 0.139 ms with .cta, the head 5.46 against 2.55)
///
///   barrier_ptr         the barrier, as it sits in this block
///   cluster_block_val   which block of the cluster: 0 .. CLUSTER - 1
__device__ __forceinline__ void attention_output_weight_grad_bf16_barrier_arrive_in_cluster(uint64_t* barrier_ptr, uint32_t cluster_block_val) {
    asm volatile("{\n"
                 ".reg .b32 remote_address;\n"
                 "mapa.shared::cluster.u32 remote_address, %0, %1;\n"
                 "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remote_address], %2;\n"
                 "}\n"
                 ::"r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr)), "r"(cluster_block_val), "r"(1) : "memory");
}

/// Waits until a barrier's phase of this parity is complete: the caller keeps count of the phases (the first is
/// 0, the next 1, then 0 again), so no token is passed round.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void attention_output_weight_grad_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    uint32_t barrier_address_val = attention_output_weight_grad_bf16_shared_address(barrier_ptr);
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
__device__ __forceinline__ void attention_output_weight_grad_bf16_copy_tile(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                            int row_val, uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(attention_output_weight_grad_bf16_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr))
                 : "memory");
}

/// Asks TMA for one box of a matrix into the shared memory of several blocks of the cluster at once (multicast):
/// read from L2 once, written at the same place in each block, its bytes counted at each block's barrier there.
///
///   tile_ptr, map_ptr, column_val, row_val, barrier_ptr   as in attention_output_weight_grad_bf16_copy_tile, as they sit in this block
///   cluster_blocks_mask_val                               bit b set: block b of the cluster gets the box
__device__ __forceinline__ void attention_output_weight_grad_bf16_copy_tile_to_cluster(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                                       int row_val, uint64_t* barrier_ptr,
                                                                       uint16_t cluster_blocks_mask_val) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                 "[%0], [%1, {%2, %3}], [%4], %5;\n"
                 ::"r"(attention_output_weight_grad_bf16_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(attention_output_weight_grad_bf16_shared_address(barrier_ptr)), "h"(cluster_blocks_mask_val)
                 : "memory");
}


// ── TMA stores ────────────────────────────────────────────────────────────────

/// Makes this thread's writes to shared memory visible to TMA (the async proxy), before a TMA store reads them.
__device__ __forceinline__ void attention_output_weight_grad_bf16_fence_for_tma(void) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/// Asks TMA to store one box of shared memory into a matrix, in this thread's bulk group.
///
///   map_ptr                TMA's description of the matrix
///   column_val, row_val    the box's corner in the matrix
///   box_ptr                the box in shared memory, in TMA's swizzled layout
__device__ __forceinline__ void attention_output_weight_grad_bf16_store_box(const CUtensorMap* map_ptr, int column_val, int row_val, const void* box_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                 ::"l"(map_ptr), "r"(column_val), "r"(row_val), "r"(attention_output_weight_grad_bf16_shared_address(box_ptr)) : "memory");
}

/// Closes this thread's bulk group: the stores asked for since the last one, waited for together.
__device__ __forceinline__ void attention_output_weight_grad_bf16_store_commit(void) {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/// Waits until TMA has read every box this thread asked it to store (the shared memory may be written again).
__device__ __forceinline__ void attention_output_weight_grad_bf16_store_wait_read(void) {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

/// Waits until every box this thread asked TMA to store is written to memory (before the block leaves).
__device__ __forceinline__ void attention_output_weight_grad_bf16_store_wait_all(void) {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// ── WGMMA's groups, the cluster, registers ────────────────────────────────────

/// Makes the sums in registers ready for the tensor cores (before a step's WGMMAs), for the whole warpgroup.
__device__ __forceinline__ void attention_output_weight_grad_bf16_wgmma_fence(void) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/// Closes the WGMMAs asked for since the last group into one group, for the whole warpgroup.
__device__ __forceinline__ void attention_output_weight_grad_bf16_wgmma_commit(void) {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/// Waits until at most LEFT_RUNNING groups of this warpgroup's WGMMAs are still running.
template <int LEFT_RUNNING>
__device__ __forceinline__ void attention_output_weight_grad_bf16_wgmma_wait(void) {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(LEFT_RUNNING) : "memory");
}

/// Every thread of the cluster at this point before any goes on. ALIGNED: every thread of each warp comes here
/// together (.aligned, faster); otherwise threads may come from different branches.
template <bool ALIGNED>
__device__ __forceinline__ void attention_output_weight_grad_bf16_cluster_sync(void) {
    if constexpr (ALIGNED) {
        asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    } else {
        asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
    }
}

/// This warpgroup keeps only REGISTERS registers a thread and gives the rest back to the block (setmaxnreg).
template <int REGISTERS>
__device__ __forceinline__ void attention_output_weight_grad_bf16_give_registers(void) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
}

/// This warpgroup takes REGISTERS registers a thread from what the block has free (setmaxnreg).
template <int REGISTERS>
__device__ __forceinline__ void attention_output_weight_grad_bf16_take_registers(void) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
}

// ── WGMMA ─────────────────────────────────────────────────────────────────────

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
__device__ __forceinline__ uint64_t attention_output_weight_grad_bf16_descriptor(const nv_bfloat16* tile_ptr, uint32_t leading_cells_val) {
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
///   right_leading_cells_val         the right descriptor's leading offset (attention_output_weight_grad_bf16_descriptor); not given: 1
template <bool LEFT_TURNED, bool RIGHT_TURNED, int GROUPS>
__device__ __forceinline__ void attention_output_weight_grad_bf16_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                               const nv_bfloat16* left_tile_ptr, const nv_bfloat16* right_tile_ptr,
                                                               uint32_t right_leading_cells_val) {
    static_assert(GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 64 || GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 128 ||
                      GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 256,
                  "N is 64, 128 or 256: the instructions written out below");
    uint64_t left_descriptor_val = attention_output_weight_grad_bf16_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = attention_output_weight_grad_bf16_descriptor(right_tile_ptr, right_leading_cells_val);
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

// ── host ──────────────────────────────────────────────────────────────────────

/// Gives TMA's description of a matrix's tiles: a bfloat16 matrix {matrix_rows, matrix_columns} sitting row after
/// row, read in boxes of {box_rows, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES} written into shared memory in the 128-byte
/// swizzle; past the edges, 0s. Made by the driver's cuTensorMapEncodeTiled, found at run time, so the program
/// needs no -lcuda.
///
///   matrix_ptr                            its first value, on a 16-byte piece
///   matrix_rows_val, matrix_columns_val   its shape as it sits; matrix_columns a multiple of 8 (rows of whole 16-byte pieces)
///   box_rows_val                          rows of a tile: HARDWARE_WGMMA_M or a product's COLUMNS (lines), or
///                                         turned ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K (values of k)
inline CUtensorMap attention_output_weight_grad_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
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
    cuuint32_t box_list[2] = {ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES, box_rows_val};
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

/// Stops with a message unless TMA can read or write a tensor: bfloat16, its last dim a multiple of
/// ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_VALUE_MULTIPLE (rows of whole 16-byte cells), starting on a 16-byte cell.
///
///   caller_ptr, name_ptr   the caller's and the tensor's names, for the message
///   tensor                 the tensor
inline void attention_output_weight_grad_bf16_check_tensor(const char* caller_ptr, const char* name_ptr, const Tensor& tensor) {
    size_t last_val = tensor.dim() >= 1 ? tensor.size(tensor.dim() - 1) : 0;
    if (tensor.dtype() != DType::BFloat16 || last_val == 0 || last_val % ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_VALUE_MULTIPLE != 0 ||
        (size_t)tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0) {
        fprintf(stderr, "error: %s needs %s bfloat16, its last dim a multiple of %d, starting on %d bytes\n", caller_ptr, name_ptr,
                ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_VALUE_MULTIPLE, HARDWARE_TMA_BYTES_PER_CELL);
        exit(EXIT_FAILURE);
    }
}

// ── the product ───────────────────────────────────────────────────────────────
void attention_output_weight_grad_bf16(const Tensor& weight_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor);
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS,
          int IN_FLIGHT = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_IN_FLIGHT, int STAGED_ROWS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROWS, int STAGED_ROW_PADDING = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>
void attention_output_weight_grad_bf16_at(const Tensor& weight_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor, int splits_val,
            int block_rows_per_group_val, int blocks_val, int sum_threads_per_block_val = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK);

template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS,
          int IN_FLIGHT = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_IN_FLIGHT, int STAGED_ROWS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROWS, int STAGED_ROW_PADDING = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>
int attention_output_weight_grad_bf16_auto_splits(size_t rows_val, size_t columns_val, size_t sum_length_val);

// ── for it ────────────────────────────────────────────────────────────────────
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS,
          int STAGED_ROW_PADDING>
int attention_output_weight_grad_bf16_blocks_at_once();
template <int COLUMNS, int STAGED_ROWS, int STAGED_ROW_PADDING>
__host__ __device__ constexpr size_t attention_output_weight_grad_bf16_room_bytes();
template <int COLUMNS, int STAGES, int ROW_GROUPS, int STAGED_ROWS, int STAGED_ROW_PADDING>
constexpr size_t attention_output_weight_grad_bf16_shared_bytes();
template <int STAGES, int ROW_GROUPS>
constexpr size_t attention_output_weight_grad_bf16_barrier_bytes();
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS,
          int STAGED_ROW_PADDING>
void attention_output_weight_grad_bf16_launch(const CUtensorMap& left_map_val, const CUtensorMap& right_map_val, const CUtensorMap& output_map_val,
                nv_bfloat16* output_ptr, float* split_sums_ptr,
                size_t rows_val, size_t columns_val, size_t sum_length_val, int splits_val, int block_rows_per_group_val, int blocks_val);
template <int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS,
          int IN_FLIGHT = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_IN_FLIGHT, int STAGED_ROWS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROWS, int STAGED_ROW_PADDING = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>
__global__ void attention_output_weight_grad_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                           const __grid_constant__ CUtensorMap output_map_val,
                           nv_bfloat16* output_ptr, float* split_sums_ptr, int rows_val, int columns_val, int sum_length_val,
                           int splits_val, int block_rows_per_group_val);
__global__ void attention_output_weight_grad_bf16_splits_sum_kernel(nv_bfloat16* output_ptr, const float* split_sums_ptr, size_t rows_val, size_t columns_val,
                                      int splits_val);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

/// Adds weight_grad += output_gradᵀ · input at this file's setting: attention_output_weight_grad_bf16_at with the defines above.
///
///   weight_grad_tensor   {out_features, in_features} bfloat16: added onto
///   output_grad_tensor   {..., out_features} bfloat16
///   input_tensor         {..., in_features} bfloat16, output_grad's rows
inline void attention_output_weight_grad_bf16(const Tensor& weight_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor) {
    attention_output_weight_grad_bf16_at<ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_GROUPS, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_COLUMNS, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STAGES, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_CLUSTER>(weight_grad_tensor, output_grad_tensor, input_tensor, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLITS,
                                                                 ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCKS);
}

/// Adds the product onto weight_grad at a setting given: checks the tensors, describes the tiles for TMA, then
/// launches attention_output_weight_grad_bf16_kernel made for ROW_GROUPS, COLUMNS, STAGES, CLUSTER and split or not; split, the splits' sums
/// added onto weight_grad by attention_output_weight_grad_bf16_splits_sum_kernel.
///
///   weight_grad_tensor, output_grad_tensor, input_tensor   as in attention_output_weight_grad_bf16
///   splits_val                  blocks a square's sum is split over (more than the steps: one a step); 0: chosen
///                               for this GPU (attention_output_weight_grad_bf16_auto_splits)
///   block_rows_per_group_val    the squares' order: rows of squares a group runs down (at least 1)
///   blocks_val                  blocks launched: 0, as many as the GPU runs at once; more than the squares: one a square
///   sum_threads_per_block_val   the splits' sum's block size; not given: ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK
/// ROW_GROUPS, COLUMNS, STAGES, CLUSTER: the square's warpgroups and columns, the sets of tiles, the blocks of a cluster.
/// PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING: as at the top of the file; not given: their defines.
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS, int STAGED_ROW_PADDING>
inline void attention_output_weight_grad_bf16_at(const Tensor& weight_grad_tensor, const Tensor& output_grad_tensor, const Tensor& input_tensor, int splits_val,
                   int block_rows_per_group_val, int blocks_val, int sum_threads_per_block_val) {
    static_assert(COLUMNS == 64 || COLUMNS == 128 || COLUMNS == 256, "COLUMNS is one WGMMA's N this file has: 64, 128 or 256");
    static_assert(STAGES >= 1 && STAGES <= ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_STAGES, "STAGES is 1 to ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_STAGES");
    static_assert(ROW_GROUPS >= 1 && ROW_GROUPS <= ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_ROW_GROUPS, "ROW_GROUPS is 1 to ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_ROW_GROUPS");
    static_assert(CLUSTER >= 1 && CLUSTER <= ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_CLUSTER, "CLUSTER is 1 to ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_CLUSTER");
    // 1. check the tensors: each one TMA can read, output_grad {..., out}, input {..., in} of its rows, weight_grad
    //    {out, in}
    attention_output_weight_grad_bf16_check_tensor("attention_output_weight_grad_bf16", "weight_grad", weight_grad_tensor);
    attention_output_weight_grad_bf16_check_tensor("attention_output_weight_grad_bf16", "output_grad", output_grad_tensor);
    attention_output_weight_grad_bf16_check_tensor("attention_output_weight_grad_bf16", "input", input_tensor);
    size_t out_features_val = output_grad_tensor.size(output_grad_tensor.dim() - 1);
    size_t in_features_val = input_tensor.size(input_tensor.dim() - 1);
    size_t sum_length_val = output_grad_tensor.numel() / out_features_val;     // the micro-batch's rows
    if (input_tensor.numel() != sum_length_val * in_features_val || weight_grad_tensor.shape() != Shape({out_features_val, in_features_val})) {
        fprintf(stderr, "error: attention_output_weight_grad_bf16 needs output_grad {..., out}, input {..., in} of output_grad's rows, weight_grad {out, in}\n");
        exit(EXIT_FAILURE);
    }
    if (block_rows_per_group_val < 1 || splits_val < 0 || blocks_val < 0) {
        fprintf(stderr, "error: attention_output_weight_grad_bf16's block_rows_per_group must be 1 or more, splits and blocks 0 or more; got %d, %d and %d\n",
                block_rows_per_group_val, splits_val, blocks_val);
        exit(EXIT_FAILURE);
    }
    size_t rows_val = out_features_val, columns_val = in_features_val;
    if (splits_val == 0)
        splits_val = attention_output_weight_grad_bf16_auto_splits<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING>(rows_val, columns_val, sum_length_val);
    int total_steps_val = (int)CEIL_DIV(sum_length_val, (size_t)ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K);
    if (splits_val > total_steps_val) splits_val = total_steps_val > 0 ? total_steps_val : 1;   // at most one a step

    // 2. TMA's descriptions: left = output_grad sitting turned {sum_length, rows}, right = input sitting turned
    //    {sum_length, columns}, both in boxes of {64 k, 64 lines}; weight_grad {rows, columns} in boxes of {64 rows, 64
    //    columns}, for its old values in and the new ones out
    CUtensorMap left_map_val = attention_output_weight_grad_bf16_tensor_map(output_grad_tensor.data_ptr<nv_bfloat16>(), sum_length_val, rows_val,
                                                           ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K);
    CUtensorMap right_map_val = attention_output_weight_grad_bf16_tensor_map(input_tensor.data_ptr<nv_bfloat16>(), sum_length_val, columns_val,
                                                            ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K);

    CUtensorMap output_map_val = attention_output_weight_grad_bf16_tensor_map(weight_grad_tensor.data_ptr<nv_bfloat16>(), rows_val, columns_val, HARDWARE_WGMMA_M);

    // 3. split: the scratch, a {rows, columns} slice of float sums per split, made the first time and grown when the
    //    product needs more; then the kernel, split or not
    nv_bfloat16* output_ptr = weight_grad_tensor.data_ptr<nv_bfloat16>();
    if (splits_val > 1) {
        static Tensor split_sums_tensor;
        if ((size_t)split_sums_tensor.numel() < (size_t)splits_val * rows_val * columns_val)
            split_sums_tensor = Tensor::empty({(size_t)splits_val * rows_val * columns_val});
        attention_output_weight_grad_bf16_launch<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, true, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING>(
            left_map_val, right_map_val, output_map_val, output_ptr, split_sums_tensor.data(), rows_val, columns_val, sum_length_val, splits_val,
            block_rows_per_group_val, blocks_val);
        // 4. the splits' sums in order, onto weight_grad, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES values a thread
        check_threads_per_block(sum_threads_per_block_val, "attention_output_weight_grad_bf16_splits_sum_kernel");
        size_t total_stores_val = rows_val * columns_val / ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES;
        attention_output_weight_grad_bf16_splits_sum_kernel<<<(unsigned)CEIL_DIV(total_stores_val, (size_t)sum_threads_per_block_val), sum_threads_per_block_val>>>(
            output_ptr, split_sums_tensor.data(), rows_val, columns_val, splits_val);
        cudaCheck(cudaGetLastError());
    } else {
        attention_output_weight_grad_bf16_launch<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, false, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING>(
            left_map_val, right_map_val, output_map_val, output_ptr, nullptr, rows_val, columns_val, sum_length_val, 1,
            block_rows_per_group_val, blocks_val);
    }
}

/// Gives the split count SPLITS 0 stands for: the one whose pieces fill the blocks' rounds best on this GPU. The
/// blocks the GPU runs at once walk the pieces in rounds; a count's cost is its rounds x (a piece's steps + what
/// a split piece's store costs, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLIT_COST_STEPS; none unsplit). The least cost wins, the smaller count on a tie.
///
///   rows_val, columns_val, sum_length_val   the product's sizes: weight_grad {rows, columns}, the sum over sum_length
/// The template parameters: as in attention_output_weight_grad_bf16_at.
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS, int STAGED_ROW_PADDING>
inline int attention_output_weight_grad_bf16_auto_splits(size_t rows_val, size_t columns_val, size_t sum_length_val) {
    // 1. the units of work unsplit (a cluster's squares), the clusters running at once, the steps of a square
    size_t units_val = CEIL_DIV(columns_val, (size_t)COLUMNS) * CEIL_DIV(rows_val, (size_t)(CLUSTER * ROW_GROUPS * HARDWARE_WGMMA_M));
    size_t clusters_at_once_val = (size_t)attention_output_weight_grad_bf16_blocks_at_once<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, true, PRODUCER_REGISTERS, IN_FLIGHT,
                                                         STAGED_ROWS, STAGED_ROW_PADDING>() / CLUSTER;
    if (clusters_at_once_val < 1) clusters_at_once_val = 1;
    size_t steps_val = CEIL_DIV(sum_length_val, (size_t)ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K);
    // 2. each count's cost in steps: rounds x (a piece's steps + the split store's)
    int best_splits_val = 1;
    size_t best_cost_val = CEIL_DIV(units_val, clusters_at_once_val) * steps_val;
    for (int splits_val = 2; splits_val <= ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_MAX_AUTO_SPLITS && (size_t)splits_val <= steps_val; splits_val++) {
        size_t cost_val = CEIL_DIV(units_val * splits_val, clusters_at_once_val) * (CEIL_DIV(steps_val, (size_t)splits_val) + ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_SPLIT_COST_STEPS);
        if (cost_val < best_cost_val) {
            best_cost_val = cost_val;
            best_splits_val = splits_val;
        }
    }
    return best_splits_val;
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Gives how many of one kernel's blocks the GPU runs at once: SMs x the blocks an SM holds. The first time for
/// each kernel, it also asks CUDA to allow the kernel's dynamic shared memory (past the 48 KB a block gets without
/// asking).
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS,
          int STAGED_ROW_PADDING>
inline int attention_output_weight_grad_bf16_blocks_at_once() {
    constexpr size_t SHARED_BYTES = attention_output_weight_grad_bf16_shared_bytes<COLUMNS, STAGES, ROW_GROUPS, STAGED_ROWS, STAGED_ROW_PADDING>();
    constexpr int THREADS = (ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS;
    static int blocks_at_once_val = 0;                       // one for each kernel this template makes
    if (blocks_at_once_val == 0) {
        auto kernel_ptr = attention_output_weight_grad_bf16_kernel<COLUMNS, STAGES, ROW_GROUPS, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS,
                                     STAGED_ROW_PADDING>;
        int device_index_val = 0, total_sms_val = 0, blocks_per_sm_val = 0;
        cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SHARED_BYTES));
        cudaCheck(cudaGetDevice(&device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&total_sms_val, cudaDevAttrMultiProcessorCount, device_index_val));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_val, kernel_ptr, THREADS, SHARED_BYTES));
        blocks_at_once_val = total_sms_val * (blocks_per_sm_val > 0 ? blocks_per_sm_val : 1);
    }
    return blocks_at_once_val;
}

/// Gives a consumer warpgroup's room for the store: its 64 rows as bfloat16 for TMA (unsplit), or a round of the
/// staged floats (split), whichever is larger, rounded up to whole HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atoms (TMA's
/// swizzle needs each box on one).
template <int COLUMNS, int STAGED_ROWS, int STAGED_ROW_PADDING>
__host__ __device__ constexpr size_t attention_output_weight_grad_bf16_room_bytes() {
    return CEIL_DIV((STAGED_ROWS * (COLUMNS + STAGED_ROW_PADDING) * sizeof(float) > HARDWARE_WGMMA_M * COLUMNS * sizeof(nv_bfloat16)
                         ? STAGED_ROWS * (COLUMNS + STAGED_ROW_PADDING) * sizeof(float)
                         : HARDWARE_WGMMA_M * COLUMNS * sizeof(nv_bfloat16)),
                    (size_t)HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) * HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
}

/// Gives the dynamic shared memory a block takes: STAGES sets of tiles (a left tile a consumer warpgroup and the
/// right tile), each consumer warpgroup's room for a round of the store, and one atom of room to start the tiles on one.
template <int COLUMNS, int STAGES, int ROW_GROUPS, int STAGED_ROWS, int STAGED_ROW_PADDING>
constexpr size_t attention_output_weight_grad_bf16_shared_bytes() {
    return (size_t)STAGES * (ROW_GROUPS * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_LEFT_TILE_VALUES + COLUMNS * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K) * sizeof(nv_bfloat16) +
           (size_t)ROW_GROUPS * attention_output_weight_grad_bf16_room_bytes<COLUMNS, STAGED_ROWS, STAGED_ROW_PADDING>() + HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
}

/// Gives the static shared memory a block takes besides: its barriers, 8 bytes each (two a set of tiles, one a
/// consumer warpgroup for its store, and one for weight_grad's old rows). The dynamic and this together must fit
/// HARDWARE_MAX_SHARED_BYTES_PER_BLOCK.
template <int STAGES, int ROW_GROUPS>
constexpr size_t attention_output_weight_grad_bf16_barrier_bytes() {
    return (size_t)(2 * STAGES + 2 * ROW_GROUPS) * sizeof(uint64_t);
}

/// Launches one attention_output_weight_grad_bf16_kernel: clusters of CLUSTER blocks, each block a producer and ROW_GROUPS consumer warpgroups,
/// with the dynamic shared memory its sets of tiles and its store take; as many blocks as the GPU runs at once
/// (attention_output_weight_grad_bf16_blocks_at_once), or blocks_val.
///
///   left_map_val, right_map_val          TMA's descriptions of left's and right's tiles
///   output_map_val                       ... and weight_grad's
///   output_ptr, split_sums_ptr           weight_grad; the splits' scratch (SPLIT), or nullptr
///   rows_val, columns_val, sum_length_val   the product's sizes
///   splits_val, block_rows_per_group_val, blocks_val   as in attention_output_weight_grad_bf16_at
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS,
          int STAGED_ROW_PADDING>
inline void attention_output_weight_grad_bf16_launch(const CUtensorMap& left_map_val, const CUtensorMap& right_map_val, const CUtensorMap& output_map_val,
                nv_bfloat16* output_ptr, float* split_sums_ptr,
                       size_t rows_val, size_t columns_val, size_t sum_length_val, int splits_val, int block_rows_per_group_val, int blocks_val) {
    constexpr size_t SHARED_BYTES = attention_output_weight_grad_bf16_shared_bytes<COLUMNS, STAGES, ROW_GROUPS, STAGED_ROWS, STAGED_ROW_PADDING>();
    constexpr int THREADS = (ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS;
    static_assert(SHARED_BYTES + attention_output_weight_grad_bf16_barrier_bytes<STAGES, ROW_GROUPS>() <= HARDWARE_MAX_SHARED_BYTES_PER_BLOCK,
                  "the sets of tiles, the store's room and the barriers fit a block's shared memory");
    // 1. the kernel, and the blocks the GPU runs at once (the first time: its shared memory allowed)
    auto kernel_ptr = attention_output_weight_grad_bf16_kernel<COLUMNS, STAGES, ROW_GROUPS, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS,
                                 STAGED_ROW_PADDING>;
    int blocks_at_once_val = attention_output_weight_grad_bf16_blocks_at_once<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS,
                                                STAGED_ROW_PADDING>();
    // 2. the clusters: a cluster's unit of work is CLUSTER squares one under the other, for one split; the blocks
    //    asked for in whole clusters, at least one, at most one a unit
    size_t total_units_val = CEIL_DIV(columns_val, (size_t)COLUMNS) * CEIL_DIV(rows_val, (size_t)(CLUSTER * ROW_GROUPS * HARDWARE_WGMMA_M)) *
                             (size_t)splits_val;
    size_t total_clusters_val = (blocks_val > 0 ? (size_t)blocks_val : (size_t)blocks_at_once_val) / CLUSTER;
    if (total_clusters_val < 1) total_clusters_val = 1;
    if (total_clusters_val > total_units_val) total_clusters_val = total_units_val;
    unsigned total_blocks_val = (unsigned)(total_clusters_val * CLUSTER);
    // 3. launch; with clusters, by cudaLaunchKernelEx, which takes the cluster's size
    if (CLUSTER == 1) {
        kernel_ptr<<<total_blocks_val, THREADS, SHARED_BYTES>>>(left_map_val, right_map_val, output_map_val, output_ptr, split_sums_ptr, (int)rows_val,
                                                                (int)columns_val, (int)sum_length_val, splits_val, block_rows_per_group_val);
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
        cudaCheck(cudaLaunchKernelEx(&config_val, kernel_ptr, left_map_val, right_map_val, output_map_val, output_ptr, split_sums_ptr, (int)rows_val,
                                     (int)columns_val, (int)sum_length_val, splits_val, block_rows_per_group_val));
    }
}

/// Walks this block's squares of weight_grad, square after square (persistent), with a producer warpgroup and
/// ROW_GROUPS consumer warpgroups (see the top of the file), and adds each square onto weight_grad through TMA
/// (SPLIT: writes it into its split's slice of the scratch).
///
///   left_map_val, right_map_val            TMA's descriptions of left's and right's tiles
///   output_map_val                         ... and weight_grad's: its old rows in, the new ones out (not SPLIT)
///   output_ptr                             weight_grad {rows, columns} (unused: TMA reads and writes it by output_map_val)
///   split_sums_ptr                         SPLIT: the scratch {splits, rows, columns}; otherwise nullptr
///   rows_val, columns_val, sum_length_val  the product's sizes
///   splits_val                             the splits of the sum (1 when not SPLIT)
///   block_rows_per_group_val               the squares' order: rows of units a group runs down
template <int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int STAGED_ROWS,
          int STAGED_ROW_PADDING>
__global__ void __launch_bounds__((ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS)
attention_output_weight_grad_bf16_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
           const __grid_constant__ CUtensorMap output_map_val, nv_bfloat16* output_ptr, float* split_sums_ptr, int rows_val, int columns_val, int sum_length_val,
           int splits_val, int block_rows_per_group_val) {
    constexpr bool LEFT_TURNED = true;                                                                      // output_grad {sum_length, rows}: turned
    constexpr bool RIGHT_TURNED = true;                                                                     // input {sum_length, columns}: turned
    constexpr int COLUMN_GROUPS = COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS;                                   // groups of 8 across the square
    constexpr int RIGHT_TILE_VALUES = COLUMNS * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K;                                   // a right tile: COLUMNS lines of a step's k
    constexpr int BLOCK_ROWS = ROW_GROUPS * HARDWARE_WGMMA_M;                                              // a square's rows
    constexpr int UNIT_ROWS = CLUSTER * BLOCK_ROWS;                                                        // a cluster's unit: its squares one under the other
    constexpr int LEFT_VALUES = ROW_GROUPS * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_LEFT_TILE_VALUES;                             // a set's left tiles, one a warpgroup
    constexpr int SET_VALUES = LEFT_VALUES + RIGHT_TILE_VALUES;                                             // one set: left tiles, then right tile
    constexpr int LEFT_RUNNING = STAGES > IN_FLIGHT ? IN_FLIGHT : STAGES - 1;                                // steps of WGMMAs left running: IN_FLIGHT, at most STAGES - 1 (the set must be free)
    static_assert(IN_FLIGHT >= 0, "IN_FLIGHT is 0 or more steps");
    constexpr int RIGHT_BOXES = RIGHT_TURNED ? COLUMNS / ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES : 1;                   // TMA boxes in a right tile
    constexpr int RIGHT_BOX_VALUES = RIGHT_TURNED ? ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K : RIGHT_TILE_VALUES;
    constexpr uint32_t SET_BYTES = SET_VALUES * sizeof(nv_bfloat16);                                         // what TMA brings a block a step
    constexpr int STAGED_ROW_FLOATS = COLUMNS + STAGED_ROW_PADDING;                                         // a staged row
    constexpr int STORE_ROUNDS = HARDWARE_WGMMA_M / STAGED_ROWS;                                            // rounds to store a warpgroup's 64 rows
    static_assert(HARDWARE_WGMMA_M % STAGED_ROWS == 0, "a warpgroup's 64 rows are staged in whole rounds");
    static_assert(STAGED_ROW_PADDING % HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP == 0, "a staged row stays on 16 bytes: its float4 reads");
    constexpr int STORES_PER_ROW = COLUMNS / ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES;                                              // a row's 8-byte stores
    constexpr int OUTPUT_BOXES = COLUMNS / ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES;                                                  // TMA boxes across a warpgroup's 64 rows
    constexpr int PRODUCER_THREADS = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_PRODUCER_WARPGROUPS * HARDWARE_WARPGROUP_THREADS;
    // the registers (the worklog's kernel 5b): a block of THREADS gets LAUNCH_REGISTERS a thread, too few for 2
    // consumer warpgroups' 128 sums each; the producer needs few, so with 2 it gives back all but
    // PRODUCER_REGISTERS and the consumers take them
    constexpr int THREADS = PRODUCER_THREADS + ROW_GROUPS * HARDWARE_WARPGROUP_THREADS;
    constexpr int LAUNCH_REGISTERS = HARDWARE_REGISTERS_PER_SM / THREADS / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    static_assert(PRODUCER_REGISTERS >= HARDWARE_SETMAXNREG_MIN && PRODUCER_REGISTERS % HARDWARE_REGISTER_ALLOCATION_UNIT == 0,
                  "setmaxnreg takes at least HARDWARE_SETMAXNREG_MIN, in whole allocation units");
    constexpr int CONSUMER_REGISTERS_FIT = (LAUNCH_REGISTERS * THREADS - PRODUCER_REGISTERS * PRODUCER_THREADS) /
                                           (ROW_GROUPS * HARDWARE_WARPGROUP_THREADS) / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    constexpr int CONSUMER_REGISTERS = CONSUMER_REGISTERS_FIT < HARDWARE_SETMAXNREG_MAX ? CONSUMER_REGISTERS_FIT : HARDWARE_SETMAXNREG_MAX;
    constexpr bool MOVE_REGISTERS = ROW_GROUPS > 1;           // 1 consumer warpgroup: 255 a thread already, plenty
    // dynamic shared memory, from its next HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom on: the sets of tiles in TMA's
    // swizzled layout, then each consumer warpgroup's room for the store. Static: each set's two barriers, and one a
    // consumer warpgroup for its store
    extern __shared__ __align__(16) uint8_t attention_output_weight_grad_bf16_shared_list[];
    __shared__ uint64_t full_barriers_list[STAGES];          // set s's bytes are in
    __shared__ uint64_t empty_barriers_list[STAGES];         // every consumer of the cluster is done with set s
    __shared__ uint64_t store_barriers_list[ROW_GROUPS];     // consumer warpgroup g's threads are all at the same point of its store
    __shared__ uint64_t old_barriers_list[ROW_GROUPS];       // consumer warpgroup g's 64 rows of the old weight_grad are in its room
    uint32_t shared_address_val = (uint32_t)__cvta_generic_to_shared(attention_output_weight_grad_bf16_shared_list);
    nv_bfloat16* tiles_ptr = reinterpret_cast<nv_bfloat16*>(
        attention_output_weight_grad_bf16_shared_list + (HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM - shared_address_val % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM);
    float* staged_rooms_ptr = reinterpret_cast<float*>(tiles_ptr + STAGES * SET_VALUES);
    // where a WGMMA's 16 values of k start in a tile: turned, 16 rows on (16 x 64 values)
    const int left_k_step_values_val = LEFT_TURNED ? HARDWARE_WGMMA_K_BF16 * HARDWARE_WGMMA_M : HARDWARE_WGMMA_K_BF16;
    const int right_k_step_values_val = RIGHT_TURNED ? HARDWARE_WGMMA_K_BF16 * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES : HARDWARE_WGMMA_K_BF16;
    // the right descriptor's leading offset, in cells: a turned right tile of more than 64 columns is boxes side by
    // side, one box apart; otherwise not used (the 128-byte swizzle's own step), the usual 1
    const uint32_t right_leading_cells_val = RIGHT_TURNED && COLUMNS > ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES
                                                 ? (uint32_t)(RIGHT_BOX_VALUES * sizeof(nv_bfloat16) / HARDWARE_WGMMA_DESCRIPTOR_UNIT)
                                                 : (uint32_t)(HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);

    // 1. the barriers, by one thread; with clusters, every block of the cluster waits until all have theirs
    if (threadIdx.x == 0) {
#pragma unroll
        for (int set_index_val = 0; set_index_val < STAGES; set_index_val++) {
            attention_output_weight_grad_bf16_barrier_init(&full_barriers_list[set_index_val], 1);                       // the producer's one arrival, and the bytes
            attention_output_weight_grad_bf16_barrier_init(&empty_barriers_list[set_index_val], ROW_GROUPS * CLUSTER);   // one arrival a consumer warpgroup of the cluster
        }
#pragma unroll
        for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++)
            attention_output_weight_grad_bf16_barrier_init(&store_barriers_list[row_group_index_val], HARDWARE_WARPGROUP_THREADS);   // every thread of the warpgroup
#pragma unroll
        for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++)
            attention_output_weight_grad_bf16_barrier_init(&old_barriers_list[row_group_index_val], 1);   // the warpgroup's first thread, and the bytes
        attention_output_weight_grad_bf16_fence_for_tma();                     // the barriers visible to TMA
    }
    __syncthreads();
    if (CLUSTER > 1) attention_output_weight_grad_bf16_cluster_sync<true>();

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
    int all_steps_val = CEIL_DIV(sum_length_val, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K);
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
        if (MOVE_REGISTERS) attention_output_weight_grad_bf16_give_registers<PRODUCER_REGISTERS>();   // the whole warpgroup
        // 3. the producer: its thread 0 asks for every step's tiles of every unit, set after set (walk_step counts
        //    the steps since the launch: its set is walk_step % STAGES, the set's lap walk_step / STAGES). From the
        //    second lap on, it first waits until the cluster's consumers are done with the set (its empty barrier's
        //    phase lap - 1). A step's ask: tell the set's full barrier how many bytes are coming, then the left tiles
        //    and the right tile's boxes: with clusters, block 0 asks for the right tile for every block. A box's
        //    corner is (column, row) of the matrix as it sits, turned: (line, k); past the edges TMA gives 0s
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
                        attention_output_weight_grad_bf16_barrier_wait(&empty_barriers_list[set_index_val], (uint32_t)((lap_index_val - 1) % 2));
                    nv_bfloat16* left_tile_ptr = tiles_ptr + set_index_val * SET_VALUES;
                    nv_bfloat16* right_tile_ptr = left_tile_ptr + LEFT_VALUES;
                    int step_first_k_index_val = (first_step_index_val + step_index_val) * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K;
                    attention_output_weight_grad_bf16_barrier_expect(&full_barriers_list[set_index_val], SET_BYTES);
#pragma unroll
                    for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++)
                        attention_output_weight_grad_bf16_copy_tile(left_tile_ptr + row_group_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_LEFT_TILE_VALUES, &left_map_val,
                                                   block_first_row_index_val + row_group_index_val * HARDWARE_WGMMA_M, step_first_k_index_val,
                                                   &full_barriers_list[set_index_val]);
#pragma unroll
                    for (int box_index_val = 0; box_index_val < RIGHT_BOXES; box_index_val++) {
                        int box_column_val = block_first_column_index_val + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES;
                        if (CLUSTER == 1)
                            attention_output_weight_grad_bf16_copy_tile(right_tile_ptr + box_index_val * RIGHT_BOX_VALUES, &right_map_val, box_column_val,
                                                       step_first_k_index_val, &full_barriers_list[set_index_val]);
                        else if (cluster_block_index_val == 0)
                            attention_output_weight_grad_bf16_copy_tile_to_cluster(right_tile_ptr + box_index_val * RIGHT_BOX_VALUES, &right_map_val,
                                                                  box_column_val, step_first_k_index_val, &full_barriers_list[set_index_val],
                                                                  (uint16_t)((1u << CLUSTER) - 1));
                    }
                }
            }
        }
    } else {
        if (MOVE_REGISTERS) attention_output_weight_grad_bf16_take_registers<CONSUMER_REGISTERS>();   // the whole warpgroup
        // 4. a consumer: walk each unit's sum on its own left tile and the shared right tile, then store its rows
        int consumer_thread_index_val = (int)threadIdx.x - PRODUCER_THREADS;       // 0 on: warpgroup g's threads 128 g on
        int row_group_index_val = consumer_thread_index_val / HARDWARE_WARPGROUP_THREADS;
        int warpgroup_thread_index_val = consumer_thread_index_val % HARDWARE_WARPGROUP_THREADS;
        bool warpgroup_first_thread_val = warpgroup_thread_index_val == 0;
        int warp_index_val = warpgroup_thread_index_val / HARDWARE_THREADS_PER_WARP;   // in the warpgroup: its rows 16 w on
        int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
        int thread_first_row_in_group_val = warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
        int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
        uint8_t* room_ptr = reinterpret_cast<uint8_t*>(staged_rooms_ptr) + row_group_index_val * attention_output_weight_grad_bf16_room_bytes<COLUMNS, STAGED_ROWS, STAGED_ROW_PADDING>();
        float* staged_ptr = reinterpret_cast<float*>(room_ptr);                    // SPLIT: the room as staged floats
        uint32_t store_phase_val = 0;                                              // the store barrier's phase to wait for next
        uint32_t old_phase_val = 0;                                                // ... and the old rows' barrier's
        // this warpgroup is done with a step's set: its first thread arrives at the set's empty barrier in every block
        // of the cluster
        auto free_set_val = [&](int set_index_val) {
            if (!warpgroup_first_thread_val) return;
            if (CLUSTER == 1) {
                attention_output_weight_grad_bf16_barrier_arrive(&empty_barriers_list[set_index_val]);
            } else {
#pragma unroll
                for (int cluster_block_val = 0; cluster_block_val < CLUSTER; cluster_block_val++)
                    attention_output_weight_grad_bf16_barrier_arrive_in_cluster(&empty_barriers_list[set_index_val], (uint32_t)cluster_block_val);
            }
        };
        // every thread of this warpgroup at the same point: all arrive at its store barrier and wait for the phase
        auto meet_warpgroup_val = [&]() {
            attention_output_weight_grad_bf16_barrier_arrive(&store_barriers_list[row_group_index_val]);
            attention_output_weight_grad_bf16_barrier_wait(&store_barriers_list[row_group_index_val], store_phase_val);
            store_phase_val ^= 1;
        };

        int walk_step_index_val = 0;                                               // steps since the launch, as the producer counts them
        for (int unit_index_val = cluster_index_val; unit_index_val < total_units_val; unit_index_val += total_clusters_val) {
            int block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val, total_steps_val;
            place_unit_val(unit_index_val, block_first_row_index_val, block_first_column_index_val, split_index_val, first_step_index_val,
                           total_steps_val);
            int group_first_row_index_val = block_first_row_index_val + row_group_index_val * HARDWARE_WGMMA_M;
            // unsplit: before the walk, the warpgroup's first thread asks TMA for its 64 rows of the old weight_grad,
            // into its room (once TMA has read the room for the unit before), so they come while the tensor cores
            // work; past the edges TMA gives 0s
            if constexpr (!SPLIT)
                if (warpgroup_first_thread_val) {
                    attention_output_weight_grad_bf16_store_wait_read();
                    attention_output_weight_grad_bf16_barrier_expect(&old_barriers_list[row_group_index_val], (uint32_t)(OUTPUT_BOXES * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_BOX_BYTES));
#pragma unroll
                    for (int box_index_val = 0; box_index_val < OUTPUT_BOXES; box_index_val++)
                        attention_output_weight_grad_bf16_copy_tile(reinterpret_cast<nv_bfloat16*>(room_ptr + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_BOX_BYTES), &output_map_val,
                                      block_first_column_index_val + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES, group_first_row_index_val,
                                      &old_barriers_list[row_group_index_val]);
                }
            float sums_list[COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};   // this thread's part of its warpgroup's 64 rows
            for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
                // a. wait until this step's set is full: its barrier's phase is the set's lap
                int set_index_val = (walk_step_index_val + step_index_val) % STAGES;
                attention_output_weight_grad_bf16_barrier_wait(&full_barriers_list[set_index_val], (uint32_t)(((walk_step_index_val + step_index_val) / STAGES) % 2));

                // b. the tensor cores: ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16 WGMMAs, each 16 more values of k
                const nv_bfloat16* left_tile_ptr = tiles_ptr + set_index_val * SET_VALUES + row_group_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_LEFT_TILE_VALUES;
                const nv_bfloat16* right_tile_ptr = tiles_ptr + set_index_val * SET_VALUES + LEFT_VALUES;
                attention_output_weight_grad_bf16_wgmma_fence();                                    // the sums ready for the tensor cores
#pragma unroll
                for (int k_step_index_val = 0; k_step_index_val < ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_BLOCK_K / HARDWARE_WGMMA_K_BF16; k_step_index_val++)
                    attention_output_weight_grad_bf16_multiply_add<LEFT_TURNED, RIGHT_TURNED>(sums_list, left_tile_ptr + k_step_index_val * left_k_step_values_val,
                                                                             right_tile_ptr + k_step_index_val * right_k_step_values_val,
                                                                             right_leading_cells_val);
                attention_output_weight_grad_bf16_wgmma_commit();                                   // the step's WGMMAs as one group

                // c. wait for the WGMMAs of LEFT_RUNNING steps back (this step's own still running); this warpgroup is then
                //    done with that step's set
                attention_output_weight_grad_bf16_wgmma_wait<LEFT_RUNNING>();
                if (step_index_val >= LEFT_RUNNING)
                    free_set_val((walk_step_index_val + step_index_val - LEFT_RUNNING) % STAGES);
            }
            // d. the last steps' WGMMAs done, and their sets free: the producer may be filling them for the next unit
            attention_output_weight_grad_bf16_wgmma_wait<0>();
            for (int step_index_val = max(total_steps_val - LEFT_RUNNING, 0); step_index_val < total_steps_val; step_index_val++)
                free_set_val((walk_step_index_val + step_index_val) % STAGES);
            walk_step_index_val += total_steps_val;

            // e. unsplit: the old rows are in the room (every thread waits). Each thread adds its pairs onto the room's
            //    pair at the same place (box b: columns 64 b on, rows of 128 bytes, the 16-byte cell c of row r at c XOR
            //    r % 8, as TMA's 128-byte swizzle has it), rounded once to bfloat16, makes its writes visible to TMA, and
            //    the first thread asks TMA to store the boxes; past the edges TMA writes nothing. Then on to the next
            //    unit: nobody waits for the store
            if constexpr (!SPLIT) {
                attention_output_weight_grad_bf16_barrier_wait(&old_barriers_list[row_group_index_val], old_phase_val);
                old_phase_val ^= 1;
#pragma unroll
                for (int group_index_val = 0; group_index_val < COLUMN_GROUPS; group_index_val++) {
                    int column_in_square_val = group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
                    int box_index_val = column_in_square_val / ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES;
                    int column_in_box_val = column_in_square_val % ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES;
                    int cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) / HARDWARE_TMA_BYTES_PER_CELL;
                    int byte_in_cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) % HARDWARE_TMA_BYTES_PER_CELL;
#pragma unroll
                    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
                        int row_in_group_val = thread_first_row_in_group_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                        int swizzled_cell_val = cell_val ^ (row_in_group_val % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN);
                        nv_bfloat162* room_pair_ptr = reinterpret_cast<nv_bfloat162*>(
                            room_ptr + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_BOX_BYTES + row_in_group_val * HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW +
                            swizzled_cell_val * HARDWARE_TMA_BYTES_PER_CELL + byte_in_cell_val);
                        float2 old_val = __bfloat1622float2(*room_pair_ptr);
                        *room_pair_ptr = __floats2bfloat162_rn(
                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] + old_val.x,
                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] + old_val.y);
                    }
                }
                attention_output_weight_grad_bf16_fence_for_tma();                                               // this thread's writes visible to TMA
                meet_warpgroup_val();                                              // every thread's
                if (warpgroup_first_thread_val) {
#pragma unroll
                    for (int box_index_val = 0; box_index_val < OUTPUT_BOXES; box_index_val++)
                        attention_output_weight_grad_bf16_store_box(&output_map_val, block_first_column_index_val + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_ROW_VALUES, group_first_row_index_val,
                                      room_ptr + box_index_val * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_BOX_BYTES);
                    attention_output_weight_grad_bf16_store_commit();
                }
                continue;
            }

            // f. split: the store, STAGED_ROWS of this warpgroup's rows a round, into its split's slice
            meet_warpgroup_val();                                                  // every thread done with the room's last round
            for (int round_index_val = 0; round_index_val < STORE_ROUNDS; round_index_val++) {
                // the round's sums into the warpgroup's room as floats, each thread's pairs at their row and column:
                // the threads whose rows are the round's
#pragma unroll
                for (int group_index_val = 0; group_index_val < COLUMN_GROUPS; group_index_val++) {
                    int column_in_square_val = group_index_val * HARDWARE_WGMMA_GROUP_COLUMNS + thread_pair_column_index_val;
#pragma unroll
                    for (int thread_row_index_val = 0; thread_row_index_val < HARDWARE_WGMMA_ROWS_PER_THREAD; thread_row_index_val++) {   // rows group and group + 8
                        int row_in_group_val = thread_first_row_in_group_val + thread_row_index_val * HARDWARE_WGMMA_ROWS_PER_WARP;
                        if (row_in_group_val / STAGED_ROWS != round_index_val) continue;
                        *reinterpret_cast<float2*>(staged_ptr + (row_in_group_val % STAGED_ROWS) * STAGED_ROW_FLOATS + column_in_square_val) =
                            make_float2(sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val],
                                        sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1]);
                    }
                }
                meet_warpgroup_val();                                              // the round's rows are in the room

                // the rows out, row by row, ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES neighbouring values a thread: the 4 float sums into the
                // split's slice of the scratch as they are, one 16-byte store; nothing past the edges (columns is a
                // multiple of 8, so a thread's 4 values are all in or all out)
                for (int store_index_val = warpgroup_thread_index_val; store_index_val < STAGED_ROWS * STORES_PER_ROW;
                     store_index_val += HARDWARE_WARPGROUP_THREADS) {
                    int row_in_round_val = store_index_val / STORES_PER_ROW;
                    int column_in_square_val = ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES * (store_index_val % STORES_PER_ROW);
                    int row_index_val = group_first_row_index_val + round_index_val * STAGED_ROWS + row_in_round_val;
                    int column_index_val = block_first_column_index_val + column_in_square_val;
                    if (row_index_val >= rows_val || column_index_val >= columns_val) continue;
                    float4 values_val = *reinterpret_cast<const float4*>(staged_ptr + row_in_round_val * STAGED_ROW_FLOATS + column_in_square_val);
                    size_t output_index_val = (size_t)row_index_val * columns_val + column_index_val;
                    *reinterpret_cast<float4*>(split_sums_ptr + (size_t)split_index_val * rows_val * columns_val + output_index_val) = values_val;
                }
                meet_warpgroup_val();                                              // every thread has read the room: the next round may write it
            }
        }
    }

    // 5. unsplit: the consumers' TMA stores all written before the block leaves
    if constexpr (!SPLIT)
        if (threadIdx.x >= PRODUCER_THREADS && (threadIdx.x - PRODUCER_THREADS) % HARDWARE_WARPGROUP_THREADS == 0) attention_output_weight_grad_bf16_store_wait_all();

    // 6. with clusters: every thread of the cluster meets before any block leaves, so no block's shared memory goes
    //    while another block may still write into it (the right tile's multicast, the arrivals at its empty
    //    barriers). Not .aligned: the producer's threads come from a branch only its thread 0 took
    if (CLUSTER > 1) attention_output_weight_grad_bf16_cluster_sync<false>();
}

/// Adds the splits' float sums of ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES neighbouring values of weight_grad, in order of the splits,
/// onto weight_grad, rounded once, one 8-byte store: what a split product's blocks leave undone. One thread per 4
/// values, the 4 a thread's neighbours and a warp's 32 neighbours too.
///
///   output_ptr              weight_grad {rows, columns}: added onto
///   split_sums_ptr          {splits, rows, columns} float: each split's sums
///   rows_val, columns_val   weight_grad's shape (columns a multiple of ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES)
///   splits_val              the slices
__global__ void attention_output_weight_grad_bf16_splits_sum_kernel(nv_bfloat16* output_ptr, const float* split_sums_ptr, size_t rows_val, size_t columns_val,
                                      int splits_val) {
    size_t output_index_val = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_STORE_VALUES;
    if (output_index_val >= rows_val * columns_val) return;
    // 1. the splits' sums, in order
    size_t slice_values_val = rows_val * columns_val;
    float4 sum_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int split_index_val = 0; split_index_val < splits_val; split_index_val++) {
        float4 split_val = *reinterpret_cast<const float4*>(split_sums_ptr + split_index_val * slice_values_val + output_index_val);
        sum_val = make_float4(sum_val.x + split_val.x, sum_val.y + split_val.y, sum_val.z + split_val.z, sum_val.w + split_val.w);
    }
    // 2. + weight_grad, rounded once, one 8-byte store
    nv_bfloat162* output_pairs_ptr = reinterpret_cast<nv_bfloat162*>(output_ptr + output_index_val);
    float2 first_old_val = __bfloat1622float2(output_pairs_ptr[0]);
    float2 second_old_val = __bfloat1622float2(output_pairs_ptr[1]);
    nv_bfloat162 pairs_list[2] = {__floats2bfloat162_rn(sum_val.x + first_old_val.x, sum_val.y + first_old_val.y),
                                  __floats2bfloat162_rn(sum_val.z + second_old_val.x, sum_val.w + second_old_val.y)};
    *reinterpret_cast<uint2*>(output_pairs_ptr) = *reinterpret_cast<const uint2*>(pairs_list);
}

#endif // ATTENTION_OUTPUT_WEIGHT_GRAD_BF16_CUH
