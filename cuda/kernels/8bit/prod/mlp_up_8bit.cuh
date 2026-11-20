// mlp_up_8bit.cuh — mlp_up_output = ln2_output · mlp_up_weightᵀ + mlp_up_bias, and gelu_output = gelu(mlp_up_output)
// beside it, in 8 bits (FP8 E4M3) on an H100's tensor cores (WGMMA, fed by TMA).
//
//     tensor        type in memory                     shape
//     input         bfloat16                           {rows, in_features}
//     weight        bfloat16                           {out_features, in_features}
//     bias          bfloat16                           {out_features}
//     gelu_input    bfloat16                           {rows, out_features}
//     gelu_output   bfloat16                           {rows, out_features}
//                   or E4M3 (mlp_up_8bit_gelu_8bit)    rows {rows, out} and turned
//                   , one scale for the tensor         {out, rows}
//     the copies    E4M3, one float scale each         MlpUp8bitCopies's: left {rows, k},
//                                                      right {columns, k}
//     sums          float                              in registers
#ifndef MLP_UP_8BIT_CUH
#define MLP_UP_8BIT_CUH

#include <type_traits>              // std::true_type: the split kernel or not
#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include <cuda_bf16.h>
#include <cuda_fp8.h>                // FP8 E4M3
#include "../../cuda_common.h"       // cudaCheck, check_threads_per_block, the HARDWARE_ numbers
#include "../../../common/tensor.cuh"
#include "../round_8bit.cuh"         // the 8-bit copies: round_8bit_per_tensor, ROUND_8BIT_ROW_LENGTH_MULTIPLE

// the setting: tune_mlp_up_8bit sweeps each and prints the fastest
#define MLP_UP_8BIT_ROW_GROUPS 2                       // consumer warpgroups a block, 64 rows each: 1 or 2
#define MLP_UP_8BIT_COLUMNS 256                        // a square's columns, its WGMMA's N: 64, 128, or 256 with PROMOTION_STEPS 0
#define MLP_UP_8BIT_STAGES 4                           // sets of tiles in shared memory: 1 to MLP_UP_8BIT_MAX_STAGES
#define MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP 4            // rows of squares a group of blocks runs down before the next column (1: row after row)
#define MLP_UP_8BIT_CLUSTER 1                          // blocks a cluster, sharing each right tile: 1 (none) or 2
#define MLP_UP_8BIT_BLOCKS 0                           // blocks a launch: 0, as many as the GPU runs at once (persistent); MLP_UP_8BIT_ONE_A_SQUARE: one a square
#define MLP_UP_8BIT_PRODUCER_REGISTERS 48                // with 2 consumer warpgroups: the producer's registers a thread (setmaxnreg), the consumers take the rest; tune_mlp_up_8bit sweeps it
#define MLP_UP_8BIT_IN_FLIGHT 1                          // steps of WGMMAs left running while the threads go on (STAGES - 1 at most: the set must be free); tune_mlp_up_8bit sweeps it
#define MLP_UP_8BIT_PROMOTION_STEPS 0                    // steps the tensor cores add up before their sum moves into float registers; 0: never. wgmma_matmul_8bit's 4: as accurate as 1 (H100 NVL); tune_mlp_up_8bit sweeps it
#define MLP_UP_8BIT_PINGPONG 1                           // the consumer warpgroups' turns: 0, cooperative (both on a square of 2 x 64 rows, sharing its right tile); 1, ping-pong (each its own squares of 64 rows, their sums in turns, one's store under the other's sums; ROW_GROUPS 2, CLUSTER 1). The 2 x 256 x 4 ping-pong square as bf16's mlp_up found best: tune_mlp_up_8bit to confirm
// the walk's own numbers
#define MLP_UP_8BIT_ONE_A_SQUARE (1 << 30)              // BLOCKS: more than any product's squares, so one block a square
#define MLP_UP_8BIT_MAX_ROW_GROUPS 2                     // the most: 2 x 128 threads, each warpgroup's two sets of sums inside a block's registers
#define MLP_UP_8BIT_MAX_STAGES 4                         // the most sets of tiles: with the room, a square must fit HARDWARE_MAX_SHARED_BYTES_PER_BLOCK
#define MLP_UP_8BIT_MAX_CLUSTER 2                        // the most blocks a cluster
#define MLP_UP_8BIT_PRODUCER_WARPGROUPS 1                // warpgroups of a block that only load tiles: the first one (its thread 0 asks TMA)
#define MLP_UP_8BIT_GELU_SCALE 0.7978845608f            // GPT-2's tanh GELU: sqrt(2 / pi), as fp32/gelu.cuh's (the same bits as the unfused GELU)
#define MLP_UP_8BIT_GELU_CUBE_FACTOR 0.044715f          // how much of x³ goes into s

// ── the hardware calls: this kernel's own, one or a few PTX instructions each ──────
//     barrier_init, _expect, _arrive, _wait  an mbarrier in shared memory: "this set is full" / "this set is free"
//     barrier_arrive_in_cluster              the same, at the barrier of another block of the cluster
//     copy_tile, copy_tile_to_cluster        TMA: one box of a matrix into shared memory (or every block's of the cluster)
//     store_box, store_commit, store_wait_*  TMA: one box of shared memory out to a matrix, and waiting for it
//     fence_for_tma                          this thread's writes to shared memory made visible to TMA
//     descriptor, multiply_add               WGMMA: where a tile sits, and sums {64, N} += left tile · right tileᵀ (E4M3)
//     wgmma_fence, wgmma_commit, wgmma_wait  WGMMA's groups: the sums ready, a step's group closed, waiting for it
//     cluster_sync                           every thread of the cluster at the same point
//     give_registers, take_registers         setmaxnreg: a warpgroup gives back or takes registers
//     tanh                                   tanh.approx.f32: tanh in one instruction, for GELU
//     tensor_map, tensor_map_8bit            (host) TMA's description of a bfloat16 matrix in 64-value boxes, an E4M3 one in 128
//     tensor_map_turned_8bit                 (host) ... and of gelu_output's turned E4M3 copy, in boxes of COLUMNS lines x 64 bytes, plain
//     check_tensor                           (host) a tensor TMA can read: bfloat16, rows of whole 16-byte cells

// What the hardware's 128-byte swizzle fixes: a tile's row is one swizzled row of 128 bytes, 128 E4M3s; the
// output's TMA boxes are bfloat16, 64 values a row.
#define MLP_UP_8BIT_BLOCK_K ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(__nv_fp8_e4m3)))     // values (bytes) of k a step loads: one swizzled row of E4M3s
#define MLP_UP_8BIT_ROW_VALUES ((int)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / sizeof(nv_bfloat16)))    // a TMA box's row: one swizzled row of bfloat16
#define MLP_UP_8BIT_LEFT_TILE_BYTES (HARDWARE_WGMMA_M * MLP_UP_8BIT_BLOCK_K)                  // a left tile: 64 rows of a step's k, a byte each
#define MLP_UP_8BIT_VALUE_MULTIPLE ((int)(HARDWARE_TMA_BYTES_PER_CELL / sizeof(nv_bfloat16)))       // a matrix row's multiple: whole 16-byte cells
#define MLP_UP_8BIT_STORE_BOX_BYTES (HARDWARE_WGMMA_M * MLP_UP_8BIT_ROW_VALUES * (int)sizeof(nv_bfloat16))   // a TMA store's box: 64 rows of one swizzled row
static_assert(MLP_UP_8BIT_BLOCK_K % HARDWARE_WGMMA_K_E4M3 == 0, "each step is whole WGMMAs");
static_assert(ROUND_8BIT_ROW_LENGTH_MULTIPLE == HARDWARE_TMA_BYTES_PER_CELL, "round_8bit's rows are whole 16-byte pieces, as TMA reads them");
static_assert(HARDWARE_WGMMA_M == MLP_UP_8BIT_ROW_VALUES, "a turned left tile's row (64 rows) is one swizzled row, as a tile's row of k is");

__device__ __forceinline__ uint64_t mlp_up_8bit_descriptor(const void* tile_ptr,
                                                                uint32_t leading_cells_val = HARDWARE_TMA_BYTES_PER_CELL / HARDWARE_WGMMA_DESCRIPTOR_UNIT);
template <int GROUPS>
__device__ __forceinline__ void mlp_up_8bit_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                              const uint8_t* left_tile_ptr, const uint8_t* right_tile_ptr, int add_onto_sums_val);

// ── barriers and TMA loads ────────────────────────────────────────────────────

/// Gives a shared-memory object's address as PTX's shared::cta instructions take it.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t mlp_up_8bit_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void mlp_up_8bit_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mlp_up_8bit_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void mlp_up_8bit_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    // Its phase completes when they are in.
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(mlp_up_8bit_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Arrives once at a barrier of this block.
///
///   barrier_ptr   the barrier
__device__ __forceinline__ void mlp_up_8bit_barrier_arrive(uint64_t* barrier_ptr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(mlp_up_8bit_shared_address(barrier_ptr)), "r"(1) : "memory");
}

/// Arrives once at the same barrier in another block of the cluster (or this one).
///
///   barrier_ptr         the barrier, as it sits in this block
///   cluster_block_val   which block of the cluster: 0 .. CLUSTER - 1
__device__ __forceinline__ void mlp_up_8bit_barrier_arrive_in_cluster(uint64_t* barrier_ptr, uint32_t cluster_block_val) {
    // The barrier at this address of block cluster_block_val's shared memory. Released at the block's scope (.cta,
    // as CUTLASS does), not the cluster's: what it orders is this block's own reads of the set (the WGMMAs,
    // already waited for); .release.cluster made every arrival wait on a cluster-wide fence, and clusters of 2
    // took twice as long (H100 PCIe, 2 x 256: qkv 0.250 ms against 0.139 ms with .cta, the head 5.46 against 2.55)
    asm volatile("{\n"
                 ".reg .b32 remote_address;\n"
                 "mapa.shared::cluster.u32 remote_address, %0, %1;\n"
                 "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remote_address], %2;\n"
                 "}\n"
                 ::"r"(mlp_up_8bit_shared_address(barrier_ptr)), "r"(cluster_block_val), "r"(1) : "memory");
}

/// Waits until a barrier's phase of this parity is complete.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void mlp_up_8bit_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    // The caller keeps count of the phases (the first is 0, the next 1, then 0 again), so no token is passed
    // round.
    uint32_t barrier_address_val = mlp_up_8bit_shared_address(barrier_ptr);
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
__device__ __forceinline__ void mlp_up_8bit_copy_tile(void* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                            int row_val, uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(mlp_up_8bit_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(mlp_up_8bit_shared_address(barrier_ptr))
                 : "memory");
}

/// Asks TMA for one box of a matrix into the shared memory of several blocks of the cluster at once (multicast).
///
///   tile_ptr, map_ptr, column_val, row_val, barrier_ptr   as in mlp_up_8bit_copy_tile, as they sit in this block
///   cluster_blocks_mask_val                               bit b set: block b of the cluster gets the box
__device__ __forceinline__ void mlp_up_8bit_copy_tile_to_cluster(void* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                                       int row_val, uint64_t* barrier_ptr,
                                                                       uint16_t cluster_blocks_mask_val) {
    // Read from L2 once, written at the same place in each block, its bytes counted at each block's barrier there.
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                 "[%0], [%1, {%2, %3}], [%4], %5;\n"
                 ::"r"(mlp_up_8bit_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(mlp_up_8bit_shared_address(barrier_ptr)), "h"(cluster_blocks_mask_val)
                 : "memory");
}


// ── TMA stores ────────────────────────────────────────────────────────────────

/// Makes this thread's writes to shared memory visible to TMA (the async proxy), before a TMA store reads them.
__device__ __forceinline__ void mlp_up_8bit_fence_for_tma(void) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/// Asks TMA to store one box of shared memory into a matrix, in this thread's bulk group.
///
///   map_ptr                TMA's description of the matrix
///   column_val, row_val    the box's corner in the matrix
///   box_ptr                the box in shared memory, in TMA's swizzled layout
__device__ __forceinline__ void mlp_up_8bit_store_box(const CUtensorMap* map_ptr, int column_val, int row_val, const void* box_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                 ::"l"(map_ptr), "r"(column_val), "r"(row_val), "r"(mlp_up_8bit_shared_address(box_ptr)) : "memory");
}

/// Closes this thread's bulk group: the stores asked for since the last one, waited for together.
__device__ __forceinline__ void mlp_up_8bit_store_commit(void) {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/// Waits until TMA has read every box this thread asked it to store (the shared memory may be written again).
__device__ __forceinline__ void mlp_up_8bit_store_wait_read(void) {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

/// Waits until every box this thread asked TMA to store is written to memory (before the block leaves).
__device__ __forceinline__ void mlp_up_8bit_store_wait_all(void) {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// ── WGMMA's groups, the cluster, registers ────────────────────────────────────

/// Makes the sums in registers ready for the tensor cores (before a step's WGMMAs), for the whole warpgroup.
__device__ __forceinline__ void mlp_up_8bit_wgmma_fence(void) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/// Closes the WGMMAs asked for since the last group into one group, for the whole warpgroup.
__device__ __forceinline__ void mlp_up_8bit_wgmma_commit(void) {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/// Waits until at most LEFT_RUNNING groups of this warpgroup's WGMMAs are still running.
template <int LEFT_RUNNING>
__device__ __forceinline__ void mlp_up_8bit_wgmma_wait(void) {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(LEFT_RUNNING) : "memory");
}

/// Holds every thread of the cluster at this point until all have come.
template <bool ALIGNED>
__device__ __forceinline__ void mlp_up_8bit_cluster_sync(void) {
    // Every thread of the cluster at this point before any goes on. ALIGNED: every thread of each warp comes here
    // together (.aligned, faster); otherwise threads may come from different branches.
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
__device__ __forceinline__ void mlp_up_8bit_give_registers(void) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
}

/// This warpgroup takes REGISTERS registers a thread from what the block has free (setmaxnreg).
template <int REGISTERS>
__device__ __forceinline__ void mlp_up_8bit_take_registers(void) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
}

/// Gives tanh(x) by the hardware's tanh.approx.f32.
///
///   x_val   one value
__device__ __forceinline__ float mlp_up_8bit_tanh(float x_val) {
    // One instruction on the special function unit, about 2^-11 off (relative), against tanhf's 20 to 30
    // instructions; GELU's result is rounded to bfloat16 (2^-8) after it.
    float tanh_val;
    asm("tanh.approx.f32 %0, %1;\n" : "=f"(tanh_val) : "f"(x_val));
    return tanh_val;
}

// ── WGMMA ─────────────────────────────────────────────────────────────────────

/// Gives the WGMMA descriptor of a tile in shared memory.
///
///   tile_ptr            the tile's first value to read, in shared memory
///   leading_cells_val   the leading offset, in 16-byte cells; not given: 1 (not used)
__device__ __forceinline__ uint64_t mlp_up_8bit_descriptor(const void* tile_ptr, uint32_t leading_cells_val) {
    // Gives the WGMMA descriptor of a tile in shared memory: where it starts and how its rows sit (rows of 128 bytes
    // in TMA's 128-byte swizzle, 8-row atoms HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM apart).
    //     K-major (rows of k): the stride offset is from 8 lines to the next 8; moving 16 values of k along a row is
    //                          32 bytes more on the start address (the swizzle is undone from the address); the leading
    //                          offset is not used
    //     MN-major (turned, rows of lines): the stride offset is from 8 values of k to the next 8; moving 16 values
    //                          of k is 16 rows (2 atoms) more on the start address; the leading offset is from 64 lines
    //                          to the next 64 (a tile of more than 64 lines is boxes side by side)
    uint64_t address_val = (uint64_t)__cvta_generic_to_shared(tile_ptr);
    uint64_t descriptor_val = ((address_val & HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS) / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT;
    descriptor_val |= (uint64_t)leading_cells_val << HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT;
    descriptor_val |= (uint64_t)(HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM / HARDWARE_WGMMA_DESCRIPTOR_UNIT) << HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT;
    descriptor_val |= HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 << HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT;
    return descriptor_val;
}

/// Adds left tile · right tileᵀ into sums {64, N} on the E4M3 tensor cores, for the whole warpgroup.
///
///   sums_list                       this thread's sums, in WGMMA's layout: GROUPS groups of 4
///   left_tile_ptr, right_tile_ptr   the 32 values (bytes) of k to use, in the shared memory tiles
///   add_onto_sums_val               1: sums += the product; 0: sums = the product (what sums held is not read)
template <int GROUPS>
__device__ __forceinline__ void mlp_up_8bit_multiply_add(float (&sums_list)[GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP],
                                                              const uint8_t* left_tile_ptr, const uint8_t* right_tile_ptr, int add_onto_sums_val) {
    // sums {64, N} += left tile {64, 32} · right tile {N, 32}ᵀ on the E4M3 tensor cores, for the whole warpgroup, N =
    // GROUPS x 8 (64, 128 or 256): wgmma.mma_async m64nNk32, E4M3 in, float sums, both tiles K-major (FP8 takes no
    // transpose), read straight from shared memory; asynchronous: done only after wgmma.wait_group. The instruction
    // and its 4 x GROUPS sum registers are written out for each N (generated, not typed).
    static_assert(GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 64 || GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 128 ||
                      GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 256,
                  "N is 64, 128 or 256: the instructions written out below");
    uint64_t left_descriptor_val = mlp_up_8bit_descriptor(left_tile_ptr);
    uint64_t right_descriptor_val = mlp_up_8bit_descriptor(right_tile_ptr);
    // scale-d (the predicate p, from add_onto_sums_val): 1 adds onto the sums, 0 writes them over; scale-a, scale-b 1
    if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 64) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
            "%32, %33, p, 1, 1;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]), "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]), "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]), "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(add_onto_sums_val));
    } else if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 128) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
            "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
            "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
            "%64, %65, p, 1, 1;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]), "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]), "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]), "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3]), "+f"(sums_list[8][0]), "+f"(sums_list[8][1]), "+f"(sums_list[8][2]), "+f"(sums_list[8][3]), "+f"(sums_list[9][0]), "+f"(sums_list[9][1]), "+f"(sums_list[9][2]), "+f"(sums_list[9][3]), "+f"(sums_list[10][0]), "+f"(sums_list[10][1]), "+f"(sums_list[10][2]), "+f"(sums_list[10][3]), "+f"(sums_list[11][0]), "+f"(sums_list[11][1]), "+f"(sums_list[11][2]), "+f"(sums_list[11][3]), "+f"(sums_list[12][0]), "+f"(sums_list[12][1]), "+f"(sums_list[12][2]), "+f"(sums_list[12][3]), "+f"(sums_list[13][0]), "+f"(sums_list[13][1]), "+f"(sums_list[13][2]), "+f"(sums_list[13][3]), "+f"(sums_list[14][0]), "+f"(sums_list[14][1]), "+f"(sums_list[14][2]), "+f"(sums_list[14][3]), "+f"(sums_list[15][0]), "+f"(sums_list[15][1]), "+f"(sums_list[15][2]), "+f"(sums_list[15][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(add_onto_sums_val));
    } else if constexpr (GROUPS * HARDWARE_WGMMA_GROUP_COLUMNS == 256) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %130, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
            "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
            "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
            "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
            "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
            "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
            "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
            "%128, %129, p, 1, 1;\n"
            "}\n"
            : "+f"(sums_list[0][0]), "+f"(sums_list[0][1]), "+f"(sums_list[0][2]), "+f"(sums_list[0][3]), "+f"(sums_list[1][0]), "+f"(sums_list[1][1]), "+f"(sums_list[1][2]), "+f"(sums_list[1][3]), "+f"(sums_list[2][0]), "+f"(sums_list[2][1]), "+f"(sums_list[2][2]), "+f"(sums_list[2][3]), "+f"(sums_list[3][0]), "+f"(sums_list[3][1]), "+f"(sums_list[3][2]), "+f"(sums_list[3][3]), "+f"(sums_list[4][0]), "+f"(sums_list[4][1]), "+f"(sums_list[4][2]), "+f"(sums_list[4][3]), "+f"(sums_list[5][0]), "+f"(sums_list[5][1]), "+f"(sums_list[5][2]), "+f"(sums_list[5][3]), "+f"(sums_list[6][0]), "+f"(sums_list[6][1]), "+f"(sums_list[6][2]), "+f"(sums_list[6][3]), "+f"(sums_list[7][0]), "+f"(sums_list[7][1]), "+f"(sums_list[7][2]), "+f"(sums_list[7][3]), "+f"(sums_list[8][0]), "+f"(sums_list[8][1]), "+f"(sums_list[8][2]), "+f"(sums_list[8][3]), "+f"(sums_list[9][0]), "+f"(sums_list[9][1]), "+f"(sums_list[9][2]), "+f"(sums_list[9][3]), "+f"(sums_list[10][0]), "+f"(sums_list[10][1]), "+f"(sums_list[10][2]), "+f"(sums_list[10][3]), "+f"(sums_list[11][0]), "+f"(sums_list[11][1]), "+f"(sums_list[11][2]), "+f"(sums_list[11][3]), "+f"(sums_list[12][0]), "+f"(sums_list[12][1]), "+f"(sums_list[12][2]), "+f"(sums_list[12][3]), "+f"(sums_list[13][0]), "+f"(sums_list[13][1]), "+f"(sums_list[13][2]), "+f"(sums_list[13][3]), "+f"(sums_list[14][0]), "+f"(sums_list[14][1]), "+f"(sums_list[14][2]), "+f"(sums_list[14][3]), "+f"(sums_list[15][0]), "+f"(sums_list[15][1]), "+f"(sums_list[15][2]), "+f"(sums_list[15][3]), "+f"(sums_list[16][0]), "+f"(sums_list[16][1]), "+f"(sums_list[16][2]), "+f"(sums_list[16][3]), "+f"(sums_list[17][0]), "+f"(sums_list[17][1]), "+f"(sums_list[17][2]), "+f"(sums_list[17][3]), "+f"(sums_list[18][0]), "+f"(sums_list[18][1]), "+f"(sums_list[18][2]), "+f"(sums_list[18][3]), "+f"(sums_list[19][0]), "+f"(sums_list[19][1]), "+f"(sums_list[19][2]), "+f"(sums_list[19][3]), "+f"(sums_list[20][0]), "+f"(sums_list[20][1]), "+f"(sums_list[20][2]), "+f"(sums_list[20][3]), "+f"(sums_list[21][0]), "+f"(sums_list[21][1]), "+f"(sums_list[21][2]), "+f"(sums_list[21][3]), "+f"(sums_list[22][0]), "+f"(sums_list[22][1]), "+f"(sums_list[22][2]), "+f"(sums_list[22][3]), "+f"(sums_list[23][0]), "+f"(sums_list[23][1]), "+f"(sums_list[23][2]), "+f"(sums_list[23][3]), "+f"(sums_list[24][0]), "+f"(sums_list[24][1]), "+f"(sums_list[24][2]), "+f"(sums_list[24][3]), "+f"(sums_list[25][0]), "+f"(sums_list[25][1]), "+f"(sums_list[25][2]), "+f"(sums_list[25][3]), "+f"(sums_list[26][0]), "+f"(sums_list[26][1]), "+f"(sums_list[26][2]), "+f"(sums_list[26][3]), "+f"(sums_list[27][0]), "+f"(sums_list[27][1]), "+f"(sums_list[27][2]), "+f"(sums_list[27][3]), "+f"(sums_list[28][0]), "+f"(sums_list[28][1]), "+f"(sums_list[28][2]), "+f"(sums_list[28][3]), "+f"(sums_list[29][0]), "+f"(sums_list[29][1]), "+f"(sums_list[29][2]), "+f"(sums_list[29][3]), "+f"(sums_list[30][0]), "+f"(sums_list[30][1]), "+f"(sums_list[30][2]), "+f"(sums_list[30][3]), "+f"(sums_list[31][0]), "+f"(sums_list[31][1]), "+f"(sums_list[31][2]), "+f"(sums_list[31][3])
            : "l"(left_descriptor_val), "l"(right_descriptor_val), "r"(add_onto_sums_val));
    }
}

// ── host ──────────────────────────────────────────────────────────────────────

/// Gives TMA's description of a matrix's tiles.
///
///   matrix_ptr                            its first value, on a 16-byte piece
///   matrix_rows_val, matrix_columns_val   its shape as it sits; matrix_columns a multiple of 8 (rows of whole 16-byte pieces)
///   box_rows_val                          rows of a tile: HARDWARE_WGMMA_M or a product's COLUMNS (lines), or
///                                         turned MLP_UP_8BIT_BLOCK_K (values of k)
inline CUtensorMap mlp_up_8bit_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                                unsigned box_rows_val) {
    // A bfloat16 matrix {matrix_rows, matrix_columns} sitting row after row, read in boxes of {box_rows,
    // MLP_UP_8BIT_ROW_VALUES} written into shared memory in the 128-byte swizzle; past the edges, 0s. Made by the
    // driver's cuTensorMapEncodeTiled, found at run time, so the program needs no -lcuda.
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function_ptr, 12000, cudaEnableDefault, &status_val));
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
    cuuint32_t box_list[2] = {MLP_UP_8BIT_ROW_VALUES, box_rows_val};
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

/// Gives TMA's description of an E4M3 copy's tiles.
///
///   matrix_ptr                                  its first byte, on a 16-byte piece
///   matrix_rows_val, matrix_bytes_per_row_val   its shape; bytes per row (= values: one E4M3 each) a multiple of 16
///   box_rows_val                                rows of a tile: HARDWARE_WGMMA_M or the product's COLUMNS
inline CUtensorMap mlp_up_8bit_tensor_map_8bit(const void* matrix_ptr, size_t matrix_rows_val, size_t matrix_bytes_per_row_val, unsigned box_rows_val) {
    // {matrix_rows, matrix_bytes_per_row} bytes sitting row after row, read in boxes of {box_rows,
    // MLP_UP_8BIT_BLOCK_K} written into shared memory in the 128-byte swizzle; past the edges, 0s. Made by the
    // driver's cuTensorMapEncodeTiled, found at run time.
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function_ptr, 12000, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: mlp_up_8bit could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: bytes of a row (k), then rows
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {matrix_bytes_per_row_val, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {matrix_bytes_per_row_val};                         // from one row to the next
    cuuint32_t box_list[2] = {MLP_UP_8BIT_BLOCK_K, box_rows_val};
    cuuint32_t element_steps_list[2] = {1, 1};                                         // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(matrix_ptr), sizes_list,
                                     row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %zu}-byte matrix at %p\n", (int)result_val,
                matrix_rows_val, matrix_bytes_per_row_val, matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Gives TMA's description of gelu_output's turned E4M3 copy for the stores.
///
///   matrix_ptr                         its first byte, on a 16-byte piece
///   lines_val, bytes_per_line_val      its shape: gelu_output's columns, and its rows (a multiple of 16)
///   box_lines_val                      lines of a box: the product's COLUMNS
inline CUtensorMap mlp_up_8bit_tensor_map_turned_8bit(const void* matrix_ptr, size_t lines_val, size_t bytes_per_line_val, unsigned box_lines_val) {
    // {lines, bytes_per_line} bytes sitting line after line (a line: one column of gelu_output, its rows' values),
    // written from shared memory in boxes of {box_lines, HARDWARE_WGMMA_M bytes} (a warpgroup's 64 rows of COLUMNS
    // columns, turned), no swizzle; past the edges, nothing written. Made by the driver's cuTensorMapEncodeTiled,
    // found at run time.
    // 1. the driver's function, found once
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function_ptr, 12000, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: mlp_up_8bit could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    // 2. the matrix and its boxes, the fastest-moving dimension first: bytes of a line (rows), then lines
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {bytes_per_line_val, lines_val};
    cuuint64_t row_bytes_list[1] = {bytes_per_line_val};                                // from one line to the next
    cuuint32_t box_list[2] = {HARDWARE_WGMMA_M, box_lines_val};
    cuuint32_t element_steps_list[2] = {1, 1};                                          // every value, no gaps
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(matrix_ptr), sizes_list,
                                     row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a turned {%zu, %zu}-byte matrix at %p\n", (int)result_val,
                lines_val, bytes_per_line_val, matrix_ptr);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

/// Stops with a message unless TMA can read or write a tensor.
///
///   caller_ptr, name_ptr   the caller's and the tensor's names, for the message
///   tensor                 the tensor
inline void mlp_up_8bit_check_tensor(const char* caller_ptr, const char* name_ptr, const Tensor& tensor) {
    // Bfloat16, its last dim a multiple of MLP_UP_8BIT_VALUE_MULTIPLE (rows of whole 16-byte cells), starting on a
    // 16-byte cell.
    size_t last_val = tensor.dim() >= 1 ? tensor.size(tensor.dim() - 1) : 0;
    if (tensor.dtype() != DType::BFloat16 || last_val == 0 || last_val % MLP_UP_8BIT_VALUE_MULTIPLE != 0 ||
        (size_t)tensor.data_ptr() % HARDWARE_TMA_BYTES_PER_CELL != 0) {
        fprintf(stderr, "error: %s needs %s bfloat16, its last dim a multiple of %d, starting on %d bytes\n", caller_ptr, name_ptr,
                MLP_UP_8BIT_VALUE_MULTIPLE, HARDWARE_TMA_BYTES_PER_CELL);
        exit(EXIT_FAILURE);
    }
}

/// The 8-bit copies of the product's two sides, each E4M3 with one float scale for the whole side.
struct MlpUp8bitCopies {
    // The 8-bit copies of the product's two sides: left {rows, k} and right {columns, k}, each E4M3
    // with one float scale for the whole side, as round_8bit_per_tensor writes them and the product reads them. Made once for a size, used
    // by every call.
    Tensor left_8bit_tensor;             // {rows, k} E4M3
    Tensor left_scale_tensor;            // {1} float: every value's scale
    Tensor right_8bit_tensor;            // {columns, k} E4M3
    Tensor right_scale_tensor;           // {1} float: every value's scale

    MlpUp8bitCopies() = default;
    /// Makes the copies, all 0, for output {rows, columns} with a sum of k (a multiple of ROUND_8BIT_ROW_LENGTH_MULTIPLE).
    MlpUp8bitCopies(size_t rows_val, size_t columns_val, size_t k_val)
        : left_8bit_tensor(Tensor::zeros({rows_val, k_val}, DType::E4M3)),
          left_scale_tensor(Tensor::zeros({1})),
          right_8bit_tensor(Tensor::zeros({columns_val, k_val}, DType::E4M3)),
          right_scale_tensor(Tensor::zeros({1})) {}
};

/// gelu_output in 8 bits: both copies mlp_down reads, and their one scale.
struct MlpUp8bitGelu8bit {
    // gelu_output in 8 bits, as mlp_up_8bit_gelu_8bit writes it: both copies mlp_down reads, their one scale (a float in GPU
    // memory, written here), and where that scale comes from. rows_8bit is mlp_down's left side (its forward), turned_8bit
    // its weight_grad's right; scale both's.
    Tensor rows_8bit_tensor;                 // {rows, columns} E4M3: written
    Tensor scale_tensor;                     // {1} float: written, the one scale of both copies
    Tensor turned_8bit_tensor;               // {columns, rows} E4M3: written
    const unsigned* scale_largest_ptr = nullptr;   // the largest |value| the scale is made from (a float's bits, in GPU memory)
    float margin_val = 1.0f;                       // ... times this (ROUND_8BIT_DELAYED_MARGIN for a delayed one)
    unsigned* next_largest_ptr = nullptr;          // this tensor's largest |value| maxed onto it (0 before), or nullptr
};

// ── the product ───────────────────────────────────────────────────────────────
void mlp_up_8bit(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val);
void mlp_up_8bit_rounded(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val);
void mlp_up_8bit_gelu_8bit(const Tensor& gelu_input_tensor, const MlpUp8bitGelu8bit& gelu_val, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val);
void mlp_up_8bit_round(const MlpUp8bitCopies& copies_val, const Tensor& input_tensor, const Tensor& weight_tensor);
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_8BIT_PRODUCER_REGISTERS, int IN_FLIGHT = MLP_UP_8BIT_IN_FLIGHT,
          int PROMOTION_STEPS = MLP_UP_8BIT_PROMOTION_STEPS, int PINGPONG = MLP_UP_8BIT_PINGPONG
          >
void mlp_up_8bit_rounded_at(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val,
                    int block_rows_per_group_val, int blocks_val);
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_8BIT_PRODUCER_REGISTERS, int IN_FLIGHT = MLP_UP_8BIT_IN_FLIGHT,
          int PROMOTION_STEPS = MLP_UP_8BIT_PROMOTION_STEPS, int PINGPONG = MLP_UP_8BIT_PINGPONG>
void mlp_up_8bit_gelu_8bit_at(const Tensor& gelu_input_tensor, const MlpUp8bitGelu8bit& gelu_val, const Tensor& bias_tensor,
                              const MlpUp8bitCopies& copies_val, int block_rows_per_group_val, int blocks_val);

// ── for it ────────────────────────────────────────────────────────────────────
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS, int PINGPONG, bool GELU_8BIT>
void mlp_up_8bit_launch(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const MlpUp8bitGelu8bit* gelu_8bit_ptr,
                        const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val, int block_rows_per_group_val, int blocks_val);
template <int COLUMNS, int STAGES, int ROW_GROUPS, int PINGPONG = 0
          >
constexpr size_t mlp_up_8bit_shared_bytes();
template <int STAGES, int ROW_GROUPS>
constexpr size_t mlp_up_8bit_barrier_bytes();
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS,
          int STAGED_ROWS, int STAGED_ROW_PADDING, int PINGPONG, bool GELU_8BIT>
int mlp_up_8bit_blocks_at_once();
template <int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER, bool SPLIT = false, int PRODUCER_REGISTERS = MLP_UP_8BIT_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_UP_8BIT_IN_FLIGHT, int PROMOTION_STEPS = MLP_UP_8BIT_PROMOTION_STEPS, int STAGED_ROWS = 0, int STAGED_ROW_PADDING = 0,
          int PINGPONG = 0, bool GELU_8BIT = false>
__global__ void mlp_up_8bit_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
                           const __grid_constant__ CUtensorMap output_map_val,
                           const __grid_constant__ CUtensorMap side_map_val,
                           const __grid_constant__ CUtensorMap side_turned_map_val,
                           const float* left_scale_ptr, const float* right_scale_ptr,
                           const nv_bfloat16* bias_ptr,
                           int rows_val, int columns_val, int sum_length_val, int block_rows_per_group_val,
                           float* side_scale_ptr, const unsigned* side_scale_largest_ptr,
                           float side_margin_val, unsigned* side_next_largest_ptr);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── the product ───────────────────────────────────────────────────────────────

/// Writes gelu_input = input · weightᵀ + bias and gelu_output = gelu(gelu_input), the product in 8 bits, at this file's setting.
///
///   gelu_input_tensor    {..., out_features} bfloat16: written; kept for the backward
///   gelu_output_tensor   gelu_input's shape, bfloat16: written
///   input_tensor         {..., in_features} bfloat16
///   weight_tensor        {out_features, in_features} bfloat16
///   bias_tensor          {out_features} bfloat16, or Tensor() for none (nothing added)
///   copies_val           MlpUp8bitCopies(rows, out_features, in_features): written
inline void mlp_up_8bit(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val) {
    // Rounds the two sides into the copies (mlp_up_8bit_round), then mlp_up_8bit_rounded_at with the defines
    // above.
    mlp_up_8bit_round(copies_val, input_tensor, weight_tensor);
    mlp_up_8bit_rounded_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER>(gelu_input_tensor, gelu_output_tensor, bias_tensor, copies_val,
                                                                         MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// Works out the product at this file's setting (the defines above) on copies already made.
inline void mlp_up_8bit_rounded(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val) {
    // By mlp_up_8bit_round, or shared with another product that rounds the same tensor (a linear layer's
    // output_grad for its input_grad and its weight_grad, round_8bit_rows_and_columns), so the tensor is rounded
    // once.
    //
    //   the tensors   as in mlp_up_8bit, but the two it rounds: their copies are copies_val's
    mlp_up_8bit_rounded_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER>(gelu_input_tensor, gelu_output_tensor, bias_tensor, copies_val,
                                                                         MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// Works out the product on copies already made, writing gelu_output straight in 8 bits.
///
///   gelu_input_tensor   {..., out_features} bfloat16: written; kept for the backward (gelu')
///   gelu_val            gelu_output's copies, their scale's largest and margin, and where this tensor's largest goes
///   bias_tensor         {out_features} bfloat16, or Tensor() for none
///   copies_val          rounded (mlp_up_8bit_round, or the model's forward copies)
inline void mlp_up_8bit_gelu_8bit(const Tensor& gelu_input_tensor, const MlpUp8bitGelu8bit& gelu_val, const Tensor& bias_tensor,
                                  const MlpUp8bitCopies& copies_val) {
    // Works out the product at this file's setting on copies already made, as mlp_up_8bit_rounded, but writes gelu_output
    // straight in 8 bits (both copies, one scale for the tensor: MlpUp8bitGelu8bit) in place of bfloat16: what mlp_down
    // reads, with no rounding pass of its own.
    mlp_up_8bit_gelu_8bit_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER>(
        gelu_input_tensor, gelu_val, bias_tensor, copies_val, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// Rounds the product's two sides to E4M3 into the copies, one power-of-2 scale for each side (round_8bit_per_tensor): input {rows, in} and weight {out, in}, both as they sit.
///
///   copies_val           MlpUp8bitCopies: written
///   input_tensor         {..., in_features} bfloat16
///   weight_tensor        {out_features, in_features} bfloat16
inline void mlp_up_8bit_round(const MlpUp8bitCopies& copies_val, const Tensor& input_tensor, const Tensor& weight_tensor) {
    // 1. the sizes: each side's rows of k; the copies made for them
    size_t k_val = input_tensor.size(input_tensor.dim() - 1), rows_val = input_tensor.numel() / k_val;
    size_t columns_val = weight_tensor.dim() == 2 && weight_tensor.size(1) == k_val ? weight_tensor.size(0) : 0;
    if (rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 || columns_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 ||
        k_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE != 0 ||
        copies_val.left_8bit_tensor.shape() != Shape({rows_val, k_val}) ||
        copies_val.right_8bit_tensor.shape() != Shape({columns_val, k_val}) ||
        copies_val.left_scale_tensor.numel() != 1 || copies_val.right_scale_tensor.numel() != 1) {
        fprintf(stderr, "error: mlp_up_8bit needs rows, columns and k multiples of %d and its copies made with MlpUp8bitCopies(rows, columns, k): "
                        "got %zu, %zu, %zu\n", ROUND_8BIT_ROW_LENGTH_MULTIPLE, rows_val, columns_val, k_val);
        exit(EXIT_FAILURE);
    }
    // 2. the copies
    round_8bit_per_tensor(copies_val.left_8bit_tensor, Tensor(), copies_val.left_scale_tensor, input_tensor);
    round_8bit_per_tensor(copies_val.right_8bit_tensor, Tensor(), copies_val.right_scale_tensor, weight_tensor);
}

/// Works out the product from the copies at a setting given, gelu_output bfloat16.
///
///   gelu_input_tensor    {..., out_features} bfloat16: written; kept for the backward
///   gelu_output_tensor   gelu_input's shape, bfloat16: written
///   bias_tensor          {out_features} bfloat16, or Tensor() for none (nothing added)
///   copies_val                     rounded (mlp_up_8bit_round)
///   block_rows_per_group_val       the squares' order: rows of squares a group runs down (at least 1)
///   blocks_val                     blocks launched: 0, as many as the GPU runs at once; more than the squares: one a square
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS, int PINGPONG
          >
inline void mlp_up_8bit_rounded_at(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val,
                           int block_rows_per_group_val, int blocks_val) {
    // mlp_up_8bit_launch made for the setting.
    //
    // The template parameters: the setting (see the top of the file); not given: the defines.
    mlp_up_8bit_check_tensor("mlp_up_8bit", "gelu_output", gelu_output_tensor);
    mlp_up_8bit_launch<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, PINGPONG, false>(
        gelu_input_tensor, gelu_output_tensor, nullptr, bias_tensor, copies_val, block_rows_per_group_val, blocks_val);
}

/// Works out the product from the copies at a setting given, gelu_output in 8 bits.
///
///   gelu_input_tensor, bias_tensor, copies_val, block_rows_per_group_val, blocks_val   as in mlp_up_8bit_rounded_at
///   gelu_val                                                                           as in mlp_up_8bit_gelu_8bit
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS, int PINGPONG>
inline void mlp_up_8bit_gelu_8bit_at(const Tensor& gelu_input_tensor, const MlpUp8bitGelu8bit& gelu_val, const Tensor& bias_tensor,
                                     const MlpUp8bitCopies& copies_val, int block_rows_per_group_val, int blocks_val) {
    // (MlpUp8bitGelu8bit): mlp_up_8bit_launch made for the setting. COLUMNS 128 or 256: a row copy's TMA box is a
    // swizzled row of 128 bytes.
    static_assert(COLUMNS % MLP_UP_8BIT_BLOCK_K == 0, "gelu_output in 8 bits: COLUMNS 128 or 256, whole swizzled rows of 128 E4M3s");
    // 1. the copies: E4M3, their shapes from the product's; the scale's largest given
    size_t rows_val = copies_val.left_8bit_tensor.size(0), columns_val = copies_val.right_8bit_tensor.size(0);
    if (!(gelu_val.rows_8bit_tensor.dtype() == DType::E4M3 && gelu_val.turned_8bit_tensor.dtype() == DType::E4M3 &&
          gelu_val.scale_tensor.dtype() == DType::Float32 && gelu_val.scale_tensor.numel() == 1 &&
          gelu_val.rows_8bit_tensor.numel() == rows_val * columns_val &&
          gelu_val.rows_8bit_tensor.size(gelu_val.rows_8bit_tensor.dim() - 1) == columns_val &&
          gelu_val.turned_8bit_tensor.shape() == Shape({columns_val, rows_val}) &&
          rows_val % ROUND_8BIT_ROW_LENGTH_MULTIPLE == 0 && gelu_val.scale_largest_ptr != nullptr)) {
        fprintf(stderr, "error: mlp_up_8bit_gelu_8bit needs E4M3 copies {rows, columns} and {columns, rows}, a float scale {1}, "
                        "rows a multiple of %d, and the scale's largest\n",
                ROUND_8BIT_ROW_LENGTH_MULTIPLE);
        exit(EXIT_FAILURE);
    }
    // 2. the launch
    mlp_up_8bit_launch<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, PINGPONG, true>(
        gelu_input_tensor, Tensor(), &gelu_val, bias_tensor, copies_val, block_rows_per_group_val, blocks_val);
}

/// Checks, describes the tiles for TMA, then launches mlp_up_8bit_kernel made for the setting.
///
///   gelu_input_tensor     {..., out_features} bfloat16: written
///   gelu_output_tensor    bfloat16 gelu_output (GELU_8BIT false): written; Tensor() with GELU_8BIT
///   gelu_8bit_ptr         gelu_output in 8 bits (GELU_8BIT true); nullptr otherwise
///   bias_tensor, copies_val, block_rows_per_group_val, blocks_val   as in mlp_up_8bit_rounded_at
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS, int PINGPONG, bool GELU_8BIT>
inline void mlp_up_8bit_launch(const Tensor& gelu_input_tensor, const Tensor& gelu_output_tensor, const MlpUp8bitGelu8bit* gelu_8bit_ptr,
                               const Tensor& bias_tensor, const MlpUp8bitCopies& copies_val, int block_rows_per_group_val, int blocks_val) {
    // What mlp_up_8bit_rounded_at and mlp_up_8bit_gelu_8bit_at share.
    constexpr int STAGED_ROWS = 0, STAGED_ROW_PADDING = 0;    // no split store
    static_assert(COLUMNS == 64 || COLUMNS == 128 || (COLUMNS == 256 && PROMOTION_STEPS == 0),
                  "COLUMNS is 64 or 128, or 256 with PROMOTION_STEPS 0: the promotion's second set of sums leaves no room for 256");
    static_assert(STAGES >= 1 && STAGES <= MLP_UP_8BIT_MAX_STAGES, "STAGES is 1 to MLP_UP_8BIT_MAX_STAGES");
    static_assert(ROW_GROUPS >= 1 && ROW_GROUPS <= MLP_UP_8BIT_MAX_ROW_GROUPS, "ROW_GROUPS is 1 to MLP_UP_8BIT_MAX_ROW_GROUPS");
    static_assert(CLUSTER >= 1 && CLUSTER <= MLP_UP_8BIT_MAX_CLUSTER, "CLUSTER is 1 to MLP_UP_8BIT_MAX_CLUSTER");
    static_assert(PROMOTION_STEPS >= 0, "PROMOTION_STEPS is 0 (never) or more");
    static_assert(PINGPONG == 0 || (PINGPONG == 1 && ROW_GROUPS == 2 && CLUSTER == 1), "PINGPONG 1 takes 2 consumer warpgroups and no clusters");
    // 1. the sizes, from the copies; gelu_input: bfloat16, rows of whole 16-byte cells, the shape
    size_t rows_val = copies_val.left_8bit_tensor.size(0), columns_val = copies_val.right_8bit_tensor.size(0);
    size_t sum_length_val = copies_val.left_8bit_tensor.size(1);
    mlp_up_8bit_check_tensor("mlp_up_8bit", "gelu_input", gelu_input_tensor);
    if (!(gelu_input_tensor.numel() == rows_val * columns_val && gelu_input_tensor.size(gelu_input_tensor.dim() - 1) == columns_val &&
          (GELU_8BIT || (gelu_output_tensor.numel() == rows_val * columns_val &&
                         gelu_output_tensor.size(gelu_output_tensor.dim() - 1) == columns_val)) &&
          (bias_tensor.numel() == 0 || (bias_tensor.dtype() == DType::BFloat16 && bias_tensor.shape() == Shape({columns_val}))))) {
        fprintf(stderr, "error: mlp_up_8bit needs its outputs {..., columns} of the copies' rows and columns, bias {columns} or none\n");
        exit(EXIT_FAILURE);
    }
    if (block_rows_per_group_val < 1 || blocks_val < 0) {
        fprintf(stderr, "error: mlp_up_8bit's block_rows_per_group must be 1 or more and the blocks 0 or more\n");
        exit(EXIT_FAILURE);
    }

    // 2. TMA's descriptions: the copies in boxes of {64 rows or COLUMNS lines, MLP_UP_8BIT_BLOCK_K bytes}; the output in
    //    bfloat16 boxes of {64 rows, 64 columns}; gelu_output as bfloat16 the same, or in 8 bits: its row copy in boxes of
    //    {64 rows, 128 bytes}, its turned copy in boxes of {COLUMNS lines, 64 bytes}
    CUtensorMap left_map_val = mlp_up_8bit_tensor_map_8bit(copies_val.left_8bit_tensor.data_ptr(), rows_val, sum_length_val, HARDWARE_WGMMA_M);
    CUtensorMap right_map_val = mlp_up_8bit_tensor_map_8bit(copies_val.right_8bit_tensor.data_ptr(), columns_val, sum_length_val, COLUMNS);
    CUtensorMap output_map_val = mlp_up_8bit_tensor_map(gelu_input_tensor.data_ptr<nv_bfloat16>(), rows_val, columns_val, HARDWARE_WGMMA_M);
    CUtensorMap side_map_val = output_map_val, side_turned_map_val = output_map_val;   // not read where not used
    float* side_scale_ptr = nullptr;
    const unsigned* side_scale_largest_ptr = nullptr;
    float side_margin_val = 1.0f;
    unsigned* side_next_largest_ptr = nullptr;
    if constexpr (GELU_8BIT) {
        side_map_val = mlp_up_8bit_tensor_map_8bit(gelu_8bit_ptr->rows_8bit_tensor.data_ptr(), rows_val, columns_val, HARDWARE_WGMMA_M);
        side_turned_map_val = mlp_up_8bit_tensor_map_turned_8bit(gelu_8bit_ptr->turned_8bit_tensor.data_ptr(), columns_val, rows_val, COLUMNS);
        side_scale_ptr = gelu_8bit_ptr->scale_tensor.data();
        side_scale_largest_ptr = gelu_8bit_ptr->scale_largest_ptr;
        side_margin_val = gelu_8bit_ptr->margin_val;
        side_next_largest_ptr = gelu_8bit_ptr->next_largest_ptr;
    } else {
        side_map_val = mlp_up_8bit_tensor_map(gelu_output_tensor.data_ptr<nv_bfloat16>(), rows_val, columns_val, HARDWARE_WGMMA_M);
    }

    // 3. the launch: as many blocks as the GPU runs at once (or blocks_val), in whole clusters, at most one a unit (a unit:
    //    a cluster's squares; ping-pong, one warpgroup's square of 64 rows)
    constexpr bool SPLIT = false;
    constexpr size_t SHARED_BYTES = mlp_up_8bit_shared_bytes<COLUMNS, STAGES, ROW_GROUPS, PINGPONG>();
    constexpr int THREADS = (MLP_UP_8BIT_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS;
    static_assert(SHARED_BYTES + mlp_up_8bit_barrier_bytes<STAGES, ROW_GROUPS>() <= HARDWARE_MAX_SHARED_BYTES_PER_BLOCK,
                  "the sets of tiles, the rooms and the barriers fit a block's shared memory");
    auto kernel_ptr = mlp_up_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, STAGED_ROWS,
                                         STAGED_ROW_PADDING, PINGPONG, GELU_8BIT>;
    int blocks_at_once_val = mlp_up_8bit_blocks_at_once<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS,
                                                        STAGED_ROWS, STAGED_ROW_PADDING, PINGPONG, GELU_8BIT>();
    size_t unit_rows_val = PINGPONG ? (size_t)HARDWARE_WGMMA_M : (size_t)(CLUSTER * ROW_GROUPS * HARDWARE_WGMMA_M);
    size_t total_units_val = CEIL_DIV(columns_val, (size_t)COLUMNS) * CEIL_DIV(rows_val, unit_rows_val);
    size_t total_clusters_val = (blocks_val > 0 ? (size_t)blocks_val : (size_t)blocks_at_once_val) / CLUSTER;
    if (total_clusters_val < 1) total_clusters_val = 1;
    if (total_clusters_val > total_units_val) total_clusters_val = total_units_val;
    unsigned total_blocks_val = (unsigned)(total_clusters_val * CLUSTER);
    const float* left_scale_ptr = copies_val.left_scale_tensor.data();
    const float* right_scale_ptr = copies_val.right_scale_tensor.data();
    const nv_bfloat16* bias_ptr = bias_tensor.numel() > 0 ? bias_tensor.data_ptr<nv_bfloat16>() : nullptr;   // nullptr: no bias
    if (CLUSTER == 1) {
        kernel_ptr<<<total_blocks_val, THREADS, SHARED_BYTES>>>(left_map_val, right_map_val, output_map_val, side_map_val, side_turned_map_val,
                                                                left_scale_ptr, right_scale_ptr, bias_ptr,
                                                                (int)rows_val, (int)columns_val, (int)sum_length_val, block_rows_per_group_val,
                                                                side_scale_ptr, side_scale_largest_ptr, side_margin_val,
                                                                side_next_largest_ptr);
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
        cudaCheck(cudaLaunchKernelEx(&config_val, kernel_ptr, left_map_val, right_map_val, output_map_val, side_map_val, side_turned_map_val,
                                     left_scale_ptr, right_scale_ptr, bias_ptr,
                                     (int)rows_val, (int)columns_val, (int)sum_length_val, block_rows_per_group_val,
                                     side_scale_ptr, side_scale_largest_ptr, side_margin_val, side_next_largest_ptr));
    }
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Gives the dynamic shared memory a block takes.
template <int COLUMNS, int STAGES, int ROW_GROUPS, int PINGPONG>
constexpr size_t mlp_up_8bit_shared_bytes() {
    // STAGES sets of tiles (a left tile a consumer warpgroup, or one with ping-pong, and the right tile), each
    // consumer warpgroup's room for its 64 rows as bfloat16 (TMA's store boxes; with gelu_output in 8 bits, its
    // row copy in the room's first half and its turned copy in the second), and one atom of room to start the
    // tiles on one.
    return (size_t)STAGES * ((PINGPONG ? 1 : ROW_GROUPS) * MLP_UP_8BIT_LEFT_TILE_BYTES + COLUMNS * MLP_UP_8BIT_BLOCK_K) +
           (size_t)ROW_GROUPS * HARDWARE_WGMMA_M * COLUMNS * sizeof(nv_bfloat16) + HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
}

/// Gives the static shared memory a block takes besides: its barriers.
template <int STAGES, int ROW_GROUPS>
constexpr size_t mlp_up_8bit_barrier_bytes() {
    // 8 bytes each (two a set of tiles, two a consumer warpgroup: its store's and its ping-pong turn's). The
    // dynamic and this together must fit HARDWARE_MAX_SHARED_BYTES_PER_BLOCK.
    return (size_t)(2 * STAGES + 2 * ROW_GROUPS) * sizeof(uint64_t);
}

/// Gives how many of one kernel's blocks the GPU runs at once.
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS,
          int STAGED_ROWS, int STAGED_ROW_PADDING, int PINGPONG, bool GELU_8BIT>
inline int mlp_up_8bit_blocks_at_once() {
    // SMs x the blocks an SM holds. The first time for each kernel, it also asks CUDA to allow the kernel's
    // dynamic shared memory (past the 48 KB a block gets without asking).
    constexpr size_t SHARED_BYTES = mlp_up_8bit_shared_bytes<COLUMNS, STAGES, ROW_GROUPS, PINGPONG>();
    constexpr int THREADS = (MLP_UP_8BIT_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS;
    static int blocks_at_once_val = 0;                       // one for each kernel this template makes
    if (blocks_at_once_val == 0) {
        auto kernel_ptr = mlp_up_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, CLUSTER, SPLIT, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, STAGED_ROWS,
                                             STAGED_ROW_PADDING, PINGPONG, GELU_8BIT>;
        int device_index_val = 0, total_sms_val = 0, blocks_per_sm_val = 0;
        cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SHARED_BYTES));
        cudaCheck(cudaGetDevice(&device_index_val));
        cudaCheck(cudaDeviceGetAttribute(&total_sms_val, cudaDevAttrMultiProcessorCount, device_index_val));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_val, kernel_ptr, THREADS, SHARED_BYTES));
        blocks_at_once_val = total_sms_val * (blocks_per_sm_val > 0 ? blocks_per_sm_val : 1);
    }
    return blocks_at_once_val;
}

/// Gives gelu(x), GPT-2's tanh GELU, with tanh by tanh.approx.f32.
///
///   x_val   one value
__device__ __forceinline__ float mlp_up_8bit_gelu_value(float x_val) {
    // Gives gelu(x) = 0.5 x (1 + tanh(s)), s = sqrt(2 / pi) (x + 0.044715 x³): GPT-2's tanh GELU, as fp32/gelu.cuh's
    // gelu_value, but tanh by tanh.approx.f32 (mlp_up_8bit_tanh): within a bfloat16 step of it once rounded.
    float inner_val = MLP_UP_8BIT_GELU_SCALE * (x_val + MLP_UP_8BIT_GELU_CUBE_FACTOR * x_val * x_val * x_val);   // s
    return 0.5f * x_val * (1.0f + mlp_up_8bit_tanh(inner_val));
}

/// Gives gelu of a pair of the product (+ bias), rounded to bfloat16 first as gelu_input holds it.
///
///   pair_val   two neighbouring results, in float
__device__ __forceinline__ float2 mlp_up_8bit_gelu_of(float2 pair_val) {
    // What gelu_output gets.
    float2 x_val = __bfloat1622float2(__floats2bfloat162_rn(pair_val.x, pair_val.y));
    return make_float2(mlp_up_8bit_gelu_value(x_val.x), mlp_up_8bit_gelu_value(x_val.y));
}

/// Walks this block's squares of output, square after square (persistent), and stores each by TMA.
///
///   left_map_val, right_map_val          TMA's descriptions of the copies' tiles
///   output_map_val                       ... and gelu_input's
///   side_map_val                         ... and gelu_output's (bfloat16), or its row copy's (GELU_8BIT)
///   side_turned_map_val                  ... its turned copy's (GELU_8BIT; not read otherwise)
///   left_scale_ptr, right_scale_ptr    the copies' scales: one float each
///   bias_ptr                             {columns}, or nullptr for none
///   rows_val, columns_val, sum_length_val   the product's sizes
///   block_rows_per_group_val             the squares' order: rows of units a group runs down
///   side_scale_ptr                               GELU_8BIT: gelu_output's one scale {1}: written
///   side_scale_largest_ptr, side_margin_val       GELU_8BIT: the scale is round_8bit_scale(largest x margin)
///   side_next_largest_ptr                         GELU_8BIT: gelu_output's largest |value| maxed onto it, or nullptr
template <int COLUMNS, int STAGES, int ROW_GROUPS, int CLUSTER, bool SPLIT, int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS,
          int STAGED_ROWS, int STAGED_ROW_PADDING, int PINGPONG, bool GELU_8BIT>
__global__ void __launch_bounds__((MLP_UP_8BIT_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS)
mlp_up_8bit_kernel(const __grid_constant__ CUtensorMap left_map_val, const __grid_constant__ CUtensorMap right_map_val,
           const __grid_constant__ CUtensorMap output_map_val,
           const __grid_constant__ CUtensorMap side_map_val,
           const __grid_constant__ CUtensorMap side_turned_map_val,
           const float* left_scale_ptr, const float* right_scale_ptr,
           const nv_bfloat16* bias_ptr,
           int rows_val, int columns_val, int sum_length_val, int block_rows_per_group_val,
           float* side_scale_ptr, const unsigned* side_scale_largest_ptr,
           float side_margin_val, unsigned* side_next_largest_ptr) {
    // Walks this block's squares of output, square after square (persistent), with a producer warpgroup and ROW_GROUPS
    // consumer warpgroups (see the top of the file), cooperative or ping-pong (PINGPONG), and stores each square by TMA:
    // gelu_input, then gelu_output as bfloat16 or (GELU_8BIT) as its two E4M3 copies.
    constexpr int COLUMN_GROUPS = COLUMNS / HARDWARE_WGMMA_GROUP_COLUMNS;                                   // groups of 8 across the square
    constexpr int RIGHT_TILE_BYTES = COLUMNS * MLP_UP_8BIT_BLOCK_K;                                                 // a right tile: COLUMNS lines of a step's k
    constexpr int BLOCK_ROWS = ROW_GROUPS * HARDWARE_WGMMA_M;                                              // a square's rows (cooperative)
    constexpr int UNIT_ROWS = PINGPONG ? HARDWARE_WGMMA_M : CLUSTER * BLOCK_ROWS;                            // a unit: the cluster's squares one under the other; ping-pong, one warpgroup's square
    constexpr int SET_LEFT_TILES = PINGPONG ? 1 : ROW_GROUPS;                                                // a set's left tiles: one a warpgroup; ping-pong, the unit's one
    constexpr int SET_USERS = PINGPONG ? 1 : ROW_GROUPS * CLUSTER;                                           // consumer warpgroups that free a set
    constexpr int LEFT_BYTES = SET_LEFT_TILES * MLP_UP_8BIT_LEFT_TILE_BYTES;                                            // a set's left tiles
    constexpr int SET_BYTES = LEFT_BYTES + RIGHT_TILE_BYTES;                                                // one set: left tiles, then right tile
    constexpr int LEFT_RUNNING = STAGES > IN_FLIGHT ? IN_FLIGHT : STAGES - 1;                                // steps of WGMMAs left running: IN_FLIGHT, at most STAGES - 1
    static_assert(IN_FLIGHT >= 0, "IN_FLIGHT is 0 or more steps");
    constexpr int PROMOTION_PERIOD = PROMOTION_STEPS > 0 ? PROMOTION_STEPS : 1;                               // what the steps are counted in (never 0)
    constexpr int OUTPUT_BOXES = COLUMNS / MLP_UP_8BIT_ROW_VALUES;                                                  // bfloat16 TMA store boxes across a warpgroup's 64 rows
    constexpr int ROOM_BYTES = HARDWARE_WGMMA_M * COLUMNS * (int)sizeof(nv_bfloat16);                       // a consumer warpgroup's room: its 64 rows as bfloat16
    // gelu_output in 8 bits: the row copy's boxes (64 rows of a swizzled row of 128 E4M3s each) in the room's first half,
    // the turned copy (COLUMNS lines of 64 E4M3s, plain) in its second
    constexpr int ROW_8BIT_BOXES = COLUMNS / MLP_UP_8BIT_BLOCK_K;
    constexpr int ROW_8BIT_BOX_BYTES = HARDWARE_WGMMA_M * MLP_UP_8BIT_BLOCK_K;
    constexpr int TURNED_8BIT_OFFSET = HARDWARE_WGMMA_M * COLUMNS;
    static_assert(!GELU_8BIT || COLUMNS % MLP_UP_8BIT_BLOCK_K == 0, "gelu_output in 8 bits: COLUMNS 128 or 256");
    constexpr int PRODUCER_THREADS = MLP_UP_8BIT_PRODUCER_WARPGROUPS * HARDWARE_WARPGROUP_THREADS;
    // the registers (the worklog's kernel 5b): a block of THREADS gets LAUNCH_REGISTERS a thread; the producer needs
    // few, so with 2 consumer warpgroups it gives back all but PRODUCER_REGISTERS and the consumers take them
    constexpr int THREADS = PRODUCER_THREADS + ROW_GROUPS * HARDWARE_WARPGROUP_THREADS;
    constexpr int LAUNCH_REGISTERS = HARDWARE_REGISTERS_PER_SM / THREADS / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    static_assert(PRODUCER_REGISTERS >= HARDWARE_SETMAXNREG_MIN && PRODUCER_REGISTERS % HARDWARE_REGISTER_ALLOCATION_UNIT == 0,
                  "setmaxnreg takes at least HARDWARE_SETMAXNREG_MIN, in whole allocation units");
    constexpr int CONSUMER_REGISTERS_FIT = (LAUNCH_REGISTERS * THREADS - PRODUCER_REGISTERS * PRODUCER_THREADS) /
                                           (ROW_GROUPS * HARDWARE_WARPGROUP_THREADS) / HARDWARE_REGISTER_ALLOCATION_UNIT * HARDWARE_REGISTER_ALLOCATION_UNIT;
    constexpr int CONSUMER_REGISTERS = CONSUMER_REGISTERS_FIT < HARDWARE_SETMAXNREG_MAX ? CONSUMER_REGISTERS_FIT : HARDWARE_SETMAXNREG_MAX;
    constexpr bool MOVE_REGISTERS = ROW_GROUPS > 1;           // 1 consumer warpgroup: 255 a thread already, plenty
    // dynamic shared memory, from its next HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM atom on: the sets of tiles in TMA's
    // swizzled layout, then each consumer warpgroup's room. Static: the barriers
    extern __shared__ __align__(16) uint8_t mlp_up_8bit_shared_list[];
    __shared__ uint64_t full_barriers_list[STAGES];          // set s's bytes are in
    __shared__ uint64_t empty_barriers_list[STAGES];         // every consumer of the cluster that used set s is done with it
    __shared__ uint64_t store_barriers_list[ROW_GROUPS];     // consumer warpgroup g's threads are all at the same point of its store
    __shared__ uint64_t turn_barriers_list[ROW_GROUPS];      // ping-pong: warpgroup g's turn at the tensor cores (the other's sums done)
    uint32_t shared_address_val = (uint32_t)__cvta_generic_to_shared(mlp_up_8bit_shared_list);
    uint8_t* tiles_ptr = mlp_up_8bit_shared_list + (HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM - shared_address_val % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) %
                                           HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
    uint8_t* rooms_ptr = tiles_ptr + STAGES * SET_BYTES;

    // 1. the barriers, by one thread; with clusters, every block of the cluster waits until all have theirs
    if (threadIdx.x == 0) {
#pragma unroll
        for (int set_index_val = 0; set_index_val < STAGES; set_index_val++) {
            mlp_up_8bit_barrier_init(&full_barriers_list[set_index_val], 1);                       // the producer's one arrival, and the bytes
            mlp_up_8bit_barrier_init(&empty_barriers_list[set_index_val], SET_USERS);              // one arrival a consumer warpgroup that used it
        }
#pragma unroll
        for (int row_group_index_val = 0; row_group_index_val < ROW_GROUPS; row_group_index_val++) {
            mlp_up_8bit_barrier_init(&store_barriers_list[row_group_index_val], HARDWARE_WARPGROUP_THREADS);   // every thread of the warpgroup
            mlp_up_8bit_barrier_init(&turn_barriers_list[row_group_index_val], 1);                             // the other warpgroup's first thread
        }
        mlp_up_8bit_fence_for_tma();                                  // the barriers visible to TMA
    }
    __syncthreads();
    if (CLUSTER > 1) mlp_up_8bit_cluster_sync<true>();

    // 2. the units of work, and which are this block's. A unit: the cluster's UNIT_ROWS x COLUMNS of output; block b of
    //    the cluster takes its square b (ping-pong: a unit is one warpgroup's 64 rows). The units are numbered in groups
    //    of block_rows_per_group rows of units, down a group first. Cluster c takes units c, c + clusters, ...
    int cluster_block_index_val = blockIdx.x % CLUSTER;
    int cluster_index_val = blockIdx.x / CLUSTER;
    int total_clusters_val = gridDim.x / CLUSTER;
    int total_unit_columns_val = CEIL_DIV(columns_val, COLUMNS);
    int total_unit_rows_val = CEIL_DIV(rows_val, UNIT_ROWS);
    int total_units_val = total_unit_columns_val * total_unit_rows_val;
    int total_steps_val = CEIL_DIV(sum_length_val, MLP_UP_8BIT_BLOCK_K);
    int units_per_group_val = block_rows_per_group_val * total_unit_columns_val;
    // gives a unit's square for this block: its first row and column
    auto place_unit_val = [&](int unit_index_val, int& block_first_row_index_val, int& block_first_column_index_val) {
        int group_first_unit_row_index_val = (unit_index_val / units_per_group_val) * block_rows_per_group_val;
        int group_unit_rows_val = min(block_rows_per_group_val, total_unit_rows_val - group_first_unit_row_index_val);
        int unit_in_group_val = unit_index_val % units_per_group_val;
        int unit_row_index_val = group_first_unit_row_index_val + unit_in_group_val % group_unit_rows_val;
        int unit_column_index_val = unit_in_group_val / group_unit_rows_val;
        block_first_row_index_val = unit_row_index_val * UNIT_ROWS + (PINGPONG ? 0 : cluster_block_index_val * BLOCK_ROWS);
        block_first_column_index_val = unit_column_index_val * COLUMNS;
    };

    if (threadIdx.x < PRODUCER_THREADS) {
        if (MOVE_REGISTERS) mlp_up_8bit_give_registers<PRODUCER_REGISTERS>();   // the whole warpgroup
        // 3. the producer: its thread 0 asks for every step's tiles of every unit, set after set (walk_step counts
        //    the steps since the launch: its set is walk_step % STAGES, the set's lap walk_step / STAGES). From the
        //    second lap on, it first waits until the consumers that used the set are done with it. A step's ask: tell
        //    the set's full barrier how many bytes are coming, then the left tiles and the right tile: with clusters,
        //    block 0 asks for the right tile for every block. Past the edges TMA gives 0s
        if (threadIdx.x == 0) {
            int walk_step_index_val = 0;
            for (int unit_index_val = cluster_index_val; unit_index_val < total_units_val; unit_index_val += total_clusters_val) {
                int block_first_row_index_val, block_first_column_index_val;
                place_unit_val(unit_index_val, block_first_row_index_val, block_first_column_index_val);
                for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++, walk_step_index_val++) {
                    int set_index_val = walk_step_index_val % STAGES;
                    int lap_index_val = walk_step_index_val / STAGES;
                    if (lap_index_val > 0)
                        mlp_up_8bit_barrier_wait(&empty_barriers_list[set_index_val], (uint32_t)((lap_index_val - 1) % 2));
                    uint8_t* left_tile_ptr = tiles_ptr + set_index_val * SET_BYTES;
                    uint8_t* right_tile_ptr = left_tile_ptr + LEFT_BYTES;
                    int step_first_k_index_val = step_index_val * MLP_UP_8BIT_BLOCK_K;
                    mlp_up_8bit_barrier_expect(&full_barriers_list[set_index_val], (uint32_t)SET_BYTES);
#pragma unroll
                    for (int row_group_index_val = 0; row_group_index_val < SET_LEFT_TILES; row_group_index_val++)
                        mlp_up_8bit_copy_tile(left_tile_ptr + row_group_index_val * MLP_UP_8BIT_LEFT_TILE_BYTES, &left_map_val, step_first_k_index_val,
                                      block_first_row_index_val + row_group_index_val * HARDWARE_WGMMA_M, &full_barriers_list[set_index_val]);
                    if (CLUSTER == 1)
                        mlp_up_8bit_copy_tile(right_tile_ptr, &right_map_val, step_first_k_index_val, block_first_column_index_val,
                                      &full_barriers_list[set_index_val]);
                    else if (cluster_block_index_val == 0)
                        mlp_up_8bit_copy_tile_to_cluster(right_tile_ptr, &right_map_val, step_first_k_index_val, block_first_column_index_val,
                                                 &full_barriers_list[set_index_val], (uint16_t)((1u << CLUSTER) - 1));
                }
            }
        }
    } else {
        if (MOVE_REGISTERS) mlp_up_8bit_take_registers<CONSUMER_REGISTERS>();   // the whole warpgroup
        // 4. a consumer: walk each unit's sum on its own left tile and the shared right tile, then store its rows
        int consumer_thread_index_val = (int)threadIdx.x - PRODUCER_THREADS;       // 0 on: warpgroup g's threads 128 g on
        int row_group_index_val = consumer_thread_index_val / HARDWARE_WARPGROUP_THREADS;
        int warpgroup_thread_index_val = consumer_thread_index_val % HARDWARE_WARPGROUP_THREADS;
        bool warpgroup_first_thread_val = warpgroup_thread_index_val == 0;
        int warp_index_val = warpgroup_thread_index_val / HARDWARE_THREADS_PER_WARP;   // in the warpgroup: its rows 16 w on
        int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
        int thread_first_row_in_group_val = warp_index_val * HARDWARE_WGMMA_M_PER_WARP + lane_index_val / HARDWARE_WGMMA_THREADS_PER_ROW;
        int thread_pair_column_index_val = HARDWARE_WGMMA_COLUMNS_PER_THREAD * (lane_index_val % HARDWARE_WGMMA_THREADS_PER_ROW);
        uint8_t* room_ptr = rooms_ptr + row_group_index_val * ROOM_BYTES;
        uint32_t store_phase_val = 0;                                              // the store barrier's phase to wait for next
        // gelu_output in 8 bits: its one scale (a power of 2, so 1 / scale is exact), and this thread's largest |value|
        // written, for the next call's scale
        float side_scale_val = 1.0f, side_to_stored_val = 1.0f, thread_largest_val = 0.0f;
        if constexpr (GELU_8BIT) {
            side_scale_val = round_8bit_scale(__uint_as_float(*side_scale_largest_ptr) * side_margin_val);
            side_to_stored_val = 1.0f / side_scale_val;
        }
        // this warpgroup is done with a step's set: its first thread arrives at the set's empty barrier in every block
        // of the cluster
        auto free_set_val = [&](int set_index_val) {
            if (!warpgroup_first_thread_val) return;
            if (CLUSTER == 1) {
                mlp_up_8bit_barrier_arrive(&empty_barriers_list[set_index_val]);
            } else {
#pragma unroll
                for (int cluster_block_val = 0; cluster_block_val < CLUSTER; cluster_block_val++)
                    mlp_up_8bit_barrier_arrive_in_cluster(&empty_barriers_list[set_index_val], (uint32_t)cluster_block_val);
            }
        };
        // every thread of this warpgroup at the same point: all arrive at its store barrier and wait for the phase
        auto meet_warpgroup_val = [&]() {
            mlp_up_8bit_barrier_arrive(&store_barriers_list[row_group_index_val]);
            mlp_up_8bit_barrier_wait(&store_barriers_list[row_group_index_val], store_phase_val);
            store_phase_val ^= 1;
        };
        // ping-pong: this warpgroup's turn at the tensor cores, its k-th wait (phase k % 2); the other's first thread
        // arrives when its sums are done. Warpgroup 0's first unit needs no turn
        uint32_t turn_phase_val = 0;
        bool first_turn_val = true;

        // Units of the block in order (ordinal o: unit cluster + o x clusters); cooperative, every one; ping-pong, this
        // warpgroup's: o = g, g + 2, ... Every unit is total_steps steps, so unit o's sets start at walk step o x total_steps
        for (int unit_ordinal_val = PINGPONG ? row_group_index_val : 0;; unit_ordinal_val += PINGPONG ? 2 : 1) {
            int unit_index_val = cluster_index_val + unit_ordinal_val * total_clusters_val;
            if (unit_index_val >= total_units_val) break;
            int walk_step_index_val = unit_ordinal_val * total_steps_val;      // the unit's first step since the launch, as the producer counts them
            int block_first_row_index_val, block_first_column_index_val;
            place_unit_val(unit_index_val, block_first_row_index_val, block_first_column_index_val);
            int group_first_row_index_val = block_first_row_index_val + (PINGPONG ? 0 : row_group_index_val * HARDWARE_WGMMA_M);
            if (PINGPONG && !(row_group_index_val == 0 && first_turn_val)) {   // wait for the other warpgroup's sums of the unit before
                mlp_up_8bit_barrier_wait(&turn_barriers_list[row_group_index_val], turn_phase_val);
                turn_phase_val ^= 1;
            }
            first_turn_val = false;
            // this thread's part of its warpgroup's 64 rows: the float sums, and the tensor cores' sums since the last
            // move (PROMOTION_STEPS 0: the tensor cores add into the float sums themselves)
            float sums_list[COLUMN_GROUPS][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
            float tensor_sums_list[PROMOTION_STEPS > 0 ? COLUMN_GROUPS : 1][HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP] = {};
            for (int step_index_val = 0; step_index_val < total_steps_val; step_index_val++) {
                // b. wait until this step's set is full: its barrier's phase is the set's lap
                int set_index_val = (walk_step_index_val + step_index_val) % STAGES;
                mlp_up_8bit_barrier_wait(&full_barriers_list[set_index_val], (uint32_t)(((walk_step_index_val + step_index_val) / STAGES) % 2));

                // c. the tensor cores: MLP_UP_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3 WGMMAs, each 32 more values (bytes) of k; the
                //    first of a fresh start writes the tensor cores' sums over (adds onto 0)
                const uint8_t* left_tile_ptr = tiles_ptr + set_index_val * SET_BYTES + (PINGPONG ? 0 : row_group_index_val * MLP_UP_8BIT_LEFT_TILE_BYTES);
                const uint8_t* right_tile_ptr = tiles_ptr + set_index_val * SET_BYTES + LEFT_BYTES;
                bool start_again_val = PROMOTION_STEPS > 0 ? step_index_val % PROMOTION_PERIOD == 0 : step_index_val == 0;
                mlp_up_8bit_wgmma_fence();                                                 // the sums ready for the tensor cores
#pragma unroll
                for (int k_step_index_val = 0; k_step_index_val < MLP_UP_8BIT_BLOCK_K / HARDWARE_WGMMA_K_E4M3; k_step_index_val++) {
                    int add_onto_sums_val = (k_step_index_val == 0 && start_again_val) ? 0 : 1;
                    if constexpr (PROMOTION_STEPS > 0)
                        mlp_up_8bit_multiply_add(tensor_sums_list, left_tile_ptr + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                                         right_tile_ptr + k_step_index_val * HARDWARE_WGMMA_K_E4M3, add_onto_sums_val);
                    else
                        mlp_up_8bit_multiply_add(sums_list, left_tile_ptr + k_step_index_val * HARDWARE_WGMMA_K_E4M3,
                                         right_tile_ptr + k_step_index_val * HARDWARE_WGMMA_K_E4M3, add_onto_sums_val);
                }
                mlp_up_8bit_wgmma_commit();                                                // the step's WGMMAs as one group

                // d. wait for the WGMMAs of LEFT_RUNNING steps back (this step's own still running); this warpgroup is then
                //    done with that step's set. When the tensor cores' sums move: wait for them all, then add them into
                //    the float sums
                bool move_val = PROMOTION_STEPS > 0 &&
                                ((step_index_val + 1) % PROMOTION_PERIOD == 0 || step_index_val + 1 == total_steps_val);
                if (move_val) {
                    mlp_up_8bit_wgmma_wait<0>();
                    if constexpr (PROMOTION_STEPS > 0) {
#pragma unroll
                        for (int group_index_val = 0; group_index_val < COLUMN_GROUPS; group_index_val++)
#pragma unroll
                            for (int c_value_index_val = 0; c_value_index_val < HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP; c_value_index_val++)
                                sums_list[group_index_val][c_value_index_val] += tensor_sums_list[group_index_val][c_value_index_val];
                    }
                } else {
                    mlp_up_8bit_wgmma_wait<LEFT_RUNNING>();
                }
                if (step_index_val >= LEFT_RUNNING)
                    free_set_val((walk_step_index_val + step_index_val - LEFT_RUNNING) % STAGES);
            }
            // e. the last steps' WGMMAs done, and their sets free: the producer may be filling them for the next unit
            mlp_up_8bit_wgmma_wait<0>();
            for (int step_index_val = max(total_steps_val - LEFT_RUNNING, 0); step_index_val < total_steps_val; step_index_val++)
                free_set_val((walk_step_index_val + step_index_val) % STAGES);
            //    ping-pong: this warpgroup's sums are done; the other's turn at the tensor cores, while this one stores
            if (PINGPONG && warpgroup_first_thread_val) mlp_up_8bit_barrier_arrive(&turn_barriers_list[1 - row_group_index_val]);

            // each of this thread's pairs, sums times both scales (+ bias), handed to what_val with its place: the
            // group of 8 columns, which of the thread's two rows, the row in the warpgroup's 64 and the column in the
            // square (the pair's first)
            float left_scale_val = *left_scale_ptr, right_scale_val = *right_scale_ptr;   // one each: every value's
            auto each_pair_val = [&](auto what_val) {
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
                        // wgmma_matmul_8bit's order: sum x left scale x right scale (+ bias)
                        float2 pair_val = make_float2(
                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val] * left_scale_val * right_scale_val
                                + bias_val.x,
                            sums_list[group_index_val][HARDWARE_WGMMA_COLUMNS_PER_THREAD * thread_row_index_val + 1] * left_scale_val * right_scale_val
                                + bias_val.y);
                        what_val(row_in_group_val, column_in_square_val, pair_val);
                    }
                }
            };
            // g. what_val's result for each pair, into the room as bfloat16 at its place (box b: columns 64 b on, rows
            //    of 128 bytes, the 16-byte cell c of row r at c XOR r % 8, as TMA's 128-byte swizzle has it)
            auto write_room_val = [&](auto what_val) {
                each_pair_val([&](int row_in_group_val, int column_in_square_val, float2 pair_val) {
                    int box_index_val = column_in_square_val / MLP_UP_8BIT_ROW_VALUES;
                    int column_in_box_val = column_in_square_val % MLP_UP_8BIT_ROW_VALUES;
                    int cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) / HARDWARE_TMA_BYTES_PER_CELL;
                    int byte_in_cell_val = column_in_box_val * (int)sizeof(nv_bfloat16) % HARDWARE_TMA_BYTES_PER_CELL;
                    int swizzled_cell_val = cell_val ^ (row_in_group_val % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN);
                    nv_bfloat162* room_pair_ptr = reinterpret_cast<nv_bfloat162*>(
                        room_ptr + box_index_val * MLP_UP_8BIT_STORE_BOX_BYTES + row_in_group_val * HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW +
                        swizzled_cell_val * HARDWARE_TMA_BYTES_PER_CELL + byte_in_cell_val);
                    float2 result_val = what_val(pair_val);
                    *room_pair_ptr = __floats2bfloat162_rn(result_val.x, result_val.y);
                });
            };
            // the room's bfloat16 boxes stored by TMA into map_ptr's tensor, once every thread's writes are visible to it
            auto store_room_val = [&](const CUtensorMap* map_ptr) {
                mlp_up_8bit_fence_for_tma();                                               // this thread's writes visible to TMA
                meet_warpgroup_val();                                              // every thread's
                if (warpgroup_first_thread_val) {
#pragma unroll
                    for (int box_index_val = 0; box_index_val < OUTPUT_BOXES; box_index_val++)
                        mlp_up_8bit_store_box(map_ptr, block_first_column_index_val + box_index_val * MLP_UP_8BIT_ROW_VALUES, group_first_row_index_val,
                                      room_ptr + box_index_val * MLP_UP_8BIT_STORE_BOX_BYTES);
                    mlp_up_8bit_store_commit();
                }
            };
            // the room free (TMA has read it): the first thread waits, then the warpgroup meets
            auto wait_room_read_val = [&]() {
                if (warpgroup_first_thread_val) mlp_up_8bit_store_wait_read();
                meet_warpgroup_val();
            };

            // f. gelu_input: once TMA has read the room for the unit before, each pair (+ bias) as bfloat16, then TMA
            //    stores the boxes. Past the edges TMA writes nothing
            wait_room_read_val();
            write_room_val([](float2 pair_val) { return pair_val; });
            store_room_val(&output_map_val);
            // h. then gelu_output, once TMA has read the room
            wait_room_read_val();
            if constexpr (!GELU_8BIT) {
                //    as bfloat16: gelu of each pair (the bfloat16 one, as gelu_input holds it), through the same room
                write_room_val([](float2 pair_val) { return mlp_up_8bit_gelu_of(pair_val); });
                store_room_val(&side_map_val);
            } else {
                //    in 8 bits: gelu of each pair rounded to bfloat16 (what round_8bit_per_tensor would read), its
                //    largest |value| kept (inside the matrix only: past the edges the sums are 0s, not values), then
                //    times 1 / scale and rounded to E4M3 as round_8bit_tile_kernel does (saturating). The row copy's
                //    pair: 2 bytes of its swizzled row; the turned copy's: one byte in each of 2 lines
                each_pair_val([&](int row_in_group_val, int column_in_square_val, float2 pair_val) {
                    float2 gelu_float_val = mlp_up_8bit_gelu_of(pair_val);
                    float2 gelu_val = __bfloat1622float2(__floats2bfloat162_rn(gelu_float_val.x, gelu_float_val.y));
                    if (group_first_row_index_val + row_in_group_val < rows_val && block_first_column_index_val + column_in_square_val < columns_val)
                        thread_largest_val = fmaxf(thread_largest_val, fmaxf(fabsf(gelu_val.x), fabsf(gelu_val.y)));
                    __nv_fp8x2_e4m3 stored_val(make_float2(gelu_val.x * side_to_stored_val, gelu_val.y * side_to_stored_val));
                    unsigned short stored_bits_val = stored_val.__x;                       // the first value in the low byte
                    int box_index_val = column_in_square_val / MLP_UP_8BIT_BLOCK_K;
                    int byte_in_row_val = column_in_square_val % MLP_UP_8BIT_BLOCK_K;
                    int swizzled_cell_val = (byte_in_row_val / HARDWARE_TMA_BYTES_PER_CELL) ^ (row_in_group_val % HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN);
                    *reinterpret_cast<unsigned short*>(room_ptr + box_index_val * ROW_8BIT_BOX_BYTES + row_in_group_val * HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW +
                                                       swizzled_cell_val * HARDWARE_TMA_BYTES_PER_CELL + byte_in_row_val % HARDWARE_TMA_BYTES_PER_CELL) =
                        stored_bits_val;
                    uint8_t* turned_ptr = room_ptr + TURNED_8BIT_OFFSET + column_in_square_val * HARDWARE_WGMMA_M + row_in_group_val;
                    turned_ptr[0] = (uint8_t)(stored_bits_val & 0xFF);
                    turned_ptr[HARDWARE_WGMMA_M] = (uint8_t)(stored_bits_val >> 8);
                });
                mlp_up_8bit_fence_for_tma();                                               // this thread's writes visible to TMA
                meet_warpgroup_val();                                              // every thread's
                if (warpgroup_first_thread_val) {
#pragma unroll
                    for (int box_index_val = 0; box_index_val < ROW_8BIT_BOXES; box_index_val++)
                        mlp_up_8bit_store_box(&side_map_val, block_first_column_index_val + box_index_val * MLP_UP_8BIT_BLOCK_K, group_first_row_index_val,
                                              room_ptr + box_index_val * ROW_8BIT_BOX_BYTES);
                    mlp_up_8bit_store_box(&side_turned_map_val, group_first_row_index_val, block_first_column_index_val, room_ptr + TURNED_8BIT_OFFSET);
                    mlp_up_8bit_store_commit();
                }
                //    the one scale, by the first square's first thread (every block works out the same)
                if (warpgroup_first_thread_val && block_first_column_index_val == 0 && group_first_row_index_val == 0)
                    *side_scale_ptr = side_scale_val;
            }
        }
        //    gelu_output in 8 bits: this thread's largest onto the next call's, a warp at a time
        if constexpr (GELU_8BIT) {
            float warp_largest_val = warp_reduce_max(thread_largest_val);
            if (side_next_largest_ptr != nullptr && lane_index_val == 0 && warp_largest_val > 0.0f)
                atomicMax(side_next_largest_ptr, __float_as_uint(warp_largest_val));
        }
    }

    // 5. the consumers' TMA stores all written before the block leaves
    if (threadIdx.x >= PRODUCER_THREADS && (threadIdx.x - PRODUCER_THREADS) % HARDWARE_WARPGROUP_THREADS == 0)
        mlp_up_8bit_store_wait_all();

    // 6. with clusters: every thread of the cluster meets before any block leaves, so no block's shared memory goes
    //    while another block may still write into it (the right tile's multicast, the arrivals at its empty
    //    barriers). Not .aligned: the producer's threads come from a branch only its thread 0 took
    if (CLUSTER > 1) mlp_up_8bit_cluster_sync<false>();
}

#endif // MLP_UP_8BIT_CUH
