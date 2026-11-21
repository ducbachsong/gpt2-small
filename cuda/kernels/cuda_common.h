// cuda_common.h — what every CUDA file here needs: error checks, and sums and maxes across a warp and a block.
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define cudaCheck(call) cuda_check((call), __FILE__, __LINE__)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))               // a / b rounded up: CEIL_DIV(1000, 256) = 4

// HARDWARE_: a number the hardware sets. In groups, each made from the groups before it:
//   bits, warps and blocks, registers, mma.sync, WGMMA, shared memory, TMA (writes shared memory), the WGMMA
//   descriptor (reads it), E4M3.

// Bits
#define HARDWARE_VALUES_PER_BIT 2                            // a bit is 0 or 1, so n bits hold 2^n values (written 1ull << n: each bit doubles)
/// Gives the bits needed to count `count` different values (0 to count - 1).
__host__ __device__ constexpr int bits_for_values(unsigned long long count) {
    // Log2(count) rounded up, so bits_for_values(14592) = 14, as 2^13 = 8192 is too few and 2^14 = 16384 enough.
    // Each bit doubles the values (HARDWARE_VALUES_PER_BIT), so it halves count, rounded up, until one value is
    // left.
    return count <= 1 ? 0 : 1 + bits_for_values(CEIL_DIV(count, 2ull));
}

// Warps and blocks
#define HARDWARE_THREADS_PER_WARP 32                         // threads that run together: CUDA's warpSize
#define HARDWARE_WARP_ALL_LANES 0xffffffff                   // a shuffle's mask: one bit per thread of the warp, all taking part
#define HARDWARE_MAX_THREADS_PER_BLOCK 1024                  // CUDA's limit on one block
#define HARDWARE_MAX_WARPS_PER_BLOCK (HARDWARE_MAX_THREADS_PER_BLOCK / HARDWARE_THREADS_PER_WARP)   // the most warps a block can have: 1024 / 32 = 32

// Registers: a thread's own
#define HARDWARE_REGISTER_BYTES 4                            // a register: 32 bits, a thread's own
#define HARDWARE_BF16_PER_REGISTER (HARDWARE_REGISTER_BYTES / 2)     // 2 bfloat16s (2 bytes each)
#define HARDWARE_E4M3_PER_REGISTER (HARDWARE_REGISTER_BYTES / 1)     // 4 E4M3s (1 byte each)
#define HARDWARE_FLOATS_PER_REGISTER (HARDWARE_REGISTER_BYTES / 4)   // 1 float (4 bytes)
#define HARDWARE_REGISTERS_PER_SM 65536                      // an SM's registers, shared by the threads of its blocks (sm_80 and on)
#define HARDWARE_REGISTER_ALLOCATION_UNIT 8                  // a thread's registers come in multiples of 8
#define HARDWARE_SETMAXNREG_MIN 24                           // setmaxnreg (sm_90): a warpgroup's registers a thread, at least 24 ...
#define HARDWARE_SETMAXNREG_MAX 256                          // ... at most 256

// mma.sync (sm_80 or later): one warp makes C {M, N} += A {M, K} · B {K, N}. Its threads first load their parts of A
// and B into registers, then hand them to the tensor core; C comes back in their registers too.
//
//                                              N 8
//                                            ┌─────┐
//                                     K_BYTES│  B  │  (bf16: K 16, E4M3: K 32, TF32: K 8)
//                                       32   └─────┘
//            K_BYTES 32                        N 8
//         ┌─────────────┐                    ┌─────┐
//    M 16 │      A      │        ·      =    │  C  │ M 16    (C += A · B, C in float)
//         └─────────────┘                    └─────┘
//
//   Who holds C {16, 8}: lane t holds 2 side-by-side columns of 2 rows (4 floats).
//
//                col 0-1   col 2-3   col 4-5   col 6-7     ← COLUMNS_PER_THREAD 2, THREADS_PER_ROW 4
//      row 0     lane 0    lane 1    lane 2    lane 3   ↑
//      row 1     lane 4    lane 5    lane 6    lane 7   │  ROWS_PER_WARP 8: the 32 lanes cover 8 rows
//       ...                                             │  (row = lane / 4, columns 2 (lane % 4) and + 1)
//      row 7     lane 28   lane 29   lane 30   lane 31  ↓
//      row 8     lane 0    lane 1    lane 2    lane 3   ↑  the same lanes again: ROWS_PER_THREAD 2
//       ...                                             │  (row + 8)
//      row 15    lane 28   lane 29   lane 30   lane 31  ↓
//
#define HARDWARE_MMA_M 16                                    // rows of A and C: a tile's rows, 2 passes of HARDWARE_MMA_ROWS_PER_WARP
#define HARDWARE_MMA_N 8                                     // columns of B and C
#define HARDWARE_MMA_K_BYTES 32                              // bytes of k an mma reads from each row of A and B
#define HARDWARE_MMA_K_BF16 (HARDWARE_MMA_K_BYTES / 2)       // ... 16 bfloat16s: m16n8k16
#define HARDWARE_MMA_K_E4M3 (HARDWARE_MMA_K_BYTES / 1)       // ... 32 E4M3s: m16n8k32
#define HARDWARE_MMA_K_TF32 (HARDWARE_MMA_K_BYTES / 4)       // ... 8 floats: m16n8k8, read as TF32 (a float with 10 bits of mantissa, not 23)
// Each thread's part of a tile: the tile's values / the warp's 32 threads / the values one register holds.
#define HARDWARE_MMA_A_REGISTERS_PER_THREAD_BF16 (HARDWARE_MMA_M * HARDWARE_MMA_K_BF16 / HARDWARE_THREADS_PER_WARP / HARDWARE_BF16_PER_REGISTER)   // A {16, 16}: 256 / 32 / 2 = 4
#define HARDWARE_MMA_B_REGISTERS_PER_THREAD_BF16 (HARDWARE_MMA_K_BF16 * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_BF16_PER_REGISTER)   // B {16, 8}: 128 / 32 / 2 = 2
#define HARDWARE_MMA_C_VALUES_PER_THREAD_BF16 (HARDWARE_MMA_M * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_FLOATS_PER_REGISTER)         // C {16, 8}: 128 / 32 / 1 = 4 floats
#define HARDWARE_MMA_A_REGISTERS_PER_THREAD_E4M3 (HARDWARE_MMA_M * HARDWARE_MMA_K_E4M3 / HARDWARE_THREADS_PER_WARP / HARDWARE_E4M3_PER_REGISTER)   // A {16, 32}: 512 / 32 / 4 = 4
#define HARDWARE_MMA_B_REGISTERS_PER_THREAD_E4M3 (HARDWARE_MMA_K_E4M3 * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_E4M3_PER_REGISTER)   // B {32, 8}: 256 / 32 / 4 = 2
#define HARDWARE_MMA_C_VALUES_PER_THREAD_E4M3 (HARDWARE_MMA_M * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_FLOATS_PER_REGISTER)         // C {16, 8}: 128 / 32 / 1 = 4 floats (sums in float too)
#define HARDWARE_MMA_A_REGISTERS_PER_THREAD_TF32 (HARDWARE_MMA_M * HARDWARE_MMA_K_TF32 / HARDWARE_THREADS_PER_WARP / HARDWARE_FLOATS_PER_REGISTER)   // A {16, 8}: 128 / 32 / 1 = 4
#define HARDWARE_MMA_B_REGISTERS_PER_THREAD_TF32 (HARDWARE_MMA_K_TF32 * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_FLOATS_PER_REGISTER)   // B {8, 8}: 64 / 32 / 1 = 2
#define HARDWARE_MMA_C_VALUES_PER_THREAD_TF32 (HARDWARE_MMA_M * HARDWARE_MMA_N / HARDWARE_THREADS_PER_WARP / HARDWARE_FLOATS_PER_REGISTER)           // C {16, 8}: 128 / 32 / 1 = 4 floats
// mma's layout: a thread holds 2 side-by-side columns of a row, so a row's 8 columns take 4 lanes, the warp's 32
#define HARDWARE_MMA_COLUMNS_PER_THREAD 2                                                       // columns of a row one thread holds
#define HARDWARE_MMA_THREADS_PER_ROW (HARDWARE_MMA_N / HARDWARE_MMA_COLUMNS_PER_THREAD)         // threads that share a row: 8 / 2 = 4
#define HARDWARE_MMA_ROWS_PER_WARP (HARDWARE_THREADS_PER_WARP / HARDWARE_MMA_THREADS_PER_ROW)   // rows the warp's 32 threads cover in one pass: 32 / 4 = 8
#define HARDWARE_MMA_ROWS_PER_THREAD (HARDWARE_MMA_M / HARDWARE_MMA_ROWS_PER_WARP)              // rows one thread holds: 16 / 8 = 2 (row and row + 8)
#define HARDWARE_MMA_BF16_SMALLEST_MAJOR 8                   // compute capability's major: the first GPUs with bfloat16 mma (8.0)
#define HARDWARE_MMA_E4M3_SMALLEST_CAPABILITY 89             // compute capability x 10: the first GPUs with E4M3 mma (8.9)

// WGMMA (Hopper, sm_90a): one warpgroup (4 warps, 128 threads) makes C {M, N} += A {M, K} · B {K, N}. M and K are fixed,
// N is the kernel's choice. No thread loads A or B: the tensor cores read them straight from shared memory, told where by
// two descriptors (below). Only C is in registers.
//
// Example: our kernels' m64n64k16 in bf16 (M 64, N 64, K 16 bf16 = K_BYTES 32):
//
//   1. The tiles, both in shared memory (written there by TMA, swizzled). B is kept as one row per column of C, so A
//      and B both have their values of k side by side ("K-major"). The tiles are 64 values of k wide; one WGMMA uses
//      16 of them, so the kernel calls it 4 times per tile, moving 16 values of k along each time.
//
//               A {64, 16}                                     B {64, 16}
//             k 0 ......... 15                               k 0 ......... 15
//            ┌────────────────┐                             ┌────────────────┐
//     row  0 │                │                   column  0 │                │
//      ...   │                │                     ...     │                │
//     row 63 │                │                   column 63 │                │
//            └────────────────┘                             └────────────────┘
//
//   2. One sum of C: row r of A times row c of B, 16 multiply-adds:
//
//        C[r][c] += A[r][0] · B[c][0] + A[r][1] · B[c][1] + ... + A[r][15] · B[c][15]
//
//      C {64, 64} has 4,096 sums, so one WGMMA does 65,536 multiply-adds.
//
//   3. Who makes which rows of C: each warp drives its own tensor core and makes 16 rows (M_PER_WARP 16 = 64 / 4).
//      It needs only its 16 rows of A, but all 64 rows of B (one for every column of C), so B is shared by all 4:
//
//        warp 0:  A rows  0-15  ·  all of B  →  C rows  0-15
//        warp 1:  A rows 16-31  ·  all of B  →  C rows 16-31
//        warp 2:  A rows 32-47  ·  all of B  →  C rows 32-47
//        warp 3:  A rows 48-63  ·  all of B  →  C rows 48-63
//
//   4. A warp's 16 rows of C are cut into groups of 8 columns (GROUP_COLUMNS 8): N 64 / 8 = 8 groups.
//
//                   group 0    group 1    group 2          group 7
//                   col 0-7    col 8-15   col 16-23  ...   col 56-63
//                 ┌──────────┬──────────┬──────────┬─────┬──────────┐
//      rows 0-15  │ {16, 8}  │ {16, 8}  │ {16, 8}  │ ... │ {16, 8}  │   warp 0 (warps 1-3: the same, 16, 32
//                 └──────────┴──────────┴──────────┴─────┴──────────┘   and 48 rows further down)
//
//   5. Each group is laid out like mma's C {16, 8} above: lane t holds rows t / 4 and t / 4 + 8 (ROWS_PER_THREAD 2) and
//      columns 2 (t % 4) and 2 (t % 4) + 1 (COLUMNS_PER_THREAD 2): 4 floats a group (C_VALUES_PER_THREAD_PER_GROUP),
//      8 groups x 4 = 32 floats a thread. For example warp 1, lane 5: rows 16 + 5 / 4 = 17 and 17 + 8 = 25, columns
//      2 (5 % 4) = 2 and 3 of every group:
//
//                     group 0        group 1          group 2                 group 7
//        row 17      col 2, 3       col 10, 11       col 18, 19     ...      col 58, 59
//        row 25      col 2, 3       col 10, 11       col 18, 19     ...      col 58, 59
//                    sums[0][0-3]   sums[1][0-3]     sums[2][0-3]            sums[7][0-3]   (sums_list, wgmma_matmul_bf16.cuh)
//
#define HARDWARE_WARPGROUP_WARPS 4                           // warps that issue one WGMMA together: one per tensor core of the SM
#define HARDWARE_WARPGROUP_THREADS (HARDWARE_WARPGROUP_WARPS * HARDWARE_THREADS_PER_WARP)   // 4 x 32 = 128
#define HARDWARE_WGMMA_M 64                                  // rows of A and C: always 64
#define HARDWARE_WGMMA_K_BYTES 32                            // bytes of k a WGMMA reads from each row of A and B
#define HARDWARE_WGMMA_K_BF16 (HARDWARE_WGMMA_K_BYTES / 2)   // ... 16 bfloat16s: m64nNk16
#define HARDWARE_WGMMA_K_E4M3 (HARDWARE_WGMMA_K_BYTES / 1)   // ... 32 E4M3s: m64nNk32
#define HARDWARE_WGMMA_K_TF32 (HARDWARE_WGMMA_K_BYTES / 4)   // ... 8 floats: m64nNk8, read as TF32
// Each thread's part of C (B always stays in shared memory; A too, or in registers, below), per group of 8 columns: the
// group's values / the warpgroup's 128 threads / the values one register holds. N / 8 groups make a thread's whole part.
#define HARDWARE_WGMMA_GROUP_COLUMNS 8                       // columns of C in one group: N is a whole number of groups
#define HARDWARE_WGMMA_MAX_N 256                             // N, at most (the kernel picks any multiple of HARDWARE_WGMMA_GROUP_COLUMNS up to it)
#define HARDWARE_WGMMA_C_VALUES_PER_THREAD_PER_GROUP (HARDWARE_WGMMA_M * HARDWARE_WGMMA_GROUP_COLUMNS / HARDWARE_WARPGROUP_THREADS / HARDWARE_FLOATS_PER_REGISTER)   // C {64, 8}: 512 / 128 / 1 = 4 floats
// WGMMA's layout: warp w holds rows 16 w .. 16 w + 15, and in each group of 8 columns lays them out as mma's C {16, 8}:
#define HARDWARE_WGMMA_M_PER_WARP (HARDWARE_WGMMA_M / HARDWARE_WARPGROUP_WARPS)                             // rows of C one warp holds: 64 / 4 = 16
#define HARDWARE_WGMMA_COLUMNS_PER_THREAD 2                                                                 // columns of a row one thread holds, in each group
#define HARDWARE_WGMMA_THREADS_PER_ROW (HARDWARE_WGMMA_GROUP_COLUMNS / HARDWARE_WGMMA_COLUMNS_PER_THREAD)   // threads that share a row: 8 / 2 = 4
#define HARDWARE_WGMMA_ROWS_PER_WARP (HARDWARE_THREADS_PER_WARP / HARDWARE_WGMMA_THREADS_PER_ROW)           // rows the warp's 32 threads cover in one pass: 32 / 4 = 8
#define HARDWARE_WGMMA_ROWS_PER_THREAD (HARDWARE_WGMMA_M_PER_WARP / HARDWARE_WGMMA_ROWS_PER_WARP)           // rows one thread holds: 16 / 8 = 2 (row and row + 8)
// A in registers in place of shared memory (the product of something just worked out, as attention's probs): each warp
// holds its 16 rows of A laid out as mma's A, so two groups of C side by side are one A {16, 16} with no move.
#define HARDWARE_WGMMA_A_REGISTERS_PER_THREAD_BF16 (HARDWARE_WGMMA_M_PER_WARP * HARDWARE_WGMMA_K_BF16 / HARDWARE_THREADS_PER_WARP / HARDWARE_BF16_PER_REGISTER)   // a warp's A {16, 16}: 256 / 32 / 2 = 4
#define HARDWARE_WGMMA_CAPABILITY 90                         // compute capability x 10: the only GPUs with WGMMA (9.0, Hopper: an H100)

// Shared memory: its size
#define HARDWARE_SHARED_BYTES_PER_SM (228 * 1024)            // the shared memory an SM has (H100), split among the blocks on it
#define HARDWARE_DEFAULT_MAX_SHARED_BYTES_PER_BLOCK (48 * 1024)   // the most a block can have without asking
#define HARDWARE_MAX_SHARED_BYTES_PER_BLOCK (227 * 1024)     // the most a block can have when it asks, with cudaFuncSetAttribute (H100)
// Shared memory is 32 banks side by side, like a table of 32 columns: each bank is a column 4 bytes wide (one word)
// and as long as the shared memory / 32; byte b sits in bank (b / 4) % 32. A bank gives one word a clock, from any level.
#define HARDWARE_SHARED_BANKS 32                             // banks side by side: the table's columns
#define HARDWARE_SHARED_BANK_WIDTH_BYTES 4                   // a bank's width: one word, what it gives in one clock
#define HARDWARE_SHARED_BANK_LENGTH_BYTES (HARDWARE_SHARED_BYTES_PER_SM / HARDWARE_SHARED_BANKS)   // a bank's length, all of shared memory in use (H100): 228 KB / 32 = 7,296
#define HARDWARE_SHARED_WORDS_PER_BANK (HARDWARE_SHARED_BANK_LENGTH_BYTES / HARDWARE_SHARED_BANK_WIDTH_BYTES)        // the levels of a bank: 7,296 / 4 = 1,824
#define HARDWARE_SHARED_BYTES_PER_LEVEL (HARDWARE_SHARED_BANKS * HARDWARE_SHARED_BANK_WIDTH_BYTES)            // one level across all the banks: 32 x 4 = 128, what they give together in one clock

// TMA: one thread's instruction copies a box of a matrix from global memory into shared memory, in the 128-byte swizzle.
// From big to small (the defines below go the other way, each made from the ones above it):
//   box     up to 256 rows x 128 bytes, a stack of atoms; the kernels choose 64 rows (8 KB, 8 atoms)
//   atom    8 rows x 128 bytes = 1024: the swizzle's whole pattern; a tile starts on one
//   row     8 cells across = 128 bytes = all 32 banks once; cell c stored at slot c XOR (row % 8)
//   column  one slot down the 8 rows = 128 bytes: what the tensor cores read at once, 8 different slots, no bank conflict
//   cell    16 bytes, never split: TMA starts and jumps only at a cell's edge, so a matrix's start, its row length and a
//           box's row are whole cells (bf16 rows a multiple of 8 values, E4M3 rows of 16)
//
//   One atom in shared memory, after TMA's swizzle (c0..c7: a row's cells in k order, bf16 k 0-7, 8-15, ...):
//
//               ←──── SWIZZLE_BYTES_PER_ROW 128 = CELLS_PER_ATOM_ROW 8 cells ────→
//              slot 0 slot 1 slot 2 slot 3 slot 4 slot 5 slot 6 slot 7
//    row 0    [ c0 ][ c1 ][ c2 ][ c3 ][ c4 ][ c5 ][ c6 ][ c7 ]    ↑
//    row 1    [ c1 ][ c0 ][ c3 ][ c2 ][ c5 ][ c4 ][ c7 ][ c6 ]    │
//    row 2    [ c2 ][ c3 ][ c0 ][ c1 ][ c6 ][ c7 ][ c4 ][ c5 ]    │  CELLS_PER_ATOM_COLUMN 8 rows
//    row 3    [ c3 ][ c2 ][ c1 ][ c0 ][ c7 ][ c6 ][ c5 ][ c4 ]    │  (BYTES_PER_ATOM_COLUMN 128 / 16):
//    row 4    [ c4 ][ c5 ][ c6 ][ c7 ][ c0 ][ c1 ][ c2 ][ c3 ]    │  c0 of the 8 rows sits in 8 different
//    row 5    [ c5 ][ c4 ][ c7 ][ c6 ][ c1 ][ c0 ][ c3 ][ c2 ]    │  slots, so the tensor cores read it
//    row 6    [ c6 ][ c7 ][ c4 ][ c5 ][ c2 ][ c3 ][ c0 ][ c1 ]    │  in one go, no bank conflict
//    row 7    [ c7 ][ c6 ][ c5 ][ c4 ][ c3 ][ c2 ][ c1 ][ c0 ]    ↓
//               └─ one cell: BYTES_PER_CELL 16           the whole atom: BYTES_PER_ATOM 8 x 8 x 16 = 1024
//    row 8..  the next atom, the same pattern again; a box stacks atoms down to MAX_ROWS_PER_BOX 256 (32 atoms)
//
#define HARDWARE_TMA_BYTES_PER_CELL 16                                                                                 // TMA's cell: every start and jump is a whole number of them
#define HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW HARDWARE_SHARED_BYTES_PER_LEVEL                         // the swizzle pattern's row: 32 x 4 = 128, always
#define HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_ROW (HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / HARDWARE_TMA_BYTES_PER_CELL)      // across: 128 / 16 = 8 cells, so 8 slots
#define HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM_COLUMN HARDWARE_SHARED_BYTES_PER_LEVEL                 // one slot down one atom, one cell from each of its rows, read at once: all 32 banks once, 32 x 4 = 128
#define HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN (HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM_COLUMN / HARDWARE_TMA_BYTES_PER_CELL)   // down: 128 / 16 = 8 rows, so row % 8
#define HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM (HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_ROW * HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN * HARDWARE_TMA_BYTES_PER_CELL)   // 8 x 8 x 16 = 1024
#define HARDWARE_TMA_SWIZZLE_MAX_BYTES_PER_BOX_ROW HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW                                  // a box's row, at most: one pattern row, 128 (the kernels use all 128)
#define HARDWARE_TMA_MAX_ROWS_PER_BOX 256                                                                               // a box's rows, at most (the kernels use 64)
#define HARDWARE_TMA_MAX_VALUES_PER_BOX_ROW 256                                                                         // a box's row with no swizzle, at most, in values
#define HARDWARE_TMA_SHARED_ALIGN_BYTES 128                                                                             // a box's start in shared memory: on 128 bytes
#define HARDWARE_TMA_SWIZZLE_MAX_BYTES_PER_BOX (HARDWARE_TMA_MAX_ROWS_PER_BOX * HARDWARE_TMA_SWIZZLE_MAX_BYTES_PER_BOX_ROW)   // a box, at most: 256 x 128 = 32 KB (32 atoms)

// A WGMMA shared memory descriptor: one 64-bit number telling the tensor cores where a tile starts and how its rows
// sit, in 4 fields (the bits between them stay 0). Each field: its first bit (_SHIFT) and how many bits (_BITS).
//
//   bit:  63 62 | 61 ... 46 | 45 ........ 32 | 31 30 | 29 ........ 16 | 15 14 | 13 ......... 0
//        swizzle   (unused)   stride offset    (unused)  leading offset  (unused)     address
//           ↑                       ↑                          ↑                         ↑
//    SWIZZLE_SHIFT 62        STRIDE_SHIFT 32          LEADING_SHIFT 16          ADDRESS_SHIFT 0
//
// A field's value starts at bit 0 and is moved up into its box with << _SHIFT, then joined in with |=.
#define HARDWARE_WGMMA_DESCRIPTOR_UNIT HARDWARE_TMA_BYTES_PER_CELL                     // the address and offsets count 16-byte cells, as TMA does
#define HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_SHIFT 0            // the address field's first bit: its value moves up 0 (it holds where the tile starts, in cells: 8192 / 16 = 512)
#define HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_BITS bits_for_values(HARDWARE_SHARED_BYTES_PER_SM / HARDWARE_WGMMA_DESCRIPTOR_UNIT)   // the address field: enough for every cell of shared memory, 228 KB / 16 = 14,592 cells, 2^13 too few, so 14
#define HARDWARE_WGMMA_DESCRIPTOR_MAX_SHARED_BYTES ((1ull << HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_BITS) * HARDWARE_WGMMA_DESCRIPTOR_UNIT)   // the most it can point at: 2^14 cells x 16 = 256 KB, rounded up from 228 KB
#define HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_MASK_BITS (HARDWARE_WGMMA_DESCRIPTOR_MAX_SHARED_BYTES - 1)   // the byte address bits kept, all ones: 256 KB - 1 = 0x3FFFF (18 bits)
#define HARDWARE_WGMMA_DESCRIPTOR_LEADING_SHIFT 16           // the leading offset field's first bit: its value moves up 16 (it holds the jump along k, in cells; the 128-byte swizzle ignores it)
#define HARDWARE_WGMMA_DESCRIPTOR_LEADING_BITS HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_BITS   // an offset can be as far as any address: 14
#define HARDWARE_WGMMA_DESCRIPTOR_STRIDE_SHIFT 32            // the stride offset field's first bit: its value moves up 32 (it holds the jump to the next 8 rows, one atom, in cells: 1024 / 16 = 64)
#define HARDWARE_WGMMA_DESCRIPTOR_STRIDE_BITS HARDWARE_WGMMA_DESCRIPTOR_ADDRESS_BITS    // 14
#define HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_SHIFT 62           // the swizzle field's first bit: its value moves up 62 (it holds which swizzle: 0 none, 1 128-byte, 2 64-byte, 3 32-byte)
#define HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_BITS 2
#define HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_128 1ull           // 1 = 128-byte swizzle
#define HARDWARE_WGMMA_DESCRIPTOR_SWIZZLE_64 2ull            // 2 = 64-byte swizzle: a tile whose rows are 64 bytes (4 cells, cell c of row r at slot c XOR ((r / 2) % 4))

// E4M3, the 8-bit float
#define HARDWARE_E4M3_LARGEST 448.0f                         // the largest FP8 E4M3 value
#define HARDWARE_E4M3_SMALLEST 0.001953125f                  // the smallest E4M3 value above 0: 2^-9



/// Stops with the file and line if a CUDA call failed.
///
///   error_val             what the call returned
///   file_ptr, line_val    where the call is
static inline void cuda_check(cudaError_t error_val, const char* file_ptr, int line_val) {
    if (error_val != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file_ptr, line_val, cudaGetErrorString(error_val));
        exit(EXIT_FAILURE);
    }
}

/// Stops with a message unless a launch's block size is whole warps and within CUDA's limit.
///
///   threads_per_block_val   the block size asked for
///   kernel_name_ptr         the launcher's name, for the message
static inline void check_threads_per_block(int threads_per_block_val, const char* kernel_name_ptr) {
    // Every launcher checks its threads_per_block with it.
    if (threads_per_block_val <= 0 || threads_per_block_val > HARDWARE_MAX_THREADS_PER_BLOCK ||
        threads_per_block_val % HARDWARE_THREADS_PER_WARP != 0) {
        fprintf(stderr, "error: %s needs threads_per_block to be a multiple of %d, at most %d; got %d\n",
                kernel_name_ptr, HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK, threads_per_block_val);
        exit(EXIT_FAILURE);
    }
}

/// Gives the sum of `value_val` over the warp's HARDWARE_THREADS_PER_WARP threads, in every thread.
///
///   value_val   this thread's part
__device__ __forceinline__ float warp_reduce_sum(float value_val) {
    for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2)
        value_val += __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, value_val, offset_val);
    return value_val;
}

/// Gives the largest `value_val` over the warp's HARDWARE_THREADS_PER_WARP threads, in every thread.
///
///   value_val   this thread's part
__device__ __forceinline__ float warp_reduce_max(float value_val) {
    // The same fold, with fmaxf.
    for (int offset_val = HARDWARE_THREADS_PER_WARP / 2; offset_val > 0; offset_val /= 2)
        value_val = fmaxf(value_val, __shfl_xor_sync(HARDWARE_WARP_ALL_LANES, value_val, offset_val));
    return value_val;
}

/// Gives the sum of `value_val` over the block, in every thread.
///
///   value_val   this thread's part
__device__ __forceinline__ float block_reduce_sum(float value_val) {
    // Each warp's total to shared memory, then warp 0 adds those. Every thread must call it; blockDim.x a multiple
    // of HARDWARE_THREADS_PER_WARP.
    // one slot per warp; warp 0's HARDWARE_THREADS_PER_WARP lanes must cover them all
    static_assert(HARDWARE_MAX_WARPS_PER_BLOCK <= HARDWARE_THREADS_PER_WARP, "warp 0 must hold one total per warp");
    __shared__ float warp_totals_list[HARDWARE_MAX_WARPS_PER_BLOCK];
    __shared__ float block_total_val;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / HARDWARE_THREADS_PER_WARP;
    // 1. each warp's total, by its lane 0
    value_val = warp_reduce_sum(value_val);
    if (lane_index_val == 0) warp_totals_list[warp_index_val] = value_val;
    __syncthreads();
    // 2. warp 0 adds the warps' totals
    if (warp_index_val == 0) {
        float total_val = warp_reduce_sum(lane_index_val < warps_in_block_val ? warp_totals_list[lane_index_val] : 0.0f);
        if (lane_index_val == 0) block_total_val = total_val;
    }
    __syncthreads();
    // 3. every thread reads it
    float result_val = block_total_val;
    __syncthreads();          // before a next call writes warp_totals_list again
    return result_val;
}

/// Gives the largest `value_val` over the block, in every thread, as block_reduce_sum.
///
///   value_val   this thread's part
__device__ __forceinline__ float block_reduce_max(float value_val) {
    // one slot per warp; warp 0's HARDWARE_THREADS_PER_WARP lanes must cover them all
    static_assert(HARDWARE_MAX_WARPS_PER_BLOCK <= HARDWARE_THREADS_PER_WARP, "warp 0 must hold one maximum per warp");
    __shared__ float warp_maxima_list[HARDWARE_MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_val;
    int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    int warp_index_val = threadIdx.x / HARDWARE_THREADS_PER_WARP;
    int warps_in_block_val = blockDim.x / HARDWARE_THREADS_PER_WARP;
    // 1. each warp's largest, by its lane 0
    value_val = warp_reduce_max(value_val);
    if (lane_index_val == 0) warp_maxima_list[warp_index_val] = value_val;
    __syncthreads();
    // 2. warp 0 takes the largest of the warps'
    if (warp_index_val == 0) {
        float maximum_val = warp_reduce_max(lane_index_val < warps_in_block_val ? warp_maxima_list[lane_index_val] : -FLT_MAX);
        if (lane_index_val == 0) block_max_val = maximum_val;
    }
    __syncthreads();
    // 3. every thread reads it
    float result_val = block_max_val;
    __syncthreads();
    return result_val;
}

#endif // CUDA_COMMON_H
