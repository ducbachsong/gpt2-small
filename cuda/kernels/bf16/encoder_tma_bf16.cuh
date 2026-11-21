// encoder_tma_bf16.cuh — encoder_bf16's forward with TMA: TMA brings the wte and wpe rows into shared memory, the threads add them, TMA writes encoded back.
//
//     tensor        type in memory   shape
//     wte           bfloat16         {vocab_size, embed_dim}
//     wpe           bfloat16         {max_seq_len, embed_dim}
//     encoded       bfloat16         {batch_size, seq_len, embed_dim}
//     token_ids     int              {batch_size, seq_len}
//     a tile        bfloat16         {tile_rows, tile_columns} of wte rows, the same of wpe rows, in shared memory
#ifndef ENCODER_TMA_BF16_CUH
#define ENCODER_TMA_BF16_CUH

#include <cuda.h>                    // CUtensorMap, the TMA tile description (the types only: no -lcuda)
#include <cudaTypedefs.h>            // PFN_cuTensorMapEncodeTiled, to get it from the driver at run time
#include "encoder_bf16.cuh"             // encoder_bf16_check_shapes

// The settings: our choices, swept by tune_encoder_tma_bf16. Defaults until a tune sets them.
#define ENCODER_TMA_BF16_VALUES 16                   // values a thread adds a round: whole 16-byte cells (8 values)
#define ENCODER_TMA_BF16_SWIZZLE 128                 // TMA's swizzle in bytes: 128, or 0 for none
#define ENCODER_TMA_BF16_WARPS 8                     // warps a block: 1 to HARDWARE_MAX_WARPS_PER_BLOCK
#define ENCODER_TMA_BF16_BLOCKS_PER_SM 2             // blocks a launch, per SM
#define ENCODER_TMA_BF16_TILE_ROWS 32                // tokens a tile: divides seq_len, at most HARDWARE_TMA_MAX_ROWS_PER_BOX
#define ENCODER_TMA_BF16_TILE_COLUMNS 128            // values of a row a tile: divides embed_dim, a multiple of the box's row
#define ENCODER_TMA_BF16_STAGES 2                    // sets of tiles in shared memory: 1 to ENCODER_TMA_BF16_MAX_STAGES
#define ENCODER_TMA_BF16_MAX_STAGES 8                // the most sets of tiles
#define ENCODER_TMA_BF16_MAX_SHARED_BYTES (HARDWARE_MAX_SHARED_BYTES_PER_BLOCK - ENCODER_TMA_BF16_MAX_STAGES * (int)sizeof(uint64_t))   // the dynamic shared memory a block can ask for: the rest is the barriers
#define ENCODER_TMA_BF16_MAX_ROWS_PER_LANE (HARDWARE_TMA_MAX_ROWS_PER_BOX / HARDWARE_THREADS_PER_WARP)   // token ids a lane of warp 0 holds: 8

// ── the forward ───────────────────────────────────────────────────────────────
struct EncoderTmaBf16Tile;
void encoder_tma_bf16_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                              const Tensor& wpe_tensor);
template <int VALUES, int SWIZZLE_BYTES, int WARPS>
bool encoder_tma_bf16_forward_at(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                                 const Tensor& wpe_tensor, int blocks_val, int tile_rows_val, int tile_columns_val, int stages_val);

// ── for it ────────────────────────────────────────────────────────────────────
bool encoder_tma_bf16_plan(EncoderTmaBf16Tile* tile_ptr, size_t* shared_bytes_ptr, int swizzle_bytes_val, int tile_rows_val,
                           int tile_columns_val, int stages_val, int batch_size_val, int seq_len_val, int embed_dim_val);
CUtensorMap encoder_tma_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                        int box_columns_val, int box_rows_val, int swizzle_bytes_val);

// ══ the code, in the order of the list above ══════════════════════════════════

/// One launch's tile: its shape, how it sits in shared memory, and the product's sizes.
struct EncoderTmaBf16Tile {
    int tile_rows;                    // tokens a tile, the rows of each box but wte's
    int tile_columns;                 // values of a row a tile
    int box_columns;                  // values of a box's row: 64 with the 128-byte swizzle, up to 256 with none
    int chunk_bytes;                  // a box of tile_rows rows in shared memory, rounded up to its alignment
    int stages;                       // sets of tiles in shared memory
    int seq_len;                      // tokens a row of the batch: wpe's rows read
    int embed_dim;                    // values a token's row
    int tokens;                       // batch_size x seq_len: encoded's rows
};

// ── the forward ───────────────────────────────────────────────────────────────

template <int VALUES, int SWIZZLE_BYTES, int WARPS>
__global__ void __launch_bounds__(WARPS * HARDWARE_THREADS_PER_WARP, 1)   // one block an SM is enough: all of its registers
encoder_tma_bf16_kernel(const __grid_constant__ CUtensorMap wte_map_val, const __grid_constant__ CUtensorMap wpe_map_val,
                        const __grid_constant__ CUtensorMap encoded_map_val, const int* token_ids_ptr, EncoderTmaBf16Tile tile_val);
template <int SWIZZLE_BYTES>
__device__ __forceinline__ int encoder_tma_bf16_cell_offset(int chunk_index_val, int row_index_val, int cell_index_val,
                                                            const EncoderTmaBf16Tile& tile_val);
__device__ __forceinline__ uint32_t encoder_tma_bf16_shared_address(const void* shared_ptr);
__device__ __forceinline__ void encoder_tma_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val);
__device__ __forceinline__ void encoder_tma_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val);
__device__ __forceinline__ void encoder_tma_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val);
__device__ __forceinline__ void encoder_tma_bf16_copy_tile(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                           int row_val, uint64_t* barrier_ptr);

/// Writes encoded = wte[token_ids] + wpe in bfloat16 at the default settings.
///
///   encoded_tensor     {batch_size, seq_len, embed_dim} bfloat16: the result
///   token_ids_tensor   {batch_size, seq_len} ints
///   wte_tensor         {vocab_size, embed_dim} bfloat16
///   wpe_tensor         {max_seq_len, embed_dim} bfloat16
inline void encoder_tma_bf16_forward(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                                     const Tensor& wpe_tensor) {
    // The ENCODER_TMA_BF16_ settings, ENCODER_TMA_BF16_BLOCKS_PER_SM blocks an SM. The same bits as
    // encoder_bf16_forward: the same float add, rounded once. Needs an H100 (TMA).
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    bool fits_val = encoder_tma_bf16_forward_at<ENCODER_TMA_BF16_VALUES, ENCODER_TMA_BF16_SWIZZLE, ENCODER_TMA_BF16_WARPS>(
        encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor, sms_val * ENCODER_TMA_BF16_BLOCKS_PER_SM, ENCODER_TMA_BF16_TILE_ROWS,
        ENCODER_TMA_BF16_TILE_COLUMNS, ENCODER_TMA_BF16_STAGES);
    if (!fits_val) {
        fprintf(stderr, "error: encoder_tma_bf16_forward's tile %d x %d does not fit seq_len %zu, embed_dim %zu\n",
                ENCODER_TMA_BF16_TILE_ROWS, ENCODER_TMA_BF16_TILE_COLUMNS, encoded_tensor.size(1), encoded_tensor.size(2));
        exit(EXIT_FAILURE);
    }
}

/// Writes encoded = wte[token_ids] + wpe in bfloat16 at the settings given, or gives false if they do not fit.
///
///   encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor   as in encoder_tma_bf16_forward
///   blocks_val                                                 blocks a launch
///   tile_rows_val, tile_columns_val, stages_val                the tile and its sets
template <int VALUES, int SWIZZLE_BYTES, int WARPS>
bool encoder_tma_bf16_forward_at(const Tensor& encoded_tensor, const Tensor& token_ids_tensor, const Tensor& wte_tensor,
                                 const Tensor& wpe_tensor, int blocks_val, int tile_rows_val, int tile_columns_val, int stages_val) {
    // Checks, plans the tile, makes the three tensor maps, then launches encoder_tma_bf16_kernel.
    // 1. check the tensors, and that the tile fits these sizes
    encoder_bf16_check_shapes("encoder_tma_bf16_forward", encoded_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    int embed_dim_val = (int)encoded_tensor.size(2);
    EncoderTmaBf16Tile tile_val;
    size_t shared_bytes_val = 0;
    if (!encoder_tma_bf16_plan(&tile_val, &shared_bytes_val, SWIZZLE_BYTES, tile_rows_val, tile_columns_val, stages_val,
                               (int)encoded_tensor.size(0), (int)encoded_tensor.size(1), embed_dim_val))
        return false;
    // 2. the maps: wte one row a box, wpe and encoded tile_rows rows a box
    CUtensorMap wte_map_val = encoder_tma_bf16_tensor_map(wte_tensor.data_ptr<nv_bfloat16>(), wte_tensor.size(0), embed_dim_val,
                                                          tile_val.box_columns, 1, SWIZZLE_BYTES);
    CUtensorMap wpe_map_val = encoder_tma_bf16_tensor_map(wpe_tensor.data_ptr<nv_bfloat16>(), wpe_tensor.size(0), embed_dim_val,
                                                          tile_val.box_columns, tile_val.tile_rows, SWIZZLE_BYTES);
    CUtensorMap encoded_map_val = encoder_tma_bf16_tensor_map(encoded_tensor.data_ptr<nv_bfloat16>(), tile_val.tokens, embed_dim_val,
                                                              tile_val.box_columns, tile_val.tile_rows, SWIZZLE_BYTES);
    // 3. launch: blocks_val blocks, each walking its tiles; the kernel's shared memory raised once
    auto kernel_ptr = encoder_tma_bf16_kernel<VALUES, SWIZZLE_BYTES, WARPS>;
    static bool shared_set_val = false;
    if (!shared_set_val) {
        cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ENCODER_TMA_BF16_MAX_SHARED_BYTES));
        shared_set_val = true;
    }
    kernel_ptr<<<blocks_val, WARPS * HARDWARE_THREADS_PER_WARP, shared_bytes_val>>>(wte_map_val, wpe_map_val, encoded_map_val,
                                                                                    token_ids_tensor.data_ptr<int>(), tile_val);
    cudaCheck(cudaGetLastError());
    return true;
}

/// Walks this block's tiles: TMA brings each tile's rows, the threads add them, TMA writes the sums out.
///
///   wte_map_val, wpe_map_val, encoded_map_val   TMA's descriptions of wte, wpe and encoded (as {tokens, embed_dim})
///   token_ids_ptr                               {tokens} ints
///   tile_val                                    the tile's shape and the sizes
template <int VALUES, int SWIZZLE_BYTES, int WARPS>
__global__ void __launch_bounds__(WARPS * HARDWARE_THREADS_PER_WARP, 1)   // one block an SM is enough: all of its registers
encoder_tma_bf16_kernel(const __grid_constant__ CUtensorMap wte_map_val, const __grid_constant__ CUtensorMap wpe_map_val,
                        const __grid_constant__ CUtensorMap encoded_map_val, const int* token_ids_ptr, EncoderTmaBf16Tile tile_val) {
    // A tile is tile_rows tokens next to each other in one row of the batch, by tile_columns values of their rows. The
    // tiles are numbered token block by token block, column block fastest; block b takes tiles b, b + blocks, ...
    //
    //     a set of tiles in shared memory (one stage):
    //       weight region:    chunks of box_columns values, each tile_rows rows: wte's rows of the tile's tokens
    //       position region:  the same shape: wpe's rows of the tile's positions
    //
    // Warp 0 asks TMA for each set: one box a wte row and chunk (a gather: the ids are not next to each other), one
    // box of tile_rows rows a wpe chunk. Every thread then adds its cells (VALUES values, 16 bytes a cell, next to each
    // other in the tile's order), writing the sums over the weight region; warp 0 then has TMA store the weight
    // region into encoded, waits until TMA has read it, and asks for the tile STAGES on into the same set.
    //
    // SWIZZLE_BYTES 128: the boxes land in TMA's 128-byte swizzle, cell c of a 128-byte row r at slot c XOR (r % 8),
    // so a quarter warp reading cells of one column (VALUES 64: 8 cells a thread, a row each) hits 8 different banks.
    // 0: rows as they sit.
    constexpr int THREADS = WARPS * HARDWARE_THREADS_PER_WARP;
    constexpr int CELLS = VALUES / VALUES_PER_BF16_LOAD;                  // 16-byte cells a thread adds a round
    static_assert(VALUES % VALUES_PER_BF16_LOAD == 0, "a thread adds whole 16-byte cells");
    static_assert(SWIZZLE_BYTES == 0 || SWIZZLE_BYTES == HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW, "the swizzle: none or 128 bytes");
    extern __shared__ __align__(16) uint8_t encoder_tma_bf16_shared_list[];
    __shared__ uint64_t full_barriers_list[ENCODER_TMA_BF16_MAX_STAGES];   // set s's bytes are in
    // the sets start on a swizzle atom (1,024 bytes): the swizzle is worked out from the address's bits 7 to 9
    uint32_t shared_address_val = (uint32_t)__cvta_generic_to_shared(encoder_tma_bf16_shared_list);
    uint8_t* sets_ptr = encoder_tma_bf16_shared_list +
                        (HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM - shared_address_val % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM) % HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
    const int chunks_val = tile_val.tile_columns / tile_val.box_columns;
    const int region_bytes_val = chunks_val * tile_val.chunk_bytes;
    const int set_bytes_val = 2 * region_bytes_val;
    const int row_bytes_val = tile_val.box_columns * (int)sizeof(nv_bfloat16);
    const int cells_per_row_val = tile_val.box_columns / VALUES_PER_BF16_LOAD;
    const int tile_cells_val = tile_val.tile_rows * tile_val.tile_columns / VALUES_PER_BF16_LOAD;
    const int column_tiles_val = tile_val.embed_dim / tile_val.tile_columns;
    const int total_tiles_val = tile_val.tokens / tile_val.tile_rows * column_tiles_val;
    const uint32_t set_tx_bytes_val = 2u * tile_val.tile_rows * tile_val.tile_columns * (uint32_t)sizeof(nv_bfloat16);
    const int lane_index_val = threadIdx.x % HARDWARE_THREADS_PER_WARP;
    const bool asking_warp_val = threadIdx.x < HARDWARE_THREADS_PER_WARP;

    // 1. the barriers, by one thread, visible to TMA
    if (threadIdx.x == 0) {
        for (int set_index_val = 0; set_index_val < tile_val.stages; set_index_val++)
            encoder_tma_bf16_barrier_init(&full_barriers_list[set_index_val], 1);   // warp 0's lane 0, and the bytes
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // the barriers visible to TMA
    }
    __syncthreads();

    // gives a tile's first token and first column
    auto place_tile_val = [&](int tile_index_val, int& first_token_val, int& first_column_val) {
        first_token_val = tile_index_val / column_tiles_val * tile_val.tile_rows;
        first_column_val = tile_index_val % column_tiles_val * tile_val.tile_columns;
    };
    // warp 0's 32 lanes ask TMA for a tile's rows into set set_index_val: lane 0 first tells the set's barrier the
    // bytes coming, then lane l asks for the wte rows l, l + 32, ... (its ids read first, all at once) and the wpe
    // chunks l, l + 32, ...
    auto ask_tile_val = [&](int tile_index_val, int set_index_val) {
        int first_token_val, first_column_val;
        place_tile_val(tile_index_val, first_token_val, first_column_val);
        uint8_t* weight_ptr = sets_ptr + set_index_val * set_bytes_val;
        uint8_t* position_ptr = weight_ptr + region_bytes_val;
        int ids_list[ENCODER_TMA_BF16_MAX_ROWS_PER_LANE];
#pragma unroll
        for (int slot_index_val = 0; slot_index_val < ENCODER_TMA_BF16_MAX_ROWS_PER_LANE; slot_index_val++) {
            int row_index_val = lane_index_val + slot_index_val * HARDWARE_THREADS_PER_WARP;
            ids_list[slot_index_val] = row_index_val < tile_val.tile_rows ? token_ids_ptr[first_token_val + row_index_val] : 0;
        }
        if (lane_index_val == 0) encoder_tma_bf16_barrier_expect(&full_barriers_list[set_index_val], set_tx_bytes_val);
        __syncwarp();
#pragma unroll
        for (int slot_index_val = 0; slot_index_val < ENCODER_TMA_BF16_MAX_ROWS_PER_LANE; slot_index_val++) {
            int row_index_val = lane_index_val + slot_index_val * HARDWARE_THREADS_PER_WARP;
            if (row_index_val >= tile_val.tile_rows) break;
            for (int chunk_index_val = 0; chunk_index_val < chunks_val; chunk_index_val++)
                encoder_tma_bf16_copy_tile(reinterpret_cast<nv_bfloat16*>(weight_ptr + chunk_index_val * tile_val.chunk_bytes +
                                                                            row_index_val * row_bytes_val),
                                            &wte_map_val, first_column_val + chunk_index_val * tile_val.box_columns,
                                            ids_list[slot_index_val], &full_barriers_list[set_index_val]);
        }
        int first_position_val = first_token_val % tile_val.seq_len;
        for (int chunk_index_val = lane_index_val; chunk_index_val < chunks_val; chunk_index_val += HARDWARE_THREADS_PER_WARP)
            encoder_tma_bf16_copy_tile(reinterpret_cast<nv_bfloat16*>(position_ptr + chunk_index_val * tile_val.chunk_bytes),
                                        &wpe_map_val, first_column_val + chunk_index_val * tile_val.box_columns, first_position_val,
                                        &full_barriers_list[set_index_val]);
    };

    // 2. warp 0 asks for the first STAGES tiles
    int my_tiles_val = (int)blockIdx.x < total_tiles_val ? (total_tiles_val - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    if (asking_warp_val)
        for (int set_index_val = 0; set_index_val < tile_val.stages && set_index_val < my_tiles_val; set_index_val++)
            ask_tile_val(blockIdx.x + set_index_val * gridDim.x, set_index_val);

    // 3. each tile: wait for its set, add, store, ask for the tile STAGES on into the same set
    for (int walk_index_val = 0; walk_index_val < my_tiles_val; walk_index_val++) {
        int set_index_val = walk_index_val % tile_val.stages;
        int tile_index_val = blockIdx.x + walk_index_val * gridDim.x;
        encoder_tma_bf16_barrier_wait(&full_barriers_list[set_index_val], (uint32_t)((walk_index_val / tile_val.stages) % 2));
        uint8_t* weight_ptr = sets_ptr + set_index_val * set_bytes_val;
        uint8_t* position_ptr = weight_ptr + region_bytes_val;

        // a. the threads add: thread t takes cells t x CELLS .. + CELLS - 1, then THREADS x CELLS on; a cell's place
        //    walks chunk by chunk, row by row, cell by cell, as TMA laid them
        for (int first_cell_val = threadIdx.x * CELLS; first_cell_val < tile_cells_val; first_cell_val += THREADS * CELLS) {
            int cells_per_chunk_val = tile_val.tile_rows * cells_per_row_val;
            int chunk_index_val = first_cell_val / cells_per_chunk_val;
            int row_index_val = first_cell_val % cells_per_chunk_val / cells_per_row_val;
            int cell_index_val = first_cell_val % cells_per_row_val;
            int offsets_list[CELLS];
            float weight_list[CELLS][VALUES_PER_BF16_LOAD];
            float position_list[CELLS][VALUES_PER_BF16_LOAD];
#pragma unroll
            for (int cell_val = 0; cell_val < CELLS; cell_val++) {
                offsets_list[cell_val] = first_cell_val + cell_val < tile_cells_val
                                             ? encoder_tma_bf16_cell_offset<SWIZZLE_BYTES>(chunk_index_val, row_index_val, cell_index_val, tile_val)
                                             : -1;
                if (offsets_list[cell_val] >= 0) {
                    read8_as_float(weight_list[cell_val], reinterpret_cast<const nv_bfloat16*>(weight_ptr + offsets_list[cell_val]));
                    read8_as_float(position_list[cell_val], reinterpret_cast<const nv_bfloat16*>(position_ptr + offsets_list[cell_val]));
                }
                // the next cell: along the row, then the next row, then the next chunk
                if (++cell_index_val == cells_per_row_val) {
                    cell_index_val = 0;
                    if (++row_index_val == tile_val.tile_rows) {
                        row_index_val = 0;
                        chunk_index_val++;
                    }
                }
            }
#pragma unroll
            for (int cell_val = 0; cell_val < CELLS; cell_val++) {
                if (offsets_list[cell_val] < 0) continue;
#pragma unroll
                for (int value_index_val = 0; value_index_val < VALUES_PER_BF16_LOAD; value_index_val++)
                    weight_list[cell_val][value_index_val] += position_list[cell_val][value_index_val];
                write8_from_float(reinterpret_cast<nv_bfloat16*>(weight_ptr + offsets_list[cell_val]), weight_list[cell_val]);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // this thread's sums visible to TMA
        __syncthreads();                                                   // every thread's

        // b. warp 0: TMA stores the sums, reads them out of the set, then the tile STAGES on comes into it
        if (asking_warp_val) {
            int first_token_val, first_column_val;
            place_tile_val(tile_index_val, first_token_val, first_column_val);
            for (int chunk_index_val = lane_index_val; chunk_index_val < chunks_val; chunk_index_val += HARDWARE_THREADS_PER_WARP)
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                             ::"l"(&encoded_map_val), "r"(first_column_val + chunk_index_val * tile_val.box_columns), "r"(first_token_val),
                               "r"(encoder_tma_bf16_shared_address(weight_ptr + chunk_index_val * tile_val.chunk_bytes))
                             : "memory");
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
            if (walk_index_val + tile_val.stages < my_tiles_val) {
                asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");   // this lane's stores have read the set
                __syncwarp();                                                    // every lane's
                ask_tile_val(tile_index_val + tile_val.stages * gridDim.x, set_index_val);
            }
        }
    }
    // 4. the last stores written before the block ends
    if (asking_warp_val) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

/// Gives a 16-byte cell's byte offset in a region of a set, as TMA laid it.
///
///   chunk_index_val, row_index_val, cell_index_val   the cell: its chunk, its row in the chunk, its cell in the row
///   tile_val                                         the tile's shape
template <int SWIZZLE_BYTES>
__device__ __forceinline__ int encoder_tma_bf16_cell_offset(int chunk_index_val, int row_index_val, int cell_index_val,
                                                            const EncoderTmaBf16Tile& tile_val) {
    // Chunks are chunk_bytes apart (a multiple of the 1,024-byte atom with the swizzle), rows box_columns x 2 bytes,
    // cells 16. The 128-byte swizzle XORs the offset's bits 4 to 6 (the cell) with bits 7 to 9 (the row % 8).
    int offset_val = chunk_index_val * tile_val.chunk_bytes + row_index_val * tile_val.box_columns * (int)sizeof(nv_bfloat16) +
                     cell_index_val * HARDWARE_TMA_BYTES_PER_CELL;
    if constexpr (SWIZZLE_BYTES == HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW) {
        const int row_bits_val = HARDWARE_TMA_SWIZZLE_CELLS_PER_ATOM_COLUMN - 1;   // row % 8: 3 bits
        offset_val ^= ((offset_val / HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW) & row_bits_val) * HARDWARE_TMA_BYTES_PER_CELL;
    }
    return offset_val;
}

/// Gives a pointer into shared memory as the 32-bit address the PTX instructions take.
///
///   shared_ptr   a pointer into the block's shared memory
__device__ __forceinline__ uint32_t encoder_tma_bf16_shared_address(const void* shared_ptr) {
    return (uint32_t)__cvta_generic_to_shared(shared_ptr);
}

/// Makes an mbarrier that completes a phase after arrivals_val arrivals (and the bytes it is told to expect).
///
///   barrier_ptr    the barrier, 8 bytes of shared memory
///   arrivals_val   arrivals a phase
__device__ __forceinline__ void encoder_tma_bf16_barrier_init(uint64_t* barrier_ptr, uint32_t arrivals_val) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(encoder_tma_bf16_shared_address(barrier_ptr)), "r"(arrivals_val) : "memory");
}

/// Arrives once at a barrier and tells it bytes_val bytes of TMA copies are coming.
///
///   barrier_ptr   the barrier
///   bytes_val     the bytes the copies asked for next bring
__device__ __forceinline__ void encoder_tma_bf16_barrier_expect(uint64_t* barrier_ptr, uint32_t bytes_val) {
    // Its phase completes when they are in.
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(encoder_tma_bf16_shared_address(barrier_ptr)), "r"(bytes_val) : "memory");
}

/// Waits until a barrier's phase of this parity is complete.
///
///   barrier_ptr   the barrier
///   parity_val    the phase waited for, 0 or 1
__device__ __forceinline__ void encoder_tma_bf16_barrier_wait(uint64_t* barrier_ptr, uint32_t parity_val) {
    // The caller keeps count of the phases (the first is 0, the next 1, then 0 again).
    uint32_t barrier_address_val = encoder_tma_bf16_shared_address(barrier_ptr);
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
///   tile_ptr               where the box goes, in shared memory
///   map_ptr                TMA's description of the matrix
///   column_val, row_val    the box's corner, as the matrix sits
///   barrier_ptr            the barrier that expects the bytes
__device__ __forceinline__ void encoder_tma_bf16_copy_tile(nv_bfloat16* tile_ptr, const CUtensorMap* map_ptr, int column_val,
                                                           int row_val, uint64_t* barrier_ptr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(encoder_tma_bf16_shared_address(tile_ptr)), "l"(map_ptr), "r"(column_val), "r"(row_val),
                   "r"(encoder_tma_bf16_shared_address(barrier_ptr))
                 : "memory");
}

// ── for it ────────────────────────────────────────────────────────────────────

/// Works out a launch's tile and its shared memory, and tells if it fits.
///
///   tile_ptr                                         written: the tile
///   shared_bytes_ptr                                 written: the dynamic shared memory a block asks for
///   swizzle_bytes_val                                128 or 0
///   tile_rows_val, tile_columns_val, stages_val      the settings
///   batch_size_val, seq_len_val, embed_dim_val       encoded's shape
inline bool encoder_tma_bf16_plan(EncoderTmaBf16Tile* tile_ptr, size_t* shared_bytes_ptr, int swizzle_bytes_val, int tile_rows_val,
                                  int tile_columns_val, int stages_val, int batch_size_val, int seq_len_val, int embed_dim_val) {
    // A box's row: 64 values (128 bytes) with the swizzle; with none, the tile's columns if at most
    // HARDWARE_TMA_MAX_VALUES_PER_BOX_ROW, else the largest of 256, 128, 64 dividing them. A wte row is its own box,
    // and a box starts on HARDWARE_TMA_SHARED_ALIGN_BYTES, so a box's row is a multiple of 128 bytes. Fits: tile_rows
    // divides seq_len (a tile's positions are next to each other in wpe) and is at most a box's rows, tile_columns
    // divides embed_dim, stages within ENCODER_TMA_BF16_MAX_STAGES, the sets within a block's shared memory.
    // 1. the box's row
    const int value_bytes_val = (int)sizeof(nv_bfloat16);
    int box_columns_val = 0;
    if (swizzle_bytes_val == HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW) {
        box_columns_val = HARDWARE_TMA_SWIZZLE_BYTES_PER_ROW / value_bytes_val;
    } else if (swizzle_bytes_val == 0 && tile_columns_val <= HARDWARE_TMA_MAX_VALUES_PER_BOX_ROW) {
        box_columns_val = tile_columns_val;
    } else if (swizzle_bytes_val == 0) {
        const int narrowest_val = HARDWARE_TMA_SHARED_ALIGN_BYTES / value_bytes_val;   // 64 values: 128 bytes
        for (int width_val = HARDWARE_TMA_MAX_VALUES_PER_BOX_ROW; width_val >= narrowest_val && box_columns_val == 0; width_val /= 2)
            if (tile_columns_val % width_val == 0) box_columns_val = width_val;
        if (box_columns_val == 0) return false;
    } else {
        return false;
    }
    // 2. the fits
    if (tile_rows_val < 1 || tile_rows_val > HARDWARE_TMA_MAX_ROWS_PER_BOX || seq_len_val % tile_rows_val != 0 ||
        tile_columns_val < box_columns_val || tile_columns_val % box_columns_val != 0 || embed_dim_val % tile_columns_val != 0 ||
        box_columns_val * value_bytes_val % HARDWARE_TMA_SHARED_ALIGN_BYTES != 0 || stages_val < 1 ||
        stages_val > ENCODER_TMA_BF16_MAX_STAGES)
        return false;
    // 3. shared memory: chunks on their alignment (the swizzle's atom, or a box's), two regions a set, the slack to
    //    start the sets on an atom
    int align_val = swizzle_bytes_val != 0 ? HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM : HARDWARE_TMA_SHARED_ALIGN_BYTES;
    int chunk_bytes_val = CEIL_DIV(tile_rows_val * box_columns_val * value_bytes_val, align_val) * align_val;
    size_t shared_bytes_val = (size_t)stages_val * 2 * (tile_columns_val / box_columns_val) * chunk_bytes_val +
                              HARDWARE_TMA_SWIZZLE_BYTES_PER_ATOM;
    if (shared_bytes_val > (size_t)ENCODER_TMA_BF16_MAX_SHARED_BYTES) return false;
    *tile_ptr = EncoderTmaBf16Tile{tile_rows_val, tile_columns_val, box_columns_val, chunk_bytes_val, stages_val, seq_len_val,
                                   embed_dim_val, batch_size_val * seq_len_val};
    *shared_bytes_ptr = shared_bytes_val;
    return true;
}

/// Makes TMA's description of a bfloat16 matrix, with boxes of box_rows rows of box_columns values.
///
///   matrix_ptr                               the matrix in GPU memory, on 16 bytes
///   matrix_rows_val, matrix_columns_val      its shape, rows of matrix_columns_val values
///   box_columns_val, box_rows_val            a box
///   swizzle_bytes_val                        128 or 0
inline CUtensorMap encoder_tma_bf16_tensor_map(const nv_bfloat16* matrix_ptr, size_t matrix_rows_val, size_t matrix_columns_val,
                                               int box_columns_val, int box_rows_val, int swizzle_bytes_val) {
    // The driver's cuTensorMapEncodeTiled, found at run time; past the edges: 0s.
    static PFN_cuTensorMapEncodeTiled_v12000 encode_ptr = [] {
        void* function_ptr = nullptr;
        cudaDriverEntryPointQueryResult status_val;
        cudaCheck(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function_ptr, 12000, cudaEnableDefault, &status_val));
        if (status_val != cudaDriverEntryPointSuccess || function_ptr == nullptr) {
            fprintf(stderr, "error: encoder_tma_bf16 could not find the driver's cuTensorMapEncodeTiled (a driver for CUDA 12 or later)\n");
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function_ptr);
    }();
    CUtensorMap map_val;
    cuuint64_t sizes_list[2] = {matrix_columns_val, matrix_rows_val};
    cuuint64_t row_bytes_list[1] = {matrix_columns_val * sizeof(nv_bfloat16)};
    cuuint32_t box_list[2] = {(cuuint32_t)box_columns_val, (cuuint32_t)box_rows_val};
    cuuint32_t element_steps_list[2] = {1, 1};
    CUresult result_val = encode_ptr(&map_val, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bfloat16*>(matrix_ptr), sizes_list,
                                     row_bytes_list, box_list, element_steps_list, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     swizzle_bytes_val != 0 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                     CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result_val != CUDA_SUCCESS) {
        fprintf(stderr, "error: cuTensorMapEncodeTiled gave %d for a {%zu, %zu} matrix, box {%d, %d}, swizzle %d\n", (int)result_val,
                matrix_rows_val, matrix_columns_val, box_rows_val, box_columns_val, swizzle_bytes_val);
        exit(EXIT_FAILURE);
    }
    return map_val;
}

#endif // ENCODER_TMA_BF16_CUH
