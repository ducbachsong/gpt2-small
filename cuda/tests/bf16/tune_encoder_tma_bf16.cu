// tune_encoder_tma_bf16.cu — the timings of cuda/kernels/bf16/encoder_tma_bf16.cuh: every setting, father to son, against encoder_bf16_forward.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include <algorithm>                  // std::sort
#include <chrono>                     // the sweep's progress
#include <string>
#include <vector>
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../tune_helpers.h"            // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gpu_folder_name
#include "../../kernels/bf16/encoder_bf16.cuh"
#include "../../kernels/bf16/encoder_tma_bf16.cuh"

/// An encoder's sizes.
struct EncoderSize
{
    const char *name_ptr;
    size_t batch_size, seq_len, embed_dim;
    int vocab_size;
};
static const EncoderSize D24_SIZE = {"nanochat d24", 16, 2048, 1536, 32768};   // the trainer's: 16 rows of 2,048 tokens
static const EncoderSize GPT2_SIZE = {"GPT-2 small", 4, 1024, 768, 50304};     // tune_encoder_bf16's

// The sweep, father to son: blocks, warps, values a thread, the tile (rows, columns, swizzle), sets of tiles
static constexpr int BLOCKS_PER_SM_LIST[] = {1, 2, 4, 8, 16};
#define WARPS_PACK 1, 2, 4, 8, 16, 32
static constexpr int WARPS_LIST[] = {WARPS_PACK};
static constexpr int VALUES_LIST[] = {8, 16, 32, 64, 128};
static constexpr int TILE_ROWS_LIST[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
static constexpr int TILE_COLUMNS_LIST[] = {64, 128, 256, 384, 768, 1536, 3072, 5120, 6144, 7680, 15360};
static constexpr int SWIZZLE_LIST[] = {128, 0};
static constexpr int STAGES_LIST[] = {1, 2, 3, 4};
static constexpr int BASELINE_BLOCKS_PER_SM_LIST[] = {1, 2, 4, 8, 16, 32};   // encoder_bf16_forward's launches
static constexpr int BASELINE_THREADS_LIST[] = {128, 256, 512, 1024};
#define COUNT(list) ((int)(sizeof(list) / sizeof((list)[0])))
#define FIRST_PASS_RUNS 3             // timed runs of each setting in the first pass, averaged
#define SLOW_CUT 20.0f                // a setting this many times the current kernel's best: timed once in the first pass
#define FINALISTS 100                 // the first pass's fastest, timed again over SWEEP_ROUNDS rounds taking turns
#define TOP_PRINTED 20                // finalists printed
#define PROGRESS_EVERY 2000           // settings between progress lines
#define MEGA 1e6

using EncoderTmaKernel = void (*)(const CUtensorMap, const CUtensorMap, const CUtensorMap, const int *, EncoderTmaBf16Tile);
static EncoderTmaKernel kernels_list[COUNT(VALUES_LIST)][COUNT(SWIZZLE_LIST)][COUNT(WARPS_LIST)];

/// One kernel's registers, and whether the sweep uses it.
struct KernelInfo
{
    int registers;
    size_t local_bytes;               // > 0: registers spilled to local memory
    bool usable;                      // no spill here, nor at fewer values a thread
};
static KernelInfo infos_list[COUNT(VALUES_LIST)][COUNT(SWIZZLE_LIST)][COUNT(WARPS_LIST)];

/// One setting of the sweep, its tile and maps, and its times.
struct Setting
{
    alignas(64) CUtensorMap maps_list[3];   // wte, wpe, encoded
    int blocks_per_sm_val, warps_index_val, values_index_val, tile_rows_val, tile_columns_val, swizzle_index_val, stages_val;
    EncoderTmaBf16Tile tile_val;
    size_t shared_bytes_val;
    float first_ms_val = 0.0f;
    int first_runs_val = 0;
    bool same_val = false;
    float final_ms_val = 0.0f;
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes an int tensor {batch_size_val, seq_len_val} of random token ids below vocab_size_val on the GPU.
///
///   batch_size_val, seq_len_val   its shape
///   vocab_size_val                the ids' bound
static Tensor random_token_ids(size_t batch_size_val, size_t seq_len_val, int vocab_size_val)
{
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, vocab_size_val - 1);
    std::vector<int> ids_list(batch_size_val * seq_len_val);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    Tensor tensor = Tensor::empty({batch_size_val, seq_len_val}, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

/// Tells if two bfloat16 tensors hold the same bits.
///
///   first_tensor, second_tensor   the same shape, on the GPU
static bool same_bits(const Tensor &first_tensor, const Tensor &second_tensor)
{
    torch::TensorOptions shorts_val = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);
    torch::Tensor first_val = torch::from_blob(first_tensor.data_ptr(), {(int64_t)first_tensor.numel()}, shorts_val);
    torch::Tensor second_val = torch::from_blob(second_tensor.data_ptr(), {(int64_t)second_tensor.numel()}, shorts_val);
    return torch::equal(first_val, second_val);
}

/// The encoder's tensors at one size, and encoder_bf16_forward's result on them.
struct EncoderData
{
    EncoderSize size_val;
    Tensor wte_tensor, wpe_tensor, token_ids_tensor, expected_tensor, encoded_tensor;

    /// Makes the tensors and the expected result.
    ///
    ///   size_val   the sizes
    explicit EncoderData(const EncoderSize &size_val) : size_val(size_val)
    {
        torch::manual_seed(0);
        wte_tensor = random_bf16_tensor({(size_t)size_val.vocab_size, size_val.embed_dim});
        wpe_tensor = random_bf16_tensor({size_val.seq_len, size_val.embed_dim});
        token_ids_tensor = random_token_ids(size_val.batch_size, size_val.seq_len, size_val.vocab_size);
        expected_tensor = Tensor::zeros({size_val.batch_size, size_val.seq_len, size_val.embed_dim}, DType::BFloat16);
        encoded_tensor = Tensor::zeros({size_val.batch_size, size_val.seq_len, size_val.embed_dim}, DType::BFloat16);
        encoder_bf16_forward(expected_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    }

    /// Gives the bytes the forward asks for: wte's and wpe's rows read, encoded written, the ids.
    double bytes(void) const
    {
        double tokens_val = (double)size_val.batch_size * size_val.seq_len;
        return 3.0 * tokens_val * size_val.embed_dim * sizeof(nv_bfloat16) + tokens_val * sizeof(int);
    }
};

/// Fills one row of kernels_list: one value count and swizzle, every warp count.
///
///   row_list   kernels_list[values][swizzle]
template <int VALUES, int SWIZZLE, int... WARPS>
static void fill_warps(EncoderTmaKernel (&row_list)[sizeof...(WARPS)])
{
    int index_val = 0;
    ((row_list[index_val++] = encoder_tma_bf16_kernel<VALUES, SWIZZLE, WARPS>), ...);
}

/// Fills kernels_list, every kernel the sweep can launch, and reads each one's registers into infos_list.
static void fill_kernels(void)
{
    // A kernel is usable if it spills no register, nor does the same kernel with fewer values a thread: for each warp
    // count and swizzle, the values go up only to the most that do not spill.
    static_assert(COUNT(VALUES_LIST) == 5 && COUNT(SWIZZLE_LIST) == 2, "one line a value count and swizzle below");
    fill_warps<8, 128, WARPS_PACK>(kernels_list[0][0]);
    fill_warps<8, 0, WARPS_PACK>(kernels_list[0][1]);
    fill_warps<16, 128, WARPS_PACK>(kernels_list[1][0]);
    fill_warps<16, 0, WARPS_PACK>(kernels_list[1][1]);
    fill_warps<32, 128, WARPS_PACK>(kernels_list[2][0]);
    fill_warps<32, 0, WARPS_PACK>(kernels_list[2][1]);
    fill_warps<64, 128, WARPS_PACK>(kernels_list[3][0]);
    fill_warps<64, 0, WARPS_PACK>(kernels_list[3][1]);
    fill_warps<128, 128, WARPS_PACK>(kernels_list[4][0]);
    fill_warps<128, 0, WARPS_PACK>(kernels_list[4][1]);
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
        {
            bool spilled_val = false;
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                EncoderTmaKernel kernel_ptr = kernels_list[values_index_val][swizzle_index_val][warps_index_val];
                cudaCheck(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, ENCODER_TMA_BF16_MAX_SHARED_BYTES));
                cudaFuncAttributes attributes_val;
                cudaCheck(cudaFuncGetAttributes(&attributes_val, kernel_ptr));
                spilled_val = spilled_val || attributes_val.localSizeBytes > 0;
                infos_list[values_index_val][swizzle_index_val][warps_index_val] = {attributes_val.numRegs, attributes_val.localSizeBytes,
                                                                                    !spilled_val};
            }
        }
}

/// Launches one setting.
///
///   setting_val   the setting
///   data_val      the tensors
///   sms_val       this GPU's SMs
static void launch_setting(const Setting &setting_val, const EncoderData &data_val, int sms_val)
{
    EncoderTmaKernel kernel_ptr = kernels_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val];
    kernel_ptr<<<sms_val * setting_val.blocks_per_sm_val, WARPS_LIST[setting_val.warps_index_val] * HARDWARE_THREADS_PER_WARP,
                 setting_val.shared_bytes_val>>>(setting_val.maps_list[0], setting_val.maps_list[1], setting_val.maps_list[2],
                                                 data_val.token_ids_tensor.data_ptr<int>(), setting_val.tile_val);
    cudaCheck(cudaGetLastError());
}

/// Gives a setting as one line: its warps, values, tile, swizzle, sets, registers and shared memory.
///
///   setting_val   the setting
static std::string describe(const Setting &setting_val)
{
    char line_list[160];
    snprintf(line_list, sizeof(line_list), "%2d warps, %3d values, tile %3d x %4d, swizzle %3d, %d sets  (%3d registers, %6zu B shared)",
             WARPS_LIST[setting_val.warps_index_val], VALUES_LIST[setting_val.values_index_val], setting_val.tile_rows_val,
             setting_val.tile_columns_val, SWIZZLE_LIST[setting_val.swizzle_index_val], setting_val.stages_val,
             infos_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val].registers,
             setting_val.shared_bytes_val);
    return line_list;
}

// ══ the tune ══════════════════════════════════════════════════════════════════

/// Times encoder_bf16_forward_kernel at every launch of the baseline lists, and gives its fastest.
///
///   data_val                   the tensors
///   sms_val                    this GPU's SMs
///   best_blocks_ptr, best_threads_ptr   written: its fastest launch
static float time_the_current_kernel(const EncoderData &data_val, int sms_val, int *best_blocks_ptr, int *best_threads_ptr)
{
    // SWEEP_ROUNDS rounds, each timing every launch once.
    const int launches_val = COUNT(BASELINE_BLOCKS_PER_SM_LIST) * COUNT(BASELINE_THREADS_LIST);
    std::vector<float> total_ms_list(launches_val, 0.0f);
    auto launch_val = [&](int launch_index_val) {
        int blocks_val = sms_val * BASELINE_BLOCKS_PER_SM_LIST[launch_index_val / COUNT(BASELINE_THREADS_LIST)];
        int threads_val = BASELINE_THREADS_LIST[launch_index_val % COUNT(BASELINE_THREADS_LIST)];
        encoder_bf16_forward_kernel<<<blocks_val, threads_val>>>(
            data_val.encoded_tensor.data_ptr<nv_bfloat16>(), data_val.token_ids_tensor.data_ptr<int>(),
            data_val.wte_tensor.data_ptr<nv_bfloat16>(), data_val.wpe_tensor.data_ptr<nv_bfloat16>(), data_val.size_val.batch_size,
            data_val.size_val.seq_len, data_val.size_val.embed_dim);
        cudaCheck(cudaGetLastError());
    };
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
        for (int launch_index_val = 0; launch_index_val < launches_val; launch_index_val++)
            total_ms_list[launch_index_val] += gpu_milliseconds([&] { launch_val(launch_index_val); });
    int best_index_val = (int)(std::min_element(total_ms_list.begin(), total_ms_list.end()) - total_ms_list.begin());
    *best_blocks_ptr = sms_val * BASELINE_BLOCKS_PER_SM_LIST[best_index_val / COUNT(BASELINE_THREADS_LIST)];
    *best_threads_ptr = BASELINE_THREADS_LIST[best_index_val % COUNT(BASELINE_THREADS_LIST)];
    return total_ms_list[best_index_val] / SWEEP_ROUNDS;
}

/// Sweeps every setting, father to son, and prints the tree of the fastest, the finalists, and every setting in a .csv.
///
///   size_val   the encoder's sizes
///   csv_path   where every setting's row goes
static void time_every_setting(const EncoderSize &size_val, const std::string &csv_path)
{
    // Father to son: blocks a launch (x this GPU's SMs), warps a block, values a thread, then the tile: its rows, its
    // columns, the swizzle, the sets in shared memory. A setting is left out if its tile does not fit the sizes or
    // shared memory, its kernel spills (or one with fewer values a thread does), or no block of it fits on an SM.
    //
    // First pass, in that order: each setting's result checked against encoder_bf16_forward's, to the last bit, then
    // timed FIRST_PASS_RUNS times (once if slower than SLOW_CUT x the current kernel's best). Second pass: the
    // FINALISTS fastest and the current kernel's best launch, timed over SWEEP_ROUNDS rounds taking turns.
    //
    // Expect: every setting gives the same bits.

    // 1. the tensors, the kernels, the current kernel's fastest launch
    EncoderData data_val(size_val);
    fill_kernels();
    int sms_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    warm_up_gpu([&] { encoder_bf16_forward(data_val.encoded_tensor, data_val.token_ids_tensor, data_val.wte_tensor, data_val.wpe_tensor); });
    int baseline_blocks_val = 0, baseline_threads_val = 0;
    float baseline_ms_val = time_the_current_kernel(data_val, sms_val, &baseline_blocks_val, &baseline_threads_val);

    // 2. every setting, father to son; the ones that cannot run left out, counted by why
    std::vector<Setting> settings_list;
    long no_fit_val = 0, spilled_val = 0, no_room_val = 0;
    for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
        for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
                for (int tile_rows_val : TILE_ROWS_LIST)
                    for (int tile_columns_val : TILE_COLUMNS_LIST)
                        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
                            for (int stages_val : STAGES_LIST)
                            {
                                Setting setting_val;
                                setting_val.blocks_per_sm_val = blocks_per_sm_val;
                                setting_val.warps_index_val = warps_index_val;
                                setting_val.values_index_val = values_index_val;
                                setting_val.tile_rows_val = tile_rows_val;
                                setting_val.tile_columns_val = tile_columns_val;
                                setting_val.swizzle_index_val = swizzle_index_val;
                                setting_val.stages_val = stages_val;
                                int swizzle_val = SWIZZLE_LIST[swizzle_index_val];
                                if (!encoder_tma_bf16_plan(&setting_val.tile_val, &setting_val.shared_bytes_val, swizzle_val, tile_rows_val,
                                                           tile_columns_val, stages_val, (int)size_val.batch_size, (int)size_val.seq_len,
                                                           (int)size_val.embed_dim))
                                {
                                    no_fit_val++;
                                    continue;
                                }
                                if (!infos_list[values_index_val][swizzle_index_val][warps_index_val].usable)
                                {
                                    spilled_val++;
                                    continue;
                                }
                                int blocks_on_sm_val = 0;
                                cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                    &blocks_on_sm_val, kernels_list[values_index_val][swizzle_index_val][warps_index_val],
                                    WARPS_LIST[warps_index_val] * HARDWARE_THREADS_PER_WARP, setting_val.shared_bytes_val));
                                if (blocks_on_sm_val == 0)
                                {
                                    no_room_val++;
                                    continue;
                                }
                                int box_columns_val = setting_val.tile_val.box_columns;
                                setting_val.maps_list[0] = encoder_tma_bf16_tensor_map(data_val.wte_tensor.data_ptr<nv_bfloat16>(), size_val.vocab_size,
                                                                                       size_val.embed_dim, box_columns_val, 1, swizzle_val);
                                setting_val.maps_list[1] = encoder_tma_bf16_tensor_map(data_val.wpe_tensor.data_ptr<nv_bfloat16>(), size_val.seq_len,
                                                                                       size_val.embed_dim, box_columns_val, tile_rows_val, swizzle_val);
                                setting_val.maps_list[2] = encoder_tma_bf16_tensor_map(data_val.encoded_tensor.data_ptr<nv_bfloat16>(),
                                                                                       setting_val.tile_val.tokens, size_val.embed_dim,
                                                                                       box_columns_val, tile_rows_val, swizzle_val);
                                settings_list.push_back(setting_val);
                            }
    printf("    %zu settings to run; left out: %ld tiles that do not fit, %ld kernels that spill, %ld with no room on an SM\n",
           settings_list.size(), no_fit_val, spilled_val, no_room_val);

    // 3. first pass: check, then time
    auto start_val = std::chrono::steady_clock::now();
    for (size_t setting_index_val = 0; setting_index_val < settings_list.size(); setting_index_val++)
    {
        Setting &setting_val = settings_list[setting_index_val];
        data_val.encoded_tensor.zero_();
        launch_setting(setting_val, data_val, sms_val);
        cudaCheck(cudaDeviceSynchronize());
        setting_val.same_val = same_bits(data_val.encoded_tensor, data_val.expected_tensor);
        float ms_val = gpu_milliseconds([&] { launch_setting(setting_val, data_val, sms_val); });
        setting_val.first_runs_val = 1;
        if (ms_val < SLOW_CUT * baseline_ms_val)
            for (; setting_val.first_runs_val < FIRST_PASS_RUNS; setting_val.first_runs_val++)
                ms_val += gpu_milliseconds([&] { launch_setting(setting_val, data_val, sms_val); });
        setting_val.first_ms_val = ms_val / setting_val.first_runs_val;
        if ((setting_index_val + 1) % PROGRESS_EVERY == 0)
            printf("    ... %zu of %zu settings, %.0f s\n", setting_index_val + 1, settings_list.size(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - start_val).count());
    }

    // 4. second pass: the finalists and the current kernel's best, taking turns
    std::vector<int> order_list(settings_list.size());
    for (size_t index_val = 0; index_val < order_list.size(); index_val++)
        order_list[index_val] = (int)index_val;
    std::sort(order_list.begin(), order_list.end(), [&](int first_val, int second_val) {
        return settings_list[first_val].first_ms_val < settings_list[second_val].first_ms_val;
    });
    int finalists_val = std::min((int)order_list.size(), FINALISTS);
    std::vector<float> total_ms_list(finalists_val, 0.0f);
    float baseline_total_ms_val = 0.0f;
    for (int round_index_val = 0; round_index_val < SWEEP_ROUNDS; round_index_val++)
    {
        baseline_total_ms_val += gpu_milliseconds([&] {
            encoder_bf16_forward_kernel<<<baseline_blocks_val, baseline_threads_val>>>(
                data_val.encoded_tensor.data_ptr<nv_bfloat16>(), data_val.token_ids_tensor.data_ptr<int>(),
                data_val.wte_tensor.data_ptr<nv_bfloat16>(), data_val.wpe_tensor.data_ptr<nv_bfloat16>(), size_val.batch_size,
                size_val.seq_len, size_val.embed_dim);
        });
        for (int finalist_index_val = 0; finalist_index_val < finalists_val; finalist_index_val++)
            total_ms_list[finalist_index_val] +=
                gpu_milliseconds([&] { launch_setting(settings_list[order_list[finalist_index_val]], data_val, sms_val); });
    }
    for (int finalist_index_val = 0; finalist_index_val < finalists_val; finalist_index_val++)
        settings_list[order_list[finalist_index_val]].final_ms_val = total_ms_list[finalist_index_val] / SWEEP_ROUNDS;
    float baseline_final_ms_val = baseline_total_ms_val / SWEEP_ROUNDS;

    // 5. the registers of each kernel
    double bytes_val = data_val.bytes();
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    %s: %zu rows of %zu tokens, embed_dim %zu, vocab %d: %.1f MB; this GPU's peak %.0f GB/s\n", size_val.name_ptr,
           size_val.batch_size, size_val.seq_len, size_val.embed_dim, size_val.vocab_size, bytes_val / MEGA, peak_val);
    printf("    registers a thread (S: spills, so left out with every larger value count)\n");
    printf("    warps  swizzle ");
    for (int values_val : VALUES_LIST)
        printf("  %3d values", values_val);
    printf("\n");
    for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        for (int swizzle_index_val = 0; swizzle_index_val < COUNT(SWIZZLE_LIST); swizzle_index_val++)
        {
            printf("    %5d  %7d ", WARPS_LIST[warps_index_val], SWIZZLE_LIST[swizzle_index_val]);
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                const KernelInfo &info_val = infos_list[values_index_val][swizzle_index_val][warps_index_val];
                printf("  %8d %c", info_val.registers, info_val.local_bytes > 0 ? 'S' : ' ');
            }
            printf("\n");
        }

    // 6. the tree: the fastest setting under each blocks, each blocks x warps, each blocks x warps x values (first pass)
    printf("    the current kernel: fastest %d blocks x %d threads, %.4f ms, %.0f GB/s\n", baseline_blocks_val, baseline_threads_val,
           baseline_ms_val, gigabytes_per_second(bytes_val, baseline_ms_val));
    printf("    the tree, first pass: the fastest under each father (ms, GB/s, x the current kernel's best)\n");
    auto fastest_val = [&](auto matches_val) {
        int best_val = -1;
        for (size_t index_val = 0; index_val < settings_list.size(); index_val++)
            if (matches_val(settings_list[index_val]) && (best_val < 0 || settings_list[index_val].first_ms_val < settings_list[best_val].first_ms_val))
                best_val = (int)index_val;
        return best_val;
    };
    auto print_node_val = [&](int depth_val, const char *label_ptr, int best_val) {
        if (best_val < 0)
        {
            printf("    %*s%-18s nothing runs\n", 4 * depth_val, "", label_ptr);
            return;
        }
        const Setting &setting_val = settings_list[best_val];
        printf("    %*s%-18s %8.4f  %5.0f  %5.2fx   %s\n", 4 * depth_val, "", label_ptr, setting_val.first_ms_val,
               gigabytes_per_second(bytes_val, setting_val.first_ms_val), baseline_ms_val / setting_val.first_ms_val,
               describe(setting_val).c_str());
    };
    char label_list[32];
    for (int blocks_per_sm_val : BLOCKS_PER_SM_LIST)
    {
        snprintf(label_list, sizeof(label_list), "blocks %d (%dx SMs)", sms_val * blocks_per_sm_val, blocks_per_sm_val);
        print_node_val(0, label_list, fastest_val([&](const Setting &s) { return s.blocks_per_sm_val == blocks_per_sm_val; }));
        for (int warps_index_val = 0; warps_index_val < COUNT(WARPS_LIST); warps_index_val++)
        {
            snprintf(label_list, sizeof(label_list), "warps %d", WARPS_LIST[warps_index_val]);
            print_node_val(1, label_list, fastest_val([&](const Setting &s) {
                               return s.blocks_per_sm_val == blocks_per_sm_val && s.warps_index_val == warps_index_val;
                           }));
            for (int values_index_val = 0; values_index_val < COUNT(VALUES_LIST); values_index_val++)
            {
                snprintf(label_list, sizeof(label_list), "values %d", VALUES_LIST[values_index_val]);
                print_node_val(2, label_list, fastest_val([&](const Setting &s) {
                                   return s.blocks_per_sm_val == blocks_per_sm_val && s.warps_index_val == warps_index_val &&
                                          s.values_index_val == values_index_val;
                               }));
            }
        }
    }

    // 7. the finalists, timed again taking turns with the current kernel's best
    printf("    the %d fastest, timed again over %d rounds with the current kernel (%d x %d: %.4f ms, %.0f GB/s, MBU %.0f%%)\n",
           TOP_PRINTED, SWEEP_ROUNDS, baseline_blocks_val, baseline_threads_val, baseline_final_ms_val,
           gigabytes_per_second(bytes_val, baseline_final_ms_val), 100.0 * gigabytes_per_second(bytes_val, baseline_final_ms_val) / peak_val);
    std::vector<int> final_order_list(order_list.begin(), order_list.begin() + finalists_val);
    std::sort(final_order_list.begin(), final_order_list.end(),
              [&](int first_val, int second_val) { return settings_list[first_val].final_ms_val < settings_list[second_val].final_ms_val; });
    printf("         ms    GB/s   MBU   x current   blocks   setting\n");
    for (int rank_val = 0; rank_val < std::min(TOP_PRINTED, finalists_val); rank_val++)
    {
        const Setting &setting_val = settings_list[final_order_list[rank_val]];
        double speed_val = gigabytes_per_second(bytes_val, setting_val.final_ms_val);
        printf("    %7.4f  %6.0f  %3.0f%%     %5.2fx   %6d   %s\n", setting_val.final_ms_val, speed_val, 100.0 * speed_val / peak_val,
               baseline_final_ms_val / setting_val.final_ms_val, sms_val * setting_val.blocks_per_sm_val, describe(setting_val).c_str());
    }

    // 8. every setting into the .csv
    FILE *csv_ptr = fopen(csv_path.c_str(), "w");
    if (csv_ptr != nullptr)
    {
        fprintf(csv_ptr, "blocks,warps,values,tile_rows,tile_columns,swizzle,stages,registers,shared_bytes,first_ms,first_runs,"
                         "final_ms,gb_per_s,same\n");
        for (const Setting &setting_val : settings_list)
            fprintf(csv_ptr, "%d,%d,%d,%d,%d,%d,%d,%d,%zu,%.5f,%d,%.5f,%.0f,%d\n", sms_val * setting_val.blocks_per_sm_val,
                    WARPS_LIST[setting_val.warps_index_val], VALUES_LIST[setting_val.values_index_val], setting_val.tile_rows_val,
                    setting_val.tile_columns_val, SWIZZLE_LIST[setting_val.swizzle_index_val], setting_val.stages_val,
                    infos_list[setting_val.values_index_val][setting_val.swizzle_index_val][setting_val.warps_index_val].registers,
                    setting_val.shared_bytes_val, setting_val.first_ms_val, setting_val.first_runs_val, setting_val.final_ms_val,
                    gigabytes_per_second(bytes_val, setting_val.first_ms_val), setting_val.same_val ? 1 : 0);
        fclose(csv_ptr);
        printf("    every setting: %s\n", csv_path.c_str());
    }
    else
    {
        printf("    could not write %s\n", csv_path.c_str());
    }

    // 9. every setting gave encoder_bf16_forward's bits
    long same_count_val = (long)std::count_if(settings_list.begin(), settings_list.end(), [](const Setting &s) { return s.same_val; });
    ASSERT_EQ(same_count_val, (long)settings_list.size());
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    // ./tune_encoder_tma_bf16: nanochat d24's size; gpt2: GPT-2 small's; x<N>: d24's with embed_dim x N (x4, x10).
    std::string size_name_val = argc > 1 ? argv[1] : "";
    EncoderSize size_val = size_name_val == "gpt2" ? GPT2_SIZE : D24_SIZE;
    std::string name_list;
    if (size_name_val.size() > 1 && size_name_val[0] == 'x')
    {
        int times_val = std::stoi(size_name_val.substr(1));
        size_val.embed_dim *= times_val;
        name_list = "nanochat d24, embed_dim x " + std::to_string(times_val);
        size_val.name_ptr = name_list.c_str();
    }
    std::string folder_val = "cuda/tests/results/" + gpu_folder_name();
    std::string log_path_val = start_run_log(folder_val.c_str(), __FILE__, "tune_", "txt");
    print_gpu();
    std::string csv_path_val = log_path_val.substr(0, log_path_val.size() - 3) + "csv";
    time_every_setting(size_val, csv_path_val);

    printf("\nall tunes done\n");
    return 0;
}
