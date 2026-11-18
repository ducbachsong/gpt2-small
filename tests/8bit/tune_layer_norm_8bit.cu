// tune_layer_norm_8bit.cu — the timings of llmc/kernels/8bit/layer_norm_8bit.cuh, apart from
// tests/8bit/test_layer_norm_8bit.cu so the tests only check numbers: time_8bit_forward_against_bf16_and_roundings
// (both copies made inside LayerNorm against LayerNorm then the forward's and the backward's rounding passes),
// time_each_tile_and_block_size (the tile's rows and the block size, swept). With the bytes each moves, its GB/s and
// MBU. The backward is layer_norm_bf16's: tests/bf16/tune_layer_norm_bf16.cu.
//
// Built as tests/8bit/test_layer_norm_8bit.cu is (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/layer_norm_8bit_<date>_<time>.txt. Any GPU.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"          // ASSERT_EQ
#include "../tune_helpers.h"          // gpu_milliseconds, warm_up_gpu, peak_gigabytes_per_second, gigabytes_per_second
#include "../../llmc/kernels/8bit/layer_norm_8bit.cuh"

#define GPT2_BATCH_SIZE 8             // a training micro-batch: 8 rows of 2048 tokens (the trainers' default)
#define GPT2_SEQ_LEN 2048
#define GPT2_ROWS (GPT2_BATCH_SIZE * GPT2_SEQ_LEN)
#define GPT2_EMBED_DIM 1536           // the model's width (nanochat's d24: 24 layers x 1,536)
#define GPT2_LAYER_NORMS 49           // LayerNorms in a d24 forward: 2 a block, 24 blocks, and lnf
#define MEGA 1e6                      // bytes in an MB

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// The d24 model's LayerNorm (16384 random rows of 1536) with both 8-bit copies' tensors, a bfloat16 output for the
/// way before, and a largest for the scale.
struct Gpt2LayerNorm
{
    Tensor input_tensor = random_bf16_tensor({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM});
    Tensor weight_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor bias_tensor = random_bf16_tensor({GPT2_EMBED_DIM});
    Tensor output_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, DType::BFloat16);
    Tensor mean_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
    Tensor rstd_tensor = Tensor::zeros({GPT2_BATCH_SIZE, GPT2_SEQ_LEN});
    Tensor rows_8bit_tensor = Tensor::zeros({GPT2_ROWS, GPT2_EMBED_DIM}, DType::E4M3);
    Tensor turned_8bit_tensor = Tensor::zeros({GPT2_EMBED_DIM, GPT2_ROWS}, DType::E4M3);
    Tensor scale_tensor = Tensor::zeros({1});
    Tensor largest_tensor = Tensor::zeros({2});        // [the largest the scale comes from, the next largest], float bits

    Gpt2LayerNorm(void)
    {
        std::vector<float> slots_list = {4.0f, 0.0f};    // a LayerNorm output's largest is a few: any will do for the time
        largest_tensor.copy_from_cpu(slots_list.data());
    }

    /// Gives the copies' tensors, the scale from largest_tensor's first slot x ROUND_8BIT_DELAYED_MARGIN, this tensor's
    /// largest onto its second.
    LayerNorm8bitCopies copies(void) const
    {
        unsigned *slots_ptr = reinterpret_cast<unsigned *>(largest_tensor.data());
        LayerNorm8bitCopies copies_val;
        copies_val.rows_8bit_tensor = rows_8bit_tensor;
        copies_val.turned_8bit_tensor = turned_8bit_tensor;
        copies_val.scale_tensor = scale_tensor;
        copies_val.scale_largest_ptr = slots_ptr;
        copies_val.margin_val = ROUND_8BIT_DELAYED_MARGIN;
        copies_val.next_largest_ptr = slots_ptr + 1;
        return copies_val;
    }

    /// Runs layer_norm_8bit_forward_at<TILE_ROWS> as the model does (no bfloat16 output) at a block size given.
    template <int TILE_ROWS>
    void forward_8bit(int threads_val) const
    {
        layer_norm_8bit_forward_at<TILE_ROWS>(Tensor(), copies(), mean_tensor, rstd_tensor, input_tensor, weight_tensor, bias_tensor,
                                              threads_val);
    }
};

/// Gives the bytes asked for, in run order: the 8-bit forward, layer_norm_bf16_forward, the forward's rounding of its
/// output (the row copy), the backward's (the turned copy). Weight and bias are left out.
///
///     8-bit forward      reads input; writes both copies, the scale, mean, rstd            2 + 1 + 1 bytes a value
///     bfloat16 forward   reads input; writes output, mean, rstd                            2 + 2
///     forward rounding   reads output; writes the row copy, the scale                      2 + 1
///     backward rounding  reads output; writes the turned copy, the scale                   2 + 1
static void run_bytes(double (&bytes_list)[4])
{
    double values_val = (double)GPT2_ROWS * GPT2_EMBED_DIM;
    double row_floats_bytes_val = (double)GPT2_ROWS * sizeof(float);
    double scale_bytes_val = sizeof(float);
    bytes_list[0] = values_val * (sizeof(nv_bfloat16) + 2 * sizeof(__nv_fp8_e4m3)) + 2 * row_floats_bytes_val + scale_bytes_val;
    bytes_list[1] = values_val * 2 * sizeof(nv_bfloat16) + 2 * row_floats_bytes_val;
    bytes_list[2] = values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + scale_bytes_val;
    bytes_list[3] = values_val * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3)) + scale_bytes_val;
}

// ══ part 1: both copies made inside LayerNorm against LayerNorm then two roundings ═

static void time_8bit_forward_against_bf16_and_roundings(void)
{
    // Prints, on the d24 model's sizes (16384 rows of 1536), each the average of rounds WARM_UP_STEPS + 1 ..
    // SWEEP_ROUNDS taking turns: layer_norm_8bit_forward (both copies, no bfloat16 output), and the way the model
    // took before: layer_norm_bf16_forward, the forward's round_8bit_per_tensor of its output into the row copy (at a
    // delayed scale) and the backward's into the turned copy (its exact largest, known), with the bytes each asks for,
    // GB/s and MBU; then the two ways over a forward's GPT2_LAYER_NORMS LayerNorms, and the memory the bfloat16 outputs
    // kept for the backward took.
    //
    // Expect: the 8-bit forward a little slower than the bfloat16 one alone (one byte a value more written, and a
    // tile turned through shared memory), and well under the three passes (which read the output back twice). Both
    // copies the bits round_8bit_per_tensor gives from my output at the same delayed scale.

    // 1. the d24 model's LayerNorm; both histories started by the same exact first call, then both ways once, checked
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    const Gpt2LayerNorm &l = layer_norm_val;
    Tensor reference_rows_8bit_tensor = Tensor::zeros({GPT2_ROWS, GPT2_EMBED_DIM}, DType::E4M3);
    Tensor reference_turned_8bit_tensor = Tensor::zeros({GPT2_EMBED_DIM, GPT2_ROWS}, DType::E4M3);
    Tensor reference_scale_tensor = Tensor::zeros({1});
    Round8bitHistory mine_history_val, reference_history_val;
    layer_norm_bf16_forward(l.output_tensor, l.mean_tensor, l.rstd_tensor, l.input_tensor, l.weight_tensor, l.bias_tensor);
    for (Round8bitHistory *history_ptr : {&mine_history_val, &reference_history_val})
        round_8bit_per_tensor(reference_rows_8bit_tensor, reference_turned_8bit_tensor, reference_scale_tensor, l.output_tensor, history_ptr);
    LayerNorm8bitCopies mine_copies_val = l.copies();                // its scale and next largest from the history instead
    round_8bit_delayed_largest(mine_history_val, mine_copies_val.scale_largest_ptr, mine_copies_val.next_largest_ptr);
    layer_norm_8bit_forward(l.output_tensor, mine_copies_val, l.mean_tensor, l.rstd_tensor, l.input_tensor, l.weight_tensor,
                            l.bias_tensor);
    round_8bit_per_tensor(reference_rows_8bit_tensor, reference_turned_8bit_tensor, reference_scale_tensor, l.output_tensor,
                          &reference_history_val);
    bool same_val = l.rows_8bit_tensor.equal(reference_rows_8bit_tensor) && l.turned_8bit_tensor.equal(reference_turned_8bit_tensor) &&
                    l.scale_tensor.tolist() == reference_scale_tensor.tolist();

    // 2. the 4 runs, taking turns over SWEEP_ROUNDS rounds
    const int runs_val = 4;
    auto run_val = [&](int run_index_val) {
        if (run_index_val == 0)
            l.forward_8bit<LAYER_NORM_8BIT_TILE_ROWS>(LAYER_NORM_8BIT_THREADS_PER_BLOCK);
        else if (run_index_val == 1)
            layer_norm_bf16_forward(l.output_tensor, l.mean_tensor, l.rstd_tensor, l.input_tensor, l.weight_tensor, l.bias_tensor);
        else if (run_index_val == 2)
            round_8bit_per_tensor(reference_rows_8bit_tensor, Tensor(), reference_scale_tensor, l.output_tensor, &reference_history_val);
        else
            round_8bit_per_tensor(Tensor(), reference_turned_8bit_tensor, reference_scale_tensor, l.output_tensor,
                                  &reference_history_val, true);
    };
    warm_up_gpu([&] { run_val(0); });
    float total_ms_list[runs_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
        {
            float ms_val = gpu_milliseconds([&] { run_val(run_index_val); });
            if (round_index_val > WARM_UP_STEPS)
                total_ms_list[run_index_val] += ms_val;
        }

    // 3. print each with its GB/s and MBU, then the two ways and the memory
    double bytes_list[runs_val];
    run_bytes(bytes_list);
    const char *names_list[runs_val] = {"layer_norm_8bit_forward", "layer_norm_bf16_forward", "forward's rounding (rows)",
                                        "backward's rounding (turned)"};
    float ms_list[runs_val];
    double peak_val = peak_gigabytes_per_second();
    print_test_name(__func__);
    printf("    this GPU's peak %.0f GB/s; both copies round_8bit_per_tensor's bits: %s\n", peak_val, same_val ? "yes" : "NO");
    printf("    run                               MB        ms     GB/s    MBU\n");
    for (int run_index_val = 0; run_index_val < runs_val; run_index_val++)
    {
        ms_list[run_index_val] = total_ms_list[run_index_val] / (SWEEP_ROUNDS - WARM_UP_STEPS);
        double speed_val = gigabytes_per_second(bytes_list[run_index_val], ms_list[run_index_val]);
        printf("    %-30s   %6.1f   %7.4f   %6.0f   %3.0f%%\n", names_list[run_index_val], bytes_list[run_index_val] / MEGA,
               ms_list[run_index_val], speed_val, 100.0 * speed_val / peak_val);
    }
    float before_ms_val = ms_list[1] + ms_list[2] + ms_list[3];
    double kept_bf16_bytes_val = (double)GPT2_ROWS * GPT2_EMBED_DIM * sizeof(nv_bfloat16);
    double kept_8bit_bytes_val = (double)GPT2_ROWS * GPT2_EMBED_DIM * sizeof(__nv_fp8_e4m3);
    printf("    both copies:   in one kernel %.4f ms   LayerNorm then two roundings %.4f ms (%.2fx)   over %d LayerNorms: %.3f ms saved\n",
           ms_list[0], before_ms_val, before_ms_val / ms_list[0], GPT2_LAYER_NORMS, GPT2_LAYER_NORMS * (before_ms_val - ms_list[0]));
    printf("    kept for the backward, a LayerNorm: %.1f MB bfloat16 before, %.1f MB the turned copy now; over %d: %.2f GB less\n",
           kept_bf16_bytes_val / MEGA, kept_8bit_bytes_val / MEGA, GPT2_LAYER_NORMS,
           GPT2_LAYER_NORMS * (kept_bf16_bytes_val - kept_8bit_bytes_val) / GIGA);

    // 4. check
    ASSERT_EQ(same_val, true);
}

// ══ part 2: the tile's rows and the block size ═════════════════════════════════

/// One tile the forward can take: its rows and its run.
struct Tile
{
    int tile_rows_val;
    void (*run_ptr)(const Gpt2LayerNorm &, int);
};

/// Runs the forward at tile TILE_ROWS and a block size given.
template <int TILE_ROWS>
static void forward_at_tile(const Gpt2LayerNorm &layer_norm_val, int threads_val)
{
    layer_norm_val.forward_8bit<TILE_ROWS>(threads_val);
}

static void time_each_tile_and_block_size(void)
{
    // Prints layer_norm_8bit_forward on the d24 model's sizes at tiles of 16, 32 and 64 rows (16 x 1536 .. 64 x 1536
    // bytes of shared memory a block: 24, 48, 96 KB) and 2, 4, 8 and 16 warps a block, each the average of rounds
    // WARM_UP_STEPS + 1 .. SWEEP_ROUNDS taking turns, with GB/s and MBU, and the fastest. A tile of more rows makes
    // each column's turned piece longer (more of a 32-byte sector written at once) but fewer blocks fit on an SM.
    //
    // Expect: every setting gives the default's copies, scale and next largest to the last bit. Only the time
    // changes.
    const Tile tiles_list[] = {{16, forward_at_tile<16>}, {32, forward_at_tile<32>}, {64, forward_at_tile<64>}};
    const int threads_list[] = {2 * HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                16 * HARDWARE_THREADS_PER_WARP};
    const int tiles_val = sizeof(tiles_list) / sizeof(tiles_list[0]);
    const int threads_val = sizeof(threads_list) / sizeof(threads_list[0]);

    // 1. the d24 model's LayerNorm; the default's results
    torch::manual_seed(0);
    Gpt2LayerNorm layer_norm_val;
    layer_norm_val.forward_8bit<LAYER_NORM_8BIT_TILE_ROWS>(LAYER_NORM_8BIT_THREADS_PER_BLOCK);
    Tensor usual_rows_tensor = layer_norm_val.rows_8bit_tensor.clone();
    Tensor usual_turned_tensor = layer_norm_val.turned_8bit_tensor.clone();
    std::vector<float> usual_scale_list = layer_norm_val.scale_tensor.tolist();
    warm_up_gpu([&] { layer_norm_val.forward_8bit<LAYER_NORM_8BIT_TILE_ROWS>(LAYER_NORM_8BIT_THREADS_PER_BLOCK); });

    // 2. time every setting, taking turns over SWEEP_ROUNDS rounds
    float total_ms_list[tiles_val][threads_val] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int tile_index_val = 0; tile_index_val < tiles_val; tile_index_val++)
            for (int threads_index_val = 0; threads_index_val < threads_val; threads_index_val++)
            {
                float ms_val = gpu_milliseconds([&] { tiles_list[tile_index_val].run_ptr(layer_norm_val, threads_list[threads_index_val]); });
                if (round_index_val > WARM_UP_STEPS)
                    total_ms_list[tile_index_val][threads_index_val] += ms_val;
            }

    // 3. print each with GB/s and MBU, and check its results against the default's
    double bytes_list[4];
    run_bytes(bytes_list);
    double peak_val = peak_gigabytes_per_second();
    int settings_with_the_same_result_val = 0;
    float fastest_ms_val = 0.0f;
    int fastest_tile_rows_val = 0, fastest_threads_val = 0;
    print_test_name(__func__);
    printf("    %.1f MB asked for; this GPU's peak %.0f GB/s\n", bytes_list[0] / MEGA, peak_val);
    printf("    tile rows   threads   blocks        ms     GB/s    MBU   same result\n");
    for (int tile_index_val = 0; tile_index_val < tiles_val; tile_index_val++)
        for (int threads_index_val = 0; threads_index_val < threads_val; threads_index_val++)
        {
            const Tile &tile_val = tiles_list[tile_index_val];
            int block_threads_val = threads_list[threads_index_val];
            layer_norm_val.rows_8bit_tensor.zero_();
            layer_norm_val.turned_8bit_tensor.zero_();
            layer_norm_val.scale_tensor.zero_();
            tile_val.run_ptr(layer_norm_val, block_threads_val);
            bool same_val = layer_norm_val.rows_8bit_tensor.equal(usual_rows_tensor) &&
                            layer_norm_val.turned_8bit_tensor.equal(usual_turned_tensor) &&
                            layer_norm_val.scale_tensor.tolist() == usual_scale_list;
            if (same_val)
                settings_with_the_same_result_val++;
            float ms_val = total_ms_list[tile_index_val][threads_index_val] / (SWEEP_ROUNDS - WARM_UP_STEPS);
            if (fastest_tile_rows_val == 0 || ms_val < fastest_ms_val)
            {
                fastest_ms_val = ms_val;
                fastest_tile_rows_val = tile_val.tile_rows_val;
                fastest_threads_val = block_threads_val;
            }
            double speed_val = gigabytes_per_second(bytes_list[0], ms_val);
            bool usual_setting_val = tile_val.tile_rows_val == LAYER_NORM_8BIT_TILE_ROWS && block_threads_val == LAYER_NORM_8BIT_THREADS_PER_BLOCK;
            printf("    %9d   %7d   %6d   %7.4f   %6.0f   %3.0f%%   %-11s%s\n", tile_val.tile_rows_val, block_threads_val,
                   (int)CEIL_DIV((size_t)GPT2_ROWS, (size_t)tile_val.tile_rows_val), ms_val, speed_val, 100.0 * speed_val / peak_val,
                   same_val ? "yes" : "NO", usual_setting_val ? "   <- this file's" : "");
        }
    printf("    fastest: tile %d rows, %d threads, %.4f ms\n", fastest_tile_rows_val, fastest_threads_val, fastest_ms_val);

    // 4. every setting gave the same results
    ASSERT_EQ(settings_with_the_same_result_val, tiles_val * threads_val);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_8bit_forward_against_bf16_and_roundings();
    time_each_tile_and_block_size();

    printf("\nall tunes done\n");
    return 0;
}
