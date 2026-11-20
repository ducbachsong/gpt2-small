// tune_mlp_up_8bit.cu — the timings of cuda/kernels/8bit/prod/mlp_up_8bit.cuh (mlp_up_8bit): mlp_up_output = ln2_output · mlp_up_weightᵀ + mlp_up_bias, and gelu_output = gelu(mlp_up_output) beside it, the product in 8 bits, on a training
// micro-batch of GPT2_ROWS rows; apart from test_mlp_up_8bit.cu so the tests only check. Each setting is timed taking turns
// with the others over SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged:
//     part 1   this file's setting, with its rounding (mlp_up_8bit) and without (the product alone), against PyTorch's
//              bfloat16 matmul (cuBLAS)
//              (cuBLAS without GELU: mine does it in the product)
//     part 2   every cooperative square (1 or 2 warpgroups x 64, 128 or, with promotion 0, 256 columns x 1 to 4 sets)
//     part 2b  this file's setting against every ping-pong square (128 or 256 columns x 2 to 4 sets) at orders 1, 4, 16
//     part 2c  gelu_output as the model makes it: bfloat16 then round_8bit_per_tensor (both copies), against
//              mlp_up_8bit_gelu_8bit writing both copies itself
//     part 3   at part 2's fastest square: every block order, block count and cluster
//     part 4   at this file's square: the producer's registers x the steps in flight, then every promotion (at 128
//              columns when the square has 256: the promotion's second set of sums needs the room)
// then the fastest as the #define lines of mlp_up_8bit.cuh. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a; the log goes to cuda/tests/results/<GPU>/.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <vector>
#include "../../common/test_helpers.h"          // print_test_name
#include "../../tune_helpers.h"          // gpu_milliseconds, peak_bf16_flops, start_tune_log
#include "../../../kernels/8bit/prod/mlp_up_8bit.cuh"

#define GPT2_ROWS 16384               // a training micro-batch: 8 rows of 2048 tokens (train_bf16.cu's and train_8bit.cu's default)
#define GPT2_EMBED_DIM 1536           // the model's width (nanochat's d24: 24 layers x 1,536, 12 heads)
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 32768  // nanochat's tokenizer: 2^15 ids, no padding
#define FLOPS_PER_MULTIPLY_ADD 2      // a product's flops: 2 x rows x columns x sum_length
#define ROWS GPT2_ROWS               // the product's output {ROWS, COLUMNS} and its sum
#define COLUMNS GPT2_MLP_DIM
#define SUM_LENGTH GPT2_EMBED_DIM

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

/// Makes a PyTorch bfloat16 tensor on the GPU with a copy of my bfloat16 tensor's values.
static torch::Tensor pytorch_bf16_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
        .clone()
        .to(torch::kBFloat16);
}

/// The product's operands, outputs and copies at GPT-2's size.
struct Product
{
    Tensor input_tensor = random_bf16_tensor({ROWS, SUM_LENGTH});
    Tensor weight_tensor = random_bf16_tensor({COLUMNS, SUM_LENGTH});
    Tensor bias_tensor = random_bf16_tensor({COLUMNS});
    Tensor output_tensor = Tensor::zeros({ROWS, COLUMNS}, DType::BFloat16);
    Tensor gelu_output_tensor = Tensor::zeros({ROWS, COLUMNS}, DType::BFloat16);
    MlpUp8bitCopies copies_val = MlpUp8bitCopies(ROWS, COLUMNS, SUM_LENGTH);
};

/// Runs mlp_up_8bit's product on the copies at a setting given (the template choices not given: this file's).
template <int ROW_GROUPS, int COLUMNS_ARG, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_8BIT_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_UP_8BIT_IN_FLIGHT, int PROMOTION_STEPS = MLP_UP_8BIT_PROMOTION_STEPS, int PINGPONG = MLP_UP_8BIT_PINGPONG
          >
static void mine_at(const Product &product_val,
                    int block_rows_per_group_val, int blocks_val
                    )
{
    mlp_up_8bit_rounded_at<ROW_GROUPS, COLUMNS_ARG, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, PINGPONG>(
        product_val.output_tensor, product_val.gelu_output_tensor, product_val.bias_tensor, product_val.copies_val, block_rows_per_group_val, blocks_val);
}

/// One cooperative square run in a cluster of cluster_val: what the sweeps call.
template <int ROW_GROUPS, int COLUMNS_ARG, int STAGES>
static void square_at(const Product &product_val, int cluster_val,
                      int block_rows_per_group_val, int blocks_val)
{
    if (cluster_val == 2)
        mine_at<ROW_GROUPS, COLUMNS_ARG, STAGES, 2, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 0>(
            product_val, block_rows_per_group_val, blocks_val);
    else
        mine_at<ROW_GROUPS, COLUMNS_ARG, STAGES, 1, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 0>(
            product_val, block_rows_per_group_val, blocks_val);
}

/// A square and its run.
struct Square
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Product &, int, int, int);
};

/// Every cooperative square that fits.
static const Square SQUARES_LIST[] = {
    // 1 or 2 warpgroups x 64 or 128 columns x 1 to 4 sets of tiles; with promotion 0, 256 columns (1 x 1 to 4
    // sets, 2 x 1 to 3).
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
    {1, 256, 1, square_at<1, 256, 1>},
    {1, 256, 2, square_at<1, 256, 2>},
    {1, 256, 3, square_at<1, 256, 3>},
    {1, 256, 4, square_at<1, 256, 4>},
    {2, 256, 1, square_at<2, 256, 1>},
    {2, 256, 2, square_at<2, 256, 2>},
    {2, 256, 3, square_at<2, 256, 3>},
#endif
    {1, 64, 1, square_at<1, 64, 1>},
    {1, 64, 2, square_at<1, 64, 2>},
    {1, 64, 3, square_at<1, 64, 3>},
    {1, 64, 4, square_at<1, 64, 4>},
    {1, 128, 1, square_at<1, 128, 1>},
    {1, 128, 2, square_at<1, 128, 2>},
    {1, 128, 3, square_at<1, 128, 3>},
    {1, 128, 4, square_at<1, 128, 4>},
    {2, 64, 1, square_at<2, 64, 1>},
    {2, 64, 2, square_at<2, 64, 2>},
    {2, 64, 3, square_at<2, 64, 3>},
    {2, 64, 4, square_at<2, 64, 4>},
    {2, 128, 1, square_at<2, 128, 1>},
    {2, 128, 2, square_at<2, 128, 2>},
    {2, 128, 3, square_at<2, 128, 3>},
    {2, 128, 4, square_at<2, 128, 4>},
};
static const int SQUARE_COUNT = (int)(sizeof(SQUARES_LIST) / sizeof(SQUARES_LIST[0]));

/// Runs one variant of this file's setting, with these template choices.
///
///   product_val   the product's tensors
template <int PRODUCER_REGISTERS, int IN_FLIGHT, int PROMOTION_STEPS
          >
static void variant_at(const Product &product_val
                       )
{
    constexpr int COLUMNS_ARG = PROMOTION_STEPS > 0 && MLP_UP_8BIT_COLUMNS > 128 ? 128 : MLP_UP_8BIT_COLUMNS;   // a promotion: no room for 256
    mine_at<MLP_UP_8BIT_ROW_GROUPS, COLUMNS_ARG, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS>(product_val,
                                                                                                                 MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// A variant: its template choices and its run.
struct Variant
{
    int producer_registers_val;
    int in_flight_val;
    int promotion_steps_val;
    void (*run_ptr)(const Product &);
};

/// The variants: the producer's registers x the steps in flight, then every promotion, the rest at this file's.
static const Variant VARIANTS_LIST[] = {
    {24, 0, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<24, 0, MLP_UP_8BIT_PROMOTION_STEPS>},
    {24, 1, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<24, 1, MLP_UP_8BIT_PROMOTION_STEPS>},
    {24, 2, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<24, 2, MLP_UP_8BIT_PROMOTION_STEPS>},
    {24, 3, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<24, 3, MLP_UP_8BIT_PROMOTION_STEPS>},
    {32, 0, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<32, 0, MLP_UP_8BIT_PROMOTION_STEPS>},
    {32, 1, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<32, 1, MLP_UP_8BIT_PROMOTION_STEPS>},
    {32, 2, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<32, 2, MLP_UP_8BIT_PROMOTION_STEPS>},
    {32, 3, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<32, 3, MLP_UP_8BIT_PROMOTION_STEPS>},
    {40, 0, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<40, 0, MLP_UP_8BIT_PROMOTION_STEPS>},
    {40, 1, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<40, 1, MLP_UP_8BIT_PROMOTION_STEPS>},
    {40, 2, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<40, 2, MLP_UP_8BIT_PROMOTION_STEPS>},
    {40, 3, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<40, 3, MLP_UP_8BIT_PROMOTION_STEPS>},
    {48, 0, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<48, 0, MLP_UP_8BIT_PROMOTION_STEPS>},
    {48, 1, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<48, 1, MLP_UP_8BIT_PROMOTION_STEPS>},
    {48, 2, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<48, 2, MLP_UP_8BIT_PROMOTION_STEPS>},
    {48, 3, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<48, 3, MLP_UP_8BIT_PROMOTION_STEPS>},
    {56, 0, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<56, 0, MLP_UP_8BIT_PROMOTION_STEPS>},
    {56, 1, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<56, 1, MLP_UP_8BIT_PROMOTION_STEPS>},
    {56, 2, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<56, 2, MLP_UP_8BIT_PROMOTION_STEPS>},
    {56, 3, MLP_UP_8BIT_PROMOTION_STEPS, variant_at<56, 3, MLP_UP_8BIT_PROMOTION_STEPS>},
    {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 0, variant_at<MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 0>},
    {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 1, variant_at<MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 1>},
    {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 2, variant_at<MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 2>},
    {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 4, variant_at<MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 4>},
    {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 8, variant_at<MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, 8>},
};
static const int VARIANT_COUNT = (int)(sizeof(VARIANTS_LIST) / sizeof(VARIANTS_LIST[0]));

/// Gives TFLOPS for the product taking milliseconds_val.
static double tflops_of(double milliseconds_val)
{
    return (double)FLOPS_PER_MULTIPLY_ADD * ROWS * COLUMNS * SUM_LENGTH / (milliseconds_val / MILLISECONDS_PER_SECOND) / TERA;
}

// ══ part 1: against PyTorch (cuBLAS) ══════════════════════════════════════════

static void time_against_cublas(const Product &product_val)
{
    // Prints this file's setting's time with its rounding and without, and PyTorch's bfloat16 matmul's (cuBLAS),
    // TIMED_STEPS of each taking turns, steps WARM_UP_STEPS + 1 on averaged, with TFLOPS (of the product) and the share
    // of this GPU's bfloat16 peak (8-bit's peak is twice it).
    torch::Tensor input_tensor = pytorch_bf16_copy_of(product_val.input_tensor);
    torch::Tensor weight_turned_tensor = pytorch_bf16_copy_of(product_val.weight_tensor).t();
    torch::Tensor bias_tensor = pytorch_bf16_copy_of(product_val.bias_tensor);
    torch::Tensor pytorch_output_tensor = torch::zeros({ROWS, COLUMNS}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    double totals_list[3] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[3];
        times_list[0] = gpu_milliseconds([&] { mlp_up_8bit(product_val.output_tensor, product_val.gelu_output_tensor, product_val.input_tensor, product_val.weight_tensor, product_val.bias_tensor, product_val.copies_val); });
        times_list[1] = gpu_milliseconds([&] {
            mine_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER>(product_val, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
        });
        times_list[2] = gpu_milliseconds([&] { torch::addmm_out(pytorch_output_tensor, bias_tensor, input_tensor, weight_turned_tensor); });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int side_index_val = 0; side_index_val < 3; side_index_val++)
            totals_list[side_index_val] += times_list[side_index_val];
    }
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    const char *names_list[3] = {"mine, rounding included", "mine, the product alone", "PyTorch (cuBLAS, bfloat16)"};
    print_test_name(__func__);
    printf("    mlp up: output {%d, %d}, a sum of %d: %.1f GFLOP\n", ROWS, COLUMNS, SUM_LENGTH,
           (double)FLOPS_PER_MULTIPLY_ADD * ROWS * COLUMNS * SUM_LENGTH / GIGA);
    printf("    average of steps %d..%d:              ms     TFLOPS   of bf16 peak\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    for (int side_index_val = 0; side_index_val < 3; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        printf("        %-28s %9.3f   %8.1f   %6.1f%%\n", names_list[side_index_val], milliseconds_val, tflops_of(milliseconds_val),
               100.0 * tflops_of(milliseconds_val) * TERA / peak_bf16_flops());
    }
    printf("    cuBLAS takes %.2fx as long as mine with the rounding, %.2fx as long as the product alone\n", totals_list[2] / totals_list[0],
           totals_list[2] / totals_list[1]);
}

// ══ part 2: every square ══════════════════════════════════════════════════════

static int time_each_square(const Product &product_val)
{
    // Prints each square's time at this file's block order, block count and cluster
    // All taking turns over SWEEP_ROUNDS rounds; gives the fastest one's place in SQUARES_LIST.
    std::vector<double> totals_list(SQUARE_COUNT, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int square_index_val = 0; square_index_val < SQUARE_COUNT; square_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] {
                SQUARES_LIST[square_index_val].run_ptr(product_val, MLP_UP_8BIT_CLUSTER, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[square_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    int fastest_index_val = 0;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d; warpgroups x columns x sets of tiles\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS);
    for (int square_index_val = 0; square_index_val < SQUARE_COUNT; square_index_val++)
    {
        const Square &square_val = SQUARES_LIST[square_index_val];
        double milliseconds_val = totals_list[square_index_val] / averaged_val;
        bool this_files_val = MLP_UP_8BIT_PINGPONG == 0 && square_val.row_groups_val == MLP_UP_8BIT_ROW_GROUPS &&
                              square_val.columns_val == MLP_UP_8BIT_COLUMNS && square_val.stages_val == MLP_UP_8BIT_STAGES;
        printf("        %d x %3d x %d   %8.3f ms   %6.1f TFLOPS%s\n", square_val.row_groups_val, square_val.columns_val, square_val.stages_val,
               milliseconds_val, tflops_of(milliseconds_val), this_files_val ? "   (this file's)" : "");
        if (totals_list[square_index_val] < totals_list[fastest_index_val])
            fastest_index_val = square_index_val;
    }
    printf("    fastest: %d x %d x %d\n", SQUARES_LIST[fastest_index_val].row_groups_val, SQUARES_LIST[fastest_index_val].columns_val,
           SQUARES_LIST[fastest_index_val].stages_val);
    return fastest_index_val;
}

/// A launch choice at a square: its cluster, block order and block count.
struct Choice
{
    int cluster_val;
    int block_rows_per_group_val;
    int blocks_val;
};

// ══ part 2b: ping-pong against this file's setting ════════════════════════════

/// One ping-pong square (2 warpgroups x COLUMNS x STAGES, clusters of 1) at a block order: what the sweep below times.
template <int COLUMNS_ARG, int STAGES>
static void pingpong_at(const Product &product_val, int block_rows_per_group_val)
{
    mine_at<2, COLUMNS_ARG, STAGES, 1, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 1>(product_val,
                                                                                                                    block_rows_per_group_val, 0);
}

/// A ping-pong square and its run.
struct PingpongSquare
{
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Product &, int);
};

/// The ping-pong squares: 128 columns, or 256 with promotion 0, x 2 to 4 sets (256 x 4 fits: a set holds one left tile).
static const PingpongSquare PINGPONG_SQUARES_LIST[] = {
    {128, 2, pingpong_at<128, 2>}, {128, 3, pingpong_at<128, 3>}, {128, 4, pingpong_at<128, 4>},
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
    {256, 2, pingpong_at<256, 2>}, {256, 3, pingpong_at<256, 3>}, {256, 4, pingpong_at<256, 4>},
#endif
};

/// What part 2b found fastest: ping-pong or not, and its square and order.
struct PingpongChoice
{
    bool pingpong_val;
    int columns_val;
    int stages_val;
    int block_rows_per_group_val;
};

static PingpongChoice time_pingpong_against_this_files_setting(const Product &product_val, const Choice &cooperative_val, int square_index_val)
{
    // Prints part 2 and 3's fastest cooperative setting and this file's setting against each ping-pong square (the 2
    // warpgroups each on their own 64-row squares, their sums in turns, one's store, GELU's included, under the
    // other's sums) at block orders 1, 4 and 16, persistent, all taking turns over SWEEP_ROUNDS rounds; then the
    // fastest, and as how many times this file's setting's time. Gives the fastest.
    struct Run
    {
        const char *name_ptr;
        int row_groups_val;
        int columns_val;
        int stages_val;
        int block_rows_per_group_val;
        const PingpongSquare *square_ptr;   // nullptr: this file's setting, or (cooperative) part 2's square
        bool cooperative_val;
    };
    const Square &square_val = SQUARES_LIST[square_index_val];
    std::vector<Run> runs_list = {
        {"this file's setting", MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, nullptr, false},
        {"cooperative", square_val.row_groups_val, square_val.columns_val, square_val.stages_val, cooperative_val.block_rows_per_group_val, nullptr, true}};
    for (const PingpongSquare &pingpong_val : PINGPONG_SQUARES_LIST)
        for (int block_rows_per_group_val : {1, 4, 16})
            runs_list.push_back({"ping-pong", 2, pingpong_val.columns_val, pingpong_val.stages_val, block_rows_per_group_val, &pingpong_val, false});
    std::vector<double> totals_list(runs_list.size(), 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (size_t run_index_val = 0; run_index_val < runs_list.size(); run_index_val++)
        {
            const Run &run_val = runs_list[run_index_val];
            float milliseconds_val = gpu_milliseconds([&] {
                if (run_val.square_ptr != nullptr)
                    run_val.square_ptr->run_ptr(product_val, run_val.block_rows_per_group_val);
                else if (run_val.cooperative_val)
                    square_val.run_ptr(product_val, cooperative_val.cluster_val, cooperative_val.block_rows_per_group_val, cooperative_val.blocks_val);
                else
                    mine_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER>(product_val,
                                                                                                 MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[run_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    size_t fastest_index_val = 1;                                     // this file's setting is one of the others: only they are chosen
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS);
    for (size_t run_index_val = 0; run_index_val < runs_list.size(); run_index_val++)
    {
        const Run &run_val = runs_list[run_index_val];
        double milliseconds_val = totals_list[run_index_val] / averaged_val;
        printf("        %-20s %d x %3d x %d, %2d rows a group   %8.3f ms   %6.1f TFLOPS\n", run_val.name_ptr, run_val.row_groups_val,
               run_val.columns_val, run_val.stages_val, run_val.block_rows_per_group_val, milliseconds_val, tflops_of(milliseconds_val));
        if (run_index_val > 0 && totals_list[run_index_val] < totals_list[fastest_index_val])
            fastest_index_val = run_index_val;
    }
    const Run &fastest_val = runs_list[fastest_index_val];
    printf("    fastest: %s %d x %d x %d, %d rows a group: %.3f ms, this file's setting %.2fx as long\n", fastest_val.name_ptr,
           fastest_val.row_groups_val, fastest_val.columns_val, fastest_val.stages_val, fastest_val.block_rows_per_group_val,
           totals_list[fastest_index_val] / averaged_val, totals_list[0] / totals_list[fastest_index_val]);
    return {fastest_val.square_ptr != nullptr, fastest_val.columns_val, fastest_val.stages_val, fastest_val.block_rows_per_group_val};
}

// ══ part 2c: gelu_output in 8 bits against bfloat16 and a rounding pass ═══════

static void time_gelu_8bit_against_bf16_and_rounding(const Product &product_val)
{
    // Prints, at this file's setting, all taking turns over SWEEP_ROUNDS rounds: gelu_output as bfloat16 then rounded by
    // round_8bit_per_tensor into both 8-bit copies at a delayed scale (what the model did before MODEL_8BIT_GELU_8BIT: a
    // write of 2 bytes a value, then a read of them and 2 bytes written), the rounding pass alone, and
    // mlp_up_8bit_gelu_8bit writing both copies itself (2 bytes a value written, nothing read back).
    Tensor rows_8bit_tensor = Tensor::zeros({ROWS, COLUMNS}, DType::E4M3);
    Tensor turned_8bit_tensor = Tensor::zeros({COLUMNS, ROWS}, DType::E4M3);
    Tensor scale_tensor = Tensor::zeros({1});
    Round8bitHistory bf16_history_val, gelu_8bit_history_val;
    mlp_up_8bit_rounded(product_val.output_tensor, product_val.gelu_output_tensor, product_val.bias_tensor, product_val.copies_val);
    for (Round8bitHistory *history_ptr : {&bf16_history_val, &gelu_8bit_history_val})   // the first call, exact: the histories started
        round_8bit_per_tensor(rows_8bit_tensor, turned_8bit_tensor, scale_tensor, product_val.gelu_output_tensor, history_ptr);
    auto round_val = [&] {
        round_8bit_per_tensor(rows_8bit_tensor, turned_8bit_tensor, scale_tensor, product_val.gelu_output_tensor,
                              &bf16_history_val);
    };
    auto gelu_8bit_val = [&] {
        MlpUp8bitGelu8bit gelu_val;
        gelu_val.rows_8bit_tensor = rows_8bit_tensor;
        gelu_val.turned_8bit_tensor = turned_8bit_tensor;
        gelu_val.scale_tensor = scale_tensor;
        gelu_val.margin_val = ROUND_8BIT_DELAYED_MARGIN;
        round_8bit_delayed_largest(gelu_8bit_history_val, gelu_val.scale_largest_ptr, gelu_val.next_largest_ptr);
        mlp_up_8bit_gelu_8bit(product_val.output_tensor, gelu_val, product_val.bias_tensor, product_val.copies_val);
    };
    const char *names_list[4] = {"bfloat16 gelu_output", "  + round_8bit_per_tensor", "the rounding pass alone", "mlp_up_8bit_gelu_8bit"};
    double totals_list[4] = {};
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
    {
        float times_list[4];
        times_list[0] = gpu_milliseconds([&] { mlp_up_8bit_rounded(product_val.output_tensor, product_val.gelu_output_tensor, product_val.bias_tensor,
                                                                   product_val.copies_val); });
        times_list[1] = gpu_milliseconds([&] {
            mlp_up_8bit_rounded(product_val.output_tensor, product_val.gelu_output_tensor, product_val.bias_tensor, product_val.copies_val);
            round_val();
        });
        times_list[2] = gpu_milliseconds(round_val);
        times_list[3] = gpu_milliseconds(gelu_8bit_val);
        if (round_index_val > WARM_UP_STEPS)
            for (int run_index_val = 0; run_index_val < 4; run_index_val++)
                totals_list[run_index_val] += times_list[run_index_val];
    }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, at this file's setting\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS);
    for (int run_index_val = 0; run_index_val < 4; run_index_val++)
        printf("        %-28s %8.3f ms\n", names_list[run_index_val], totals_list[run_index_val] / averaged_val);
    printf("    gelu_output in 8 bits: %.3f ms against %.3f (%.2fx); the backward's turned rounding of it, gone too, is not timed here\n",
           totals_list[3] / averaged_val, totals_list[1] / averaged_val, totals_list[1] / totals_list[3]);
}

// ══ part 3: order, blocks and cluster ═════════════════════════════════════════

static Choice time_each_choice(const Product &product_val, int square_index_val)
{
    // Prints, at part 2's fastest square, each cluster (1, 2), block order (1, 4, 16, 64) and block count (as many as
    // run at once, one a square)
    // All taking turns over SWEEP_ROUNDS rounds; gives the fastest.
    std::vector<Choice> choices_list;
    for (int cluster_val : {1, 2})
        for (int block_rows_per_group_val : {1, 4, 16, 64})
            for (int blocks_val : {0, MLP_UP_8BIT_ONE_A_SQUARE})
                choices_list.push_back({cluster_val, block_rows_per_group_val, blocks_val});
    const Square &square_val = SQUARES_LIST[square_index_val];
    std::vector<double> totals_list(choices_list.size(), 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (size_t choice_index_val = 0; choice_index_val < choices_list.size(); choice_index_val++)
        {
            const Choice &choice_val = choices_list[choice_index_val];
            float milliseconds_val = gpu_milliseconds([&] {
                square_val.run_ptr(product_val, choice_val.cluster_val, choice_val.block_rows_per_group_val, choice_val.blocks_val);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[choice_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    size_t fastest_index_val = 0;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, at %d x %d x %d\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS, square_val.row_groups_val,
           square_val.columns_val, square_val.stages_val);
    for (size_t choice_index_val = 0; choice_index_val < choices_list.size(); choice_index_val++)
    {
        const Choice &choice_val = choices_list[choice_index_val];
        double milliseconds_val = totals_list[choice_index_val] / averaged_val;
        printf("        clusters of %d, %2d rows a group, %-14s   %8.3f ms   %6.1f TFLOPS\n", choice_val.cluster_val,
               choice_val.block_rows_per_group_val, choice_val.blocks_val == 0 ? "persistent" : "one a square", milliseconds_val,
               tflops_of(milliseconds_val));
        if (totals_list[choice_index_val] < totals_list[fastest_index_val])
            fastest_index_val = choice_index_val;
    }
    return choices_list[fastest_index_val];
}

// ══ part 4: the template choices, at this file's square ══════════════════════

static Variant time_each_variant(const Product &product_val)
{
    // Prints, at this file's square, cluster, block order and block count
    // Each variant's time, all taking turns over SWEEP_ROUNDS rounds; steps in flight past STAGES - 1 and producer
    // registers with 1 consumer warpgroup (none are moved) are left out. These are template choices compiled for this
    // file's square: when parts 2 and 3 change the square, update its defines and run again. Gives the fastest of
    // each group (registers and steps in flight; promotion
    // ), each in its own fields.
    std::vector<double> totals_list(VARIANT_COUNT, 0.0);
    auto counts_val = [](const Variant &variant_val) {
        return variant_val.in_flight_val <= MLP_UP_8BIT_STAGES - 1 && (MLP_UP_8BIT_ROW_GROUPS > 1 || variant_val.producer_registers_val == MLP_UP_8BIT_PRODUCER_REGISTERS);
    };
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int variant_index_val = 0; variant_index_val < VARIANT_COUNT; variant_index_val++)
        {
            const Variant &variant_val = VARIANTS_LIST[variant_index_val];
            if (!counts_val(variant_val))
                continue;
            float milliseconds_val = gpu_milliseconds([&] {
                variant_val.run_ptr(product_val);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[variant_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    Variant best_val = {MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS,
                        nullptr};
    double best_registers_val = 1e30, best_promotion_val = 1e30, best_staged_val = 1e30;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, at %d x %d x %d\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS, MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES);
    for (int variant_index_val = 0; variant_index_val < VARIANT_COUNT; variant_index_val++)
    {
        const Variant &variant_val = VARIANTS_LIST[variant_index_val];
        if (!counts_val(variant_val))
            continue;
        double milliseconds_val = totals_list[variant_index_val] / averaged_val;
        printf("        producer %2d registers, %d in flight, promotion %d   %8.3f ms   %6.1f TFLOPS\n", variant_val.producer_registers_val,
               variant_val.in_flight_val, variant_val.promotion_steps_val, milliseconds_val, tflops_of(milliseconds_val));
        bool files_staged_val = true;
        bool files_registers_val = variant_val.producer_registers_val == MLP_UP_8BIT_PRODUCER_REGISTERS && variant_val.in_flight_val == MLP_UP_8BIT_IN_FLIGHT;
        bool files_promotion_val = variant_val.promotion_steps_val == MLP_UP_8BIT_PROMOTION_STEPS;
        if (files_staged_val && files_promotion_val && milliseconds_val < best_registers_val)
        {
            best_registers_val = milliseconds_val;
            best_val.producer_registers_val = variant_val.producer_registers_val;
            best_val.in_flight_val = variant_val.in_flight_val;
        }
        if (files_staged_val && files_registers_val && milliseconds_val < best_promotion_val)
        {
            best_promotion_val = milliseconds_val;
            best_val.promotion_steps_val = variant_val.promotion_steps_val;
        }
    }
    (void)best_staged_val;
    printf("    (the promotion changes the bits a little: every 4 steps was as accurate as every step on an H100 NVL; 0, never,\n"
           "     keeps the whole sum in the tensor cores, less accurate over a long sum)\n");
    return best_val;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and cuda/tests/results/<GPU>/<kernel>_<date>_<time>.txt
    torch::manual_seed(0);
    Product product_val;
    mlp_up_8bit_round(product_val.copies_val, product_val.input_tensor, product_val.weight_tensor);
    time_against_cublas(product_val);
    int square_index_val = time_each_square(product_val);
    Choice choice_val = time_each_choice(product_val, square_index_val);
    PingpongChoice pingpong_val = time_pingpong_against_this_files_setting(product_val, choice_val, square_index_val);
    time_gelu_8bit_against_bf16_and_rounding(product_val);
    Variant variant_val = time_each_variant(product_val);
    // the fastest, as the defines of the product's file: ping-pong's square and order if part 2b found it fastest
    const Square &square_val = SQUARES_LIST[square_index_val];
    printf("\n    for cuda/kernels/8bit/prod/mlp_up_8bit.cuh:\n");
    if (pingpong_val.pingpong_val)
    {
        printf("#define MLP_UP_8BIT_ROW_GROUPS 2\n#define MLP_UP_8BIT_COLUMNS %d\n#define MLP_UP_8BIT_STAGES %d\n", pingpong_val.columns_val,
               pingpong_val.stages_val);
        printf("#define MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP %d\n#define MLP_UP_8BIT_CLUSTER 1\n#define MLP_UP_8BIT_BLOCKS 0\n", pingpong_val.block_rows_per_group_val);
    }
    else
    {
        printf("#define MLP_UP_8BIT_ROW_GROUPS %d\n#define MLP_UP_8BIT_COLUMNS %d\n#define MLP_UP_8BIT_STAGES %d\n", square_val.row_groups_val, square_val.columns_val,
               square_val.stages_val);
        printf("#define MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP %d\n#define MLP_UP_8BIT_CLUSTER %d\n#define MLP_UP_8BIT_BLOCKS %s\n", choice_val.block_rows_per_group_val,
               choice_val.cluster_val, choice_val.blocks_val == 0 ? "0" : "MLP_UP_8BIT_ONE_A_SQUARE");
    }
    printf("#define MLP_UP_8BIT_PRODUCER_REGISTERS %d\n#define MLP_UP_8BIT_IN_FLIGHT %d\n#define MLP_UP_8BIT_PROMOTION_STEPS %d\n", variant_val.producer_registers_val,
           variant_val.in_flight_val, variant_val.promotion_steps_val);
    printf("#define MLP_UP_8BIT_PINGPONG %d\n", pingpong_val.pingpong_val ? 1 : 0);
    printf("    (part 4's at this file's square: if parts 2 and 3 changed it, update the defines and run again; the promotion\n"
           "     changes the bits: take a faster one only if the tests stay within their tolerances)\n");

    printf("\nall tunes done\n");
    return 0;
}
