// tune_wgmma_matmul_bf16.cu — the timings of llmc/kernels/bf16/wgmma_matmul_bf16.cuh, apart from tests/bf16/test_wgmma_matmul_bf16.cu so the tests only check
// numbers: time_20_products_of_each_shape. Mine (the tensor cores, by WGMMA and TMA) against mma_matmul_bf16.cuh (mma.sync),
// fma_matmul_bf16.cuh (the CUDA cores) and PyTorch's bfloat16 matmul (cuBLAS), on GPT-2 small's 5 linear layers, each layer's three
// products (the forward, and the backward's input_grad and weight_grad, with left or right turned round), and
// the worklog's 4096 x 4096 x 4096.
//
// Built as tests/bf16/test_wgmma_matmul_bf16.cu is, with -arch=sm_90a (see tests/README.md); what it prints also goes to
// tests/results/<GPU>/wgmma_matmul_bf16_<date>_<time>.txt. Needs an H100.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <utility>                    // std::pair: a product's two operands
#include "../test_helpers.h"    // ASSERT_EQ, ASSERT_NEAR
#include "../tune_helpers.h"            // gpu_milliseconds, peak_bf16_flops, print_gpu
#include "../../llmc/kernels/bf16/wgmma_matmul_bf16.cuh"
#include "../../llmc/kernels/bf16/mma_matmul_bf16.cuh" // mma_matmul_bf16: the tensor cores by mma.sync, Ampere's way
#include "../../llmc/kernels/bf16/fma_matmul_bf16.cuh" // fma_matmul_bf16: the CUDA cores

static const double BFLOAT16_TOLERANCE = 1e-2; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_ROWS 16384               // a training micro-batch: 16 rows of 1024 tokens, tune_model's best
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define WORKLOG_SIZE 4096             // the worklog's M = N = K
#define FLOPS_PER_MULTIPLY_ADD 2      // a product's flops: 2 x rows x columns x sum_length

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch
/// on the GPU, then rounded to bfloat16.
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU, bfloat16 turned into
/// floats exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// One product's shape: output {rows, columns} = L {rows, sum_length} · R {columns, sum_length}ᵀ, left and right
/// holding L and R as they sit, or turned round.
struct ProductShape
{
    const char *name_ptr;
    size_t rows_val;
    size_t columns_val;
    size_t sum_length_val;
    bool left_turned_val;
    bool right_turned_val;
    const char *setting_name_ptr;   // model_bf16.cuh's define of this product's setting; nullptr: not one of GPT-2's
};

/// GPT-2 small's 5 linear layers (a micro-batch of GPT2_ROWS rows), each with its forward product and the backward's
/// two (input_grad = output_grad · W: right turned; weight_grad = output_gradᵀ · x: both turned, the sum over the
/// GPT2_ROWS rows), then the worklog's 4096³.
static const ProductShape GPT2_SHAPES_LIST[] = {
    {"qkv", GPT2_ROWS, GPT2_QKV_DIM, GPT2_EMBED_DIM, false, false, "MODEL_BF16_QKV_SETTING"},
    {"qkv input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_QKV_DIM, false, true, "MODEL_BF16_QKV_INPUT_GRAD_SETTING"},
    {"qkv weight_grad", GPT2_QKV_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true, "MODEL_BF16_QKV_WEIGHT_GRAD_SETTING"},
    {"attention output", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, false, false, "MODEL_BF16_ATTENTION_OUTPUT_SETTING"},
    {"attention output input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, false, true, "MODEL_BF16_ATTENTION_OUTPUT_INPUT_GRAD_SETTING"},
    {"attention output weight_grad", GPT2_EMBED_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true, "MODEL_BF16_ATTENTION_OUTPUT_WEIGHT_GRAD_SETTING"},
    {"mlp up", GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, false, false, "MODEL_BF16_MLP_UP_SETTING"},
    {"mlp up input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, false, true, "MODEL_BF16_MLP_UP_INPUT_GRAD_SETTING"},
    {"mlp up weight_grad", GPT2_MLP_DIM, GPT2_EMBED_DIM, GPT2_ROWS, true, true, "MODEL_BF16_MLP_UP_WEIGHT_GRAD_SETTING"},
    {"mlp down", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, false, false, "MODEL_BF16_MLP_DOWN_SETTING"},
    {"mlp down input_grad", GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, false, true, "MODEL_BF16_MLP_DOWN_INPUT_GRAD_SETTING"},
    {"mlp down weight_grad", GPT2_EMBED_DIM, GPT2_MLP_DIM, GPT2_ROWS, true, true, "MODEL_BF16_MLP_DOWN_WEIGHT_GRAD_SETTING"},
    {"head", GPT2_ROWS, GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, false, false, "MODEL_BF16_HEAD_SETTING"},
    {"head input_grad", GPT2_ROWS, GPT2_EMBED_DIM, GPT2_PADDED_VOCAB_SIZE, false, true, "MODEL_BF16_HEAD_INPUT_GRAD_SETTING"},
    {"head weight_grad", GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, GPT2_ROWS, true, true, "MODEL_BF16_HEAD_WEIGHT_GRAD_SETTING"},
    {"worklog 4096^3", WORKLOG_SIZE, WORKLOG_SIZE, WORKLOG_SIZE, false, false, nullptr},
};

// ══ part 1: timed against PyTorch, mma_matmul_bf16 and fma_matmul_bf16 ═══════════════════════════════

/// Prints the averages of 20 products of one shape, steps 3..20, for each side, with TFLOPS and the share of this
/// GPU's bfloat16 peak (peak_bf16_flops); then checks mine against PyTorch's within BFLOAT16_TOLERANCE.
static void time_20_products_of_one_shape(const ProductShape &shape_val)
{
    // 1. random bfloat16 left and right as they sit; the sides' outputs
    torch::manual_seed(0);
    bool left_turned_val = shape_val.left_turned_val;
    bool right_turned_val = shape_val.right_turned_val;
    Tensor left_tensor = random_bf16_tensor(left_turned_val ? Shape{shape_val.sum_length_val, shape_val.rows_val}
                                                            : Shape{shape_val.rows_val, shape_val.sum_length_val});
    Tensor right_tensor = random_bf16_tensor(right_turned_val ? Shape{shape_val.sum_length_val, shape_val.columns_val}
                                                              : Shape{shape_val.columns_val, shape_val.sum_length_val});
    Tensor output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    Tensor other_output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
    torch::Tensor pytorch_left_tensor = pytorch_copy_of(left_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_right_tensor = pytorch_copy_of(right_tensor).to(torch::kBFloat16);
    torch::Tensor pytorch_left_matrix_tensor = left_turned_val ? pytorch_left_tensor.t() : pytorch_left_tensor;     // views, no copies:
    torch::Tensor pytorch_right_matrix_tensor = right_turned_val ? pytorch_right_tensor.t() : pytorch_right_tensor; // cuBLAS reads them turned
    torch::Tensor pytorch_output_tensor;

    // 2. time TIMED_STEPS products of each side; the first WARM_UP_STEPS do first-time work (loading kernels,
    //    cuBLAS picking its own), so the averages are of the rest
    const char *sides_list[] = {"mine (WGMMA + TMA)", "mma_matmul_bf16 (mma.sync)", "fma_matmul_bf16 (CUDA cores)", "PyTorch (cuBLAS)"};
    const int sides_val = (int)(sizeof(sides_list) / sizeof(sides_list[0]));
    double totals_list[sides_val] = {};
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float times_list[sides_val];
        times_list[0] = gpu_milliseconds([&] { wgmma_matmul_bf16(output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val, right_turned_val); });
        times_list[1] = gpu_milliseconds([&] { mma_matmul_bf16(other_output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val, right_turned_val); });
        times_list[2] = gpu_milliseconds([&] { fma_matmul_bf16(other_output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val, right_turned_val); });
        times_list[3] = gpu_milliseconds([&] {
            pytorch_output_tensor = pytorch_left_matrix_tensor.matmul(pytorch_right_matrix_tensor.t());
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        for (int side_index_val = 0; side_index_val < sides_val; side_index_val++)
            totals_list[side_index_val] += times_list[side_index_val];
    }

    // 3. print the averages: ms, TFLOPS, and the share of the peak
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double flops_val = (double)FLOPS_PER_MULTIPLY_ADD * shape_val.rows_val * shape_val.columns_val * shape_val.sum_length_val;
    double peak_val = peak_bf16_flops();
    print_test_name(__func__);
    printf("    %s: rows %zu, columns %zu, sum_length %zu%s%s: %.1f GFLOP\n", shape_val.name_ptr, shape_val.rows_val,
           shape_val.columns_val, shape_val.sum_length_val, left_turned_val ? ", left turned" : "",
           right_turned_val ? ", right turned" : "", flops_val / GIGA);
    printf("    average of steps %d..%d:                ms     TFLOPS   of bf16 peak\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    for (int side_index_val = 0; side_index_val < sides_val; side_index_val++)
    {
        double milliseconds_val = totals_list[side_index_val] / averaged_val;
        double tflops_val = flops_val / (milliseconds_val / MILLISECONDS_PER_SECOND) / TERA;
        printf("        %-30s %9.3f   %8.1f   %6.1f%%\n", sides_list[side_index_val], milliseconds_val, tflops_val,
               100.0 * tflops_val * TERA / peak_val);
    }
    printf("    PyTorch takes %.2fx as long as mine\n", totals_list[sides_val - 1] / totals_list[0]);

    // 4. mine agrees with PyTorch at the end
    torch::Tensor expected_tensor = pytorch_left_matrix_tensor.to(torch::kFloat).matmul(pytorch_right_matrix_tensor.to(torch::kFloat).t());
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

static void time_20_products_of_each_shape(void)
{
    // Prints, for each of GPT-2 small's 5 linear layers (a micro-batch of GPT2_ROWS rows), its forward product and
    // the backward's two (input_grad = output_grad · W: right turned; weight_grad = output_gradᵀ · x: both turned,
    // the sum over the GPT2_ROWS rows), then the worklog's 4096³, each side's times. The worklog reached 317 TFLOPS with this step's
    // kernel on an H100 SXM at 4096³ (cuBLAS 716); an H100 PCIe has about 3/4 of an SXM's tensor-core flops.
    for (const ProductShape &shape_val : GPT2_SHAPES_LIST)
        time_20_products_of_one_shape(shape_val);
}

// ══ part 2: the square's warpgroups, columns and sets of tiles swept ═══════════════════════

#define GPT2_LAYERS 12                // a step runs each layer's products once per layer, the head's once

/// One product at COLUMNS, STAGES and ROW_GROUPS, the layout, the splits and the block order given at run time:
/// what the sweeps time for each setting.
///
///   output_tensor, left_tensor, right_tensor, left_turned_val, right_turned_val   as in wgmma_matmul_bf16
///   splits_val, block_rows_per_group_val                                          as in wgmma_matmul_bf16_at
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void product_at(const Tensor &output_tensor, const Tensor &left_tensor, const Tensor &right_tensor, bool left_turned_val,
                       bool right_turned_val, int splits_val, int block_rows_per_group_val)
{
    wgmma_matmul_bf16_at<ROW_GROUPS, COLUMNS, STAGES>(output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val,
                                                      right_turned_val, WgmmaMatmulBf16Store(), splits_val, block_rows_per_group_val);
}

/// A setting of the product: its warpgroups (64 rows each), its square's columns, its sets of tiles, and the product
/// made for them.
struct ProductSetting
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*product_ptr)(const Tensor &, const Tensor &, const Tensor &, bool, bool, int, int);
};

static const ProductSetting PRODUCT_SETTINGS_LIST[] = {
    {1, 64, 1, product_at<64, 1, 1>}, {1, 64, 2, product_at<64, 2, 1>}, {1, 64, 3, product_at<64, 3, 1>}, {1, 64, 4, product_at<64, 4, 1>},
    {1, 128, 1, product_at<128, 1, 1>}, {1, 128, 2, product_at<128, 2, 1>}, {1, 128, 3, product_at<128, 3, 1>}, {1, 128, 4, product_at<128, 4, 1>},
    {1, 256, 1, product_at<256, 1, 1>}, {1, 256, 2, product_at<256, 2, 1>}, {1, 256, 3, product_at<256, 3, 1>}, {1, 256, 4, product_at<256, 4, 1>},
    {2, 64, 1, product_at<64, 1, 2>}, {2, 64, 2, product_at<64, 2, 2>}, {2, 64, 3, product_at<64, 3, 2>}, {2, 64, 4, product_at<64, 4, 2>},
    {2, 128, 1, product_at<128, 1, 2>}, {2, 128, 2, product_at<128, 2, 2>}, {2, 128, 3, product_at<128, 3, 2>}, {2, 128, 4, product_at<128, 4, 2>},
    {2, 256, 1, product_at<256, 1, 2>}, {2, 256, 2, product_at<256, 2, 2>}, {2, 256, 3, product_at<256, 3, 2>}, {2, 256, 4, product_at<256, 4, 2>},
};

#define GPT2_SHAPE_COUNT (sizeof(GPT2_SHAPES_LIST) / sizeof(GPT2_SHAPES_LIST[0]))
static int fastest_setting_index_list[GPT2_SHAPE_COUNT];        // part 2's fastest square of each product, for part 3
static double default_milliseconds_list[GPT2_SHAPE_COUNT];      // ... and the defaults' time, to compare part 3's with

static void time_each_setting_on_gpt2_products(void)
{
    // Prints, for each of GPT-2 small's 15 products, every setting's time (rows x columns x sets of tiles, the rows
    // 64 a warpgroup), taking turns
    // over SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged, and the fastest; then a step's worth of each
    // setting (each layer's products GPT2_LAYERS times, the head's once, for one micro-batch of GPT2_ROWS rows) and the
    // setting with the least, against the default's.
    const int setting_count_val = (int)(sizeof(PRODUCT_SETTINGS_LIST) / sizeof(PRODUCT_SETTINGS_LIST[0]));
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    std::vector<double> step_totals_list(setting_count_val, 0.0);
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, each setting rows x columns x sets of tiles\n    %-34s", WARM_UP_STEPS + 1, SWEEP_ROUNDS, "product");
    for (const ProductSetting &setting_val : PRODUCT_SETTINGS_LIST)
    {
        char label_list[32];
        snprintf(label_list, sizeof(label_list), "%dx%dx%d", setting_val.row_groups_val * HARDWARE_WGMMA_M, setting_val.columns_val,
                 setting_val.stages_val);
        printf(" %9s", label_list);
    }
    printf("   fastest\n");
    for (const ProductShape &shape_val : GPT2_SHAPES_LIST)
    {
        // 1. the step's weight of this product: none for the worklog's
        if (strncmp(shape_val.name_ptr, "worklog", 7) == 0)
            continue;
        double weight_val = strncmp(shape_val.name_ptr, "head", 4) == 0 ? 1.0 : (double)GPT2_LAYERS;
        // 2. random left and right, as they sit
        torch::manual_seed(0);
        Tensor left_tensor = random_bf16_tensor(shape_val.left_turned_val ? Shape{shape_val.sum_length_val, shape_val.rows_val}
                                                                          : Shape{shape_val.rows_val, shape_val.sum_length_val});
        Tensor right_tensor = random_bf16_tensor(shape_val.right_turned_val ? Shape{shape_val.sum_length_val, shape_val.columns_val}
                                                                            : Shape{shape_val.columns_val, shape_val.sum_length_val});
        Tensor output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
        // 3. SWEEP_ROUNDS rounds, each timing every setting once
        std::vector<double> totals_list(setting_count_val, 0.0);
        for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
            for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
            {
                const ProductSetting &setting_val = PRODUCT_SETTINGS_LIST[setting_index_val];
                float milliseconds_val = gpu_milliseconds([&] {
                    setting_val.product_ptr(output_tensor, left_tensor, right_tensor, shape_val.left_turned_val, shape_val.right_turned_val,
                                            WGMMA_MATMUL_BF16_SPLITS, WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP);
                });
                if (round_index_val > WARM_UP_STEPS)
                    totals_list[setting_index_val] += milliseconds_val;
            }
        // 4. print the row and its fastest; add to the step's totals
        int fastest_index_val = 0;
        printf("    %-34s", shape_val.name_ptr);
        for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
        {
            double milliseconds_val = totals_list[setting_index_val] / averaged_val;
            printf(" %9.3f", milliseconds_val);
            step_totals_list[setting_index_val] += weight_val * milliseconds_val;
            if (totals_list[setting_index_val] < totals_list[fastest_index_val])
                fastest_index_val = setting_index_val;
        }
        size_t shape_index_val = (size_t)(&shape_val - GPT2_SHAPES_LIST);
        fastest_setting_index_list[shape_index_val] = fastest_index_val;
        for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
            if (PRODUCT_SETTINGS_LIST[setting_index_val].row_groups_val == WGMMA_MATMUL_BF16_ROW_GROUPS &&
                PRODUCT_SETTINGS_LIST[setting_index_val].columns_val == WGMMA_MATMUL_BF16_COLUMNS &&
                PRODUCT_SETTINGS_LIST[setting_index_val].stages_val == WGMMA_MATMUL_BF16_STAGES)
                default_milliseconds_list[shape_index_val] = totals_list[setting_index_val] / averaged_val;
        printf("   %dx%dx%d\n", PRODUCT_SETTINGS_LIST[fastest_index_val].row_groups_val * HARDWARE_WGMMA_M,
               PRODUCT_SETTINGS_LIST[fastest_index_val].columns_val, PRODUCT_SETTINGS_LIST[fastest_index_val].stages_val);
    }

    // 5. a step's worth of each setting, the least, and the default's
    int best_index_val = 0;
    int default_index_val = 0;
    printf("    %-34s", "a step's products, one micro-batch");
    for (int setting_index_val = 0; setting_index_val < setting_count_val; setting_index_val++)
    {
        printf(" %9.2f", step_totals_list[setting_index_val]);
        if (step_totals_list[setting_index_val] < step_totals_list[best_index_val])
            best_index_val = setting_index_val;
        if (PRODUCT_SETTINGS_LIST[setting_index_val].row_groups_val == WGMMA_MATMUL_BF16_ROW_GROUPS &&
            PRODUCT_SETTINGS_LIST[setting_index_val].columns_val == WGMMA_MATMUL_BF16_COLUMNS &&
            PRODUCT_SETTINGS_LIST[setting_index_val].stages_val == WGMMA_MATMUL_BF16_STAGES)
            default_index_val = setting_index_val;
    }
    printf("\n    best for a step: %d warpgroups, %d columns, %d sets of tiles, %.2f ms a micro-batch; the default (%d x %d x %d) %.2f ms, %.2fx as long\n",
           PRODUCT_SETTINGS_LIST[best_index_val].row_groups_val, PRODUCT_SETTINGS_LIST[best_index_val].columns_val,
           PRODUCT_SETTINGS_LIST[best_index_val].stages_val, step_totals_list[best_index_val], WGMMA_MATMUL_BF16_ROW_GROUPS,
           WGMMA_MATMUL_BF16_COLUMNS, WGMMA_MATMUL_BF16_STAGES,
           step_totals_list[default_index_val], step_totals_list[default_index_val] / step_totals_list[best_index_val]);
}

// ══ part 3: the splits and the block order swept, at each product's square ════════

/// Makes one product's left and right, random, as they sit: what parts 3 and 4 time on.
///
///   shape_val   the product
static std::pair<Tensor, Tensor> random_operands_of(const ProductShape &shape_val)
{
    torch::manual_seed(0);
    Tensor left_tensor = random_bf16_tensor(shape_val.left_turned_val ? Shape{shape_val.sum_length_val, shape_val.rows_val}
                                                                      : Shape{shape_val.rows_val, shape_val.sum_length_val});
    Tensor right_tensor = random_bf16_tensor(shape_val.right_turned_val ? Shape{shape_val.sum_length_val, shape_val.columns_val}
                                                                        : Shape{shape_val.columns_val, shape_val.sum_length_val});
    return {left_tensor, right_tensor};
}

static void time_each_split_and_block_order_at_each_products_square(void)
{
    // Prints, for each of GPT-2 small's 15 products at its fastest square from part 2, the time of every split count
    // (1, 2, 4, 8, 16) with every block order (groups of 1, 4, 16, 64 rows of squares), taking turns over
    // SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged, and the fastest; then a step's worth (each layer's
    // products GPT2_LAYERS times, the head's once) at each product's fastest, against all at the defaults; then
    // model_bf16.cuh's 15 setting lines, to paste in.
    const int splits_list[] = {1, 2, 4, 8, 16};
    const int block_rows_per_group_list[] = {1, 4, 16, 64};
    const int splits_count_val = (int)(sizeof(splits_list) / sizeof(splits_list[0]));
    const int order_count_val = (int)(sizeof(block_rows_per_group_list) / sizeof(block_rows_per_group_list[0]));
    const int choice_count_val = splits_count_val * order_count_val;
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    int best_choice_index_list[GPT2_SHAPE_COUNT] = {};
    double default_step_val = 0.0;
    double best_step_val = 0.0;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, at each product's square; splits/rows of squares a group runs down\n    %-34s %9s",
           WARM_UP_STEPS + 1, SWEEP_ROUNDS, "product", "square");
    for (int splits_val : splits_list)
        for (int block_rows_per_group_val : block_rows_per_group_list)
        {
            char label_list[16];
            snprintf(label_list, sizeof(label_list), "%d/%d", splits_val, block_rows_per_group_val);
            printf(" %7s", label_list);
        }
    printf("   fastest\n");
    for (size_t shape_index_val = 0; shape_index_val < GPT2_SHAPE_COUNT; shape_index_val++)
    {
        // 1. the step's weight of this product: none for the worklog's; its square from part 2
        const ProductShape &shape_val = GPT2_SHAPES_LIST[shape_index_val];
        if (shape_val.setting_name_ptr == nullptr)
            continue;
        double weight_val = strncmp(shape_val.name_ptr, "head", 4) == 0 ? 1.0 : (double)GPT2_LAYERS;
        const ProductSetting &setting_val = PRODUCT_SETTINGS_LIST[fastest_setting_index_list[shape_index_val]];
        // 2. random left and right, as they sit
        std::pair<Tensor, Tensor> operands_val = random_operands_of(shape_val);
        Tensor output_tensor = Tensor::zeros({shape_val.rows_val, shape_val.columns_val}, DType::BFloat16);
        // 3. SWEEP_ROUNDS rounds, each timing every split count and order once
        std::vector<double> totals_list(choice_count_val, 0.0);
        for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
            for (int choice_index_val = 0; choice_index_val < choice_count_val; choice_index_val++)
            {
                float milliseconds_val = gpu_milliseconds([&] {
                    setting_val.product_ptr(output_tensor, operands_val.first, operands_val.second, shape_val.left_turned_val,
                                            shape_val.right_turned_val, splits_list[choice_index_val / order_count_val],
                                            block_rows_per_group_list[choice_index_val % order_count_val]);
                });
                if (round_index_val > WARM_UP_STEPS)
                    totals_list[choice_index_val] += milliseconds_val;
            }
        // 4. print the row and its fastest; add to the step's totals
        char square_list[32];
        snprintf(square_list, sizeof(square_list), "%dx%dx%d", setting_val.row_groups_val * HARDWARE_WGMMA_M, setting_val.columns_val,
                 setting_val.stages_val);
        printf("    %-34s %9s", shape_val.name_ptr, square_list);
        int fastest_index_val = 0;
        for (int choice_index_val = 0; choice_index_val < choice_count_val; choice_index_val++)
        {
            printf(" %7.3f", totals_list[choice_index_val] / averaged_val);
            if (totals_list[choice_index_val] < totals_list[fastest_index_val])
                fastest_index_val = choice_index_val;
        }
        best_choice_index_list[shape_index_val] = fastest_index_val;
        printf("   %d/%d\n", splits_list[fastest_index_val / order_count_val], block_rows_per_group_list[fastest_index_val % order_count_val]);
        default_step_val += weight_val * default_milliseconds_list[shape_index_val];
        best_step_val += weight_val * totals_list[fastest_index_val] / averaged_val;
    }

    // 5. a step's worth, and the lines for model_bf16.cuh
    printf("    a step's products, one micro-batch: %.2f ms at each product's fastest; %.2f ms all at the defaults, %.2fx as long\n",
           best_step_val, default_step_val, default_step_val / best_step_val);
    printf("    for llmc/model_bf16.cuh:\n");
    for (size_t shape_index_val = 0; shape_index_val < GPT2_SHAPE_COUNT; shape_index_val++)
    {
        const ProductShape &shape_val = GPT2_SHAPES_LIST[shape_index_val];
        if (shape_val.setting_name_ptr == nullptr)
            continue;
        const ProductSetting &setting_val = PRODUCT_SETTINGS_LIST[fastest_setting_index_list[shape_index_val]];
        int choice_index_val = best_choice_index_list[shape_index_val];
        printf("#define %s WgmmaMatmulBf16Setting<%d, %d, %d, %d, %d>\n", shape_val.setting_name_ptr, setting_val.row_groups_val,
               setting_val.columns_val, setting_val.stages_val, splits_list[choice_index_val / order_count_val],
               block_rows_per_group_list[choice_index_val % order_count_val]);
    }
}

// ══ part 4: the splits' sum's block size swept ═════════════════════════════════

static void time_each_splits_sum_block_size(void)
{
    // Prints the time of a split product at each of the splits' sum's block sizes (64 to 1024 threads), taking
    // turns over SWEEP_ROUNDS rounds, rounds WARM_UP_STEPS + 1 on averaged: GPT-2's attention output weight_grad
    // (768 x 768 out of GPT2_ROWS values of k) and mlp up weight_grad (3072 x 768), split in 8 at the default square;
    // then the fastest for both against the default (WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK). The product's
    // own time is the same in every column, so the differences are the sum's.
    const int threads_list[] = {64, 128, 256, 512, 1024};
    const int threads_count_val = (int)(sizeof(threads_list) / sizeof(threads_list[0]));
    const int splits_val = 8;
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    const char *names_list[] = {"attention output weight_grad", "mlp up weight_grad"};
    std::vector<double> both_list(threads_count_val, 0.0);
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, %d splits; threads a block of the splits' sum\n    %-34s", WARM_UP_STEPS + 1, SWEEP_ROUNDS,
           splits_val, "product");
    for (int threads_per_block_val : threads_list)
        printf(" %7d", threads_per_block_val);
    printf("   fastest\n");
    for (const char *name_ptr : names_list)
    {
        // 1. the product's shape and its operands
        const ProductShape *shape_ptr = nullptr;
        for (const ProductShape &shape_val : GPT2_SHAPES_LIST)
            if (strcmp(shape_val.name_ptr, name_ptr) == 0)
                shape_ptr = &shape_val;
        std::pair<Tensor, Tensor> operands_val = random_operands_of(*shape_ptr);
        Tensor output_tensor = Tensor::zeros({shape_ptr->rows_val, shape_ptr->columns_val}, DType::BFloat16);
        // 2. SWEEP_ROUNDS rounds, each timing every block size once
        std::vector<double> totals_list(threads_count_val, 0.0);
        for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
            for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
            {
                float milliseconds_val = gpu_milliseconds([&] {
                    wgmma_matmul_bf16_at<WGMMA_MATMUL_BF16_ROW_GROUPS, WGMMA_MATMUL_BF16_COLUMNS, WGMMA_MATMUL_BF16_STAGES>(
                        output_tensor, operands_val.first, operands_val.second, Tensor(), false, shape_ptr->left_turned_val,
                        shape_ptr->right_turned_val, WgmmaMatmulBf16Store(), splits_val, WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP,
                        threads_list[threads_index_val]);
                });
                if (round_index_val > WARM_UP_STEPS)
                    totals_list[threads_index_val] += milliseconds_val;
            }
        // 3. the row and its fastest
        int fastest_index_val = 0;
        printf("    %-34s", name_ptr);
        for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
        {
            printf(" %7.3f", totals_list[threads_index_val] / averaged_val);
            both_list[threads_index_val] += totals_list[threads_index_val] / averaged_val;
            if (totals_list[threads_index_val] < totals_list[fastest_index_val])
                fastest_index_val = threads_index_val;
        }
        printf("   %d\n", threads_list[fastest_index_val]);
    }
    // 4. the two together: the fastest, and the default's
    int best_index_val = 0;
    int default_index_val = 0;
    for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
    {
        if (both_list[threads_index_val] < both_list[best_index_val])
            best_index_val = threads_index_val;
        if (threads_list[threads_index_val] == WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK)
            default_index_val = threads_index_val;
    }
    printf("    best for both: %d threads, %.3f ms; the default (%d) %.3f ms, %.2fx as long\n", threads_list[best_index_val],
           both_list[best_index_val], WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK, both_list[default_index_val],
           both_list[default_index_val] / both_list[best_index_val]);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    time_20_products_of_each_shape();
    time_each_setting_on_gpt2_products();
    time_each_split_and_block_order_at_each_products_square();
    time_each_splits_sum_block_size();

    printf("\nall tunes done\n");
    return 0;
}
