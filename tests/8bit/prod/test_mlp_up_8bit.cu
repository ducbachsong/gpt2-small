// test_mlp_up_8bit.cu — tests for llmc/kernels/8bit/prod/mlp_up_8bit.cuh (mlp_up_8bit): mlp_up_output = ln2_output · mlp_up_weightᵀ + mlp_up_bias, and gelu_output = gelu(mlp_up_output) beside it, the product in 8 bits.
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks against PyTorch: on the same rounded values (the E4M3 copies read back as floats) within bfloat16's
// rounding, and on the bfloat16 values within E4M3's; part 2 checks the bits: the same output as wgmma_matmul_8bit (the
// general 8-bit product) on the same copies at each promotion (gelu_output within a bfloat16 step of its: mine takes
// tanh from tanh.approx.f32, its epilogue from tanhf), and the same at every square, block order, block count,
// cluster, variant and ping-pong square; gelu_output in 8 bits (mlp_up_8bit_gelu_8bit) gives the bits of the
// bfloat16 gelu_output rounded by round_8bit_per_tensor at the same delayed scale, at every square. Part 3 checks
// that the kernels' registers fit their block. Run it on an H100 (compute capability 9.0), built with -arch=sm_90a.
// The timings are in tune_mlp_up_8bit.cu.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <string>
#include <vector>
#include "../../test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../../llmc/kernels/8bit/prod/mlp_up_8bit.cuh"
#include "../../../llmc/kernels/8bit/wgmma_matmul_8bit.cuh"   // the reference: the general 8-bit product
#include "../../../llmc/kernels/bf16/gelu_bf16.cuh"           // ... with GELU's epilogues, as gelu_8bit.cuh passes them

// Against PyTorch: on the same rounded values, the sums are float on both sides, added in another order, then mine are
// rounded to bfloat16 (2^-9 of a value); on the bfloat16 values, each E4M3 is off by up to 2^-4 (6%) of its value too.
// Each tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double EIGHT_BIT_TOLERANCE = 1e-1;
// gelu_output against the general kernel's on the same sums: tanh.approx.f32 is about 2^-11 off, so after bfloat16's
// rounding a value lands at most a step (2^-8 of it) away; as a fraction of the largest value.
static const double APPROXIMATE_TANH_TOLERANCE = 1e-2;

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define ODD_ROWS 208                  // sizes no square divides, multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE: 3 x 64 + 16 rows, ...
#define ODD_COLUMNS 272               // ... 4 x 64 + 16 columns ...
#define ODD_SUM_LENGTH 1040           // ... and 8 steps of 128 values of k + 16
#define MOST_SPILL_OF_THE_LARGEST_SQUARE 64   // bytes the largest squares' kernels may spill (see every_kernel_fits_its_block): 2 x 128 x 2 sets, clusters of 1, spilled 64 on an H100 PCIe 2026-11-15 (2 x 128 x 2, clusters of 2, and the other sets: 0); the tune times it as it is
#define MOST_SPILL_AT_256_COLUMNS 64          // ... and at 256 columns: 128 sums a thread with GELU's math beside them (bf16's mlp_up spilled 16)
#define COOPERATIVE_STAGES (MLP_UP_8BIT_COLUMNS == 256 ? 3 : MLP_UP_8BIT_STAGES)   // the cooperative walk at this file's square: 2 x 256 fits 3 sets, not 4 (two left tiles a set)

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU, then rounded to bfloat16.
static Tensor random_tensor(const Shape &shape_list)
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

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value, worked
/// out on the GPU.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives a copy's rounded values turned back by PyTorch: each E4M3 (read as float8_e4m3fn) times its row's scale,
/// exact. {rows, k} float.
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scales_tensor)
{
    int64_t rows_val = (int64_t)rounded_tensor.size(0), k_val = (int64_t)rounded_tensor.size(1);
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, k_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scales_tensor).unsqueeze(1);
}

/// The product's operands, as mlp_up_8bit takes them, for output {rows, columns} with a sum of sum_length, and their copies.
struct Operands
{
    Tensor input_tensor;              // {rows, sum_length}
    Tensor weight_tensor;             // {columns, sum_length}
    Tensor bias_tensor;               // {columns}
    MlpUp8bitCopies copies_val;              // the 8-bit copies, rounded once by mlp_up_8bit_round
};

/// What the product writes.
struct Outputs
{
    Tensor output_tensor;             // {rows, columns}
    Tensor gelu_output_tensor;        // {rows, columns}: gelu of output
};

/// Makes random operands for output {rows, columns} with a sum of sum_length, from seed_val, and rounds them.
static Operands random_operands(size_t rows_val, size_t columns_val, size_t sum_length_val, uint64_t seed_val)
{
    torch::manual_seed(seed_val);
    Operands operands_val;
    operands_val.input_tensor = random_tensor({rows_val, sum_length_val});
    operands_val.weight_tensor = random_tensor({columns_val, sum_length_val});
    operands_val.bias_tensor = random_tensor({columns_val});
    operands_val.copies_val = MlpUp8bitCopies(rows_val, columns_val, sum_length_val);
    mlp_up_8bit_round(operands_val.copies_val, operands_val.input_tensor, operands_val.weight_tensor);
    return operands_val;
}

/// Makes the outputs for output {rows, columns}, filled with 7s
/// so a value not written shows.
static Outputs fresh_outputs(const Operands &operands_val, size_t rows_val, size_t columns_val)
{
    Outputs outputs_val;
    outputs_val.output_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
    outputs_val.output_tensor.fill_(7.0);
    outputs_val.gelu_output_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
    outputs_val.gelu_output_tensor.fill_(7.0);
    return outputs_val;
}

/// Runs mlp_up_8bit's product on the operands' copies at a setting given (the template choices not given: this file's).
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_8BIT_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_UP_8BIT_IN_FLIGHT, int PROMOTION_STEPS = MLP_UP_8BIT_PROMOTION_STEPS, int PINGPONG = MLP_UP_8BIT_PINGPONG
          >
static void mine_at(const Outputs &outputs_val, const Operands &operands_val,
                    int block_rows_per_group_val, int blocks_val)
{
    mlp_up_8bit_rounded_at<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, PINGPONG>(
        outputs_val.output_tensor, outputs_val.gelu_output_tensor, operands_val.bias_tensor, operands_val.copies_val, block_rows_per_group_val, blocks_val);
}

/// Runs mlp_up_8bit at this file's setting (its defines): rounds the operands (again, the same copies) and the product.
static void mine(const Outputs &outputs_val, const Operands &operands_val)
{
    mlp_up_8bit(outputs_val.output_tensor, outputs_val.gelu_output_tensor, operands_val.input_tensor, operands_val.weight_tensor, operands_val.bias_tensor, operands_val.copies_val);
}

/// Runs the same product with wgmma_matmul_8bit (the general 8-bit kernel) on the same copies, at promotion_steps_val:
/// the bits mine must give unsplit.
static void reference(const Outputs &outputs_val, const Operands &operands_val, int promotion_steps_val)
{
    const MlpUp8bitCopies &copies_val = operands_val.copies_val;
    wgmma_matmul_8bit(outputs_val.output_tensor, copies_val.left_8bit_tensor, copies_val.left_scales_tensor, copies_val.right_8bit_tensor,
                      copies_val.right_scales_tensor, operands_val.bias_tensor, false, GeluBf16ForwardEpilogue{outputs_val.gelu_output_tensor.data_ptr<nv_bfloat16>()}, promotion_steps_val);
}

/// Gives PyTorch's float product of the same rounded values (the copies read back): the output mine must be near
/// within bfloat16's rounding.
static torch::Tensor pytorch_on_rounded(const Operands &operands_val)
{
    const MlpUp8bitCopies &copies_val = operands_val.copies_val;
    torch::Tensor expected_tensor = turned_back(copies_val.left_8bit_tensor, copies_val.left_scales_tensor)
                                        .matmul(turned_back(copies_val.right_8bit_tensor, copies_val.right_scales_tensor).t());
    if (operands_val.bias_tensor.numel() > 0)                                // Tensor(): no bias
        expected_tensor = expected_tensor + pytorch_copy_of(operands_val.bias_tensor);
    return expected_tensor;
}

/// Gives PyTorch's float product of the bfloat16 values themselves: the output mine must be near within E4M3's rounding.
static torch::Tensor pytorch_on_bf16(const Operands &operands_val)
{
    torch::Tensor expected_tensor = pytorch_copy_of(operands_val.input_tensor).matmul(pytorch_copy_of(operands_val.weight_tensor).t());
    if (operands_val.bias_tensor.numel() > 0)                                // Tensor(): no bias
        expected_tensor = expected_tensor + pytorch_copy_of(operands_val.bias_tensor);
    return expected_tensor;
}

/// True when two runs' outputs hold the same bits.
static bool same_bits(const Outputs &first_val, const Outputs &second_val)
{
    bool same_val = torch::equal(pytorch_copy_of(first_val.output_tensor), pytorch_copy_of(second_val.output_tensor));
    same_val = same_val && torch::equal(pytorch_copy_of(first_val.gelu_output_tensor), pytorch_copy_of(second_val.gelu_output_tensor));
    return same_val;
}

/// True when mine and the general kernel's give the same output's bits and gelu_output within
/// APPROXIMATE_TANH_TOLERANCE: the same sums, but tanh by tanh.approx.f32 in mine and by tanhf in its epilogue.
static bool same_output_and_gelu_near(const Outputs &mine_val, const Outputs &reference_val)
{
    bool same_val = torch::equal(pytorch_copy_of(mine_val.output_tensor), pytorch_copy_of(reference_val.output_tensor));
    double gelu_difference_val = relative_difference_on_gpu(mine_val.gelu_output_tensor, pytorch_copy_of(reference_val.gelu_output_tensor));
    return same_val && gelu_difference_val <= APPROXIMATE_TANH_TOLERANCE;
}

/// Gives a tensor's bytes, copied from the GPU: for comparing 8-bit copies and their scales bit for bit.
static std::vector<uint8_t> bytes_of(const Tensor &tensor)
{
    std::vector<uint8_t> bytes_list(tensor.numel() * tensor.element_size());
    cudaCheck(cudaMemcpy(bytes_list.data(), tensor.data_ptr(), bytes_list.size(), cudaMemcpyDeviceToHost));
    return bytes_list;
}

/// Gives gelu'(x) of GPT-2's tanh GELU, by PyTorch: 0.5 (1 + tanh u) + 0.5 x (1 - tanh² u) √(2/π) (1 + 3 · 0.044715 x²),
/// u = √(2/π) (x + 0.044715 x³).
static torch::Tensor pytorch_gelu_slope(const torch::Tensor &x_tensor)
{
    const double root_val = std::sqrt(2.0 / M_PI), cube_val = 0.044715;
    torch::Tensor tanh_tensor = torch::tanh(root_val * (x_tensor + cube_val * x_tensor * x_tensor * x_tensor));
    return 0.5 * (1.0 + tanh_tensor) + 0.5 * x_tensor * (1.0 - tanh_tensor * tanh_tensor) * root_val * (1.0 + 3.0 * cube_val * x_tensor * x_tensor);
}

// ══ part 1: against PyTorch ═══════════════════════════════════════════════════

/// Runs mlp_up_8bit at its setting on random operands for output {rows, columns} with a sum of sum_length, and checks it
/// against PyTorch's float product of the same rounded values within BFLOAT16_TOLERANCE.
static void matches_pytorch_on_the_rounded_values(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val)
{
    Operands operands_val = random_operands(rows_val, columns_val, sum_length_val, 0);
    Outputs outputs_val = fresh_outputs(operands_val, rows_val, columns_val);
    mine(outputs_val, operands_val);
    torch::Tensor expected_tensor = pytorch_on_rounded(operands_val);
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu\n", rows_val, columns_val, sum_length_val);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.gelu_output_tensor, torch::gelu(expected_tensor, "tanh")), 0.0, BFLOAT16_TOLERANCE);
}

static void matches_pytorch_on_gpt2_size(void)
{
    // Expect: GPT-2 small's mlp up on a micro-batch of GPT2_ROWS rows, within 1e-2 of PyTorch's float product of
    // the same rounded values.
    matches_pytorch_on_the_rounded_values(__func__, GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM);
}

static void matches_pytorch_with_sizes_no_tile_divides(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, no square nor step dividing them: TMA gives
    // 0s past the edges and stores nothing there; within 1e-2 of PyTorch on the same rounded values.
    matches_pytorch_on_the_rounded_values(__func__, ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH);
}

static void matches_pytorch_on_the_bfloat16_values(void)
{
    // Expect: GPT-2's size, within 1e-1 of PyTorch's float product of the bfloat16 values themselves: each E4M3 is off by
    // up to 6% of its value, and the sums average much of it out.
    Operands operands_val = random_operands(GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, 1);
    Outputs outputs_val = fresh_outputs(operands_val, GPT2_ROWS, GPT2_MLP_DIM);
    mine(outputs_val, operands_val);
    print_test_name(__func__);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, pytorch_on_bf16(operands_val)), 0.0, EIGHT_BIT_TOLERANCE);
}

static void matches_pytorch_and_the_general_kernel_with_no_bias(void)
{
    // Expect: bias Tensor(), as a model with no linear bias passes it (GPT2Config::linear_bias false): nothing is
    // added. Output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH: within 1e-2 of PyTorch's float product of
    // the same rounded values with no bias (and gelu_output, GELU of it), and the same output's bits as wgmma_matmul_8bit given no
    // bias on the same copies at MLP_UP_8BIT_PROMOTION_STEPS (gelu_output within APPROXIMATE_TANH_TOLERANCE of its).
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 3);
    operands_val.bias_tensor = Tensor();
    Outputs mine_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    Outputs reference_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    mine(mine_val, operands_val);
    reference(reference_val, operands_val, MLP_UP_8BIT_PROMOTION_STEPS);
    torch::Tensor expected_tensor = pytorch_on_rounded(operands_val);
    print_test_name(__func__);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.gelu_output_tensor, torch::gelu(expected_tensor, "tanh")), 0.0, BFLOAT16_TOLERANCE);
    ASSERT_EQ(same_output_and_gelu_near(mine_val, reference_val), true);
}

// ══ part 2: the bits ══════════════════════════════════════════════════════════

/// Runs the product at this file's square at promotion PROMOTION_STEPS, unsplit; a promotion with 256 columns (no
/// room for its second set of sums) at 128.
template <int PROMOTION_STEPS>
static void at_promotion(const Outputs &outputs_val, const Operands &operands_val)
{
    constexpr int COLUMNS = PROMOTION_STEPS > 0 && MLP_UP_8BIT_COLUMNS > 128 ? 128 : MLP_UP_8BIT_COLUMNS;
    mine_at<MLP_UP_8BIT_ROW_GROUPS, COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, PROMOTION_STEPS>(
        outputs_val, operands_val, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

static void every_promotion_gives_wgmma_matmul_8bits_bits(void)
{
    // Expect: at GPT-2 small's size and at sizes no square divides, mlp_up_8bit at its square, unsplit, gives the same output's
    // bits as wgmma_matmul_8bit on the same copies at the same promotion (0: never, 1, 2, 4, 8 steps): the same sums in the
    // same order, the tensor cores' sums moved at the same steps, times the scales the same way; gelu_output within
    // APPROXIMATE_TANH_TOLERANCE of its GELU epilogue's (tanh.approx.f32 against tanhf). So training gives the numbers it
    // gave with wgmma_matmul_8bit, but GELU's last bit.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM}, {ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH}};
    void (*runs_list[5])(const Outputs &, const Operands &) = {at_promotion<0>, at_promotion<1>, at_promotion<2>, at_promotion<4>, at_promotion<8>};
    const int promotions_list[5] = {0, 1, 2, 4, 8};
    for (const auto &size_val : sizes_list)
    {
        Operands operands_val = random_operands(size_val[0], size_val[1], size_val[2], 3);
        for (int promotion_index_val = 0; promotion_index_val < 5; promotion_index_val++)
        {
            Outputs mine_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
            Outputs reference_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
            runs_list[promotion_index_val](mine_val, operands_val);
            reference(reference_val, operands_val, promotions_list[promotion_index_val]);
            printf("    rows %zu, columns %zu, sum_length %zu, promotion every %d steps:\n", size_val[0], size_val[1], size_val[2],
                   promotions_list[promotion_index_val]);
            ASSERT_EQ(same_output_and_gelu_near(mine_val, reference_val), true);
        }
    }
}

/// One square (warpgroups, columns, sets of tiles) run in a cluster of cluster_val, cooperative: what the sweeps below
/// call (ping-pong's: pingpong_at).
template <int ROW_GROUPS, int COLUMNS, int STAGES>
static void square_at(const Outputs &outputs_val, const Operands &operands_val, int cluster_val,
                      int block_rows_per_group_val, int blocks_val)
{
    if (cluster_val == 2)
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 2, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 0>(
            outputs_val, operands_val, block_rows_per_group_val, blocks_val);
    else
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 1, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 0>(
            outputs_val, operands_val, block_rows_per_group_val, blocks_val);
}

/// A square that fits a block's shared memory, and its run.
struct Square
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Outputs &, const Operands &, int, int, int);
};

/// Every square that fits a block's shared memory: 1 or 2 warpgroups x 64 or 128 columns x 1 to 4 sets of tiles, and with
/// promotion 0, 256 columns (1 warpgroup x 1 to 4 sets, 2 x 1 to 3).
static const Square SQUARES_LIST[] = {
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

static void every_square_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH: every square gives the bits of this file's:
    // each value's sum walks k in the same order whatever the square.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 4);
    print_test_name(__func__);
    Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    mine(default_val, operands_val);
    for (const Square &square_val : SQUARES_LIST)
    {
        Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        square_val.run_ptr(outputs_val, operands_val, 1, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
        printf("    %d warpgroups x %d columns x %d sets:\n", square_val.row_groups_val, square_val.columns_val, square_val.stages_val);
        ASSERT_EQ(same_bits(outputs_val, default_val), true);
    }
}

static void every_order_block_count_and_cluster_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's square: every block order (1,
    // 4, 16, 64 rows of squares a group), block count (as many as run at once, 1, 3, one a square) and cluster (1, 2)
    // gives the bits of this file's setting: who works out a square changes, never a sum. The cooperative walk at
    // COOPERATIVE_STAGES sets (ping-pong's orders and counts: pingpong_gives_the_default_bits).
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 5);
    print_test_name(__func__);
    {
        Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine(default_val, operands_val);
        for (int cluster_val : {1, 2})
            for (int block_rows_per_group_val : {1, 4, 16, 64})
                for (int blocks_val : {0, 1, 3, MLP_UP_8BIT_ONE_A_SQUARE})
                {
                    Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
                    square_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, COOPERATIVE_STAGES>(outputs_val, operands_val, cluster_val, block_rows_per_group_val,
                                                                     blocks_val);
                    printf("    clusters of %d, %d rows a group, %d blocks:\n", cluster_val, block_rows_per_group_val, blocks_val);
                    ASSERT_EQ(same_bits(outputs_val, default_val), true);
                }
    }
}

/// One ping-pong square (2 warpgroups x COLUMNS x STAGES, clusters of 1) at a block order and block count: what the
/// ping-pong check below calls.
template <int COLUMNS, int STAGES>
static void pingpong_at(const Outputs &outputs_val, const Operands &operands_val, int block_rows_per_group_val, int blocks_val)
{
    mine_at<2, COLUMNS, STAGES, 1, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, 1>(
        outputs_val, operands_val, block_rows_per_group_val, blocks_val);
}

/// A ping-pong square and its run.
struct PingpongSquare
{
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Outputs &, const Operands &, int, int);
};

/// The ping-pong squares: 128 columns, or 256 with promotion 0, x 2 to 4 sets (256 x 4 fits: a set holds one left tile).
static const PingpongSquare PINGPONG_SQUARES_LIST[] = {
    {128, 2, pingpong_at<128, 2>}, {128, 3, pingpong_at<128, 3>}, {128, 4, pingpong_at<128, 4>},
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
    {256, 2, pingpong_at<256, 2>}, {256, 3, pingpong_at<256, 3>}, {256, 4, pingpong_at<256, 4>},
#endif
};

static void pingpong_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, and GPT-2's size: every ping-pong square
    // (the 2 warpgroups each on their own 64-row squares, their sums in turns), at block orders 1 and 4 and block
    // counts as many as run at once, 1 (one block takes every square, its warpgroups taking turns over all of them),
    // 3 (a warpgroup with one square fewer) and one a square, gives the bits of this file's setting: each value's sum
    // is the same 64 x k walk in the same order, only who works it out and when changes.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH}, {GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM}};
    for (const auto &size_val : sizes_list)
    {
        Operands operands_val = random_operands(size_val[0], size_val[1], size_val[2], 7);
        Outputs default_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
        mine(default_val, operands_val);
        for (const PingpongSquare &square_val : PINGPONG_SQUARES_LIST)
            for (int block_rows_per_group_val : {1, 4})
                for (int blocks_val : {0, 1, 3, MLP_UP_8BIT_ONE_A_SQUARE})
                {
                    if (size_val[0] == GPT2_ROWS && blocks_val == 1)
                        continue;                                // one block for the whole of GPT-2's size: slow, and 3 covers it
                    Outputs outputs_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
                    square_val.run_ptr(outputs_val, operands_val, block_rows_per_group_val, blocks_val);
                    printf("    rows %zu: ping-pong 2 x %d x %d sets, %d rows a group, %d blocks:\n", size_val[0], square_val.columns_val,
                           square_val.stages_val, block_rows_per_group_val, blocks_val);
                    ASSERT_EQ(same_bits(outputs_val, default_val), true);
                }
    }
}

// ── gelu_output in 8 bits ─────────────────────────────────────────────────────

/// gelu_output's two 8-bit copies, one scale for the tensor, as mlp_up_8bit_gelu_8bit and round_8bit_per_tensor write
/// them, for output {rows, columns}; filled with 7s first, so a byte not written shows.
struct Gelu8bitCopies
{
    Tensor rows_8bit_tensor;          // {rows, columns} E4M3
    Tensor row_scales_tensor;         // {rows}
    Tensor turned_8bit_tensor;        // {columns, rows} E4M3
    Tensor turned_scales_tensor;      // {columns}
    Round8bitHistory history_val;     // the place's history: where the delayed scale comes from

    Gelu8bitCopies(size_t rows_val, size_t columns_val)
        : rows_8bit_tensor(Tensor::zeros({rows_val, columns_val}, DType::E4M3)),
          row_scales_tensor(Tensor::zeros({rows_val})),
          turned_8bit_tensor(Tensor::zeros({columns_val, rows_val}, DType::E4M3)),
          turned_scales_tensor(Tensor::zeros({columns_val}))
    {
        cudaCheck(cudaMemset(rows_8bit_tensor.data_ptr(), 7, rows_8bit_tensor.numel() * rows_8bit_tensor.element_size()));
        cudaCheck(cudaMemset(turned_8bit_tensor.data_ptr(), 7, turned_8bit_tensor.numel() * turned_8bit_tensor.element_size()));
        row_scales_tensor.fill_(7.0);
        turned_scales_tensor.fill_(7.0);
    }
};

/// Runs mlp_up_8bit_gelu_8bit on the operands' copies at a setting given, its scale the history's delayed one
/// (round_8bit_delayed_largest): what the model's forward does after a layer's first micro-batch.
template <int ROW_GROUPS, int COLUMNS, int STAGES, int PINGPONG>
static void gelu_8bit_at(const Tensor &gelu_input_tensor, Gelu8bitCopies &copies_val, const Operands &operands_val)
{
    MlpUp8bitGelu8bit gelu_val;
    gelu_val.rows_8bit_tensor = copies_val.rows_8bit_tensor;
    gelu_val.row_scales_tensor = copies_val.row_scales_tensor;
    gelu_val.turned_8bit_tensor = copies_val.turned_8bit_tensor;
    gelu_val.turned_scales_tensor = copies_val.turned_scales_tensor;
    gelu_val.margin_val = ROUND_8BIT_DELAYED_MARGIN;
    if (!round_8bit_delayed_largest(copies_val.history_val, gelu_val.scale_largest_ptr, gelu_val.next_largest_ptr))
    {
        fprintf(stderr, "gelu_8bit_at needs a history with a call already\n");
        exit(EXIT_FAILURE);
    }
    mlp_up_8bit_gelu_8bit_at<ROW_GROUPS, COLUMNS, STAGES, 1, MLP_UP_8BIT_PRODUCER_REGISTERS, MLP_UP_8BIT_IN_FLIGHT, MLP_UP_8BIT_PROMOTION_STEPS, PINGPONG>(
        gelu_input_tensor, gelu_val, operands_val.bias_tensor, operands_val.copies_val, MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// True when two sets of gelu_output's 8-bit copies hold the same bytes, scales and next largest (the history's slot
/// this call filled).
static bool same_gelu_8bit_bits(const Gelu8bitCopies &first_val, const Gelu8bitCopies &second_val)
{
    return bytes_of(first_val.rows_8bit_tensor) == bytes_of(second_val.rows_8bit_tensor) &&
           bytes_of(first_val.turned_8bit_tensor) == bytes_of(second_val.turned_8bit_tensor) &&
           bytes_of(first_val.row_scales_tensor) == bytes_of(second_val.row_scales_tensor) &&
           bytes_of(first_val.turned_scales_tensor) == bytes_of(second_val.turned_scales_tensor) &&
           bytes_of(first_val.history_val.largest_tensor) == bytes_of(second_val.history_val.largest_tensor);
}

static void gelu_8bit_gives_round_8bit_per_tensors_bits(void)
{
    // Expect: at GPT-2's size and at sizes no square divides, at this file's setting, mlp_up_8bit_gelu_8bit writes
    // gelu_input with the bits mlp_up_8bit writes, and gelu_output's two 8-bit copies, their scales lists and the
    // history's next largest with the bits round_8bit_per_tensor gives from mlp_up_8bit's bfloat16 gelu_output at the
    // same delayed scale (the tensor's largest of the call before, x 2): each value is GELU rounded to bfloat16,
    // divided by the same power of 2 and rounded to E4M3 the same way. So the model trains as it did with a bfloat16
    // gelu_output and a rounding pass, without them.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM}, {ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH}};
    for (const auto &size_val : sizes_list)
    {
        size_t rows_val = size_val[0], columns_val = size_val[1];
        Operands operands_val = random_operands(rows_val, columns_val, size_val[2], 8);
        // 1. the reference: bfloat16 gelu_output, rounded twice by round_8bit_per_tensor (the second at the delayed scale)
        Outputs outputs_val = fresh_outputs(operands_val, rows_val, columns_val);
        mine(outputs_val, operands_val);
        Gelu8bitCopies reference_val(rows_val, columns_val);
        round_8bit_per_tensor(reference_val.rows_8bit_tensor, reference_val.row_scales_tensor, reference_val.turned_8bit_tensor,
                              reference_val.turned_scales_tensor, outputs_val.gelu_output_tensor, &reference_val.history_val);
        round_8bit_per_tensor(reference_val.rows_8bit_tensor, reference_val.row_scales_tensor, reference_val.turned_8bit_tensor,
                              reference_val.turned_scales_tensor, outputs_val.gelu_output_tensor, &reference_val.history_val);
        // 2. mine: a history with the same first call, then mlp_up_8bit_gelu_8bit into fresh copies
        Gelu8bitCopies mine_val(rows_val, columns_val);
        Gelu8bitCopies first_call_val(rows_val, columns_val);
        round_8bit_per_tensor(first_call_val.rows_8bit_tensor, first_call_val.row_scales_tensor, first_call_val.turned_8bit_tensor,
                              first_call_val.turned_scales_tensor, outputs_val.gelu_output_tensor, &mine_val.history_val);
        Tensor gelu_input_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
        gelu_input_tensor.fill_(7.0);
        gelu_8bit_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_PINGPONG>(gelu_input_tensor, mine_val, operands_val);
        printf("    rows %zu, columns %zu, sum_length %zu:\n", rows_val, columns_val, size_val[2]);
        ASSERT_EQ(torch::equal(pytorch_copy_of(gelu_input_tensor), pytorch_copy_of(outputs_val.output_tensor)), true);
        ASSERT_EQ(same_gelu_8bit_bits(mine_val, reference_val), true);
        ASSERT_EQ(mine_val.history_val.calls_val, reference_val.history_val.calls_val);
    }
}

/// One square run with gelu_output in 8 bits, and its run.
struct Gelu8bitSquare
{
    const char *name_ptr;
    void (*run_ptr)(const Tensor &, Gelu8bitCopies &, const Operands &);
};

/// The squares with gelu_output in 8 bits: cooperative 1 x 128 x 4, 2 x 128 x 4 and, with promotion 0, 2 x 256 x 3;
/// ping-pong 2 x 128 x 4 and, with promotion 0, 2 x 256 x 4.
static const Gelu8bitSquare GELU_8BIT_SQUARES_LIST[] = {
    {"cooperative 1 x 128 x 4", gelu_8bit_at<1, 128, 4, 0>},
    {"cooperative 2 x 128 x 4", gelu_8bit_at<2, 128, 4, 0>},
    {"ping-pong 2 x 128 x 4", gelu_8bit_at<2, 128, 4, 1>},
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
    {"cooperative 2 x 256 x 3", gelu_8bit_at<2, 256, 3, 0>},
    {"ping-pong 2 x 256 x 4", gelu_8bit_at<2, 256, 4, 1>},
#endif
};

static void gelu_8bit_at_every_square_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH: with gelu_output in 8 bits, every square
    // (cooperative and ping-pong, 128 and 256 columns) gives the bits of this file's setting, its two copies, scales
    // and next largest: who writes a value's byte, into which half of which room, changes; never the byte.
    print_test_name(__func__);
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 9);
    Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    mine(outputs_val, operands_val);
    // every run's history: the same exact first call
    auto fresh_copies_val = [&](Gelu8bitCopies &copies_val) {
        Gelu8bitCopies first_call_val(ODD_ROWS, ODD_COLUMNS);
        round_8bit_per_tensor(first_call_val.rows_8bit_tensor, first_call_val.row_scales_tensor, first_call_val.turned_8bit_tensor,
                              first_call_val.turned_scales_tensor, outputs_val.gelu_output_tensor, &copies_val.history_val);
    };
    Tensor gelu_input_tensor = Tensor::zeros({ODD_ROWS, ODD_COLUMNS}, DType::BFloat16);
    Gelu8bitCopies default_val(ODD_ROWS, ODD_COLUMNS);
    fresh_copies_val(default_val);
    gelu_8bit_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_PINGPONG>(gelu_input_tensor, default_val, operands_val);
    for (const Gelu8bitSquare &square_val : GELU_8BIT_SQUARES_LIST)
    {
        Gelu8bitCopies copies_val(ODD_ROWS, ODD_COLUMNS);
        fresh_copies_val(copies_val);
        square_val.run_ptr(gelu_input_tensor, copies_val, operands_val);
        printf("    %s:\n", square_val.name_ptr);
        ASSERT_EQ(same_gelu_8bit_bits(copies_val, default_val), true);
    }
}

/// One variant of this file's setting, with these template choices
/// .
template <int PRODUCER_REGISTERS, int IN_FLIGHT
          >
static void variant_at(const Outputs &outputs_val, const Operands &operands_val
                       )
{
    mine_at<MLP_UP_8BIT_ROW_GROUPS, MLP_UP_8BIT_COLUMNS, MLP_UP_8BIT_STAGES, MLP_UP_8BIT_CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT>(outputs_val, operands_val,
                                                                                                MLP_UP_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_8BIT_BLOCKS);
}

/// A variant: its template choices and its run.
struct Variant
{
    int producer_registers_val;
    int in_flight_val;
    void (*run_ptr)(const Outputs &, const Operands &);
};

/// The variants: the producer's registers (24 to 56) x the steps in flight (0 to 3)
/// .
static const Variant VARIANTS_LIST[] = {
    {24, 0, variant_at<24, 0>},
    {24, 1, variant_at<24, 1>},
    {24, 2, variant_at<24, 2>},
    {24, 3, variant_at<24, 3>},
    {32, 0, variant_at<32, 0>},
    {32, 1, variant_at<32, 1>},
    {32, 2, variant_at<32, 2>},
    {32, 3, variant_at<32, 3>},
    {40, 0, variant_at<40, 0>},
    {40, 1, variant_at<40, 1>},
    {40, 2, variant_at<40, 2>},
    {40, 3, variant_at<40, 3>},
    {48, 0, variant_at<48, 0>},
    {48, 1, variant_at<48, 1>},
    {48, 2, variant_at<48, 2>},
    {48, 3, variant_at<48, 3>},
    {56, 0, variant_at<56, 0>},
    {56, 1, variant_at<56, 1>},
    {56, 2, variant_at<56, 2>},
    {56, 3, variant_at<56, 3>},
};

static void every_variant_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's setting: every producer
    // register count and steps in flight
    // gives the bits of this file's setting: they move registers, waits and shared memory, never a sum.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 6);
    print_test_name(__func__);
    Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    mine(default_val, operands_val);
    for (const Variant &variant_val : VARIANTS_LIST)
    {
        Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        variant_val.run_ptr(outputs_val, operands_val);
        printf("    producer %d registers, %d steps in flight:\n", variant_val.producer_registers_val, variant_val.in_flight_val);
        ASSERT_EQ(same_bits(outputs_val, default_val), true);
    }
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

/// Adds one square's kernels (clusters of 1 and 2
/// ) to the lists, with their names and block sizes.
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void add_kernels_of_square(std::vector<const void *> &kernels_list, std::vector<std::string> &names_list, std::vector<int> &threads_list)
{
    kernels_list.push_back((const void *)mlp_up_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 1>);
    kernels_list.push_back((const void *)mlp_up_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 2>);
    for (const char *kind_ptr : {"clusters of 1", "clusters of 2"})
    {
        names_list.push_back(std::string("mlp_up_8bit_kernel<") + std::to_string(ROW_GROUPS) + " warpgroups, " + std::to_string(COLUMNS) +
                             " columns, " + std::to_string(STAGES) + " sets, " + kind_ptr + ">");
        threads_list.push_back((MLP_UP_8BIT_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS);   // the producer and the consumers
    }
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel (every square, in clusters of 1 and 2
    // , at this file's promotion) can be launched with its warpgroups' threads, and nothing is spilled, save up to
    // MOST_SPILL_OF_THE_LARGEST_SQUARE bytes at 2 warpgroups x 128 columns
    // (and at 128 columns with 1 warpgroup: GELU's math beside the sums)
    // : two sets of 64 sums a thread (the float ones and the tensor cores'), with the store's addresses. cudaFuncGetAttributes'
    // numRegs, maxThreadsPerBlock and localSizeBytes, read without running anything.
    std::vector<const void *> kernels_list;
    std::vector<std::string> names_list;
    std::vector<int> threads_list;
    add_kernels_of_square<64, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 4, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 4, 2>(kernels_list, names_list, threads_list);
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
    add_kernels_of_square<256, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 3, 2>(kernels_list, names_list, threads_list);
#endif
    // the ping-pong squares (2 warpgroups, clusters of 1), and gelu_output in 8 bits (GELU_8BIT), cooperative and ping-pong
    constexpr int P = MLP_UP_8BIT_PRODUCER_REGISTERS, F = MLP_UP_8BIT_IN_FLIGHT, S = MLP_UP_8BIT_PROMOTION_STEPS;
    std::vector<std::pair<const void *, const char *>> more_list = {
        {(const void *)mlp_up_8bit_kernel<128, 4, 2, 1, false, P, F, S, 0, 0, 1, false>, "mlp_up_8bit_kernel<ping-pong, 128 columns, 4 sets>"},
        {(const void *)mlp_up_8bit_kernel<128, 4, 2, 1, false, P, F, S, 0, 0, 0, true>, "mlp_up_8bit_kernel<gelu 8 bits, 2 warpgroups, 128 columns, 4 sets>"},
        {(const void *)mlp_up_8bit_kernel<128, 4, 2, 1, false, P, F, S, 0, 0, 1, true>, "mlp_up_8bit_kernel<gelu 8 bits, ping-pong, 128 columns, 4 sets>"},
#if MLP_UP_8BIT_PROMOTION_STEPS == 0
        {(const void *)mlp_up_8bit_kernel<256, 3, 2, 1, false, P, F, S, 0, 0, 1, false>, "mlp_up_8bit_kernel<ping-pong, 256 columns, 3 sets>"},
        {(const void *)mlp_up_8bit_kernel<256, 4, 2, 1, false, P, F, S, 0, 0, 1, false>, "mlp_up_8bit_kernel<ping-pong, 256 columns, 4 sets>"},
        {(const void *)mlp_up_8bit_kernel<256, 3, 2, 1, false, P, F, S, 0, 0, 0, true>, "mlp_up_8bit_kernel<gelu 8 bits, 2 warpgroups, 256 columns, 3 sets>"},
        {(const void *)mlp_up_8bit_kernel<256, 4, 2, 1, false, P, F, S, 0, 0, 1, true>, "mlp_up_8bit_kernel<gelu 8 bits, ping-pong, 256 columns, 4 sets>"},
#endif
    };
    for (const auto &kernel_val : more_list)
    {
        kernels_list.push_back(kernel_val.first);
        names_list.push_back(kernel_val.second);
        threads_list.push_back((MLP_UP_8BIT_PRODUCER_WARPGROUPS + 2) * HARDWARE_WARPGROUP_THREADS);
    }
    for (size_t kernel_index_val = 0; kernel_index_val < kernels_list.size(); kernel_index_val++)
    {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val].c_str(),
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        size_t most_spill_val = names_list[kernel_index_val].find("256 columns") != std::string::npos   ? (size_t)MOST_SPILL_AT_256_COLUMNS
                                : names_list[kernel_index_val].find("128 columns") != std::string::npos ? (size_t)MOST_SPILL_OF_THE_LARGEST_SQUARE
                                                                                                         : (size_t)0;
        ASSERT_EQ(attributes_val.localSizeBytes <= most_spill_val, true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_size();
    matches_pytorch_with_sizes_no_tile_divides();
    matches_pytorch_on_the_bfloat16_values();
    matches_pytorch_and_the_general_kernel_with_no_bias();

    every_promotion_gives_wgmma_matmul_8bits_bits();
    every_square_gives_the_default_bits();
    every_order_block_count_and_cluster_gives_the_default_bits();
    pingpong_gives_the_default_bits();
    every_variant_gives_the_default_bits();
    gelu_8bit_gives_round_8bit_per_tensors_bits();
    gelu_8bit_at_every_square_gives_the_default_bits();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
