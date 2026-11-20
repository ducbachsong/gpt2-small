// test_mlp_up_weight_grad_8bit.cu — tests for cuda/kernels/8bit/prod/mlp_up_weight_grad_8bit.cuh (mlp_up_weight_grad_8bit): mlp_up_weight_grad += mlp_up_output_gradᵀ · ln2_output, the product in 8 bits.
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks against PyTorch: on the same rounded values (the E4M3 copies read back as floats) within bfloat16's
// rounding, and on the bfloat16 values within E4M3's; part 2 checks the bits: the same as wgmma_matmul_8bit (the
// general 8-bit product) on the same copies at each promotion, and the same at every square, block order, block count,
// cluster and variant
// (at each split count: split, the sums are added in another order, so the bits are each split count's own)
// . Part 3 checks that the kernels' registers fit their block. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a. The timings are in tune_mlp_up_weight_grad_8bit.cu.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <string>
#include <vector>
#include "../../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../../kernels/8bit/prod/mlp_up_weight_grad_8bit.cuh"
#include "../../../kernels/8bit/wgmma_matmul_8bit.cuh"   // the reference: the general 8-bit product

// Against PyTorch: on the same rounded values, the sums are float on both sides, added in another order, then mine are
// rounded to bfloat16 (2^-9 of a value); on the bfloat16 values, each E4M3 is off by up to 2^-4 (6%) of its value too.
// Each tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double EIGHT_BIT_TOLERANCE = 1e-1;

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define ODD_ROWS 208                  // sizes no square divides, multiples of ROUND_8BIT_ROW_LENGTH_MULTIPLE: 3 x 64 + 16 rows, ...
#define ODD_COLUMNS 272               // ... 4 x 64 + 16 columns ...
#define ODD_SUM_LENGTH 1040           // ... and 8 steps of 128 values of k + 16
#define MOST_SPILL_OF_THE_LARGEST_SQUARE 16   // bytes the largest squares' kernels may spill (see every_kernel_fits_its_block)

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   bfloat16, on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU, bfloat16 turned into floats exactly.
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value.
///
///   mine_tensor      my result
///   pytorch_tensor   PyTorch's, the same shape
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    // Worked out on the GPU.
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Gives a copy's rounded values turned back by PyTorch: each E4M3 times the tensor's one scale, exact.
///
///   rounded_tensor   {rows, k} E4M3
///   scale_tensor     {1} float: the tensor's scale
static torch::Tensor turned_back(const Tensor &rounded_tensor, const Tensor &scale_tensor)
{
    // Gives a copy's rounded values turned back by PyTorch: each E4M3 (read as float8_e4m3fn) times the tensor's one
    // scale, exact. {rows, k} float.
    int64_t rows_val = (int64_t)rounded_tensor.size(0), k_val = (int64_t)rounded_tensor.size(1);
    torch::Tensor e4m3_tensor = torch::from_blob(rounded_tensor.data_ptr(), {rows_val, k_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scale_tensor);
}

/// Gives a copy's one scale as a list of every row's, for the general kernel (one scale a row).
///
///   scale_tensor   {1} float: the tensor's scale
///   rows_val       how many rows
static Tensor scales_list(const Tensor &scale_tensor, size_t rows_val)
{
    // {rows_val} float, every value the scale.
    Tensor scales_tensor = Tensor::zeros({rows_val});
    scales_tensor.fill_(scale_tensor.tolist<float>()[0]);
    return scales_tensor;
}

/// The product's operands, as mlp_up_weight_grad_8bit takes them, for output {rows, columns} with a sum of sum_length, and their copies.
struct Operands
{
    Tensor output_grad_tensor;        // {sum_length, rows}
    Tensor input_tensor;              // {sum_length, columns}
    Tensor old_tensor;                // {rows, columns}: what weight_grad holds before, added onto
    MlpUpWeightGrad8bitCopies copies_val;              // the 8-bit copies, rounded once by mlp_up_weight_grad_8bit_round
};

/// What the product writes.
struct Outputs
{
    Tensor output_tensor;             // {rows, columns}
};

/// Makes random operands for output {rows, columns} with a sum of sum_length, from seed_val, and rounds them.
static Operands random_operands(size_t rows_val, size_t columns_val, size_t sum_length_val, uint64_t seed_val)
{
    torch::manual_seed(seed_val);
    Operands operands_val;
    operands_val.output_grad_tensor = random_tensor({sum_length_val, rows_val});
    operands_val.input_tensor = random_tensor({sum_length_val, columns_val});
    operands_val.old_tensor = random_tensor({rows_val, columns_val});
    operands_val.copies_val = MlpUpWeightGrad8bitCopies(rows_val, columns_val, sum_length_val);
    mlp_up_weight_grad_8bit_round(operands_val.copies_val, operands_val.output_grad_tensor, operands_val.input_tensor);
    return operands_val;
}

/// Makes the outputs for output {rows, columns}, filled with 7s so a value not written shows.
///
///   operands_val   the product's operands (weight_grad: a copy of their old values, added onto)
///   rows_val       output's rows
///   columns_val    output's columns
static Outputs fresh_outputs(const Operands &operands_val, size_t rows_val, size_t columns_val)
{
    Outputs outputs_val;
    outputs_val.output_tensor = operands_val.old_tensor.clone();
    return outputs_val;
}

/// Runs mlp_up_weight_grad_8bit's product on the operands' copies at a setting given (the template choices not given: this file's).
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, int PROMOTION_STEPS = MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS
          , int STAGED_ROWS = MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, int STAGED_ROW_PADDING = MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING
          >
static void mine_at(const Outputs &outputs_val, const Operands &operands_val,
                    int splits_val,
                    int block_rows_per_group_val, int blocks_val)
{
    mlp_up_weight_grad_8bit_rounded_at<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PROMOTION_STEPS, STAGED_ROWS, STAGED_ROW_PADDING>(
        outputs_val.output_tensor, operands_val.copies_val, splits_val, block_rows_per_group_val, blocks_val);
}

/// Runs mlp_up_weight_grad_8bit at this file's setting (its defines): rounds the operands (again, the same copies) and the product.
static void mine(const Outputs &outputs_val, const Operands &operands_val)
{
    mlp_up_weight_grad_8bit(outputs_val.output_tensor, operands_val.output_grad_tensor, operands_val.input_tensor, operands_val.copies_val);
}

/// Runs the same product with wgmma_matmul_8bit (the general 8-bit kernel) on the same copies.
///
///   outputs_val           written
///   operands_val          the product's operands and copies
///   promotion_steps_val   the general kernel's promotion
static void reference(const Outputs &outputs_val, const Operands &operands_val, int promotion_steps_val)
{
    // At promotion_steps_val: the bits mine must give unsplit.
    const MlpUpWeightGrad8bitCopies &copies_val = operands_val.copies_val;
    wgmma_matmul_8bit(outputs_val.output_tensor, copies_val.left_8bit_tensor,
                      scales_list(copies_val.left_scale_tensor, copies_val.left_8bit_tensor.size(0)), copies_val.right_8bit_tensor,
                      scales_list(copies_val.right_scale_tensor, copies_val.right_8bit_tensor.size(0)), Tensor(), true, WgmmaMatmul8bitStore(), promotion_steps_val);
}

/// Gives PyTorch's float product of the same rounded values (the copies read back).
///
///   operands_val   the product's operands and copies
static torch::Tensor pytorch_on_rounded(const Operands &operands_val)
{
    // The output mine must be near within bfloat16's rounding.
    const MlpUpWeightGrad8bitCopies &copies_val = operands_val.copies_val;
    torch::Tensor expected_tensor = turned_back(copies_val.left_8bit_tensor, copies_val.left_scale_tensor)
                                        .matmul(turned_back(copies_val.right_8bit_tensor, copies_val.right_scale_tensor).t());
    return pytorch_copy_of(operands_val.old_tensor) + expected_tensor;
}

/// Gives PyTorch's float product of the bfloat16 values themselves: the output mine must be near within E4M3's rounding.
static torch::Tensor pytorch_on_bf16(const Operands &operands_val)
{
    return pytorch_copy_of(operands_val.old_tensor) +
           pytorch_copy_of(operands_val.output_grad_tensor).t().matmul(pytorch_copy_of(operands_val.input_tensor));
}

/// True when two runs' outputs hold the same bits.
static bool same_bits(const Outputs &first_val, const Outputs &second_val)
{
    bool same_val = torch::equal(pytorch_copy_of(first_val.output_tensor), pytorch_copy_of(second_val.output_tensor));
    return same_val;
}

// ══ part 1: against PyTorch ═══════════════════════════════════════════════════

/// Runs mlp_up_weight_grad_8bit at its setting on random operands and checks it against PyTorch's float product.
///
///   test_ptr         the test's name, for the message
///   rows_val         output's rows
///   columns_val      output's columns
///   sum_length_val   the product's k
static void matches_pytorch_on_the_rounded_values(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val)
{
    // Runs mlp_up_weight_grad_8bit at its setting on random operands for output {rows, columns} with a sum of sum_length, and checks it
    // against PyTorch's float product of the same rounded values within BFLOAT16_TOLERANCE.
    Operands operands_val = random_operands(rows_val, columns_val, sum_length_val, 0);
    Outputs outputs_val = fresh_outputs(operands_val, rows_val, columns_val);
    mine(outputs_val, operands_val);
    torch::Tensor expected_tensor = pytorch_on_rounded(operands_val);
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu\n", rows_val, columns_val, sum_length_val);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

static void matches_pytorch_on_gpt2_size(void)
{
    // Expect: GPT-2 small's mlp up weight grad on a micro-batch of GPT2_ROWS rows, within 1e-2 of PyTorch's float product of
    // the same rounded values.
    matches_pytorch_on_the_rounded_values(__func__, GPT2_MLP_DIM, GPT2_EMBED_DIM, GPT2_ROWS);
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
    Operands operands_val = random_operands(GPT2_MLP_DIM, GPT2_EMBED_DIM, GPT2_ROWS, 1);
    Outputs outputs_val = fresh_outputs(operands_val, GPT2_MLP_DIM, GPT2_EMBED_DIM);
    mine(outputs_val, operands_val);
    print_test_name(__func__);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, pytorch_on_bf16(operands_val)), 0.0, EIGHT_BIT_TOLERANCE);
}

static void every_split_count_matches_pytorch(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS}, a sum of ODD_SUM_LENGTH (9 steps), added onto random old values: split in 0
    // (chosen for this GPU), 1, 2, 3, 4, 8 and 32 (more than the steps: one a step), each within 1e-2 of PyTorch's old +
    // the product of the same rounded values.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 2);
    torch::Tensor expected_tensor = pytorch_on_rounded(operands_val);
    print_test_name(__func__);
    for (int splits_val : {0, 1, 2, 3, 4, 8, 32})
    {
        Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER>(outputs_val, operands_val, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
        printf("    %d splits:\n", splits_val);
        ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
    }
}

// ══ part 2: the bits ══════════════════════════════════════════════════════════

/// Runs the product at this file's square at promotion PROMOTION_STEPS, unsplit.
template <int PROMOTION_STEPS>
static void at_promotion(const Outputs &outputs_val, const Operands &operands_val)
{
    mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER, MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, PROMOTION_STEPS>(
        outputs_val, operands_val, 1, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
}

static void every_promotion_gives_wgmma_matmul_8bits_bits(void)
{
    // Expect: at GPT-2 small's size and at sizes no square divides, mlp_up_weight_grad_8bit at its square, unsplit, gives the same bits as
    // wgmma_matmul_8bit on the same copies at the same promotion (0: never, 1, 2, 4, 8 steps): the same sums in the
    // same order, the tensor cores' sums moved at the same steps, times the scales the same way
    // . So training gives the numbers it gave with wgmma_matmul_8bit.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{GPT2_MLP_DIM, GPT2_EMBED_DIM, GPT2_ROWS}, {ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH}};
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
            ASSERT_EQ(same_bits(mine_val, reference_val), true);
        }
    }
}

/// One square (warpgroups, columns, sets of tiles) run in a cluster of cluster_val: what the sweeps below call.
template <int ROW_GROUPS, int COLUMNS, int STAGES>
static void square_at(const Outputs &outputs_val, const Operands &operands_val, int cluster_val,
                      int splits_val,
                      int block_rows_per_group_val, int blocks_val)
{
    if (cluster_val == 2)
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 2>(outputs_val, operands_val, splits_val, block_rows_per_group_val, blocks_val);
    else
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 1>(outputs_val, operands_val, splits_val, block_rows_per_group_val, blocks_val);
}

/// A square that fits a block's shared memory, and its run.
struct Square
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Outputs &, const Operands &, int, int, int, int);
};

/// Every square: 1 or 2 warpgroups x 64 or 128 columns x 1 to 4 sets of tiles (all fit a block's shared memory).
static const Square SQUARES_LIST[] = {
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

/// The split counts the bit checks run at: 1, and 3 (the split kernel and its staged store).
static const int CHECKED_SPLITS_LIST[] = {1, 3};

static void every_square_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH: every square gives the bits of this file's:
    // each value's sum walks k in the same order whatever the square.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 4);
    print_test_name(__func__);
    for (int splits_val : CHECKED_SPLITS_LIST)
    {
        Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER>(default_val, operands_val, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
        for (const Square &square_val : SQUARES_LIST)
        {
            Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
            square_val.run_ptr(outputs_val, operands_val, 1, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
            printf("    %d splits, %d warpgroups x %d columns x %d sets:\n", splits_val, square_val.row_groups_val, square_val.columns_val,
                   square_val.stages_val);
            ASSERT_EQ(same_bits(outputs_val, default_val), true);
        }
    }
}

static void every_order_block_count_and_cluster_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's square: every block order (1,
    // 4, 16, 64 rows of squares a group), block count (as many as run at once, 1, 3, one a square) and cluster (1, 2)
    // gives the bits of this file's setting: who works out a square changes, never a sum.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 5);
    print_test_name(__func__);
    for (int splits_val : CHECKED_SPLITS_LIST)
    {
        Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER>(default_val, operands_val, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
        for (int cluster_val : {1, 2})
            for (int block_rows_per_group_val : {1, 4, 16, 64})
                for (int blocks_val : {0, 1, 3, MLP_UP_WEIGHT_GRAD_8BIT_ONE_A_SQUARE})
                {
                    Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
                    square_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES>(outputs_val, operands_val, cluster_val, splits_val, block_rows_per_group_val,
                                                                     blocks_val);
                    printf("    %d splits, clusters of %d, %d rows a group, %d blocks:\n", splits_val, cluster_val, block_rows_per_group_val,
                           blocks_val);
                    ASSERT_EQ(same_bits(outputs_val, default_val), true);
                }
    }
}

/// Runs one variant of this file's setting, with these template choices and the split count given.
///
///   outputs_val    written
///   operands_val   the product's operands and copies
///   splits_val     the split count
template <int PRODUCER_REGISTERS, int IN_FLIGHT
          , int STAGED_ROWS, int STAGED_ROW_PADDING
          >
static void variant_at(const Outputs &outputs_val, const Operands &operands_val
                       , int splits_val
                       )
{
    mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS, STAGED_ROWS, STAGED_ROW_PADDING>(
        outputs_val, operands_val, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
}

/// A variant: its template choices and its run.
struct Variant
{
    int producer_registers_val;
    int in_flight_val;
    int staged_rows_val;
    int staged_row_padding_val;
    void (*run_ptr)(const Outputs &, const Operands &, int);
};

/// The variants: the producer's registers x the steps in flight, then every staged rows x padding.
static const Variant VARIANTS_LIST[] = {
    // The producer's registers (24 to 56) x the steps in flight (0 to 3), then every staged rows (16, 32, 64)
    // x padding (0, 4, 8, 16) (all fit at 64 or 128 columns).
    {24, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<24, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {24, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<24, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {24, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<24, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {24, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<24, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {32, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<32, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {32, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<32, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {32, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<32, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {32, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<32, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {40, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<40, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {40, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<40, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {40, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<40, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {40, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<40, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {48, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<48, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {48, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<48, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {48, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<48, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {48, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<48, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {56, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<56, 0, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {56, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<56, 1, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {56, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<56, 2, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {56, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING, variant_at<56, 3, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 0, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 0>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 4, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 4>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 8, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 8>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 16, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 16, 16>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 0, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 0>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 4, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 4>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 8, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 8>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 16, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 32, 16>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 0, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 0>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 4, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 4>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 8, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 8>},
    {MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 16, variant_at<MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, 64, 16>},
};

static void every_variant_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's setting: every producer
    // register count and steps in flight
    // And every staged rows x padding (at 1 and 3 splits: the staged store is the split one's)
    // gives the bits of this file's setting: they move registers, waits and shared memory, never a sum.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 6);
    print_test_name(__func__);
    for (int splits_val : CHECKED_SPLITS_LIST)
    {
        Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine_at<MLP_UP_WEIGHT_GRAD_8BIT_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_8BIT_COLUMNS, MLP_UP_WEIGHT_GRAD_8BIT_STAGES, MLP_UP_WEIGHT_GRAD_8BIT_CLUSTER>(default_val, operands_val, splits_val, MLP_UP_WEIGHT_GRAD_8BIT_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_8BIT_BLOCKS);
        for (const Variant &variant_val : VARIANTS_LIST)
        {
            Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
            variant_val.run_ptr(outputs_val, operands_val, splits_val);
            printf("    %d splits, producer %d registers, %d steps in flight, %d staged rows, padding %d:\n", splits_val,
                   variant_val.producer_registers_val, variant_val.in_flight_val, variant_val.staged_rows_val, variant_val.staged_row_padding_val);
            ASSERT_EQ(same_bits(outputs_val, default_val), true);
        }
    }
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

/// Adds one square's kernels (clusters of 1 and 2, unsplit and split) to the lists, with their names and block sizes.
///
///   kernels_list   added to
///   names_list     added to
///   threads_list   added to
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void add_kernels_of_square(std::vector<const void *> &kernels_list, std::vector<std::string> &names_list, std::vector<int> &threads_list)
{
    constexpr int R = MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROWS, D = MLP_UP_WEIGHT_GRAD_8BIT_STAGED_ROW_PADDING;
    kernels_list.push_back((const void *)mlp_up_weight_grad_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 1, false, MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS, R, D>);
    kernels_list.push_back((const void *)mlp_up_weight_grad_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 2, false, MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS, R, D>);
    kernels_list.push_back((const void *)mlp_up_weight_grad_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 1, true, MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS, R, D>);
    kernels_list.push_back((const void *)mlp_up_weight_grad_8bit_kernel<COLUMNS, STAGES, ROW_GROUPS, 2, true, MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_8BIT_IN_FLIGHT, MLP_UP_WEIGHT_GRAD_8BIT_PROMOTION_STEPS, R, D>);
    for (const char *kind_ptr : {"clusters of 1", "clusters of 2", "clusters of 1, split", "clusters of 2, split"})
    {
        names_list.push_back(std::string("mlp_up_weight_grad_8bit_kernel<") + std::to_string(ROW_GROUPS) + " warpgroups, " + std::to_string(COLUMNS) +
                             " columns, " + std::to_string(STAGES) + " sets, " + kind_ptr + ">");
        threads_list.push_back((MLP_UP_WEIGHT_GRAD_8BIT_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS);   // the producer and the consumers
    }
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel (every square, in clusters of 1 and 2
    // Unsplit and split
    // At this file's promotion) can be launched with its warpgroups' threads, and nothing is spilled, save up to
    // MOST_SPILL_OF_THE_LARGEST_SQUARE bytes at 2 warpgroups x 128 columns
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
    for (size_t kernel_index_val = 0; kernel_index_val < kernels_list.size(); kernel_index_val++)
    {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val].c_str(),
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        bool largest_square_val = names_list[kernel_index_val].find("2 warpgroups, 128 columns") != std::string::npos;
        ASSERT_EQ(attributes_val.localSizeBytes <= (largest_square_val ? (size_t)MOST_SPILL_OF_THE_LARGEST_SQUARE : (size_t)0), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_size();
    matches_pytorch_with_sizes_no_tile_divides();
    matches_pytorch_on_the_bfloat16_values();
    every_split_count_matches_pytorch();

    every_promotion_gives_wgmma_matmul_8bits_bits();
    every_square_gives_the_default_bits();
    every_order_block_count_and_cluster_gives_the_default_bits();
    every_variant_gives_the_default_bits();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
