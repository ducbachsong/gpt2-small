// test_mlp_down_input_grad_bf16.cu — tests for cuda/kernels/bf16/prod/mlp_down_input_grad_bf16.cuh (mlp_down_input_grad_bf16): mlp_up_output_grad = (branch_output_grad · mlp_down_weight) · gelu'(mlp_up_output): back through the second linear and GELU in one.
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks against PyTorch's float product of the same values, at GPT-2 small's size and at sizes no square
// divides; part 2 checks the bits: the same as wgmma_matmul_bf16 (the general product this kernel was split from) at
// the same setting, and the same at every square, block order, block count, cluster, producer register count and
// steps in flight (weight_grads: and staged rows and padding)
// Part 3 checks that the kernels' registers fit their block. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a. The timings are in tune_mlp_down_input_grad_bf16.cu.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <string>
#include <vector>
#include "../../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../../kernels/bf16/prod/mlp_down_input_grad_bf16.cuh"
#include "../../../kernels/bf16/gelu_bf16.cuh"   // the reference: wgmma_matmul_bf16 (through linear_bf16.cuh) and GELU's epilogues

// Against PyTorch in float on the same bfloat16 values: the sums are float on both sides, added in another order,
// then mine are rounded to bfloat16, by up to 2^-9 (0.2%) of each value. The tolerance is the largest difference,
// as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define ODD_ROWS 200                  // sizes no square divides: 3 x 64 + 8 rows, ...
#define ODD_COLUMNS 264               // ... 4 x 64 + 8 columns ...
// the sets of tiles for the cooperative walk's checks (the general kernel's, and this file's square in clusters): the
// sets never change a bit, and ping-pong's 4 (a set holding one left tile) do not fit a cooperative block
#define COOPERATIVE_STAGES (MLP_DOWN_INPUT_GRAD_BF16_STAGES < 3 ? MLP_DOWN_INPUT_GRAD_BF16_STAGES : 3)
#define ODD_SUM_LENGTH 1032           // ... and 16 steps of 64 values of k + 8

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

/// The product's operands, as mlp_down_input_grad_bf16 takes them, for output {rows, columns} with a sum of sum_length.
struct Operands
{
    Tensor output_grad_tensor;        // {rows, sum_length}
    Tensor weight_tensor;             // {sum_length, columns}
    Tensor gelu_input_tensor;         // {rows, columns}
};

/// What the product writes.
struct Outputs
{
    Tensor output_tensor;             // {rows, columns}
};

/// Makes random operands for output {rows, columns} with a sum of sum_length, from seed_val.
static Operands random_operands(size_t rows_val, size_t columns_val, size_t sum_length_val, uint64_t seed_val)
{
    torch::manual_seed(seed_val);
    Operands operands_val;
    operands_val.output_grad_tensor = random_tensor({rows_val, sum_length_val});
    operands_val.weight_tensor = random_tensor({sum_length_val, columns_val});
    operands_val.gelu_input_tensor = random_tensor({rows_val, columns_val});
    return operands_val;
}

/// Makes the outputs for output {rows, columns}, filled with 7s so a value not written shows.
///
///   operands_val   the product's operands
///   rows_val       output's rows
///   columns_val    output's columns
static Outputs fresh_outputs(const Operands &operands_val, size_t rows_val, size_t columns_val)
{
    Outputs outputs_val;
    outputs_val.output_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
    outputs_val.output_tensor.fill_(7.0);
    return outputs_val;
}

/// Runs mlp_down_input_grad_bf16 at a setting given (the template choices not given: this file's).
template <int ROW_GROUPS, int COLUMNS, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, int PINGPONG = MLP_DOWN_INPUT_GRAD_BF16_PINGPONG
          >
static void mine_at(const Outputs &outputs_val, const Operands &operands_val,
                    int block_rows_per_group_val, int blocks_val)
{
    mlp_down_input_grad_bf16_at<ROW_GROUPS, COLUMNS, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, PINGPONG>(outputs_val.output_tensor, operands_val.output_grad_tensor, operands_val.weight_tensor,
                                                 operands_val.gelu_input_tensor, block_rows_per_group_val, blocks_val);
}

/// Runs mlp_down_input_grad_bf16 at this file's setting (its defines).
static void mine(const Outputs &outputs_val, const Operands &operands_val)
{
    mlp_down_input_grad_bf16(outputs_val.output_tensor, operands_val.output_grad_tensor, operands_val.weight_tensor, operands_val.gelu_input_tensor);
}

/// Runs the same product with wgmma_matmul_bf16 (the general kernel) at this file's setting: the bits mine must give.
///
///   outputs_val    written
///   operands_val   the product's operands
static void reference(const Outputs &outputs_val, const Operands &operands_val
                      )
{
    wgmma_matmul_bf16_at<MLP_DOWN_INPUT_GRAD_BF16_ROW_GROUPS, MLP_DOWN_INPUT_GRAD_BF16_COLUMNS, COOPERATIVE_STAGES, MLP_DOWN_INPUT_GRAD_BF16_CLUSTER>(
        outputs_val.output_tensor, operands_val.output_grad_tensor, operands_val.weight_tensor, Tensor(), false, false, true,
        GeluBf16BackwardEpilogue{operands_val.gelu_input_tensor.data_ptr<nv_bfloat16>()}, 1, MLP_DOWN_INPUT_GRAD_BF16_BLOCK_ROWS_PER_GROUP);
}

/// Gives PyTorch's float product of the same values: the output mine must be near.
static torch::Tensor pytorch_expected(const Operands &operands_val)
{
    torch::Tensor expected_tensor = pytorch_copy_of(operands_val.output_grad_tensor).matmul(pytorch_copy_of(operands_val.weight_tensor));
    // times gelu'(x) of GPT-2's tanh GELU: 0.5 (1 + tanh u) + 0.5 x (1 - tanh² u) √(2/π) (1 + 3 · 0.044715 x²), u = √(2/π) (x + 0.044715 x³)
    torch::Tensor x_tensor = pytorch_copy_of(operands_val.gelu_input_tensor);
    const double root_val = std::sqrt(2.0 / M_PI), cube_val = 0.044715;
    torch::Tensor tanh_tensor = torch::tanh(root_val * (x_tensor + cube_val * x_tensor * x_tensor * x_tensor));
    torch::Tensor slope_tensor = 0.5 * (1.0 + tanh_tensor) +
                                 0.5 * x_tensor * (1.0 - tanh_tensor * tanh_tensor) * root_val * (1.0 + 3.0 * cube_val * x_tensor * x_tensor);
    expected_tensor = expected_tensor * slope_tensor;
    return expected_tensor;
}

/// True when two runs' outputs hold the same bits.
static bool same_bits(const Outputs &first_val, const Outputs &second_val)
{
    bool same_val = torch::equal(pytorch_copy_of(first_val.output_tensor), pytorch_copy_of(second_val.output_tensor));
    return same_val;
}

// ══ part 1: against PyTorch ═══════════════════════════════════════════════════

/// Runs mlp_down_input_grad_bf16 at its setting on random operands and checks it against PyTorch's float product.
///
///   test_ptr         the test's name
///   rows_val         output's rows
///   columns_val      output's columns
///   sum_length_val   the product's k
static void matches_pytorch(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val)
{
    // Runs mlp_down_input_grad_bf16 at its setting on random operands for output {rows, columns} with a sum of sum_length, and checks it
    // against PyTorch's float product within BFLOAT16_TOLERANCE.
    // 1. random operands; mine
    Operands operands_val = random_operands(rows_val, columns_val, sum_length_val, 0);
    Outputs outputs_val = fresh_outputs(operands_val, rows_val, columns_val);
    mine(outputs_val, operands_val);
    // 2. PyTorch's, then check
    torch::Tensor expected_tensor = pytorch_expected(operands_val);
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu\n", rows_val, columns_val, sum_length_val);
    ASSERT_NEAR(relative_difference_on_gpu(outputs_val.output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
}

static void matches_pytorch_on_gpt2_size(void)
{
    // Expect: GPT-2 small's mlp down input grad on a micro-batch of GPT2_ROWS rows (output {rows, 4 x embed_dim}, a sum
    // of embed_dim), within 1e-2 of PyTorch's float product.
    matches_pytorch(__func__, GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM);
}

static void matches_pytorch_with_sizes_no_tile_divides(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, no square nor step dividing them: TMA
    // gives 0s past the edges and stores nothing there; within 1e-2 of PyTorch's float product.
    matches_pytorch(__func__, ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH);
}

// ══ part 2: the bits ══════════════════════════════════════════════════════════

static void gives_wgmma_matmul_bf16s_bits(void)
{
    // Expect: at GPT-2 small's size and at sizes no square divides, mlp_down_input_grad_bf16 at its setting gives the same bits as
    // wgmma_matmul_bf16 at the same setting
    // with GELU's epilogue
    // : the same sums in the same order, rounded once the same way, so training gives the same numbers.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM}, {ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH}};
    for (const auto &size_val : sizes_list)
    {
        Operands operands_val = random_operands(size_val[0], size_val[1], size_val[2], 2);
        Outputs mine_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
        Outputs reference_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
        mine(mine_val, operands_val);
        reference(reference_val, operands_val);
        printf("    rows %zu, columns %zu, sum_length %zu:\n", size_val[0], size_val[1], size_val[2]);
        ASSERT_EQ(same_bits(mine_val, reference_val), true);
    }
}

/// One square (warpgroups, columns, sets of tiles) run in a cluster of cluster_val: what the sweeps below call.
template <int ROW_GROUPS, int COLUMNS, int STAGES>
static void square_at(const Outputs &outputs_val, const Operands &operands_val, int cluster_val,
                      int block_rows_per_group_val, int blocks_val)
{
    if (cluster_val == 2)
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 2, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 0>(outputs_val, operands_val, block_rows_per_group_val, blocks_val);
    else
        mine_at<ROW_GROUPS, COLUMNS, STAGES, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 0>(outputs_val, operands_val, block_rows_per_group_val, blocks_val);
}

/// A square that fits a block's shared memory, and its run.
struct Square
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Outputs &, const Operands &, int, int, int);
};

/// Every square that fits a block's shared memory (mlp_down_input_grad_bf16_shared_bytes at most HARDWARE_MAX_SHARED_BYTES_PER_BLOCK).
static const Square SQUARES_LIST[] = {
    {1, 64, 1, square_at<1, 64, 1>},
    {1, 64, 2, square_at<1, 64, 2>},
    {1, 64, 3, square_at<1, 64, 3>},
    {1, 64, 4, square_at<1, 64, 4>},
    {1, 128, 1, square_at<1, 128, 1>},
    {1, 128, 2, square_at<1, 128, 2>},
    {1, 128, 3, square_at<1, 128, 3>},
    {1, 128, 4, square_at<1, 128, 4>},
    {1, 256, 1, square_at<1, 256, 1>},
    {1, 256, 2, square_at<1, 256, 2>},
    {1, 256, 3, square_at<1, 256, 3>},
    {1, 256, 4, square_at<1, 256, 4>},
    {2, 64, 1, square_at<2, 64, 1>},
    {2, 64, 2, square_at<2, 64, 2>},
    {2, 64, 3, square_at<2, 64, 3>},
    {2, 64, 4, square_at<2, 64, 4>},
    {2, 128, 1, square_at<2, 128, 1>},
    {2, 128, 2, square_at<2, 128, 2>},
    {2, 128, 3, square_at<2, 128, 3>},
    {2, 128, 4, square_at<2, 128, 4>},
    {2, 256, 1, square_at<2, 256, 1>},
    {2, 256, 2, square_at<2, 256, 2>},
    {2, 256, 3, square_at<2, 256, 3>},
};

static void every_square_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH: every square that fits (1 or 2 warpgroups x
    // 64, 128 or 256 columns x 1 to 4 sets of tiles) gives the bits of this file's setting: each value's sum walks k
    // in the same order whatever the square.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 3);
    print_test_name(__func__);
    Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
    mine(default_val, operands_val);
    for (const Square &square_val : SQUARES_LIST)
    {
        Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        square_val.run_ptr(outputs_val, operands_val, 1, MLP_DOWN_INPUT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, MLP_DOWN_INPUT_GRAD_BF16_BLOCKS);
        printf("    %d warpgroups x %d columns x %d sets:\n", square_val.row_groups_val, square_val.columns_val, square_val.stages_val);
        ASSERT_EQ(same_bits(outputs_val, default_val), true);
    }
}

static void every_order_block_count_and_cluster_gives_the_default_bits(void)
{
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's square: every block order
    // (1, 4, 16, 64 rows of squares a group), block count (as many as run at once, 1, 3, one a square) and cluster
    // (1, 2) gives the bits of this file's setting: who works out a square changes, never a sum. The cooperative
    // walk at COOPERATIVE_STAGES sets (ping-pong's orders and counts: pingpong_gives_the_default_bits).
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 4);
    print_test_name(__func__);
    {
        Outputs default_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
        mine(default_val, operands_val);
        for (int cluster_val : {1, 2})
            for (int block_rows_per_group_val : {1, 4, 16, 64})
                for (int blocks_val : {0, 1, 3, MLP_DOWN_INPUT_GRAD_BF16_ONE_A_SQUARE})
                {
                    Outputs outputs_val = fresh_outputs(operands_val, ODD_ROWS, ODD_COLUMNS);
                    square_at<MLP_DOWN_INPUT_GRAD_BF16_ROW_GROUPS, MLP_DOWN_INPUT_GRAD_BF16_COLUMNS, COOPERATIVE_STAGES>(outputs_val, operands_val, cluster_val, block_rows_per_group_val,
                                                                     blocks_val);
                    printf("    clusters of %d, %d rows a group, %d blocks:\n", cluster_val, block_rows_per_group_val, blocks_val);
                    ASSERT_EQ(same_bits(outputs_val, default_val), true);
                }
    }
}

/// Runs one ping-pong square (2 warpgroups x COLUMNS x STAGES, clusters of 1) at a block order and count.
///
///   outputs_val                written
///   operands_val               the product's operands and copies
///   block_rows_per_group_val   the block order
///   blocks_val                 how many blocks
template <int COLUMNS, int STAGES>
static void pingpong_at(const Outputs &outputs_val, const Operands &operands_val, int block_rows_per_group_val, int blocks_val)
{
    // One ping-pong square (2 warpgroups x COLUMNS x STAGES, clusters of 1) at a block order and block count: what the
    // ping-pong check below calls.
    mine_at<2, COLUMNS, STAGES, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 1>(outputs_val, operands_val,
                                                                                            block_rows_per_group_val, blocks_val);
}

/// A ping-pong square and its run.
struct PingpongSquare
{
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Outputs &, const Operands &, int, int);
};

/// The ping-pong squares: 128 or 256 columns, 2 to 4 sets (256 x 4 fits: a set holds one left tile).
static const PingpongSquare PINGPONG_SQUARES_LIST[] = {
    {128, 2, pingpong_at<128, 2>}, {128, 3, pingpong_at<128, 3>}, {128, 4, pingpong_at<128, 4>},
    {256, 2, pingpong_at<256, 2>}, {256, 3, pingpong_at<256, 3>}, {256, 4, pingpong_at<256, 4>},
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
        Operands operands_val = random_operands(size_val[0], size_val[1], size_val[2], 6);
        Outputs default_val = fresh_outputs(operands_val, size_val[0], size_val[1]);
        mine(default_val, operands_val);
        for (const PingpongSquare &square_val : PINGPONG_SQUARES_LIST)
            for (int block_rows_per_group_val : {1, 4})
                for (int blocks_val : {0, 1, 3, MLP_DOWN_INPUT_GRAD_BF16_ONE_A_SQUARE})
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

/// Runs one variant of this file's setting, with these template choices.
///
///   outputs_val    written
///   operands_val   the product's operands
template <int PRODUCER_REGISTERS, int IN_FLIGHT
          >
static void variant_at(const Outputs &outputs_val, const Operands &operands_val
                       )
{
    // One variant of this file's setting: its square, cluster, block order and block count
    // With these template choices.
    mine_at<MLP_DOWN_INPUT_GRAD_BF16_ROW_GROUPS, MLP_DOWN_INPUT_GRAD_BF16_COLUMNS, MLP_DOWN_INPUT_GRAD_BF16_STAGES, MLP_DOWN_INPUT_GRAD_BF16_CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT>(outputs_val, operands_val, MLP_DOWN_INPUT_GRAD_BF16_BLOCK_ROWS_PER_GROUP,
                                                                                       MLP_DOWN_INPUT_GRAD_BF16_BLOCKS);
}

/// A variant: its template choices and its run.
struct Variant
{
    int producer_registers_val;
    int in_flight_val;
    void (*run_ptr)(const Outputs &, const Operands &);
};

/// The variants: the producer's registers (24 to 56) x the steps in flight (0 to 3).
static const Variant VARIANTS_LIST[] = {
    // The variants: the producer's registers (24 to 56) x the steps in flight (0 to 3; more than STAGES - 1 runs as
    // STAGES - 1).
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
    // Expect: output {ODD_ROWS, ODD_COLUMNS} with a sum of ODD_SUM_LENGTH, at this file's square, cluster, order and
    // blocks: every producer register count (24 to 56) and steps in flight (0 to 3)
    // gives the bits of this file's setting: they move registers, waits and shared memory, never a sum.
    Operands operands_val = random_operands(ODD_ROWS, ODD_COLUMNS, ODD_SUM_LENGTH, 5);
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

static const size_t MOST_SPILL_OF_THE_LARGEST_SQUARE = 16;   // bytes the 2 warpgroups x 256 columns kernels may spill (see below)

/// Adds one square's kernels (clusters of 1 and 2) to the lists, with their names and block sizes.
///
///   kernels_list   added to
///   names_list     added to
///   threads_list   added to
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void add_kernels_of_square(std::vector<const void *> &kernels_list, std::vector<std::string> &names_list, std::vector<int> &threads_list)
{
    kernels_list.push_back((const void *)mlp_down_input_grad_bf16_kernel<COLUMNS, STAGES, ROW_GROUPS, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 0>);
    kernels_list.push_back((const void *)mlp_down_input_grad_bf16_kernel<COLUMNS, STAGES, ROW_GROUPS, 2, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 0>);
    for (const char *kind_ptr : {"clusters of 1", "clusters of 2"})
    {
        names_list.push_back(std::string("mlp_down_input_grad_bf16_kernel<") + std::to_string(ROW_GROUPS) + " warpgroups, " + std::to_string(COLUMNS) +
                             " columns, " + std::to_string(STAGES) + " sets, " + kind_ptr + ">");
        threads_list.push_back((MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_WARPGROUPS + ROW_GROUPS) * HARDWARE_WARPGROUP_THREADS);   // the producer and the consumers
    }
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel (every square that fits, in clusters of 1 and 2
    // ) can be launched with its warpgroups' threads, and nothing is spilled. Save 2 warpgroups x 256 columns: 384
    // threads get 168 registers each, and the 128 sums a consumer thread holds leave a word or two over: at most
    // MOST_SPILL_OF_THE_LARGEST_SQUARE bytes, a cost the tune measures. cudaFuncGetAttributes' numRegs,
    // maxThreadsPerBlock and localSizeBytes (0 if nothing spills), read without running anything.
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
    add_kernels_of_square<256, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<64, 4, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<128, 4, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_square<256, 3, 2>(kernels_list, names_list, threads_list);
    // the ping-pong squares (2 warpgroups, clusters of 1)
    const void *pingpong_kernels_list[] = {(const void *)mlp_down_input_grad_bf16_kernel<128, 4, 2, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 1>,
                                           (const void *)mlp_down_input_grad_bf16_kernel<256, 3, 2, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 1>,
                                           (const void *)mlp_down_input_grad_bf16_kernel<256, 4, 2, 1, MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_REGISTERS, MLP_DOWN_INPUT_GRAD_BF16_IN_FLIGHT, 1>};
    const char *pingpong_names_list[] = {"mlp_down_input_grad_bf16_kernel<ping-pong, 128 columns, 4 sets>", "mlp_down_input_grad_bf16_kernel<ping-pong, 256 columns, 3 sets>",
                                         "mlp_down_input_grad_bf16_kernel<ping-pong, 256 columns, 4 sets>"};
    for (int pingpong_index_val = 0; pingpong_index_val < 3; pingpong_index_val++)
    {
        kernels_list.push_back(pingpong_kernels_list[pingpong_index_val]);
        names_list.push_back(pingpong_names_list[pingpong_index_val]);
        threads_list.push_back((MLP_DOWN_INPUT_GRAD_BF16_PRODUCER_WARPGROUPS + 2) * HARDWARE_WARPGROUP_THREADS);
    }
    for (size_t kernel_index_val = 0; kernel_index_val < kernels_list.size(); kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val].c_str(),
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        // 2. check; 2 warpgroups x 256 columns may spill MOST_SPILL_OF_THE_LARGEST_SQUARE bytes
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        bool largest_square_val = names_list[kernel_index_val].find("2 warpgroups, 256 columns") != std::string::npos;
        ASSERT_EQ(attributes_val.localSizeBytes <= (largest_square_val ? MOST_SPILL_OF_THE_LARGEST_SQUARE : (size_t)0), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_size();
    matches_pytorch_with_sizes_no_tile_divides();

    gives_wgmma_matmul_bf16s_bits();
    every_square_gives_the_default_bits();
    every_order_block_count_and_cluster_gives_the_default_bits();
    every_variant_gives_the_default_bits();
    pingpong_gives_the_default_bits();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
