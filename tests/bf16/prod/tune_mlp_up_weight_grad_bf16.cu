// tune_mlp_up_weight_grad_bf16.cu — the timings of llmc/kernels/bf16/prod/mlp_up_weight_grad_bf16.cuh (mlp_up_weight_grad_bf16): mlp_up_weight_grad += mlp_up_output_gradᵀ · ln2_output: the MLP's first linear's weight gradient, on a training micro-batch of
// GPT2_ROWS rows; apart from test_mlp_up_weight_grad_bf16.cu so the tests only check. Each setting is timed taking turns with the others
// over SWEEP_ROUNDS rounds (the GPU's clock and heat move the same for all), rounds WARM_UP_STEPS + 1 on averaged:
//     part 1   this file's setting against PyTorch's bfloat16 matmul (cuBLAS)
//              (cuBLAS adding onto weight_grad, addmm_ with beta 1, as mine does)
//     part 2   every square that fits (1 or 2 warpgroups x 64, 128 or 256 columns x 1 to 4 sets of tiles)
//     part 3   at part 2's fastest square: every block order, block count and cluster
//              and split count
//     part 4   at part 3's fastest split count (when it splits): the splits' sum's block size
//     part 5   at this file's square: the producer's registers x the steps in flight
//              , then the staged rows x padding
// then the fastest as the #define lines of mlp_up_weight_grad_bf16.cuh. Run it on an H100 (compute capability 9.0), built with
// -arch=sm_90a; the log goes to tests/results/<GPU>/.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include <vector>
#include "../../test_helpers.h"          // print_test_name
#include "../../tune_helpers.h"          // gpu_milliseconds, peak_bf16_flops, start_tune_log
#include "../../../llmc/kernels/bf16/prod/mlp_up_weight_grad_bf16.cuh"

#define GPT2_ROWS 16384               // a training micro-batch: 16 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304
#define FLOPS_PER_MULTIPLY_ADD 2      // a product's flops: 2 x rows x columns x sum_length
#define ROWS GPT2_MLP_DIM               // the product's output {ROWS, COLUMNS} and its sum
#define COLUMNS GPT2_EMBED_DIM
#define SUM_LENGTH GPT2_ROWS

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

/// Makes a PyTorch bfloat16 tensor on the GPU with a copy of my bfloat16 tensor's values.
static torch::Tensor pytorch_bf16_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
        .clone()
        .to(torch::kBFloat16);
}

/// The product's operands and outputs at GPT-2's size, mine and PyTorch's copies.
struct Product
{
    Tensor output_grad_tensor = random_bf16_tensor({SUM_LENGTH, ROWS});
    Tensor input_tensor = random_bf16_tensor({SUM_LENGTH, COLUMNS});
    Tensor output_tensor = Tensor::zeros({ROWS, COLUMNS}, DType::BFloat16);
};

/// Runs mlp_up_weight_grad_bf16 at a setting given (the template choices not given: this file's).
template <int ROW_GROUPS, int COLUMNS_ARG, int STAGES, int CLUSTER, int PRODUCER_REGISTERS = MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS,
          int IN_FLIGHT = MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT
          , int STAGED_ROWS = MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, int STAGED_ROW_PADDING = MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING
          >
static void mine_at(const Product &product_val,
                    int splits_val,
                    int block_rows_per_group_val, int blocks_val
                    , int sum_threads_per_block_val = MLP_UP_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK
                    )
{
    mlp_up_weight_grad_bf16_at<ROW_GROUPS, COLUMNS_ARG, STAGES, CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING>(
        product_val.output_tensor, product_val.output_grad_tensor, product_val.input_tensor, splits_val, block_rows_per_group_val, blocks_val,
        sum_threads_per_block_val);
}

/// One square (warpgroups, columns, sets of tiles) run in a cluster of cluster_val: what the sweeps call.
template <int ROW_GROUPS, int COLUMNS_ARG, int STAGES>
static void square_at(const Product &product_val, int cluster_val,
                      int splits_val,
                      int block_rows_per_group_val, int blocks_val)
{
    if (cluster_val == 2)
        mine_at<ROW_GROUPS, COLUMNS_ARG, STAGES, 2>(product_val, splits_val, block_rows_per_group_val, blocks_val);
    else
        mine_at<ROW_GROUPS, COLUMNS_ARG, STAGES, 1>(product_val, splits_val, block_rows_per_group_val, blocks_val);
}

/// A square that fits a block's shared memory, and its run.
struct Square
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*run_ptr)(const Product &, int, int, int, int);
};

/// Every square that fits a block's shared memory (mlp_up_weight_grad_bf16_shared_bytes at most HARDWARE_MAX_SHARED_BYTES_PER_BLOCK).
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
static const int SQUARE_COUNT = (int)(sizeof(SQUARES_LIST) / sizeof(SQUARES_LIST[0]));

/// True when this file's square fits a block's shared memory with STAGED_ROWS x STAGED_ROW_PADDING's room (and the
/// barriers).
template <int STAGED_ROWS, int STAGED_ROW_PADDING>
static constexpr bool variant_fits(void)
{
    return mlp_up_weight_grad_bf16_shared_bytes<MLP_UP_WEIGHT_GRAD_BF16_COLUMNS, MLP_UP_WEIGHT_GRAD_BF16_STAGES, MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS, STAGED_ROWS, STAGED_ROW_PADDING>() +
               mlp_up_weight_grad_bf16_barrier_bytes<MLP_UP_WEIGHT_GRAD_BF16_STAGES, MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS>() <= HARDWARE_MAX_SHARED_BYTES_PER_BLOCK;
}

/// One variant of this file's setting: its square, cluster, block order and block count
/// (and the split count given)
/// , with these template choices.
template <int PRODUCER_REGISTERS, int IN_FLIGHT
          , int STAGED_ROWS, int STAGED_ROW_PADDING
          >
static void variant_at(const Product &product_val
                       , int splits_val
                       )
{
    if constexpr (variant_fits<STAGED_ROWS, STAGED_ROW_PADDING>())
        mine_at<MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_BF16_COLUMNS, MLP_UP_WEIGHT_GRAD_BF16_STAGES, MLP_UP_WEIGHT_GRAD_BF16_CLUSTER, PRODUCER_REGISTERS, IN_FLIGHT, STAGED_ROWS, STAGED_ROW_PADDING>(
            product_val, splits_val, MLP_UP_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_BF16_BLOCKS);
}

/// A variant: its template choices
/// , whether it fits a block's shared memory at this file's square,
/// and its run.
struct Variant
{
    int producer_registers_val;
    int in_flight_val;
    int staged_rows_val;
    int staged_row_padding_val;
    bool fits_val;
    void (*run_ptr)(const Product &, int);
};

/// The variants: the producer's registers (24 to 56) x the steps in flight (0 to 3; more than STAGES - 1 runs as
/// STAGES - 1)
/// at this file's staged store, then every staged rows (16, 32, 64) x padding (0, 4, 8, 16) at this file's registers
/// and steps in flight
/// .
static const Variant VARIANTS_LIST[] = {
    {24, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<24, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {24, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<24, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {24, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<24, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {24, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<24, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {32, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<32, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {32, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<32, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {32, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<32, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {32, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<32, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {40, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<40, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {40, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<40, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {40, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<40, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {40, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<40, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {48, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<48, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {48, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<48, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {48, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<48, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {48, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<48, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {56, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<56, 0, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {56, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<56, 1, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {56, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<56, 2, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {56, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING, variant_fits<MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>(),
     variant_at<56, 3, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS, MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 0, variant_fits<16, 0>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 0>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 4, variant_fits<16, 4>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 4>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 8, variant_fits<16, 8>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 8>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 16, variant_fits<16, 16>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 16, 16>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 0, variant_fits<32, 0>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 0>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 4, variant_fits<32, 4>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 4>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 8, variant_fits<32, 8>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 8>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 16, variant_fits<32, 16>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 32, 16>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 0, variant_fits<64, 0>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 0>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 4, variant_fits<64, 4>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 4>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 8, variant_fits<64, 8>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 8>},
    {MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 16, variant_fits<64, 16>(),
     variant_at<MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS, MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT, 64, 16>},
};

/// Gives TFLOPS for the product taking milliseconds_val.
static double tflops_of(double milliseconds_val)
{
    return (double)FLOPS_PER_MULTIPLY_ADD * ROWS * COLUMNS * SUM_LENGTH / (milliseconds_val / MILLISECONDS_PER_SECOND) / TERA;
}

// ══ part 1: against PyTorch (cuBLAS) ══════════════════════════════════════════

static void time_against_cublas(const Product &product_val)
{
    // Prints this file's setting's time and PyTorch's bfloat16 matmul's (cuBLAS), TIMED_STEPS of each taking turns,
    // steps WARM_UP_STEPS + 1 on averaged, with TFLOPS and the share of this GPU's bfloat16 peak.
    // 1. PyTorch's copies, as cuBLAS takes them (views turned round, no copies)
    torch::Tensor output_grad_turned_tensor = pytorch_bf16_copy_of(product_val.output_grad_tensor).t();
    torch::Tensor input_tensor = pytorch_bf16_copy_of(product_val.input_tensor);
    torch::Tensor pytorch_output_tensor = torch::zeros({ROWS, COLUMNS}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    // 2. take turns
    double mine_total_val = 0.0, cublas_total_val = 0.0;
    for (int step_index_val = 1; step_index_val <= TIMED_STEPS; step_index_val++)
    {
        float mine_val = gpu_milliseconds([&] {
            mine_at<MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_BF16_COLUMNS, MLP_UP_WEIGHT_GRAD_BF16_STAGES, MLP_UP_WEIGHT_GRAD_BF16_CLUSTER>(product_val,
                                                                           MLP_UP_WEIGHT_GRAD_BF16_SPLITS,
                                                                           MLP_UP_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_BF16_BLOCKS);
        });
        float cublas_val = gpu_milliseconds([&] {
            pytorch_output_tensor.addmm_(output_grad_turned_tensor, input_tensor);
        });
        if (step_index_val <= WARM_UP_STEPS)
            continue;
        mine_total_val += mine_val;
        cublas_total_val += cublas_val;
    }
    // 3. print
    const int averaged_val = TIMED_STEPS - WARM_UP_STEPS;
    double mine_ms_val = mine_total_val / averaged_val, cublas_ms_val = cublas_total_val / averaged_val;
    print_test_name(__func__);
    printf("    mlp up weight grad: output {%d, %d}, a sum of %d: %.1f GFLOP\n", ROWS, COLUMNS, SUM_LENGTH,
           (double)FLOPS_PER_MULTIPLY_ADD * ROWS * COLUMNS * SUM_LENGTH / GIGA);
    printf("    average of steps %d..%d:            ms     TFLOPS   of bf16 peak\n", WARM_UP_STEPS + 1, TIMED_STEPS);
    printf("        mine (this file's setting) %9.3f   %8.1f   %6.1f%%\n", mine_ms_val, tflops_of(mine_ms_val),
           100.0 * tflops_of(mine_ms_val) * TERA / peak_bf16_flops());
    printf("        PyTorch (cuBLAS)           %9.3f   %8.1f   %6.1f%%\n", cublas_ms_val, tflops_of(cublas_ms_val),
           100.0 * tflops_of(cublas_ms_val) * TERA / peak_bf16_flops());
    printf("    cuBLAS takes %.2fx as long as mine\n", cublas_ms_val / mine_ms_val);
}

// ══ part 2: every square ══════════════════════════════════════════════════════

static int time_each_square(const Product &product_val)
{
    // Prints each square's time at this file's block order, block count and cluster
    // and split count
    // , all taking turns over SWEEP_ROUNDS rounds; gives the fastest one's place in SQUARES_LIST.
    std::vector<double> totals_list(SQUARE_COUNT, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int square_index_val = 0; square_index_val < SQUARE_COUNT; square_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] {
                SQUARES_LIST[square_index_val].run_ptr(product_val, MLP_UP_WEIGHT_GRAD_BF16_CLUSTER, MLP_UP_WEIGHT_GRAD_BF16_SPLITS, MLP_UP_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_BF16_BLOCKS);
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
        bool this_files_val = square_val.row_groups_val == MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS && square_val.columns_val == MLP_UP_WEIGHT_GRAD_BF16_COLUMNS &&
                              square_val.stages_val == MLP_UP_WEIGHT_GRAD_BF16_STAGES;
        printf("        %d x %3d x %d   %8.3f ms   %6.1f TFLOPS%s\n", square_val.row_groups_val, square_val.columns_val, square_val.stages_val,
               milliseconds_val, tflops_of(milliseconds_val), this_files_val ? "   (this file's)" : "");
        if (totals_list[square_index_val] < totals_list[fastest_index_val])
            fastest_index_val = square_index_val;
    }
    printf("    fastest: %d x %d x %d\n", SQUARES_LIST[fastest_index_val].row_groups_val, SQUARES_LIST[fastest_index_val].columns_val,
           SQUARES_LIST[fastest_index_val].stages_val);
    return fastest_index_val;
}

// ══ part 3: order, blocks and cluster ═════════════════════════════════════════

/// A launch choice at a square: its cluster, block order and block count
/// , and split count
/// .
struct Choice
{
    int cluster_val;
    int block_rows_per_group_val;
    int blocks_val;
    int splits_val;
};

static Choice time_each_choice(const Product &product_val, int square_index_val)
{
    // Prints, at part 2's fastest square, each cluster (1, 2), block order (1, 4, 16, 64 rows of squares a group) and
    // block count (as many as run at once, one a square)
    // and split count (0: chosen for this GPU by mlp_up_weight_grad_bf16_auto_splits; 1 to 8, 12, 16)
    // , all taking turns over SWEEP_ROUNDS rounds; gives the fastest.
    std::vector<Choice> choices_list;
    for (int cluster_val : {1, 2})
        for (int block_rows_per_group_val : {1, 4, 16, 64})
            for (int blocks_val : {0, MLP_UP_WEIGHT_GRAD_BF16_ONE_A_SQUARE})
                for (int splits_val : {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16})
                    choices_list.push_back({cluster_val, block_rows_per_group_val, blocks_val, splits_val});
    const Square &square_val = SQUARES_LIST[square_index_val];
    std::vector<double> totals_list(choices_list.size(), 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (size_t choice_index_val = 0; choice_index_val < choices_list.size(); choice_index_val++)
        {
            const Choice &choice_val = choices_list[choice_index_val];
            float milliseconds_val = gpu_milliseconds([&] {
                square_val.run_ptr(product_val, choice_val.cluster_val, choice_val.splits_val, choice_val.block_rows_per_group_val,
                                   choice_val.blocks_val);
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
        printf("        clusters of %d, %2d rows a group, %-14s, %2d splits   %8.3f ms   %6.1f TFLOPS\n", choice_val.cluster_val,
               choice_val.block_rows_per_group_val, choice_val.blocks_val == 0 ? "persistent" : "one a square", choice_val.splits_val,
               milliseconds_val, tflops_of(milliseconds_val));
        if (totals_list[choice_index_val] < totals_list[fastest_index_val])
            fastest_index_val = choice_index_val;
    }
    return choices_list[fastest_index_val];
}

// ══ part 4: the splits' sum's block size ══════════════════════════════════════

static int time_each_sum_block_size(const Product &product_val, int splits_val)
{
    // Prints, at this file's square and splits_val splits, the time at each of the splits' sum's block sizes (64 to
    // 1024 threads), taking turns over SWEEP_ROUNDS rounds: the product's own time is the same in every column, so
    // the differences are the sum's. Gives the fastest.
    const int threads_list[] = {64, 128, 256, 512, 1024};
    const int threads_count_val = (int)(sizeof(threads_list) / sizeof(threads_list[0]));
    std::vector<double> totals_list(threads_count_val, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
        {
            float milliseconds_val = gpu_milliseconds([&] {
                mine_at<MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_BF16_COLUMNS, MLP_UP_WEIGHT_GRAD_BF16_STAGES, MLP_UP_WEIGHT_GRAD_BF16_CLUSTER>(product_val, splits_val, MLP_UP_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP, MLP_UP_WEIGHT_GRAD_BF16_BLOCKS,
                                                                               threads_list[threads_index_val]);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[threads_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    int fastest_index_val = 0;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, %d splits; threads a block of the splits' sum\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS, splits_val);
    for (int threads_index_val = 0; threads_index_val < threads_count_val; threads_index_val++)
    {
        printf("        %4d   %8.3f ms\n", threads_list[threads_index_val], totals_list[threads_index_val] / averaged_val);
        if (totals_list[threads_index_val] < totals_list[fastest_index_val])
            fastest_index_val = threads_index_val;
    }
    return threads_list[fastest_index_val];
}

// ══ part 5: the template choices, at this file's square ══════════════════════

/// True when a variant is worth timing: its steps in flight at most STAGES - 1 (more runs as STAGES - 1), and its
/// producer register count this file's when there is 1 consumer warpgroup (no registers are moved then)
/// , and it fits a block's shared memory
/// .
static bool variant_counts(const Variant &variant_val)
{
    bool counts_val = variant_val.in_flight_val <= MLP_UP_WEIGHT_GRAD_BF16_STAGES - 1 &&
                      (MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS > 1 || variant_val.producer_registers_val == MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS);
    counts_val = counts_val && variant_val.fits_val;
    return counts_val;
}

/// The fastest variants of part 5: the producer's registers and steps in flight
/// (at this file's staged store), and the staged rows and padding (at this file's registers and steps in flight)
/// .
struct VariantChoice
{
    int producer_registers_val;
    int in_flight_val;
    int staged_rows_val;
    int staged_row_padding_val;
};

static VariantChoice time_each_variant(const Product &product_val
                                       , int splits_val
                                       )
{
    // Prints, at this file's square, cluster, block order and block count
    // (and splits_val splits)
    // , each variant's time, all taking turns over SWEEP_ROUNDS rounds: the producer's registers (only with 2
    // consumer warpgroups) x the steps in flight (0 to STAGES - 1)
    // , then the staged rows x padding that fit
    // . These are template choices compiled for this file's square: when parts 2 and 3 change the square, update
    // its defines and run again to time these at the new one.
    const int variant_count_val = (int)(sizeof(VARIANTS_LIST) / sizeof(VARIANTS_LIST[0]));
    std::vector<double> totals_list(variant_count_val, 0.0);
    for (int round_index_val = 1; round_index_val <= SWEEP_ROUNDS; round_index_val++)
        for (int variant_index_val = 0; variant_index_val < variant_count_val; variant_index_val++)
        {
            const Variant &variant_val = VARIANTS_LIST[variant_index_val];
            if (!variant_counts(variant_val))
                continue;
            float milliseconds_val = gpu_milliseconds([&] {
                variant_val.run_ptr(product_val, splits_val);
            });
            if (round_index_val > WARM_UP_STEPS)
                totals_list[variant_index_val] += milliseconds_val;
        }
    const int averaged_val = SWEEP_ROUNDS - WARM_UP_STEPS;
    int fastest_index_val = -1;
    int fastest_staged_index_val = -1;
    print_test_name(__func__);
    printf("    ms, average of rounds %d..%d, at %d x %d x %d\n", WARM_UP_STEPS + 1, SWEEP_ROUNDS, MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS, MLP_UP_WEIGHT_GRAD_BF16_COLUMNS, MLP_UP_WEIGHT_GRAD_BF16_STAGES);
    for (int variant_index_val = 0; variant_index_val < variant_count_val; variant_index_val++)
    {
        const Variant &variant_val = VARIANTS_LIST[variant_index_val];
        if (!variant_counts(variant_val))
            continue;
        double milliseconds_val = totals_list[variant_index_val] / averaged_val;
        printf("        producer %2d registers, %d steps in flight, %2d staged rows, padding %2d   %8.3f ms   %6.1f TFLOPS\n",
               variant_val.producer_registers_val, variant_val.in_flight_val, variant_val.staged_rows_val, variant_val.staged_row_padding_val,
               milliseconds_val, tflops_of(milliseconds_val));
        bool files_staged_val = variant_val.staged_rows_val == MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS && variant_val.staged_row_padding_val == MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING;
        bool files_registers_val = variant_val.producer_registers_val == MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS && variant_val.in_flight_val == MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT;
        if (files_staged_val && (fastest_index_val < 0 || totals_list[variant_index_val] < totals_list[fastest_index_val]))
            fastest_index_val = variant_index_val;
        if (files_registers_val && (fastest_staged_index_val < 0 || totals_list[variant_index_val] < totals_list[fastest_staged_index_val]))
            fastest_staged_index_val = variant_index_val;
    }
    VariantChoice choice_val;
    choice_val.producer_registers_val = VARIANTS_LIST[fastest_index_val].producer_registers_val;
    choice_val.in_flight_val = VARIANTS_LIST[fastest_index_val].in_flight_val;
    choice_val.staged_rows_val = VARIANTS_LIST[fastest_staged_index_val].staged_rows_val;
    choice_val.staged_row_padding_val = VARIANTS_LIST[fastest_staged_index_val].staged_row_padding_val;
    return choice_val;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_tune_log(__FILE__);                   // the console, and tests/results/<GPU>/<kernel>_<date>_<time>.txt
    torch::manual_seed(0);
    Product product_val;
    time_against_cublas(product_val);
    int square_index_val = time_each_square(product_val);
    Choice choice_val = time_each_choice(product_val, square_index_val);
    int sum_threads_val = MLP_UP_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK;
    if (choice_val.splits_val > 1)
        sum_threads_val = time_each_sum_block_size(product_val, choice_val.splits_val);
    VariantChoice variant_choice_val = time_each_variant(product_val, MLP_UP_WEIGHT_GRAD_BF16_SPLITS);
    // the fastest, as the defines of the product's file
    const Square &square_val = SQUARES_LIST[square_index_val];
    printf("\n    for llmc/kernels/bf16/prod/mlp_up_weight_grad_bf16.cuh:\n");
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_ROW_GROUPS %d\n#define MLP_UP_WEIGHT_GRAD_BF16_COLUMNS %d\n#define MLP_UP_WEIGHT_GRAD_BF16_STAGES %d\n", square_val.row_groups_val, square_val.columns_val,
           square_val.stages_val);
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_SPLITS %d\n", choice_val.splits_val);
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_BLOCK_ROWS_PER_GROUP %d\n#define MLP_UP_WEIGHT_GRAD_BF16_CLUSTER %d\n#define MLP_UP_WEIGHT_GRAD_BF16_BLOCKS %s\n", choice_val.block_rows_per_group_val,
           choice_val.cluster_val, choice_val.blocks_val == 0 ? "0" : "MLP_UP_WEIGHT_GRAD_BF16_ONE_A_SQUARE");
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_SPLITS_SUM_THREADS_PER_BLOCK %d\n", sum_threads_val);
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_PRODUCER_REGISTERS %d\n#define MLP_UP_WEIGHT_GRAD_BF16_IN_FLIGHT %d\n", variant_choice_val.producer_registers_val,
           variant_choice_val.in_flight_val);
    printf("#define MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROWS %d\n#define MLP_UP_WEIGHT_GRAD_BF16_STAGED_ROW_PADDING %d\n", variant_choice_val.staged_rows_val,
           variant_choice_val.staged_row_padding_val);
    printf("    (part 5's at this file's square: if parts 2 and 3 changed it, update the defines and run again)\n");

    printf("\nall tunes done\n");
    return 0;
}
