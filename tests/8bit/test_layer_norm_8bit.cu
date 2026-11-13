// test_layer_norm_8bit.cu — tests for llmc/kernels/8bit/layer_norm_8bit.cuh (layer_norm_8bit_forward,
// layer_norm_8bit_backward): LayerNorm for a bfloat16 model whose forward also writes the output's 8-bit copy (FP8
// E4M3, one scale per row) for the 8-bit product after it. Every check prints its test case, input, expected and
// actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the bfloat16 output against layer_norm_bf16's, the copy against round_8bit's on that output, and the
// backward against layer_norm_bf16's; part 2 checks numbers worked out by hand; part 3 checks that the forward's
// registers fit its block, and that any block size gives the same bits. Any GPU. The timings are in
// tests/8bit/tune_layer_norm_8bit.cu.
#include <torch/types.h>              // torch::from_blob, float8_e4m3fn; libtorch before any CUDA header
#include "../test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/8bit/layer_norm_8bit.cuh"

// The output against layer_norm_bf16's: the same math, so the same bits but where the compiler orders the float math
// another way inside the kernel: then by one bfloat16 rounding. The largest difference, as a fraction of the largest
// value.
static const double ONE_ROUNDING_TOLERANCE = 1e-2;
// weight_grad and bias_grad between two backwards: each block's sums land in whatever order the blocks finish, each
// rounded to bfloat16.
static const double BFLOAT16_TOLERANCE = 2e-2;

#define GPT2_BATCH_SIZE 4             // a micro-batch: 4 rows of 1024 tokens
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define SMALL_EMBED_DIM ROUND_8BIT_ROW_LENGTH_MULTIPLE   // part 2's rows: the narrowest an 8-bit product takes, 16

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded to
/// bfloat16.
static Tensor bf16_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values (bfloat16 turned into floats exactly).
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);         // the tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of my tensor as a CPU list of floats (bfloat16 turned into floats, exactly).
static std::vector<float> floats_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// One 8-bit LayerNorm's tensors: the inputs, the forward's results (the bfloat16 output, its E4M3 copy and scales,
/// mean and rstd) and the gradients, all results 0 at first.
struct LayerNormTensors
{
    size_t rows_val;
    size_t embed_dim_val;
    Tensor input_tensor;
    Tensor weight_tensor;
    Tensor bias_tensor;
    Tensor output_grad_tensor;
    Tensor output_tensor;
    Tensor output_8bit_tensor;         // {rows, embed_dim / 4} Int32
    Tensor output_scales_tensor;       // {rows} float
    Tensor mean_tensor;
    Tensor rstd_tensor;
    Tensor input_grad_tensor;
    Tensor weight_grad_tensor;
    Tensor bias_grad_tensor;

    /// Gives a copy with the same inputs and results and gradients of its own, all 0.
    LayerNormTensors with_fresh_results(void) const
    {
        LayerNormTensors tensors_val = *this;
        tensors_val.output_tensor = Tensor::zeros(output_tensor.shape(), DType::BFloat16);
        tensors_val.output_8bit_tensor = Tensor::zeros(output_8bit_tensor.shape(), DType::Int32);
        tensors_val.output_scales_tensor = Tensor::zeros(output_scales_tensor.shape());
        tensors_val.mean_tensor = Tensor::zeros(mean_tensor.shape());
        tensors_val.rstd_tensor = Tensor::zeros(rstd_tensor.shape());
        tensors_val.input_grad_tensor = Tensor::zeros(input_grad_tensor.shape(), DType::BFloat16);
        tensors_val.weight_grad_tensor = Tensor::zeros(weight_grad_tensor.shape(), DType::BFloat16);
        tensors_val.bias_grad_tensor = Tensor::zeros(bias_grad_tensor.shape(), DType::BFloat16);
        return tensors_val;
    }

    /// Runs layer_norm_8bit_forward at this block size (the default if not given).
    void forward(int threads_per_block_val = LAYER_NORM_8BIT_THREADS_PER_BLOCK) const
    {
        layer_norm_8bit_forward(output_tensor, output_8bit_tensor, output_scales_tensor, mean_tensor, rstd_tensor, input_tensor,
                                weight_tensor, bias_tensor, threads_per_block_val);
    }

    /// Gives the copy turned back by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times its row's scale,
    /// exact. {rows, embed_dim} float.
    torch::Tensor turned_back(void) const
    {
        torch::Tensor e4m3_tensor = torch::from_blob(output_8bit_tensor.data_ptr(), {(int64_t)rows_val, (int64_t)embed_dim_val},
                                                     torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(output_scales_tensor).unsqueeze(1);
    }
};

/// Makes an 8-bit LayerNorm's tensors of {batch_size, seq_len, embed_dim}: input, weight, bias, output_grad from
/// `input_list`, `weight_list`, `bias_list` (random where empty) and random output_grad, in bfloat16.
static LayerNormTensors layer_norm_of(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val,
                                      const std::vector<float> &input_list = {}, const std::vector<float> &weight_list = {},
                                      const std::vector<float> &bias_list = {})
{
    Shape shape_list = {batch_size_val, seq_len_val, embed_dim_val};
    LayerNormTensors tensors_val;
    tensors_val.rows_val = batch_size_val * seq_len_val;
    tensors_val.embed_dim_val = embed_dim_val;
    tensors_val.input_tensor = bf16_of(shape_list, input_list.empty() ? random_values(shape_list) : input_list);
    tensors_val.weight_tensor = bf16_of({embed_dim_val}, weight_list.empty() ? random_values({embed_dim_val}) : weight_list);
    tensors_val.bias_tensor = bf16_of({embed_dim_val}, bias_list.empty() ? random_values({embed_dim_val}) : bias_list);
    tensors_val.output_grad_tensor = bf16_of(shape_list, random_values(shape_list));
    tensors_val.output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    tensors_val.output_8bit_tensor = Tensor::zeros({tensors_val.rows_val, embed_dim_val / ROUND_8BIT_VALUES_PER_INT}, DType::Int32);
    tensors_val.output_scales_tensor = Tensor::zeros({tensors_val.rows_val});
    tensors_val.mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.input_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    tensors_val.weight_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    tensors_val.bias_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
    return tensors_val;
}

// ══ part 1: mine against layer_norm_bf16's and round_8bit's ═══════════════════

static void output_is_layer_norm_bf16_s(void)
{
    // Expect: GPT-2 small's micro-batch, 4096 random rows of 768. mean and rstd the same bits as layer_norm_bf16_forward's
    // (the same row_mean_rstd on the same values); output within ONE_ROUNDING_TOLERANCE of its (the same math, but the
    // compiler may order it another way in this kernel). Whether every bit is the same is printed.
    // 1. random tensors; mine and layer_norm_bf16's
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = layer_norm_of(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM);
    LayerNormTensors bf16_val = tensors_val.with_fresh_results();
    tensors_val.forward();
    layer_norm_bf16_forward(bf16_val.output_tensor, bf16_val.mean_tensor, bf16_val.rstd_tensor, bf16_val.input_tensor,
                            bf16_val.weight_tensor, bf16_val.bias_tensor);
    // 2. check
    bool same_bits_val = torch::equal(pytorch_copy_of(tensors_val.output_tensor), pytorch_copy_of(bf16_val.output_tensor));
    print_test_name(__func__);
    printf("    output the same bits as layer_norm_bf16's: %s\n", same_bits_val ? "yes" : "no (within one rounding?)");
    ASSERT_EQ(tensors_val.mean_tensor.tolist(), bf16_val.mean_tensor.tolist());
    ASSERT_EQ(tensors_val.rstd_tensor.tolist(), bf16_val.rstd_tensor.tolist());
    ASSERT_NEAR(floats_of(tensors_val.output_tensor), floats_of(bf16_val.output_tensor), ONE_ROUNDING_TOLERANCE);
}

static void the_copy_is_round_8bit_s_of_the_output(void)
{
    // Expect: the same 4096 random rows of 768. The copy and scales the forward writes are round_8bit's on the
    // bfloat16 output it wrote, to the last bit: the same largest |value| of each row as stored, the same power-of-2
    // scale, the same rounding to E4M3. So an 8-bit product can take this copy in place of rounding the output again.
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = layer_norm_of(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM);
    tensors_val.forward();
    // 2. round_8bit's on my output
    LayerNormTensors rounded_val = tensors_val.with_fresh_results();
    round_8bit(rounded_val.output_8bit_tensor, rounded_val.output_scales_tensor, tensors_val.output_tensor);
    // 3. the same bits
    ASSERT_EQ(tensors_val.output_8bit_tensor.tolist<int>() == rounded_val.output_8bit_tensor.tolist<int>(), true);
    ASSERT_EQ(tensors_val.output_scales_tensor.tolist(), rounded_val.output_scales_tensor.tolist());
}

static void backward_is_layer_norm_bf16_s(void)
{
    // Expect: 256 random rows of 768, backward from 0 here and with layer_norm_bf16_backward: the forward's rounding
    // passes the gradient through (straight-through), so input_grad is the same bits (one warp per row), weight_grad
    // and bias_grad within BFLOAT16_TOLERANCE (the blocks' adds land in another order).
    // 1. random tensors; one forward, two backwards
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = layer_norm_of(4, 64, GPT2_EMBED_DIM);
    tensors_val.forward();
    LayerNormTensors bf16_val = tensors_val.with_fresh_results();
    layer_norm_8bit_backward(tensors_val.input_grad_tensor, tensors_val.weight_grad_tensor, tensors_val.bias_grad_tensor,
                             tensors_val.output_grad_tensor, tensors_val.input_tensor, tensors_val.weight_tensor,
                             tensors_val.mean_tensor, tensors_val.rstd_tensor);
    layer_norm_bf16_backward(bf16_val.input_grad_tensor, bf16_val.weight_grad_tensor, bf16_val.bias_grad_tensor,
                             tensors_val.output_grad_tensor, tensors_val.input_tensor, tensors_val.weight_tensor,
                             tensors_val.mean_tensor, tensors_val.rstd_tensor);
    // 2. check
    ASSERT_EQ(floats_of(tensors_val.input_grad_tensor), floats_of(bf16_val.input_grad_tensor));
    ASSERT_NEAR(floats_of(tensors_val.weight_grad_tensor), floats_of(bf16_val.weight_grad_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(floats_of(tensors_val.bias_grad_tensor), floats_of(bf16_val.bias_grad_tensor), BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Rows of 16: the row 1 x 8, 3 x 8 has mean 2, variance 1, rstd 1 / sqrt(1.00001) = 0.999995, and norm -0.999995 x 8,
// 0.999995 x 8: -1 and 1 in bfloat16. E4M3 keeps 4 significant bits; between 256 and 448 it steps by 32. A row's scale
// is the smallest power of 2 with largest |value| / scale <= 448.

/// Gives the 16 values of a part-2 row: `low_val` eight times, then `high_val` eight times.
static std::vector<float> row_of(float low_val, float high_val)
{
    std::vector<float> values_list(SMALL_EMBED_DIM / 2, low_val);
    values_list.resize(SMALL_EMBED_DIM, high_val);
    return values_list;
}

static void a_row_e4m3_holds_is_copied_exactly(void)
{
    // Expect: the row 1 x 8, 3 x 8, weight 1, bias 0: output -1 x 8, 1 x 8 in bfloat16. Its largest |value| is 1:
    // 1 / 448 is between 2^-9 and 2^-8, so the scale is 2^-8 = 0.00390625, and each value over it is -256 or 256,
    // which E4M3 holds: the copy turned back is the output, exactly.
    // 1. the forward
    LayerNormTensors tensors_val = layer_norm_of(1, 1, SMALL_EMBED_DIM, row_of(1, 3), row_of(1, 1), row_of(0, 0));
    tensors_val.forward();
    // 2. check: output, scale, the copy turned back
    torch::Tensor turned_back_tensor = tensors_val.turned_back().cpu();
    ASSERT_EQ(floats_of(tensors_val.output_tensor), row_of(-1, 1));
    ASSERT_EQ(tensors_val.output_scales_tensor.tolist(), std::vector<float>({0.00390625f}));
    ASSERT_EQ(std::vector<float>(turned_back_tensor.data_ptr<float>(), turned_back_tensor.data_ptr<float>() + SMALL_EMBED_DIM),
              row_of(-1, 1));
}

static void a_value_e4m3_cannot_hold_is_rounded_in_the_copy_only(void)
{
    // Expect: the same row, weight 1, bias 18 in column 0 and 0 elsewhere: output 17 in column 0 (-0.999995 + 18 =
    // 17.000005: 17 in bfloat16), then -1 x 7, 1 x 8. Largest |value| 17: 17 / 448 is between 2^-5 and 2^-4, so the
    // scale is 2^-4 = 0.0625, and 17 / 0.0625 = 272 sits halfway between E4M3's 256 and 288; a tie goes to the even
    // last bit: 256, turned back 256 x 0.0625 = 16. The ±1s are ±16 over the scale, held exactly. So the copy has 16
    // where the bfloat16 output keeps 17: the 8 bits' cost.
    // 1. the forward
    std::vector<float> bias_list(SMALL_EMBED_DIM, 0.0f);
    bias_list[0] = 18.0f;
    LayerNormTensors tensors_val = layer_norm_of(1, 1, SMALL_EMBED_DIM, row_of(1, 3), row_of(1, 1), bias_list);
    tensors_val.forward();
    // 2. check: output, scale, the copy turned back
    std::vector<float> expected_output_list = row_of(-1, 1);
    expected_output_list[0] = 17.0f;
    std::vector<float> expected_copy_list = row_of(-1, 1);
    expected_copy_list[0] = 16.0f;
    torch::Tensor turned_back_tensor = tensors_val.turned_back().cpu();
    ASSERT_EQ(floats_of(tensors_val.output_tensor), expected_output_list);
    ASSERT_EQ(tensors_val.output_scales_tensor.tolist(), std::vector<float>({0.0625f}));
    ASSERT_EQ(std::vector<float>(turned_back_tensor.data_ptr<float>(), turned_back_tensor.data_ptr<float>() + SMALL_EMBED_DIM),
              expected_copy_list);
}

static void each_row_gets_its_own_scale(void)
{
    // Expect: two rows, 1 x 8, 3 x 8 and a flat row of 0s; weight 1, bias 0. Row 0's output is ±1, scale 2^-8 as
    // above. Row 1 is flat, so its output is the bias, all 0: largest |value| 0, scale 1 (round_8bit's for a row of
    // 0s), its copy all 0. Each row's scale comes from its own values only.
    // 1. the forward on two rows
    std::vector<float> input_list = row_of(1, 3);
    for (float value_val : row_of(0, 0))
        input_list.push_back(value_val);
    LayerNormTensors tensors_val = layer_norm_of(1, 2, SMALL_EMBED_DIM, input_list, row_of(1, 1), row_of(0, 0));
    tensors_val.forward();
    // 2. check: the two scales, row 1's copy all 0
    torch::Tensor turned_back_tensor = tensors_val.turned_back().cpu();
    ASSERT_EQ(tensors_val.output_scales_tensor.tolist(), std::vector<float>({0.00390625f, 1.0f}));
    ASSERT_EQ(std::vector<float>(turned_back_tensor.data_ptr<float>() + SMALL_EMBED_DIM,
                                 turned_back_tensor.data_ptr<float>() + 2 * SMALL_EMBED_DIM),
              row_of(0, 0));
}

// ══ part 3: the forward fits its launch, and any block size works ═════════════

static void the_forward_fits_its_block(void)
{
    // Expect: layer_norm_8bit_forward_kernel can be launched with LAYER_NORM_8BIT_THREADS_PER_BLOCK (256), and nothing
    // is spilled: a lane's 8 values stay in registers. (The backward is layer_norm_bf16's, checked in
    // tests/bf16/test_layer_norm_bf16.cu.)
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    // 1. what the compiler wrote down
    cudaFuncAttributes attributes_val;
    cudaCheck(cudaFuncGetAttributes(&attributes_val, (const void *)layer_norm_8bit_forward_kernel));
    print_test_name(__func__);
    printf("    layer_norm_8bit_forward_kernel: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
           attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
    // 2. check
    ASSERT_EQ(attributes_val.maxThreadsPerBlock >= LAYER_NORM_8BIT_THREADS_PER_BLOCK, true);
    ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: 256 random rows of 768, the forward called with threads_per_block 32, 128, 256 and 512 (one warp up).
    // Each row is one warp's whatever the block size, so output, mean, rstd, the copy and its scales are the
    // default's to the last bit.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                    16 * HARDWARE_THREADS_PER_WARP};
    // 1. random tensors; the default's results
    Tensor::manual_seed(0);
    LayerNormTensors usual_val = layer_norm_of(4, 64, GPT2_EMBED_DIM);
    usual_val.forward();

    // 2. each block size, from the same inputs, against the default's
    for (int threads_per_block_val : block_sizes_list)
    {
        LayerNormTensors tensors_val = usual_val.with_fresh_results();
        tensors_val.forward(threads_per_block_val);
        ASSERT_EQ(floats_of(tensors_val.output_tensor), floats_of(usual_val.output_tensor));
        ASSERT_EQ(tensors_val.mean_tensor.tolist(), usual_val.mean_tensor.tolist());
        ASSERT_EQ(tensors_val.rstd_tensor.tolist(), usual_val.rstd_tensor.tolist());
        ASSERT_EQ(tensors_val.output_8bit_tensor.tolist<int>() == usual_val.output_8bit_tensor.tolist<int>(), true);
        ASSERT_EQ(tensors_val.output_scales_tensor.tolist(), usual_val.output_scales_tensor.tolist());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    output_is_layer_norm_bf16_s();
    the_copy_is_round_8bit_s_of_the_output();
    backward_is_layer_norm_bf16_s();

    a_row_e4m3_holds_is_copied_exactly();
    a_value_e4m3_cannot_hold_is_rounded_in_the_copy_only();
    each_row_gets_its_own_scale();

    the_forward_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
