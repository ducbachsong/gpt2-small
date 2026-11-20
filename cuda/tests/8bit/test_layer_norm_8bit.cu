// test_layer_norm_8bit.cu — tests for cuda/kernels/8bit/layer_norm_8bit.cuh (layer_norm_8bit_forward,
// layer_norm_8bit_backward): LayerNorm for a bfloat16 model whose output is written straight in 8 bits, both copies
// the products read (FP8 E4M3, as it sits and turned round), one scale for the tensor. Every check prints its test
// case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks the bfloat16 output against layer_norm_bf16's, both copies against round_8bit_per_tensor's on that
// output at the same delayed scale (as the model's forward runs it), and the backward against layer_norm_bf16's;
// part 2 checks numbers worked out by hand; part 3 checks that the forward's registers fit its block, and that any
// block size and any tile give the same bits. Any GPU. The timings are in cuda/tests/8bit/tune_layer_norm_8bit.cu.
#include <torch/types.h>              // torch::from_blob, float8_e4m3fn; libtorch before any CUDA header
#include "../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../kernels/8bit/layer_norm_8bit.cuh"

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
#define ODD_SEQ_LEN 48                // rows no tile divides: 48 = 32 + 16, one short tile at 32 and 64
#define ODD_EMBED_DIM 80              // a row of 10 loads, fewer than a warp's lanes
#define SMALL_EMBED_DIM ROUND_8BIT_ROW_LENGTH_MULTIPLE   // part 2's rows: the narrowest an 8-bit product takes, 16
#define SMALL_ROWS ROUND_8BIT_ROW_LENGTH_MULTIPLE        // part 2's row count: the turned copy's rows are whole 16-byte stores, 16

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a bfloat16 tensor on the GPU holding `values_list`, each rounded to bfloat16.
///
///   shape_list    the tensor's shape
///   values_list   in row order
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

/// Gives every value of a PyTorch float tensor as a CPU list, in row order.
static std::vector<float> floats_of(const torch::Tensor &tensor)
{
    torch::Tensor cpu_tensor = tensor.to(torch::kFloat).contiguous().cpu();
    return std::vector<float>(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
}

/// Gives an E4M3 copy turned back by PyTorch: each E4M3 times the tensor's one scale, exact.
///
///   copy_8bit_tensor   the E4M3 copy
///   scale_tensor       {1} float: its scale
///   rows_val           the copy's rows
///   row_length_val     its row length
static torch::Tensor turned_back(const Tensor &copy_8bit_tensor, const Tensor &scale_tensor, size_t rows_val, size_t row_length_val)
{
    // Gives an E4M3 copy turned back by PyTorch: each E4M3 (read as PyTorch's float8_e4m3fn) times the tensor's one
    // scale, exact. {rows, row_length} float.
    torch::Tensor e4m3_tensor = torch::from_blob(copy_8bit_tensor.data_ptr(), {(int64_t)rows_val, (int64_t)row_length_val},
                                                 torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    return e4m3_tensor.to(torch::kFloat) * pytorch_copy_of(scale_tensor);
}

/// One 8-bit LayerNorm's tensors: the inputs, the forward's results and the gradients.
struct LayerNormTensors
{
    // One 8-bit LayerNorm's tensors: the inputs, the forward's results (the bfloat16 output, both E4M3 copies and their
    // one scale, mean and rstd, the place's history) and the gradients. Results 0 at first; the copies and their scale 7s,
    // so a byte not written shows.
    size_t rows_val;
    size_t embed_dim_val;
    Tensor input_tensor;
    Tensor weight_tensor;
    Tensor bias_tensor;
    Tensor output_grad_tensor;
    Tensor output_tensor;              // bfloat16; Tensor(): not written
    Tensor rows_8bit_tensor;           // {rows, embed_dim} E4M3
    Tensor turned_8bit_tensor;         // {embed_dim, rows} E4M3
    Tensor scale_tensor;               // {1} float: both copies' one scale
    Tensor mean_tensor;
    Tensor rstd_tensor;
    Tensor input_grad_tensor;
    Tensor weight_grad_tensor;
    Tensor bias_grad_tensor;
    Round8bitHistory history_val;      // the place's history: where the delayed scale comes from
    Tensor largest_tensor;             // forward_with_largest's: [the largest given, the next largest] as float bits

    /// Gives a copy with the same inputs, and results, gradients and a history of its own, all fresh.
    LayerNormTensors with_fresh_results(void) const
    {
        LayerNormTensors tensors_val = *this;
        tensors_val.output_tensor = Tensor::zeros(input_tensor.shape(), DType::BFloat16);
        tensors_val.rows_8bit_tensor = Tensor::zeros({rows_val, embed_dim_val}, DType::E4M3);
        tensors_val.turned_8bit_tensor = Tensor::zeros({embed_dim_val, rows_val}, DType::E4M3);
        tensors_val.scale_tensor = Tensor::zeros({1});
        cudaCheck(cudaMemset(tensors_val.rows_8bit_tensor.data_ptr(), 7, rows_val * embed_dim_val));
        cudaCheck(cudaMemset(tensors_val.turned_8bit_tensor.data_ptr(), 7, rows_val * embed_dim_val));
        tensors_val.scale_tensor.fill_(7.0);
        tensors_val.mean_tensor = Tensor::zeros(mean_tensor.shape());
        tensors_val.rstd_tensor = Tensor::zeros(rstd_tensor.shape());
        tensors_val.input_grad_tensor = Tensor::zeros(input_tensor.shape(), DType::BFloat16);
        tensors_val.weight_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
        tensors_val.bias_grad_tensor = Tensor::zeros({embed_dim_val}, DType::BFloat16);
        tensors_val.history_val = Round8bitHistory();
        tensors_val.largest_tensor = Tensor::zeros({2});
        return tensors_val;
    }

    /// Gives the copies' tensors with a scale from a largest times a margin.
    ///
    ///   scale_largest_ptr   the largest the scale comes from
    ///   margin_val          multiplies it
    ///   next_largest_ptr    this tensor's largest is maxed onto it
    LayerNorm8bitCopies copies(const unsigned *scale_largest_ptr, float margin_val, unsigned *next_largest_ptr) const
    {
        LayerNorm8bitCopies copies_val;
        copies_val.rows_8bit_tensor = rows_8bit_tensor;
        copies_val.turned_8bit_tensor = turned_8bit_tensor;
        copies_val.scale_tensor = scale_tensor;
        copies_val.scale_largest_ptr = scale_largest_ptr;
        copies_val.margin_val = margin_val;
        copies_val.next_largest_ptr = next_largest_ptr;
        return copies_val;
    }

    /// Starts the history as the model's first micro-batch does.
    ///
    ///   first_tensor   rounded exactly, its largest kept
    void start_history(const Tensor &first_tensor) const
    {
        // `first_tensor` rounded exactly by round_8bit_per_tensor (into copies of its own, thrown away), its largest
        // kept.
        round_8bit_per_tensor(Tensor::zeros({rows_val, embed_dim_val}, DType::E4M3), Tensor::zeros({embed_dim_val, rows_val}, DType::E4M3),
                              Tensor::zeros({1}), first_tensor, &history_val);
    }

    /// Runs layer_norm_8bit_forward with the history's delayed scale.
    void forward(void) const
    {
        // (round_8bit_delayed_largest, x ROUND_8BIT_DELAYED_MARGIN): what the model's forward does after a layer's
        // first micro-batch.
        const unsigned *scale_largest_ptr = nullptr;
        unsigned *next_largest_ptr = nullptr;
        if (!round_8bit_delayed_largest(history_val, scale_largest_ptr, next_largest_ptr))
        {
            fprintf(stderr, "forward needs a history with a call already (start_history)\n");
            exit(EXIT_FAILURE);
        }
        layer_norm_8bit_forward(output_tensor, copies(scale_largest_ptr, ROUND_8BIT_DELAYED_MARGIN, next_largest_ptr), mean_tensor,
                                rstd_tensor, input_tensor, weight_tensor, bias_tensor);
    }

    /// Runs layer_norm_8bit_forward_at<TILE_ROWS> with a scale from `largest_val` x `margin_val`.
    ///
    ///   largest_val             the largest the scale comes from
    ///   margin_val              multiplies it
    ///   threads_per_block_val   the block size; the default if not given
    template <int TILE_ROWS = LAYER_NORM_8BIT_TILE_ROWS>
    void forward_with_largest(float largest_val, float margin_val, int threads_per_block_val = LAYER_NORM_8BIT_THREADS_PER_BLOCK) const
    {
        // This tensor's largest into largest_tensor's second slot (0 before), at a block size given (the default if
        // not).
        std::vector<float> slots_list = {largest_val, 0.0f};           // a float's bits are what the kernel reads
        largest_tensor.copy_from_cpu(slots_list.data());
        unsigned *slots_ptr = reinterpret_cast<unsigned *>(largest_tensor.data());
        layer_norm_8bit_forward_at<TILE_ROWS>(output_tensor, copies(slots_ptr, margin_val, slots_ptr + 1), mean_tensor, rstd_tensor,
                                              input_tensor, weight_tensor, bias_tensor, threads_per_block_val);
    }

    /// Gives the row copy turned back: {rows, embed_dim} float.
    torch::Tensor rows_turned_back(void) const
    {
        return turned_back(rows_8bit_tensor, scale_tensor, rows_val, embed_dim_val);
    }

    /// Gives the turned copy turned back: {embed_dim, rows} float.
    torch::Tensor turned_turned_back(void) const
    {
        return turned_back(turned_8bit_tensor, scale_tensor, embed_dim_val, rows_val);
    }

    /// Gives the largest |value| forward_with_largest's run found for the next call.
    float next_largest(void) const
    {
        return largest_tensor.tolist()[1];
    }
};

// Makes an 8-bit LayerNorm's tensors of {batch_size, seq_len, embed_dim}: input, weight, bias, output_grad from
// `input_list`, `weight_list`, `bias_list` (random where empty) and random output_grad, in bfloat16.
/// Makes an 8-bit LayerNorm's tensors of {batch_size, seq_len, embed_dim}, random where not given.
///
///   batch_size_val   rows of the batch
///   seq_len_val      tokens a row
///   embed_dim_val    the row's width
///   input_list       input's values; random if empty
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
    tensors_val.mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    return tensors_val.with_fresh_results();
}

/// True when two LayerNorms' copies hold the same bytes and scale.
static bool same_copies(const LayerNormTensors &first_val, const LayerNormTensors &second_val)
{
    return first_val.rows_8bit_tensor.equal(second_val.rows_8bit_tensor) &&
           first_val.turned_8bit_tensor.equal(second_val.turned_8bit_tensor) &&
           first_val.scale_tensor.tolist() == second_val.scale_tensor.tolist();
}

// ══ part 1: mine against layer_norm_bf16's and round_8bit_per_tensor's ════════

static void output_is_layer_norm_bf16_s(void)
{
    // Expect: GPT-2 small's micro-batch, 4096 random rows of 768. mean and rstd the same bits as layer_norm_bf16_forward's
    // (the same row_mean_rstd on the same values); output within ONE_ROUNDING_TOLERANCE of its (the same math, but the
    // compiler may order it another way in this kernel). Whether every bit is the same is printed.
    // 1. random tensors; mine (a scale from largest 4, x 2) and layer_norm_bf16's
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = layer_norm_of(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM);
    LayerNormTensors bf16_val = tensors_val.with_fresh_results();
    tensors_val.forward_with_largest(4.0f, ROUND_8BIT_DELAYED_MARGIN);
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

static void the_copies_are_round_8bit_per_tensor_s_of_the_output(void)
{
    // Expect: as the model's forward runs it after a layer's first micro-batch, at GPT-2 small's size and at one no
    // tile divides (48 rows of 80): a history started by an exact first call, then mine at its delayed scale (the
    // first call's largest x 2). Both copies, their one scale and the history (its next largest, its calls) are
    // round_8bit_per_tensor's on the bfloat16 output mine wrote, at the same delayed scale, to the last bit: each value
    // rounded to bfloat16, divided by the same power of 2 and rounded to E4M3 the same way. So the products can take
    // these copies in place of rounding the output, forward and backward.
    print_test_name(__func__);
    const size_t sizes_list[2][3] = {{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM}, {1, ODD_SEQ_LEN, ODD_EMBED_DIM}};
    for (const auto &size_val : sizes_list)
    {
        // 1. random tensors; the first micro-batch's output (layer_norm_bf16's), which starts both histories
        Tensor::manual_seed(0);
        LayerNormTensors mine_val = layer_norm_of(size_val[0], size_val[1], size_val[2]);
        LayerNormTensors reference_val = mine_val.with_fresh_results();
        layer_norm_bf16_forward(reference_val.output_tensor, reference_val.mean_tensor, reference_val.rstd_tensor,
                                reference_val.input_tensor, reference_val.weight_tensor, reference_val.bias_tensor);
        mine_val.start_history(reference_val.output_tensor);
        reference_val.start_history(reference_val.output_tensor);
        // 2. mine at the delayed scale; the reference: round_8bit_per_tensor's on my output, the same delayed turn
        mine_val.forward();
        round_8bit_per_tensor(reference_val.rows_8bit_tensor, reference_val.turned_8bit_tensor, reference_val.scale_tensor,
                              mine_val.output_tensor, &reference_val.history_val);
        // 3. the same bits
        printf("    rows %zu, embed_dim %zu:\n", mine_val.rows_val, mine_val.embed_dim_val);
        ASSERT_EQ(same_copies(mine_val, reference_val), true);
        ASSERT_EQ(mine_val.history_val.largest_tensor.tolist(), reference_val.history_val.largest_tensor.tolist());
        ASSERT_EQ(mine_val.history_val.calls_val, reference_val.history_val.calls_val);
    }
}

static void without_an_output_the_copies_are_the_same(void)
{
    // Expect: 256 random rows of 768, the forward run twice from the same inputs and scale: once writing the bfloat16
    // output, once with Tensor() in its place (the model's way: nothing but the copies kept). The copies, their scale,
    // mean, rstd and the next largest are the same bits; the bfloat16 output is only written, never read.
    // 1. random tensors; with and without the output
    Tensor::manual_seed(0);
    LayerNormTensors with_val = layer_norm_of(4, 64, GPT2_EMBED_DIM);
    LayerNormTensors without_val = with_val.with_fresh_results();
    without_val.output_tensor = Tensor();
    with_val.forward_with_largest(4.0f, ROUND_8BIT_DELAYED_MARGIN);
    without_val.forward_with_largest(4.0f, ROUND_8BIT_DELAYED_MARGIN);
    // 2. check
    ASSERT_EQ(same_copies(without_val, with_val), true);
    ASSERT_EQ(without_val.mean_tensor.tolist(), with_val.mean_tensor.tolist());
    ASSERT_EQ(without_val.rstd_tensor.tolist(), with_val.rstd_tensor.tolist());
    ASSERT_EQ(without_val.next_largest(), with_val.next_largest());
}

static void backward_is_layer_norm_bf16_s(void)
{
    // Expect: 256 random rows of 768, backward from 0 here and with layer_norm_bf16_backward: the forward's rounding
    // passes the gradient through (straight-through), so input_grad is the same bits (one warp per row), weight_grad
    // and bias_grad within BFLOAT16_TOLERANCE (the blocks' adds land in another order).
    // 1. random tensors; one forward, two backwards
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = layer_norm_of(4, 64, GPT2_EMBED_DIM);
    tensors_val.forward_with_largest(4.0f, ROUND_8BIT_DELAYED_MARGIN);
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
// 16 rows of 16 (the turned copy's rows are whole 16-byte stores). The row 1 x 8, 3 x 8 has mean 2, variance 1, rstd
// 1 / sqrt(1.00001) = 0.999995, and norm -0.999995 x 8, 0.999995 x 8: -1 and 1 in bfloat16. E4M3 keeps 4 significant
// bits; between 256 and 448 it steps by 32, and past 448 it saturates. The scale is the smallest power of 2 with
// largest x margin / scale <= 448; both copies hold the same E4M3s, the turned one column by column.

/// Gives the 16 values of a part-2 row: `low_val` eight times, then `high_val` eight times.
static std::vector<float> row_of(float low_val, float high_val)
{
    std::vector<float> values_list(SMALL_EMBED_DIM / 2, low_val);
    values_list.resize(SMALL_EMBED_DIM, high_val);
    return values_list;
}

/// Gives `rows_val` part-2 rows, each `row_list`, one after the other.
static std::vector<float> rows_of(size_t rows_val, const std::vector<float> &row_list)
{
    std::vector<float> values_list;
    for (size_t row_index_val = 0; row_index_val < rows_val; row_index_val++)
        values_list.insert(values_list.end(), row_list.begin(), row_list.end());
    return values_list;
}

static void a_row_e4m3_holds_is_copied_exactly(void)
{
    // Expect: 16 rows of 1 x 8, 3 x 8, weight 1, bias 0: output -1 x 8, 1 x 8 in bfloat16. A scale from largest 1,
    // margin 1: 1 / 448 is between 2^-9 and 2^-8, so the scale is 2^-8 = 0.00390625, and each value over it is -256 or
    // 256, which E4M3 holds: both copies turned back are the output, exactly (the turned one its columns), every scale
    // 2^-8, and the next largest 1.
    // 1. the forward
    LayerNormTensors tensors_val = layer_norm_of(1, SMALL_ROWS, SMALL_EMBED_DIM, rows_of(SMALL_ROWS, row_of(1, 3)), row_of(1, 1),
                                                 row_of(0, 0));
    tensors_val.forward_with_largest(1.0f, 1.0f);
    // 2. check: output, scale, both copies turned back, the next largest
    std::vector<float> expected_list = rows_of(SMALL_ROWS, row_of(-1, 1));
    ASSERT_EQ(floats_of(tensors_val.output_tensor), expected_list);
    ASSERT_EQ(tensors_val.scale_tensor.tolist(), std::vector<float>(1, 0.00390625f));
    ASSERT_EQ(floats_of(tensors_val.rows_turned_back()), expected_list);
    ASSERT_EQ(floats_of(tensors_val.turned_turned_back().t()), expected_list);
    ASSERT_EQ(tensors_val.next_largest(), 1.0f);
}

static void a_value_e4m3_cannot_hold_is_rounded_in_the_copies_only(void)
{
    // Expect: the same rows, weight 1, bias 18 in column 0 and 0 elsewhere: output 17 in column 0 (-0.999995 + 18 =
    // 17.000005: 17 in bfloat16), then -1 x 7, 1 x 8. A scale from largest 17, margin 1: 17 / 448 is between 2^-5 and
    // 2^-4, so the scale is 2^-4 = 0.0625, and 17 / 0.0625 = 272 sits halfway between E4M3's 256 and 288; a tie goes
    // to the even last bit: 256, turned back 256 x 0.0625 = 16. The ±1s are ±16 over the scale, held exactly. So both
    // copies have 16 where the bfloat16 output keeps 17: the 8 bits' cost. The next largest is 17.
    // 1. the forward
    std::vector<float> bias_list(SMALL_EMBED_DIM, 0.0f);
    bias_list[0] = 18.0f;
    LayerNormTensors tensors_val = layer_norm_of(1, SMALL_ROWS, SMALL_EMBED_DIM, rows_of(SMALL_ROWS, row_of(1, 3)), row_of(1, 1),
                                                 bias_list);
    tensors_val.forward_with_largest(17.0f, 1.0f);
    // 2. check: output, scale, both copies turned back, the next largest
    std::vector<float> output_row_list = row_of(-1, 1);
    output_row_list[0] = 17.0f;
    std::vector<float> copy_row_list = row_of(-1, 1);
    copy_row_list[0] = 16.0f;
    ASSERT_EQ(floats_of(tensors_val.output_tensor), rows_of(SMALL_ROWS, output_row_list));
    ASSERT_EQ(tensors_val.scale_tensor.tolist(), std::vector<float>(1, 0.0625f));
    ASSERT_EQ(floats_of(tensors_val.rows_turned_back()), rows_of(SMALL_ROWS, copy_row_list));
    ASSERT_EQ(floats_of(tensors_val.turned_turned_back().t()), rows_of(SMALL_ROWS, copy_row_list));
    ASSERT_EQ(tensors_val.next_largest(), 17.0f);
}

static void a_value_past_the_delayed_scale_is_clamped(void)
{
    // Expect: the same rows and bias 18 in column 0, but the scale a delayed one from a call before whose largest was 1:
    // 1 x ROUND_8BIT_DELAYED_MARGIN (2) / 448 is between 2^-8 and 2^-7, so the scale is 2^-7 = 0.0078125, and the
    // copies hold at most 448 x 2^-7 = 3.5. 17 / 2^-7 = 2176 is past 448: saturated to 448, turned back 3.5. The ±1s
    // are ±128, held exactly. The next largest is 17, so the next call's scale has room for it: the delayed scale's cost
    // when a tensor jumps past its margin.
    // 1. the forward
    std::vector<float> bias_list(SMALL_EMBED_DIM, 0.0f);
    bias_list[0] = 18.0f;
    LayerNormTensors tensors_val = layer_norm_of(1, SMALL_ROWS, SMALL_EMBED_DIM, rows_of(SMALL_ROWS, row_of(1, 3)), row_of(1, 1),
                                                 bias_list);
    tensors_val.forward_with_largest(1.0f, ROUND_8BIT_DELAYED_MARGIN);
    // 2. check: scale, both copies turned back, the next largest
    std::vector<float> copy_row_list = row_of(-1, 1);
    copy_row_list[0] = 3.5f;
    ASSERT_EQ(tensors_val.scale_tensor.tolist(), std::vector<float>(1, 0.0078125f));
    ASSERT_EQ(floats_of(tensors_val.rows_turned_back()), rows_of(SMALL_ROWS, copy_row_list));
    ASSERT_EQ(floats_of(tensors_val.turned_turned_back().t()), rows_of(SMALL_ROWS, copy_row_list));
    ASSERT_EQ(tensors_val.next_largest(), 17.0f);
}

static void every_scale_is_the_tensor_s_one(void)
{
    // Expect: 8 rows of 1 x 8, 3 x 8, then 8 flat rows of 2s; weight 1, bias 0. The flat rows' output is the bias, all
    // 0. A scale from largest 1, margin 1: 2^-8 for every row, the flat ones too (no scale of their own, unlike
    // round_8bit's per-row 1), and for every column; the flat rows' copies all 0. The next largest is 1.
    // 1. the forward
    std::vector<float> input_list = rows_of(SMALL_ROWS / 2, row_of(1, 3));
    std::vector<float> flat_list = rows_of(SMALL_ROWS / 2, row_of(2, 2));
    input_list.insert(input_list.end(), flat_list.begin(), flat_list.end());
    LayerNormTensors tensors_val = layer_norm_of(1, SMALL_ROWS, SMALL_EMBED_DIM, input_list, row_of(1, 1), row_of(0, 0));
    tensors_val.forward_with_largest(1.0f, 1.0f);
    // 2. check: every scale 2^-8, both copies turned back, the next largest
    std::vector<float> expected_list = rows_of(SMALL_ROWS / 2, row_of(-1, 1));
    std::vector<float> zeros_list = rows_of(SMALL_ROWS / 2, row_of(0, 0));
    expected_list.insert(expected_list.end(), zeros_list.begin(), zeros_list.end());
    ASSERT_EQ(tensors_val.scale_tensor.tolist(), std::vector<float>(1, 0.00390625f));
    ASSERT_EQ(floats_of(tensors_val.rows_turned_back()), expected_list);
    ASSERT_EQ(floats_of(tensors_val.turned_turned_back().t()), expected_list);
    ASSERT_EQ(tensors_val.next_largest(), 1.0f);
}

// ══ part 3: the forward fits its launch, and any block size and tile work ═════

/// One tile the forward can take, its kernel and its run.
struct Tile
{
    int tile_rows_val;
    const void *kernel_ptr;
    void (*run_ptr)(const LayerNormTensors &, int);
};

/// Runs the forward at tile TILE_ROWS and a block size given, the scale from largest 4 x 2.
template <int TILE_ROWS>
static void forward_at_tile(const LayerNormTensors &tensors_val, int threads_per_block_val)
{
    tensors_val.forward_with_largest<TILE_ROWS>(4.0f, ROUND_8BIT_DELAYED_MARGIN, threads_per_block_val);
}

/// The tiles tune_layer_norm_8bit sweeps: 16, 32 and 64 rows a block.
static const Tile TILES_LIST[] = {
    {16, (const void *)layer_norm_8bit_forward_kernel<16>, forward_at_tile<16>},
    {32, (const void *)layer_norm_8bit_forward_kernel<32>, forward_at_tile<32>},
    {64, (const void *)layer_norm_8bit_forward_kernel<64>, forward_at_tile<64>},
};

static void the_forward_fits_its_block(void)
{
    // Expect: at every tile, layer_norm_8bit_forward_kernel can be launched with LAYER_NORM_8BIT_THREADS_PER_BLOCK
    // (256), and nothing is spilled: a lane's 8 values and a turned store's 4 words stay in registers. (The backward
    // is layer_norm_bf16's, checked in cuda/tests/bf16/test_layer_norm_bf16.cu.)
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    print_test_name(__func__);
    for (const Tile &tile_val : TILES_LIST)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, tile_val.kernel_ptr));
        printf("    layer_norm_8bit_forward_kernel<%d>: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               tile_val.tile_rows_val, attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= LAYER_NORM_8BIT_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_and_tile_gives_the_same_bits(void)
{
    // Expect: 4 x 48 random rows of 768 (192 rows: a short last tile at 64), the forward called at every tile with
    // threads_per_block 32, 128, 256 and 512. Each row is one warp's and each value rounded alone, whatever the block
    // and the tile, so output, mean, rstd, both copies, their scale and the next largest are the default's to the
    // last bit.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 8 * HARDWARE_THREADS_PER_WARP,
                                    16 * HARDWARE_THREADS_PER_WARP};
    // 1. random tensors; the default's results
    Tensor::manual_seed(0);
    LayerNormTensors usual_val = layer_norm_of(4, ODD_SEQ_LEN, GPT2_EMBED_DIM);
    usual_val.forward_with_largest(4.0f, ROUND_8BIT_DELAYED_MARGIN);

    // 2. each tile and block size, from the same inputs, against the default's
    for (const Tile &tile_val : TILES_LIST)
        for (int threads_per_block_val : block_sizes_list)
        {
            LayerNormTensors tensors_val = usual_val.with_fresh_results();
            tile_val.run_ptr(tensors_val, threads_per_block_val);
            print_test_name(__func__);
            printf("    tile %d rows, %d threads:\n", tile_val.tile_rows_val, threads_per_block_val);
            ASSERT_EQ(floats_of(tensors_val.output_tensor), floats_of(usual_val.output_tensor));
            ASSERT_EQ(tensors_val.mean_tensor.tolist(), usual_val.mean_tensor.tolist());
            ASSERT_EQ(tensors_val.rstd_tensor.tolist(), usual_val.rstd_tensor.tolist());
            ASSERT_EQ(same_copies(tensors_val, usual_val), true);
            ASSERT_EQ(tensors_val.next_largest(), usual_val.next_largest());
        }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    output_is_layer_norm_bf16_s();
    the_copies_are_round_8bit_per_tensor_s_of_the_output();
    without_an_output_the_copies_are_the_same();
    backward_is_layer_norm_bf16_s();

    a_row_e4m3_holds_is_copied_exactly();
    a_value_e4m3_cannot_hold_is_rounded_in_the_copies_only();
    a_value_past_the_delayed_scale_is_clamped();
    every_scale_is_the_tensor_s_one();

    the_forward_fits_its_block();
    any_block_size_and_tile_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
