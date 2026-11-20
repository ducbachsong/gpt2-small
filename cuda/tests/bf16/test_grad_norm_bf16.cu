// test_grad_norm_bf16.cu — tests for cuda/kernels/bf16/grad_norm_bf16.cuh (GradNormBf16): the norm of bfloat16
// gradients, 8 values a thread (one 16-byte load). Every check prints its test case, input, expected and actual
// value, and ok; the first FAIL stops the program.
//
// Part 1 checks the norm against PyTorch's, from torch::nn::utils::clip_grad_norm_ (it returns the norm of all the
// gradients it was given), and against fp32/grad_norm.cuh's bfloat16 path (4 values a thread); part 2 checks numbers
// worked out by hand (the cases of cuda/tests/fp32/test_grad_norm.cu, and the values before the first 16-byte boundary a
// GPU's slice can start with), how the work is shared out between the blocks, and any block size. Any GPU. The
// timings are in cuda/tests/bf16/tune_grad_norm_bf16.cu.
#include <ATen/ops/_foreach_norm.h>   // at::_foreach_norm: what Python's clip_grad_norm_ uses
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"          // ASSERT_EQ, ASSERT_NEAR
#include "../../common/tensor_buffer.cuh"  // TensorBuffer: float values in one allocation, around the norm
#include "../../kernels/fp32/grad_norm.cuh"      // the 4-value path, to compare with
#include "../../kernels/bf16/grad_norm_bf16.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the norm

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn, times `scale_val`, as a CPU list.
static std::vector<float> random_values(const Shape &shape_list, float scale_val = 1.0f)
{
    std::vector<float> numbers_list = Tensor::randn(shape_list).tolist();
    for (float &number_val : numbers_list)
        number_val *= scale_val;
    return numbers_list;
}

/// Gives a Shape as the sizes PyTorch takes: {2, 4} as int64_t.
static std::vector<int64_t> sizes_of(const Shape &shape_list)
{
    return std::vector<int64_t>(shape_list.begin(), shape_list.end());
}

/// Makes a bfloat16 tensor of `shape_list` holding `values_list` on the GPU, each value rounded to bfloat16.
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
    return torch::from_blob(floats_tensor.data(), sizes_of(tensor.shape()), on_gpu_val).clone();
}

/// A GradNormBf16 with block sums and a norm of its own, all 0.
struct NormOfOwn
{
    // A GradNormBf16 with block sums and a norm of its own: block_sums {2, GRAD_NORM_BF16_BLOCKS} (row 0 the block sums,
    // row 1 value 0 the counter), grad_norm {1}, all 0.
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BF16_BLOCKS});
    Tensor grad_norm_tensor = Tensor::zeros({1});
    GradNormBf16 grad_norm_val{block_sums_tensor};

    /// Gives the norm of `grads_tensor` (bfloat16), at this block size (the default if not given).
    float of(const Tensor &grads_tensor, int threads_per_block_val = GRAD_NORM_BF16_THREADS)
    {
        grad_norm_val.compute(grads_tensor, grad_norm_tensor, threads_per_block_val);
        return grad_norm_tensor.item();
    }
};

// ══ part 1: mine against PyTorch's and against the 4-value path ═══════════════

static void whole_model_norm_matches_pytorch(void)
{
    // Expect: the gradients of a tiny GPT-2 of 1 layer, 16 wide (16 tensors, 4464 values), in one flat bfloat16
    // tensor, as the optimizer keeps them. One compute on all 4464 values equals PyTorch's norm of the 16 separate
    // .grad tensors holding the same values as floats, within TOLERANCE (each is read as a float, squared and added in
    // float: bfloat16 changes only what is read). Two rounds, as two training steps: big gradients (norm about
    // sqrt(4464) = 67, above a max_norm of 1: clipped), then small ones (about 0.067, below it: left alone).

    // 1. the shapes, one flat bfloat16 tensor, and PyTorch's 16 separate parameters
    Tensor::manual_seed(0);
    std::vector<Shape> param_shapes_list = {
        {64, 16}, {8, 16}, {48, 16}, {16, 16}, {64, 16}, {16, 64},          // wte, wpe, qkvw, attprojw, fcw, fcprojw
        {16}, {16}, {48}, {16}, {16}, {16}, {64}, {16},                     // ln1w, ln1b, qkvb, attprojb, ln2w, ln2b, fcb, fcprojb
        {16}, {16},                                                         // lnfw, lnfb
    };
    size_t total_val = 0;                                         // 4464
    for (const Shape &shape_list : param_shapes_list)
    {
        size_t numel_val = 1;
        for (size_t size_val : shape_list)
            numel_val *= size_val;
        total_val += numel_val;
    }
    std::vector<torch::Tensor> pytorch_params_list;               // only their .grad matters here
    for (const Shape &shape_list : param_shapes_list)
        pytorch_params_list.push_back(torch::zeros(sizes_of(shape_list), torch::kCUDA).requires_grad_(true));
    NormOfOwn norm_val;

    // 2. two rounds: big gradients, then small ones
    for (int round_index_val = 1; round_index_val <= 2; round_index_val++)
    {
        bool big_val = round_index_val == 1;
        // the gradients: random, rounded to bfloat16; PyTorch's .grad the same values as floats
        Tensor grads_tensor = bf16_of({total_val}, random_values({total_val}, big_val ? 1.0f : 1e-3f));
        torch::Tensor all_tensor = pytorch_copy_of(grads_tensor);
        int64_t start_val = 0;
        for (size_t param_index_val = 0; param_index_val < param_shapes_list.size(); param_index_val++)
        {
            std::vector<int64_t> sizes_list = sizes_of(param_shapes_list[param_index_val]);
            int64_t numel_val = 1;
            for (int64_t size_val : sizes_list)
                numel_val *= size_val;
            pytorch_params_list[param_index_val].mutable_grad() = all_tensor.narrow(0, start_val, numel_val).reshape(sizes_list).clone();
            start_val += numel_val;
        }
        // my norm, and PyTorch's: clip_grad_norm_ with max_norm 1e9 only measures (it multiplies each gradient by 1)
        float my_norm_val = norm_val.of(grads_tensor);
        double pytorch_norm_val = torch::nn::utils::clip_grad_norm_(pytorch_params_list, 1e9);
        ASSERT_NEAR(my_norm_val, pytorch_norm_val, TOLERANCE * pytorch_norm_val);
        ASSERT_EQ(my_norm_val > 1.0f, big_val);
    }
}

static void norm_with_a_head_and_a_tail_matches_pytorch(void)
{
    // Expect: 10,016 random bfloat16 values, of which the compute is given a view of 10,007 starting at value 3 (6
    // bytes past a 16-byte boundary): a head of 5 values before the next boundary, 1250 whole loads, and a tail of 2.
    // The norm equals PyTorch's norm of the same 10,007 values as floats within TOLERANCE.
    // 1. random values; my norm of the view
    Tensor::manual_seed(0);
    Tensor all_tensor = bf16_of({10016}, random_values({10016}));
    Tensor view_tensor = all_tensor.narrow(0, 3, 10007);
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(view_tensor);
    // 2. PyTorch's, of the same values as floats
    double pytorch_norm_val = pytorch_copy_of(view_tensor).norm().item<double>();
    // 3. check
    ASSERT_NEAR(my_norm_val, pytorch_norm_val, TOLERANCE * pytorch_norm_val);
}

static void within_tolerance_of_the_4_value_path(void)
{
    // Expect: 1,000,003 random bfloat16 values. The norm here (8 values a thread) and fp32/grad_norm.cuh's GradNorm
    // on bfloat16 (4 values a thread) are within TOLERANCE: the same squares, added in another order.
    // 1. random values; both norms
    Tensor::manual_seed(0);
    Tensor grads_tensor = Tensor::randn({1000003}).to(DType::BFloat16);
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(grads_tensor);
    Tensor four_block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    Tensor four_norm_tensor = Tensor::zeros({1});
    GradNorm(four_block_sums_tensor).compute(grads_tensor, four_norm_tensor);
    // 2. check
    ASSERT_NEAR(my_norm_val, four_norm_tensor.item(), TOLERANCE * four_norm_tensor.item());
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════

static void computes_the_norm_of_all_gradients_together(void)
{
    // Expect: a weight's gradient {1, 2, 3, 4} and a bias's {5, 6}, one after the other in one flat bfloat16 tensor,
    // have the norm of all six as one long list: sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) = 9.539. (C#:
    // ClipReturnsTheNormOfAllGradientsTogether.) 6 values: no whole load, a tail of 6.
    // 1. the norm
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(bf16_of({6}, {1, 2, 3, 4, 5, 6}));
    // 2. check
    ASSERT_NEAR(my_norm_val, sqrt(91.0), 1e-5);
}

static void a_3_and_minus_4_give_5(void)
{
    // Expect: {3, -4} has norm sqrt(9 + 16) = sqrt(25) = 5, exactly: a minus sign makes no difference once squared.
    NormOfOwn norm_val;
    ASSERT_EQ(norm_val.of(bf16_of({2}, {3.0f, -4.0f})), 5.0f);
}

static void zeros_give_0(void)
{
    // Expect: 1000 zeros have norm 0, not NaN: sqrt(0) = 0. The norm tensor starts at 7, so a 0 shows the compute
    // really wrote it.
    NormOfOwn norm_val;
    norm_val.grad_norm_tensor.fill_(7);
    ASSERT_EQ(norm_val.of(Tensor::zeros({1000}, DType::BFloat16)), 0.0f);
}

static void each_block_adds_its_share_of_the_values(void)
{
    // Expect: 4,194,304 ones = 524,288 loads of 8 = exactly one round of the 512 x 1024 = 524,288 threads, so each
    // thread reads 1 load: 8 squares (the 4-value path's threads read 2 loads of 4). Each block: 8 x 1024 = 8192, so
    // all 512 block sums are 8192. They add up to 512 x 8192 = 4,194,304, and the norm is sqrt(4,194,304) = 2048,
    // exactly.
    // 1. the norm of ones
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(Tensor::ones({4194304}, DType::BFloat16));
    // 2. check every block sum, and the norm
    ASSERT_EQ(norm_val.block_sums_tensor[0].tolist(), std::vector<float>(GRAD_NORM_BF16_BLOCKS, 8192.0f));
    ASSERT_EQ(my_norm_val, 2048.0f);
}

static void a_last_round_that_is_not_full_still_counts(void)
{
    // Expect: 4,840,000 values of 0.5 = 605,000 loads. 605,000 = 524,288 + 80,712: one full round, then a part round,
    // so threads 0 .. 80,711 read 2 loads and the rest 1. Each load is 8 squares of 0.25 = 2:
    //
    //     block 0     threads 0 .. 1023, all with 2:                       1024 x 2 x 2 = 4096
    //     block 78    threads 79,872 .. 80,895: 840 with 2, 184 with 1:     (840 x 2 + 184) x 2 = 3728
    //     block 511   threads 523,264 .. 524,287, all with 1:               1024 x 1 x 2 = 2048
    //
    // All of them: 4,840,000 x 0.25 = 1,210,000, so the norm is sqrt(1,210,000) = 1100, exactly.
    // 1. the norm of 0.5s
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(Tensor::zeros({4840000}, DType::BFloat16).fill_(0.5));
    // 2. check three block sums, and the norm
    ASSERT_EQ(norm_val.block_sums_tensor[0][0].item(), 4096.0f);
    ASSERT_EQ(norm_val.block_sums_tensor[0][78].item(), 3728.0f);
    ASSERT_EQ(norm_val.block_sums_tensor[0][511].item(), 2048.0f);
    ASSERT_EQ(my_norm_val, 1100.0f);
}

static void values_after_the_last_whole_load_still_count(void)
{
    // Expect: 15 values, eight 1s then seven 2s: one whole load (the 1s) and a tail of 7 (the 2s), which threads 0 .. 6
    // read one each. Block 0 has them all: 8 x 1 + 7 x 4 = 36, and the norm is sqrt(36) = 6, exactly. Without the tail
    // it would be sqrt(8) = 2.83.
    // 1. the norm
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(bf16_of({15}, {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2}));
    // 2. check
    ASSERT_EQ(norm_val.block_sums_tensor[0][0].item(), 36.0f);
    ASSERT_EQ(my_norm_val, 6.0f);
}

static void values_before_the_first_16_byte_boundary_still_count(void)
{
    // Expect: 24 bfloat16 values starting on a 16-byte boundary (a fresh allocation), and the compute given a view of
    // 18 from value 1, as a GPU's slice can start 2 bytes past one:
    //
    //     value      0     1 .. 7        8 .. 15       16  17  18     19 .. 23
    //                100   1 (the head)  2 (a load)    3   1   0      100 (not in the view)
    //
    // The head's 7 values (up to the boundary at value 8) are read one each by threads 0 .. 6, the load by thread 0,
    // the tail's 3 by threads 0 .. 2: 7 x 1 + 8 x 4 + 9 + 1 + 0 = 49, all in block 0, and the norm is sqrt(49) = 7,
    // exactly. Without the head it would be sqrt(42) = 6.48; reading past the view, far more.
    // 1. the values; the norm of the view
    std::vector<float> values_list(24, 100.0f);
    for (int value_index_val = 1; value_index_val <= 7; value_index_val++)
        values_list[value_index_val] = 1.0f;
    for (int value_index_val = 8; value_index_val <= 15; value_index_val++)
        values_list[value_index_val] = 2.0f;
    values_list[16] = 3.0f, values_list[17] = 1.0f, values_list[18] = 0.0f;
    Tensor all_tensor = bf16_of({24}, values_list);
    NormOfOwn norm_val;
    float my_norm_val = norm_val.of(all_tensor.narrow(0, 1, 18));
    // 2. check
    ASSERT_EQ(norm_val.block_sums_tensor[0][0].item(), 49.0f);
    ASSERT_EQ(my_norm_val, 7.0f);
}

static void the_same_values_give_the_same_norm_every_time(void)
{
    // Expect: 1,000,000 random values, 5 computes: all 5 norms are the same, to the last bit. The last block adds the
    // block sums in the same order every time; atomicAdd into the norm would not. Each compute also puts the counter
    // back to 0, ready for the next.
    // 1. the first norm
    Tensor::manual_seed(0);
    Tensor gradient_tensor = Tensor::randn({1000000}).to(DType::BFloat16);
    NormOfOwn norm_val;
    float first_norm_val = norm_val.of(gradient_tensor);
    // 2. four more, each the same
    for (int run_index_val = 2; run_index_val <= 5; run_index_val++)
        ASSERT_EQ(norm_val.of(gradient_tensor), first_norm_val);
    // 3. the counter is back to 0
    ASSERT_EQ(norm_val.block_sums_tensor[1][0].item(), 0.0f);
}

static void only_the_values_in_the_view_count(void)
{
    // Expect: of {3, 4, 100, 100}, the compute is given a view of the first 2 (narrow(0, 0, 2): the address of the 3,
    // numel 2): norm 5. A kernel that read past the view would take a 100 too.
    NormOfOwn norm_val;
    ASSERT_EQ(norm_val.of(bf16_of({4}, {3.0f, 4.0f, 100.0f, 100.0f}).narrow(0, 0, 2)), 5.0f);
}

static void changes_nothing_but_the_norm(void)
{
    // Expect: the values 1 2 3 4 then 7 7 7 7 in bfloat16, the compute given the first 4; one float buffer, every
    // value 7 at first:
    //
    //     [ block sums: 512 x 7, then row 1: 0 (blocks done) and 511 x 7 | grad_norm: 7 | after: 7 7 7 7 ]
    //
    // After the compute, the norm is sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.477 (not more: the 7s right after the values
    // are not read). Block sum 0 is 30 (block 0 has all 4 values), the other 511 are 0. The counter is back to 0. The
    // values, the 4 after them and the 4 floats after the norm are as they were. (The counter is set to 0 after the
    // fill: it must start at 0.)
    // 1. the values; the buffer, all 7, the counter 0
    Tensor values_tensor = bf16_of({8}, {1, 2, 3, 4, 7, 7, 7, 7});
    TensorBuffer buffer_val;
    int block_sums_index_val = buffer_val.add({2, GRAD_NORM_BF16_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index_val = buffer_val.add({1});
    int after_index_val = buffer_val.add({4});
    buffer_val.allocate();
    buffer_val.flat().fill_(7);
    buffer_val.tensor(block_sums_index_val)[1][0].zero_();
    // 2. the norm
    GradNormBf16 grad_norm_val(buffer_val.tensor(block_sums_index_val));
    grad_norm_val.compute(values_tensor.narrow(0, 0, 4), buffer_val.tensor(grad_norm_index_val));
    // 3. check the norm, the block sums, the counter, and that nothing else changed
    ASSERT_NEAR(buffer_val.tensor(grad_norm_index_val).item(), sqrt(30.0), 1e-5);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[0][0].item(), 30.0f);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[0][GRAD_NORM_BF16_BLOCKS - 1].item(), 0.0f);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[1][0].item(), 0.0f);
    ASSERT_EQ(values_tensor.to(DType::Float32).tolist(), std::vector<float>({1, 2, 3, 4, 7, 7, 7, 7}));
    ASSERT_EQ(buffer_val.tensor(after_index_val).tolist(), std::vector<float>(4, 7.0f));
}

static void any_block_size_gives_the_same_norm_within_tolerance(void)
{
    // Expect: 1,000,003 random bfloat16 values (whole loads and 3 left over). compute called with threads_per_block
    // 32, 128, 512 and 1024 (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK) gives the default's norm
    // (GRAD_NORM_BF16_THREADS) within TOLERANCE, not to the last bit: another block size gives each thread other
    // values and each block other threads, so the squares are added in another order. The same block size twice gives
    // the same bits, and every compute puts the counter back to 0.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                    HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random gradients; the default's norm
    Tensor::manual_seed(0);
    Tensor gradient_tensor = Tensor::randn({1000003}).to(DType::BFloat16);
    NormOfOwn norm_val;
    float usual_norm_val = norm_val.of(gradient_tensor);

    // 2. each block size: within TOLERANCE of the default's, the same bits twice, the counter back to 0
    for (int threads_per_block_val : block_sizes_list)
    {
        float block_size_norm_val = norm_val.of(gradient_tensor, threads_per_block_val);
        ASSERT_NEAR(block_size_norm_val, usual_norm_val, TOLERANCE * usual_norm_val);
        ASSERT_EQ(norm_val.of(gradient_tensor, threads_per_block_val), block_size_norm_val);
        ASSERT_EQ(norm_val.block_sums_tensor[1][0].item(), 0.0f);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    whole_model_norm_matches_pytorch();
    norm_with_a_head_and_a_tail_matches_pytorch();
    within_tolerance_of_the_4_value_path();

    computes_the_norm_of_all_gradients_together();
    a_3_and_minus_4_give_5();
    zeros_give_0();
    each_block_adds_its_share_of_the_values();
    a_last_round_that_is_not_full_still_counts();
    values_after_the_last_whole_load_still_count();
    values_before_the_first_16_byte_boundary_still_count();
    the_same_values_give_the_same_norm_every_time();
    only_the_values_in_the_view_count();
    changes_nothing_but_the_norm();
    any_block_size_gives_the_same_norm_within_tolerance();

    printf("\nall tests passed\n");
    return 0;
}
