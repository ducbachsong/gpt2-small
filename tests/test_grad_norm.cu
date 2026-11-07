// test_grad_norm.cu — tests for llmc/kernels/grad_norm.cuh (GradNorm). Every check
// prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks the norm against PyTorch's, from torch::nn::utils::clip_grad_norm_ (it
// returns the norm of all the gradients it was given); part 2 checks numbers worked out by
// hand, how the work is shared out between the blocks, and any block size. The timings are in
// tunes/tune_grad_norm.cu.
#include <ATen/ops/_foreach_norm.h>   // at::_foreach_norm: what Python's clip_grad_norm_ uses
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the gradients in one allocation, as the optimizer keeps them
#include "../llmc/kernels/grad_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the norm

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn, times `scale_val`, as a CPU list: the same numbers
/// for both sides of a test.
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

/// Makes a new PyTorch tensor on the GPU from values in row order: the tests' CPU lists.
static torch::Tensor to_gpu(const std::vector<float> &row_values_list, const Shape &shape_list)
{
    return torch::tensor(row_values_list).reshape(sizes_of(shape_list)).to(torch::kCUDA);
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU: for
/// values too many to pass through a CPU list quickly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    return torch::from_blob(tensor.data(), sizes_of(tensor.shape()), on_gpu_val).clone();
}

/// Makes a tensor of `shape_list` holding `values_list` on the GPU.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

// ══ part 1: my GradNorm against PyTorch's ═══════════════════════════════════

static void whole_model_norm_matches_pytorch(void)
{
    // Expect: the gradients of a tiny GPT-2 of 1 layer, 16 wide (16 tensors, 4464 values), in
    // one flat TensorBuffer, as the optimizer keeps them. One compute on the gradients' view
    // over all 4464 values equals PyTorch's norm of the 16 separate .grad tensors. Two rounds, as two training
    // steps: big gradients (norm about sqrt(4464) = 67, above a max_norm of 1: clipped), then
    // small ones (about 0.067, below it: left alone), so both sides of clipping are covered.
    // (C#: BackwardThroughTheModelWritesIntoTheFlatGradients checked the same, norm of the
    // flat buffer against norm of each .grad, with gradients from backward(); the C trainer
    // has no backward yet, so the gradients here are random.)

    // 1. the shapes, one flat buffer, and PyTorch's 16 separate parameters
    Tensor::manual_seed(0);
    std::vector<Shape> param_shapes_list = {
        // the 6 matrices: 4224 values
        {64, 16}, // wte: vocab 64
        {8, 16},  // wpe: sequence 8
        {48, 16}, // qkvw
        {16, 16}, // attprojw
        {64, 16}, // fcw
        {16, 64}, // fcprojw
        // the 10 biases and LayerNorm tensors: 240 values
        {16}, {16}, {48}, {16}, {16}, {16}, {64}, {16}, // ln1w, ln1b, qkvb, attprojb, ln2w, ln2b, fcb, fcprojb
        {16}, {16},                                     // lnfw, lnfb
    };
    int num_params_val = (int)param_shapes_list.size();
    // [ grads: 4464 | block sums: 2 x GRAD_NORM_BLOCKS | grad_norm: 1 ]
    TensorBuffer buffer_val;
    int grads_group_val = buffer_val.add_many(param_shapes_list);
    int block_sums_index_val = buffer_val.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index_val = buffer_val.add({1});
    buffer_val.allocate();
    const Tensor &grads_tensor = buffer_val.group(grads_group_val); // every gradient, one flat tensor
    const Tensor &grad_norm_tensor = buffer_val.tensor(grad_norm_index_val);
    GradNorm grad_norm_kernel_val(buffer_val.tensor(block_sums_index_val));
    std::vector<torch::Tensor> pytorch_params_list;               // only their .grad matters here
    for (const Shape &shape_list : param_shapes_list)
        pytorch_params_list.push_back(torch::zeros(sizes_of(shape_list), torch::kCUDA).requires_grad_(true));

    // 2. two rounds: big gradients, then small ones
    for (int round_val = 1; round_val <= 2; round_val++)
    {
        bool big_val = round_val == 1;
        // the gradients: random, the same on both sides
        for (int param_index_val = 0; param_index_val < num_params_val; param_index_val++)
        {
            std::vector<float> grad_values_list = random_values(param_shapes_list[param_index_val], big_val ? 1.0f : 1e-3f);
            buffer_val.tensor(grads_group_val, param_index_val).copy_from_cpu(grad_values_list.data());
            pytorch_params_list[param_index_val].mutable_grad() = to_gpu(grad_values_list, param_shapes_list[param_index_val]);
        }
        // my norm: one compute on the flat gradients' view, all 4464 values
        grad_norm_kernel_val.compute(grads_tensor, grad_norm_tensor);
        // PyTorch's: clip_grad_norm_ returns the norm of all 16 .grad together. max_norm 1e9 is far above any
        // norm here, so it only measures (it multiplies each gradient by 1).
        double pytorch_norm_val = torch::nn::utils::clip_grad_norm_(pytorch_params_list, 1e9);
        // compare: within 1e-5 of the norm, not exactly, as the two add the squares in a different order; and
        // the norm is on the side of max_norm 1 this round is meant for
        ASSERT_NEAR(grad_norm_tensor.item(), pytorch_norm_val, TOLERANCE * pytorch_norm_val);
        ASSERT_EQ(grad_norm_tensor.item() > 1.0f, big_val);
    }
}

static void norm_of_bfloat16_gradients_matches_pytorch(void)
{
    // Expect: 10,007 random gradients rounded to bfloat16 (2501 whole float4s and 3 left over, which the
    // threads 0 .. 2 add on their own). Each is read as a float and squared and added in float, so the norm is
    // PyTorch's norm of the same values as floats within 1e-5: bfloat16 changes only what is read. block_sums
    // and grad_norm stay float.
    // 1. random gradients in bfloat16; my norm
    Tensor::manual_seed(0);
    Tensor grads_tensor = tensor_of({10007}, random_values({10007})).to(DType::BFloat16);
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(grads_tensor, grad_norm_tensor);
    // 2. PyTorch's, of the same values as floats
    double pytorch_norm_val = pytorch_copy_of(grads_tensor.to(DType::Float32)).norm().item<double>();
    // 3. check
    ASSERT_NEAR(grad_norm_tensor.item(), pytorch_norm_val, TOLERANCE * pytorch_norm_val);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════

static void computes_the_norm_of_all_gradients_together(void)
{
    // Expect: a weight's gradient {1, 2, 3, 4} and a bias's {5, 6}, one after the other in a
    // flat buffer, have the norm of all six as one long list:
    // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) = 9.539. (C#: ClipReturnsTheNormOfAllGradientsTogether.)
    // The buffer: [ weight's grad: 4 | bias's grad: 2 | block sums: 2 x GRAD_NORM_BLOCKS | grad_norm: 1 ]
    // 1. the buffer, with the two gradients
    TensorBuffer buffer_val;
    int grads_group_val = buffer_val.add_many({{2, 2}, {2}});
    int block_sums_index_val = buffer_val.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index_val = buffer_val.add({1});
    buffer_val.allocate();
    std::vector<float> weight_grad_list = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> bias_grad_list = {5.0f, 6.0f};
    buffer_val.tensor(grads_group_val, 0).copy_from_cpu(weight_grad_list.data());
    buffer_val.tensor(grads_group_val, 1).copy_from_cpu(bias_grad_list.data());
    // 2. the norm
    const Tensor &grads_tensor = buffer_val.group(grads_group_val);
    const Tensor &grad_norm_tensor = buffer_val.tensor(grad_norm_index_val);
    GradNorm grad_norm_kernel_val(buffer_val.tensor(block_sums_index_val));
    grad_norm_kernel_val.compute(grads_tensor, grad_norm_tensor);
    // 3. check
    ASSERT_NEAR(grad_norm_tensor.item(), sqrt(91.0), 1e-5);
}

static void a_3_and_minus_4_give_5(void)
{
    // Expect: {3, -4} has norm sqrt(9 + 16) = sqrt(25) = 5, exactly: a minus sign makes no
    // difference once squared.
    // 1. the norm
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(tensor_of({2}, {3.0f, -4.0f}), grad_norm_tensor);
    // 2. check
    ASSERT_EQ(grad_norm_tensor.item(), 5.0f);
}

static void zeros_give_0(void)
{
    // Expect: 1000 zeros have norm 0, not NaN: sqrt(0) = 0. The norm tensor starts at 7, so
    // a 0 shows the compute really wrote it.
    // 1. the norm of zeros, into a 7
    Tensor gradient_tensor = Tensor::zeros({1000});
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1}).fill_(7);
    grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
    // 2. check
    ASSERT_EQ(grad_norm_tensor.item(), 0.0f);
}

static void each_block_adds_its_share_of_the_values(void)
{
    // Expect: 4,194,304 ones = 1,048,576 float4s = 2 rounds of the 512 x 1024 = 524,288 threads,
    // so each thread reads 2 float4s: 8 squares. Each block: 8 x 1024 = 8192, so all 512 block
    // sums are 8192. They add up to 512 x 8192 = 4,194,304, and the norm is sqrt(4,194,304) =
    // 2048, exactly.
    // 1. the norm of ones
    Tensor gradient_tensor = Tensor::ones({4194304});
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
    // 2. check every block sum, and the norm
    ASSERT_EQ(block_sums_tensor[0].tolist(), std::vector<float>(GRAD_NORM_BLOCKS, 8192.0f));
    ASSERT_EQ(grad_norm_tensor.item(), 2048.0f);
}

static void a_last_round_that_is_not_full_still_counts(void)
{
    // Expect: 4,000,000 values of 0.5 = 1,000,000 float4s. 1,000,000 = 524,288 + 475,712: one
    // full round, then a part round, so threads 0 .. 475,711 read 2 float4s and the rest 1. Each
    // float4 is 4 squares of 0.25 = 1:
    //
    //     block 0     threads 0 .. 1023, all with 2:                       1024 x 2 = 2048
    //     block 464   threads 475,136 .. 476,159: 576 with 2, 448 with 1:   576 x 2 + 448 = 1600
    //     block 511   threads 523,264 .. 524,287, all with 1:               1024 x 1 = 1024
    //
    // All of them: 4,000,000 x 0.25 = 1,000,000, so the norm is sqrt(1,000,000) = 1000, exactly.
    // 1. the norm of 0.5s
    Tensor gradient_tensor = Tensor::zeros({4000000}).fill_(0.5);
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
    // 2. check three block sums, and the norm
    ASSERT_EQ(block_sums_tensor[0][0].item(), 2048.0f);
    ASSERT_EQ(block_sums_tensor[0][464].item(), 1600.0f);
    ASSERT_EQ(block_sums_tensor[0][511].item(), 1024.0f);
    ASSERT_EQ(grad_norm_tensor.item(), 1000.0f);
}

static void values_after_the_last_whole_float4_still_count(void)
{
    // Expect: 7 values {1, 1, 1, 1, 2, 2, 2}: one whole float4 (the four 1s) and 3 left over (the
    // 2s), which threads 0, 1 and 2 read one float each. Block 0 has them all: 1 + 1 + 1 + 1 +
    // 4 + 4 + 4 = 16, and the norm is sqrt(16) = 4, exactly. Without the leftovers it would be 2.
    // 1. the norm
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(tensor_of({7}, {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}), grad_norm_tensor);
    // 2. check
    ASSERT_EQ(block_sums_tensor[0][0].item(), 16.0f);
    ASSERT_EQ(grad_norm_tensor.item(), 4.0f);
}

static void the_same_values_give_the_same_norm_every_time(void)
{
    // Expect: 1,000,000 random values, 5 computes: all 5 norms are the same, to the last bit.
    // The last block adds the block sums in the same order every time; atomicAdd into the
    // norm would not. Each compute also puts the counter back to 0, ready for the next.
    // 1. the first norm
    Tensor::manual_seed(0);
    Tensor gradient_tensor = Tensor::randn({1000000});
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
    float first_norm_val = grad_norm_tensor.item();
    // 2. four more, each the same
    for (int run_val = 2; run_val <= 5; run_val++)
    {
        grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
        ASSERT_EQ(grad_norm_tensor.item(), first_norm_val);
    }
    // 3. the counter is back to 0
    ASSERT_EQ(block_sums_tensor[1][0].item(), 0.0f);
}

static void only_the_values_in_the_view_count(void)
{
    // Expect: of {3, 4, 100, 100}, the compute is given a view of the first 2 (narrow(0, 0, 2):
    // the address of the 3, numel 2): norm 5. A kernel that read past the view would take a 100 too.
    // 1. the norm of a view
    Tensor all_values_tensor = tensor_of({4}, {3.0f, 4.0f, 100.0f, 100.0f});
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel_val(block_sums_tensor);
    Tensor grad_norm_tensor = Tensor::zeros({1});
    grad_norm_kernel_val.compute(all_values_tensor.narrow(0, 0, 2), grad_norm_tensor);
    // 2. check
    ASSERT_EQ(grad_norm_tensor.item(), 5.0f);
}

static void changes_nothing_but_the_norm(void)
{
    // Expect: one flat buffer, every value 7 at first:
    //
    //     [ values: 1 2 3 4 | block sums: 512 x 7, then row 1: 0 (blocks done) and 511 x 7 | grad_norm: 7 | after: 7 7 7 7 ]
    //
    // After the compute, the norm is sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.477 (not sqrt(30 +
    // 49): the 7 right after the values is not read). Block sum 0 is 30 (block 0 has all 4
    // values), the other 511 are 0. The counter is back to 0. The values and the 4 after the
    // norm are as they were. (The counter is set to 0 after the fill: it must start at 0.)
    // 1. the buffer, all 7, the counter 0, the values 1 2 3 4
    TensorBuffer buffer_val;
    int values_index_val = buffer_val.add({4});
    int block_sums_index_val = buffer_val.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index_val = buffer_val.add({1});
    int after_index_val = buffer_val.add({4});
    buffer_val.allocate();
    buffer_val.flat().fill_(7);
    buffer_val.tensor(block_sums_index_val)[1][0].zero_();
    std::vector<float> values_list = {1.0f, 2.0f, 3.0f, 4.0f};
    buffer_val.tensor(values_index_val).copy_from_cpu(values_list.data());
    // 2. the norm
    GradNorm grad_norm_kernel_val(buffer_val.tensor(block_sums_index_val));
    grad_norm_kernel_val.compute(buffer_val.tensor(values_index_val), buffer_val.tensor(grad_norm_index_val));
    // 3. check the norm, the block sums, the counter, and that nothing else changed
    ASSERT_NEAR(buffer_val.tensor(grad_norm_index_val).item(), sqrt(30.0), 1e-5);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[0][0].item(), 30.0f);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[0][GRAD_NORM_BLOCKS - 1].item(), 0.0f);
    ASSERT_EQ(buffer_val.tensor(block_sums_index_val)[1][0].item(), 0.0f);
    ASSERT_EQ(buffer_val.tensor(values_index_val).tolist(), values_list);
    ASSERT_EQ(buffer_val.tensor(after_index_val).tolist(), std::vector<float>(4, 7.0f));
}

static void any_block_size_gives_the_same_norm_within_tolerance(void)
{
    // Expect: 1,000,003 random values (whole float4s and 3 left over), in float and in bfloat16. compute
    // called with threads_per_block 32, 128, 512 and 1024 (one warp up to MAX_THREADS_PER_BLOCK) gives the
    // default's norm (GRAD_NORM_THREADS) within TOLERANCE, not to the last bit: another block size gives each
    // thread other values and each block other threads, so the squares are added in another order. The
    // same block size twice gives the same bits, and every compute puts the counter back to 0.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random gradients; the default's norm
        Tensor::manual_seed(0);
        Tensor gradient_tensor = Tensor::randn({1000003}).to(dtype_val);
        Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the block sums; row 1, value 0: blocks done
        GradNorm grad_norm_kernel_val(block_sums_tensor);
        Tensor grad_norm_tensor = Tensor::zeros({1});
        grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor);
        float usual_norm_val = grad_norm_tensor.item();

        // 2. each block size: within TOLERANCE of the default's, the same bits twice, the counter back to 0
        for (int threads_per_block_val : block_sizes_list)
        {
            grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor, threads_per_block_val);
            float norm_val = grad_norm_tensor.item();
            ASSERT_NEAR(norm_val, usual_norm_val, TOLERANCE * usual_norm_val);
            grad_norm_kernel_val.compute(gradient_tensor, grad_norm_tensor, threads_per_block_val);
            ASSERT_EQ(grad_norm_tensor.item(), norm_val);
            ASSERT_EQ(block_sums_tensor[1][0].item(), 0.0f);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    whole_model_norm_matches_pytorch();
    norm_of_bfloat16_gradients_matches_pytorch();

    computes_the_norm_of_all_gradients_together();
    a_3_and_minus_4_give_5();
    zeros_give_0();
    each_block_adds_its_share_of_the_values();
    a_last_round_that_is_not_full_still_counts();
    values_after_the_last_whole_float4_still_count();
    the_same_values_give_the_same_norm_every_time();
    only_the_values_in_the_view_count();
    changes_nothing_but_the_norm();
    any_block_size_gives_the_same_norm_within_tolerance();

    printf("\nall tests passed\n");
    return 0;
}
