// test_grad_norm.cu — tests for llmc/kernels/grad_norm.cuh (GradNorm). Every check
// prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks the norm against PyTorch's, from torch::nn::utils::clip_grad_norm_ (it
// returns the norm of all the gradients it was given); part 2 checks numbers worked out by
// hand, and how the work is shared out between the blocks.
#include <torch/nn/utils/clip_grad.h> // torch::nn::utils::clip_grad_norm_; libtorch before any CUDA header
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/tensor_buffer.cuh"  // TensorBuffer: the gradients in one allocation, as the optimizer keeps them
#include "../llmc/kernels/grad_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the norm

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn, times `scale`, as a CPU list: the same numbers
/// for both sides of a test.
static std::vector<float> random_values(const Shape &shape, float scale = 1.0f)
{
    std::vector<float> numbers = Tensor::randn(shape).tolist();
    for (float &number : numbers)
        number *= scale;
    return numbers;
}

/// Gives a Shape as the sizes PyTorch takes: {2, 4} as int64_t.
static std::vector<int64_t> sizes_of(const Shape &shape)
{
    return std::vector<int64_t>(shape.begin(), shape.end());
}

/// Makes a new PyTorch tensor on the GPU from values in row order: the tests' CPU lists.
static torch::Tensor to_gpu(const std::vector<float> &row_values, const Shape &shape)
{
    return torch::tensor(row_values).reshape(sizes_of(shape)).to(torch::kCUDA);
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU: for
/// values too many to pass through a CPU list quickly.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    return torch::from_blob(tensor.data(), sizes_of(tensor.shape()), on_gpu).clone();
}

/// Measures how long `run` takes on the GPU, in milliseconds: a mark on the GPU's queue
/// before it and one after; the GPU writes down the time as it reaches each.
template <typename Run>
static float gpu_milliseconds(Run run)
{
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    run();
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end)); // wait until the GPU has passed the second mark
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    return milliseconds;
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
    Tensor::manual_seed(0);
    std::vector<Shape> param_shapes = {
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
    int num_params = (int)param_shapes.size();
    // one flat buffer: [ grads: 4464 | block sums: 2 x 512 | grad_norm: 1 ]
    TensorBuffer buffer;
    int grads_group = buffer.add_many(param_shapes);
    int block_sums_index = buffer.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index = buffer.add({1});
    buffer.allocate();
    const Tensor &grads = buffer.group(grads_group); // every gradient, one flat tensor
    const Tensor &grad_norm = buffer.tensor(grad_norm_index);
    GradNorm grad_norm_kernel(buffer.tensor(block_sums_index));
    // PyTorch's: 16 separate parameters; only their .grad matters here
    std::vector<torch::Tensor> pytorch_params;
    for (const Shape &shape : param_shapes)
        pytorch_params.push_back(torch::zeros(sizes_of(shape), torch::kCUDA).requires_grad_(true));

    for (int round = 1; round <= 2; round++)
    {
        bool big = round == 1;
        // 1. the gradients: random, the same on both sides
        for (int k = 0; k < num_params; k++)
        {
            std::vector<float> grad_values = random_values(param_shapes[k], big ? 1.0f : 1e-3f);
            buffer.tensor(grads_group, k).copy_from_cpu(grad_values.data());
            pytorch_params[k].mutable_grad() = to_gpu(grad_values, param_shapes[k]);
        }
        // 2. my norm: one compute on the flat gradients' view, all 4464 values
        grad_norm_kernel.compute(grads, grad_norm);
        // 3. PyTorch's: clip_grad_norm_ returns the norm of all 16 .grad together. max_norm 1e9
        //    is far above any norm here, so it only measures (it multiplies each gradient by 1).
        double pytorch_norm = torch::nn::utils::clip_grad_norm_(pytorch_params, 1e9);
        // 4. compare: within 1e-5 of the norm, not exactly, as the two add the squares in a
        //    different order; and the norm is on the side of max_norm 1 this round is meant for
        ASSERT_NEAR(grad_norm.item(), pytorch_norm, TOLERANCE * pytorch_norm);
        ASSERT_EQ(grad_norm.item() > 1.0f, big);
    }
}

static void time_20_norms_on_gpt2_small_size(void)
{
    // Prints the times on as many values as GPT-2 small has: 162,030 x 768 + 768 =
    // 124,439,808. Each norm reads every value once: 124,439,808 x 4 bytes = 498 MB, so at a
    // T4's 320 GB/s, at least 1.56 ms. PyTorch's side is what clip_grad_norm_ does before it
    // clips: the norm of each tensor, then the norm of those norms (timed without the clip,
    // which would also multiply every gradient by its factor). Checks at the end that both
    // measured the same norm.
    Tensor::manual_seed(0);
    std::vector<Shape> param_shapes = {{162030, 768}, {768}};
    // one flat buffer: [ grads: 124,439,808 | block sums: 2 x 512 | grad_norm: 1 ]
    TensorBuffer buffer;
    int grads_group = buffer.add_many(param_shapes);
    int block_sums_index = buffer.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index = buffer.add({1});
    buffer.allocate();
    for (int k = 0; k < 2; k++)
        buffer.tensor(grads_group, k).normal_(0.0f, 1.0f); // made on the CPU: a few seconds
    std::vector<torch::Tensor> pytorch_grads;
    for (int k = 0; k < 2; k++)
        pytorch_grads.push_back(pytorch_copy_of(buffer.tensor(grads_group, k)));
    GradNorm grad_norm_kernel(buffer.tensor(block_sums_index));
    const Tensor &grad_norm = buffer.tensor(grad_norm_index);
    torch::Tensor pytorch_norm;
    const Tensor &grads = buffer.group(grads_group); // every gradient, one flat view
    size_t n = grads.numel();

    // Step 1 does first-time work (CUDA loads each side's kernels), so the average is of 2..20.
    print_test_name(__func__);
    printf("    time of one norm, in ms:   step   my GradNorm   PyTorch\n");
    double my_total = 0.0, pytorch_total = 0.0;
    for (int t = 1; t <= 20; t++)
    {
        float my_ms = gpu_milliseconds([&] { grad_norm_kernel.compute(grads, grad_norm); });
        float pytorch_ms = gpu_milliseconds([&] {
            std::vector<torch::Tensor> norms;
            for (const torch::Tensor &grad : pytorch_grads)
                norms.push_back(grad.norm());
            pytorch_norm = torch::stack(norms).norm();
        });
        printf("                               %4d   %13.3f   %7.3f\n", t, my_ms, pytorch_ms);
        if (t == 1)
            continue;
        my_total += my_ms;
        pytorch_total += pytorch_ms;
    }
    double bytes = (double)n * sizeof(float);
    printf("    average of steps 2..20:           %13.3f   %7.3f   PyTorch takes %.2fx as long\n", my_total / 19,
           pytorch_total / 19, pytorch_total / my_total);
    printf("    memory read, in GB/s:             %13.0f   %7.0f\n", bytes / (my_total / 19) / 1e6,
           bytes / (pytorch_total / 19) / 1e6);
    double expected = pytorch_norm.item<double>();
    ASSERT_NEAR(grad_norm.item(), expected, TOLERANCE * expected);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════

static void computes_the_norm_of_all_gradients_together(void)
{
    // Expect: a weight's gradient {1, 2, 3, 4} and a bias's {5, 6}, one after the other in a
    // flat buffer, have the norm of all six as one long list:
    // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) = 9.539. (C#: ClipReturnsTheNormOfAllGradientsTogether.)
    // The buffer: [ weight's grad: 4 | bias's grad: 2 | block sums: 2 x 512 | grad_norm: 1 ]
    TensorBuffer buffer;
    int grads_group = buffer.add_many({{2, 2}, {2}});
    int block_sums_index = buffer.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index = buffer.add({1});
    buffer.allocate();
    std::vector<float> weight_grad = {1.0f, 2.0f, 3.0f, 4.0f}, bias_grad = {5.0f, 6.0f};
    buffer.tensor(grads_group, 0).copy_from_cpu(weight_grad.data());
    buffer.tensor(grads_group, 1).copy_from_cpu(bias_grad.data());
    const Tensor &grads = buffer.group(grads_group);
    const Tensor &grad_norm = buffer.tensor(grad_norm_index);
    GradNorm grad_norm_kernel(buffer.tensor(block_sums_index));
    grad_norm_kernel.compute(grads, grad_norm);
    ASSERT_NEAR(grad_norm.item(), sqrt(91.0), 1e-5);
}

static void a_3_and_minus_4_give_5(void)
{
    // Expect: {3, -4} has norm sqrt(9 + 16) = sqrt(25) = 5, exactly: a minus sign makes no
    // difference once squared.
    std::vector<float> values = {3.0f, -4.0f};
    Tensor gradient = Tensor::zeros({2});
    gradient.copy_from_cpu(values.data());
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(gradient, grad_norm);
    ASSERT_EQ(grad_norm.item(), 5.0f);
}

static void zeros_give_0(void)
{
    // Expect: 1000 zeros have norm 0, not NaN: sqrt(0) = 0. The norm tensor starts at 7, so
    // a 0 shows the compute really wrote it.
    Tensor gradient = Tensor::zeros({1000});
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1}).fill_(7);
    grad_norm_kernel.compute(gradient, grad_norm);
    ASSERT_EQ(grad_norm.item(), 0.0f);
}

static void each_block_adds_its_share_of_the_values(void)
{
    // Expect: 4,194,304 ones = 1,048,576 float4s = 2 rounds of the 512 x 1024 = 524,288 threads,
    // so each thread reads 2 float4s: 8 squares. Each block: 8 x 1024 = 8192, so all 512 block
    // sums are 8192. They add up to 512 x 8192 = 4,194,304, and the norm is sqrt(4,194,304) =
    // 2048, exactly.
    Tensor gradient = Tensor::ones({4194304});
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(gradient, grad_norm);
    ASSERT_EQ(block_sums[0].tolist(), std::vector<float>(512, 8192.0f));
    ASSERT_EQ(grad_norm.item(), 2048.0f);
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
    Tensor gradient = Tensor::zeros({4000000}).fill_(0.5);
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(gradient, grad_norm);
    ASSERT_EQ(block_sums[0][0].item(), 2048.0f);
    ASSERT_EQ(block_sums[0][464].item(), 1600.0f);
    ASSERT_EQ(block_sums[0][511].item(), 1024.0f);
    ASSERT_EQ(grad_norm.item(), 1000.0f);
}

static void values_after_the_last_whole_float4_still_count(void)
{
    // Expect: 7 values {1, 1, 1, 1, 2, 2, 2}: one whole float4 (the four 1s) and 3 left over (the
    // 2s), which threads 0, 1 and 2 read one float each. Block 0 has them all: 1 + 1 + 1 + 1 +
    // 4 + 4 + 4 = 16, and the norm is sqrt(16) = 4, exactly. Without the leftovers it would be 2.
    std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    Tensor gradient = Tensor::zeros({7});
    gradient.copy_from_cpu(values.data());
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(gradient, grad_norm);
    ASSERT_EQ(block_sums[0][0].item(), 16.0f);
    ASSERT_EQ(grad_norm.item(), 4.0f);
}

static void the_same_values_give_the_same_norm_every_time(void)
{
    // Expect: 1,000,000 random values, 5 computes: all 5 norms are the same, to the last bit.
    // The last block adds the 512 block sums in the same order every time; atomicAdd into the
    // norm would not. Each compute also puts the counter back to 0, ready for the next.
    Tensor::manual_seed(0);
    Tensor gradient = Tensor::randn({1000000});
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(gradient, grad_norm);
    float first_norm = grad_norm.item();
    for (int run = 2; run <= 5; run++)
    {
        grad_norm_kernel.compute(gradient, grad_norm);
        ASSERT_EQ(grad_norm.item(), first_norm);
    }
    ASSERT_EQ(block_sums[1][0].item(), 0.0f);
}

static void only_the_values_in_the_view_count(void)
{
    // Expect: of {3, 4, 100, 100}, the compute is given a view of the first 2 (narrow(0, 0, 2):
    // the address of the 3, numel 2): norm 5. A kernel that read past the view would take a 100 too.
    std::vector<float> values = {3.0f, 4.0f, 100.0f, 100.0f};
    Tensor all_values = Tensor::zeros({4});
    all_values.copy_from_cpu(values.data());
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}); // row 0: the 512 block sums; row 1, value 0: blocks done
    GradNorm grad_norm_kernel(block_sums);
    Tensor grad_norm = Tensor::zeros({1});
    grad_norm_kernel.compute(all_values.narrow(0, 0, 2), grad_norm);
    ASSERT_EQ(grad_norm.item(), 5.0f);
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
    TensorBuffer buffer;
    int values_index = buffer.add({4});
    int block_sums_index = buffer.add({2, GRAD_NORM_BLOCKS});  // row 0: block sums; row 1, value 0: blocks done
    int grad_norm_index = buffer.add({1});
    int after_index = buffer.add({4});
    buffer.allocate();
    buffer.flat().fill_(7);
    buffer.tensor(block_sums_index)[1][0].zero_();
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    buffer.tensor(values_index).copy_from_cpu(values.data());
    GradNorm grad_norm_kernel(buffer.tensor(block_sums_index));
    grad_norm_kernel.compute(buffer.tensor(values_index), buffer.tensor(grad_norm_index));
    ASSERT_NEAR(buffer.tensor(grad_norm_index).item(), sqrt(30.0), 1e-5);
    ASSERT_EQ(buffer.tensor(block_sums_index)[0][0].item(), 30.0f);
    ASSERT_EQ(buffer.tensor(block_sums_index)[0][511].item(), 0.0f);
    ASSERT_EQ(buffer.tensor(block_sums_index)[1][0].item(), 0.0f);
    ASSERT_EQ(buffer.tensor(values_index).tolist(), values);
    ASSERT_EQ(buffer.tensor(after_index).tolist(), std::vector<float>(4, 7.0f));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    whole_model_norm_matches_pytorch();
    time_20_norms_on_gpt2_small_size();

    computes_the_norm_of_all_gradients_together();
    a_3_and_minus_4_give_5();
    zeros_give_0();
    each_block_adds_its_share_of_the_values();
    a_last_round_that_is_not_full_still_counts();
    values_after_the_last_whole_float4_still_count();
    the_same_values_give_the_same_norm_every_time();
    only_the_values_in_the_view_count();
    changes_nothing_but_the_norm();

    printf("\nall tests passed\n");
    return 0;
}
