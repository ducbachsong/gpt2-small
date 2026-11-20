// test_residual_bf16.cu — tests for cuda/kernels/bf16/residual_bf16.cuh (residual_bf16_forward, residual_bf16_backward):
// the residual add in bfloat16, 8 values a thread (one 16-byte load). Every check prints its test case, input,
// expected and actual value, and ok; the first FAIL stops the program.
//
// Part 1 checks both against PyTorch's + (in float on the same values, rounded to bfloat16) and against
// fp32/residual.cuh's bfloat16 path (4 values a thread); part 2 checks numbers worked out by hand, the cases of
// cuda/tests/fp32/test_residual.cu on 8 values; part 3 checks that each kernel's registers fit its block, and that any
// block size gives the same bits. Any GPU. The timings are in cuda/tests/bf16/tune_residual_bf16.cu.
#include <torch/types.h>              // torch::randn; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"          // ASSERT_EQ: every check here is exact
#include "../../kernels/fp32/residual.cuh"       // the 4-value path, to compare with
#include "../../kernels/bf16/residual_bf16.cuh"

// ── helpers ───────────────────────────────────────────────────────────────────

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

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    // Makes a bfloat16 tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
    // the GPU (Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 3.1M values), then rounded.
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values (bfloat16 turned into floats exactly).
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    Tensor floats_tensor = tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

/// Gives every value of my tensor as a CPU list of floats (bfloat16 turned into floats, exactly).
static std::vector<float> floats_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

// ══ part 1: mine against PyTorch's and against the 4-value path ═══════════════

/// Runs one forward and one backward on random values and checks them against PyTorch to the last bit.
///
///   shape_list   the tensors' shape
static void forward_and_backward_match_pytorch(const Shape &shape_list)
{
    // Runs one forward and one backward on random bfloat16 values of `shape_list`, and checks output against PyTorch's
    // float add of the same values rounded to bfloat16 by PyTorch, to the last bit (both round to the nearest, a tie to
    // the even one), and both gradients against output_grad itself: the backward only copies.
    // 1. random inputs; mine
    torch::manual_seed(0);
    Tensor residual_tensor = random_bf16_tensor(shape_list);
    Tensor branch_output_tensor = random_bf16_tensor(shape_list);
    Tensor output_grad_tensor = random_bf16_tensor(shape_list);
    Tensor output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor residual_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor branch_output_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    residual_bf16_forward(output_tensor, residual_tensor, branch_output_tensor);
    residual_bf16_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);

    // 2. PyTorch's float add of the same values
    torch::Tensor pytorch_output_tensor = pytorch_copy_of(residual_tensor) + pytorch_copy_of(branch_output_tensor);

    // 3. the same bits once rounded; the gradients output_grad itself
    ASSERT_EQ(floats_of(output_tensor), list_of(pytorch_output_tensor.to(torch::kBFloat16).to(torch::kFloat)));
    ASSERT_EQ(floats_of(residual_grad_tensor), floats_of(output_grad_tensor));
    ASSERT_EQ(floats_of(branch_output_grad_tensor), floats_of(output_grad_tensor));
}

static void matches_pytorch_on_gpt2_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens of 768, the residual stream of GPT-2 small: 12,288 values,
    // 1536 loads, 6 full blocks. output equals PyTorch's residual + branch_output rounded to bfloat16, and both
    // gradients output_grad, exactly.
    forward_and_backward_match_pytorch({2, 8, 768});
}

static void matches_pytorch_with_a_part_full_last_block(void)
{
    // Expect: {3, 7, 104}: 2184 values, 273 loads, 2 blocks of 256 threads; the last has 17 loads for 256 threads, and
    // its other 239 threads must write nothing. The same as PyTorch, exactly.
    forward_and_backward_match_pytorch({3, 7, 104});
}

static void the_same_bits_as_the_4_value_path(void)
{
    // Expect: GPT-2 small's residual stream, 4 rows of 1024 tokens of 768, random bfloat16 values. Here (8 values a
    // thread) and in fp32/residual.cuh's bfloat16 path (4 values a thread): output, residual_grad and
    // branch_output_grad the same bits, as each value is one float add of the same two numbers rounded once (or one
    // copy), whichever thread does it.
    // 1. random inputs; both sides
    torch::manual_seed(0);
    Shape shape_list = {4, 1024, 768};
    Tensor residual_tensor = random_bf16_tensor(shape_list);
    Tensor branch_output_tensor = random_bf16_tensor(shape_list);
    Tensor output_grad_tensor = random_bf16_tensor(shape_list);
    Tensor results_list[2][3];                               // [side][output, residual_grad, branch_output_grad]: 0 here, 1 the 4-value path
    for (int side_index_val = 0; side_index_val < 2; side_index_val++)
        for (int result_index_val = 0; result_index_val < 3; result_index_val++)
            results_list[side_index_val][result_index_val] = Tensor::zeros(shape_list, DType::BFloat16);
    residual_bf16_forward(results_list[0][0], residual_tensor, branch_output_tensor);
    residual_bf16_backward(results_list[0][1], results_list[0][2], output_grad_tensor);
    residual_forward(results_list[1][0], residual_tensor, branch_output_tensor);
    residual_backward(results_list[1][1], results_list[1][2], output_grad_tensor);
    // 2. the same bits, checked on the GPU
    for (int result_index_val = 0; result_index_val < 3; result_index_val++)
        ASSERT_EQ(torch::equal(pytorch_copy_of(results_list[0][result_index_val]), pytorch_copy_of(results_list[1][result_index_val])),
                  true);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example and 4 more values, 8 in all (one load: thread 0 has it all), every one a
// number bfloat16 holds exactly:
//
//     residual             1      2      3      4      5     6      7     8
//     branch_output        0.5   -2      0      2     -5     0.25   1    -8
//     output               1.5    0      3      6      0     6.25   8     0       F1: y = r + b
//     output_grad          1     -2      0.5    3     -1     4      0.25  2
//     residual_grad        1     -2      0.5    3     -1     4      0.25  2       B2: dr(1) = dy
//     branch_output_grad   1     -2      0.5    3     -1     4      0.25  2       B1: db = dy

static const std::vector<float> NOTE_RESIDUAL = {1, 2, 3, 4, 5, 6, 7, 8};
static const std::vector<float> NOTE_BRANCH_OUTPUT = {0.5f, -2, 0, 2, -5, 0.25f, 1, -8};
static const std::vector<float> NOTE_OUTPUT = {1.5f, 0, 3, 6, 0, 6.25f, 8, 0};
static const std::vector<float> NOTE_OUTPUT_GRAD = {1, -2, 0.5f, 3, -1, 4, 0.25f, 2};

static void forward_of_the_note_s_example(void)
{
    // Expect: 1 + 0.5, 2 - 2, 3 + 0, 4 + 2, 5 - 5, 6 + 0.25, 7 + 1, 8 - 8 = 1.5 0 3 6 0 6.25 8 0, exactly: every sum
    // is a bfloat16.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({8}, DType::BFloat16);
    residual_bf16_forward(output_tensor, bf16_of({8}, NOTE_RESIDUAL), bf16_of({8}, NOTE_BRANCH_OUTPUT));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), NOTE_OUTPUT);
}

static void backward_gives_output_grad_to_both(void)
{
    // Expect: the slope of r + b is 1 for r and 1 for b, so residual_grad and branch_output_grad are both
    // output_grad, exactly. Not halved: the add copies the gradient, it does not split it.
    // 1. the backward
    Tensor residual_grad_tensor = Tensor::zeros({8}, DType::BFloat16);
    Tensor branch_output_grad_tensor = Tensor::zeros({8}, DType::BFloat16);
    residual_bf16_backward(residual_grad_tensor, branch_output_grad_tensor, bf16_of({8}, NOTE_OUTPUT_GRAD));
    // 2. check
    ASSERT_EQ(floats_of(residual_grad_tensor), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(floats_of(branch_output_grad_tensor), NOTE_OUTPUT_GRAD);
}

static void a_branch_of_0_leaves_the_row_as_it_is(void)
{
    // Expect: branch_output all 0, as a block that has learned nothing gives: output is residual, 1 .. 8, exactly. The
    // block is then the identity, and the row passes through it unchanged.
    // 1. the forward with a branch of 0
    Tensor output_tensor = Tensor::zeros({8}, DType::BFloat16);
    residual_bf16_forward(output_tensor, bf16_of({8}, NOTE_RESIDUAL), Tensor::zeros({8}, DType::BFloat16));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), NOTE_RESIDUAL);
}

static void adding_minus_itself_gives_0(void)
{
    // Expect: x + (-x) = 0 exactly for any bfloat16, also 1000000 (999,424 as a bfloat16) and -3.25: 0 x 8.
    // 1. the forward
    Tensor output_tensor = Tensor::ones({8}, DType::BFloat16);
    residual_bf16_forward(output_tensor, bf16_of({8}, {1.5f, -2, 1000000, -3.25f, 7, -0.125f, 300, 96}),
                          bf16_of({8}, {-1.5f, 2, -1000000, 3.25f, -7, 0.125f, -300, -96}));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), std::vector<float>(8, 0.0f));
}

static void adds_in_float_then_rounds_once(void)
{
    // Expect: a bfloat16 has 8 significant bits, so from 1 to 2 the bfloat16s are 2^-7 = 0.0078125 apart, from 64 to
    // 128 0.5 apart, from 128 to 256 1 apart, from 256 to 512 2 apart, from 512 to 1024 4 apart. Each pair is added in
    // float (exactly here) and rounded once to the nearer bfloat16, to the even one on a tie:
    //
    //     256 + 1           = 257, halfway between 256 and 258: 256
    //     1 + 0.001         = 1.001, much nearer 1 than 1.0078125: 1
    //     256 + 2           = 258, a bfloat16: 258
    //     1000 + 1          = 1001, between 1000 and 1004: 1000
    //     1 + 0.00390625    = 1.00390625, halfway between 1 and 1.0078125: 1
    //     128 + 0.5         = 128.5, halfway between 128 and 129: 128
    //     64 + 0.25         = 64.25, halfway between 64 and 64.5: 64
    //     1 + 0.0078125     = 1.0078125, a bfloat16: 1.0078125
    //
    // PyTorch's float + then .to(bfloat16) rounds the same way, so part 1 can ask for the same bits.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({8}, DType::BFloat16);
    residual_bf16_forward(output_tensor, bf16_of({8}, {256, 1, 256, 1000, 1, 128, 64, 1}),
                          bf16_of({8}, {1, 0.001f, 2, 1, 0.00390625f, 0.5f, 0.25f, 0.0078125f}));
    // 2. check
    ASSERT_EQ(floats_of(output_tensor), std::vector<float>({256, 1, 258, 1000, 1, 128, 64, 1.0078125f}));
}

static void forward_can_write_into_residual_itself(void)
{
    // Expect: the residual tensor passed as output too, as a trainer that keeps one residual stream would: each
    // thread reads its residual before writing its output, so the result is the note's output.
    // 1. the forward, in place
    Tensor residual_tensor = bf16_of({8}, NOTE_RESIDUAL);
    residual_bf16_forward(residual_tensor, residual_tensor, bf16_of({8}, NOTE_BRANCH_OUTPUT));
    // 2. check
    ASSERT_EQ(floats_of(residual_tensor), NOTE_OUTPUT);
}

static void backward_writes_over_both_grads(void)
{
    // Expect: residual_grad and branch_output_grad start at 10 (left from somewhere else); the backward writes over
    // them, so after it both are output_grad, with no 10 added. A second backward gives the same: the layer after
    // (LayerNorm) adds its part onto residual_grad, not the residual add.
    // 1. grads that start at 10
    Tensor output_grad_tensor = bf16_of({8}, NOTE_OUTPUT_GRAD);
    Tensor residual_grad_tensor = Tensor::zeros({8}, DType::BFloat16).fill_(10);
    Tensor branch_output_grad_tensor = Tensor::zeros({8}, DType::BFloat16).fill_(10);
    // 2. one backward: written over
    residual_bf16_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    ASSERT_EQ(floats_of(residual_grad_tensor), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(floats_of(branch_output_grad_tensor), NOTE_OUTPUT_GRAD);
    // 3. a second backward: the same
    residual_bf16_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    ASSERT_EQ(floats_of(residual_grad_tensor), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(floats_of(branch_output_grad_tensor), NOTE_OUTPUT_GRAD);
}

static void backward_can_write_into_output_grad_itself(void)
{
    // Expect: the output_grad tensor passed as residual_grad too, as llm.c's trainer does (and model_bf16.cuh's block_bf16_backward,
    // input_grad from residual2_grad): each thread reads its output_grad before writing, so residual_grad stays the
    // note's output_grad and branch_output_grad gets the same.
    // 1. the backward, in place
    Tensor grad_tensor = bf16_of({8}, NOTE_OUTPUT_GRAD);
    Tensor branch_output_grad_tensor = Tensor::zeros({8}, DType::BFloat16);
    residual_bf16_backward(grad_tensor, branch_output_grad_tensor, grad_tensor);
    // 2. check
    ASSERT_EQ(floats_of(grad_tensor), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(floats_of(branch_output_grad_tensor), NOTE_OUTPUT_GRAD);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with RESIDUAL_BF16_THREADS_PER_BLOCK (256), and nothing is spilled: a
    // thread's 8 values (and the 16-byte loads they are read into) stay in registers. cudaFuncGetAttributes reads what
    // the compiler wrote down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)residual_bf16_forward_kernel, (const void *)residual_bf16_backward_kernel};
    const char *names_list[] = {"residual_bf16_forward_kernel", "residual_bf16_backward_kernel"};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= RESIDUAL_BF16_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: residual_bf16_forward and residual_bf16_backward called with threads_per_block 32, 128, 512 and 1024
    // (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK) give the same output and gradients as with the default,
    // RESIDUAL_BF16_THREADS_PER_BLOCK, to the last bit: each value is one add or one copy, the same whichever thread
    // does it. {3, 7, 104}: 273 loads, so every size has a part-full last block.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP,
                                    HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random inputs; the default launch's results
    torch::manual_seed(0);
    Shape shape_list = {3, 7, 104};
    Tensor residual_tensor = random_bf16_tensor(shape_list);
    Tensor branch_output_tensor = random_bf16_tensor(shape_list);
    Tensor output_grad_tensor = random_bf16_tensor(shape_list);
    Tensor usual_output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor usual_residual_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor usual_branch_output_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    residual_bf16_forward(usual_output_tensor, residual_tensor, branch_output_tensor);
    residual_bf16_backward(usual_residual_grad_tensor, usual_branch_output_grad_tensor, output_grad_tensor);

    // 2. each block size, against the default's
    for (int threads_per_block_val : block_sizes_list)
    {
        Tensor output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
        Tensor residual_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
        Tensor branch_output_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
        residual_bf16_forward(output_tensor, residual_tensor, branch_output_tensor, threads_per_block_val);
        residual_bf16_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor, threads_per_block_val);
        ASSERT_EQ(floats_of(output_tensor), floats_of(usual_output_tensor));
        ASSERT_EQ(floats_of(residual_grad_tensor), floats_of(usual_residual_grad_tensor));
        ASSERT_EQ(floats_of(branch_output_grad_tensor), floats_of(usual_branch_output_grad_tensor));
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_rows();
    matches_pytorch_with_a_part_full_last_block();
    the_same_bits_as_the_4_value_path();

    forward_of_the_note_s_example();
    backward_gives_output_grad_to_both();
    a_branch_of_0_leaves_the_row_as_it_is();
    adding_minus_itself_gives_0();
    adds_in_float_then_rounds_once();
    forward_can_write_into_residual_itself();
    backward_writes_over_both_grads();
    backward_can_write_into_output_grad_itself();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
