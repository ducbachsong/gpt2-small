// test_residual.cu — tests for llmc/kernels/residual.cuh (residual_forward, residual_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's + with autograd for the backward; part 2 checks numbers worked out
// by hand; part 3 checks that each kernel's registers fit its block, and that any block size gives the same
// bits. The timings are in tunes/tune_residual.cu.
// (The C# trainer had no residual test: TorchSharp's + was trusted.)
#include <torch/types.h>              // torch::randn, backward(); libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "test_helpers.h"             // ASSERT_EQ: every check here is exact
#include "../llmc/kernels/residual.cuh"
// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 3.1M values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives every value of a PyTorch tensor as a CPU list in row order, to compare with mine.
static std::vector<float> list_of(const torch::Tensor &tensor)
{
    torch::Tensor on_cpu_tensor = tensor.detach().contiguous().cpu();
    return std::vector<float>(on_cpu_tensor.data_ptr<float>(), on_cpu_tensor.data_ptr<float>() + on_cpu_tensor.numel());
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random values, mine and PyTorch's residual + branch_output with
/// autograd, and checks that output, residual_grad and branch_output_grad are the same to the last bit:
/// both sides do the same one float add per value, and the backward only copies.
static void forward_and_backward_match_pytorch(const Shape &shape_list)
{
    // 1. random inputs; mine
    torch::manual_seed(0);
    Tensor residual_tensor = random_tensor_on_gpu(shape_list);
    Tensor branch_output_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list);
    Tensor output_tensor = Tensor::zeros(shape_list);
    Tensor residual_grad_tensor = Tensor::zeros(shape_list);
    Tensor branch_output_grad_tensor = Tensor::zeros(shape_list);
    residual_forward(output_tensor, residual_tensor, branch_output_tensor);
    residual_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);

    // 2. PyTorch's, with autograd
    torch::Tensor pytorch_residual_tensor = pytorch_copy_of(residual_tensor).requires_grad_(true);
    torch::Tensor pytorch_branch_output_tensor = pytorch_copy_of(branch_output_tensor).requires_grad_(true);
    torch::Tensor pytorch_output_tensor = pytorch_residual_tensor + pytorch_branch_output_tensor;
    pytorch_output_tensor.backward(pytorch_copy_of(output_grad_tensor));

    // 3. the same bits
    ASSERT_EQ(output_tensor.tolist(), list_of(pytorch_output_tensor));
    ASSERT_EQ(residual_grad_tensor.tolist(), list_of(pytorch_residual_tensor.grad()));
    ASSERT_EQ(branch_output_grad_tensor.tolist(), list_of(pytorch_branch_output_tensor.grad()));
}

static void matches_pytorch_on_gpt2_rows(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens of 768, the residual stream of GPT-2 small:
    // 12,288 values, 3072 float4s, 12 full blocks. output equals PyTorch's residual + branch_output, and
    // both gradients its autograd's, exactly.
    forward_and_backward_match_pytorch({2, 8, 768});
}

static void matches_pytorch_with_a_part_full_last_block(void)
{
    // Expect: {3, 7, 100}: 2100 values, 525 float4s, 3 blocks of 256 threads; the last has 13 float4s
    // for 256 threads, and its other 243 threads must write nothing. The same as PyTorch, exactly.
    forward_and_backward_match_pytorch({3, 7, 100});
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: {2, 8, 768} random values rounded to bfloat16, added in bfloat16 tensors. Each value is read as
    // a float, added in float, and rounded once to bfloat16 as it is stored: the same as PyTorch's float add
    // of the same values, rounded to bfloat16 by PyTorch, to the last bit (both round to the nearest, a tie
    // to the even one). The gradients are output_grad copied, so the same bfloat16s exactly.

    // 1. random inputs in bfloat16; mine
    torch::manual_seed(0);
    Shape shape_list = {2, 8, 768};
    Tensor residual_tensor = random_tensor_on_gpu(shape_list).to(DType::BFloat16);
    Tensor branch_output_tensor = random_tensor_on_gpu(shape_list).to(DType::BFloat16);
    Tensor output_grad_tensor = random_tensor_on_gpu(shape_list).to(DType::BFloat16);
    Tensor output_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor residual_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    Tensor branch_output_grad_tensor = Tensor::zeros(shape_list, DType::BFloat16);
    residual_forward(output_tensor, residual_tensor, branch_output_tensor);
    residual_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);

    // 2. PyTorch's float add of the same values
    torch::Tensor pytorch_output_tensor = pytorch_copy_of(residual_tensor.to(DType::Float32)) +
                                          pytorch_copy_of(branch_output_tensor.to(DType::Float32));

    // 3. the same bits once rounded; the gradients output_grad itself
    ASSERT_EQ(output_tensor.to(DType::Float32).tolist(), list_of(pytorch_output_tensor.to(torch::kBFloat16).to(torch::kFloat)));
    ASSERT_EQ(residual_grad_tensor.to(DType::Float32).tolist(), output_grad_tensor.to(DType::Float32).tolist());
    ASSERT_EQ(branch_output_grad_tensor.to(DType::Float32).tolist(), output_grad_tensor.to(DType::Float32).tolist());
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example, 4 values (one float4: thread 0 has it all):
//
//     residual             1      2      3      4
//     branch_output        0.5   -2      0      2
//     output               1.5    0      3      6        F1: y = r + b
//     output_grad          1     -2      0.5    3
//     residual_grad        1     -2      0.5    3        B2: dr(1) = dy
//     branch_output_grad   1     -2      0.5    3        B1: db = dy

static const std::vector<float> NOTE_RESIDUAL = {1, 2, 3, 4};
static const std::vector<float> NOTE_BRANCH_OUTPUT = {0.5f, -2, 0, 2};
static const std::vector<float> NOTE_OUTPUT = {1.5f, 0, 3, 6};
static const std::vector<float> NOTE_OUTPUT_GRAD = {1, -2, 0.5f, 3};

static void forward_of_the_note_s_example(void)
{
    // Expect: 1 + 0.5, 2 - 2, 3 + 0, 4 + 2 = 1.5 0 3 6, exactly: every value is a float.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({4});
    residual_forward(output_tensor, tensor_of({4}, NOTE_RESIDUAL), tensor_of({4}, NOTE_BRANCH_OUTPUT));
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), NOTE_OUTPUT);
}

static void backward_gives_output_grad_to_both(void)
{
    // Expect: the slope of r + b is 1 for r and 1 for b, so residual_grad and branch_output_grad are both
    // output_grad, 1 -2 0.5 3, exactly. Not halved: the add copies the gradient, it does not split it.
    // 1. the backward
    Tensor residual_grad_tensor = Tensor::zeros({4});
    Tensor branch_output_grad_tensor = Tensor::zeros({4});
    residual_backward(residual_grad_tensor, branch_output_grad_tensor, tensor_of({4}, NOTE_OUTPUT_GRAD));
    // 2. check
    ASSERT_EQ(residual_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
}

static void a_branch_of_0_leaves_the_row_as_it_is(void)
{
    // Expect: branch_output all 0, as a block that has learned nothing gives: output is residual,
    // 1 2 3 4, exactly. The block is then the identity, and the row passes through it unchanged.
    // 1. the forward with a branch of 0
    Tensor output_tensor = Tensor::zeros({4});
    residual_forward(output_tensor, tensor_of({4}, NOTE_RESIDUAL), Tensor::zeros({4}));
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), NOTE_RESIDUAL);
}

static void adding_minus_itself_gives_0(void)
{
    // Expect: x + (-x) = 0 exactly for any float, also 1000000 and -3.25: 0 0 0 0.
    // 1. the forward
    Tensor output_tensor = Tensor::ones({4});
    residual_forward(output_tensor, tensor_of({4}, {1.5f, -2, 1000000, -3.25f}),
                     tensor_of({4}, {-1.5f, 2, -1000000, 3.25f}));
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>(4, 0.0f));
}

static void adds_in_float_as_pytorch_does(void)
{
    // Expect: a float has 24 bits of digits, so above 2^24 = 16,777,216 the floats are 2 apart, and
    // above 10^8 they are 8 apart. A value between two floats is rounded to the nearer one, to the even
    // one on a tie:
    //
    //     16777216 + 1      = 16777217, halfway between 16777216 and 16777218: 16777216
    //     1 + 0.00000001    = 1.00000001, much nearer 1 than the next float, 1.0000001: 1
    //     16777216 + 2      = 16777218, a float: 16777218
    //     100000000 + 1     = 100000001, nearest float 100000000
    //
    // PyTorch's + rounds the same way, so part 1 can ask for the same bits.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({4});
    residual_forward(output_tensor, tensor_of({4}, {16777216, 1, 16777216, 100000000}), tensor_of({4}, {1, 1e-8f, 2, 1}));
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({16777216, 1, 16777218, 100000000}));
}

static void forward_can_write_into_residual_itself(void)
{
    // Expect: the residual tensor passed as output too, as a trainer that keeps one residual stream
    // would: each thread reads its residual before writing its output, so the result is 1.5 0 3 6.
    // 1. the forward, in place
    Tensor residual_tensor = tensor_of({4}, NOTE_RESIDUAL);
    residual_forward(residual_tensor, residual_tensor, tensor_of({4}, NOTE_BRANCH_OUTPUT));
    // 2. check
    ASSERT_EQ(residual_tensor.tolist(), NOTE_OUTPUT);
}

static void backward_writes_over_both_grads(void)
{
    // Expect: residual_grad and branch_output_grad start at 10 (left from somewhere else); the backward
    // writes over them, so after it both are output_grad, with no 10 added. A second backward gives the
    // same: the layer after (LayerNorm) adds its part onto residual_grad, not the residual add.
    // 1. grads that start at 10
    Tensor output_grad_tensor = tensor_of({4}, NOTE_OUTPUT_GRAD);
    Tensor residual_grad_tensor = Tensor::zeros({4}).fill_(10);
    Tensor branch_output_grad_tensor = Tensor::zeros({4}).fill_(10);
    // 2. one backward: written over
    residual_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    ASSERT_EQ(residual_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
    // 3. a second backward: the same
    residual_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor);
    ASSERT_EQ(residual_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
}

static void backward_can_write_into_output_grad_itself(void)
{
    // Expect: the output_grad tensor passed as residual_grad too, as llm.c's trainer does: each thread
    // reads its output_grad before writing, so residual_grad stays 1 -2 0.5 3 and branch_output_grad
    // gets the same.
    // 1. the backward, in place
    Tensor grad_tensor = tensor_of({4}, NOTE_OUTPUT_GRAD);
    Tensor branch_output_grad_tensor = Tensor::zeros({4});
    residual_backward(grad_tensor, branch_output_grad_tensor, grad_tensor);
    // 2. check
    ASSERT_EQ(grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
    ASSERT_EQ(branch_output_grad_tensor.tolist(), NOTE_OUTPUT_GRAD);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with RESIDUAL_THREADS_PER_BLOCK (256), and nothing is spilled.
    // The compiler picks each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote
    // down, without running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)residual_forward_kernel<float>,
                                  (const void *)residual_backward_kernel<float>};
    const char *names_list[] = {"residual_forward_kernel", "residual_backward_kernel"};
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
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= RESIDUAL_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: residual_forward and residual_backward called with threads_per_block 32, 128, 512 and 1024
    // (one warp up to MAX_THREADS_PER_BLOCK) give the same output and gradients as with the default,
    // RESIDUAL_THREADS_PER_BLOCK, to the last bit, in float and in bfloat16: each value is one add or one copy,
    // the same whichever thread does it. {3, 7, 100}: 525 float4s, so every size has a part-full last block.
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 16 * THREADS_PER_WARP, MAX_THREADS_PER_BLOCK};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random inputs; the default launch's results
        torch::manual_seed(0);
        Shape shape_list = {3, 7, 100};
        Tensor residual_tensor = random_tensor_on_gpu(shape_list).to(dtype_val);
        Tensor branch_output_tensor = random_tensor_on_gpu(shape_list).to(dtype_val);
        Tensor output_grad_tensor = random_tensor_on_gpu(shape_list).to(dtype_val);
        Tensor usual_output_tensor = Tensor::zeros(shape_list, dtype_val);
        Tensor usual_residual_grad_tensor = Tensor::zeros(shape_list, dtype_val);
        Tensor usual_branch_output_grad_tensor = Tensor::zeros(shape_list, dtype_val);
        residual_forward(usual_output_tensor, residual_tensor, branch_output_tensor);
        residual_backward(usual_residual_grad_tensor, usual_branch_output_grad_tensor, output_grad_tensor);

        // 2. each block size, against the default's
        for (int threads_per_block_val : block_sizes_list)
        {
            Tensor output_tensor = Tensor::zeros(shape_list, dtype_val);
            Tensor residual_grad_tensor = Tensor::zeros(shape_list, dtype_val);
            Tensor branch_output_grad_tensor = Tensor::zeros(shape_list, dtype_val);
            residual_forward(output_tensor, residual_tensor, branch_output_tensor, threads_per_block_val);
            residual_backward(residual_grad_tensor, branch_output_grad_tensor, output_grad_tensor, threads_per_block_val);
            ASSERT_EQ(output_tensor.to(DType::Float32).tolist(), usual_output_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(residual_grad_tensor.to(DType::Float32).tolist(), usual_residual_grad_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(branch_output_grad_tensor.to(DType::Float32).tolist(),
                      usual_branch_output_grad_tensor.to(DType::Float32).tolist());
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_rows();
    matches_pytorch_with_a_part_full_last_block();
    forward_and_backward_in_bfloat16();

    forward_of_the_note_s_example();
    backward_gives_output_grad_to_both();
    a_branch_of_0_leaves_the_row_as_it_is();
    adding_minus_itself_gives_0();
    adds_in_float_as_pytorch_does();
    forward_can_write_into_residual_itself();
    backward_writes_over_both_grads();
    backward_can_write_into_output_grad_itself();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
