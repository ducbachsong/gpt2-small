// test_qk_norm_8bit.cu — tests for cuda/kernels/8bit/qk_norm_8bit.cuh (qk_norm_8bit_forward, qk_norm_8bit_backward):
// QK-norm before the 8-bit model's attention, each head row of query and key scaled to a root mean square of
// QK_NORM_8BIT_GAIN in bfloat16 qkv. Every check prints its test case, input, expected and actual value, and ok; the
// first FAIL stops the program.
//
// Part 1 checks against PyTorch in float on the same values, the backward by autograd; part 2 checks numbers worked
// out by hand; part 3 checks that both kernels fit their block. Any GPU.
#include <torch/types.h>              // torch::randn, torch::rsqrt, backward(); before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include "../common/test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../kernels/8bit/qk_norm_8bit.cuh"

// y and dx are rounded to bfloat16 here (2^-9 of a value); the backward also reads y rounded: within 1e-2 of the
// largest value. rstd is float on both sides: 1e-5.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double FLOAT_TOLERANCE = 1e-5;

#define D24_EMBED_DIM 1536            // the d24 model's: 12 heads of 128
#define D24_NUM_HEADS 12

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation `std_val`, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
///   std_val      the standard deviation; 1 if not given
static Tensor random_bf16_tensor(const Shape &shape_list, float std_val = 1.0f)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA)) * std_val;
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes a bfloat16 tensor of `shape_list` on the GPU holding `values_list`, given in row order, each rounded.
static Tensor bf16_tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor.to(DType::BFloat16);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values.
///
///   source_tensor   on the GPU
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    // GPU to GPU (a bfloat16 tensor's values turned into floats first, exactly).
    Tensor floats_tensor = source_tensor.to(DType::Float32);
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end());
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between two PyTorch tensors, as a fraction of the second's largest value.
static double relative_difference(const torch::Tensor &mine_tensor, const torch::Tensor &expected_tensor)
{
    return ((mine_tensor - expected_tensor).abs().max() / expected_tensor.abs().max()).item<double>();
}

/// Gives PyTorch's QK-norm of head rows x {..., head_size}: gain · x / sqrt(mean(x²) + eps), in float.
static torch::Tensor pytorch_qk_norm(const torch::Tensor &x_tensor)
{
    return QK_NORM_8BIT_GAIN * x_tensor * torch::rsqrt(x_tensor.pow(2).mean(-1, true) + QK_NORM_8BIT_EPS);
}

/// Splits a PyTorch qkv {B, T, 3D} into its query, key and value as head rows {B, NH, T, HS}.
static std::vector<torch::Tensor> pytorch_parts_of(const torch::Tensor &qkv_tensor, int64_t num_heads_val)
{
    int64_t B = qkv_tensor.size(0), T = qkv_tensor.size(1), D = qkv_tensor.size(2) / 3;
    std::vector<torch::Tensor> parts_list;
    for (const torch::Tensor &part_tensor : qkv_tensor.split(D, -1))
        parts_list.push_back(part_tensor.reshape({B, T, num_heads_val, D / num_heads_val}).transpose(1, 2));
    return parts_list;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs QK-norm's forward and backward on random values and checks them against PyTorch.
///
///   test_ptr         the test's name
///   batch_size_val   rows of the batch
///   seq_len_val      tokens a row
///   embed_dim_val    the model's width
///   num_heads_val    heads
static void matches_pytorch(const char *test_ptr, size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val,
                            size_t num_heads_val)
{
    // Runs the forward on random qkv (std 3, so the norm has work to do), then the backward on random dy with the
    // forward's y as qkv_rounded; PyTorch's QK-norm in float on the same values, dx by autograd. Checks: query's and
    // key's y and dx within BFLOAT16_TOLERANCE, the rstds within FLOAT_TOLERANCE, value's part of qkv and of qkv_grad
    // untouched to the bit.
    // 1. mine: the forward, then the backward
    torch::manual_seed(0);
    size_t B = batch_size_val, T = seq_len_val, D = embed_dim_val, NH = num_heads_val;
    Tensor qkv_tensor = random_bf16_tensor({B, T, 3 * D}, 3.0f);
    Tensor x_copy_tensor = qkv_tensor.to(DType::Float32).to(DType::BFloat16);    // x, before the forward writes over it
    Tensor query_rstd_tensor = Tensor::zeros({B, NH, T});
    Tensor key_rstd_tensor = Tensor::zeros({B, NH, T});
    qk_norm_8bit_forward(qkv_tensor, query_rstd_tensor, key_rstd_tensor);
    Tensor dy_tensor = random_bf16_tensor({B, T, 3 * D});
    Tensor qkv_grad_tensor = dy_tensor.to(DType::Float32).to(DType::BFloat16);  // written over by the backward
    qk_norm_8bit_backward(qkv_grad_tensor, qkv_tensor, query_rstd_tensor, key_rstd_tensor);
    // 2. PyTorch's, in float on the same x and dy
    std::vector<torch::Tensor> x_list = pytorch_parts_of(pytorch_copy_of(x_copy_tensor), (int64_t)NH);
    std::vector<torch::Tensor> dy_list = pytorch_parts_of(pytorch_copy_of(dy_tensor), (int64_t)NH);
    std::vector<torch::Tensor> y_list = pytorch_parts_of(pytorch_copy_of(qkv_tensor), (int64_t)NH);
    std::vector<torch::Tensor> dx_list = pytorch_parts_of(pytorch_copy_of(qkv_grad_tensor), (int64_t)NH);
    torch::Tensor rstd_list[2] = {pytorch_copy_of(query_rstd_tensor), pytorch_copy_of(key_rstd_tensor)};
    // 3. check: query (0) and key (1); then value
    print_test_name(test_ptr);
    printf("    qkv {%zu, %zu, %zu}, %zu heads of %zu\n", B, T, 3 * D, NH, D / NH);
    for (int part_val = 0; part_val < 2; part_val++) {
        torch::Tensor x_tensor = x_list[part_val].clone().requires_grad_(true);
        torch::Tensor expected_y_tensor = pytorch_qk_norm(x_tensor);
        (expected_y_tensor * dy_list[part_val]).sum().backward();
        torch::Tensor expected_rstd_tensor = torch::rsqrt(x_list[part_val].pow(2).mean(-1) + QK_NORM_8BIT_EPS);
        printf("    %s:\n", part_val == 0 ? "query" : "key");
        ASSERT_NEAR(relative_difference(y_list[part_val], expected_y_tensor.detach()), 0.0, BFLOAT16_TOLERANCE);
        ASSERT_NEAR(relative_difference(rstd_list[part_val], expected_rstd_tensor), 0.0, FLOAT_TOLERANCE);
        ASSERT_NEAR(relative_difference(dx_list[part_val], x_tensor.grad()), 0.0, BFLOAT16_TOLERANCE);
    }
    printf("    value, untouched:\n");
    ASSERT_EQ((y_list[2] - x_list[2]).abs().max().item<float>(), 0.0f);
    ASSERT_EQ((dx_list[2] - dy_list[2]).abs().max().item<float>(), 0.0f);
}

static void matches_pytorch_on_d24_rows(void)
{
    // Expect: 2 x 256 tokens of the d24 model's qkv (12 heads of 128): y, rstd and dx of query and key as PyTorch's,
    // value's parts untouched.
    matches_pytorch(__func__, 2, 256, D24_EMBED_DIM, D24_NUM_HEADS);
}

static void matches_pytorch_with_heads_of_64(void)
{
    // Expect: 3 x 5 tokens, 4 heads of 64 (half a warp's float4s a row: lanes 16 .. 31 idle), a token count not a
    // multiple of the block's 8 warps: the same checks.
    matches_pytorch(__func__, 3, 5, 256, 4);
}

// ══ part 2: worked out by hand ════════════════════════════════════════════════

static void one_head_row_by_hand(void)
{
    // Expect: 1 token, 1 head of 4: q 1 2 3 4, k 2 2 2 2, v 5 6 7 8. q: mean(q²) = 30 / 4 = 7.5, rstd 1 / sqrt(7.5) =
    // 0.3651484, y = 1.2 x 0.3651484 x q = 0.4381780 0.8763561 1.3145341 1.7527122 (each to bfloat16); k: mean 4, rstd
    // 0.5, y = 1.2 each; v as it is. Backward with dy = 1 0 0 0 on q: n = y / 1.2 = q · rstd, mean(dy · n) =
    // 0.3651484 / 4 = 0.0912871, dx = 1.2 x 0.3651484 x (dy - n x 0.0912871) = 0.4381780 x (0.9666667, -0.0666667,
    // -0.1, -0.1333333) = 0.4235721 -0.0292119 -0.0438178 -0.0584237 (y and dx rounded to bfloat16: within 2e-3).
    Tensor qkv_tensor = bf16_tensor_of({1, 1, 12}, {1, 2, 3, 4, 2, 2, 2, 2, 5, 6, 7, 8});
    Tensor query_rstd_tensor = Tensor::zeros({1, 1, 1});
    Tensor key_rstd_tensor = Tensor::zeros({1, 1, 1});
    qk_norm_8bit_forward(qkv_tensor, query_rstd_tensor, key_rstd_tensor);
    std::vector<float> y_list = qkv_tensor.to(DType::Float32).tolist();
    print_test_name(__func__);
    ASSERT_NEAR(query_rstd_tensor.tolist()[0], 0.3651484f, 1e-6);
    ASSERT_NEAR(key_rstd_tensor.tolist()[0], 0.5f, 1e-6);
    const float expected_y_list[12] = {0.4381780f, 0.8763561f, 1.3145341f, 1.7527122f, 1.2f, 1.2f, 1.2f, 1.2f, 5, 6, 7, 8};
    for (int index_val = 0; index_val < 12; index_val++)
        ASSERT_NEAR(y_list[index_val], expected_y_list[index_val], 1e-2);
    Tensor qkv_grad_tensor = bf16_tensor_of({1, 1, 12}, {1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1});
    qk_norm_8bit_backward(qkv_grad_tensor, qkv_tensor, query_rstd_tensor, key_rstd_tensor);
    std::vector<float> dx_list = qkv_grad_tensor.to(DType::Float32).tolist();
    const float expected_dx_list[12] = {0.4235721f, -0.0292119f, -0.0438178f, -0.0584237f, 0, 0, 0, 0, 1, 1, 1, 1};
    for (int index_val = 0; index_val < 12; index_val++)
        ASSERT_NEAR(dx_list[index_val], expected_dx_list[index_val], 2e-3);
}

// ══ part 3: the kernels' blocks ═══════════════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: both kernels can be launched with HARDWARE_MAX_THREADS_PER_BLOCK threads and nothing is spilled:
    // cudaFuncGetAttributes' maxThreadsPerBlock and localSizeBytes, read without running anything.
    const void *kernels_list[] = {(const void *)qk_norm_8bit_forward_kernel, (const void *)qk_norm_8bit_backward_kernel};
    const char *names_list[] = {"qk_norm_8bit_forward_kernel", "qk_norm_8bit_backward_kernel"};
    for (int kernel_index_val = 0; kernel_index_val < 2; kernel_index_val++) {
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names_list[kernel_index_val],
               attributes_val.numRegs, attributes_val.maxThreadsPerBlock, attributes_val.localSizeBytes);
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= HARDWARE_MAX_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_d24_rows();
    matches_pytorch_with_heads_of_64();

    one_head_row_by_hand();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
