// test_fma_matmul_bf16.cu — tests for llmc/kernels/bf16/fma_matmul_bf16.cuh (fma_matmul_bf16): output = left · rightᵀ (+ bias) (+ output),
// on the CUDA cores, bfloat16 or float, the sums in float. Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks against PyTorch's float product of the same values: GPT-2 small's 5 linear layers, the
// backward's two products (left or right turned round), and sizes no square divides; part 2 checks numbers that
// must come out exactly, and that a turned tensor gives the same bits as the plain one; part 3 checks that the
// kernels' registers fit their block. Any GPU. The timings are in tunes/tune_fma_matmul_bf16.cu.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/bf16/fma_matmul_bf16.cuh"

// Against PyTorch in float on the same bfloat16 values: the sums are float on both sides, added in another
// order (about 1e-6 of a value apart), then mine are rounded to bfloat16, by up to 2^-9 (0.2%) of each value.
// The tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;
static const double FLOAT_TOLERANCE = 1e-5;   // float against float: only the order of the sums differs

#define GPT2_ROWS 4096                // a micro-batch of 4 rows of 1024 tokens
#define GPT2_EMBED_DIM 768            // GPT-2 small's widths
#define GPT2_QKV_DIM (3 * GPT2_EMBED_DIM)
#define GPT2_MLP_DIM (4 * GPT2_EMBED_DIM)
#define GPT2_PADDED_VOCAB_SIZE 50304

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a float tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU; then, unless `dtype_val` is float, rounded to it.
static Tensor random_tensor(const Shape &shape_list, DType dtype_val = DType::BFloat16)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(dtype_val);
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU. A bfloat16 tensor's
/// values are turned into floats first, exactly.
static torch::Tensor pytorch_copy_of(const Tensor &source_tensor)
{
    Tensor floats_tensor = source_tensor.to(DType::Float32);  // the source_tensor itself if it is float already
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(source_tensor.shape().begin(), source_tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(floats_tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Makes my tensor of `dtype_val` from a PyTorch float tensor on the GPU, GPU to GPU (exact when its values are
/// already `dtype_val`'s).
static Tensor tensor_from_pytorch(const torch::Tensor &source_tensor, DType dtype_val)
{
    torch::Tensor floats_tensor = source_tensor.contiguous();
    Shape shape_list(floats_tensor.sizes().begin(), floats_tensor.sizes().end());
    Tensor mine_tensor = Tensor::empty(shape_list);
    cudaCheck(cudaMemcpy(mine_tensor.data(), floats_tensor.data_ptr<float>(), mine_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return mine_tensor.to(dtype_val);
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: the head's output is 206M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Makes the identity {size, size} in bfloat16.
static Tensor identity_bf16(size_t size_val)
{
    std::vector<float> identity_list(size_val * size_val, 0.0f);
    for (size_t index_val = 0; index_val < size_val; index_val++)
        identity_list[index_val * size_val + index_val] = 1.0f;
    Tensor identity_tensor = Tensor::empty({size_val, size_val});
    identity_tensor.copy_from_cpu(identity_list.data());
    return identity_tensor.to(DType::BFloat16);
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs fma_matmul_bf16 on random left (L {rows, sum_length}, or turned: Lᵀ {sum_length, rows}), right (R {columns,
/// sum_length}, or turned Rᵀ) and bias {columns} (or none), and checks the output against PyTorch's float L · Rᵀ +
/// bias of the same values within `tolerance_val`.
static void matmul_matches_pytorch(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val,
                                   bool with_bias_val, bool left_turned_val = false, bool right_turned_val = false,
                                   DType dtype_val = DType::BFloat16, double tolerance_val = BFLOAT16_TOLERANCE)
{
    // 1. random values, each matrix as it sits; mine
    torch::manual_seed(0);
    Tensor left_tensor = random_tensor(left_turned_val ? Shape{sum_length_val, rows_val} : Shape{rows_val, sum_length_val}, dtype_val);
    Tensor right_tensor = random_tensor(right_turned_val ? Shape{sum_length_val, columns_val} : Shape{columns_val, sum_length_val}, dtype_val);
    Tensor bias_tensor = with_bias_val ? random_tensor({columns_val}, dtype_val) : Tensor();
    Tensor output_tensor = Tensor::zeros({rows_val, columns_val}, dtype_val);
    fma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor, false, left_turned_val, right_turned_val);
    // 2. PyTorch's, in float: L and R turned back where they sit turned
    torch::Tensor left_matrix_tensor = left_turned_val ? pytorch_copy_of(left_tensor).t() : pytorch_copy_of(left_tensor);
    torch::Tensor right_matrix_tensor = right_turned_val ? pytorch_copy_of(right_tensor).t() : pytorch_copy_of(right_tensor);
    torch::Tensor expected_tensor = left_matrix_tensor.matmul(right_matrix_tensor.t());
    if (with_bias_val)
        expected_tensor = expected_tensor + pytorch_copy_of(bias_tensor);
    // 3. check
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu%s%s%s\n", rows_val, columns_val, sum_length_val, with_bias_val ? ", bias" : "",
           left_turned_val ? ", left turned" : "", right_turned_val ? ", right turned" : "");
    ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, tolerance_val);
}

static void matches_pytorch_on_gpt2_qkv(void)
{
    // Expect: x {4096, 768} · W {2304, 768}ᵀ + b. Within 1e-2.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_QKV_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_attention_output(void)
{
    // Expect: {4096, 768} · {768, 768}ᵀ + b. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_EMBED_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_mlp_up(void)
{
    // Expect: {4096, 768} · {3072, 768}ᵀ + b. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_MLP_DIM, GPT2_EMBED_DIM, true);
}

static void matches_pytorch_on_gpt2_mlp_down(void)
{
    // Expect: {4096, 3072} · {768, 3072}ᵀ + b: the longest sum of the five. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_EMBED_DIM, GPT2_MLP_DIM, true);
}

static void matches_pytorch_on_gpt2_head(void)
{
    // Expect: {4096, 768} · wte {50304, 768}ᵀ, no bias: the widest. The same.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_PADDED_VOCAB_SIZE, GPT2_EMBED_DIM, false);
}

static void matches_pytorch_on_gpt2_qkv_input_grad(void)
{
    // Expect: the qkv layer's input_grad = output_grad {4096, 2304} · W {2304, 768}: in output = L · Rᵀ, R is
    // {768, 2304} and W sits as Rᵀ, so right is turned. Within 1e-2.
    matmul_matches_pytorch(__func__, GPT2_ROWS, GPT2_EMBED_DIM, GPT2_QKV_DIM, false, false, true);
}

static void matches_pytorch_on_gpt2_qkv_weight_grad(void)
{
    // Expect: the qkv layer's weight_grad = output_gradᵀ · x: L {2304, 4096} sits as output_grad {4096, 2304}, R
    // {768, 4096} as x {4096, 768}, so both are turned; the sum is over the 4096 rows. The same.
    matmul_matches_pytorch(__func__, GPT2_QKV_DIM, GPT2_EMBED_DIM, GPT2_ROWS, false, true, true);
}

static void matches_pytorch_with_sizes_no_tile_divides(void)
{
    // Expect: {100, 72} · {200, 72}ᵀ + b: 100 rows, 200 columns, 72 values of k, none a whole number of squares
    // or steps, so every square at an edge is part-full and the last step mostly past the edge: 0s are read
    // there, which add nothing, and the stores past the edges are skipped. Within 1e-2.
    matmul_matches_pytorch(__func__, 100, 200, 72, true);
}

static void matches_pytorch_turned_with_sizes_no_tile_divides(void)
{
    // Expect: 104 rows, 200 columns, 72 values of k, left turned, then right turned, then both: the turned
    // tiles' edges as part-full as the plain ones'. Within 1e-2 each.
    matmul_matches_pytorch(__func__, 104, 200, 72, true, true, false);
    matmul_matches_pytorch(__func__, 104, 200, 72, true, false, true);
    matmul_matches_pytorch(__func__, 104, 200, 72, true, true, true);
}

static void matches_pytorch_in_float(void)
{
    // Expect: the same product in float, the exact reference linear.cuh uses: qkv's shape with 256 rows, plain,
    // then the backward's ways (right turned; both turned). Float sums of float values, added in another order
    // than PyTorch's: within FLOAT_TOLERANCE (1e-5).
    matmul_matches_pytorch(__func__, 256, GPT2_QKV_DIM, GPT2_EMBED_DIM, true, false, false, DType::Float32, FLOAT_TOLERANCE);
    matmul_matches_pytorch(__func__, 256, GPT2_EMBED_DIM, GPT2_QKV_DIM, false, false, true, DType::Float32, FLOAT_TOLERANCE);
    matmul_matches_pytorch(__func__, GPT2_QKV_DIM, GPT2_EMBED_DIM, 256, false, true, true, DType::Float32, FLOAT_TOLERANCE);
}

// ══ part 2: numbers that must come out exactly ════════════════════════════════

static void times_the_identity_gives_right_turned_round(void)
{
    // Expect: left = the identity {64, 64}, right random {128, 64}: output[i][j] = sum over k of I[i][k] · right[j][k]
    // = right[j][i], one product of 1 and a bfloat16, exact in the float sums and again when rounded back. So
    // output = rightᵀ to the last bit, and the 7s it started with are all written over.
    // 1. the identity, random right, output full of 7s
    torch::manual_seed(1);
    Tensor left_tensor = identity_bf16(64);
    Tensor right_tensor = random_tensor({128, 64});
    Tensor output_tensor = Tensor::zeros({64, 128}, DType::BFloat16);
    output_tensor.fill_(7.0);
    // 2. the product
    fma_matmul_bf16(output_tensor, left_tensor, right_tensor);
    // 3. check against rightᵀ
    torch::Tensor expected_tensor = pytorch_copy_of(right_tensor).t().contiguous();
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
}

static void zero_left_gives_the_bias_in_every_row(void)
{
    // Expect: left all 0, random right and bias {96}: every sum is 0, so output row r = bias exactly, in all 70 rows.
    // 1. the tensors
    torch::manual_seed(2);
    Tensor left_tensor = Tensor::zeros({70, 64}, DType::BFloat16);
    Tensor right_tensor = random_tensor({96, 64});
    Tensor bias_tensor = random_tensor({96});
    Tensor output_tensor = Tensor::zeros({70, 96}, DType::BFloat16);
    // 2. the product
    fma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor);
    // 3. check every row against bias
    torch::Tensor expected_tensor = pytorch_copy_of(bias_tensor).unsqueeze(0).expand({70, 96}).contiguous();
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
}

static void add_onto_adds_to_what_output_holds(void)
{
    // Expect: left the identity {64, 64}, right random {64, 64}, output starting at random values: with add_onto
    // the output is old + rightᵀ, each rounded once to bfloat16, as PyTorch's (old + rightᵀ) rounded to bfloat16;
    // without it, rightᵀ alone (times_the_identity_gives_right_turned_round).
    // 1. the tensors
    torch::manual_seed(3);
    Tensor left_tensor = identity_bf16(64);
    Tensor right_tensor = random_tensor({64, 64});
    Tensor output_tensor = random_tensor({64, 64});
    torch::Tensor expected_tensor =
        (pytorch_copy_of(output_tensor) + pytorch_copy_of(right_tensor).t()).to(torch::kBFloat16).to(torch::kFloat);
    // 2. the product, added onto
    fma_matmul_bf16(output_tensor, left_tensor, right_tensor, Tensor(), true);
    // 3. check
    ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
}

static void turned_tensors_give_the_same_bits(void)
{
    // Expect: random L {136, 96} and R {200, 96}, and copies of them turned round (Lᵀ {96, 136}, Rᵀ {96, 200}):
    // the product of each pair of ways (plain or turned) is the same to the last bit. The turned copies hold the
    // same values, and each sum adds the same products in the same order of k; only where they are read from
    // changes.
    // 1. random L and R, their turned-round copies
    torch::manual_seed(4);
    Tensor left_tensor = random_tensor({136, 96});
    Tensor right_tensor = random_tensor({200, 96});
    Tensor left_turned_tensor = tensor_from_pytorch(pytorch_copy_of(left_tensor).t(), DType::BFloat16);
    Tensor right_turned_tensor = tensor_from_pytorch(pytorch_copy_of(right_tensor).t(), DType::BFloat16);
    // 2. the plain product
    Tensor plain_tensor = Tensor::zeros({136, 200}, DType::BFloat16);
    fma_matmul_bf16(plain_tensor, left_tensor, right_tensor);
    torch::Tensor expected_tensor = pytorch_copy_of(plain_tensor);
    // 3. each turned way against it
    for (int way_index_val = 1; way_index_val < 4; way_index_val++)
    {
        bool left_turned_val = (way_index_val & 1) != 0;
        bool right_turned_val = (way_index_val & 2) != 0;
        Tensor output_tensor = Tensor::zeros({136, 200}, DType::BFloat16);
        fma_matmul_bf16(output_tensor, left_turned_val ? left_turned_tensor : left_tensor, right_turned_val ? right_turned_tensor : right_tensor,
               Tensor(), false, left_turned_val, right_turned_val);
        print_test_name(__func__);
        printf("    left %s, right %s\n", left_turned_val ? "turned" : "plain", right_turned_val ? "turned" : "plain");
        ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
    }
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each of the kernels (one per turned flags, float and bfloat16) can be launched with FMA_MATMUL_BF16_THREADS_PER_BLOCK
    // threads, and nothing is spilled: cudaFuncGetAttributes' numRegs (registers per thread), maxThreadsPerBlock
    // (the largest block) and localSizeBytes (0 if nothing spills), read without running anything.
    const void *kernels_list[] = {(const void *)fma_matmul_bf16_kernel<float, false, false>,
                                  (const void *)fma_matmul_bf16_kernel<float, false, true>,
                                  (const void *)fma_matmul_bf16_kernel<float, true, false>,
                                  (const void *)fma_matmul_bf16_kernel<float, true, true>,
                                  (const void *)fma_matmul_bf16_kernel<nv_bfloat16, false, false>,
                                  (const void *)fma_matmul_bf16_kernel<nv_bfloat16, false, true>,
                                  (const void *)fma_matmul_bf16_kernel<nv_bfloat16, true, false>,
                                  (const void *)fma_matmul_bf16_kernel<nv_bfloat16, true, true>};
    const char *names_list[] = {"fma_matmul_bf16_kernel<float, false, false>",
                                "fma_matmul_bf16_kernel<float, false, true>",
                                "fma_matmul_bf16_kernel<float, true, false>",
                                "fma_matmul_bf16_kernel<float, true, true>",
                                "fma_matmul_bf16_kernel<nv_bfloat16, false, false>",
                                "fma_matmul_bf16_kernel<nv_bfloat16, false, true>",
                                "fma_matmul_bf16_kernel<nv_bfloat16, true, false>",
                                "fma_matmul_bf16_kernel<nv_bfloat16, true, true>"};
    const int kernel_count_val = (int)(sizeof(kernels_list) / sizeof(kernels_list[0]));
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory, %zu bytes shared memory\n",
               names_list[kernel_index_val], attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes, attributes_val.sharedSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= FMA_MATMUL_BF16_THREADS_PER_BLOCK, true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    matches_pytorch_on_gpt2_qkv();
    matches_pytorch_on_gpt2_attention_output();
    matches_pytorch_on_gpt2_mlp_up();
    matches_pytorch_on_gpt2_mlp_down();
    matches_pytorch_on_gpt2_head();
    matches_pytorch_on_gpt2_qkv_input_grad();
    matches_pytorch_on_gpt2_qkv_weight_grad();
    matches_pytorch_with_sizes_no_tile_divides();
    matches_pytorch_turned_with_sizes_no_tile_divides();
    matches_pytorch_in_float();

    times_the_identity_gives_right_turned_round();
    zero_left_gives_the_bias_in_every_row();
    add_onto_adds_to_what_output_holds();
    turned_tensors_give_the_same_bits();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
