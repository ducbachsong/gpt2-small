// test_wgmma_matmul_bf16.cu — tests for llmc/kernels/bf16/wgmma_matmul_bf16.cuh (wgmma_matmul_bf16): output = left · rightᵀ (+ bias) (+ output),
// in bfloat16 on an H100's tensor cores (WGMMA, fed by TMA). Every check prints its test case, input, expected and actual value, and ok; the first FAIL stops the
// program.
//
// Part 1 checks against PyTorch's float product of the same values: GPT-2 small's 5 linear layers, the
// backward's two products (left or right turned round), and sizes no square divides; part 2 checks numbers that
// must come out exactly, that a turned tensor gives the same bits as the plain one, that every setting gives the
// default's bits and every split of the sum PyTorch's product; part 3 checks that the
// kernels' registers fit their block. Run it on an H100 (compute capability 9.0), built with -arch=sm_90a in place of -arch=native (WGMMA and
// TMA are Hopper's own instructions). The timings are in tests/bf16/tune_wgmma_matmul_bf16.cu.
#include <torch/types.h>              // torch::randn, torch::matmul; before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/bf16/wgmma_matmul_bf16.cuh"

// Against PyTorch in float on the same bfloat16 values: the sums are float on both sides, added in another
// order (about 1e-6 of a value apart), then mine are rounded to bfloat16, by up to 2^-9 (0.2%) of each value.
// The tolerance is the largest difference, as a fraction of the largest value.
static const double BFLOAT16_TOLERANCE = 1e-2;

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

/// Runs wgmma_matmul_bf16 on random left (L {rows, sum_length}, or turned: Lᵀ {sum_length, rows}), right (R {columns,
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
    wgmma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor, false, left_turned_val, right_turned_val);
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
    wgmma_matmul_bf16(output_tensor, left_tensor, right_tensor);
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
    wgmma_matmul_bf16(output_tensor, left_tensor, right_tensor, bias_tensor);
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
    wgmma_matmul_bf16(output_tensor, left_tensor, right_tensor, Tensor(), true);
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
    wgmma_matmul_bf16(plain_tensor, left_tensor, right_tensor);
    torch::Tensor expected_tensor = pytorch_copy_of(plain_tensor);
    // 3. each turned way against it
    for (int way_index_val = 1; way_index_val < 4; way_index_val++)
    {
        bool left_turned_val = (way_index_val & 1) != 0;
        bool right_turned_val = (way_index_val & 2) != 0;
        Tensor output_tensor = Tensor::zeros({136, 200}, DType::BFloat16);
        wgmma_matmul_bf16(output_tensor, left_turned_val ? left_turned_tensor : left_tensor, right_turned_val ? right_turned_tensor : right_tensor,
               Tensor(), false, left_turned_val, right_turned_val);
        print_test_name(__func__);
        printf("    left %s, right %s\n", left_turned_val ? "turned" : "plain", right_turned_val ? "turned" : "plain");
        ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
    }
}

// ══ part 2b: every square's warpgroups, columns and sets of tiles ══════════════════════════

/// One product at COLUMNS, STAGES and ROW_GROUPS, the layout given at run time: what the checks below call for each setting.
///
///   output_tensor, left_tensor, right_tensor, left_turned_val, right_turned_val   as in wgmma_matmul_bf16
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void product_at(const Tensor &output_tensor, const Tensor &left_tensor, const Tensor &right_tensor, bool left_turned_val,
                       bool right_turned_val)
{
    wgmma_matmul_bf16_at<ROW_GROUPS, COLUMNS, STAGES>(output_tensor, left_tensor, right_tensor, Tensor(), false, left_turned_val,
                                                      right_turned_val, WgmmaMatmulBf16Store(), WGMMA_MATMUL_BF16_SPLITS,
                                                      WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP);
}

/// A setting of the product: its warpgroups (64 rows each), its square's columns, its sets of tiles, and the product
/// made for them.
struct ProductSetting
{
    int row_groups_val;
    int columns_val;
    int stages_val;
    void (*product_ptr)(const Tensor &, const Tensor &, const Tensor &, bool, bool);
};

static const ProductSetting PRODUCT_SETTINGS_LIST[] = {
    {1, 64, 1, product_at<64, 1, 1>}, {1, 64, 2, product_at<64, 2, 1>}, {1, 64, 3, product_at<64, 3, 1>}, {1, 64, 4, product_at<64, 4, 1>},
    {1, 128, 1, product_at<128, 1, 1>}, {1, 128, 2, product_at<128, 2, 1>}, {1, 128, 3, product_at<128, 3, 1>}, {1, 128, 4, product_at<128, 4, 1>},
    {1, 256, 1, product_at<256, 1, 1>}, {1, 256, 2, product_at<256, 2, 1>}, {1, 256, 3, product_at<256, 3, 1>}, {1, 256, 4, product_at<256, 4, 1>},
    {2, 64, 1, product_at<64, 1, 2>}, {2, 64, 2, product_at<64, 2, 2>}, {2, 64, 3, product_at<64, 3, 2>}, {2, 64, 4, product_at<64, 4, 2>},
    {2, 128, 1, product_at<128, 1, 2>}, {2, 128, 2, product_at<128, 2, 2>}, {2, 128, 3, product_at<128, 3, 2>}, {2, 128, 4, product_at<128, 4, 2>},
    {2, 256, 1, product_at<256, 1, 2>}, {2, 256, 2, product_at<256, 2, 2>}, {2, 256, 3, product_at<256, 3, 2>}, {2, 256, 4, product_at<256, 4, 2>},
};

/// For random L {rows, sum_length} and R {columns, sum_length}, each plain and turned round: every setting's product,
/// in each of the 4 ways (plain or turned on each side), is the default's to the last bit.
static void every_setting_gives_the_default_bits(const char *test_ptr, size_t rows_val, size_t columns_val, size_t sum_length_val)
{
    // 1. random L and R, their turned-round copies
    torch::manual_seed(8);
    Tensor left_tensor = random_tensor({rows_val, sum_length_val});
    Tensor right_tensor = random_tensor({columns_val, sum_length_val});
    Tensor left_turned_tensor = tensor_from_pytorch(pytorch_copy_of(left_tensor).t(), DType::BFloat16);
    Tensor right_turned_tensor = tensor_from_pytorch(pytorch_copy_of(right_tensor).t(), DType::BFloat16);
    print_test_name(test_ptr);
    printf("    rows %zu, columns %zu, sum_length %zu: %zu steps of %d values of k\n", rows_val, columns_val, sum_length_val,
           (size_t)CEIL_DIV(sum_length_val, (size_t)WGMMA_MATMUL_BF16_BLOCK_K), WGMMA_MATMUL_BF16_BLOCK_K);
    // 2. each way: the default's product, then every setting's against it
    for (int way_index_val = 0; way_index_val < 4; way_index_val++)
    {
        bool left_turned_val = (way_index_val & 1) != 0;
        bool right_turned_val = (way_index_val & 2) != 0;
        const Tensor &left_used_tensor = left_turned_val ? left_turned_tensor : left_tensor;
        const Tensor &right_used_tensor = right_turned_val ? right_turned_tensor : right_tensor;
        Tensor default_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
        wgmma_matmul_bf16(default_tensor, left_used_tensor, right_used_tensor, Tensor(), false, left_turned_val, right_turned_val);
        torch::Tensor expected_tensor = pytorch_copy_of(default_tensor);
        for (const ProductSetting &setting_val : PRODUCT_SETTINGS_LIST)
        {
            Tensor output_tensor = Tensor::zeros({rows_val, columns_val}, DType::BFloat16);
            output_tensor.fill_(7.0);
            setting_val.product_ptr(output_tensor, left_used_tensor, right_used_tensor, left_turned_val, right_turned_val);
            printf("    left %s, right %s, %d warpgroups, %d columns, %d sets of tiles:\n", left_turned_val ? "turned" : "plain",
                   right_turned_val ? "turned" : "plain", setting_val.row_groups_val, setting_val.columns_val, setting_val.stages_val);
            ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
        }
    }
}

static void every_setting_gives_the_default_bits_with_sizes_no_tile_divides(void)
{
    // Expect: L {136, 336}, R {520, 336}: 136 rows (squares of 64 or 128 rows: the last part-full), 520 columns (no
    // square of 64, 128 or 256 divides it), 336 values of k (5 whole steps and one of 16: the last step mostly past the
    // edge, where TMA gives 0s). Every warpgroups, columns and sets of tiles adds the same products in the same order of k, so
    // every way, plain or turned, is the default's product to the last bit; the 7s it started with all written over.
    // 6 steps go round the 4 sets of tiles more than once.
    every_setting_gives_the_default_bits(__func__, 136, 520, 336);
}

static void every_setting_gives_the_default_bits_with_fewer_steps_than_sets(void)
{
    // Expect: L {72, 64}, R {200, 64}: one step only, fewer than the sets of tiles, so the tiles asked for ahead stop
    // at the sum's end. Every setting's product is the default's to the last bit.
    every_setting_gives_the_default_bits(__func__, 72, 200, 64);
}

static void every_setting_matches_pytorch_on_gpt2_qkv_weight_grad(void)
{
    // Expect: the qkv layer's weight_grad, output_gradᵀ · x with output_grad {4096, 2304} and x {4096, 768} both
    // sitting turned round, 64 steps: at every setting within 1e-2 of PyTorch's float product.
    // 1. random values; PyTorch's product
    torch::manual_seed(9);
    Tensor left_tensor = random_tensor({GPT2_ROWS, GPT2_QKV_DIM});
    Tensor right_tensor = random_tensor({GPT2_ROWS, GPT2_EMBED_DIM});
    torch::Tensor expected_tensor = pytorch_copy_of(left_tensor).t().matmul(pytorch_copy_of(right_tensor));
    // 2. every setting against it
    print_test_name(__func__);
    for (const ProductSetting &setting_val : PRODUCT_SETTINGS_LIST)
    {
        Tensor output_tensor = Tensor::zeros({GPT2_QKV_DIM, GPT2_EMBED_DIM}, DType::BFloat16);
        setting_val.product_ptr(output_tensor, left_tensor, right_tensor, true, true);
        printf("    %d warpgroups, %d columns, %d sets of tiles:\n", setting_val.row_groups_val, setting_val.columns_val, setting_val.stages_val);
        ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
    }
}

static void every_block_order_gives_the_default_bits(void)
{
    // Expect: L {296, 336}, R {520, 336}: rows of squares of the default's rows (the last part-full), columns of
    // the default's columns. The blocks run in
    // groups of 1 (row after row), 2, 3, 4, 8 and 64 rows of squares, down a group first; the last group gets the rows
    // left (296 rows: 1 of 2, 2 of 3, 1 of 4 ...; a multiple of 8, as a turned left's last dim must be). The order changes only which block runs when, never a sum, so
    // every way, plain or turned, is the default order's product to the last bit, the 7s all written over.
    // 1. random L and R, their turned-round copies
    torch::manual_seed(10);
    Tensor left_tensor = random_tensor({296, 336});
    Tensor right_tensor = random_tensor({520, 336});
    Tensor left_turned_tensor = tensor_from_pytorch(pytorch_copy_of(left_tensor).t(), DType::BFloat16);
    Tensor right_turned_tensor = tensor_from_pytorch(pytorch_copy_of(right_tensor).t(), DType::BFloat16);
    print_test_name(__func__);
    // 2. each way: the default order's product, then every order's against it
    const int block_rows_per_group_list[] = {1, 2, 3, 4, 8, 64};
    for (int way_index_val = 0; way_index_val < 4; way_index_val++)
    {
        bool left_turned_val = (way_index_val & 1) != 0;
        bool right_turned_val = (way_index_val & 2) != 0;
        const Tensor &left_used_tensor = left_turned_val ? left_turned_tensor : left_tensor;
        const Tensor &right_used_tensor = right_turned_val ? right_turned_tensor : right_tensor;
        Tensor default_tensor = Tensor::zeros({296, 520}, DType::BFloat16);
        wgmma_matmul_bf16(default_tensor, left_used_tensor, right_used_tensor, Tensor(), false, left_turned_val, right_turned_val);
        torch::Tensor expected_tensor = pytorch_copy_of(default_tensor);
        for (int block_rows_per_group_val : block_rows_per_group_list)
        {
            Tensor output_tensor = Tensor::zeros({296, 520}, DType::BFloat16);
            output_tensor.fill_(7.0);
            wgmma_matmul_bf16_at<WGMMA_MATMUL_BF16_ROW_GROUPS, WGMMA_MATMUL_BF16_COLUMNS, WGMMA_MATMUL_BF16_STAGES>(
                output_tensor, left_used_tensor, right_used_tensor, Tensor(), false, left_turned_val, right_turned_val,
                WgmmaMatmulBf16Store(), WGMMA_MATMUL_BF16_SPLITS, block_rows_per_group_val);
            printf("    left %s, right %s, groups of %d rows of squares:\n", left_turned_val ? "turned" : "plain",
                   right_turned_val ? "turned" : "plain", block_rows_per_group_val);
            ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
        }
    }
}

// ══ part 2c: the sum split over blocks ════════════════════════════════════════

/// One product at the default square and sets of tiles, split in `splits_val`, with bias, add_onto and an
/// epilogue: what the split checks below call.
///
///   output_tensor ... epilogue_val   as in wgmma_matmul_bf16
///   splits_val                       blocks a square's sum is split over
///   sum_threads_per_block_val        the splits' sum's block size
template <typename EPILOGUE = WgmmaMatmulBf16Store>
static void split_product(const Tensor &output_tensor, const Tensor &left_tensor, const Tensor &right_tensor, const Tensor &bias_tensor,
                          bool add_onto_val, bool left_turned_val, bool right_turned_val, int splits_val,
                          const EPILOGUE &epilogue_val = EPILOGUE(),
                          int sum_threads_per_block_val = WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK)
{
    wgmma_matmul_bf16_at<WGMMA_MATMUL_BF16_ROW_GROUPS, WGMMA_MATMUL_BF16_COLUMNS, WGMMA_MATMUL_BF16_STAGES>(
        output_tensor, left_tensor, right_tensor, bias_tensor, add_onto_val, left_turned_val, right_turned_val, epilogue_val,
        splits_val, WGMMA_MATMUL_BF16_BLOCK_ROWS_PER_GROUP, sum_threads_per_block_val);
}

/// A test epilogue: each pair times 2, so a split product's epilogue can be seen to run once on every pair.
struct TimesTwoEpilogue
{
    /// Gives the pair doubled.
    ///
    ///   pair_val    two neighbouring results, in float
    ///   index_val   the first one's index in output (unused)
    __device__ __forceinline__ float2 operator()(float2 pair_val, size_t index_val) const
    {
        return make_float2(2.0f * pair_val.x, 2.0f * pair_val.y);
    }
};

static void every_split_count_matches_pytorch_with_bias_and_add_onto(void)
{
    // Expect: L {136, 336}, R {520, 336} (6 steps of 64 k, the last of 16), bias {520}, output starting at random
    // values and added onto, in each of the 4 ways (plain or turned): split in 1, 2, 3, 4, 6 and 8 (8 more than the
    // steps: one a step), each within 1e-2 of PyTorch's old + L · Rᵀ + bias in float. Split, the sums are added in
    // another order, so the bits may differ from 1 split's, but never by more than bfloat16's rounding.
    // 1. random L, R, bias and the old output; their turned-round copies
    torch::manual_seed(11);
    Tensor left_tensor = random_tensor({136, 336});
    Tensor right_tensor = random_tensor({520, 336});
    Tensor bias_tensor = random_tensor({520});
    Tensor old_tensor = random_tensor({136, 520});
    Tensor left_turned_tensor = tensor_from_pytorch(pytorch_copy_of(left_tensor).t(), DType::BFloat16);
    Tensor right_turned_tensor = tensor_from_pytorch(pytorch_copy_of(right_tensor).t(), DType::BFloat16);
    torch::Tensor expected_tensor = pytorch_copy_of(old_tensor) + pytorch_copy_of(left_tensor).matmul(pytorch_copy_of(right_tensor).t()) +
                                    pytorch_copy_of(bias_tensor);
    print_test_name(__func__);
    // 2. each way and split count against it
    const int splits_list[] = {1, 2, 3, 4, 6, 8};
    for (int way_index_val = 0; way_index_val < 4; way_index_val++)
        for (int splits_val : splits_list)
        {
            bool left_turned_val = (way_index_val & 1) != 0;
            bool right_turned_val = (way_index_val & 2) != 0;
            Tensor output_tensor = old_tensor.clone();
            split_product(output_tensor, left_turned_val ? left_turned_tensor : left_tensor, right_turned_val ? right_turned_tensor : right_tensor,
                          bias_tensor, true, left_turned_val, right_turned_val, splits_val);
            printf("    left %s, right %s, %d splits:\n", left_turned_val ? "turned" : "plain", right_turned_val ? "turned" : "plain", splits_val);
            ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
        }
}

static void every_split_count_matches_pytorch_on_gpt2_attention_output_weight_grad(void)
{
    // Expect: the attention output layer's weight_grad, output_gradᵀ · x with output_grad {4096, 768} and x {4096,
    // 768} both sitting turned round: 72 squares of 64 x 128 and 64 steps, the shape split-K is for. Split in 1, 2,
    // 4, 8, 16 and 64, each within 1e-2 of PyTorch's float product.
    // 1. random values; PyTorch's product
    torch::manual_seed(12);
    Tensor left_tensor = random_tensor({GPT2_ROWS, GPT2_EMBED_DIM});
    Tensor right_tensor = random_tensor({GPT2_ROWS, GPT2_EMBED_DIM});
    torch::Tensor expected_tensor = pytorch_copy_of(left_tensor).t().matmul(pytorch_copy_of(right_tensor));
    // 2. every split count against it
    print_test_name(__func__);
    const int splits_list[] = {1, 2, 4, 8, 16, 64};
    for (int splits_val : splits_list)
    {
        Tensor output_tensor = Tensor::zeros({GPT2_EMBED_DIM, GPT2_EMBED_DIM}, DType::BFloat16);
        split_product(output_tensor, left_tensor, right_tensor, Tensor(), false, true, true, splits_val);
        printf("    %d splits:\n", splits_val);
        ASSERT_NEAR(relative_difference_on_gpu(output_tensor, expected_tensor), 0.0, BFLOAT16_TOLERANCE);
    }
}

static void every_split_count_runs_the_epilogue_once_on_each_pair(void)
{
    // Expect: L {136, 336}, R {520, 336}, bias {520}: at each split count (1, 2, 3, 6), the product with an epilogue
    // that doubles each pair is 2 x the product without it, to the last bit (doubling a float and rounding is
    // rounding and doubling: bfloat16 has the same digits at twice the size). Once on each pair, after the bias:
    // never on a split's part alone, never twice.
    // 1. random L, R and bias
    torch::manual_seed(13);
    Tensor left_tensor = random_tensor({136, 336});
    Tensor right_tensor = random_tensor({520, 336});
    Tensor bias_tensor = random_tensor({520});
    print_test_name(__func__);
    // 2. each split count: with the epilogue against 2 x without it
    const int splits_list[] = {1, 2, 3, 6};
    for (int splits_val : splits_list)
    {
        Tensor plain_tensor = Tensor::zeros({136, 520}, DType::BFloat16);
        Tensor doubled_tensor = Tensor::zeros({136, 520}, DType::BFloat16);
        split_product(plain_tensor, left_tensor, right_tensor, bias_tensor, false, false, false, splits_val);
        split_product(doubled_tensor, left_tensor, right_tensor, bias_tensor, false, false, false, splits_val, TimesTwoEpilogue());
        printf("    %d splits:\n", splits_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(doubled_tensor), 2.0f * pytorch_copy_of(plain_tensor)), true);
    }
}

static void every_splits_sum_block_size_gives_the_same_bits(void)
{
    // Expect: L {136, 336}, R {520, 336}, bias {520}, split in 3: the splits' sum at 32, 64, 128, 256, 512 and 1024
    // threads a block gives the same bits as at the default (WGMMA_MATMUL_BF16_SPLITS_SUM_THREADS_PER_BLOCK): a
    // thread's 4 values are added in the same order of splits whatever block it is in.
    // 1. random L, R and bias; the default block size's product
    torch::manual_seed(14);
    Tensor left_tensor = random_tensor({136, 336});
    Tensor right_tensor = random_tensor({520, 336});
    Tensor bias_tensor = random_tensor({520});
    Tensor default_tensor = Tensor::zeros({136, 520}, DType::BFloat16);
    split_product(default_tensor, left_tensor, right_tensor, bias_tensor, false, false, false, 3);
    torch::Tensor expected_tensor = pytorch_copy_of(default_tensor);
    print_test_name(__func__);
    // 2. every block size against it
    const int threads_list[] = {32, 64, 128, 256, 512, 1024};
    for (int threads_per_block_val : threads_list)
    {
        Tensor output_tensor = Tensor::zeros({136, 520}, DType::BFloat16);
        output_tensor.fill_(7.0);
        split_product(output_tensor, left_tensor, right_tensor, bias_tensor, false, false, false, 3, WgmmaMatmulBf16Store(),
                      threads_per_block_val);
        printf("    %d threads a block:\n", threads_per_block_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(output_tensor), expected_tensor), true);
    }
}

// ══ part 3: the kernels fit their launch ══════════════════════════════════════

/// Adds the 4 kernels (one per turned flags) of one warpgroups, columns and sets of tiles to the lists, with their
/// names and block sizes.
template <int COLUMNS, int STAGES, int ROW_GROUPS>
static void add_kernels_of_setting(std::vector<const void *> &kernels_list, std::vector<std::string> &names_list,
                                   std::vector<int> &threads_list)
{
    kernels_list.push_back((const void *)wgmma_matmul_bf16_kernel<false, false, WgmmaMatmulBf16Store, COLUMNS, STAGES, ROW_GROUPS>);
    kernels_list.push_back((const void *)wgmma_matmul_bf16_kernel<false, true, WgmmaMatmulBf16Store, COLUMNS, STAGES, ROW_GROUPS>);
    kernels_list.push_back((const void *)wgmma_matmul_bf16_kernel<true, false, WgmmaMatmulBf16Store, COLUMNS, STAGES, ROW_GROUPS>);
    kernels_list.push_back((const void *)wgmma_matmul_bf16_kernel<true, true, WgmmaMatmulBf16Store, COLUMNS, STAGES, ROW_GROUPS>);
    const char *ways_list[] = {"false, false", "false, true", "true, false", "true, true"};
    for (const char *way_ptr : ways_list)
    {
        names_list.push_back(std::string("wgmma_matmul_bf16_kernel<") + way_ptr + ", " + std::to_string(ROW_GROUPS) + " warpgroups, " +
                             std::to_string(COLUMNS) + " columns, " + std::to_string(STAGES) + " sets>");
        threads_list.push_back(ROW_GROUPS * HARDWARE_WARPGROUP_THREADS);
    }
}

static void every_kernel_fits_its_block(void)
{
    // Expect: each of the kernels (one per turned flags, warpgroups, columns and sets of tiles: 96) can be launched
    // with its warpgroups' threads, and nothing is spilled, the 128 sums a thread of 256 columns included; the
    // splits' sum with up to HARDWARE_MAX_THREADS_PER_BLOCK, the largest it may be given:
    // cudaFuncGetAttributes' numRegs (registers per thread), maxThreadsPerBlock (the largest block) and
    // localSizeBytes (0 if nothing spills), read without running anything. The tiles are dynamic shared memory, so
    // sharedSizeBytes is only the barriers.
    std::vector<const void *> kernels_list;
    std::vector<std::string> names_list;
    std::vector<int> threads_list;
    add_kernels_of_setting<64, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 1, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 2, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 3, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 4, 1>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<64, 4, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<128, 4, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 1, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 2, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 3, 2>(kernels_list, names_list, threads_list);
    add_kernels_of_setting<256, 4, 2>(kernels_list, names_list, threads_list);
    kernels_list.push_back((const void *)wgmma_matmul_bf16_splits_sum_kernel<WgmmaMatmulBf16Store>);   // the splits' sum, any block size
    names_list.push_back("wgmma_matmul_bf16_splits_sum_kernel<WgmmaMatmulBf16Store>");
    threads_list.push_back(HARDWARE_MAX_THREADS_PER_BLOCK);
    for (size_t kernel_index_val = 0; kernel_index_val < kernels_list.size(); kernel_index_val++)
    {
        // 1. what the compiler wrote down
        cudaFuncAttributes attributes_val;
        cudaCheck(cudaFuncGetAttributes(&attributes_val, kernels_list[kernel_index_val]));
        print_test_name(__func__);
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory, %zu bytes shared memory\n",
               names_list[kernel_index_val].c_str(), attributes_val.numRegs, attributes_val.maxThreadsPerBlock,
               attributes_val.localSizeBytes, attributes_val.sharedSizeBytes);
        // 2. check
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
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

    times_the_identity_gives_right_turned_round();
    zero_left_gives_the_bias_in_every_row();
    add_onto_adds_to_what_output_holds();
    turned_tensors_give_the_same_bits();

    every_setting_gives_the_default_bits_with_sizes_no_tile_divides();
    every_setting_gives_the_default_bits_with_fewer_steps_than_sets();
    every_setting_matches_pytorch_on_gpt2_qkv_weight_grad();
    every_block_order_gives_the_default_bits();

    every_split_count_matches_pytorch_with_bias_and_add_onto();
    every_split_count_matches_pytorch_on_gpt2_attention_output_weight_grad();
    every_split_count_runs_the_epilogue_once_on_each_pair();
    every_splits_sum_block_size_gives_the_same_bits();

    every_kernel_fits_its_block();

    printf("\nall tests passed\n");
    return 0;
}
