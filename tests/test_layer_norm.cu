// test_layer_norm.cu — tests for llmc/kernels/layer_norm.cuh (layer_norm_forward, layer_norm_backward).
// Every check prints its test case, input, expected and actual value, and ok; the first FAIL
// stops the program.
//
// Part 1 checks both against PyTorch's layer_norm, with autograd for the backward; part 2 checks numbers
// worked out by hand; part 3 checks that each kernel's registers fit its block, and other block sizes. The
// timings are in tunes/tune_layer_norm.cu. (The C# trainer had no LayerNorm test: TorchSharp's was
// trusted.)
#include <torch/types.h>              // torch::layer_norm, backward(); libtorch before any CUDA header
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/kernels/layer_norm.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value
// In bfloat16, against PyTorch in float on the same values: a bfloat16 keeps 8 significant bits, so each
// rounding moves a value by up to 2^-8, about 0.4% of it. weight_grad and bias_grad are rounded once per
// block that adds onto them, 8 times here.
static const double BFLOAT16_TOLERANCE = 2e-2;
// Between block sizes in bfloat16: the backward always launches LAYER_NORM_BACKWARD_BLOCKS (80) blocks, and
// each block with rows rounds its sum into weight_grad and bias_grad once. 256 rows at 16 warps a block (the
// default) fill 16 blocks: 16 roundings; at one warp a block every one of the 80 has rows: 80 roundings, and
// the two land 2.6% apart on an H100 PCIe. A block that lost its rows would be far off.
static const double BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE = 5e-2;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes random values from Tensor::randn as a CPU list: the same numbers for both sides of a test.
static std::vector<float> random_values(const Shape &shape_list)
{
    return Tensor::randn(shape_list).tolist();
}

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
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

/// One LayerNorm's tensors: the inputs, the forward's results and the gradients, float or bfloat16 (mean
/// and rstd always float), the results and gradients all 0 at first.
struct LayerNormTensors
{
    Tensor input_tensor;
    Tensor weight_tensor;
    Tensor bias_tensor;
    Tensor output_grad_tensor;
    Tensor output_tensor;
    Tensor mean_tensor;
    Tensor rstd_tensor;
    Tensor input_grad_tensor;
    Tensor weight_grad_tensor;
    Tensor bias_grad_tensor;
};

/// Makes a LayerNorm's tensors of {batch_size, seq_len, embed_dim} from random values, in `dtype_val`.
static LayerNormTensors random_layer_norm(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val, DType dtype_val)
{
    Shape shape_list = {batch_size_val, seq_len_val, embed_dim_val};
    LayerNormTensors tensors_val;
    tensors_val.input_tensor = tensor_of(shape_list, random_values(shape_list)).to(dtype_val);
    tensors_val.weight_tensor = tensor_of({embed_dim_val}, random_values({embed_dim_val})).to(dtype_val);
    tensors_val.bias_tensor = tensor_of({embed_dim_val}, random_values({embed_dim_val})).to(dtype_val);
    tensors_val.output_grad_tensor = tensor_of(shape_list, random_values(shape_list)).to(dtype_val);
    tensors_val.output_tensor = Tensor::zeros(shape_list, dtype_val);
    tensors_val.mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    tensors_val.input_grad_tensor = Tensor::zeros(shape_list, dtype_val);
    tensors_val.weight_grad_tensor = Tensor::zeros({embed_dim_val}, dtype_val);
    tensors_val.bias_grad_tensor = Tensor::zeros({embed_dim_val}, dtype_val);
    return tensors_val;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

static void forward_matches_pytorch(void)
{
    // Expect: batch_size = 2 rows of seq_len = 8 tokens, embed_dim = 768 as in GPT-2 small (each
    // thread of a warp takes 6 of a row's 192 float4s), random input, weight and bias: output, mean
    // and rstd equal what PyTorch's native_layer_norm gives, within 1e-5. Not to the last bit: the
    // two add up a row's 768 values in a different order.
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(2, 8, 768, DType::Float32);
    layer_norm_forward(tensors_val.output_tensor, tensors_val.mean_tensor, tensors_val.rstd_tensor,
                       tensors_val.input_tensor, tensors_val.weight_tensor, tensors_val.bias_tensor);
    // 2. PyTorch's
    std::vector<int64_t> normalized_shape_list = {768};               // normalize over the last dim
    auto [pytorch_output_tensor, pytorch_mean_tensor, pytorch_rstd_tensor] =
        torch::native_layer_norm(pytorch_copy_of(tensors_val.input_tensor), normalized_shape_list,
                                 pytorch_copy_of(tensors_val.weight_tensor), pytorch_copy_of(tensors_val.bias_tensor), 1e-5);
    // 3. check
    ASSERT_NEAR(tensors_val.output_tensor.tolist(), list_of(pytorch_output_tensor), TOLERANCE);
    ASSERT_NEAR(tensors_val.mean_tensor.tolist(), list_of(pytorch_mean_tensor), TOLERANCE);
    ASSERT_NEAR(tensors_val.rstd_tensor.tolist(), list_of(pytorch_rstd_tensor), TOLERANCE);
}

static void backward_matches_pytorch(void)
{
    // Expect: batch_size = 4 rows of seq_len = 64 tokens, embed_dim = 768: 256 rows, 8 blocks of the
    // backward, so weight_grad and bias_grad get adds from 8 blocks. Random input, weight, bias and
    // output_grad, backward from 0: input_grad, weight_grad and bias_grad equal PyTorch's .grad after
    // autograd's backward, within 1e-5.
    // 1. random tensors; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(4, 64, 768, DType::Float32);
    layer_norm_forward(tensors_val.output_tensor, tensors_val.mean_tensor, tensors_val.rstd_tensor,
                       tensors_val.input_tensor, tensors_val.weight_tensor, tensors_val.bias_tensor);
    layer_norm_backward(tensors_val.input_grad_tensor, tensors_val.weight_grad_tensor, tensors_val.bias_grad_tensor,
                        tensors_val.output_grad_tensor, tensors_val.input_tensor, tensors_val.weight_tensor,
                        tensors_val.mean_tensor, tensors_val.rstd_tensor);
    // 2. PyTorch's, with autograd
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(tensors_val.input_tensor).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(tensors_val.weight_tensor).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = pytorch_copy_of(tensors_val.bias_tensor).requires_grad_(true);
    std::vector<int64_t> normalized_shape_list = {768};
    torch::Tensor pytorch_output_tensor =
        torch::layer_norm(pytorch_input_tensor, normalized_shape_list, pytorch_weight_tensor, pytorch_bias_tensor, 1e-5);
    pytorch_output_tensor.backward(pytorch_copy_of(tensors_val.output_grad_tensor));
    // 3. check
    ASSERT_NEAR(tensors_val.input_grad_tensor.tolist(), list_of(pytorch_input_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(tensors_val.weight_grad_tensor.tolist(), list_of(pytorch_weight_tensor.grad()), TOLERANCE);
    ASSERT_NEAR(tensors_val.bias_grad_tensor.tolist(), list_of(pytorch_bias_tensor.grad()), TOLERANCE);
}

static void forward_and_backward_in_bfloat16(void)
{
    // Expect: backward_matches_pytorch's 256 rows of 768 (8 blocks of the backward), with input, weight,
    // bias, output_grad and the gradients in bfloat16; mean and rstd stay float. Every sum is in float from
    // the same bfloat16 values PyTorch is given as floats, so mean and rstd are PyTorch's within 1e-5; output
    // and input_grad are rounded once as they are stored, weight_grad and bias_grad once per block: within
    // BFLOAT16_TOLERANCE.
    // 1. random tensors in bfloat16; mine
    Tensor::manual_seed(0);
    LayerNormTensors tensors_val = random_layer_norm(4, 64, 768, DType::BFloat16);
    layer_norm_forward(tensors_val.output_tensor, tensors_val.mean_tensor, tensors_val.rstd_tensor,
                       tensors_val.input_tensor, tensors_val.weight_tensor, tensors_val.bias_tensor);
    layer_norm_backward(tensors_val.input_grad_tensor, tensors_val.weight_grad_tensor, tensors_val.bias_grad_tensor,
                        tensors_val.output_grad_tensor, tensors_val.input_tensor, tensors_val.weight_tensor,
                        tensors_val.mean_tensor, tensors_val.rstd_tensor);
    // 2. PyTorch's, in float on the same values
    torch::Tensor pytorch_input_tensor = pytorch_copy_of(tensors_val.input_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_weight_tensor = pytorch_copy_of(tensors_val.weight_tensor.to(DType::Float32)).requires_grad_(true);
    torch::Tensor pytorch_bias_tensor = pytorch_copy_of(tensors_val.bias_tensor.to(DType::Float32)).requires_grad_(true);
    std::vector<int64_t> normalized_shape_list = {768};
    auto [pytorch_output_tensor, pytorch_mean_tensor, pytorch_rstd_tensor] = torch::native_layer_norm(
        pytorch_input_tensor, normalized_shape_list, pytorch_weight_tensor, pytorch_bias_tensor, 1e-5);
    pytorch_output_tensor.backward(pytorch_copy_of(tensors_val.output_grad_tensor.to(DType::Float32)));
    // 3. check: mean and rstd within TOLERANCE, the rest within one or a few roundings
    ASSERT_NEAR(tensors_val.mean_tensor.tolist(), list_of(pytorch_mean_tensor), TOLERANCE);
    ASSERT_NEAR(tensors_val.rstd_tensor.tolist(), list_of(pytorch_rstd_tensor), TOLERANCE);
    ASSERT_NEAR(tensors_val.output_tensor.to(DType::Float32).tolist(), list_of(pytorch_output_tensor), BFLOAT16_TOLERANCE);
    ASSERT_NEAR(tensors_val.input_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_input_tensor.grad()),
                BFLOAT16_TOLERANCE);
    ASSERT_NEAR(tensors_val.weight_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_weight_tensor.grad()),
                BFLOAT16_TOLERANCE);
    ASSERT_NEAR(tensors_val.bias_grad_tensor.to(DType::Float32).tolist(), list_of(pytorch_bias_tensor.grad()),
                BFLOAT16_TOLERANCE);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use rows of 4 (one float4: thread 0 of the warp has it all, threads 1 .. THREADS_PER_WARP - 1 have
// nothing to read). The row 1 2 3 4 works out as:
//
//     mean = 10 / 4 = 2.5
//     var  = (1.5² + 0.5² + 0.5² + 1.5²) / 4 = 5 / 4 = 1.25
//     rstd = 1 / sqrt(1.25 + 0.00001) = 0.8944236
//     norm = -1.5, -0.5, 0.5, 1.5 times rstd = -1.3416354, -0.4472118, 0.4472118, 1.3416354

static const std::vector<float> NORM_OF_1_2_3_4 = {-1.3416354f, -0.4472118f, 0.4472118f, 1.3416354f};

static void forward_moves_a_row_to_mean_0_and_standard_deviation_1(void)
{
    // Expect: the row 1 2 3 4, weight 1, bias 0: output is norm, -1.3416354 -0.4472118 0.4472118
    // 1.3416354 (mean 0, standard deviation 1); mean is 2.5 exactly, rstd 0.8944236.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, 4});
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, tensor_of({1, 1, 4}, {1, 2, 3, 4}), Tensor::ones({4}),
                       Tensor::zeros({4}));
    // 2. check
    ASSERT_NEAR(output_tensor.tolist(), NORM_OF_1_2_3_4, 1e-6);
    ASSERT_EQ(mean_tensor.tolist(), std::vector<float>({2.5f}));
    ASSERT_NEAR(rstd_tensor.tolist()[0], 0.8944236, 1e-6);
}

static void weight_and_bias_scale_and_shift_each_column(void)
{
    // Expect: the row 1 2 3 4, weight 2 for every column, bias 10 20 30 40: output = 2 x norm + bias
    // = 10 - 2.6832708, 20 - 0.8944236, 30 + 0.8944236, 40 + 2.6832708.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, 4});
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, tensor_of({1, 1, 4}, {1, 2, 3, 4}),
                       tensor_of({4}, {2, 2, 2, 2}), tensor_of({4}, {10, 20, 30, 40}));
    // 2. check
    ASSERT_NEAR(output_tensor.tolist(), std::vector<float>({7.3167292f, 19.1055764f, 30.8944236f, 42.6832708f}), 1e-6);
}

static void a_flat_row_gives_the_bias(void)
{
    // Expect: the row 5 5 5 5: mean 5, every x - mean is exactly 0, so output is the bias, 10 20 30
    // 40, exactly. The variance is 0 and rstd = 1 / sqrt(0 + 0.00001) = 316.22777: without the 1e-5
    // it would be 1 / 0 = infinity, and 0 x infinity = NaN.
    // 1. the forward
    Tensor output_tensor = Tensor::zeros({1, 1, 4});
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, tensor_of({1, 1, 4}, {5, 5, 5, 5}),
                       tensor_of({4}, {2, 2, 2, 2}), tensor_of({4}, {10, 20, 30, 40}));
    // 2. check
    ASSERT_EQ(output_tensor.tolist(), std::vector<float>({10, 20, 30, 40}));
    ASSERT_NEAR(rstd_tensor.tolist()[0], 316.22777, 1e-3);
}

static void each_row_is_normalized_on_its_own(void)
{
    // Expect: two rows, 1 2 3 4 and 103 106 109 112 (= 100 + 3 x the first). Each row gets its own
    // mean, 2.5 and 107.5, and its own rstd, so both come out as the same norm: shifting a row or
    // scaling it does not change its output. Within 1e-5, not exactly: the 1e-5 added to the
    // variance counts for less next to 11.25 than next to 1.25.
    // 1. the forward on two rows
    Tensor output_tensor = Tensor::zeros({1, 2, 4});
    Tensor mean_tensor = Tensor::zeros({1, 2});
    Tensor rstd_tensor = Tensor::zeros({1, 2});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, tensor_of({1, 2, 4}, {1, 2, 3, 4, 103, 106, 109, 112}),
                       Tensor::ones({4}), Tensor::zeros({4}));
    // 2. check
    ASSERT_EQ(mean_tensor.tolist(), std::vector<float>({2.5f, 107.5f}));
    ASSERT_NEAR(output_tensor[0][0].tolist(), NORM_OF_1_2_3_4, 1e-5);
    ASSERT_NEAR(output_tensor[0][1].tolist(), NORM_OF_1_2_3_4, 1e-5);
}

static void backward_adds_up_bias_grad_and_weight_grad_over_the_rows(void)
{
    // Expect: two rows, 1 2 3 4 and 4 3 2 1 (its norm is the first's backwards: -norm), weight 1,
    // output_grad rows of 1s and of 2s.
    //
    //     bias_grad   = 1 + 2                       = 3 3 3 3, exactly
    //     weight_grad = 1 x norm + 2 x (-norm) = -norm  = 1.3416354 0.4472118 -0.4472118 -1.3416354
    // 1. the forward
    Tensor input_tensor = tensor_of({1, 2, 4}, {1, 2, 3, 4, 4, 3, 2, 1});
    Tensor weight_tensor = Tensor::ones({4});
    Tensor output_tensor = Tensor::zeros({1, 2, 4});
    Tensor mean_tensor = Tensor::zeros({1, 2});
    Tensor rstd_tensor = Tensor::zeros({1, 2});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, Tensor::zeros({4}));
    // 2. the backward
    Tensor input_grad_tensor = Tensor::zeros({1, 2, 4});
    Tensor weight_grad_tensor = Tensor::zeros({4});
    Tensor bias_grad_tensor = Tensor::zeros({4});
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, tensor_of({1, 2, 4}, {1, 1, 1, 1, 2, 2, 2, 2}),
                        input_tensor, weight_tensor, mean_tensor, rstd_tensor);
    // 3. check
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({3, 3, 3, 3}));
    ASSERT_NEAR(weight_grad_tensor.tolist(), std::vector<float>({1.3416354f, 0.4472118f, -0.4472118f, -1.3416354f}), 1e-6);
}

static void the_same_output_grad_on_a_whole_row_gives_no_input_grad(void)
{
    // Expect: the row 1 2 3 4, weight 1, output_grad 5 5 5 5: input_grad is 0 (up to rounding).
    // Raising every output of a row by the same amount is something no change of the input can do,
    // since the forward subtracts the row's mean: norm_grad - mean(norm_grad) = 5 - 5 = 0, and
    // mean(norm_grad x norm) = 5 x mean(norm) = 0.
    // 1. the forward
    Tensor input_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor weight_tensor = Tensor::ones({4});
    Tensor output_tensor = Tensor::zeros({1, 1, 4});
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, Tensor::zeros({4}));
    // 2. the backward
    Tensor input_grad_tensor = Tensor::zeros({1, 1, 4});
    Tensor weight_grad_tensor = Tensor::zeros({4});
    Tensor bias_grad_tensor = Tensor::zeros({4});
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, tensor_of({1, 1, 4}, {5, 5, 5, 5}),
                        input_tensor, weight_tensor, mean_tensor, rstd_tensor);
    // 3. check
    ASSERT_NEAR(input_grad_tensor.tolist(), std::vector<float>(4, 0.0f), 1e-5);
}

static void backward_adds_onto_the_gradients_already_there(void)
{
    // Expect: every gradient starts at 10 (as if an earlier micro-batch, or for input_grad the
    // residual stream, had left it there). The row 1 2 3 4, weight 1, output_grad 1 1 1 1:
    // bias_grad becomes 11 exactly, weight_grad 10 + norm = 8.6583646 9.5527882 10.4472118 11.3416354,
    // input_grad 10 + 0 (as in the test above). A second backward adds again: bias_grad 12.
    // 1. the forward
    Tensor input_tensor = tensor_of({1, 1, 4}, {1, 2, 3, 4});
    Tensor weight_tensor = Tensor::ones({4});
    Tensor output_tensor = Tensor::zeros({1, 1, 4});
    Tensor mean_tensor = Tensor::zeros({1, 1});
    Tensor rstd_tensor = Tensor::zeros({1, 1});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, Tensor::zeros({4}));
    // 2. one backward onto gradients of 10: added onto
    Tensor output_grad_tensor = Tensor::ones({1, 1, 4});
    Tensor input_grad_tensor = Tensor::zeros({1, 1, 4}).fill_(10);
    Tensor weight_grad_tensor = Tensor::zeros({4}).fill_(10);
    Tensor bias_grad_tensor = Tensor::zeros({4}).fill_(10);
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                        weight_tensor, mean_tensor, rstd_tensor);
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({11, 11, 11, 11}));
    ASSERT_NEAR(weight_grad_tensor.tolist(), std::vector<float>({8.6583646f, 9.5527882f, 10.4472118f, 11.3416354f}), 1e-6);
    ASSERT_NEAR(input_grad_tensor.tolist(), std::vector<float>({10, 10, 10, 10}), 1e-6);
    // 3. a second backward: added again
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor, output_grad_tensor, input_tensor,
                        weight_tensor, mean_tensor, rstd_tensor);
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>({12, 12, 12, 12}));
}

static void many_rows_adding_onto_bias_grad_lose_nothing(void)
{
    // Expect: GPT-2 small's sizes, batch_size = 4 rows of seq_len = 1024 tokens, embed_dim = 768,
    // random input, weight 1, output_grad all 1. All 4096 rows add onto the same 768 values of
    // bias_grad: 51 or 52 rows into each block's shared copy, then 80 blocks into GPU memory, many at the
    // same moment. With atomicAdd no add is lost: 4096 x 1 = 4096 in every column, exactly (whole
    // numbers this small add exactly in floats). input_grad is 0 in every row, up to rounding, as
    // in the_same_output_grad_on_a_whole_row_gives_no_input_grad.
    // 1. the forward
    Tensor::manual_seed(0);
    size_t batch_size_val = 4;
    size_t seq_len_val = 1024;
    size_t embed_dim_val = 768;
    Tensor input_tensor = Tensor::randn({batch_size_val, seq_len_val, embed_dim_val});      // made on the CPU: a few seconds
    Tensor weight_tensor = Tensor::ones({embed_dim_val});
    Tensor output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor mean_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    Tensor rstd_tensor = Tensor::zeros({batch_size_val, seq_len_val});
    layer_norm_forward(output_tensor, mean_tensor, rstd_tensor, input_tensor, weight_tensor, Tensor::zeros({embed_dim_val}));
    // 2. the backward with output_grad all 1
    Tensor input_grad_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
    Tensor weight_grad_tensor = Tensor::zeros({embed_dim_val});
    Tensor bias_grad_tensor = Tensor::zeros({embed_dim_val});
    layer_norm_backward(input_grad_tensor, weight_grad_tensor, bias_grad_tensor,
                        Tensor::ones({batch_size_val, seq_len_val, embed_dim_val}), input_tensor, weight_tensor, mean_tensor,
                        rstd_tensor);
    // 3. check
    ASSERT_EQ(bias_grad_tensor.tolist(), std::vector<float>(embed_dim_val, 4096.0f));
    ASSERT_NEAR(input_grad_tensor.tolist(), std::vector<float>(batch_size_val * seq_len_val * embed_dim_val, 0.0f), 1e-5);
}

// ══ part 3: the kernels fit their launch, and other block sizes work ═════════

static void every_kernel_fits_its_block(void)
{
    // Expect: the forward can be launched with LAYER_NORM_THREADS_PER_BLOCK (512) and the backward
    // with LAYER_NORM_BACKWARD_THREADS_PER_BLOCK (512), and nothing is spilled. The compiler picks
    // each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote down, without
    // running anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const void *kernels_list[] = {(const void *)layer_norm_forward_kernel<float>,
                                  (const void *)layer_norm_backward_kernel<float>};
    const char *names_list[] = {"layer_norm_forward_kernel", "layer_norm_backward_kernel"};
    int threads_list[] = {LAYER_NORM_THREADS_PER_BLOCK, LAYER_NORM_BACKWARD_THREADS_PER_BLOCK};
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
        ASSERT_EQ(attributes_val.maxThreadsPerBlock >= threads_list[kernel_index_val], true);
        ASSERT_EQ(attributes_val.localSizeBytes, (size_t)0);
    }
}

static void other_block_sizes_give_the_same_result(void)
{
    // Expect: 256 random rows of 768, in float and in bfloat16, both launchers called with threads_per_block
    // 32, 128, 256 and 512 (one warp up; not 1024: the kernels' 66 registers per thread allow at most 896, see
    // tunes/tune_layer_norm.cu). Each row is one warp's whatever the block size, so output, mean, rstd
    // and input_grad are the defaults' to the last bit. weight_grad and bias_grad are each block's shared sums
    // added onto GPU memory in whatever order the blocks finish: within TOLERANCE in float,
    // BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE in bfloat16 (more blocks with rows, more roundings).
    const int block_sizes_list[] = {THREADS_PER_WARP, 4 * THREADS_PER_WARP, 8 * THREADS_PER_WARP, 16 * THREADS_PER_WARP};
    DType dtypes_list[] = {DType::Float32, DType::BFloat16};
    for (DType dtype_val : dtypes_list)
    {
        // 1. random tensors; the defaults' results
        Tensor::manual_seed(0);
        double tolerance_val = dtype_val == DType::BFloat16 ? BLOCK_SIZES_BFLOAT16_GRAD_TOLERANCE : TOLERANCE;
        LayerNormTensors usual_val = random_layer_norm(4, 64, 768, dtype_val);
        layer_norm_forward(usual_val.output_tensor, usual_val.mean_tensor, usual_val.rstd_tensor, usual_val.input_tensor,
                           usual_val.weight_tensor, usual_val.bias_tensor);
        layer_norm_backward(usual_val.input_grad_tensor, usual_val.weight_grad_tensor, usual_val.bias_grad_tensor,
                            usual_val.output_grad_tensor, usual_val.input_tensor, usual_val.weight_tensor,
                            usual_val.mean_tensor, usual_val.rstd_tensor);

        // 2. each block size, from the same inputs, against the defaults'
        for (int threads_per_block_val : block_sizes_list)
        {
            LayerNormTensors tensors_val = usual_val;              // the same inputs; results of its own
            tensors_val.output_tensor = Tensor::zeros(usual_val.output_tensor.shape(), dtype_val);
            tensors_val.mean_tensor = Tensor::zeros(usual_val.mean_tensor.shape());
            tensors_val.rstd_tensor = Tensor::zeros(usual_val.rstd_tensor.shape());
            tensors_val.input_grad_tensor = Tensor::zeros(usual_val.input_grad_tensor.shape(), dtype_val);
            tensors_val.weight_grad_tensor = Tensor::zeros(usual_val.weight_grad_tensor.shape(), dtype_val);
            tensors_val.bias_grad_tensor = Tensor::zeros(usual_val.bias_grad_tensor.shape(), dtype_val);
            layer_norm_forward(tensors_val.output_tensor, tensors_val.mean_tensor, tensors_val.rstd_tensor,
                               tensors_val.input_tensor, tensors_val.weight_tensor, tensors_val.bias_tensor,
                               threads_per_block_val);
            layer_norm_backward(tensors_val.input_grad_tensor, tensors_val.weight_grad_tensor, tensors_val.bias_grad_tensor,
                                tensors_val.output_grad_tensor, tensors_val.input_tensor, tensors_val.weight_tensor,
                                tensors_val.mean_tensor, tensors_val.rstd_tensor, threads_per_block_val);
            ASSERT_EQ(tensors_val.output_tensor.to(DType::Float32).tolist(), usual_val.output_tensor.to(DType::Float32).tolist());
            ASSERT_EQ(tensors_val.mean_tensor.tolist(), usual_val.mean_tensor.tolist());
            ASSERT_EQ(tensors_val.rstd_tensor.tolist(), usual_val.rstd_tensor.tolist());
            ASSERT_EQ(tensors_val.input_grad_tensor.to(DType::Float32).tolist(),
                      usual_val.input_grad_tensor.to(DType::Float32).tolist());
            ASSERT_NEAR(tensors_val.weight_grad_tensor.to(DType::Float32).tolist(),
                        usual_val.weight_grad_tensor.to(DType::Float32).tolist(), tolerance_val);
            ASSERT_NEAR(tensors_val.bias_grad_tensor.to(DType::Float32).tolist(),
                        usual_val.bias_grad_tensor.to(DType::Float32).tolist(), tolerance_val);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    forward_matches_pytorch();
    backward_matches_pytorch();
    forward_and_backward_in_bfloat16();

    forward_moves_a_row_to_mean_0_and_standard_deviation_1();
    weight_and_bias_scale_and_shift_each_column();
    a_flat_row_gives_the_bias();
    each_row_is_normalized_on_its_own();
    backward_adds_up_bias_grad_and_weight_grad_over_the_rows();
    the_same_output_grad_on_a_whole_row_gives_no_input_grad();
    backward_adds_onto_the_gradients_already_there();
    many_rows_adding_onto_bias_grad_lose_nothing();

    every_kernel_fits_its_block();
    other_block_sizes_give_the_same_result();

    printf("\nall tests passed\n");
    return 0;
}
