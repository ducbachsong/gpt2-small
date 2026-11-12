// test_attention.cu — tests for llmc/kernels/fp32/attention.cuh: float flash attention (attention_flash_forward,
// attention_flash_backward). Every check prints its test case, input, expected and actual value, and ok; the
// first FAIL stops the program.
//
// Part 1 checks both against PyTorch's scaled_dot_product_attention (causal), with autograd for the backward,
// and logsumexp against PyTorch's logsumexp of the masked scores; part 2 checks numbers worked out by hand and
// what the mask must give; part 3 checks that each kernel's registers fit its block, and other block sizes.
// Flash attention takes head_size ATTENTION_FLASH_HEAD_SIZE (64) and seq_len a multiple of ATTENTION_FLASH_TILE
// (32), so every test is of those sizes. bfloat16 (its steps and its flash attention):
// tests/bf16/test_attention_bf16.cu; 8-bit: tests/test_attention_8bit.cu. The timings are in tunes/tune_attention.cu.
// (The C# trainer had no attention test: TorchSharp's was trusted.)
#include <torch/types.h>              // torch::scaled_dot_product_attention, torch::randn, backward(); before CUDA headers
#include <torch/utils.h>              // torch::manual_seed
#include "../test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../../llmc/kernels/fp32/attention.cuh"

static const double TOLERANCE = 1e-5; // largest difference from PyTorch, as a fraction of the largest value

#define GPT2_NUM_HEADS 12             // GPT-2 small's heads
#define GPT2_SEQ_LEN 1024             // its tokens per row
#define HEAD_SIZE ATTENTION_FLASH_HEAD_SIZE   // 64, the only head_size flash attention takes
#define TILE ATTENTION_FLASH_TILE             // 32, the fewest tokens it takes

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a tensor of `shape_list` on the GPU holding `values_list`, given in row order.
static Tensor tensor_of(const Shape &shape_list, const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Makes a tensor of `shape_list` with random values, mean 0 and standard deviation 1, made by PyTorch on
/// the GPU: Tensor::randn makes them on the CPU, which takes seconds for GPT-2's 9.4M qkv values.
static Tensor random_tensor_on_gpu(const Shape &shape_list)
{
    Tensor tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(tensor.data(), values_tensor.data_ptr<float>(), tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return tensor;
}

/// Makes a PyTorch float tensor on the GPU with a copy of my tensor's values, GPU to GPU.
static torch::Tensor pytorch_copy_of(const Tensor &tensor)
{
    torch::TensorOptions on_gpu_val = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    std::vector<int64_t> sizes_list(tensor.shape().begin(), tensor.shape().end()); // PyTorch takes int64_t sizes
    return torch::from_blob(tensor.data(), sizes_list, on_gpu_val).clone();
}

/// Gives the largest difference between my tensor and PyTorch's, as a fraction of PyTorch's largest value,
/// worked out on the GPU: GPT-2's qkv_grad is 9.4M values, too many to copy to the CPU and print.
static double relative_difference_on_gpu(const Tensor &mine_tensor, const torch::Tensor &pytorch_tensor)
{
    torch::Tensor difference_tensor = (pytorch_copy_of(mine_tensor).reshape(pytorch_tensor.sizes()) - pytorch_tensor).abs().max();
    return difference_tensor.item<double>() / pytorch_tensor.abs().max().item<double>();
}

/// Every tensor attention_flash_forward and attention_flash_backward take, for one size, all float, made with 0s
/// except qkv and output_grad, which the test fills.
struct FlashTensors
{
    // forward
    Tensor qkv_tensor;
    Tensor output_tensor;
    Tensor logsumexp_tensor;
    // backward
    Tensor output_grad_tensor;
    Tensor qkv_grad_tensor;
    Tensor row_dot_tensor;

    /// Makes them for batch_size x num_heads heads of seq_len tokens of HEAD_SIZE columns.
    FlashTensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
    {
        size_t embed_dim_val = num_heads_val * HEAD_SIZE;
        qkv_tensor = Tensor::zeros({batch_size_val, seq_len_val, 3 * embed_dim_val});
        output_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val});
        logsumexp_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
        output_grad_tensor = Tensor::zeros(output_tensor.shape());
        qkv_grad_tensor = Tensor::zeros(qkv_tensor.shape());
        row_dot_tensor = Tensor::zeros({batch_size_val, num_heads_val, seq_len_val});
    }

    /// Runs attention_flash_forward on these tensors.
    void forward()
    {
        attention_flash_forward(output_tensor, logsumexp_tensor, qkv_tensor);
    }

    /// Runs attention_flash_backward on these tensors, row_dot at this block size (the default if not given).
    void backward(int threads_per_block_val = ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK)
    {
        attention_flash_backward(qkv_grad_tensor, output_grad_tensor, output_tensor, qkv_tensor, logsumexp_tensor,
                                 row_dot_tensor, threads_per_block_val);
    }
};

/// PyTorch's answers for one qkv and output_grad, the plain PyTorch way, as GPT-2's model.py does it: qkv
/// split into q, k, v, each cut into heads, then scaled_dot_product_attention with is_causal, and autograd.
/// logsumexp is worked out on the side: logsumexp of each row of masked scores.
struct PytorchAttention
{
    torch::Tensor qkv_tensor;
    torch::Tensor output_tensor;
    torch::Tensor logsumexp_tensor;

    /// Works out output, qkv.grad() and logsumexp from my qkv and output_grad.
    PytorchAttention(const FlashTensors &mine_val)
    {
        // 1. the sizes
        int64_t batch_size_val = mine_val.logsumexp_tensor.size(0);
        int64_t num_heads_val = mine_val.logsumexp_tensor.size(1);
        int64_t seq_len_val = mine_val.logsumexp_tensor.size(2);
        int64_t embed_dim_val = num_heads_val * HEAD_SIZE;
        // 2. qkv split and cut into heads, then the causal attention and its backward
        qkv_tensor = pytorch_copy_of(mine_val.qkv_tensor).requires_grad_(true);
        std::vector<torch::Tensor> parts_list = qkv_tensor.split(embed_dim_val, 2);
        auto heads_val = [&](const torch::Tensor &part_tensor) {
            return part_tensor.view({batch_size_val, seq_len_val, num_heads_val, (int64_t)HEAD_SIZE}).transpose(1, 2);
        };
        torch::Tensor q_tensor = heads_val(parts_list[0]);
        torch::Tensor k_tensor = heads_val(parts_list[1]);
        torch::Tensor v_tensor = heads_val(parts_list[2]);
        torch::Tensor y_tensor = torch::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor, {}, 0.0, true);
        output_tensor = y_tensor.transpose(1, 2).contiguous().view({batch_size_val, seq_len_val, embed_dim_val});
        output_tensor.backward(pytorch_copy_of(mine_val.output_grad_tensor));
        // 3. logsumexp, on the side
        torch::NoGradGuard no_grad_val;
        torch::Tensor scores_tensor = q_tensor.matmul(k_tensor.transpose(-2, -1)) / std::sqrt((double)HEAD_SIZE);
        torch::Tensor after_tensor = torch::ones({seq_len_val, seq_len_val}, scores_tensor.options().dtype(torch::kBool)).triu(1);
        logsumexp_tensor = scores_tensor.masked_fill(after_tensor, -INFINITY).logsumexp(-1);
    }
};

/// Makes FlashTensors for {batch_size, num_heads, seq_len} with random qkv and output_grad.
static FlashTensors random_flash_tensors(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
{
    FlashTensors tensors_val(batch_size_val, num_heads_val, seq_len_val);
    tensors_val.qkv_tensor = random_tensor_on_gpu(tensors_val.qkv_tensor.shape());
    tensors_val.output_grad_tensor = random_tensor_on_gpu(tensors_val.output_tensor.shape());
    return tensors_val;
}

// ══ part 1: mine against PyTorch's ═══════════════════════════════════════════

/// Runs one forward and one backward on random qkv and output_grad, mine and PyTorch's, and checks that
/// output, logsumexp and qkv_grad agree within TOLERANCE.
static void forward_and_backward_match_pytorch(size_t batch_size_val, size_t num_heads_val, size_t seq_len_val)
{
    // 1. random qkv and output_grad; mine
    torch::manual_seed(0);
    FlashTensors mine_val = random_flash_tensors(batch_size_val, num_heads_val, seq_len_val);
    mine_val.forward();
    mine_val.backward();
    // 2. PyTorch's
    PytorchAttention pytorch_val(mine_val);
    // 3. check
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.output_tensor, pytorch_val.output_tensor), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.logsumexp_tensor, pytorch_val.logsumexp_tensor), 0.0, TOLERANCE);
    ASSERT_NEAR(relative_difference_on_gpu(mine_val.qkv_grad_tensor, pytorch_val.qkv_tensor.grad()), 0.0, TOLERANCE);
}

static void needs_head_size_64_and_whole_tiles(void)
{
    // Expect: attention_flash_fits says no to a seq_len that is not a multiple of 32 (1000 = 31 x 32 + 8) and to
    // a head_size other than 64 (32); yes to 1024 and 64, and to 96 = 3 tiles, on any GPU.
    ASSERT_EQ(attention_flash_fits(1000, HEAD_SIZE), false);
    ASSERT_EQ(attention_flash_fits(GPT2_SEQ_LEN, HEAD_SIZE / 2), false);
    ASSERT_EQ(attention_flash_fits(GPT2_SEQ_LEN, HEAD_SIZE), true);
    ASSERT_EQ(attention_flash_fits(3 * TILE, HEAD_SIZE), true);
}

static void matches_pytorch_on_one_tile(void)
{
    // Expect: {1, 2, 32}: one tile of 32 queries and 32 keys, all on the diagonal, so every score goes through the
    // mask. output, logsumexp and qkv_grad within 1e-5 of PyTorch's.
    forward_and_backward_match_pytorch(1, 2, TILE);
}

static void matches_pytorch_with_3_batch_rows_of_96_tokens(void)
{
    // Expect: {3, 2, 96}: 3 tiles per head and 3 batch rows, so each block finds its batch row and head from its
    // number, and the key blocks walk 3, 2, 1 query tiles. The same.
    forward_and_backward_match_pytorch(3, 2, 3 * TILE);
}

static void matches_pytorch_on_gpt2_heads_of_256_tokens(void)
{
    // Expect: {2, 12, 256}: 8 tiles per head, so tiles below the diagonal (no mask), the online softmax's max
    // moving from tile to tile, the backward's key blocks walking 8, 7, ..., 1 query tiles. The same.
    forward_and_backward_match_pytorch(2, GPT2_NUM_HEADS, 8 * TILE);
}

static void matches_pytorch_on_gpt2_small_size(void)
{
    // Expect: {4, 12, 1024}, GPT-2 small's size: 32 tiles per head, 1,536 blocks; each row's sum runs over up to
    // 1024 keys in 32 tiles. The same.
    forward_and_backward_match_pytorch(4, GPT2_NUM_HEADS, GPT2_SEQ_LEN);
}

// ══ part 2: numbers worked out by hand ════════════════════════════════════════
// Most tests here use the note's example: one head of 3 tokens with head_size 4, scale 1 / sqrt(4) = 0.5:
//
//     token   q            k            v            scores (0.5 q · k), columns 0 .. token   attention_probs
//     0       2 0 0 0      1 0 0 0      1 0 0 0      1                                        1
//     1       0 2 0 0      0 1 0 0      0 1 0 0      0  1                                     0.2689414  0.7310586
//     2       2 2 0 0      1 1 0 0      0 0 1 0      1  1  2                                  0.2119416  0.2119416  0.5761169
//
// Token 1: e^0 / (e^0 + e^1) = 1 / 3.7182818 = 0.2689414. Each v has one 1, so output row t is attention_probs
// row t followed by 0s. With output_grad 1 0 0 0 in every row (the loss wants output column 0 up: the
// attention_prob on token 0):
//
//     da = output_grad · vᵀ        1 0 0 in every row
//     ds = a (da - row_dot)        token 1: 0.2689414 x (1 - 0.2689414) = 0.1966119, and -0.1966119
//                                  token 2: 0.1670223  -0.0449192  -0.1221031         (token 0: 0)
//     dq = 0.5 ds · k,  dk = 0.5 dsᵀ · q,  dv = aᵀ · output_grad
//
// Flash attention takes head_size 64 and 32 tokens, so the example sits in a bigger one: each q, k, v in the
// first 4 of its 64 columns, the other 60 columns 0, and tokens 3 .. 31 all 0 with output_grad 0. Tokens 0 .. 2
// cannot see the later ones, and the later ones send nothing back (their output_grad is 0), so rows 0 .. 2 are
// the note's. The scale is now 1 / sqrt(64) = 1 / 8, so q is given 4 times larger: q · k / 8 = (4 q) · k / 8 =
// the note's 0.5 q · k, the same scores. Then dk = dsᵀ · (4 q) / 8 is the note's, and dq = ds · k / 8 is the
// note's divided by 4.

#define NOTE_TOKENS 3                 // the note's example
#define NOTE_COLUMNS 4                // its head_size
#define NOTE_Q_SCALE 4.0f             // sqrt(64) / sqrt(4): the same scores with flash attention's scale

static const float NOTE_Q[NOTE_TOKENS][NOTE_COLUMNS] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {2, 2, 0, 0}};
static const float NOTE_K[NOTE_TOKENS][NOTE_COLUMNS] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}};
static const float NOTE_V[NOTE_TOKENS][NOTE_COLUMNS] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
static const float NOTE_ATTENTION_PROBS[NOTE_TOKENS][NOTE_TOKENS] = {
    {1, 0, 0}, {0.2689414f, 0.7310586f, 0}, {0.2119416f, 0.2119416f, 0.5761169f}};

/// Makes the note's example inside one head of TILE tokens: q (times NOTE_Q_SCALE), k, v in the first
/// NOTE_COLUMNS columns of tokens 0 .. 2, everything else 0; output_grad 1 0 0 ... in rows 0 .. 2, 0 after.
static FlashTensors note_example(void)
{
    FlashTensors note_val(1, 1, TILE);
    std::vector<float> qkv_list(TILE * 3 * HEAD_SIZE, 0.0f);
    std::vector<float> output_grad_list(TILE * HEAD_SIZE, 0.0f);
    for (int token_index_val = 0; token_index_val < NOTE_TOKENS; token_index_val++)
    {
        float *row_ptr = qkv_list.data() + token_index_val * 3 * HEAD_SIZE;        // q | k | v, HEAD_SIZE each
        for (int column_index_val = 0; column_index_val < NOTE_COLUMNS; column_index_val++)
        {
            row_ptr[column_index_val] = NOTE_Q_SCALE * NOTE_Q[token_index_val][column_index_val];
            row_ptr[HEAD_SIZE + column_index_val] = NOTE_K[token_index_val][column_index_val];
            row_ptr[2 * HEAD_SIZE + column_index_val] = NOTE_V[token_index_val][column_index_val];
        }
        output_grad_list[token_index_val * HEAD_SIZE] = 1.0f;
    }
    note_val.qkv_tensor = tensor_of(note_val.qkv_tensor.shape(), qkv_list);
    note_val.output_grad_tensor = tensor_of(note_val.output_tensor.shape(), output_grad_list);
    return note_val;
}

static void forward_of_the_note_s_example(void)
{
    // Expect: output rows 0 .. 2 = the note's attention_probs row t, then 0s to 64 columns, within 1e-6; and
    // logsumexp = log of each row's sum of e^score: token 0 log(e^1) = 1, token 1 log(e^0 + e^1) = 1.3132617,
    // token 2 log(2 e^1 + e^2) = log(12.8256198) = 2.5514448.
    // 1. the forward
    FlashTensors note_val = note_example();
    note_val.forward();
    // 2. the expected rows 0 .. 2
    std::vector<float> expected_list(NOTE_TOKENS * HEAD_SIZE, 0.0f);
    for (int token_index_val = 0; token_index_val < NOTE_TOKENS; token_index_val++)
        for (int column_index_val = 0; column_index_val < NOTE_TOKENS; column_index_val++)
            expected_list[token_index_val * HEAD_SIZE + column_index_val] = NOTE_ATTENTION_PROBS[token_index_val][column_index_val];
    // 3. check
    std::vector<float> output_list = note_val.output_tensor.tolist();
    std::vector<float> logsumexp_list = note_val.logsumexp_tensor.tolist();
    ASSERT_NEAR(std::vector<float>(output_list.begin(), output_list.begin() + NOTE_TOKENS * HEAD_SIZE), expected_list, 1e-6);
    ASSERT_NEAR(std::vector<float>(logsumexp_list.begin(), logsumexp_list.begin() + NOTE_TOKENS),
                std::vector<float>({1.0f, 1.3132617f, 2.5514448f}), 1e-6);
}

static void backward_of_the_note_s_example(void)
{
    // Expect: qkv_grad row t = dq | dk | dv of token t, worked out above (dq divided by NOTE_Q_SCALE), in the
    // first 2 columns of each part; every other value 0. Within 1e-6:
    //
    //     token   dq                       dk                          dv
    //     0       0          0             0.1670223  0.3636343        1.4808830  0
    //     1       0.0245765  -0.0245765    -0.0449192 -0.2415312       0.9430001  0
    //     2       0.0056149  -0.0208778    -0.1221031 -0.1221031       0.5761169  0
    //
    // dv of token 0 is 1 + 0.2689414 + 0.2119416: every token's attention_prob on token 0.
    // 1. the forward, then the backward
    FlashTensors note_val = note_example();
    note_val.forward();
    note_val.backward();
    // 2. the expected qkv_grad
    const float dq_list[NOTE_TOKENS][2] = {{0, 0}, {0.0245765f, -0.0245765f}, {0.0056149f, -0.0208778f}};
    const float dk_list[NOTE_TOKENS][2] = {{0.1670223f, 0.3636343f}, {-0.0449192f, -0.2415312f}, {-0.1221031f, -0.1221031f}};
    const float dv_list[NOTE_TOKENS] = {1.4808830f, 0.9430001f, 0.5761169f};
    std::vector<float> expected_list(TILE * 3 * HEAD_SIZE, 0.0f);
    for (int token_index_val = 0; token_index_val < NOTE_TOKENS; token_index_val++)
    {
        float *row_ptr = expected_list.data() + token_index_val * 3 * HEAD_SIZE;
        for (int column_index_val = 0; column_index_val < 2; column_index_val++)
        {
            row_ptr[column_index_val] = dq_list[token_index_val][column_index_val];
            row_ptr[HEAD_SIZE + column_index_val] = dk_list[token_index_val][column_index_val];
        }
        row_ptr[2 * HEAD_SIZE] = dv_list[token_index_val];
    }
    // 3. check
    ASSERT_NEAR(note_val.qkv_grad_tensor.tolist(), expected_list, 1e-6);
}

static void the_first_token_sees_only_itself(void)
{
    // Expect: token 0 sees key 0 only, so its attention_prob is exp(0) / exp(0) = 1 exactly, and its output is
    // its own v, exactly: 1 x v / 1. Random qkv, {1, 2, 32}: output row 0 = v of token 0 for both heads, qkv
    // row 0's last 128 values.
    // 1. the forward on random qkv
    torch::manual_seed(2);
    FlashTensors tensors_val = random_flash_tensors(1, 2, TILE);
    tensors_val.forward();
    // 2. check output row 0 against qkv row 0's v
    std::vector<float> qkv_list = tensors_val.qkv_tensor.tolist();
    std::vector<float> output_list = tensors_val.output_tensor.tolist();
    ASSERT_EQ(std::vector<float>(output_list.begin(), output_list.begin() + 2 * HEAD_SIZE),
              std::vector<float>(qkv_list.begin() + 4 * HEAD_SIZE, qkv_list.begin() + 6 * HEAD_SIZE));
}

static void a_later_token_does_not_change_earlier_outputs(void)
{
    // Expect: random qkv, {1, 2, 32}; then token 31's whole qkv row is set to 100 and the forward run again.
    // Tokens 0 .. 30 cannot see token 31: its scores are masked out of their rows before any sum, so their
    // output rows are the same to the last bit.
    // 1. the forward on random qkv
    torch::manual_seed(3);
    FlashTensors tensors_val = random_flash_tensors(1, 2, TILE);
    tensors_val.forward();
    std::vector<float> before_list = tensors_val.output_tensor.tolist();
    // 2. token 31 set to 100, the forward again
    tensors_val.qkv_tensor[0][TILE - 1].fill_(100);
    tensors_val.forward();
    std::vector<float> after_list = tensors_val.output_tensor.tolist();
    // 3. check the first 31 rows
    size_t first_rows_val = (TILE - 1) * 2 * HEAD_SIZE;
    ASSERT_EQ(std::vector<float>(after_list.begin(), after_list.begin() + first_rows_val),
              std::vector<float>(before_list.begin(), before_list.begin() + first_rows_val));
}

static void equal_scores_give_the_mean_of_the_values(void)
{
    // Expect: q = 0, so every score is 0 and token t gives each of its t + 1 keys 1 / (t + 1): its output is the
    // mean of v over tokens 0 .. t. With v of token t = t in all 64 columns, output row t is the mean of 0 .. t
    // = t / 2 in all 64 columns, within 1e-5. One head of 32 tokens.
    // 1. qkv: q and k 0, v of token t all t
    FlashTensors tensors_val(1, 1, TILE);
    std::vector<float> qkv_list(TILE * 3 * HEAD_SIZE, 0.0f);
    std::vector<float> expected_list(TILE * HEAD_SIZE);
    for (int token_index_val = 0; token_index_val < TILE; token_index_val++)
        for (int column_index_val = 0; column_index_val < HEAD_SIZE; column_index_val++)
        {
            qkv_list[token_index_val * 3 * HEAD_SIZE + 2 * HEAD_SIZE + column_index_val] = (float)token_index_val;
            expected_list[token_index_val * HEAD_SIZE + column_index_val] = 0.5f * token_index_val;
        }
    tensors_val.qkv_tensor = tensor_of(tensors_val.qkv_tensor.shape(), qkv_list);
    // 2. the forward
    tensors_val.forward();
    // 3. check
    ASSERT_NEAR(tensors_val.output_tensor.tolist(), expected_list, 1e-5);
}

static void large_scores_do_not_overflow(void)
{
    // Expect: q = 4000 0 0 ... for every token, k of token j = j 0 0 ..., so token t's scores are 4000 x j / 8 =
    // 500 j: up to 15,500. exp(15,500) is infinity in a float, but after taking the row's max, exp(-500) and
    // below are 0 and exp(0) is 1: each token puts all its weight on itself, and its output is its own v =
    // (token + 1) 0 0 ...: 1, 2, ..., 32 in column 0, exactly.
    // 1. qkv as above
    std::vector<float> qkv_list(TILE * 3 * HEAD_SIZE, 0.0f);
    std::vector<float> expected_list(TILE * HEAD_SIZE, 0.0f);
    for (int token_index_val = 0; token_index_val < TILE; token_index_val++)
    {
        float *row_ptr = qkv_list.data() + token_index_val * 3 * HEAD_SIZE;
        row_ptr[0] = 4000;
        row_ptr[HEAD_SIZE] = (float)token_index_val;
        row_ptr[2 * HEAD_SIZE] = (float)(token_index_val + 1);
        expected_list[token_index_val * HEAD_SIZE] = (float)(token_index_val + 1);
    }
    // 2. the forward
    FlashTensors tensors_val(1, 1, TILE);
    tensors_val.qkv_tensor = tensor_of(tensors_val.qkv_tensor.shape(), qkv_list);
    tensors_val.forward();
    // 3. check
    ASSERT_EQ(tensors_val.output_tensor.tolist(), expected_list);
}

static void heads_do_not_mix(void)
{
    // Expect: random qkv, {1, 2, 32}; then head 1's columns of q, k and v are set to 100 for every token and the
    // forward run again. Head 0's half of every output row (its first 64 columns) is the same to the last bit:
    // each head's scores, attention_probs and mix use its own columns only.
    // 1. the forward on random qkv
    torch::manual_seed(4);
    FlashTensors tensors_val = random_flash_tensors(1, 2, TILE);
    tensors_val.forward();
    std::vector<float> before_list = tensors_val.output_tensor.tolist();
    // 2. head 1's columns set to 100, the forward again
    std::vector<float> qkv_list = tensors_val.qkv_tensor.tolist();
    for (size_t token_index_val = 0; token_index_val < TILE; token_index_val++)
        for (size_t part_index_val = 0; part_index_val < 3; part_index_val++)          // q, k, v: 128 columns each, head 1 the last 64
            for (size_t column_index_val = HEAD_SIZE; column_index_val < 2 * HEAD_SIZE; column_index_val++)
                qkv_list[token_index_val * 6 * HEAD_SIZE + part_index_val * 2 * HEAD_SIZE + column_index_val] = 100;
    tensors_val.qkv_tensor.copy_from_cpu(qkv_list.data());
    tensors_val.forward();
    std::vector<float> after_list = tensors_val.output_tensor.tolist();
    // 3. head 0's columns of every row, before and after
    std::vector<float> head_0_before_list;
    std::vector<float> head_0_after_list;
    for (size_t token_index_val = 0; token_index_val < TILE; token_index_val++)
        for (size_t column_index_val = 0; column_index_val < HEAD_SIZE; column_index_val++)
        {
            head_0_before_list.push_back(before_list[token_index_val * 2 * HEAD_SIZE + column_index_val]);
            head_0_after_list.push_back(after_list[token_index_val * 2 * HEAD_SIZE + column_index_val]);
        }
    // 4. check
    ASSERT_EQ(head_0_after_list, head_0_before_list);
}

static void a_gradient_on_the_first_token_reaches_only_its_v(void)
{
    // Expect: random qkv, {1, 2, 32}, and output_grad random in row 0, 0 in the others. Output row 0 is 1 x v of
    // token 0 (the_first_token_sees_only_itself), so v of token 0 gets output_grad row 0 itself, exactly. q and
    // k of token 0 get ds = 1 x (da - row_dot) with da = output_grad · v and row_dot = output_grad · output, the
    // same dot added in another order: 0 within 1e-6. Every other token gets 0.
    // 1. random qkv; output_grad random in row 0 only
    torch::manual_seed(5);
    FlashTensors tensors_val = random_flash_tensors(1, 2, TILE);
    std::vector<float> output_grad_list(TILE * 2 * HEAD_SIZE, 0.0f), row_0_list = Tensor::randn({2 * HEAD_SIZE}).tolist();
    std::copy(row_0_list.begin(), row_0_list.end(), output_grad_list.begin());
    tensors_val.output_grad_tensor = tensor_of(tensors_val.output_tensor.shape(), output_grad_list);
    // 2. the forward, then the backward
    tensors_val.forward();
    tensors_val.backward();
    // 3. check: row 0's v gets output_grad row 0, everything else 0
    std::vector<float> qkv_grad_list = tensors_val.qkv_grad_tensor.tolist();
    std::vector<float> expected_list(TILE * 6 * HEAD_SIZE, 0.0f);   // row 0: q 0, k 0, v = output_grad row 0
    std::copy(row_0_list.begin(), row_0_list.end(), expected_list.begin() + 4 * HEAD_SIZE);
    ASSERT_EQ(std::vector<float>(qkv_grad_list.begin() + 4 * HEAD_SIZE, qkv_grad_list.begin() + 6 * HEAD_SIZE), row_0_list);
    ASSERT_NEAR(qkv_grad_list, expected_list, 1e-6);
}

static void backward_writes_over_qkv_grad_the_same_every_time(void)
{
    // Expect: qkv_grad filled with 7s first: after a backward, no 7 is left (every value of its three parts is
    // written), and a second backward gives the same values to the last bit (no two blocks add onto one value).
    // 1. a forward, then a backward onto 7s
    torch::manual_seed(0);
    FlashTensors tensors_val = random_flash_tensors(2, GPT2_NUM_HEADS, 8 * TILE);
    tensors_val.forward();
    tensors_val.qkv_grad_tensor.fill_(7.0);
    tensors_val.backward();
    torch::Tensor first_tensor = pytorch_copy_of(tensors_val.qkv_grad_tensor);
    ASSERT_EQ((first_tensor == 7.0f).sum().item<int64_t>(), (int64_t)0);
    // 2. a second backward: the same bits
    tensors_val.backward();
    ASSERT_EQ(torch::equal(pytorch_copy_of(tensors_val.qkv_grad_tensor), first_tensor), true);
}

// ══ part 3: the kernels fit their launch, and any block size works ═══════════

static void every_kernel_fits_its_block(void)
{
    // Expect: each kernel can be launched with its threads per block (256), and nothing is spilled. The compiler
    // picks each kernel's registers per thread; cudaFuncGetAttributes reads what it wrote down, without running
    // anything:
    //
    //     numRegs              registers per thread
    //     maxThreadsPerBlock   the largest block it can be launched with
    //     localSizeBytes       local memory per thread (spills and stack): 0 if nothing spills
    const int kernel_count_val = 4;
    const void *kernels_list[kernel_count_val] = {(const void *)attention_flash_forward_kernel,
                                                  (const void *)attention_flash_row_dot_kernel,
                                                  (const void *)attention_flash_key_grad_kernel,
                                                  (const void *)attention_flash_query_grad_kernel};
    const char *names_list[kernel_count_val] = {"attention_flash_forward_kernel", "attention_flash_row_dot_kernel",
                                                "attention_flash_key_grad_kernel", "attention_flash_query_grad_kernel"};
    int threads_list[kernel_count_val] = {ATTENTION_FLASH_THREADS, ATTENTION_FLASH_ROW_DOT_THREADS_PER_BLOCK,
                                          ATTENTION_FLASH_THREADS, ATTENTION_FLASH_THREADS};
    for (int kernel_index_val = 0; kernel_index_val < kernel_count_val; kernel_index_val++)
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

static void any_block_size_gives_the_same_bits(void)
{
    // Expect: random qkv and output_grad, {2, 12, 256}. The backward with row_dot's block size set to 32, 128,
    // 512 and 1024 (one warp up to HARDWARE_MAX_THREADS_PER_BLOCK) gives the default's qkv_grad to the last bit: each
    // row_dot is one warp's whatever the block. The tile kernels' block is always ATTENTION_FLASH_THREADS.
    const int block_sizes_list[] = {HARDWARE_THREADS_PER_WARP, 4 * HARDWARE_THREADS_PER_WARP, 16 * HARDWARE_THREADS_PER_WARP, HARDWARE_MAX_THREADS_PER_BLOCK};
    // 1. random qkv and output_grad; the default's qkv_grad
    torch::manual_seed(0);
    FlashTensors tensors_val = random_flash_tensors(2, GPT2_NUM_HEADS, 8 * TILE);
    tensors_val.forward();
    tensors_val.backward();
    torch::Tensor usual_qkv_grad_tensor = pytorch_copy_of(tensors_val.qkv_grad_tensor);
    // 2. each block size against it
    for (int threads_per_block_val : block_sizes_list)
    {
        tensors_val.qkv_grad_tensor.zero_();
        tensors_val.backward(threads_per_block_val);
        ASSERT_EQ(torch::equal(pytorch_copy_of(tensors_val.qkv_grad_tensor), usual_qkv_grad_tensor), true);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    needs_head_size_64_and_whole_tiles();
    matches_pytorch_on_one_tile();
    matches_pytorch_with_3_batch_rows_of_96_tokens();
    matches_pytorch_on_gpt2_heads_of_256_tokens();
    matches_pytorch_on_gpt2_small_size();

    forward_of_the_note_s_example();
    backward_of_the_note_s_example();
    the_first_token_sees_only_itself();
    a_later_token_does_not_change_earlier_outputs();
    equal_scores_give_the_mean_of_the_values();
    large_scores_do_not_overflow();
    heads_do_not_mix();
    a_gradient_on_the_first_token_reaches_only_its_v();
    backward_writes_over_qkv_grad_the_same_every_time();

    every_kernel_fits_its_block();
    any_block_size_gives_the_same_bits();

    printf("\nall tests passed\n");
    return 0;
}
