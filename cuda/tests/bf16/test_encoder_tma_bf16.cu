// test_encoder_tma_bf16.cu — encoder_tma_bf16_forward gives the same bits as encoder_bf16_forward, at the default and at the edge settings.
#include <torch/types.h>              // torch::randn, torch::equal; libtorch before any CUDA header
#include <torch/utils.h>              // torch::manual_seed
#include <random>                     // std::mt19937: random token ids
#include <algorithm>                  // std::count
#include "../common/test_helpers.h"    // ASSERT_EQ, start_run_log, print_test_name
#include "../../kernels/bf16/encoder_bf16.cuh"
#include "../../kernels/bf16/encoder_tma_bf16.cuh"

#define D24_BATCH_SIZE 16             // nanochat d24's micro-batch: 16 rows of 2,048 tokens
#define D24_SEQ_LEN 2048
#define D24_EMBED_DIM 1536
#define D24_VOCAB_SIZE 32768
#define GPT2_BATCH_SIZE 4             // GPT-2 small's
#define GPT2_SEQ_LEN 1024
#define GPT2_EMBED_DIM 768
#define GPT2_VOCAB_SIZE 50304

// ── helpers ───────────────────────────────────────────────────────────────────

/// Makes a bfloat16 tensor with random values, mean 0 and standard deviation 1, made by PyTorch on the GPU.
///
///   shape_list   the tensor's shape
static Tensor random_bf16_tensor(const Shape &shape_list)
{
    Tensor floats_tensor = Tensor::empty(shape_list);
    torch::Tensor values_tensor = torch::randn({(int64_t)floats_tensor.numel()}, torch::TensorOptions().device(torch::kCUDA));
    cudaCheck(cudaMemcpy(floats_tensor.data(), values_tensor.data_ptr<float>(), floats_tensor.numel() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    return floats_tensor.to(DType::BFloat16);
}

/// Makes an int tensor {batch_size_val, seq_len_val} of random token ids below vocab_size_val on the GPU.
///
///   batch_size_val, seq_len_val   its shape
///   vocab_size_val                the ids' bound
static Tensor random_token_ids(size_t batch_size_val, size_t seq_len_val, int vocab_size_val)
{
    std::mt19937 generator_val(0);
    std::uniform_int_distribution<int> pick_val(0, vocab_size_val - 1);
    std::vector<int> ids_list(batch_size_val * seq_len_val);
    for (int &id_val : ids_list)
        id_val = pick_val(generator_val);
    Tensor tensor = Tensor::empty({batch_size_val, seq_len_val}, DType::Int32);
    tensor.copy_from_cpu(ids_list.data());
    return tensor;
}

/// Tells if two bfloat16 tensors hold the same bits.
///
///   first_tensor, second_tensor   the same shape, on the GPU
static bool same_bits(const Tensor &first_tensor, const Tensor &second_tensor)
{
    // Each read as 16-bit ints, with no copy into floats.
    torch::TensorOptions shorts_val = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);
    torch::Tensor first_val = torch::from_blob(first_tensor.data_ptr(), {(int64_t)first_tensor.numel()}, shorts_val);
    torch::Tensor second_val = torch::from_blob(second_tensor.data_ptr(), {(int64_t)second_tensor.numel()}, shorts_val);
    return torch::equal(first_val, second_val);
}

/// An encoder's tensors at one size, and encoder_bf16_forward's result on them.
struct EncoderCase
{
    Tensor wte_tensor, wpe_tensor, token_ids_tensor, expected_tensor, encoded_tensor;

    /// Makes the tensors and the expected result.
    ///
    ///   batch_size_val, seq_len_val, embed_dim_val, vocab_size_val   the sizes
    EncoderCase(size_t batch_size_val, size_t seq_len_val, size_t embed_dim_val, int vocab_size_val)
    {
        torch::manual_seed(0);
        wte_tensor = random_bf16_tensor({(size_t)vocab_size_val, embed_dim_val});
        wpe_tensor = random_bf16_tensor({seq_len_val, embed_dim_val});
        token_ids_tensor = random_token_ids(batch_size_val, seq_len_val, vocab_size_val);
        expected_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        encoded_tensor = Tensor::zeros({batch_size_val, seq_len_val, embed_dim_val}, DType::BFloat16);
        encoder_bf16_forward(expected_tensor, token_ids_tensor, wte_tensor, wpe_tensor);
    }
};

// ── tests ─────────────────────────────────────────────────────────────────────

static void the_default_settings_give_encoder_bf16s_bits(void)
{
    // At nanochat d24's size and GPT-2 small's.
    EncoderCase d24_val(D24_BATCH_SIZE, D24_SEQ_LEN, D24_EMBED_DIM, D24_VOCAB_SIZE);
    encoder_tma_bf16_forward(d24_val.encoded_tensor, d24_val.token_ids_tensor, d24_val.wte_tensor, d24_val.wpe_tensor);
    EncoderCase gpt2_val(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM, GPT2_VOCAB_SIZE);
    encoder_tma_bf16_forward(gpt2_val.encoded_tensor, gpt2_val.token_ids_tensor, gpt2_val.wte_tensor, gpt2_val.wpe_tensor);
    cudaCheck(cudaDeviceSynchronize());
    print_test_name(__func__);
    ASSERT_EQ(same_bits(d24_val.encoded_tensor, d24_val.expected_tensor), true);
    ASSERT_EQ(same_bits(gpt2_val.encoded_tensor, gpt2_val.expected_tensor), true);
}

static void the_edge_settings_give_encoder_bf16s_bits(void)
{
    // GPT-2 small's size, at the sweep's ends: one row a tile and 256, one box a row and many, both swizzles, one set
    // and several, one warp and 32, one cell a thread and 16, few blocks (each walks many tiles) and many.
    EncoderCase case_val(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM, GPT2_VOCAB_SIZE);
    std::vector<bool> fits_list;
    std::vector<bool> same_list;
    auto check_val = [&](bool fits_val) {
        cudaCheck(cudaDeviceSynchronize());
        fits_list.push_back(fits_val);
        same_list.push_back(fits_val && same_bits(case_val.encoded_tensor, case_val.expected_tensor));
        case_val.encoded_tensor.zero_();
    };
    const Tensor &encoded_tensor = case_val.encoded_tensor, &ids_tensor = case_val.token_ids_tensor;
    const Tensor &wte_tensor = case_val.wte_tensor, &wpe_tensor = case_val.wpe_tensor;
    check_val(encoder_tma_bf16_forward_at<8, 128, 1>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 7, 1, 64, 1));
    check_val(encoder_tma_bf16_forward_at<8, 0, 32>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 1000, 256, 64, 3));
    check_val(encoder_tma_bf16_forward_at<128, 128, 4>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 132, 8, 768, 2));
    check_val(encoder_tma_bf16_forward_at<16, 0, 8>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 264, 16, 768, 4));
    check_val(encoder_tma_bf16_forward_at<64, 128, 16>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 33, 64, 384, 2));
    check_val(encoder_tma_bf16_forward_at<32, 0, 2>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 5, 2, 384, 8));
    print_test_name(__func__);
    ASSERT_EQ((long)std::count(fits_list.begin(), fits_list.end(), true), (long)fits_list.size());
    ASSERT_EQ((long)std::count(same_list.begin(), same_list.end(), true), (long)same_list.size());
}

static void a_tile_that_does_not_fit_is_refused(void)
{
    // tile_rows not dividing seq_len, tile_columns not dividing embed_dim, a set too big for shared memory: false, and
    // nothing launched.
    EncoderCase case_val(GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_EMBED_DIM, GPT2_VOCAB_SIZE);
    const Tensor &encoded_tensor = case_val.encoded_tensor, &ids_tensor = case_val.token_ids_tensor;
    const Tensor &wte_tensor = case_val.wte_tensor, &wpe_tensor = case_val.wpe_tensor;
    print_test_name(__func__);
    ASSERT_EQ((encoder_tma_bf16_forward_at<8, 128, 8>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 132, 24, 64, 1)), false);
    ASSERT_EQ((encoder_tma_bf16_forward_at<8, 128, 8>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 132, 32, 320, 1)), false);
    ASSERT_EQ((encoder_tma_bf16_forward_at<8, 128, 8>(encoded_tensor, ids_tensor, wte_tensor, wpe_tensor, 132, 256, 768, 1)), false);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    the_default_settings_give_encoder_bf16s_bits();
    the_edge_settings_give_encoder_bf16s_bits();
    a_tile_that_does_not_fit_is_refused();

    printf("\nall tests passed\n");
    return 0;
}
