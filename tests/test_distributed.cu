// test_distributed.cu — tests for llmc/distributed.cuh (Distributed), and for one AdamW step shared by several
// GPUs the way train_gpt2.cu does it. Needs 2 GPUs or more: NCCL puts each rank on a GPU of its own. Every rank
// runs every check, so each line prints once per rank; the first FAIL stops the program.
//
//     nvcc -O3 -std=c++17 -arch=native tests/test_distributed.cu -o test_distributed -lnccl
//     torchrun --standalone --nproc_per_node=2 --no-python ./test_distributed
//
// No libtorch: every expected value is worked out by hand, from the rank. The values are small whole numbers or
// halves, so bfloat16 (8 significant bits: whole numbers up to 256) holds them and every sum of them exactly,
// for up to 8 ranks: NCCL may add them in any order.
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/distributed.cuh"
#include "../llmc/optimizer.cuh"
#include "../llmc/kernels/grad_norm.cuh"

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives `count` values first, first + step, first + 2 step, ...: the values a rank fills a tensor with.
static std::vector<float> values_from(float first, float step, size_t count)
{
    std::vector<float> values(count);
    for (size_t i = 0; i < count; i++)
        values[i] = first + step * i;
    return values;
}

/// Makes a tensor of this dtype on the GPU holding these float values (rounded, for bfloat16).
static Tensor tensor_of(const std::vector<float> &values, DType dtype)
{
    Tensor floats = Tensor::empty({values.size()});
    floats.copy_from_cpu(values.data());
    return floats.to(dtype);
}

// ══ the collectives ═══════════════════════════════════════════════════════════

static void reduce_scatter_mean_averages_each_ranks_slice(const Distributed &distributed, DType dtype)
{
    // Expect: N ranks, 4 N values each, rank r's value i = 2 r + (i mod 8): the mean over the ranks of value i is
    // 2 (0 + 1 + ... + N - 1) / N + (i mod 8) = (N - 1) + (i mod 8). After reduce_scatter_mean, rank r's own
    // slice (values 4 r .. 4 r + 3) holds that mean; its other values are its own, as before. With 2 ranks:
    // rank 0 {0, ..., 7} becomes {1, 2, 3, 4, 4, 5, 6, 7}, rank 1 {2, ..., 9} becomes {2, 3, 4, 5, 5, 6, 7, 8}.
    int r = distributed.rank(), N = distributed.world_size();
    std::vector<float> values(4 * N), expected(4 * N);
    for (int i = 0; i < 4 * N; i++)
    {
        values[i] = 2.0f * r + (i % 8);
        bool in_own_slice = i >= 4 * r && i < 4 * r + 4;
        expected[i] = in_own_slice ? (N - 1.0f) + (i % 8) : values[i];
    }
    Tensor flat = tensor_of(values, dtype);
    distributed.reduce_scatter_mean(flat);
    ASSERT_EQ(flat.to(DType::Float32).tolist(), expected);
}

static void all_gather_gives_every_rank_every_slice(const Distributed &distributed, DType dtype)
{
    // Expect: N ranks, 4 N values each, all -1 but rank r's own slice, 10 r + j for j = 0 .. 3. After all_gather
    // every rank holds every slice: {0, 1, 2, 3, 10, 11, 12, 13, ...}.
    int r = distributed.rank(), N = distributed.world_size();
    std::vector<float> values(4 * N, -1.0f), expected(4 * N);
    for (int slice = 0; slice < N; slice++)
        for (int j = 0; j < 4; j++)
            expected[4 * slice + j] = 10.0f * slice + j;
    for (int j = 0; j < 4; j++)
        values[4 * r + j] = expected[4 * r + j];
    Tensor flat = tensor_of(values, dtype);
    distributed.all_gather(flat);
    ASSERT_EQ(flat.to(DType::Float32).tolist(), expected);
}

static void slice_norms_add_up_to_the_norm_of_all(const Distributed &distributed)
{
    // Expect: rank r's norm is r + 1, the norm of a slice; together sqrt(1² + 2² + ... + N²). With 2 ranks
    // sqrt(5) = 2.2360680, with 8 sqrt(204) = 14.2828569; on every rank.
    int r = distributed.rank(), N = distributed.world_size();
    float norm_value = r + 1.0f, sum_squares = 0.0f;
    for (int rank = 0; rank < N; rank++)
        sum_squares += (rank + 1.0f) * (rank + 1.0f);
    Tensor norm = Tensor::zeros({1});
    norm.copy_from_cpu(&norm_value);
    distributed.slice_norms_to_norm(norm);
    ASSERT_NEAR(norm.item<float>(), sqrtf(sum_squares), 1e-6);
}

static void mean_of_cpu_value_averages_a_number_over_the_ranks(const Distributed &distributed)
{
    // Expect: rank r gives r; every rank gets (0 + 1 + ... + N - 1) / N = (N - 1) / 2: 0.5 with 2 ranks.
    int r = distributed.rank(), N = distributed.world_size();
    ASSERT_NEAR(distributed.mean_of_cpu_value((float)r), (N - 1) / 2.0, 1e-6);
}

// ══ one AdamW step on N GPUs, as train_gpt2.cu takes it ═════════════════════════

static void a_shared_step_equals_one_adamw_on_the_mean_grads(const Distributed &distributed)
{
    // Expect: a {6, 4} weight and an {8} bias, 32 values (decayed_count 24), bfloat16. Each rank's grads differ:
    // value i is -4 + (i mod 16) / 2 + r, so their mean is -4 + (i mod 16) / 2 + (N - 1) / 2. Every value is a
    // multiple of 1/2 small enough that bfloat16 holds it, and every partial sum too, so the average comes out
    // exact whatever order NCCL adds in. Each rank runs train_gpt2.cu's step: reduce_scatter_mean of the grads,
    // GradNorm of its slice then slice_norms_to_norm, AdamW on its slice, all_gather of the params. On every
    // rank, a world-of-1 AdamW given the mean grads directly is the answer: its grad norm within 1e-6 (the
    // squares added in another order), and the params after 3 steps the same to the last bit (no clipping, so
    // the norm's last bit does not reach them).
    int r = distributed.rank(), N = distributed.world_size();
    AdamWConfig config;
    config.learning_rate = 1e-2f;
    std::vector<Shape> shapes = {{6, 4}, {8}};
    AdamW shared(shapes, config, DType::BFloat16, r, N), one(shapes, config, DType::BFloat16);
    Tensor block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS}), one_block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    GradNorm shared_norm(block_sums), one_norm(one_block_sums);
    Tensor start = tensor_of(values_from(-0.5f, 1.0f / 32, 32), DType::BFloat16);
    for (AdamW *adamw : {&shared, &one})
    {
        adamw->all_params().copy_(start);
        adamw->copy_params_to_master();
    }
    std::vector<float> rank_grads(32), mean_grads(32);
    for (int i = 0; i < 32; i++)
    {
        rank_grads[i] = -4.0f + (i % 16) / 2.0f + r;
        mean_grads[i] = -4.0f + (i % 16) / 2.0f + (N - 1) / 2.0f;
    }
    Tensor grads_slice = shared.all_grads().narrow(0, shared.slice_start(), shared.slice_size());
    for (int t = 1; t <= 3; t++)
    {
        shared.all_grads().copy_(tensor_of(rank_grads, DType::BFloat16));
        distributed.reduce_scatter_mean(shared.all_grads());
        shared_norm.compute(grads_slice, shared.grad_norm());
        distributed.slice_norms_to_norm(shared.grad_norm());
        shared.step();
        distributed.all_gather(shared.all_params());

        one.all_grads().copy_(tensor_of(mean_grads, DType::BFloat16));
        one_norm.compute(one.all_grads(), one.grad_norm());
        one.step();

        ASSERT_NEAR(shared.grad_norm().item<float>(), one.grad_norm().item<float>(), 1e-6);
        ASSERT_EQ(shared.all_params().to(DType::Float32).tolist(), one.all_params().to(DType::Float32).tolist());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    const char *run_id = getenv("TORCHELASTIC_RUN_ID");       // torchrun: new for every run
    Distributed distributed(std::string("/tmp/test_distributed_nccl_id_") + (run_id != nullptr ? run_id : "") + ".bin");
    if (distributed.world_size() < 2)
    {
        printf("needs 2 GPUs or more: torchrun --standalone --nproc_per_node=2 --no-python ./test_distributed\n");
        return 1;
    }
    printf("rank %d of %d, on GPU %d\n", distributed.rank(), distributed.world_size(), distributed.local_rank());

    reduce_scatter_mean_averages_each_ranks_slice(distributed, DType::Float32);
    reduce_scatter_mean_averages_each_ranks_slice(distributed, DType::BFloat16);
    all_gather_gives_every_rank_every_slice(distributed, DType::Float32);
    all_gather_gives_every_rank_every_slice(distributed, DType::BFloat16);
    slice_norms_add_up_to_the_norm_of_all(distributed);
    mean_of_cpu_value_averages_a_number_over_the_ranks(distributed);
    a_shared_step_equals_one_adamw_on_the_mean_grads(distributed);

    printf("\nrank %d: all tests passed\n", distributed.rank());
    return 0;
}
