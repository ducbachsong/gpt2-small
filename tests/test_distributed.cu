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

#define SLICE_VALUES 4                // values in each rank's slice, in the collectives' tests
#define SHAPES_VALUES 32              // the AdamW step's values: a {6, 4} weight and an {8} bias

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives `count_val` values first, first + step, first + 2 step, ...: the values a rank fills a tensor with.
static std::vector<float> values_from(float first_val, float step_val, size_t count_val)
{
    std::vector<float> values_list(count_val);
    for (size_t i_val = 0; i_val < count_val; i_val++)
        values_list[i_val] = first_val + step_val * i_val;
    return values_list;
}

/// Makes a tensor of this dtype on the GPU holding these float values (rounded, for bfloat16).
static Tensor tensor_of(const std::vector<float> &values_list, DType dtype_val)
{
    Tensor floats_tensor = Tensor::empty({values_list.size()});
    floats_tensor.copy_from_cpu(values_list.data());
    return floats_tensor.to(dtype_val);
}

// ══ the collectives ═══════════════════════════════════════════════════════════

static void reduce_scatter_mean_averages_each_ranks_slice(const Distributed &distributed_val, DType dtype_val)
{
    // Expect: N ranks, 4 N values each, rank r's value i = 2 r + (i mod 8): the mean over the ranks of value i is
    // 2 (0 + 1 + ... + N - 1) / N + (i mod 8) = (N - 1) + (i mod 8). After reduce_scatter_mean, rank r's own
    // slice (values 4 r .. 4 r + 3) holds that mean; its other values are its own, as before. With 2 ranks:
    // rank 0 {0, ..., 7} becomes {1, 2, 3, 4, 4, 5, 6, 7}, rank 1 {2, ..., 9} becomes {2, 3, 4, 5, 5, 6, 7, 8}.
    // 1. this rank's values, and what they should become
    int r_val = distributed_val.rank();
    int n_val = distributed_val.world_size();
    std::vector<float> values_list(SLICE_VALUES * n_val), expected_list(SLICE_VALUES * n_val);
    for (int i_val = 0; i_val < SLICE_VALUES * n_val; i_val++)
    {
        values_list[i_val] = 2.0f * r_val + (i_val % 8);
        bool in_own_slice_val = i_val >= SLICE_VALUES * r_val && i_val < SLICE_VALUES * (r_val + 1);
        expected_list[i_val] = in_own_slice_val ? (n_val - 1.0f) + (i_val % 8) : values_list[i_val];
    }
    // 2. reduce_scatter_mean
    Tensor flat_tensor = tensor_of(values_list, dtype_val);
    distributed_val.reduce_scatter_mean(flat_tensor);
    // 3. check
    ASSERT_EQ(flat_tensor.to(DType::Float32).tolist(), expected_list);
}

static void all_gather_gives_every_rank_every_slice(const Distributed &distributed_val, DType dtype_val)
{
    // Expect: N ranks, 4 N values each, all -1 but rank r's own slice, 10 r + j for j = 0 .. 3. After all_gather
    // every rank holds every slice: {0, 1, 2, 3, 10, 11, 12, 13, ...}.
    // 1. this rank's values: its own slice, -1 elsewhere; and every slice, expected
    int r_val = distributed_val.rank();
    int n_val = distributed_val.world_size();
    std::vector<float> values_list(SLICE_VALUES * n_val, -1.0f), expected_list(SLICE_VALUES * n_val);
    for (int slice_val = 0; slice_val < n_val; slice_val++)
        for (int j_val = 0; j_val < SLICE_VALUES; j_val++)
            expected_list[SLICE_VALUES * slice_val + j_val] = 10.0f * slice_val + j_val;
    for (int j_val = 0; j_val < SLICE_VALUES; j_val++)
        values_list[SLICE_VALUES * r_val + j_val] = expected_list[SLICE_VALUES * r_val + j_val];
    // 2. all_gather
    Tensor flat_tensor = tensor_of(values_list, dtype_val);
    distributed_val.all_gather(flat_tensor);
    // 3. check
    ASSERT_EQ(flat_tensor.to(DType::Float32).tolist(), expected_list);
}

static void slice_norms_add_up_to_the_norm_of_all(const Distributed &distributed_val)
{
    // Expect: rank r's norm is r + 1, the norm of a slice; together sqrt(1² + 2² + ... + N²). With 2 ranks
    // sqrt(5) = 2.2360680, with 8 sqrt(204) = 14.2828569; on every rank.
    // 1. this rank's norm, and the norm of all, worked out
    int r_val = distributed_val.rank();
    int n_val = distributed_val.world_size();
    float norm_value_val = r_val + 1.0f;
    float sum_squares_val = 0.0f;
    for (int rank_val = 0; rank_val < n_val; rank_val++)
        sum_squares_val += (rank_val + 1.0f) * (rank_val + 1.0f);
    // 2. slice_norms_to_norm
    Tensor norm_tensor = Tensor::zeros({1});
    norm_tensor.copy_from_cpu(&norm_value_val);
    distributed_val.slice_norms_to_norm(norm_tensor);
    // 3. check
    ASSERT_NEAR(norm_tensor.item<float>(), sqrtf(sum_squares_val), 1e-6);
}

static void mean_of_cpu_value_averages_a_number_over_the_ranks(const Distributed &distributed_val)
{
    // Expect: rank r gives r; every rank gets (0 + 1 + ... + N - 1) / N = (N - 1) / 2: 0.5 with 2 ranks.
    int r_val = distributed_val.rank();
    int n_val = distributed_val.world_size();
    ASSERT_NEAR(distributed_val.mean_of_cpu_value((float)r_val), (n_val - 1) / 2.0, 1e-6);
}

// ══ one AdamW step on N GPUs, as train_gpt2.cu takes it ═════════════════════════

static void a_shared_step_equals_one_adamw_on_the_mean_grads(const Distributed &distributed_val)
{
    // Expect: a {6, 4} weight and an {8} bias, 32 values (decayed_count 24), bfloat16. Each rank's grads differ:
    // value i is -4 + (i mod 16) / 2 + r, so their mean is -4 + (i mod 16) / 2 + (N - 1) / 2. Every value is a
    // multiple of 1/2 small enough that bfloat16 holds it, and every partial sum too, so the average comes out
    // exact whatever order NCCL adds in. Each rank runs train_gpt2.cu's step: reduce_scatter_mean of the grads,
    // GradNorm of its slice then slice_norms_to_norm, AdamW on its slice, all_gather of the params. On every
    // rank, a world-of-1 AdamW given the mean grads directly is the answer: its grad norm within 1e-6 (the
    // squares added in another order), and the params after 3 steps the same to the last bit (no clipping, so
    // the norm's last bit does not reach them).

    // 1. a shared AdamW (this rank's slice) and a world-of-1 one, from the same start
    int r_val = distributed_val.rank();
    int n_val = distributed_val.world_size();
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    std::vector<Shape> shapes_list = {{6, 4}, {8}};
    AdamW shared_val(shapes_list, config_val, DType::BFloat16, r_val, n_val), one_val(shapes_list, config_val, DType::BFloat16);
    Tensor block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    Tensor one_block_sums_tensor = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    GradNorm shared_norm_val(block_sums_tensor), one_norm_val(one_block_sums_tensor);
    Tensor start_tensor = tensor_of(values_from(-0.5f, 1.0f / SHAPES_VALUES, SHAPES_VALUES), DType::BFloat16);
    for (AdamW *adamw_ptr : {&shared_val, &one_val})
    {
        adamw_ptr->all_params().copy_(start_tensor);
        adamw_ptr->copy_params_to_master();
    }

    // 2. this rank's grads, and the mean of all ranks' grads
    std::vector<float> rank_grads_list(SHAPES_VALUES), mean_grads_list(SHAPES_VALUES);
    for (int i_val = 0; i_val < SHAPES_VALUES; i_val++)
    {
        rank_grads_list[i_val] = -4.0f + (i_val % 16) / 2.0f + r_val;
        mean_grads_list[i_val] = -4.0f + (i_val % 16) / 2.0f + (n_val - 1) / 2.0f;
    }

    // 3. three steps: the shared one as train_gpt2.cu takes it, the world-of-1 one on the mean grads
    Tensor grads_slice_tensor = shared_val.all_grads().narrow(0, shared_val.slice_start(), shared_val.slice_size());
    for (int t_val = 1; t_val <= 3; t_val++)
    {
        shared_val.all_grads().copy_(tensor_of(rank_grads_list, DType::BFloat16));
        distributed_val.reduce_scatter_mean(shared_val.all_grads());
        shared_norm_val.compute(grads_slice_tensor, shared_val.grad_norm());
        distributed_val.slice_norms_to_norm(shared_val.grad_norm());
        shared_val.step();
        distributed_val.all_gather(shared_val.all_params());

        one_val.all_grads().copy_(tensor_of(mean_grads_list, DType::BFloat16));
        one_norm_val.compute(one_val.all_grads(), one_val.grad_norm());
        one_val.step();

        ASSERT_NEAR(shared_val.grad_norm().item<float>(), one_val.grad_norm().item<float>(), 1e-6);
        ASSERT_EQ(shared_val.all_params().to(DType::Float32).tolist(), one_val.all_params().to(DType::Float32).tolist());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    // 1. NCCL across the ranks torchrun started
    const char *run_id_ptr = getenv("TORCHELASTIC_RUN_ID");   // torchrun: new for every run
    Distributed distributed_val(std::string("/tmp/test_distributed_nccl_id_") + (run_id_ptr != nullptr ? run_id_ptr : "") +
                                ".bin");
    if (distributed_val.world_size() < 2)
    {
        printf("needs 2 GPUs or more: torchrun --standalone --nproc_per_node=2 --no-python ./test_distributed\n");
        return 1;
    }
    printf("rank %d of %d, on GPU %d\n", distributed_val.rank(), distributed_val.world_size(), distributed_val.local_rank());

    // 2. the tests
    reduce_scatter_mean_averages_each_ranks_slice(distributed_val, DType::Float32);
    reduce_scatter_mean_averages_each_ranks_slice(distributed_val, DType::BFloat16);
    all_gather_gives_every_rank_every_slice(distributed_val, DType::Float32);
    all_gather_gives_every_rank_every_slice(distributed_val, DType::BFloat16);
    slice_norms_add_up_to_the_norm_of_all(distributed_val);
    mean_of_cpu_value_averages_a_number_over_the_ranks(distributed_val);
    a_shared_step_equals_one_adamw_on_the_mean_grads(distributed_val);

    printf("\nrank %d: all tests passed\n", distributed_val.rank());
    return 0;
}
