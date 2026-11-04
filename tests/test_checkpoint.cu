// test_checkpoint.cu — tests for llmc/checkpoint.cuh (checkpoint_save, checkpoint_load, checkpoint_problem). Every
// check prints its test case, input, expected and actual value, and ok; the first FAIL stops the program.
//
// One GPU, one process (Distributed with no RANK set); NCCL is linked but not used. Each test writes its own
// token file and checkpoint folders in the folder it runs from (test_checkpoint_*).
//
//     nvcc -O3 -std=c++17 -arch=native tests/test_checkpoint.cu -o test_checkpoint -lnccl && ./test_checkpoint
#include <filesystem>
#include "test_helpers.h"             // ASSERT_EQ, ASSERT_NEAR
#include "../llmc/checkpoint.cuh"

// A {6, 4} weight and an {8} bias, 32 values, in bfloat16: the smallest AdamW with a matrix and a vector.
static const std::vector<Shape> SHAPES = {{6, 4}, {8}};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives 32 values that differ from test to test by `seed`, between -0.5 and 0.5: params or grads.
static std::vector<float> some_values(int seed)
{
    std::vector<float> values(32);
    for (int i = 0; i < 32; i++)
        values[i] = (float)((i * 37 + seed * 11) % 17 - 8) / 16.0f;
    return values;
}

/// Makes a float tensor on the GPU holding these values, to copy_ into params or grads (rounded to bfloat16).
static Tensor tensor_of(const std::vector<float> &values)
{
    Tensor tensor = Tensor::empty({values.size()});
    tensor.copy_from_cpu(values.data());
    return tensor;
}

/// Writes a token file of tokens 0 .. count - 1, as data/fineweb.py does: the 256-int header, then uint16s.
static void write_token_file(const std::string &path, int count)
{
    int header[TOKEN_FILE_HEADER_INTS] = {TOKEN_FILE_MAGIC, TOKEN_FILE_VERSION, count};
    std::vector<uint16_t> tokens(count);
    for (int i = 0; i < count; i++)
        tokens[i] = (uint16_t)i;
    FILE *file = fopen(path.c_str(), "wb");
    assert(file != nullptr && "cannot write a test file in this folder");
    fwrite(header, sizeof(int), TOKEN_FILE_HEADER_INTS, file);
    fwrite(tokens.data(), sizeof(uint16_t), tokens.size(), file);
    fclose(file);
}

/// Gives every value of a float or bfloat16 tensor as a float list on the CPU.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Gives the text of a small file (the checkpoint's `latest`), without its last newline.
static std::string text_of(const std::string &path)
{
    char text[64] = "";
    FILE *file = fopen(path.c_str(), "r");
    if (file != nullptr)
    {
        if (fscanf(file, "%63s", text) != 1)
            text[0] = '\0';
        fclose(file);
    }
    return text;
}

// ══ the tests ═════════════════════════════════════════════════════════════════

static void loading_goes_on_exactly_where_saving_stopped(const Distributed &distributed)
{
    // Expect: a bfloat16 AdamW and a loader on a file of 40 tokens (batches of {2, 4}) take 2 steps, then save
    // at step 2 of a 10-step run. A new AdamW and loader load it: 2 steps done, the same master, m, v and
    // bfloat16 params, the loader at the same file, token 16, epoch 1. Then both take step 3 on the same grads:
    // the same next batch (token_ids 16 .. 23) and the same params after it, to the last bit. The file holds a
    // 256-byte header and 3 x 32 floats: 640 bytes.
    std::filesystem::remove_all("test_checkpoint_run");
    write_token_file("test_checkpoint_tokens.bin", 40);
    AdamWConfig config;
    config.learning_rate = 1e-2f;
    AdamW first(SHAPES, config, DType::BFloat16), second(SHAPES, config, DType::BFloat16);
    DataLoader first_loader("test_checkpoint_tokens.bin", 2, 4), second_loader("test_checkpoint_tokens.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    first.all_params().copy_(tensor_of(some_values(0)));
    first.copy_params_to_master();
    for (int t = 1; t <= 2; t++)
    {
        first_loader.next_batch(token_ids, targets);
        first.all_grads().copy_(tensor_of(some_values(t)));
        first.step();
    }
    checkpoint_save("test_checkpoint_run", 2, 10, first, first_loader, distributed);
    ASSERT_EQ((size_t)std::filesystem::file_size("test_checkpoint_run/step_00000002/rank_0.bin"), (size_t)640);

    int steps_done = checkpoint_load("test_checkpoint_run", 10, second, second_loader, distributed);
    ASSERT_EQ(steps_done, 2);
    ASSERT_EQ(second.steps_taken(), 2);
    ASSERT_EQ(list_of(second.master_slice()), list_of(first.master_slice()));
    ASSERT_EQ(list_of(second.m_slice()), list_of(first.m_slice()));
    ASSERT_EQ(list_of(second.v_slice()), list_of(first.v_slice()));
    ASSERT_EQ(list_of(second.all_params()), list_of(first.all_params()));
    ASSERT_EQ(second_loader.current_file(), first_loader.current_file());
    ASSERT_EQ(second_loader.position(), (size_t)16);
    ASSERT_EQ(second_loader.epoch(), 1);

    Tensor second_ids = Tensor::zeros({2, 4}, DType::Int32), second_targets = Tensor::zeros({2, 4}, DType::Int32);
    first_loader.next_batch(token_ids, targets);
    second_loader.next_batch(second_ids, second_targets);
    ASSERT_EQ(second_ids.tolist<int>(), token_ids.tolist<int>());
    for (AdamW *adamw : {&first, &second})
    {
        adamw->all_grads().copy_(tensor_of(some_values(3)));
        adamw->step();
    }
    ASSERT_EQ(list_of(second.all_params()), list_of(first.all_params()));
    ASSERT_EQ(list_of(second.master_slice()), list_of(first.master_slice()));
}

static void latest_names_the_newest_and_two_are_kept(const Distributed &distributed)
{
    // Expect: saves at steps 1, 2 and 3. `latest` says step_00000003; the folders of steps 2 and 3 are there,
    // step 1's was deleted (keep 2); no .partial file is left.
    std::filesystem::remove_all("test_checkpoint_keep");
    write_token_file("test_checkpoint_tokens.bin", 40);
    AdamWConfig config;
    AdamW adamw(SHAPES, config, DType::BFloat16);
    DataLoader loader("test_checkpoint_tokens.bin", 2, 4);
    for (int steps_done = 1; steps_done <= 3; steps_done++)
        checkpoint_save("test_checkpoint_keep", steps_done, 10, adamw, loader, distributed);
    ASSERT_EQ(text_of("test_checkpoint_keep/latest") == "step_00000003", true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000001"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000002/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000003/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000003/rank_0.bin.partial"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/latest.partial"), false);
}

static void a_checkpoint_of_another_run_is_refused(const Distributed &distributed)
{
    // Expect: a checkpoint of a 10-step bfloat16 run on 1 GPU. checkpoint_problem finds nothing wrong for the
    // same run; but for a 20-step run (its learning rate schedule is another), for rank 0 of 2 GPUs, for a
    // float run, and for a folder with no checkpoint, it says what is wrong (printed below each check), and
    // checkpoint_load would stop with that.
    std::filesystem::remove_all("test_checkpoint_other");
    write_token_file("test_checkpoint_tokens.bin", 40);
    AdamWConfig config;
    AdamW same(SHAPES, config, DType::BFloat16), two_gpus(SHAPES, config, DType::BFloat16, 0, 2);
    AdamW in_float(SHAPES, config, DType::Float32);
    DataLoader loader("test_checkpoint_tokens.bin", 2, 4);
    checkpoint_save("test_checkpoint_other", 1, 10, same, loader, distributed);
    ASSERT_EQ(checkpoint_problem("test_checkpoint_other", 10, same).empty(), true);
    for (std::string problem : {checkpoint_problem("test_checkpoint_other", 20, same),
                                checkpoint_problem("test_checkpoint_other", 10, two_gpus),
                                checkpoint_problem("test_checkpoint_other", 10, in_float),
                                checkpoint_problem("test_checkpoint_nothing", 10, same)})
    {
        ASSERT_EQ(problem.empty(), false);
        printf("        %s\n", problem.c_str());
    }
}

static void folder_names_sort_as_the_steps_do(void)
{
    // Expect: step_ and 8 digits: step 1000 is "step_00001000", and step 999 sorts before step 1000 as text.
    ASSERT_EQ(checkpoint_folder_name(1000) == "step_00001000", true);
    ASSERT_EQ(checkpoint_folder_name(999) < checkpoint_folder_name(1000), true);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    Distributed distributed("");                            // no RANK set: one process, no NCCL
    loading_goes_on_exactly_where_saving_stopped(distributed);
    latest_names_the_newest_and_two_are_kept(distributed);
    a_checkpoint_of_another_run_is_refused(distributed);
    folder_names_sort_as_the_steps_do();

    printf("\nall tests passed\n");
    return 0;
}
