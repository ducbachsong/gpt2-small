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
#define SHAPES_VALUES 32              // the values SHAPES holds: 6 x 4 + 8
#define TOKENS_IN_FILE 40             // the token file's tokens: 5 batches of {2, 4}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives SHAPES_VALUES values that differ from test to test by `seed_val`, between -0.5 and 0.5: params or grads.
static std::vector<float> some_values(int seed_val)
{
    std::vector<float> values_list(SHAPES_VALUES);
    for (int i_val = 0; i_val < SHAPES_VALUES; i_val++)
        values_list[i_val] = (float)((i_val * 37 + seed_val * 11) % 17 - 8) / 16.0f;
    return values_list;
}

/// Makes a float tensor on the GPU holding these values, to copy_ into params or grads (rounded to bfloat16).
static Tensor tensor_of(const std::vector<float> &values_list)
{
    Tensor tensor = Tensor::empty({values_list.size()});
    tensor.copy_from_cpu(values_list.data());
    return tensor;
}

/// Writes a token file of tokens 0 .. count - 1, as data/fineweb.py does: the TOKEN_FILE_HEADER_INTS-int header,
/// then uint16s.
static void write_token_file(const std::string &path_val, int count_val)
{
    int header_list[TOKEN_FILE_HEADER_INTS] = {TOKEN_FILE_MAGIC, TOKEN_FILE_VERSION, count_val};
    std::vector<uint16_t> tokens_list(count_val);
    for (int i_val = 0; i_val < count_val; i_val++)
        tokens_list[i_val] = (uint16_t)i_val;
    FILE *file_ptr = fopen(path_val.c_str(), "wb");
    assert(file_ptr != nullptr && "cannot write a test file in this folder");
    fwrite(header_list, sizeof(int), TOKEN_FILE_HEADER_INTS, file_ptr);
    fwrite(tokens_list.data(), sizeof(uint16_t), tokens_list.size(), file_ptr);
    fclose(file_ptr);
}

/// Gives every value of a float or bfloat16 tensor as a float list on the CPU.
static std::vector<float> list_of(const Tensor &tensor)
{
    return tensor.to(DType::Float32).tolist();
}

/// Gives the text of a small file (the checkpoint's `latest`), without its last newline.
static std::string text_of(const std::string &path_val)
{
    char text_list[64] = "";
    FILE *file_ptr = fopen(path_val.c_str(), "r");
    if (file_ptr != nullptr)
    {
        if (fscanf(file_ptr, "%63s", text_list) != 1)
            text_list[0] = '\0';
        fclose(file_ptr);
    }
    return text_list;
}

// ══ the tests ═════════════════════════════════════════════════════════════════

static void loading_goes_on_exactly_where_saving_stopped(const Distributed &distributed_val)
{
    // Expect: a bfloat16 AdamW and a loader on a file of 40 tokens (batches of {2, 4}) take 2 steps, then save
    // at step 2 of a 10-step run. A new AdamW and loader load it: 2 steps done, the same master, m, v and
    // bfloat16 params, the loader at the same file, token 16, epoch 1. Then both take step 3 on the same grads:
    // the same next batch (token_ids 16 .. 23) and the same params after it, to the last bit. The file holds a
    // 256-byte header and 3 x 32 floats: 640 bytes.

    // 1. two steps, then a save at step 2
    std::filesystem::remove_all("test_checkpoint_run");
    write_token_file("test_checkpoint_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    AdamW first_val(SHAPES, config_val, DType::BFloat16), second_val(SHAPES, config_val, DType::BFloat16);
    DataLoader first_loader_val("test_checkpoint_tokens.bin", 2, 4), second_loader_val("test_checkpoint_tokens.bin", 2, 4);
    Tensor token_ids_tensor = Tensor::zeros({2, 4}, DType::Int32);
    Tensor targets_tensor = Tensor::zeros({2, 4}, DType::Int32);
    first_val.all_params().copy_(tensor_of(some_values(0)));
    first_val.copy_params_to_master();
    for (int t_val = 1; t_val <= 2; t_val++)
    {
        first_loader_val.next_batch(token_ids_tensor, targets_tensor);
        first_val.all_grads().copy_(tensor_of(some_values(t_val)));
        first_val.step();
    }
    checkpoint_save("test_checkpoint_run", 2, 10, first_val, first_loader_val, distributed_val);
    size_t expected_bytes_val = CHECKPOINT_HEADER_INTS * sizeof(int64_t) + 3 * SHAPES_VALUES * sizeof(float);   // 640
    ASSERT_EQ((size_t)std::filesystem::file_size("test_checkpoint_run/step_00000002/rank_0.bin"), expected_bytes_val);

    // 2. a load into a new AdamW and loader: the same state
    int steps_done_val = checkpoint_load("test_checkpoint_run", 10, second_val, second_loader_val, distributed_val);
    ASSERT_EQ(steps_done_val, 2);
    ASSERT_EQ(second_val.steps_taken(), 2);
    ASSERT_EQ(list_of(second_val.master_slice()), list_of(first_val.master_slice()));
    ASSERT_EQ(list_of(second_val.m_slice()), list_of(first_val.m_slice()));
    ASSERT_EQ(list_of(second_val.v_slice()), list_of(first_val.v_slice()));
    ASSERT_EQ(list_of(second_val.all_params()), list_of(first_val.all_params()));
    ASSERT_EQ(second_loader_val.current_file(), first_loader_val.current_file());
    ASSERT_EQ(second_loader_val.position(), (size_t)16);
    ASSERT_EQ(second_loader_val.epoch(), 1);

    // 3. step 3 on both: the same batch and the same params after it
    Tensor second_ids_tensor = Tensor::zeros({2, 4}, DType::Int32);
    Tensor second_targets_tensor = Tensor::zeros({2, 4}, DType::Int32);
    first_loader_val.next_batch(token_ids_tensor, targets_tensor);
    second_loader_val.next_batch(second_ids_tensor, second_targets_tensor);
    ASSERT_EQ(second_ids_tensor.tolist<int>(), token_ids_tensor.tolist<int>());
    for (AdamW *adamw_ptr : {&first_val, &second_val})
    {
        adamw_ptr->all_grads().copy_(tensor_of(some_values(3)));
        adamw_ptr->step();
    }
    ASSERT_EQ(list_of(second_val.all_params()), list_of(first_val.all_params()));
    ASSERT_EQ(list_of(second_val.master_slice()), list_of(first_val.master_slice()));
}

static void latest_names_the_newest_and_two_are_kept(const Distributed &distributed_val)
{
    // Expect: saves at steps 1, 2 and 3. `latest` says step_00000003; the folders of steps 2 and 3 are there,
    // step 1's was deleted (keep 2); no .partial file is left.
    // 1. three saves
    std::filesystem::remove_all("test_checkpoint_keep");
    write_token_file("test_checkpoint_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    AdamW adamw_val(SHAPES, config_val, DType::BFloat16);
    DataLoader loader_val("test_checkpoint_tokens.bin", 2, 4);
    for (int steps_done_val = 1; steps_done_val <= 3; steps_done_val++)
        checkpoint_save("test_checkpoint_keep", steps_done_val, 10, adamw_val, loader_val, distributed_val);
    // 2. check what is left
    ASSERT_EQ(text_of("test_checkpoint_keep/latest") == "step_00000003", true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000001"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000002/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000003/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/step_00000003/rank_0.bin.partial"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_keep/latest.partial"), false);
}

static void a_checkpoint_of_another_run_is_refused(const Distributed &distributed_val)
{
    // Expect: a checkpoint of a 10-step bfloat16 run on 1 GPU. checkpoint_problem finds nothing wrong for the
    // same run; but for a 20-step run (its learning rate schedule is another), for rank 0 of 2 GPUs, for a
    // float run, and for a folder with no checkpoint, it says what is wrong (printed below each check), and
    // checkpoint_load would stop with that.
    // 1. a save of a 10-step bfloat16 run on 1 GPU
    std::filesystem::remove_all("test_checkpoint_other");
    write_token_file("test_checkpoint_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    AdamW same_val(SHAPES, config_val, DType::BFloat16), two_gpus_val(SHAPES, config_val, DType::BFloat16, 0, 2);
    AdamW in_float_val(SHAPES, config_val, DType::Float32);
    DataLoader loader_val("test_checkpoint_tokens.bin", 2, 4);
    checkpoint_save("test_checkpoint_other", 1, 10, same_val, loader_val, distributed_val);
    // 2. the same run: no problem
    ASSERT_EQ(checkpoint_problem("test_checkpoint_other", 10, same_val).empty(), true);
    // 3. every other run: a problem, printed
    for (std::string problem_val : {checkpoint_problem("test_checkpoint_other", 20, same_val),
                                    checkpoint_problem("test_checkpoint_other", 10, two_gpus_val),
                                    checkpoint_problem("test_checkpoint_other", 10, in_float_val),
                                    checkpoint_problem("test_checkpoint_nothing", 10, same_val)})
    {
        ASSERT_EQ(problem_val.empty(), false);
        printf("        %s\n", problem_val.c_str());
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
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    Distributed distributed_val("");                        // no RANK set: one process, no NCCL
    loading_goes_on_exactly_where_saving_stopped(distributed_val);
    latest_names_the_newest_and_two_are_kept(distributed_val);
    a_checkpoint_of_another_run_is_refused(distributed_val);
    folder_names_sort_as_the_steps_do();

    printf("\nall tests passed\n");
    return 0;
}
