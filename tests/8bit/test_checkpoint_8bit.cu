// test_checkpoint_8bit.cu — tests for llmc/helper/checkpoint_8bit.cuh (checkpoint_8bit_save, checkpoint_8bit_load,
// checkpoint_8bit_problem), the 8-bit run's own checkpoints: the cases of tests/test_checkpoint.cu (the bfloat16
// run's), on AdamW8bit and a few rounding histories. Every check prints its test case, input, expected and actual
// value, and ok; the first FAIL stops the program.
//
// One GPU, one process (Distributed with no RANK set); NCCL is linked but not used. Each test writes its own
// token file and checkpoint folders in the folder it runs from (test_checkpoint_8bit_*).
//
//     nvcc -O3 -std=c++17 -arch=sm_90a tests/8bit/test_checkpoint_8bit.cu -o test_checkpoint_8bit -lnccl && ./test_checkpoint_8bit
#include <filesystem>
#include "../test_helpers.h"                     // ASSERT_EQ
#include "../../llmc/helper/checkpoint_8bit.cuh"
#include "../../llmc/model_bf16.cuh"             // AdamWBf16: a bfloat16 run's checkpoint, refused

// A {6, 4} weight and an {8} bias, 32 values: the smallest AdamW with a matrix and a vector. 32 values take one
// scale of m and one of sqrt(v) (ADAMW_VALUES_PER_SCALE is more).
static const std::vector<Shape> SHAPES = {{6, 4}, {8}};
#define SHAPES_VALUES 32              // the values SHAPES holds: 6 x 4 + 8
#define TOKENS_IN_FILE 40             // the token file's tokens: 5 batches of {2, 4}
#define HISTORIES 3                   // the rounding histories each test keeps: a model's are 8 a layer, then 2

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives SHAPES_VALUES values that differ from test to test by `seed_val`, between -0.5 and 0.5: params or grads.
static std::vector<float> some_values(int seed_val)
{
    std::vector<float> values_list(SHAPES_VALUES);
    for (int value_index_val = 0; value_index_val < SHAPES_VALUES; value_index_val++)
        values_list[value_index_val] = (float)((value_index_val * 37 + seed_val * 11) % 17 - 8) / 16.0f;
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
    for (int value_index_val = 0; value_index_val < count_val; value_index_val++)
        tokens_list[value_index_val] = (uint16_t)value_index_val;
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

/// Gives every byte of an E4M3 tensor (m or sqrt(v) as kept) as an int list on the CPU: equal lists, the same bits.
static std::vector<int> bytes_of(const Tensor &tensor)
{
    std::vector<__nv_fp8_e4m3> values_list = tensor.tolist<__nv_fp8_e4m3>();
    std::vector<int> bytes_list(values_list.size());
    for (size_t value_index_val = 0; value_index_val < values_list.size(); value_index_val++)
        bytes_list[value_index_val] = values_list[value_index_val].__x;
    return bytes_list;
}

/// Tells if any byte of the list is not 0: m and sqrt(v) after a step, so the checks on them are not of zeros.
static bool any_byte_set(const std::vector<int> &bytes_list)
{
    for (int byte_val : bytes_list)
        if (byte_val != 0)
            return true;
    return false;
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
    // Expect: an AdamW8bit and a loader on a file of 40 tokens (batches of {2, 4}) take 2 steps, with 3 rounding
    // histories (one started at scale 1, one with largests {2.5, 7} after 5 calls, one never used); then a save at
    // step 2 of a 10-step run. A new AdamW8bit, histories and loader load it: 2 steps done, the same master, the
    // same m and sqrt(v) bytes and scales, the same bfloat16 params, the same histories, the loader at the same
    // file, token 16, epoch 1. Then both take step 3 on the same grads: the same next batch (token_ids 16 .. 23)
    // and the same params after it, to the last bit. The file holds a 256-byte header, 32 floats of master, 32
    // bytes and 1 scale for m, the same for sqrt(v), 3 x 2 float largests and 3 int64 calls:
    // 256 + 128 + 36 + 36 + 24 + 24 = 504 bytes.

    // 1. two steps, the histories given values, then a save at step 2
    std::filesystem::remove_all("test_checkpoint_8bit_run");
    write_token_file("test_checkpoint_8bit_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    config_val.learning_rate = 1e-2f;
    AdamW8bit first_val(SHAPES, config_val), second_val(SHAPES, config_val);
    std::vector<Round8bitHistory> first_histories_list(HISTORIES), second_histories_list(HISTORIES);
    round_8bit_start_at_scale_one(first_histories_list[0]);
    std::vector<float> largests_list = {2.5f, 7.0f};
    first_histories_list[1].largest_tensor.copy_from_cpu(largests_list.data());
    first_histories_list[1].calls_val = 5;
    DataLoader first_loader_val("test_checkpoint_8bit_tokens.bin", 2, 4), second_loader_val("test_checkpoint_8bit_tokens.bin", 2, 4);
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
    ASSERT_EQ(any_byte_set(bytes_of(first_val.m_bytes())), true);
    ASSERT_EQ(any_byte_set(bytes_of(first_val.sqrt_v_bytes())), true);
    checkpoint_8bit_save("test_checkpoint_8bit_run", 2, 10, first_val, first_histories_list, first_loader_val, distributed_val);
    size_t scales_val = CEIL_DIV((size_t)SHAPES_VALUES, (size_t)ADAMW_VALUES_PER_SCALE);   // 1
    size_t expected_bytes_val = CHECKPOINT_HEADER_INTS * sizeof(int64_t) + SHAPES_VALUES * sizeof(float) +
                                2 * (SHAPES_VALUES + scales_val * sizeof(float)) +
                                HISTORIES * (2 * sizeof(float) + sizeof(int64_t));                    // 504
    ASSERT_EQ((size_t)std::filesystem::file_size("test_checkpoint_8bit_run/step_00000002/rank_0.bin"), expected_bytes_val);

    // 2. a load into a new AdamW8bit, histories and loader: the same state
    int steps_done_val = checkpoint_8bit_load("test_checkpoint_8bit_run", 10, second_val, second_histories_list,
                                              second_loader_val, distributed_val);
    ASSERT_EQ(steps_done_val, 2);
    ASSERT_EQ(second_val.steps_taken(), 2);
    ASSERT_EQ(list_of(second_val.master_slice()), list_of(first_val.master_slice()));
    ASSERT_EQ(bytes_of(second_val.m_bytes()), bytes_of(first_val.m_bytes()));
    ASSERT_EQ(list_of(second_val.m_scales()), list_of(first_val.m_scales()));
    ASSERT_EQ(bytes_of(second_val.sqrt_v_bytes()), bytes_of(first_val.sqrt_v_bytes()));
    ASSERT_EQ(list_of(second_val.sqrt_v_scales()), list_of(first_val.sqrt_v_scales()));
    ASSERT_EQ(list_of(second_val.all_params()), list_of(first_val.all_params()));
    for (int history_index_val = 0; history_index_val < HISTORIES; history_index_val++)
    {
        ASSERT_EQ(list_of(second_histories_list[history_index_val].largest_tensor),
                  list_of(first_histories_list[history_index_val].largest_tensor));
        ASSERT_EQ(second_histories_list[history_index_val].calls_val, first_histories_list[history_index_val].calls_val);
    }
    ASSERT_EQ(second_loader_val.current_file(), first_loader_val.current_file());
    ASSERT_EQ(second_loader_val.position(), (size_t)16);
    ASSERT_EQ(second_loader_val.epoch(), 1);

    // 3. step 3 on both: the same batch and the same params, master, m and sqrt(v) after it
    Tensor second_ids_tensor = Tensor::zeros({2, 4}, DType::Int32);
    Tensor second_targets_tensor = Tensor::zeros({2, 4}, DType::Int32);
    first_loader_val.next_batch(token_ids_tensor, targets_tensor);
    second_loader_val.next_batch(second_ids_tensor, second_targets_tensor);
    ASSERT_EQ(second_ids_tensor.tolist<int>(), token_ids_tensor.tolist<int>());
    for (AdamW8bit *adamw_ptr : {&first_val, &second_val})
    {
        adamw_ptr->all_grads().copy_(tensor_of(some_values(3)));
        adamw_ptr->step();
    }
    ASSERT_EQ(list_of(second_val.all_params()), list_of(first_val.all_params()));
    ASSERT_EQ(list_of(second_val.master_slice()), list_of(first_val.master_slice()));
    ASSERT_EQ(bytes_of(second_val.m_bytes()), bytes_of(first_val.m_bytes()));
    ASSERT_EQ(bytes_of(second_val.sqrt_v_bytes()), bytes_of(first_val.sqrt_v_bytes()));
}

static void latest_names_the_newest_and_two_are_kept(const Distributed &distributed_val)
{
    // Expect: saves at steps 1, 2 and 3. `latest` says step_00000003; the folders of steps 2 and 3 are there,
    // step 1's was deleted (keep 2); no .partial file is left.
    // 1. three saves
    std::filesystem::remove_all("test_checkpoint_8bit_keep");
    write_token_file("test_checkpoint_8bit_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    AdamW8bit adamw_val(SHAPES, config_val);
    std::vector<Round8bitHistory> histories_list(HISTORIES);
    DataLoader loader_val("test_checkpoint_8bit_tokens.bin", 2, 4);
    for (int steps_done_val = 1; steps_done_val <= 3; steps_done_val++)
        checkpoint_8bit_save("test_checkpoint_8bit_keep", steps_done_val, 10, adamw_val, histories_list, loader_val,
                             distributed_val);
    // 2. check what is left
    ASSERT_EQ(text_of("test_checkpoint_8bit_keep/latest") == "step_00000003", true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_8bit_keep/step_00000001"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_8bit_keep/step_00000002/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_8bit_keep/step_00000003/rank_0.bin"), true);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_8bit_keep/step_00000003/rank_0.bin.partial"), false);
    ASSERT_EQ(std::filesystem::exists("test_checkpoint_8bit_keep/latest.partial"), false);
}

static void a_checkpoint_of_another_run_is_refused(const Distributed &distributed_val)
{
    // Expect: a checkpoint of a 10-step 8-bit run on 1 GPU with 3 rounding histories. checkpoint_8bit_problem finds
    // nothing wrong for the same run; but for a 20-step run (its learning rate schedule is another), for rank 0 of
    // 2 GPUs, for a model with 4 histories (another depth), for a bfloat16 run's checkpoint (checkpoint.cuh's: its m
    // and v are tensors, not bytes and scales), and for a folder with no checkpoint, it says what is wrong (printed
    // below each check), and checkpoint_8bit_load would stop with that. The bfloat16 run's own checkpoint_problem
    // refuses the 8-bit file too.
    // 1. a save of a 10-step 8-bit run on 1 GPU, and of a bfloat16 run
    std::filesystem::remove_all("test_checkpoint_8bit_other");
    std::filesystem::remove_all("test_checkpoint_8bit_bf16");
    write_token_file("test_checkpoint_8bit_tokens.bin", TOKENS_IN_FILE);
    AdamWConfig config_val;
    AdamW8bit same_val(SHAPES, config_val), two_gpus_val(SHAPES, config_val, 0, 2);
    AdamWBf16 in_bf16_val(SHAPES, config_val);
    std::vector<Round8bitHistory> histories_list(HISTORIES);
    DataLoader loader_val("test_checkpoint_8bit_tokens.bin", 2, 4);
    checkpoint_8bit_save("test_checkpoint_8bit_other", 1, 10, same_val, histories_list, loader_val, distributed_val);
    checkpoint_save("test_checkpoint_8bit_bf16", 1, 10, in_bf16_val, loader_val, distributed_val);
    // 2. the same run: no problem
    ASSERT_EQ(checkpoint_8bit_problem("test_checkpoint_8bit_other", 10, same_val, HISTORIES).empty(), true);
    // 3. every other run: a problem, printed
    for (std::string problem_val : {checkpoint_8bit_problem("test_checkpoint_8bit_other", 20, same_val, HISTORIES),
                                    checkpoint_8bit_problem("test_checkpoint_8bit_other", 10, two_gpus_val, HISTORIES),
                                    checkpoint_8bit_problem("test_checkpoint_8bit_other", 10, same_val, HISTORIES + 1),
                                    checkpoint_8bit_problem("test_checkpoint_8bit_bf16", 10, same_val, HISTORIES),
                                    checkpoint_8bit_problem("test_checkpoint_8bit_nothing", 10, same_val, HISTORIES),
                                    checkpoint_problem("test_checkpoint_8bit_other", 10, in_bf16_val)})
    {
        ASSERT_EQ(problem_val.empty(), false);
        printf("        %s\n", problem_val.c_str());
    }
}

static void folder_names_sort_as_the_steps_do(void)
{
    // Expect: step_ and 8 digits, as the bfloat16 run's (the 8-bit run's folders use the same name): step 1000 is
    // "step_00001000", and step 999 sorts before step 1000 as text.
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
