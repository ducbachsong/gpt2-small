// test_dataloader.cu — tests for llmc/dataloader.cuh (DataLoader). Every check prints its test case, input,
// expected and actual value, and ok; the first FAIL stops the program.
//
// The last four cases have a writer at work (a `writing` file, and files written by a child process a
// second later), so they take about 3 seconds.
//
// Each test writes its own small token files (test_dataloader_*.bin, in the folder it runs from), with
// tokens counting up, 0, 1, 2, ..., so every batch can be worked out by hand. Batches are {2, 4}: 8 tokens,
// read as 9. The first four cases are TokenFeedTests.cs's (the C# branch), for files instead of Python.
//
// Needs Linux (Colab): the cases where the loader must stop with an error run it in a child process, with
// fork(). No libtorch:
//
//     nvcc -O3 -std=c++17 -arch=native tests/test_dataloader.cu -o test_dataloader && ./test_dataloader
#include <string>
#include <sys/wait.h>                 // waitpid, WIFEXITED, WEXITSTATUS
#include <unistd.h>                   // fork, _exit
#include "test_helpers.h"             // ASSERT_EQ
#include "../llmc/dataloader.cuh"

#define BATCH_TOKENS 8                // a {2, 4} batch's token_ids; it reads one more for the targets

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives `count_val` numbers counting up from `first_val`: count_up(100, 3) is {100, 101, 102}. The tokens of a
/// test file, and the ids and targets expected from it.
static std::vector<int> count_up(int first_val, int count_val)
{
    std::vector<int> values_list(count_val);
    for (int i_val = 0; i_val < count_val; i_val++)
        values_list[i_val] = first_val + i_val;
    return values_list;
}

/// Writes a token file as data/fineweb.py does: the TOKEN_FILE_HEADER_INTS-int header, then each token as a
/// uint16. magic_val and header_tokens_val (the N the header says; -1 for the real count) can be set wrong, for the
/// tests of bad files.
static void write_token_file(const std::string &path_val, const std::vector<int> &tokens_list,
                             int magic_val = TOKEN_FILE_MAGIC, int header_tokens_val = -1)
{
    int header_list[TOKEN_FILE_HEADER_INTS] = {magic_val, TOKEN_FILE_VERSION,
                                               header_tokens_val >= 0 ? header_tokens_val : (int)tokens_list.size()};
    std::vector<uint16_t> values_list(tokens_list.begin(), tokens_list.end());
    FILE *file_ptr = fopen(path_val.c_str(), "wb");
    assert(file_ptr != nullptr && "cannot write a test file in this folder");
    fwrite(header_list, sizeof(int), TOKEN_FILE_HEADER_INTS, file_ptr);
    fwrite(values_list.data(), sizeof(uint16_t), values_list.size(), file_ptr);
    fclose(file_ptr);
}

/// Runs `run_val` in a child process and tells whether it stopped there with EXIT_FAILURE, as the loader does on
/// a bad file: fork() makes a copy of this program, the copy runs `run_val`, and this one waits for it to end.
/// The loader's error message shows on stderr. Called before anything touches the GPU, so the copy has
/// no GPU state.
template <typename Run>
static bool stops_with_an_error(Run run_val)
{
    fflush(stdout);                                          // else the copy prints this program's unprinted lines too
    pid_t child_val = fork();
    if (child_val == 0)
    {
        run_val();
        _exit(0);                                            // got through: no error
    }
    int status_val = 0;
    waitpid(child_val, &status_val, 0);
    return WIFEXITED(status_val) && WEXITSTATUS(status_val) == EXIT_FAILURE;
}

/// Writes a token file one second from now, from a child process, as data/fineweb.py does while a trainer
/// runs; then deletes `writing` if asked, as fineweb.py does at its end. With no tokens, writes nothing.
/// The child never touches the GPU.
static void write_token_file_soon(const std::string &path_val, const std::vector<int> &tokens_list, bool then_done_val)
{
    fflush(stdout);
    if (fork() == 0)
    {
        sleep(1);
        if (!tokens_list.empty())
            write_token_file(path_val, tokens_list);
        if (then_done_val)
            remove("writing");
        _exit(0);
    }
}

/// Makes the `writing` file in this folder, as data/fineweb.py does while it writes token files.
static void start_writing(void)
{
    FILE *marker_ptr = fopen("writing", "w");
    assert(marker_ptr != nullptr && "cannot write a test file in this folder");
    fclose(marker_ptr);
}

/// A batch's token_ids and targets, {rows, columns} ints, both 0 at first.
struct Batch
{
    Tensor token_ids_tensor;
    Tensor targets_tensor;
    explicit Batch(size_t rows_val = 2, size_t columns_val = 4)
        : token_ids_tensor(Tensor::zeros({rows_val, columns_val}, DType::Int32)),
          targets_tensor(Tensor::zeros({rows_val, columns_val}, DType::Int32))
    {
    }
    /// Reads the loader's next batch into these.
    void next_from(DataLoader &loader_val) { loader_val.next_batch(token_ids_tensor, targets_tensor); }
    std::vector<int> ids() const { return token_ids_tensor.tolist<int>(); }
    std::vector<int> targets() const { return targets_tensor.tolist<int>(); }
};

// ══ the cases that must stop with an error: first, before any GPU work ═════════

static void no_files_is_an_error(void)
{
    // C#: AnEmptyDatasetIsAnError. Expect: a pattern that matches no file, and a single path that is not
    // there, each stop with "no token files match".
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_none_%06d.bin", 2, 4); }), true);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_none.bin", 2, 4); }), true);
}

static void a_file_smaller_than_a_batch_is_refused(void)
{
    // C#: AMicroBatchLargerThanABatchIsRefused. Expect: a batch of {2, 4} reads 9 tokens; a file of 8 stops
    // with "fewer than one batch", and so does a pattern whose second file has 8. 9 is enough
    // (a_file_of_exactly_one_batch_gives_it_every_time).
    // 1. one file of 8
    write_token_file("test_dataloader_short.bin", count_up(0, BATCH_TOKENS));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_short.bin", 2, 4); }), true);
    // 2. a pattern whose second file has 8
    write_token_file("test_dataloader_short_000000.bin", count_up(0, 20));
    write_token_file("test_dataloader_short_000001.bin", count_up(0, BATCH_TOKENS));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_short_%06d.bin", 2, 4); }), true);
}

static void a_wrong_magic_number_is_an_error(void)
{
    // Expect: a file whose header starts with 12345 instead of 20240520 is not a token file: stops.
    write_token_file("test_dataloader_magic.bin", count_up(0, 20), 12345);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_magic.bin", 2, 4); }), true);
}

static void a_file_shorter_than_its_header_says_is_an_error(void)
{
    // Expect: a header that says 30 tokens on a file of 20 (1024 + 40 bytes, not 1024 + 60): a file cut
    // short while being written. Stops, rather than reading past its end later.
    write_token_file("test_dataloader_cut.bin", count_up(0, 20), TOKEN_FILE_MAGIC, 30);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_cut.bin", 2, 4); }), true);
}

static void a_file_smaller_than_a_round_is_refused(void)
{
    // Expect: with 2 ranks a round is 2 batches of {2, 4} and the next token: 17 tokens. A file of 16 is
    // enough for one rank alone, but stops here; and rank 2 of a world of 2 (ranks are 0 and 1) stops.
    // 1. a file of 16, 2 ranks
    write_token_file("test_dataloader_round.bin", count_up(0, 2 * BATCH_TOKENS));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_round.bin", 2, 4, 1, 2); }), true);
    // 2. a rank past the world
    write_token_file("test_dataloader_ranks.bin", count_up(0, 40));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader_val("test_dataloader_ranks.bin", 2, 4, 2, 2); }), true);
}

static void seek_to_a_place_no_loader_could_be_is_an_error(void)
{
    // Expect: a file of 40 tokens, batches of {2, 4}: a loader starts batches at 0, 8, 16, 24 only. seek to
    // token 4 (not the start of a batch), to token 32 (32 + 9 > 40: no batch fits) and to file 1 (there is
    // only file 0) each stop.
    write_token_file("test_dataloader_seek.bin", count_up(0, 40));
    ASSERT_EQ(stops_with_an_error([] { DataLoader("test_dataloader_seek.bin", 2, 4).seek(0, 4, 1); }), true);
    ASSERT_EQ(stops_with_an_error([] { DataLoader("test_dataloader_seek.bin", 2, 4).seek(0, 32, 1); }), true);
    ASSERT_EQ(stops_with_an_error([] { DataLoader("test_dataloader_seek.bin", 2, 4).seek(1, 0, 1); }), true);
}

// ══ batches ═══════════════════════════════════════════════════════════════════

static void batches_walk_a_file_in_order_then_start_again(void)
{
    // C#: MicroBatchesWalkEachBatchInOrderThenEnd; a loader never ends, it starts the files again. Expect,
    // from one file of tokens 0 .. 19:
    //
    //     batch   token_ids         targets           then position
    //     1       0 1 2 3 4 5 6 7   1 2 3 4 5 6 7 8   8: 8 + 9 = 17 <= 20 tokens, one more batch fits
    //     2       8 ... 15          9 ... 16          16: 16 + 9 = 25 > 20, so back to the start, epoch 2
    //     3       0 ... 7           1 ... 8           tokens 17, 18, 19 never read

    // 1. a loader on one file
    write_token_file("test_dataloader_walk.bin", count_up(0, 20));
    DataLoader loader_val("test_dataloader_walk.bin", 2, 4);
    Batch batch_val;
    ASSERT_EQ(loader_val.num_files(), 1);
    ASSERT_EQ(loader_val.epoch(), 1);
    // 2. batch 1
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
    ASSERT_EQ(batch_val.targets(), count_up(1, BATCH_TOKENS));
    ASSERT_EQ(loader_val.position(), (size_t)8);
    // 3. batch 2: then back to the start
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(8, BATCH_TOKENS));
    ASSERT_EQ(batch_val.targets(), count_up(9, BATCH_TOKENS));
    ASSERT_EQ(loader_val.position(), (size_t)0);
    ASSERT_EQ(loader_val.epoch(), 2);
    // 4. batch 3: the first again
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
    ASSERT_EQ(batch_val.targets(), count_up(1, BATCH_TOKENS));
}

static void held_out_batches_are_the_same_after_every_reset(void)
{
    // C#: HeldOutIsTheFirstBatchsFirstRows. Here the held-out tokens are a file of their own, the val file,
    // read by a loader of its own. Expect: two batches, then reset(): the next batch is the first again,
    // token_ids 0 .. 7, at file 0, position 8, epoch 1, so every val loss reads the same tokens.
    // 1. two batches
    write_token_file("test_dataloader_val.bin", count_up(0, 40));
    DataLoader loader_val("test_dataloader_val.bin", 2, 4);
    Batch batch_val;
    batch_val.next_from(loader_val);
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(8, BATCH_TOKENS));
    // 2. reset, then the first batch again
    loader_val.reset();
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
    ASSERT_EQ(batch_val.targets(), count_up(1, BATCH_TOKENS));
    ASSERT_EQ(loader_val.current_file(), 0);
    ASSERT_EQ(loader_val.position(), (size_t)8);
    ASSERT_EQ(loader_val.epoch(), 1);
}

static void the_short_tail_of_a_file_is_skipped_then_the_next_file_comes(void)
{
    // Expect: a pattern with files 1 and 2 and no file 0, as fineweb.py writes the train files: file 1 holds
    // tokens 0 .. 19, file 2 tokens 100 .. 129; 50 tokens in all.
    //
    //     batch   file   token_ids        targets          then
    //     1       1      0 ... 7          1 ... 8          position 8
    //     2       1      8 ... 15         9 ... 16         16 + 9 > 20: file 2 (tokens 17, 18, 19 skipped)
    //     3       2      100 ... 107      101 ... 108      position 8
    //     4       2      108 ... 115      109 ... 116      position 16: 16 + 9 = 25 <= 30
    //     5       2      116 ... 123      117 ... 124      24 + 9 > 30: file 1 again, epoch 2
    //     6       1      0 ... 7          1 ... 8

    // 1. a loader on the two files
    write_token_file("test_dataloader_files_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_files_000002.bin", count_up(100, 30));
    DataLoader loader_val("test_dataloader_files_%06d.bin", 2, 4);
    Batch batch_val;
    ASSERT_EQ(loader_val.num_files(), 2);
    ASSERT_EQ(loader_val.file_path(0) == "test_dataloader_files_000001.bin", true);    // print_value has no string
    ASSERT_EQ(loader_val.file_path(1) == "test_dataloader_files_000002.bin", true);
    ASSERT_EQ(loader_val.file_tokens(0), (size_t)20);
    ASSERT_EQ(loader_val.file_tokens(1), (size_t)30);
    ASSERT_EQ(loader_val.total_tokens(), (size_t)50);
    // 2. six batches, as in the table
    int first_ids_list[6] = {0, 8, 100, 108, 116, 0};
    int files_after_list[6] = {0, 1, 1, 1, 0, 0};
    for (int batch_index_val = 0; batch_index_val < 6; batch_index_val++)
    {
        batch_val.next_from(loader_val);
        ASSERT_EQ(batch_val.ids(), count_up(first_ids_list[batch_index_val], BATCH_TOKENS));
        ASSERT_EQ(batch_val.targets(), count_up(first_ids_list[batch_index_val] + 1, BATCH_TOKENS));
        ASSERT_EQ(loader_val.current_file(), files_after_list[batch_index_val]);
    }
    ASSERT_EQ(loader_val.epoch(), 2);
}

// ══ files still being written: training and data/fineweb.py at the same time ═══

static void a_file_written_after_the_start_is_read_next(void)
{
    // Expect: `writing` there and only file 1 (tokens 0 .. 19) when the loader starts: 1 file, more coming.
    // File 2 (tokens 100 .. 129) written after that is found at the end of file 1: batch 3 is 100 .. 107
    // from file 2, still epoch 1, not file 1 again.
    // 1. a loader started with one file, more coming
    start_writing();
    write_token_file("test_dataloader_growing_000001.bin", count_up(0, 20));
    DataLoader loader_val("test_dataloader_growing_%06d.bin", 2, 4);
    Batch batch_val;
    ASSERT_EQ(loader_val.num_files(), 1);
    ASSERT_EQ(loader_val.more_files_coming(), true);
    // 2. file 2 written; batch 3 is its first
    write_token_file("test_dataloader_growing_000002.bin", count_up(100, 30));
    batch_val.next_from(loader_val);
    batch_val.next_from(loader_val);
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(100, BATCH_TOKENS));
    ASSERT_EQ(loader_val.num_files(), 2);
    ASSERT_EQ(loader_val.current_file(), 1);
    ASSERT_EQ(loader_val.epoch(), 1);
    // 3. the writer done
    remove("writing");
    ASSERT_EQ(loader_val.more_files_coming(), false);
}

static void a_half_written_file_is_waited_for_until_it_is_whole(void)
{
    // Expect: `writing` there, file 1 whole, file 2 half written (its header says 30 tokens, 10 are there):
    // the loader starts with file 1 alone, no error. At the end of file 1 it waits; a second later the
    // writer finishes file 2 (tokens 100 .. 129) and deletes `writing`: batch 3 is 100 .. 107 from file 2.
    // 1. a loader started with file 1 whole and file 2 half written
    start_writing();
    write_token_file("test_dataloader_half_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_half_000002.bin", count_up(100, 10), TOKEN_FILE_MAGIC, 30);
    DataLoader loader_val("test_dataloader_half_%06d.bin", 2, 4);
    Batch batch_val;
    ASSERT_EQ(loader_val.num_files(), 1);
    // 2. file 2 finished a second later; batch 3 is its first
    write_token_file_soon("test_dataloader_half_000002.bin", count_up(100, 30), true);
    batch_val.next_from(loader_val);
    batch_val.next_from(loader_val);                         // the end of file 1: waits for file 2
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(100, BATCH_TOKENS));
    ASSERT_EQ(batch_val.targets(), count_up(101, BATCH_TOKENS));
    ASSERT_EQ(loader_val.num_files(), 2);
    ASSERT_EQ(loader_val.epoch(), 1);
    ASSERT_EQ(loader_val.more_files_coming(), false);
}

static void the_first_file_is_waited_for_when_there_is_none_yet(void)
{
    // Expect: `writing` there and no file at all: the loader waits instead of stopping with "no token
    // files"; a second later file 1 (tokens 0 .. 19) is written and `writing` deleted: it starts with it.
    // 1. no file yet; file 1 written a second later
    start_writing();
    write_token_file_soon("test_dataloader_first_000001.bin", count_up(0, 20), true);
    // 2. the loader waits for it, then reads it
    DataLoader loader_val("test_dataloader_first_%06d.bin", 2, 4);
    Batch batch_val;
    ASSERT_EQ(loader_val.num_files(), 1);
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
}

static void when_the_writer_ends_with_no_new_file_the_first_file_comes_again(void)
{
    // Expect: `writing` there, file 1 alone. At the end of file 1 the loader waits; a second later the writer
    // deletes `writing` with no new file: the loader goes back to file 1, epoch 2, batch 3 is 0 .. 7.
    // 1. a loader on file 1, the writer ending a second later with no new file
    start_writing();
    write_token_file("test_dataloader_ends_000001.bin", count_up(0, 20));
    DataLoader loader_val("test_dataloader_ends_%06d.bin", 2, 4);
    Batch batch_val;
    write_token_file_soon("", {}, true);
    // 2. three batches: the third is file 1's first again
    batch_val.next_from(loader_val);
    batch_val.next_from(loader_val);                         // the end of file 1: waits, then the writer ends
    batch_val.next_from(loader_val);
    ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
    ASSERT_EQ(loader_val.num_files(), 1);
    ASSERT_EQ(loader_val.epoch(), 2);
}

static void a_file_of_exactly_one_batch_gives_it_every_time(void)
{
    // Expect: a file of 9 tokens, 0 .. 8, is exactly one {2, 4} batch: token_ids 0 .. 7, targets 1 .. 8.
    // After it 8 + 9 > 9, so every batch is that one, and each starts a new epoch.
    // 1. a loader on 9 tokens
    write_token_file("test_dataloader_one.bin", count_up(0, BATCH_TOKENS + 1));
    DataLoader loader_val("test_dataloader_one.bin", 2, 4);
    Batch batch_val;
    // 2. three batches, each the same, each a new epoch
    for (int epoch_val = 2; epoch_val <= 4; epoch_val++)
    {
        batch_val.next_from(loader_val);
        ASSERT_EQ(batch_val.ids(), count_up(0, BATCH_TOKENS));
        ASSERT_EQ(batch_val.targets(), count_up(1, BATCH_TOKENS));
        ASSERT_EQ(loader_val.epoch(), epoch_val);
    }
}

static void ids_above_32767_stay_positive(void)
{
    // Expect: tokens are uint16s, 0 .. 65535. Read as int16s, 50256 would come out -15280; read as uint16s
    // and turned into ints, they stay as they are. File 50256, 32768, 65535, 0, 1 with a batch of {1, 4}:
    // token_ids 50256, 32768, 65535, 0, targets 32768, 65535, 0, 1.
    // 1. one batch of {1, 4}
    write_token_file("test_dataloader_large.bin", {50256, 32768, 65535, 0, 1});
    DataLoader loader_val("test_dataloader_large.bin", 1, 4);
    Batch batch_val(1, 4);
    batch_val.next_from(loader_val);
    // 2. check
    ASSERT_EQ(batch_val.ids(), std::vector<int>({50256, 32768, 65535, 0}));
    ASSERT_EQ(batch_val.targets(), std::vector<int>({32768, 65535, 0, 1}));
}

static void ranks_take_turns_batch_by_batch(void)
{
    // Expect: 2 ranks on one file of tokens 0 .. 39, batches of {2, 4}. In each round rank 0 reads the first
    // 8 tokens and rank 1 the next 8, then the round moves 16 on:
    //
    //     round   rank 0 token_ids   rank 1 token_ids   then position
    //     1       0 ... 7            8 ... 15           16: 16 + 17 = 33 <= 40, one more round fits
    //     2       16 ... 23          24 ... 31          32: 32 + 17 = 49 > 40, back to the start, epoch 2
    //     3       0 ... 7            8 ... 15
    //
    // Rounds 1 and 2 together are tokens 0 .. 31 once each: what one loader alone reads in 4 batches.
    // Both loaders keep the same position and epoch. (Tokens 32 .. 39 are never read.)
    // 1. a loader for each of 2 ranks
    write_token_file("test_dataloader_ranks.bin", count_up(0, 40));
    DataLoader rank0_val("test_dataloader_ranks.bin", 2, 4, 0, 2), rank1_val("test_dataloader_ranks.bin", 2, 4, 1, 2);
    Batch batch_val;
    // 2. three rounds, as in the table
    int rank0_first_list[3] = {0, 16, 0};
    int rank1_first_list[3] = {8, 24, 8};
    size_t positions_after_list[3] = {16, 0, 16};
    int epochs_after_list[3] = {1, 2, 2};
    for (int round_val = 0; round_val < 3; round_val++)
    {
        batch_val.next_from(rank0_val);
        ASSERT_EQ(batch_val.ids(), count_up(rank0_first_list[round_val], BATCH_TOKENS));
        ASSERT_EQ(batch_val.targets(), count_up(rank0_first_list[round_val] + 1, BATCH_TOKENS));
        batch_val.next_from(rank1_val);
        ASSERT_EQ(batch_val.ids(), count_up(rank1_first_list[round_val], BATCH_TOKENS));
        ASSERT_EQ(batch_val.targets(), count_up(rank1_first_list[round_val] + 1, BATCH_TOKENS));
        ASSERT_EQ(rank0_val.position(), positions_after_list[round_val]);
        ASSERT_EQ(rank1_val.position(), positions_after_list[round_val]);
        ASSERT_EQ(rank1_val.epoch(), epochs_after_list[round_val]);
    }
}

static void seek_continues_where_another_loader_was(void)
{
    // Expect: the_short_tail_of_a_file_is_skipped_then_the_next_file_comes's two files. A loader that has read 3
    // batches is at file 1 (the second), token 8, epoch 1, as a checkpoint saves it. A new loader put there with
    // seek gives the same next 3 batches as the first one: token_ids 108 .., 116 .., then 0 .. (epoch 2).
    // 1. the first loader reads 3 batches
    write_token_file("test_dataloader_files_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_files_000002.bin", count_up(100, 30));
    DataLoader first_val("test_dataloader_files_%06d.bin", 2, 4), second_val("test_dataloader_files_%06d.bin", 2, 4);
    Batch first_batch_val;
    for (int batch_index_val = 0; batch_index_val < 3; batch_index_val++)
        first_batch_val.next_from(first_val);
    ASSERT_EQ(first_val.current_file(), 1);
    ASSERT_EQ(first_val.position(), (size_t)8);
    // 2. the second seeks there; the next 3 batches are the same
    second_val.seek(first_val.current_file(), first_val.position(), first_val.epoch());
    Batch second_batch_val;
    for (int batch_index_val = 0; batch_index_val < 3; batch_index_val++)
    {
        first_batch_val.next_from(first_val);
        second_batch_val.next_from(second_val);
        ASSERT_EQ(second_batch_val.ids(), first_batch_val.ids());
        ASSERT_EQ(second_batch_val.targets(), first_batch_val.targets());
        ASSERT_EQ(second_val.epoch(), first_val.epoch());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    remove("writing");                                       // one left by a stopped run would make loaders wait
    // the error cases first: fork() before this program has touched the GPU
    no_files_is_an_error();
    a_file_smaller_than_a_batch_is_refused();
    a_wrong_magic_number_is_an_error();
    a_file_shorter_than_its_header_says_is_an_error();
    a_file_smaller_than_a_round_is_refused();
    seek_to_a_place_no_loader_could_be_is_an_error();

    batches_walk_a_file_in_order_then_start_again();
    held_out_batches_are_the_same_after_every_reset();
    the_short_tail_of_a_file_is_skipped_then_the_next_file_comes();
    a_file_of_exactly_one_batch_gives_it_every_time();
    ids_above_32767_stay_positive();
    ranks_take_turns_batch_by_batch();
    seek_continues_where_another_loader_was();

    a_file_written_after_the_start_is_read_next();
    a_half_written_file_is_waited_for_until_it_is_whole();
    the_first_file_is_waited_for_when_there_is_none_yet();
    when_the_writer_ends_with_no_new_file_the_first_file_comes_again();

    printf("\nall tests passed\n");
    return 0;
}
