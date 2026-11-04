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

// ── helpers ───────────────────────────────────────────────────────────────────

/// Gives `count` numbers counting up from `first`: count_up(100, 3) is {100, 101, 102}. The tokens of a test
/// file, and the ids and targets expected from it.
static std::vector<int> count_up(int first, int count)
{
    std::vector<int> values(count);
    for (int i = 0; i < count; i++)
        values[i] = first + i;
    return values;
}

/// Writes a token file as data/fineweb.py does: the 256-int header, then each token as a uint16. magic and
/// header_tokens (the N the header says; -1 for the real count) can be set wrong, for the tests of bad files.
static void write_token_file(const std::string &path, const std::vector<int> &tokens, int magic = TOKEN_FILE_MAGIC,
                             int header_tokens = -1)
{
    int header[TOKEN_FILE_HEADER_INTS] = {magic, TOKEN_FILE_VERSION,
                                          header_tokens >= 0 ? header_tokens : (int)tokens.size()};
    std::vector<uint16_t> values(tokens.begin(), tokens.end());
    FILE *file = fopen(path.c_str(), "wb");
    assert(file != nullptr && "cannot write a test file in this folder");
    fwrite(header, sizeof(int), TOKEN_FILE_HEADER_INTS, file);
    fwrite(values.data(), sizeof(uint16_t), values.size(), file);
    fclose(file);
}

/// Runs `run` in a child process and tells whether it stopped there with EXIT_FAILURE, as the loader does on
/// a bad file: fork() makes a copy of this program, the copy runs `run`, and this one waits for it to end.
/// The loader's error message shows on stderr. Called before anything touches the GPU, so the copy has
/// no GPU state.
template <typename Run>
static bool stops_with_an_error(Run run)
{
    fflush(stdout);                                          // else the copy prints this program's unprinted lines too
    pid_t child = fork();
    if (child == 0)
    {
        run();
        _exit(0);                                            // got through: no error
    }
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE;
}

/// Writes a token file one second from now, from a child process, as data/fineweb.py does while a trainer
/// runs; then deletes `writing` if asked, as fineweb.py does at its end. With no tokens, writes nothing.
/// The child never touches the GPU.
static void write_token_file_soon(const std::string &path, const std::vector<int> &tokens, bool then_done)
{
    fflush(stdout);
    if (fork() == 0)
    {
        sleep(1);
        if (!tokens.empty())
            write_token_file(path, tokens);
        if (then_done)
            remove("writing");
        _exit(0);
    }
}

/// Makes the `writing` file in this folder, as data/fineweb.py does while it writes token files.
static void start_writing(void)
{
    FILE *marker = fopen("writing", "w");
    assert(marker != nullptr && "cannot write a test file in this folder");
    fclose(marker);
}

// ══ the cases that must stop with an error: first, before any GPU work ═════════

static void no_files_is_an_error(void)
{
    // C#: AnEmptyDatasetIsAnError. Expect: a pattern that matches no file, and a single path that is not
    // there, each stop with "no token files match".
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_none_%06d.bin", 2, 4); }), true);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_none.bin", 2, 4); }), true);
}

static void a_file_smaller_than_a_batch_is_refused(void)
{
    // C#: AMicroBatchLargerThanABatchIsRefused. Expect: a batch of {2, 4} reads 9 tokens; a file of 8 stops
    // with "fewer than one batch", and so does a pattern whose second file has 8. 9 is enough
    // (a_file_of_exactly_one_batch_gives_it_every_time).
    write_token_file("test_dataloader_short.bin", count_up(0, 8));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_short.bin", 2, 4); }), true);
    write_token_file("test_dataloader_short_000000.bin", count_up(0, 20));
    write_token_file("test_dataloader_short_000001.bin", count_up(0, 8));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_short_%06d.bin", 2, 4); }), true);
}

static void a_wrong_magic_number_is_an_error(void)
{
    // Expect: a file whose header starts with 12345 instead of 20240520 is not a token file: stops.
    write_token_file("test_dataloader_magic.bin", count_up(0, 20), 12345);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_magic.bin", 2, 4); }), true);
}

static void a_file_shorter_than_its_header_says_is_an_error(void)
{
    // Expect: a header that says 30 tokens on a file of 20 (1024 + 40 bytes, not 1024 + 60): a file cut
    // short while being written. Stops, rather than reading past its end later.
    write_token_file("test_dataloader_cut.bin", count_up(0, 20), TOKEN_FILE_MAGIC, 30);
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_cut.bin", 2, 4); }), true);
}

static void a_file_smaller_than_a_round_is_refused(void)
{
    // Expect: with 2 ranks a round is 2 batches of {2, 4} and the next token: 17 tokens. A file of 16 is
    // enough for one rank alone, but stops here; and rank 2 of a world of 2 (ranks are 0 and 1) stops.
    write_token_file("test_dataloader_round.bin", count_up(0, 16));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_round.bin", 2, 4, 1, 2); }), true);
    write_token_file("test_dataloader_ranks.bin", count_up(0, 40));
    ASSERT_EQ(stops_with_an_error([] { DataLoader loader("test_dataloader_ranks.bin", 2, 4, 2, 2); }), true);
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
    write_token_file("test_dataloader_walk.bin", count_up(0, 20));
    DataLoader loader("test_dataloader_walk.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    ASSERT_EQ(loader.num_files(), 1);
    ASSERT_EQ(loader.epoch(), 1);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
    ASSERT_EQ(targets.tolist<int>(), count_up(1, 8));
    ASSERT_EQ(loader.position(), (size_t)8);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(8, 8));
    ASSERT_EQ(targets.tolist<int>(), count_up(9, 8));
    ASSERT_EQ(loader.position(), (size_t)0);
    ASSERT_EQ(loader.epoch(), 2);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
    ASSERT_EQ(targets.tolist<int>(), count_up(1, 8));
}

static void held_out_batches_are_the_same_after_every_reset(void)
{
    // C#: HeldOutIsTheFirstBatchsFirstRows. Here the held-out tokens are a file of their own, the val file,
    // read by a loader of its own. Expect: two batches, then reset(): the next batch is the first again,
    // token_ids 0 .. 7, at file 0, position 8, epoch 1, so every val loss reads the same tokens.
    write_token_file("test_dataloader_val.bin", count_up(0, 40));
    DataLoader loader("test_dataloader_val.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    loader.next_batch(token_ids, targets);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(8, 8));
    loader.reset();
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
    ASSERT_EQ(targets.tolist<int>(), count_up(1, 8));
    ASSERT_EQ(loader.current_file(), 0);
    ASSERT_EQ(loader.position(), (size_t)8);
    ASSERT_EQ(loader.epoch(), 1);
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
    write_token_file("test_dataloader_files_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_files_000002.bin", count_up(100, 30));
    DataLoader loader("test_dataloader_files_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    ASSERT_EQ(loader.num_files(), 2);
    ASSERT_EQ(loader.file_path(0) == "test_dataloader_files_000001.bin", true);    // print_value has no string
    ASSERT_EQ(loader.file_path(1) == "test_dataloader_files_000002.bin", true);
    ASSERT_EQ(loader.file_tokens(0), (size_t)20);
    ASSERT_EQ(loader.file_tokens(1), (size_t)30);
    ASSERT_EQ(loader.total_tokens(), (size_t)50);
    int first_ids[6] = {0, 8, 100, 108, 116, 0};
    int files_after[6] = {0, 1, 1, 1, 0, 0};
    for (int batch = 0; batch < 6; batch++)
    {
        loader.next_batch(token_ids, targets);
        ASSERT_EQ(token_ids.tolist<int>(), count_up(first_ids[batch], 8));
        ASSERT_EQ(targets.tolist<int>(), count_up(first_ids[batch] + 1, 8));
        ASSERT_EQ(loader.current_file(), files_after[batch]);
    }
    ASSERT_EQ(loader.epoch(), 2);
}

// ══ files still being written: training and data/fineweb.py at the same time ═══

static void a_file_written_after_the_start_is_read_next(void)
{
    // Expect: `writing` there and only file 1 (tokens 0 .. 19) when the loader starts: 1 file, more coming.
    // File 2 (tokens 100 .. 129) written after that is found at the end of file 1: batch 3 is 100 .. 107
    // from file 2, still epoch 1, not file 1 again.
    start_writing();
    write_token_file("test_dataloader_growing_000001.bin", count_up(0, 20));
    DataLoader loader("test_dataloader_growing_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    ASSERT_EQ(loader.num_files(), 1);
    ASSERT_EQ(loader.more_files_coming(), true);
    write_token_file("test_dataloader_growing_000002.bin", count_up(100, 30));
    loader.next_batch(token_ids, targets);
    loader.next_batch(token_ids, targets);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(100, 8));
    ASSERT_EQ(loader.num_files(), 2);
    ASSERT_EQ(loader.current_file(), 1);
    ASSERT_EQ(loader.epoch(), 1);
    remove("writing");
    ASSERT_EQ(loader.more_files_coming(), false);
}

static void a_half_written_file_is_waited_for_until_it_is_whole(void)
{
    // Expect: `writing` there, file 1 whole, file 2 half written (its header says 30 tokens, 10 are there):
    // the loader starts with file 1 alone, no error. At the end of file 1 it waits; a second later the
    // writer finishes file 2 (tokens 100 .. 129) and deletes `writing`: batch 3 is 100 .. 107 from file 2.
    start_writing();
    write_token_file("test_dataloader_half_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_half_000002.bin", count_up(100, 10), TOKEN_FILE_MAGIC, 30);
    DataLoader loader("test_dataloader_half_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    ASSERT_EQ(loader.num_files(), 1);
    write_token_file_soon("test_dataloader_half_000002.bin", count_up(100, 30), true);
    loader.next_batch(token_ids, targets);
    loader.next_batch(token_ids, targets);                   // the end of file 1: waits for file 2
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(100, 8));
    ASSERT_EQ(targets.tolist<int>(), count_up(101, 8));
    ASSERT_EQ(loader.num_files(), 2);
    ASSERT_EQ(loader.epoch(), 1);
    ASSERT_EQ(loader.more_files_coming(), false);
}

static void the_first_file_is_waited_for_when_there_is_none_yet(void)
{
    // Expect: `writing` there and no file at all: the loader waits instead of stopping with "no token
    // files"; a second later file 1 (tokens 0 .. 19) is written and `writing` deleted: it starts with it.
    start_writing();
    write_token_file_soon("test_dataloader_first_000001.bin", count_up(0, 20), true);
    DataLoader loader("test_dataloader_first_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    ASSERT_EQ(loader.num_files(), 1);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
}

static void when_the_writer_ends_with_no_new_file_the_first_file_comes_again(void)
{
    // Expect: `writing` there, file 1 alone. At the end of file 1 the loader waits; a second later the writer
    // deletes `writing` with no new file: the loader goes back to file 1, epoch 2, batch 3 is 0 .. 7.
    start_writing();
    write_token_file("test_dataloader_ends_000001.bin", count_up(0, 20));
    DataLoader loader("test_dataloader_ends_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    write_token_file_soon("", {}, true);
    loader.next_batch(token_ids, targets);
    loader.next_batch(token_ids, targets);                   // the end of file 1: waits, then the writer ends
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
    ASSERT_EQ(loader.num_files(), 1);
    ASSERT_EQ(loader.epoch(), 2);
}

static void a_file_of_exactly_one_batch_gives_it_every_time(void)
{
    // Expect: a file of 9 tokens, 0 .. 8, is exactly one {2, 4} batch: token_ids 0 .. 7, targets 1 .. 8.
    // After it 8 + 9 > 9, so every batch is that one, and each starts a new epoch.
    write_token_file("test_dataloader_one.bin", count_up(0, 9));
    DataLoader loader("test_dataloader_one.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    for (int epoch = 2; epoch <= 4; epoch++)
    {
        loader.next_batch(token_ids, targets);
        ASSERT_EQ(token_ids.tolist<int>(), count_up(0, 8));
        ASSERT_EQ(targets.tolist<int>(), count_up(1, 8));
        ASSERT_EQ(loader.epoch(), epoch);
    }
}

static void ids_above_32767_stay_positive(void)
{
    // Expect: tokens are uint16s, 0 .. 65535. Read as int16s, 50256 would come out -15280; read as uint16s
    // and turned into ints, they stay as they are. File 50256, 32768, 65535, 0, 1 with a batch of {1, 4}:
    // token_ids 50256, 32768, 65535, 0, targets 32768, 65535, 0, 1.
    write_token_file("test_dataloader_large.bin", {50256, 32768, 65535, 0, 1});
    DataLoader loader("test_dataloader_large.bin", 1, 4);
    Tensor token_ids = Tensor::zeros({1, 4}, DType::Int32), targets = Tensor::zeros({1, 4}, DType::Int32);
    loader.next_batch(token_ids, targets);
    ASSERT_EQ(token_ids.tolist<int>(), std::vector<int>({50256, 32768, 65535, 0}));
    ASSERT_EQ(targets.tolist<int>(), std::vector<int>({32768, 65535, 0, 1}));
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
    write_token_file("test_dataloader_ranks.bin", count_up(0, 40));
    DataLoader rank0("test_dataloader_ranks.bin", 2, 4, 0, 2), rank1("test_dataloader_ranks.bin", 2, 4, 1, 2);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    int rank0_first[3] = {0, 16, 0}, rank1_first[3] = {8, 24, 8};
    size_t positions_after[3] = {16, 0, 16};
    int epochs_after[3] = {1, 2, 2};
    for (int round = 0; round < 3; round++)
    {
        rank0.next_batch(token_ids, targets);
        ASSERT_EQ(token_ids.tolist<int>(), count_up(rank0_first[round], 8));
        ASSERT_EQ(targets.tolist<int>(), count_up(rank0_first[round] + 1, 8));
        rank1.next_batch(token_ids, targets);
        ASSERT_EQ(token_ids.tolist<int>(), count_up(rank1_first[round], 8));
        ASSERT_EQ(targets.tolist<int>(), count_up(rank1_first[round] + 1, 8));
        ASSERT_EQ(rank0.position(), positions_after[round]);
        ASSERT_EQ(rank1.position(), positions_after[round]);
        ASSERT_EQ(rank1.epoch(), epochs_after[round]);
    }
}

static void seek_continues_where_another_loader_was(void)
{
    // Expect: the_short_tail_of_a_file_is_skipped_then_the_next_file_comes's two files. A loader that has read 3
    // batches is at file 1 (the second), token 8, epoch 1, as a checkpoint saves it. A new loader put there with
    // seek gives the same next 3 batches as the first one: token_ids 108 .., 116 .., then 0 .. (epoch 2).
    write_token_file("test_dataloader_files_000001.bin", count_up(0, 20));
    write_token_file("test_dataloader_files_000002.bin", count_up(100, 30));
    DataLoader first("test_dataloader_files_%06d.bin", 2, 4), second("test_dataloader_files_%06d.bin", 2, 4);
    Tensor token_ids = Tensor::zeros({2, 4}, DType::Int32), targets = Tensor::zeros({2, 4}, DType::Int32);
    for (int batch = 0; batch < 3; batch++)
        first.next_batch(token_ids, targets);
    ASSERT_EQ(first.current_file(), 1);
    ASSERT_EQ(first.position(), (size_t)8);
    second.seek(first.current_file(), first.position(), first.epoch());
    Tensor second_ids = Tensor::zeros({2, 4}, DType::Int32), second_targets = Tensor::zeros({2, 4}, DType::Int32);
    for (int batch = 0; batch < 3; batch++)
    {
        first.next_batch(token_ids, targets);
        second.next_batch(second_ids, second_targets);
        ASSERT_EQ(second_ids.tolist<int>(), token_ids.tolist<int>());
        ASSERT_EQ(second_targets.tolist<int>(), targets.tolist<int>());
        ASSERT_EQ(second.epoch(), first.epoch());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void)
{
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
