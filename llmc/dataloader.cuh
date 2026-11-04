// dataloader.cuh — batches of tokens for training, read from the token files data/fineweb.py writes.
//
// A batch is batch_size rows of seq_len tokens, token_ids, and the same tokens moved one on, targets: at
// every position the model must predict the next token. Both come from B x T + 1 tokens in a row of a file:
//
//     tokens      [ 464  3290  286  6954  ...  13  50256  383 ]     B x T + 1 of them: 4097 for {4, 1024}
//     token_ids   [ 464  3290  286  6954  ...  13  50256 ]          the first B x T, as {B, T}
//     targets          [ 3290  286  6954  ...  13  50256  383 ]     the last B x T, as {B, T}
//
// The next batch starts B x T tokens on, so its first token is this batch's last target: no token is
// skipped. When a file has fewer than B x T + 1 tokens left, the loader moves to the next file; after the
// last file it starts the first one again (the next epoch). A 100M-token file gives 24,414 batches of
// {4, 1024}; its last 256 tokens are skipped.
//
// A token file, as llm.c writes them:
//
//     bytes 0 .. 1023      the header: 256 int32s. [0] = 20240520 (a token file), [1] = 1 (version),
//                          [2] = N, how many tokens follow; the rest 0
//     bytes 1024 ...       N tokens, a uint16 each (GPT-2's ids, 0 .. 50256, fit in 16 bits)
//
// so token i is at byte 1024 + 2 i. The files are found from a pattern with one %d, for the file's number:
//
//     "data/fineweb10B/fineweb_train_%06d.bin"   ->  fineweb_train_000001.bin, ..._000002.bin, ...
//
// starting at 0, or at 1 if there is no file 0 (fineweb.py's train files start at 1: 0 is the val file),
// up to the first number missing. A path without % is one file.
//
// On several GPUs (distributed.cuh), each process has a loader of its own, told its rank and the world
// size. They take turns: in each round, rank r reads batch r, then the round moves world_size batches on.
// With 2 ranks and batches of 4096 tokens:
//
//     round 1   rank 0: tokens 0 .. 4096       rank 1: tokens 4096 .. 8192
//     round 2   rank 0: tokens 8192 .. 12288   rank 1: tokens 12288 .. 16384
//
// so no token is read by two ranks, and together they read what one loader alone would. A file moves on
// when the next round does not fit: fewer than world_size x B x T + 1 tokens left.
//
// Training can start while data/fineweb.py is still writing the files. fineweb.py keeps a file named
// `writing` in the files' folder while it works; while it is there, the loader:
//
//     skips a file not yet whole (its size not what its header says): it is still being written
//     at the end of each file, looks for files written since, and waits for the next one if it is not there
//     yet (it gives up after an hour: a writer that stopped without deleting `writing`)
//
// Once `writing` is gone, the end of the last file goes back to the first one, the next epoch, as before.
// The writer makes about 3M tokens a second, far more than training reads, so the wait happens only at
// the start, for the first file.
#ifndef DATALOADER_CUH
#define DATALOADER_CUH

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include "tensor.cuh"

#define TOKEN_FILE_MAGIC 20240520            // header[0]: "this is a token file"
#define TOKEN_FILE_VERSION 1                 // header[1]
#define TOKEN_FILE_HEADER_INTS 256           // the header: 256 int32s, 1024 bytes

class DataLoader {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    DataLoader(const std::string& pattern, size_t batch_size, size_t seq_len,   // finds and checks the files
               int rank = 0, int world_size = 1);
    ~DataLoader();                                                      // closes the open file
    DataLoader(const DataLoader&) = delete;                             // one loader per open file
    DataLoader& operator=(const DataLoader&) = delete;

    // ── batches ─────────────────────────────────────────────────────────────────
    void next_batch(const Tensor& token_ids, const Tensor& targets);    // the next {B, T} of each, onto the GPU
    void reset();                                                       // back to file 0's first token, epoch 1
    void seek(int file, size_t position, int epoch);                    // to where a checkpoint saved it

    // ── reading ─────────────────────────────────────────────────────────────────
    int num_files() const;                                              // how many whole files it found so far
    const std::string& file_path(int k) const;                          // file k's path
    size_t file_tokens(int k) const;                                    // file k's N, from its header
    size_t total_tokens() const;                                        // all the files' N together, so far
    int current_file() const;                                           // the file the next round comes from
    size_t position() const;                                            // the next round's first token in it
    int epoch() const;                                                  // 1 on the first pass over the files
    bool more_files_coming() const;                                     // the writer is still writing files

private:
    size_t batch_size_, seq_len_;
    int rank_, world_size_;
    std::string pattern_;                    // the files' paths, with %d for the file's number
    std::string writing_marker_;             // <the files' folder>/writing: there while fineweb.py writes
    int next_number_ = 0;                    // the number to look for after the files found
    std::vector<std::string> file_paths_;
    std::vector<size_t> file_tokens_;
    int current_file_ = 0;
    size_t position_ = 0;
    int epoch_ = 1;
    FILE* file_ = nullptr;                   // the current file, open
    std::vector<uint16_t> file_values_;      // the B x T + 1 tokens just read, as in the file
    std::vector<int> token_id_values_;       // the first B x T of them as ints, for token_ids
    std::vector<int> target_values_;         // the last B x T, for targets

    void open_file(int k);                                              // file k, at its first token
    void find_new_files();                                              // adds the whole files written since
    void wait_for_file(int k);                                          // while the writer works, until file k
    std::string path_of(int number) const;                              // the pattern with this number put in
    static bool is_whole(const std::string& path);                      // its size is what its header says
    static size_t read_header(const std::string& path);                 // checks the header, gives N
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── making it ─────────────────────────────────────────────────────────────────

/// Finds the files the pattern matches, checks each one's header and that each holds at least one round
/// (world_size x B x T + 1 tokens), and opens the first. Stops with an error naming the file if one is
/// wrong. While the writer still works, it takes the whole files there are, and waits for the first if
/// there is none yet. Reads only the headers: no tokens and nothing on the GPU yet.
inline DataLoader::DataLoader(const std::string& pattern, size_t batch_size, size_t seq_len, int rank,
                              int world_size)
    : batch_size_(batch_size), seq_len_(seq_len), rank_(rank), world_size_(world_size), pattern_(pattern) {
    if (world_size < 1 || rank < 0 || rank >= world_size) {
        fprintf(stderr, "error: a data loader needs 0 <= rank < world_size; got rank %d of %d\n", rank, world_size);
        exit(EXIT_FAILURE);
    }
    size_t folder_end = pattern.find_last_of('/');
    writing_marker_ = (folder_end == std::string::npos ? "" : pattern.substr(0, folder_end + 1)) + "writing";
    wait_for_file(0);
    if (file_paths_.empty()) {
        fprintf(stderr, "error: no token files match %s (make them with data/fineweb.py)\n", pattern.c_str());
        exit(EXIT_FAILURE);
    }
    size_t batch_tokens = batch_size * seq_len;
    file_values_.resize(batch_tokens + 1);
    token_id_values_.resize(batch_tokens);
    target_values_.resize(batch_tokens);
    open_file(0);
}

/// Closes the file the loader has open.
inline DataLoader::~DataLoader() {
    if (file_ != nullptr) fclose(file_);
}

// ── batches ───────────────────────────────────────────────────────────────────

/// Writes this rank's next batch into token_ids and targets, both {batch_size, seq_len} ints on the GPU:
/// B x T + 1 tokens read from the current file, rank x B x T into the round, the first B x T into token_ids,
/// the last B x T into targets. Then moves the round on by world_size x B x T tokens, to the next file if
/// this one has fewer than world_size x B x T + 1 left (one written since counts; while the writer works it
/// waits for it). With one rank: B x T on, batch after batch.
inline void DataLoader::next_batch(const Tensor& token_ids, const Tensor& targets) {
    Shape batch = {batch_size_, seq_len_};
    if (token_ids.shape() != batch || targets.shape() != batch || token_ids.dtype() != DType::Int32 ||
        targets.dtype() != DType::Int32) {
        fprintf(stderr, "error: the data loader needs token_ids and targets {%zu, %zu} ints\n", batch_size_,
                seq_len_);
        exit(EXIT_FAILURE);
    }
    size_t batch_tokens = batch_size_ * seq_len_;
    size_t round_tokens = world_size_ * batch_tokens;
    size_t first = position_ + rank_ * batch_tokens;                    // this rank's batch in the round
    const std::string& path = file_paths_[current_file_];
    long byte = (long)(TOKEN_FILE_HEADER_INTS * sizeof(int) + first * sizeof(uint16_t));
    if (fseek(file_, byte, SEEK_SET) != 0 ||
        fread(file_values_.data(), sizeof(uint16_t), batch_tokens + 1, file_) != batch_tokens + 1) {
        fprintf(stderr, "error: cannot read tokens %zu .. %zu of %s\n", first, first + batch_tokens, path.c_str());
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < batch_tokens; i++) {
        token_id_values_[i] = file_values_[i];             // a uint16 into an int: 50256 stays 50256
        target_values_[i] = file_values_[i + 1];
    }
    token_ids.copy_from_cpu(token_id_values_.data());
    targets.copy_from_cpu(target_values_.data());

    position_ += round_tokens;
    if (position_ + round_tokens + 1 > file_tokens_[current_file_]) {   // not one more round here
        int next = current_file_ + 1;
        wait_for_file(next);                                            // written since, or still to come
        if (next >= num_files()) {                                      // the last file: the next epoch
            next = 0;
            epoch_++;
        }
        open_file(next);
    }
}

/// Goes back to the first token of the first file, epoch 1: the val loss reads the same batches every time.
inline void DataLoader::reset() {
    open_file(0);
    epoch_ = 1;
}

/// Puts the loader where a checkpoint saved it, current_file(), position() and epoch(), so the next batch is the
/// one it would have read: a resumed run trains on the tokens the stopped run had not reached yet. Stops if the
/// place is not one this loader could be at (the files changed, or another batch size).
inline void DataLoader::seek(int file, size_t position, int epoch) {
    if (file >= 0) wait_for_file(file);                                 // a file written since the start
    size_t round_tokens = world_size_ * batch_size_ * seq_len_;
    if (file < 0 || file >= num_files() || position % round_tokens != 0 ||
        position + round_tokens + 1 > file_tokens_[file] || epoch < 1) {
        fprintf(stderr, "error: the data loader cannot be at file %d, token %zu, epoch %d (%d files; rounds of %zu "
                        "tokens)\n", file, position, epoch, num_files(), round_tokens);
        exit(EXIT_FAILURE);
    }
    open_file(file);
    position_ = position;
    epoch_ = epoch;
}

// ── reading ───────────────────────────────────────────────────────────────────

/// Tells how many whole files the loader has found so far (more come while the writer works).
inline int DataLoader::num_files() const { return (int)file_paths_.size(); }

/// Gives file k's path, as the pattern made it.
inline const std::string& DataLoader::file_path(int k) const { return file_paths_.at(k); }

/// Tells how many tokens file k holds, from its header.
inline size_t DataLoader::file_tokens(int k) const { return file_tokens_.at(k); }

/// Tells how many tokens the files found so far hold together.
inline size_t DataLoader::total_tokens() const {
    size_t total = 0;
    for (size_t tokens : file_tokens_) total += tokens;
    return total;
}

/// Tells which file the next round comes from.
inline int DataLoader::current_file() const { return current_file_; }

/// Tells where the next round starts in the current file: the index of its first token, the same on every
/// rank (rank r's batch starts r x B x T after it). With one rank, where the next batch starts.
inline size_t DataLoader::position() const { return position_; }

/// Tells which pass over the files this is: 1 at first, 2 once the last file has been read and the
/// first one started again.
inline int DataLoader::epoch() const { return epoch_; }

/// Tells if data/fineweb.py is still writing token files into this folder: its `writing` file is there.
inline bool DataLoader::more_files_coming() const {
    FILE* marker = fopen(writing_marker_.c_str(), "rb");
    if (marker == nullptr) return false;
    fclose(marker);
    return true;
}

// ── private ───────────────────────────────────────────────────────────────────

/// Opens file k, closing the one open before, and puts the next batch at its first token.
inline void DataLoader::open_file(int k) {
    if (file_ != nullptr) fclose(file_);
    file_ = fopen(file_paths_[k].c_str(), "rb");
    if (file_ == nullptr) {
        fprintf(stderr, "error: cannot open %s\n", file_paths_[k].c_str());
        exit(EXIT_FAILURE);
    }
    current_file_ = k;
    position_ = 0;
}

/// Adds the files written since the last look, in order: the pattern with the next numbers put in, while the
/// file is there (a missing file 0 is skipped: train files start at 1); a pattern without % is one file.
/// While the writer works, it stops at a file not yet whole, to take it next time. Checks each file's
/// header and that it holds one round (world_size x B x T + 1 tokens); stops with an error if not.
inline void DataLoader::find_new_files() {
    bool writing = more_files_coming();
    if (pattern_.find('%') == std::string::npos && !file_paths_.empty()) return;   // one file, found
    int number = file_paths_.empty() ? 0 : next_number_;
    for (;; number++) {
        std::string path = path_of(number);
        FILE* probe = fopen(path.c_str(), "rb");
        if (probe == nullptr) {
            if (number == 0 && file_paths_.empty() && pattern_.find('%') != std::string::npos)
                continue;                                   // no file 0: try file 1
            break;
        }
        fclose(probe);
        if (writing && !is_whole(path)) break;              // still being written: next time
        size_t tokens = read_header(path);
        size_t round_tokens = world_size_ * batch_size_ * seq_len_;
        if (tokens < round_tokens + 1) {
            fprintf(stderr, "error: %s has %zu tokens, fewer than one round of %d x %zu x %zu + 1\n", path.c_str(),
                    tokens, world_size_, batch_size_, seq_len_);
            exit(EXIT_FAILURE);
        }
        file_paths_.push_back(path);
        file_tokens_.push_back(tokens);
        if (pattern_.find('%') == std::string::npos) break;
    }
    next_number_ = number;
}

/// Makes sure file k is found if it can be: looks for new files, and while the writer still works and file k
/// is not there yet, looks again every second (saying so once), for at most an hour. Returns with file k
/// found, or with the writer done and no file k (the caller goes back to file 0).
inline void DataLoader::wait_for_file(int k) {
    find_new_files();
    for (int seconds = 0; k >= num_files() && more_files_coming(); seconds++) {
        if (seconds == 0 && rank_ == 0)
            printf("data loader: waiting for token file %d of %s (the writer is still working)\n", k, pattern_.c_str());
        if (seconds >= 3600) {
            fprintf(stderr, "error: waited an hour for token file %d of %s; if data/fineweb.py has stopped, delete "
                            "%s\n", k, pattern_.c_str(), writing_marker_.c_str());
            exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        find_new_files();
    }
    find_new_files();       // the writer may have finished the last file just before deleting `writing`
}

/// Gives the path of file `number`: the pattern with the number put in for its %d (the pattern as it is
/// without one).
inline std::string DataLoader::path_of(int number) const {
    if (pattern_.find('%') == std::string::npos) return pattern_;
    char path[1024];
    snprintf(path, sizeof(path), pattern_.c_str(), number);
    return path;
}

/// Tells if a token file is all there: its 256-int header can be read and the file is as long as the header
/// says. A file the writer is still writing is not.
inline bool DataLoader::is_whole(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    int header[TOKEN_FILE_HEADER_INTS];
    size_t header_read = fread(header, sizeof(int), TOKEN_FILE_HEADER_INTS, file);
    fseek(file, 0, SEEK_END);
    long file_bytes = ftell(file);
    fclose(file);
    if (header_read != TOKEN_FILE_HEADER_INTS) return false;
    size_t tokens = (size_t)(unsigned int)header[2];
    return file_bytes == (long)(TOKEN_FILE_HEADER_INTS * sizeof(int) + tokens * sizeof(uint16_t));
}

/// Checks a token file's header (the magic number, the version, and that the file holds the N tokens it
/// says) and gives N. Stops with an error naming the file if anything is wrong.
inline size_t DataLoader::read_header(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    int header[TOKEN_FILE_HEADER_INTS];
    size_t header_read = fread(header, sizeof(int), TOKEN_FILE_HEADER_INTS, file);
    fseek(file, 0, SEEK_END);
    long file_bytes = ftell(file);
    fclose(file);
    if (header_read != TOKEN_FILE_HEADER_INTS || header[0] != TOKEN_FILE_MAGIC || header[1] != TOKEN_FILE_VERSION) {
        fprintf(stderr, "error: %s is not a version-%d token file (header %d, %d; make it with data/fineweb.py)\n",
                path.c_str(), TOKEN_FILE_VERSION, header_read > 0 ? header[0] : 0, header_read > 1 ? header[1] : 0);
        exit(EXIT_FAILURE);
    }
    size_t tokens = (size_t)(unsigned int)header[2];
    long expected_bytes = (long)(TOKEN_FILE_HEADER_INTS * sizeof(int) + tokens * sizeof(uint16_t));
    if (file_bytes != expected_bytes) {
        fprintf(stderr, "error: %s says it holds %zu tokens (%ld bytes with the header), but has %ld bytes\n",
                path.c_str(), tokens, expected_bytes, file_bytes);
        exit(EXIT_FAILURE);
    }
    return tokens;
}

#endif // DATALOADER_CUH
