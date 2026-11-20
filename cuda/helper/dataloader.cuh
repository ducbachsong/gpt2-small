// dataloader.cuh — batches of tokens for training, from the token files python/fineweb.py writes.
//
//     tensor      type in memory   shape
//     token_ids   int              {batch_size, seq_len}
//     targets     int              {batch_size, seq_len}
#ifndef DATALOADER_CUH
#define DATALOADER_CUH

#include <cerrno>
#include <chrono>
#include <csignal>                           // kill(pid, 0): is the writer's process still there
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include "../common/tensor.cuh"

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
    void wait_for_tokens(size_t tokens);                                // while the writer works, until that many are there

    // ── reading ─────────────────────────────────────────────────────────────────
    int num_files() const;                                              // how many whole files it found so far
    const std::string& file_path(int k) const;                          // file k's path
    size_t file_tokens(int k) const;                                    // file k's N, from its header
    size_t total_tokens() const;                                        // all the files' N together, so far
    int current_file() const;                                           // the file the next round comes from
    size_t position() const;                                            // the next round's first token in it
    int epoch() const;                                                  // 1 on the first pass over the files
    bool more_files_coming() const;                                     // the writer is still writing files
    double seconds_reading() const;                                     // all its batches so far: file to GPU
    double seconds_waiting() const;                                     // all its waits for the writer so far

private:
    size_t batch_size_;
    size_t seq_len_;
    int rank_;
    int world_size_;
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
    double seconds_reading_ = 0.0;           // in next_batch, its waits for a file left out
    double seconds_waiting_ = 0.0;           // in wait_for_file and wait_for_tokens

    void open_file(int k);                                              // file k, at its first token
    void find_new_files();                                              // adds the whole files written since
    void wait_for_file(int k);                                          // while the writer works, until file k
    void wait_a_second(int seconds_so_far, const char* what) const;     // one second of a wait; stops if it is hopeless
    long writer_process() const;                                        // the process id in `writing`; 0: none there
    std::string path_of(int number) const;                              // the pattern with this number put in
    static bool is_whole(const std::string& path);                      // its size is what its header says
    static size_t read_header(const std::string& path);                 // checks the header, gives N
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── making it ─────────────────────────────────────────────────────────────────

/// Finds the token files, checks their headers and opens the first.
///
///   pattern                the files' path with one %d: "python/tokenizer/data/fineweb10B/fineweb_train_%06d.bin"
///   batch_size, seq_len    B and T
///   rank, world_size       this process of how many: they take turns
inline DataLoader::DataLoader(const std::string& pattern, size_t batch_size, size_t seq_len, int rank,
                              int world_size)
    : batch_size_(batch_size), seq_len_(seq_len), rank_(rank), world_size_(world_size), pattern_(pattern) {
    // Finds the files, checks each header and that each holds one round, and opens the first; waits for the first
    // while the writer works. Reads only the headers.
    if (world_size < 1 || rank < 0 || rank >= world_size) {
        fprintf(stderr, "error: a data loader needs 0 <= rank < world_size; got rank %d of %d\n", rank, world_size);
        exit(EXIT_FAILURE);
    }
    size_t folder_end = pattern.find_last_of('/');
    writing_marker_ = (folder_end == std::string::npos ? "" : pattern.substr(0, folder_end + 1)) + "writing";
    wait_for_file(0);
    if (file_paths_.empty()) {
        fprintf(stderr, "error: no token files match %s (make them with python/fineweb.py)\n", pattern.c_str());
        exit(EXIT_FAILURE);
    }
    size_t batch_tokens = batch_size * seq_len;
    file_values_.resize(batch_tokens + 1);
    token_id_values_.resize(batch_tokens);
    target_values_.resize(batch_tokens);
    open_file(0);
}

/// Closes the open file.
inline DataLoader::~DataLoader() {
    if (file_ != nullptr) fclose(file_);
}

// ── batches ───────────────────────────────────────────────────────────────────

/// Writes this rank's next batch into token_ids and targets, then moves the round on.
///
///   token_ids, targets   {batch_size, seq_len} ints on the GPU
inline void DataLoader::next_batch(const Tensor& token_ids, const Tensor& targets) {
    // Writes this rank's next batch: B x T + 1 tokens from the current file, the first B x T into token_ids, the
    // last B x T into targets. Then moves the round on, to the next file if this one has no round left.
    Shape batch = {batch_size_, seq_len_};
    if (token_ids.shape() != batch || targets.shape() != batch || token_ids.dtype() != DType::Int32 ||
        targets.dtype() != DType::Int32) {
        fprintf(stderr, "error: the data loader needs token_ids and targets {%zu, %zu} ints\n", batch_size_,
                seq_len_);
        exit(EXIT_FAILURE);
    }
    auto started = std::chrono::steady_clock::now();
    double waiting_before = seconds_waiting_;
    size_t batch_tokens = batch_size_ * seq_len_;
    size_t round_tokens = world_size_ * batch_tokens;
    size_t first = position_ + rank_ * batch_tokens;                    // this rank's batch
    const std::string& path = file_paths_[current_file_];
    long byte = (long)(TOKEN_FILE_HEADER_INTS * sizeof(int) + first * sizeof(uint16_t));
    if (fseek(file_, byte, SEEK_SET) != 0 ||
        fread(file_values_.data(), sizeof(uint16_t), batch_tokens + 1, file_) != batch_tokens + 1) {
        fprintf(stderr, "error: cannot read tokens %zu .. %zu of %s\n", first, first + batch_tokens, path.c_str());
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < batch_tokens; i++) {
        token_id_values_[i] = file_values_[i];             // a uint16 into an int
        target_values_[i] = file_values_[i + 1];
    }
    token_ids.copy_from_cpu(token_id_values_.data());
    targets.copy_from_cpu(target_values_.data());

    position_ += round_tokens;
    if (position_ + round_tokens + 1 > file_tokens_[current_file_]) {   // not one more round here
        int next = current_file_ + 1;
        wait_for_file(next);
        if (next >= num_files()) {                                      // the last file: the next epoch
            next = 0;
            epoch_++;
            if (rank_ == 0) {
                printf("data loader: the %d file(s) of %s ran out (%zu tokens): epoch %d starts\n", num_files(),
                       pattern_.c_str(), total_tokens(), epoch_);
                fflush(stdout);
            }
        }
        open_file(next);
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    seconds_reading_ += seconds - (seconds_waiting_ - waiting_before);  // a wait for the next file is not reading
}

/// Goes back to file 0's first token, epoch 1: the val loss reads the same batches every time.
inline void DataLoader::reset() {
    open_file(0);
    epoch_ = 1;
}

/// Puts the loader where a checkpoint saved it; stops if this loader could not be there.
///
///   file, position, epoch   current_file(), position() and epoch() when it was saved
inline void DataLoader::seek(int file, size_t position, int epoch) {
    if (file >= 0) wait_for_file(file);
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

/// Waits, while the writer works, until the whole files found hold `tokens` tokens.
///
///   tokens   wanted in whole files; the writer ending with fewer is not an error here
inline void DataLoader::wait_for_tokens(size_t tokens) {
    // A lead over the writer before the first step, so the trainer does not catch up with it. Returns at once with
    // no writer, or with enough there.
    find_new_files();
    auto started = std::chrono::steady_clock::now();
    int seconds = 0;
    for (; total_tokens() < tokens && more_files_coming(); seconds++) {
        if (seconds == 0 && rank_ == 0) {
            printf("data loader: waiting for %zu tokens of %s before the first step (%zu there; the writer is still "
                   "working)\n", tokens, pattern_.c_str(), total_tokens());
            fflush(stdout);
        }
        wait_a_second(seconds, "a lead of tokens");
        find_new_files();
    }
    find_new_files();       // the last file may have finished just before `writing` went
    if (seconds == 0) return;
    seconds_waiting_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (rank_ == 0) {
        printf("data loader: waited %d s; %d file(s), %zu tokens there\n", seconds, num_files(), total_tokens());
        fflush(stdout);
    }
}

// ── reading ───────────────────────────────────────────────────────────────────

/// Gives how many whole files it found so far.
inline int DataLoader::num_files() const { return (int)file_paths_.size(); }

/// Gives file k's path.
inline const std::string& DataLoader::file_path(int k) const { return file_paths_.at(k); }

/// Gives how many tokens file k holds, from its header.
inline size_t DataLoader::file_tokens(int k) const { return file_tokens_.at(k); }

/// Gives how many tokens the files found so far hold.
inline size_t DataLoader::total_tokens() const {
    size_t total = 0;
    for (size_t tokens : file_tokens_) total += tokens;
    return total;
}

/// Gives which file the next round comes from.
inline int DataLoader::current_file() const { return current_file_; }

/// Gives the next round's first token in the current file, the same on every rank.
inline size_t DataLoader::position() const { return position_; }

/// Gives which pass over the files this is: 1 at first.
inline int DataLoader::epoch() const { return epoch_; }

/// Tells if python/fineweb.py is still writing files: its `writing` file is there.
inline bool DataLoader::more_files_coming() const {
    FILE* marker = fopen(writing_marker_.c_str(), "rb");
    if (marker == nullptr) return false;
    fclose(marker);
    return true;
}

/// Gives the seconds all its batches took so far, file to GPU, waits for the writer left out.
inline double DataLoader::seconds_reading() const { return seconds_reading_; }

/// Gives the seconds it waited for the writer so far: for a file at a file's end, and for wait_for_tokens' lead.
inline double DataLoader::seconds_waiting() const { return seconds_waiting_; }

// ── private ───────────────────────────────────────────────────────────────────

/// Opens file k, closing the one before, at its first token.
///
///   k   0 .. num_files() - 1
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

/// Adds the whole files written since the last look, in order, checking each header.
inline void DataLoader::find_new_files() {
    // Adds the whole files written since the last look, in order (no file 0: train files start at 1). Checks each
    // header and that each holds one round.
    bool writing = more_files_coming();
    if (pattern_.find('%') == std::string::npos && !file_paths_.empty()) return;   // one file, found
    int number = file_paths_.empty() ? 0 : next_number_;
    for (;; number++) {
        std::string path = path_of(number);
        FILE* probe = fopen(path.c_str(), "rb");
        if (probe == nullptr) {
            if (number == 0 && file_paths_.empty() && pattern_.find('%') != std::string::npos)
                continue;                                   // try file 1
            break;
        }
        fclose(probe);
        if (writing && !is_whole(path)) break;              // still being written
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

/// Waits for file k while the writer works, for at most an hour.
///
///   k   the file wanted
inline void DataLoader::wait_for_file(int k) {
    // Looks for file k, every second while the writer works, for at most an hour; stops at once if the writer's
    // process has died. Returns with file k found, or the writer done and no file k. Says how long it waited: the
    // GPU stood still that long.
    find_new_files();
    auto started = std::chrono::steady_clock::now();
    int seconds = 0;
    for (; k >= num_files() && more_files_coming(); seconds++) {
        if (seconds == 0 && rank_ == 0) {
            printf("data loader: waiting for token file %d of %s (the writer is still working)\n", k, pattern_.c_str());
            fflush(stdout);
        }
        wait_a_second(seconds, "the next token file");
        find_new_files();
    }
    find_new_files();       // the last file may have finished just before `writing` went
    if (seconds == 0) return;
    seconds_waiting_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (rank_ == 0) {
        printf("data loader: waited %d s for token file %d of %s: the writer is slower than the training\n", seconds,
               k, pattern_.c_str());
        fflush(stdout);
    }
}

/// Sleeps one second of a wait for the writer; stops if the writer died or after an hour.
///
///   seconds_so_far   of this wait
///   what             what is waited for, for the message
inline void DataLoader::wait_a_second(int seconds_so_far, const char* what) const {
    // Sleeps one second of a wait for the writer, after checking the wait can still end: stops with an error if
    // the process named in `writing` is gone (a writer that crashed leaves its `writing` file behind), or after an hour.
    long writer = writer_process();
    if (writer > 0 && kill((pid_t)writer, 0) != 0 && errno == ESRCH) {
        fprintf(stderr, "error: waiting for %s of %s, but the writer (process %ld, named in %s) is gone: it stopped "
                        "before its end. Run it again, or delete %s to train on the files there\n", what,
                pattern_.c_str(), writer, writing_marker_.c_str(), writing_marker_.c_str());
        exit(EXIT_FAILURE);
    }
    if (seconds_so_far >= 3600) {
        fprintf(stderr, "error: waited an hour for %s of %s; if the writer has stopped, delete %s\n", what,
                pattern_.c_str(), writing_marker_.c_str());
        exit(EXIT_FAILURE);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

/// Gives the writer's process id from its `writing` file; 0 with no such file or an empty one.
inline long DataLoader::writer_process() const {
    FILE* marker = fopen(writing_marker_.c_str(), "rb");
    if (marker == nullptr) return 0;
    long writer = 0;
    if (fscanf(marker, "%ld", &writer) != 1) writer = 0;
    fclose(marker);
    return writer;
}

/// Gives the path of file `number`: the pattern with it put in for %d.
///
///   number   the file's number
inline std::string DataLoader::path_of(int number) const {
    if (pattern_.find('%') == std::string::npos) return pattern_;
    char path[1024];
    snprintf(path, sizeof(path), pattern_.c_str(), number);
    return path;
}

/// Tells if a token file is all there: as long as its header says.
///
///   path   the file
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

/// Checks a token file's header (magic, version, length) and gives N, its tokens; stops if anything is wrong.
///
///   path   the file
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
        fprintf(stderr, "error: %s is not a version-%d token file (header %d, %d; make it with python/fineweb.py)\n",
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
