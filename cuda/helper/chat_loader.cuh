// chat_loader.cuh — the fine-tune's batches: rows of python/chat_sft.py's files, input tokens and targets (-1 where
// the loss leaves them out).
//
//     tensor      type in memory   shape
//     token_ids   int              {batch_size, seq_len}
//     targets     int              {batch_size, seq_len}
#ifndef CHAT_LOADER_CUH
#define CHAT_LOADER_CUH

#include <algorithm>                    // std::copy
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../common/tensor.cuh"

#define CHAT_LOADER_MAGIC 20261119              // header[0]: "this is a chat_sft.py rows file"
#define CHAT_LOADER_VERSION 1                   // header[1]
#define CHAT_LOADER_HEADER_INTS 256             // the header's int32s
#define CHAT_LOADER_IGNORE_TARGET (-1)          // a target the loss leaves out (SOFTMAX_LOSS_FINETUNE_BF16_IGNORE_TARGET)

class ChatLoader {
public:
    ChatLoader(const std::string& path, size_t batch_size, size_t seq_len, int rank, int world_size);
    void next_batch(const Tensor& token_ids, const Tensor& targets);    // this rank's {B, T} of the next round
    size_t graded_ahead(int rounds) const;                              // graded targets in the next rounds, every rank's
    void reset();                                                       // back to row 0, epoch 1
    void seek(int file, size_t position, int epoch);                    // to where a checkpoint saved it (file 0)
    size_t total_rows() const { return rows_; }
    size_t total_tokens() const { return rows_ * seq_len_; }            // input places, padding included
    size_t graded_targets() const { return graded_; }                   // in the whole file
    size_t rounds_per_epoch() const { return rows_ / rows_per_round_; }
    const std::string& path() const { return path_; }
    int current_file() const { return 0; }                              // one file: for checkpoint_8bit.cuh
    size_t position() const { return next_row_; }                       // the next round's first row
    int epoch() const { return epoch_; }

private:
    std::string path_;
    size_t batch_size_, seq_len_, rows_ = 0, graded_ = 0, rows_per_round_;
    int rank_;
    std::vector<int32_t> values_;                       // rows x (seq_len tokens, then seq_len targets)
    std::vector<uint32_t> graded_of_row_;               // each row's targets that are not -1
    std::vector<int32_t> token_id_values_, target_values_;   // this rank's batch on the CPU, before the copy
    size_t next_row_ = 0;
    int epoch_ = 1;
};

/// Reads the whole rows file into memory and counts each row's graded targets.
///
///   path                  python/chat_sft.py's file
///   batch_size, seq_len   the rows of one batch, and the tokens of a row (the file's)
///   rank, world_size      this GPU of how many
inline ChatLoader::ChatLoader(const std::string& path, size_t batch_size, size_t seq_len, int rank, int world_size)
    : path_(path), batch_size_(batch_size), seq_len_(seq_len), rows_per_round_((size_t)world_size * batch_size),
      rank_(rank) {
    // Stops if the file is not chat_sft.py's, its rows are not seq_len long, or it holds less than one round.
    // 1. the header
    FILE* file = fopen(path.c_str(), "rb");
    int32_t header[CHAT_LOADER_HEADER_INTS] = {};
    if (file == nullptr || fread(header, sizeof(int32_t), CHAT_LOADER_HEADER_INTS, file) != CHAT_LOADER_HEADER_INTS ||
        header[0] != CHAT_LOADER_MAGIC || header[1] != CHAT_LOADER_VERSION) {
        fprintf(stderr, "error: %s is not a chat rows file of python/chat_sft.py (run it first)\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    rows_ = (size_t)header[2];
    if ((size_t)header[3] != seq_len || rows_ < rows_per_round_) {
        fprintf(stderr, "error: %s has %zu rows of %d tokens; this run needs rows of %zu and at least %zu of them\n",
                path.c_str(), rows_, header[3], seq_len, rows_per_round_);
        exit(EXIT_FAILURE);
    }
    // 2. the rows, then each row's graded count
    values_.resize(rows_ * 2 * seq_len);
    size_t read_values = fread(values_.data(), sizeof(int32_t), values_.size(), file);
    fclose(file);
    if (read_values != values_.size()) {
        fprintf(stderr, "error: %s is cut short: %zu of %zu values\n", path.c_str(), read_values, values_.size());
        exit(EXIT_FAILURE);
    }
    graded_of_row_.resize(rows_);
    for (size_t row = 0; row < rows_; row++) {
        const int32_t* targets = values_.data() + (row * 2 + 1) * seq_len;
        uint32_t graded = 0;
        for (size_t t = 0; t < seq_len; t++) graded += targets[t] != CHAT_LOADER_IGNORE_TARGET;
        graded_of_row_[row] = graded;
        graded_ += graded;
    }
    token_id_values_.resize(batch_size * seq_len);
    target_values_.resize(batch_size * seq_len);
}

/// Copies this rank's rows of the next round onto the GPU, then moves on a round.
///
///   token_ids, targets   {batch_size, seq_len} int32, on the GPU
inline void ChatLoader::next_batch(const Tensor& token_ids, const Tensor& targets) {
    // Copies this rank's batch_size rows of the next round onto the GPU (tokens and targets), then moves on a round;
    // past the last whole round: epoch + 1, from row 0.
    if (next_row_ + rows_per_round_ > rows_) next_row_ = 0, epoch_++;
    size_t first_row = next_row_ + (size_t)rank_ * batch_size_;
    for (size_t b = 0; b < batch_size_; b++) {
        const int32_t* row = values_.data() + (first_row + b) * 2 * seq_len_;
        std::copy(row, row + seq_len_, token_id_values_.data() + b * seq_len_);
        std::copy(row + seq_len_, row + 2 * seq_len_, target_values_.data() + b * seq_len_);
    }
    token_ids.copy_from_cpu(token_id_values_.data());
    targets.copy_from_cpu(target_values_.data());
    next_row_ += rows_per_round_;
}

/// Gives the graded targets of the next `rounds` rounds, every rank's rows, without moving.
///
///   rounds   how many rounds ahead
inline size_t ChatLoader::graded_ahead(int rounds) const {
    // The count a step's loss is divided by (its micro_batches rounds), the same on every rank.
    size_t row = next_row_, graded = 0;
    for (int round = 0; round < rounds; round++) {
        if (row + rows_per_round_ > rows_) row = 0;
        for (size_t k = 0; k < rows_per_round_; k++) graded += graded_of_row_[row + k];
        row += rows_per_round_;
    }
    return graded;
}

/// Goes back to row 0, epoch 1: the val loss reads the same rows every time.
inline void ChatLoader::reset() {
    next_row_ = 0;
    epoch_ = 1;
}

/// Goes to where a checkpoint saved it.
///
///   file              0 (one file)
///   position, epoch   the saved next_row and epoch
inline void ChatLoader::seek(int file, size_t position, int epoch) {
    if (file != 0 || position > rows_) {
        fprintf(stderr, "error: a checkpoint's chat loader place (file %d, row %zu) is not in %s\n", file, position,
                path_.c_str());
        exit(EXIT_FAILURE);
    }
    next_row_ = position;
    epoch_ = epoch;
}

#endif // CHAT_LOADER_CUH
