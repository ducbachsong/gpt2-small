// checkpoint_8bit.cuh — saves an 8-bit run (AdamW8bit) every N steps, and goes on from the last save in a new
// process.
#ifndef CHECKPOINT_8BIT_CUH
#define CHECKPOINT_8BIT_CUH

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include "../common/tensor.cuh"
#include "../model_8bit.cuh"                // AdamW8bit, Round8bitHistory (round_8bit.cuh)
#include "checkpoint.cuh"                   // checkpoint_folder_name, checkpoint_rank_file, CHECKPOINT_HEADER_INTS;
                                            // CHECKPOINT_MAGIC: a bfloat16 or float run's file, told apart
#include "dataloader.cuh"
#include "distributed.cuh"

#define CHECKPOINT_8BIT_MAGIC 20261118      // header[0]: "this is an 8-bit run's checkpoint file"
#define CHECKPOINT_8BIT_VERSION 1           // header[1]

// ── saving and going on ───────────────────────────────────────────────────────
// LOADER: the train loader whose place is saved: DataLoader (train_8bit) or ChatLoader (finetune_8bit, chat_loader.cuh);
// it needs current_file(), position(), epoch() and seek(file, position, epoch).
template <typename LOADER>
void checkpoint_8bit_save(const std::string& checkpoint_dir, int steps_done, int total_steps, const AdamW8bit& optimizer,
                          const std::vector<Round8bitHistory>& histories, const LOADER& train_loader,
                          const Distributed& distributed, int keep = 2);
template <typename LOADER>
int checkpoint_8bit_load(const std::string& path, int total_steps, AdamW8bit& optimizer,
                         const std::vector<Round8bitHistory>& histories, LOADER& train_loader,
                         const Distributed& distributed);
std::string checkpoint_8bit_problem(const std::string& path, int total_steps, const AdamW8bit& optimizer,
                                    size_t histories_count);

// ── reading and writing a tensor's values ─────────────────────────────────────
template <typename T> bool checkpoint_8bit_write(FILE* file, const Tensor& tensor);
template <typename T> bool checkpoint_8bit_read(FILE* file, const Tensor& tensor);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── saving and going on ───────────────────────────────────────────────────────

/// Saves this rank's part of an 8-bit run; rank 0 then points `latest` at it and keeps the newest `keep`.
///
///   checkpoint_dir             made if needed
///   steps_done, total_steps    where the run is, and its length
///   optimizer                  the master, m and sqrt(v) of this slice
///   histories                  this rank's rounding histories (GPT2Activations8bit::round_histories)
///   train_loader               its place
///   distributed                to wait for every rank
///   keep                       how many checkpoints to keep
template <typename LOADER>
inline void checkpoint_8bit_save(const std::string& checkpoint_dir, int steps_done, int total_steps,
                                 const AdamW8bit& optimizer, const std::vector<Round8bitHistory>& histories,
                                 const LOADER& train_loader, const Distributed& distributed, int keep) {
    // Saves this rank's part of an 8-bit run into checkpoint_dir/step_<steps_done>/ (master, m and sqrt(v) as kept,
    // the rounding histories, the loader's place); then rank 0 points `latest` at it and deletes all but the newest
    // `keep`. Every rank calls it at the same step.
    namespace fs = std::filesystem;
    std::string folder = checkpoint_dir + "/" + checkpoint_folder_name(steps_done);
    fs::create_directories(folder);

    // 1. the header
    int64_t header[CHECKPOINT_HEADER_INTS] = {};
    header[0] = CHECKPOINT_8BIT_MAGIC, header[1] = CHECKPOINT_8BIT_VERSION, header[2] = steps_done;
    header[3] = total_steps, header[4] = optimizer.world_size(), header[5] = optimizer.rank();
    header[6] = (int64_t)optimizer.slice_start(), header[7] = (int64_t)optimizer.slice_size();
    header[8] = train_loader.current_file(), header[9] = (int64_t)train_loader.position();
    header[10] = train_loader.epoch(), header[11] = ADAMW_VALUES_PER_SCALE, header[12] = (int64_t)histories.size();
    std::string path = checkpoint_rank_file(folder, optimizer.rank());
    std::string partial = path + ".partial";
    FILE* file = fopen(partial.c_str(), "wb");
    bool written = file != nullptr && fwrite(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file) ==
                                          CHECKPOINT_HEADER_INTS;

    // 2. the master, m and sqrt(v), as they are kept: bytes and scales, exactly
    written = written && checkpoint_8bit_write<float>(file, optimizer.master_slice());
    written = written && checkpoint_8bit_write<__nv_fp8_e4m3>(file, optimizer.m_bytes());
    written = written && checkpoint_8bit_write<float>(file, optimizer.m_scales());
    written = written && checkpoint_8bit_write<__nv_fp8_e4m3>(file, optimizer.sqrt_v_bytes());
    written = written && checkpoint_8bit_write<float>(file, optimizer.sqrt_v_scales());

    // 3. the rounding histories: every place's two largests, then every place's calls
    for (const Round8bitHistory& history : histories)
        written = written && checkpoint_8bit_write<float>(file, history.largest_tensor);
    for (const Round8bitHistory& history : histories) {
        int64_t calls = (int64_t)history.calls_val;
        written = written && fwrite(&calls, sizeof(int64_t), 1, file) == 1;
    }

    // 4. whole: renamed
    if (file != nullptr && fclose(file) != 0) written = false;
    if (!written || rename(partial.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "error: rank %d cannot write %s\n", optimizer.rank(), path.c_str());
        exit(EXIT_FAILURE);
    }

    // 5. once every rank is done: `latest`, then the old checkpoints go
    distributed.barrier();
    if (distributed.rank() != 0) return;
    std::string latest = checkpoint_dir + "/latest";
    FILE* latest_file = fopen((latest + ".partial").c_str(), "w");
    if (latest_file == nullptr || fprintf(latest_file, "%s\n", checkpoint_folder_name(steps_done).c_str()) < 0 ||
        fclose(latest_file) != 0 || rename((latest + ".partial").c_str(), latest.c_str()) != 0) {
        fprintf(stderr, "error: cannot write %s\n", latest.c_str());
        exit(EXIT_FAILURE);
    }
    std::vector<std::string> folders;                       // the names sort as the steps do
    for (const fs::directory_entry& entry : fs::directory_iterator(checkpoint_dir))
        if (entry.is_directory() && entry.path().filename().string().rfind("step_", 0) == 0)
            folders.push_back(entry.path().filename().string());
    std::sort(folders.begin(), folders.end());
    for (size_t k = 0; k + keep < folders.size(); k++) fs::remove_all(checkpoint_dir + "/" + folders[k]);
}

/// Puts an 8-bit run back where a checkpoint left it and gives the steps done.
///
///   path                       a checkpoint folder, or a checkpoint_dir (then the one `latest` names)
///   total_steps                this run's length
///   optimizer, histories       filled in
///   train_loader               put at the saved place
///   distributed                for the all-gather of the params
template <typename LOADER>
inline int checkpoint_8bit_load(const std::string& path, int total_steps, AdamW8bit& optimizer,
                                const std::vector<Round8bitHistory>& histories, LOADER& train_loader,
                                const Distributed& distributed) {
    // Puts an 8-bit run back where a checkpoint left it (master, m and sqrt(v), step count, rounding histories, loader,
    // bfloat16 params) and gives the steps done. Stops if it is another run's. The caller then calls
    // gpt2_8bit_weights_changed, so the next forward makes the weights' 8-bit copies from the loaded params.
    // 1. the right checkpoint, or stop
    std::string problem = checkpoint_8bit_problem(path, total_steps, optimizer, histories.size());
    if (!problem.empty()) {
        fprintf(stderr, "error: cannot go on from %s: %s\n", path.c_str(), problem.c_str());
        exit(EXIT_FAILURE);
    }

    // 2. the header, the master, m and sqrt(v), the rounding histories
    std::string file_path = checkpoint_rank_file(path, optimizer.rank());
    FILE* file = fopen(file_path.c_str(), "rb");
    int64_t header[CHECKPOINT_HEADER_INTS];
    bool read = file != nullptr && fread(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file) ==
                                       CHECKPOINT_HEADER_INTS;
    read = read && checkpoint_8bit_read<float>(file, optimizer.master_slice());
    read = read && checkpoint_8bit_read<__nv_fp8_e4m3>(file, optimizer.m_bytes());
    read = read && checkpoint_8bit_read<float>(file, optimizer.m_scales());
    read = read && checkpoint_8bit_read<__nv_fp8_e4m3>(file, optimizer.sqrt_v_bytes());
    read = read && checkpoint_8bit_read<float>(file, optimizer.sqrt_v_scales());
    for (const Round8bitHistory& history : histories)
        read = read && checkpoint_8bit_read<float>(file, history.largest_tensor);
    for (const Round8bitHistory& history : histories) {
        int64_t calls = 0;
        read = read && fread(&calls, sizeof(int64_t), 1, file) == 1;
        history.calls_val = (size_t)calls;
    }
    if (file != nullptr) fclose(file);
    if (!read) {
        fprintf(stderr, "error: rank %d cannot read %s\n", optimizer.rank(), file_path.c_str());
        exit(EXIT_FAILURE);
    }

    // 3. the step count, the model's params from the master (every rank's slice to every rank), the loader
    int steps_done = (int)header[2];
    optimizer.set_steps_taken(steps_done);
    optimizer.all_params().narrow(0, optimizer.slice_start(), optimizer.slice_size()).copy_(optimizer.master_slice());
    distributed.all_gather(optimizer.all_params());
    train_loader.seek((int)header[8], (size_t)header[9], (int)header[10]);
    return steps_done;
}

/// Gives what is wrong with going on from this 8-bit checkpoint, or "".
///
///   path, total_steps, optimizer   as in checkpoint_8bit_load
///   histories_count                the model's rounding histories: a model of another depth has another count
inline std::string checkpoint_8bit_problem(const std::string& path, int total_steps, const AdamW8bit& optimizer,
                                           size_t histories_count) {
    // Tells what is wrong with going on from this checkpoint, or gives "": no file, not a checkpoint, a bfloat16 or
    // float run's (checkpoint.cuh), or another 8-bit run's (total steps, world size, slice, scale size, histories).
    std::string file_path = checkpoint_rank_file(path, optimizer.rank());
    FILE* file = fopen(file_path.c_str(), "rb");
    if (file == nullptr) return "no file " + file_path;
    int64_t header[CHECKPOINT_HEADER_INTS];
    size_t header_read = fread(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file);
    fclose(file);
    if (header_read == CHECKPOINT_HEADER_INTS && header[0] == CHECKPOINT_MAGIC)
        return file_path + " is a bfloat16 or float run's checkpoint (checkpoint.cuh), not an 8-bit run's";
    if (header_read != CHECKPOINT_HEADER_INTS || header[0] != CHECKPOINT_8BIT_MAGIC ||
        header[1] != CHECKPOINT_8BIT_VERSION)
        return file_path + " is not a version-1 8-bit checkpoint file";
    char problem[256] = "";
    if (header[3] != total_steps)
        snprintf(problem, sizeof(problem), "saved by a run of %lld steps, this one has %d", (long long)header[3],
                 total_steps);
    else if (header[4] != optimizer.world_size() || header[5] != optimizer.rank())
        snprintf(problem, sizeof(problem), "saved by rank %lld of %lld GPUs, this is rank %d of %d",
                 (long long)header[5], (long long)header[4], optimizer.rank(), optimizer.world_size());
    else if (header[6] != (int64_t)optimizer.slice_start() || header[7] != (int64_t)optimizer.slice_size())
        snprintf(problem, sizeof(problem), "its slice is %lld values from %lld, this rank's %zu from %zu",
                 (long long)header[7], (long long)header[6], optimizer.slice_size(), optimizer.slice_start());
    else if (header[11] != ADAMW_VALUES_PER_SCALE)
        snprintf(problem, sizeof(problem), "saved with a scale per %lld values of m and sqrt(v), this build has one "
                 "per %d", (long long)header[11], (int)ADAMW_VALUES_PER_SCALE);
    else if (header[12] != (int64_t)histories_count)
        snprintf(problem, sizeof(problem), "saved with %lld rounding histories, this model has %zu",
                 (long long)header[12], histories_count);
    return problem;
}

// ── reading and writing a tensor's values ─────────────────────────────────────

/// Writes every value of a GPU tensor to the file as its bytes; tells if all were written.
///
///   file     open for writing
///   tensor   of T's dtype
///   T        float or __nv_fp8_e4m3
template <typename T>
inline bool checkpoint_8bit_write(FILE* file, const Tensor& tensor) {
    std::vector<T> values = tensor.tolist<T>();
    return fwrite(values.data(), sizeof(T), values.size(), file) == values.size();
}

/// Reads the GPU tensor's values from the file, as checkpoint_8bit_write wrote them; tells if all were there.
///
///   file     open for reading, at the values
///   tensor   of T's dtype: written
///   T        float or __nv_fp8_e4m3
template <typename T>
inline bool checkpoint_8bit_read(FILE* file, const Tensor& tensor) {
    std::vector<T> values(tensor.numel());
    if (fread(values.data(), sizeof(T), values.size(), file) != values.size()) return false;
    tensor.copy_from_cpu(values.data());
    return true;
}

#endif // CHECKPOINT_8BIT_CUH
