// checkpoint.cuh — saving a training run every N steps, and going on from the last save in a new process: a run
// that outlives its machine (a session that ends, a crash) loses at most N steps. A checkpoint is a folder, one
// file per rank, and a file `latest` naming the newest one that every rank finished writing:
//
//     checkpoint_dir/
//         step_00001000/   rank_0.bin  rank_1.bin  ...  rank_7.bin
//         step_00002000/   rank_0.bin  rank_1.bin  ...  rank_7.bin
//         latest           "step_00002000"
//
// rank_R.bin is what rank R needs to go on, and nothing more:
//
//     header           32 int64s: [0] 20241103, a checkpoint file   [1] 1, the version   [2] steps done
//                      [3] the run's total steps   [4] world size   [5] rank   [6] dtype, 0 float or 1 bfloat16
//                      [7] slice start   [8] slice size   [9] the train loader's file   [10] its position
//                      [11] its epoch; the rest 0
//     master params    slice size floats: this rank's float params (optimizer.cuh)
//     m, v             slice size floats each: AdamW's
//
// so 12 bytes per value of the slice: GPT-2 small on 1 GPU 1.49 GB, on 8 GPUs 187 MB per rank. Not saved: the
// bfloat16 params (each rank rounds its master again, then they all-gather, as after a step), the grads (zero
// at every step), the val loader (reset at every val loss).
//
// Saving never spoils the last good checkpoint: each rank writes rank_R.bin.partial and renames it when it is
// whole, all ranks wait for each other, and only then rank 0 points `latest` at the new folder, the same way,
// and deletes all but the newest `keep` folders.
#ifndef CHECKPOINT_CUH
#define CHECKPOINT_CUH

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include "tensor.cuh"
#include "optimizer.cuh"
#include "dataloader.cuh"
#include "distributed.cuh"

#define CHECKPOINT_MAGIC 20241103            // header[0]: "this is a checkpoint file"
#define CHECKPOINT_VERSION 1                 // header[1]
#define CHECKPOINT_HEADER_INTS 32            // the header: 32 int64s, 256 bytes

// ── saving and going on ───────────────────────────────────────────────────────
void checkpoint_save(const std::string& checkpoint_dir, int steps_done, int total_steps, const AdamW& optimizer,
                     const DataLoader& train_loader, const Distributed& distributed, int keep = 2);
int checkpoint_load(const std::string& path, int total_steps, AdamW& optimizer, DataLoader& train_loader,
                    const Distributed& distributed);
std::string checkpoint_problem(const std::string& path, int total_steps, const AdamW& optimizer);

// ── for both ──────────────────────────────────────────────────────────────────
std::string checkpoint_folder_name(int steps_done);
std::string checkpoint_rank_file(const std::string& path, int rank);

// ══ the code, in the order of the list above ══════════════════════════════════

// ── saving and going on ───────────────────────────────────────────────────────

/// Saves where training is after steps_done steps into checkpoint_dir/step_<steps_done>/: each rank its own
/// file. Every rank calls it at the same step. Then rank 0 points `latest` at it and deletes all but the newest
/// `keep` checkpoints. The folders are made if needed.
inline void checkpoint_save(const std::string& checkpoint_dir, int steps_done, int total_steps, const AdamW& optimizer,
                            const DataLoader& train_loader, const Distributed& distributed, int keep) {
    namespace fs = std::filesystem;
    std::string folder = checkpoint_dir + "/" + checkpoint_folder_name(steps_done);
    fs::create_directories(folder);

    // 1. this rank's file, written whole under another name, then renamed
    int64_t header[CHECKPOINT_HEADER_INTS] = {};
    header[0] = CHECKPOINT_MAGIC, header[1] = CHECKPOINT_VERSION, header[2] = steps_done, header[3] = total_steps;
    header[4] = optimizer.world_size(), header[5] = optimizer.rank();
    header[6] = optimizer.dtype() == DType::BFloat16 ? 1 : 0;
    header[7] = (int64_t)optimizer.slice_start(), header[8] = (int64_t)optimizer.slice_size();
    header[9] = train_loader.current_file(), header[10] = (int64_t)train_loader.position();
    header[11] = train_loader.epoch();
    std::string path = checkpoint_rank_file(folder, optimizer.rank());
    std::string partial = path + ".partial";
    FILE* file = fopen(partial.c_str(), "wb");
    bool written = file != nullptr && fwrite(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file) ==
                                          CHECKPOINT_HEADER_INTS;
    std::vector<float> values(optimizer.slice_size());
    for (const Tensor& state : {optimizer.master_slice(), optimizer.m_slice(), optimizer.v_slice()}) {
        if (!written) break;
        state.copy_to_cpu(values.data());
        written = fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
    }
    if (file != nullptr && fclose(file) != 0) written = false;
    if (!written || rename(partial.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "error: rank %d cannot write %s\n", optimizer.rank(), path.c_str());
        exit(EXIT_FAILURE);
    }

    // 2. once every rank's file is whole: `latest`, then the old checkpoints go
    distributed.barrier();
    if (distributed.rank() != 0) return;
    std::string latest = checkpoint_dir + "/latest";
    FILE* latest_file = fopen((latest + ".partial").c_str(), "w");
    if (latest_file == nullptr || fprintf(latest_file, "%s\n", checkpoint_folder_name(steps_done).c_str()) < 0 ||
        fclose(latest_file) != 0 || rename((latest + ".partial").c_str(), latest.c_str()) != 0) {
        fprintf(stderr, "error: cannot write %s\n", latest.c_str());
        exit(EXIT_FAILURE);
    }
    std::vector<std::string> folders;                       // step_00000001 .. : the names sort as the steps do
    for (const fs::directory_entry& entry : fs::directory_iterator(checkpoint_dir))
        if (entry.is_directory() && entry.path().filename().string().rfind("step_", 0) == 0)
            folders.push_back(entry.path().filename().string());
    std::sort(folders.begin(), folders.end());
    for (size_t k = 0; k + keep < folders.size(); k++) fs::remove_all(checkpoint_dir + "/" + folders[k]);
}

/// Puts training back where a checkpoint saved it, and gives the steps done then: this rank's master, m and v,
/// the step count, the train loader's place, and all the bfloat16 params (each rank rounds its master into its
/// slice, then they all-gather). `path` is a checkpoint folder, or a checkpoint_dir: then the one `latest` names.
/// Stops if the checkpoint is of another run: checkpoint_problem says how.
inline int checkpoint_load(const std::string& path, int total_steps, AdamW& optimizer, DataLoader& train_loader,
                           const Distributed& distributed) {
    std::string problem = checkpoint_problem(path, total_steps, optimizer);
    if (!problem.empty()) {
        fprintf(stderr, "error: cannot go on from %s: %s\n", path.c_str(), problem.c_str());
        exit(EXIT_FAILURE);
    }
    std::string file_path = checkpoint_rank_file(path, optimizer.rank());
    FILE* file = fopen(file_path.c_str(), "rb");
    int64_t header[CHECKPOINT_HEADER_INTS];
    bool read = file != nullptr && fread(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file) ==
                                       CHECKPOINT_HEADER_INTS;
    std::vector<float> values(optimizer.slice_size());
    for (const Tensor& state : {optimizer.master_slice(), optimizer.m_slice(), optimizer.v_slice()}) {
        if (!read) break;
        read = fread(values.data(), sizeof(float), values.size(), file) == values.size();
        if (read) state.copy_from_cpu(values.data());
    }
    if (file != nullptr) fclose(file);
    if (!read) {
        fprintf(stderr, "error: rank %d cannot read %s\n", optimizer.rank(), file_path.c_str());
        exit(EXIT_FAILURE);
    }
    int steps_done = (int)header[2];
    optimizer.set_steps_taken(steps_done);
    if (optimizer.dtype() == DType::BFloat16)              // the model's params from the master, as a step leaves them
        optimizer.all_params().narrow(0, optimizer.slice_start(), optimizer.slice_size()).copy_(optimizer.master_slice());
    distributed.all_gather(optimizer.all_params());
    train_loader.seek((int)header[9], (size_t)header[10], (int)header[11]);
    return steps_done;
}

/// Tells what is wrong with going on from this checkpoint with this optimizer and run length, or "" if
/// nothing: no file for this rank, not a checkpoint file, or one of another run (another world size, dtype,
/// model size or number of total steps, which the learning rate schedule depends on).
inline std::string checkpoint_problem(const std::string& path, int total_steps, const AdamW& optimizer) {
    std::string file_path = checkpoint_rank_file(path, optimizer.rank());
    FILE* file = fopen(file_path.c_str(), "rb");
    if (file == nullptr) return "no file " + file_path;
    int64_t header[CHECKPOINT_HEADER_INTS];
    size_t header_read = fread(header, sizeof(int64_t), CHECKPOINT_HEADER_INTS, file);
    fclose(file);
    if (header_read != CHECKPOINT_HEADER_INTS || header[0] != CHECKPOINT_MAGIC || header[1] != CHECKPOINT_VERSION)
        return file_path + " is not a version-1 checkpoint file";
    char problem[256] = "";
    int64_t dtype = optimizer.dtype() == DType::BFloat16 ? 1 : 0;
    if (header[3] != total_steps)
        snprintf(problem, sizeof(problem), "saved by a run of %lld steps, this one has %d", (long long)header[3],
                 total_steps);
    else if (header[4] != optimizer.world_size() || header[5] != optimizer.rank())
        snprintf(problem, sizeof(problem), "saved by rank %lld of %lld GPUs, this is rank %d of %d",
                 (long long)header[5], (long long)header[4], optimizer.rank(), optimizer.world_size());
    else if (header[6] != dtype)
        snprintf(problem, sizeof(problem), "saved in %s, this run is %s", header[6] ? "bfloat16" : "float",
                 dtype ? "bfloat16" : "float");
    else if (header[7] != (int64_t)optimizer.slice_start() || header[8] != (int64_t)optimizer.slice_size())
        snprintf(problem, sizeof(problem), "its slice is %lld values from %lld, this rank's %zu from %zu",
                 (long long)header[8], (long long)header[7], optimizer.slice_size(), optimizer.slice_start());
    return problem;
}

// ── for both ──────────────────────────────────────────────────────────────────

/// Gives a checkpoint folder's name: step_ and the steps done in 8 digits, so the names sort as the steps do:
/// checkpoint_folder_name(1000) is "step_00001000".
inline std::string checkpoint_folder_name(int steps_done) {
    char name[32];
    snprintf(name, sizeof(name), "step_%08d", steps_done);
    return name;
}

/// Gives the path of rank's file in a checkpoint: path/rank_<rank>.bin if path is a checkpoint folder; if it is
/// a checkpoint_dir with a `latest` file, the same in the folder `latest` names.
inline std::string checkpoint_rank_file(const std::string& path, int rank) {
    std::string folder = path;
    FILE* latest = fopen((path + "/latest").c_str(), "r");
    if (latest != nullptr) {
        char name[64] = "";
        if (fscanf(latest, "%63s", name) == 1) folder = path + "/" + name;
        fclose(latest);
    }
    return folder + "/rank_" + std::to_string(rank) + ".bin";
}

#endif // CHECKPOINT_CUH
