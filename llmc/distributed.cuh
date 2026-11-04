// distributed.cuh — training on several GPUs of one machine at once, nanochat's way: one process per GPU,
// each with the whole model, each reading its own batches (dataloader.cuh), each updating its own slice of
// the params (optimizer.cuh). The GPUs talk through NCCL, NVIDIA's library for it. One step:
//
//     every GPU   forward + backward on its own batches              → its own grads, all of them
//     reduce_scatter_mean(all grads)                                 → each GPU: the average of ITS slice
//     slice_norms_to_norm: each GPU's norm of its slice → the norm of all the averaged grads, on every GPU
//     AdamW on its slice, clipped by that norm                       → its slice of the new params
//     all_gather(all params)                                         → every GPU: all the new params
//
// The two big ones move the same bytes as one all-reduce of the grads would, (world_size - 1) / world_size
// of them each: for GPT-2 small's 249 MB of bfloat16 grads on 8 GPUs, 218 MB sent and received per GPU by
// each, about 1 ms over an H100's NVLink in theory.
//
// Starting: each process reads who it is from the environment, as torchrun sets it:
//
//     RANK         this process: 0 .. WORLD_SIZE - 1          LOCAL_RANK   its GPU on this machine
//     WORLD_SIZE   how many processes                          none set     1 process, no NCCL at all
//
//     torchrun --standalone --nproc_per_node=8 --no-python ./train_gpt2      (or a shell loop setting them)
//
// NCCL connects the processes with an id that rank 0 makes and the others need: rank 0 writes it to a file,
// the others wait for the file and read it. All collectives go on the default stream, after the kernels
// before them, as every launch here does.
#ifndef DISTRIBUTED_CUH
#define DISTRIBUTED_CUH

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <nccl.h>
#include "tensor.cuh"

// Every NCCL call returns a result code; this stops with the file and line, as cudaCheck does.
static inline void nccl_check(ncclResult_t result, const char* file, int line) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL error at %s:%d: %s\n", file, line, ncclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#define ncclCheck(call) nccl_check((call), __FILE__, __LINE__)

class Distributed {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    explicit Distributed(const std::string& id_file);                   // picks the GPU, joins the others
    ~Distributed();                                                     // leaves NCCL
    Distributed(const Distributed&) = delete;                           // one per process
    Distributed& operator=(const Distributed&) = delete;

    // ── who this process is ─────────────────────────────────────────────────────
    int rank() const;                                                   // 0 .. world_size - 1
    int local_rank() const;                                             // its GPU on this machine
    int world_size() const;                                             // how many processes, one per GPU

    // ── the collectives ─────────────────────────────────────────────────────────
    void reduce_scatter_mean(const Tensor& flat) const;                 // each rank: the average of its slice
    void all_gather(const Tensor& flat) const;                          // each rank's slice to every rank
    void slice_norms_to_norm(const Tensor& norm) const;                 // the norm of all from each slice's
    float mean_of_cpu_value(float value) const;                         // the average of a CPU number over the ranks
    void barrier() const;                                               // waits until every rank gets here

private:
    int rank_ = 0, local_rank_ = 0, world_size_ = 1;
    ncclComm_t comm_ = nullptr;          // NCCL's link between the processes; none with one process
    Tensor one_value_;                   // 1 float on the GPU, for mean_of_cpu_value()

    size_t slice_size_of(const Tensor& flat) const;                     // flat's values / world_size, checked
    static int env_int(const char* name, int fallback);                 // an environment variable as an int
    static ncclDataType_t nccl_type(DType dtype);                       // ncclFloat or ncclBfloat16
};

// ══ the code, in the order of the list above ══════════════════════════════════

// The two 1-value kernels slice_norms_to_norm launches around its all-reduce: norm becomes norm², then back.
__global__ void distributed_square_kernel(float* value) { value[0] = value[0] * value[0]; }
__global__ void distributed_sqrt_kernel(float* value) { value[0] = sqrtf(value[0]); }

// ── making it ─────────────────────────────────────────────────────────────────

/// Reads RANK, LOCAL_RANK and WORLD_SIZE (0, 0 and 1 when they are not set), makes GPU LOCAL_RANK this
/// process's GPU, and with more than one process joins them with NCCL: rank 0 writes NCCL's id to id_file,
/// the others wait for it (up to 2 minutes) and read it. Make it first in main, before anything uses the GPU:
/// memory and cuBLAS go to the GPU chosen here. id_file must be new for every run, or a rank could read the
/// last run's id: torchrun's TORCHELASTIC_RUN_ID in its name does that.
inline Distributed::Distributed(const std::string& id_file) {
    rank_ = env_int("RANK", 0);
    local_rank_ = env_int("LOCAL_RANK", 0);
    world_size_ = env_int("WORLD_SIZE", 1);
    if (world_size_ < 1 || rank_ < 0 || rank_ >= world_size_) {
        fprintf(stderr, "error: RANK %d and WORLD_SIZE %d: need 0 <= RANK < WORLD_SIZE\n", rank_, world_size_);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(local_rank_));
    one_value_ = Tensor::zeros({1});
    if (world_size_ == 1) return;

    ncclUniqueId id;
    if (rank_ == 0) {
        ncclCheck(ncclGetUniqueId(&id));
        std::string partial = id_file + ".partial";              // written whole, then renamed: never read half
        FILE* file = fopen(partial.c_str(), "wb");
        if (file == nullptr || fwrite(&id, sizeof(id), 1, file) != 1 || fclose(file) != 0 ||
            rename(partial.c_str(), id_file.c_str()) != 0) {
            fprintf(stderr, "error: rank 0 cannot write NCCL's id to %s\n", id_file.c_str());
            exit(EXIT_FAILURE);
        }
    } else {
        FILE* file = nullptr;
        for (int waited_ms = 0; file == nullptr; waited_ms += 10) {
            file = fopen(id_file.c_str(), "rb");
            if (file != nullptr) break;
            if (waited_ms >= 120000) {
                fprintf(stderr, "error: rank %d waited 2 minutes for NCCL's id in %s\n", rank_, id_file.c_str());
                exit(EXIT_FAILURE);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (fread(&id, sizeof(id), 1, file) != 1) {
            fprintf(stderr, "error: rank %d cannot read NCCL's id from %s\n", rank_, id_file.c_str());
            exit(EXIT_FAILURE);
        }
        fclose(file);
    }
    ncclCheck(ncclCommInitRank(&comm_, world_size_, id, rank_));   // returns when every rank has joined
    if (rank_ == 0) remove(id_file.c_str());                       // everyone has read it
}

/// Leaves NCCL, after everything queued on the GPU is done.
inline Distributed::~Distributed() {
    if (comm_ != nullptr) {
        cudaDeviceSynchronize();
        ncclCommDestroy(comm_);
    }
}

// ── who this process is ───────────────────────────────────────────────────────

/// Tells which process this is: 0 .. world_size - 1. Rank 0 is the one that prints.
inline int Distributed::rank() const { return rank_; }

/// Tells which GPU of this machine this process uses.
inline int Distributed::local_rank() const { return local_rank_; }

/// Tells how many processes train together, one per GPU.
inline int Distributed::world_size() const { return world_size_; }

// ── the collectives ───────────────────────────────────────────────────────────

/// Averages flat over the ranks, but each rank gets only its own slice of the average, in place: rank r's
/// values r x slice .. (r + 1) x slice become the mean of every rank's values there; the rest of flat is left
/// as it was (this rank's own, not averaged). flat is all the grads, float or bfloat16; its values must split
/// into world_size slices, as the optimizer's do. With one process, nothing to do.
inline void Distributed::reduce_scatter_mean(const Tensor& flat) const {
    if (world_size_ == 1) return;
    size_t slice = slice_size_of(flat);
    char* slice_start = (char*)flat.data_ptr() + rank_ * slice * flat.element_size();
    ncclCheck(ncclReduceScatter(flat.data_ptr(), slice_start, slice, nccl_type(flat.dtype()), ncclAvg, comm_, 0));
}

/// Sends each rank's slice of flat to every rank, in place: after it, every rank's flat holds every rank's
/// slice. flat is all the params, after each rank updated its own slice. With one process, nothing to do.
inline void Distributed::all_gather(const Tensor& flat) const {
    if (world_size_ == 1) return;
    size_t slice = slice_size_of(flat);
    char* slice_start = (char*)flat.data_ptr() + rank_ * slice * flat.element_size();
    ncclCheck(ncclAllGather(slice_start, flat.data_ptr(), slice, nccl_type(flat.dtype()), comm_, 0));
}

/// Turns each rank's norm of its own slice into the norm of all the slices together, on every rank, in place:
/// sqrt(norm_0² + norm_1² + ... ), as the squares of all the values add up slice by slice. norm is 1 float on
/// the GPU (the optimizer's grad_norm). With one process the slice is all of it: nothing to do.
inline void Distributed::slice_norms_to_norm(const Tensor& norm) const {
    if (world_size_ == 1) return;
    distributed_square_kernel<<<1, 1>>>(norm.data());
    cudaCheck(cudaGetLastError());
    ncclCheck(ncclAllReduce(norm.data(), norm.data(), 1, ncclFloat, ncclSum, comm_, 0));
    distributed_sqrt_kernel<<<1, 1>>>(norm.data());
    cudaCheck(cudaGetLastError());
}

/// Gives the average of a CPU number over the ranks, on every rank: each step's loss, each val loss. It waits
/// for the GPU. With one process, the number itself.
inline float Distributed::mean_of_cpu_value(float value) const {
    if (world_size_ == 1) return value;
    one_value_.copy_from_cpu(&value);
    ncclCheck(ncclAllReduce(one_value_.data(), one_value_.data(), 1, ncclFloat, ncclAvg, comm_, 0));
    return one_value_.item<float>();
}

/// Waits until every rank has got here and its GPU is done with all it was given: a checkpoint is marked
/// whole only after every rank wrote its part. An all-reduce of one number, waited for. With one process, it
/// only waits for the GPU.
inline void Distributed::barrier() const {
    mean_of_cpu_value(0.0f);
    cudaCheck(cudaDeviceSynchronize());
}

// ── private ───────────────────────────────────────────────────────────────────

/// Gives how many values of flat each rank's slice holds, and stops if they do not split evenly.
inline size_t Distributed::slice_size_of(const Tensor& flat) const {
    if (flat.numel() % world_size_ != 0) {
        fprintf(stderr, "error: %zu values do not split into %d slices\n", flat.numel(), world_size_);
        exit(EXIT_FAILURE);
    }
    return flat.numel() / world_size_;
}

/// Reads an environment variable as an int; `fallback` if it is not set.
inline int Distributed::env_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value != nullptr ? atoi(value) : fallback;
}

/// Gives NCCL's name for a dtype: ncclFloat for float, ncclBfloat16 for bfloat16; stops for the others.
inline ncclDataType_t Distributed::nccl_type(DType dtype) {
    if (dtype == DType::Float32) return ncclFloat;
    if (dtype == DType::BFloat16) return ncclBfloat16;
    fprintf(stderr, "error: the collectives here take float or bfloat16\n");
    exit(EXIT_FAILURE);
}

#endif // DISTRIBUTED_CUH
