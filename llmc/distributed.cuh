// distributed.cuh — training on several GPUs of one machine, nanochat's way: one process per GPU, each with
// the whole model and its own batches, each updating its own slice of the params. NCCL moves the data. One step:
//     every GPU   forward + backward on its own batches         -> its own grads
//     reduce_scatter_mean(all grads)                            -> each GPU: the average of its slice
//     slice_norms_to_norm                                       -> the norm of all the averaged grads
//     AdamW on its slice                                        -> its slice of the new params
//     all_gather(all params)                                    -> every GPU: all the new params
// Each process reads RANK, LOCAL_RANK and WORLD_SIZE as torchrun sets them (none: one process, no NCCL):
//     torchrun --standalone --nproc_per_node=8 --no-python ./train_gpt2
#ifndef DISTRIBUTED_CUH
#define DISTRIBUTED_CUH

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <nccl.h>
#include "tensor.cuh"

/// Stops with the file and line if an NCCL call failed.
///
///   result       what the call returned
///   file, line   where the call is
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
    int rank_ = 0;
    int local_rank_ = 0;
    int world_size_ = 1;
    ncclComm_t comm_ = nullptr;          // NCCL's link between the processes; none with one process
    Tensor one_value_;                   // 1 float on the GPU, for mean_of_cpu_value()

    size_t slice_size_of(const Tensor& flat) const;                     // flat's values / world_size, checked
    static int env_int(const char* name, int fallback);                 // an environment variable as an int
    static ncclDataType_t nccl_type(DType dtype);                       // ncclFloat or ncclBfloat16
};

// ══ the code, in the order of the list above ══════════════════════════════════

// slice_norms_to_norm's two 1-value kernels: norm -> norm², then back.
__global__ void distributed_square_kernel(float* value) { value[0] = value[0] * value[0]; }
__global__ void distributed_sqrt_kernel(float* value) { value[0] = sqrtf(value[0]); }

// ── making it ─────────────────────────────────────────────────────────────────

/// Picks GPU LOCAL_RANK and, with several processes, joins them with NCCL: rank 0 writes NCCL's id to id_file,
/// the others wait for it (up to 2 minutes). Make it first in main, before anything uses the GPU.
///
///   id_file   new for every run, or a rank could read the last run's id
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
        std::string partial = id_file + ".partial";              // written whole, then renamed
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

/// Gives which process this is: 0 .. world_size - 1. Rank 0 prints.
inline int Distributed::rank() const { return rank_; }

/// Gives which GPU of this machine this process uses.
inline int Distributed::local_rank() const { return local_rank_; }

/// Gives how many processes train together, one per GPU.
inline int Distributed::world_size() const { return world_size_; }

// ── the collectives ───────────────────────────────────────────────────────────

/// Averages flat over the ranks in place, each rank getting only its own slice of the average; the rest of
/// flat is left as it was.
///
///   flat   all the grads, float or bfloat16, splitting into world_size slices
inline void Distributed::reduce_scatter_mean(const Tensor& flat) const {
    if (world_size_ == 1) return;
    size_t slice = slice_size_of(flat);
    char* slice_start = (char*)flat.data_ptr() + rank_ * slice * flat.element_size();
    ncclCheck(ncclReduceScatter(flat.data_ptr(), slice_start, slice, nccl_type(flat.dtype()), ncclAvg, comm_, 0));
}

/// Sends each rank's slice of flat to every rank, in place.
///
///   flat   all the params, after each rank updated its slice
inline void Distributed::all_gather(const Tensor& flat) const {
    if (world_size_ == 1) return;
    size_t slice = slice_size_of(flat);
    char* slice_start = (char*)flat.data_ptr() + rank_ * slice * flat.element_size();
    ncclCheck(ncclAllGather(slice_start, flat.data_ptr(), slice, nccl_type(flat.dtype()), comm_, 0));
}

/// Turns each rank's norm of its slice into the norm of all, in place: sqrt(norm_0² + norm_1² + ...).
///
///   norm   1 float on the GPU: the optimizer's grad_norm
inline void Distributed::slice_norms_to_norm(const Tensor& norm) const {
    if (world_size_ == 1) return;
    distributed_square_kernel<<<1, 1>>>(norm.data());
    cudaCheck(cudaGetLastError());
    ncclCheck(ncclAllReduce(norm.data(), norm.data(), 1, ncclFloat, ncclSum, comm_, 0));
    distributed_sqrt_kernel<<<1, 1>>>(norm.data());
    cudaCheck(cudaGetLastError());
}

/// Gives the average of a CPU number over the ranks, on every rank. Waits for the GPU.
///
///   value   this rank's: a loss
inline float Distributed::mean_of_cpu_value(float value) const {
    if (world_size_ == 1) return value;
    one_value_.copy_from_cpu(&value);
    ncclCheck(ncclAllReduce(one_value_.data(), one_value_.data(), 1, ncclFloat, ncclAvg, comm_, 0));
    return one_value_.item<float>();
}

/// Waits until every rank gets here and its GPU is done: an all-reduce of one number.
inline void Distributed::barrier() const {
    mean_of_cpu_value(0.0f);
    cudaCheck(cudaDeviceSynchronize());
}

// ── private ───────────────────────────────────────────────────────────────────

/// Gives how many values each rank's slice of flat holds; stops if they do not split evenly.
///
///   flat   the tensor to split
inline size_t Distributed::slice_size_of(const Tensor& flat) const {
    if (flat.numel() % world_size_ != 0) {
        fprintf(stderr, "error: %zu values do not split into %d slices\n", flat.numel(), world_size_);
        exit(EXIT_FAILURE);
    }
    return flat.numel() / world_size_;
}

/// Gives an environment variable as an int.
///
///   name       the variable
///   fallback   if it is not set
inline int Distributed::env_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value != nullptr ? atoi(value) : fallback;
}

/// Gives NCCL's name for a dtype; stops for others.
///
///   dtype   Float32: ncclFloat; BFloat16: ncclBfloat16
inline ncclDataType_t Distributed::nccl_type(DType dtype) {
    if (dtype == DType::Float32) return ncclFloat;
    if (dtype == DType::BFloat16) return ncclBfloat16;
    fprintf(stderr, "error: the collectives here take float or bfloat16\n");
    exit(EXIT_FAILURE);
}

#endif // DISTRIBUTED_CUH
