// tensor.cuh — a tensor on the GPU, a small version of libtorch's at::Tensor, called
// the same way: Tensor::zeros({2, 4}), t.view({8}), t[1][2].item(), ...
//
// The class below lists every function, grouped, one line each. Their code follows
// the class, in the same order and under the same headings.
//
// The values sit in row order, the last dim moving fastest. A 2x4 tensor:
//
//     data     [ a b c d | e f g h ]      shape   { 2, 4 }
//                row 0     row 1          strides { 4, 1 }: [i][j] is data[i * 4 + j * 1]
//
// Memory, as in libtorch: a tensor and its views share one piece of GPU memory,
// freed by itself when the last of them is gone. Nothing to free by hand.
#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>
#include "kernels/cuda_common.h"

/// What one value is: a float (weights, activations) or an int (token ids). Both 4 bytes.
enum class DType { Float32, Int32 };

/// A shape: {2, 4} for a 2x4 matrix, {768} for a bias, {} for a single value.
using Shape = std::vector<size_t>;

class Tensor {
public:
    // ── making tensors ──────────────────────────────────────────────────────────
    Tensor();                                                           // no values yet; give it one later
    static Tensor empty(const Shape& shape, DType dtype = DType::Float32);      // values unset
    static Tensor zeros(const Shape& shape, DType dtype = DType::Float32);      // all 0
    static Tensor ones(const Shape& shape, DType dtype = DType::Float32);       // all 1
    static Tensor randn(const Shape& shape);                            // random floats, mean 0, std 1
    static Tensor zeros_like(const Tensor& other);                      // all 0, other's shape and dtype
    static Tensor empty_like(const Tensor& other);                      // unset, other's shape and dtype
    static Tensor from_blob(void* data, const Shape& shape, DType dtype = DType::Float32);  // GPU memory you own
    static void manual_seed(uint64_t seed);                             // same seed, same random numbers

    // ── shape and info ──────────────────────────────────────────────────────────
    DType dtype() const;                                                // Float32 or Int32
    int dim() const;                                                    // how many dims: 2 for a matrix
    size_t numel() const;                                               // how many values
    const Shape& shape() const;                                         // {2, 4}
    const Shape& stride() const;                                        // {4, 1}: all the strides
    size_t size(int d) const;                                           // shape()[d], checked
    size_t stride(int d) const;                                         // stride()[d], checked
    float* data() const;                                                // first value on the GPU (float tensor)
    template <typename T> T* data_ptr() const;                          // first value as float* or int*

    // ── views: the same values, shared, not copied ──────────────────────────────
    Tensor view(const Shape& shape) const;                              // another shape: 2x4 as {8}
    Tensor narrow(int dim, size_t start, size_t length) const;          // some rows (dim 0 only)
    Tensor operator[](size_t i) const;                                  // t[1]: row 1; t[1][2]: one value

    // ── setting values (on the GPU; each returns the tensor) ────────────────────
    const Tensor& zero_() const;                                        // every value 0
    const Tensor& fill_(double value) const;                            // every value `value`
    const Tensor& normal_(float mean, float std) const;                 // random: normal_(0, 0.02)
    const Tensor& copy_(const Tensor& other) const;                     // other's values, GPU to GPU

    // ── copies and the CPU ──────────────────────────────────────────────────────
    Tensor clone() const;                                               // a copy in memory of its own
    template <typename T> void copy_from_cpu(const T* values) const;    // numel values in, from the CPU
    template <typename T> void copy_to_cpu(T* values) const;            // numel values out, to the CPU
    template <typename T = float> std::vector<T> tolist() const;        // every value, as a CPU vector
    template <typename T = float> T item() const;                       // the one value: t[1][2].item()

private:
    std::shared_ptr<void> memory_;       // frees the GPU memory after the last tensor using it; empty for from_blob
    void* data_;
    Shape shape_, strides_;
    size_t numel_;
    DType dtype_;

    Tensor(std::shared_ptr<void> memory, void* data, const Shape& shape, DType dtype);
    size_t bytes() const;                                               // numel x 4
    void* element(size_t offset) const;                                 // the address of value `offset`
    static size_t numel_of(const Shape& shape);                         // the shape multiplied
    void check_dim(int d) const;                                        // stops if there is no dim d
    template <typename T> void check_dtype() const;                     // stops if T is not the dtype

    static uint64_t& random_state();                                    // for randn and normal_
    static uint32_t random_u32();
    static float random_uniform();
    static float random_normal();
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── making tensors ────────────────────────────────────────────────────────────

///Undefined at::Tensor: Tensor t; ... t = Tensor::zeros({4});
inline Tensor::Tensor() : data_(nullptr), numel_(0), dtype_(DType::Float32) {}

/// New GPU memory of its own. Its values are whatever was there: set them before reading.
inline Tensor Tensor::empty(const Shape& shape, DType dtype) {
    size_t bytes = numel_of(shape) * 4;                                 // both dtypes are 4 bytes
    void* data;
    cudaCheck(cudaMalloc(&data, bytes));
    std::shared_ptr<void> memory(data, [](void* gpu) { cudaCheck(cudaFree(gpu)); });
    return Tensor(memory, data, shape, dtype);
}

inline Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    Tensor tensor = empty(shape, dtype);
    tensor.zero_();
    return tensor;
}

inline Tensor Tensor::ones(const Shape& shape, DType dtype) {
    Tensor tensor = empty(shape, dtype);
    tensor.fill_(1);
    return tensor;
}

inline Tensor Tensor::randn(const Shape& shape) {
    Tensor tensor = empty(shape, DType::Float32);
    tensor.normal_(0.0f, 1.0f);
    return tensor;
}

inline Tensor Tensor::zeros_like(const Tensor& other) { return zeros(other.shape_, other.dtype_); }

inline Tensor Tensor::empty_like(const Tensor& other) { return empty(other.shape_, other.dtype_); }

/// GPU memory that already exists, in row order. The tensor does not copy it and does not free it.
inline Tensor Tensor::from_blob(void* data, const Shape& shape, DType dtype) {
    return Tensor(nullptr, data, shape, dtype);
}

inline void Tensor::manual_seed(uint64_t seed) { random_state() = seed != 0 ? seed : 0x9E3779B97F4A7C15ull; }

// ── shape and info ────────────────────────────────────────────────────────────

inline DType Tensor::dtype() const { return dtype_; }
inline int Tensor::dim() const { return (int)shape_.size(); }
inline size_t Tensor::numel() const { return numel_; }
inline const Shape& Tensor::shape() const { return shape_; }
inline const Shape& Tensor::stride() const { return strides_; }

/// How many values along dim d: for a matrix, dim 0 is its rows and dim 1 its columns.
inline size_t Tensor::size(int d) const {
    check_dim(d);
    return shape_[d];
}

/// How many values one step along dim d moves in data: {4, 1} for a 2x4 matrix.
inline size_t Tensor::stride(int d) const {
    check_dim(d);
    return strides_[d];
}

inline float* Tensor::data() const { return data_ptr<float>(); }

/// data_ptr<int>() for an Int32 tensor; asking for the wrong type stops the program.
template <typename T>
inline T* Tensor::data_ptr() const {
    check_dtype<T>();
    return (T*)data_;
}

// ── views: the same values, shared, not copied ────────────────────────────────

inline Tensor Tensor::view(const Shape& shape) const {
    if (numel_of(shape) != numel_) {
        fprintf(stderr, "error: cannot view %zu values as a shape of %zu values\n", numel_, numel_of(shape));
        exit(EXIT_FAILURE);
    }
    return Tensor(memory_, data_, shape, dtype_);
}

/// Rows 1 and 2 of a 4x3 tensor are narrow(0, 1, 2), a 2x3 tensor. Only dim 0 for now:
/// other dims would give values out of row order, which this Tensor cannot describe yet.
inline Tensor Tensor::narrow(int dim, size_t start, size_t length) const {
    check_dim(dim);
    if (dim != 0) {
        fprintf(stderr, "error: narrow works on dim 0 only, not dim %d\n", dim);
        exit(EXIT_FAILURE);
    }
    if (start + length > shape_[0]) {
        fprintf(stderr, "error: narrow(0, %zu, %zu) goes past dim 0's %zu values\n", start, length, shape_[0]);
        exit(EXIT_FAILURE);
    }
    Shape shape = shape_;
    shape[0] = length;
    return Tensor(memory_, element(start * strides_[0]), shape, dtype_);
}

/// Row i along dim 0, without that dim: t[1] of a 2x4 is the 4 values of row 1, and
/// t[1][2] is one value, a tensor with no dims (read it with item()).
inline Tensor Tensor::operator[](size_t i) const {
    check_dim(0);
    if (i >= shape_[0]) {
        fprintf(stderr, "error: index %zu, but dim 0 has %zu values\n", i, shape_[0]);
        exit(EXIT_FAILURE);
    }
    Shape shape(shape_.begin() + 1, shape_.end());
    return Tensor(memory_, element(i * strides_[0]), shape, dtype_);
}

// ── setting values ────────────────────────────────────────────────────────────
// const, as in libtorch: they change the values on the GPU, not the tensor's shape
// or pointer. They return the tensor, so calls can follow: t.zero_().fill_(1).

inline const Tensor& Tensor::zero_() const {
    cudaCheck(cudaMemset(data_, 0, bytes()));
    return *this;
}

/// Sets n 32-bit values to `bits`: fill_ for both dtypes, which are both 4 bytes.
__global__ void tensor_fill_kernel(uint32_t* data, size_t n, uint32_t bits) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = bits;
}

/// fill_(1.5) for floats, fill_(7) for ints.
inline const Tensor& Tensor::fill_(double value) const {
    if (numel_ == 0) return *this;
    uint32_t bits;
    if (dtype_ == DType::Float32) {
        float as_float = (float)value;
        std::memcpy(&bits, &as_float, 4);
    } else {
        int as_int = (int)value;
        std::memcpy(&bits, &as_int, 4);
    }
    tensor_fill_kernel<<<(unsigned)CEIL_DIV(numel_, 256), 256>>>((uint32_t*)data_, numel_, bits);
    cudaCheck(cudaGetLastError());
    return *this;
}

/// The numbers are made on the CPU, then copied to the GPU.
inline const Tensor& Tensor::normal_(float mean, float std) const {
    check_dtype<float>();
    std::vector<float> values(numel_);
    for (float& value : values) value = mean + std * random_normal();
    copy_from_cpu(values.data());
    return *this;
}

/// Same dtype and numel; the shapes may differ.
inline const Tensor& Tensor::copy_(const Tensor& other) const {
    if (other.dtype_ != dtype_ || other.numel_ != numel_) {
        fprintf(stderr, "error: copy_ of %zu values into %zu, or of another dtype\n", other.numel_, numel_);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice));
    return *this;
}

// ── copies and the CPU ────────────────────────────────────────────────────────

/// Changing the copy does not change the original, and the other way round.
inline Tensor Tensor::clone() const {
    Tensor copy = empty(shape_, dtype_);
    copy.copy_(*this);
    return copy;
}

/// In row order: floats, or ints for an Int32 tensor.
template <typename T>
inline void Tensor::copy_from_cpu(const T* values) const {
    check_dtype<T>();
    cudaCheck(cudaMemcpy(data_, values, bytes(), cudaMemcpyHostToDevice));
}

template <typename T>
inline void Tensor::copy_to_cpu(T* values) const {
    check_dtype<T>();
    cudaCheck(cudaMemcpy(values, data_, bytes(), cudaMemcpyDeviceToHost));
}

/// Flat, in row order: a 2x4 gives 8 values (PyTorch's tolist nests them by dim).
/// tolist<int>() for ints.
template <typename T>
inline std::vector<T> Tensor::tolist() const {
    std::vector<T> values(numel_);
    copy_to_cpu(values.data());
    return values;
}

/// Only for a tensor of 1 value; item<int>() for ints.
template <typename T>
inline T Tensor::item() const {
    check_dtype<T>();
    if (numel_ != 1) {
        fprintf(stderr, "error: item() needs a tensor of 1 value, not %zu\n", numel_);
        exit(EXIT_FAILURE);
    }
    T value;
    cudaCheck(cudaMemcpy(&value, data_, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Works out the strides and numel from the shape. Shape {} is a tensor with no dims
/// and 1 value, as t[1][2] or torch::zeros({}).
inline Tensor::Tensor(std::shared_ptr<void> memory, void* data, const Shape& shape, DType dtype)
    : memory_(memory), data_(data), shape_(shape), strides_(shape.size()), numel_(1), dtype_(dtype) {
    for (int d = dim() - 1; d >= 0; d--) {       // from the last dim: its stride is 1
        strides_[d] = numel_;
        numel_ *= shape_[d];
    }
}

inline size_t Tensor::bytes() const { return numel_ * 4; }            // both dtypes are 4 bytes

inline void* Tensor::element(size_t offset) const { return (char*)data_ + offset * 4; }

inline size_t Tensor::numel_of(const Shape& shape) {
    size_t numel = 1;
    for (size_t size : shape) numel *= size;
    return numel;
}

inline void Tensor::check_dim(int d) const {
    if (d < 0 || d >= dim()) {
        fprintf(stderr, "error: a tensor with %d dims has no dim %d\n", dim(), d);
        exit(EXIT_FAILURE);
    }
}

template <typename T>
inline void Tensor::check_dtype() const {
    static_assert(std::is_same<T, float>::value || std::is_same<T, int>::value, "a Tensor holds float or int");
    DType wanted = std::is_same<T, float>::value ? DType::Float32 : DType::Int32;
    if (wanted != dtype_) {
        fprintf(stderr, "error: a %s tensor read or written as %s\n", dtype_ == DType::Float32 ? "float" : "int",
                wanted == DType::Float32 ? "float" : "int");
        exit(EXIT_FAILURE);
    }
}

// Random numbers for randn and normal_: xorshift64* for bits, Box-Muller for normals.

inline uint64_t& Tensor::random_state() {
    static uint64_t state = 0x9E3779B97F4A7C15ull;                      // one per program, set by manual_seed
    return state;
}

inline uint32_t Tensor::random_u32() {
    uint64_t& state = random_state();
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 0x2545F4914F6CDD1Dull) >> 32);
}

/// Uniform in (0, 1]: never 0, so the log in random_normal is safe.
inline float Tensor::random_uniform() { return ((random_u32() >> 8) + 1) / 16777216.0f; }

/// Mean 0, std 1: two uniforms make one normal.
inline float Tensor::random_normal() {
    float u1 = random_uniform(), u2 = random_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

#endif // TENSOR_CUH
