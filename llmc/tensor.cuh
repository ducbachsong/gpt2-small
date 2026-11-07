// tensor.cuh — a tensor on the GPU, a small version of libtorch's at::Tensor, called the same way:
// Tensor::zeros({2, 4}), t.view({8}), t[1][2].item(), ... Values in row order, the last dim fastest:
//     data     [ a b c d | e f g h ]      shape   { 2, 4 }
//                row 0     row 1          strides { 4, 1 }: [i][j] is data[i * 4 + j * 1]
// A tensor and its views share one piece of GPU memory, freed when the last of them is gone.
//
//     dtype      significant bits   about     largest   bits
//     float      24                 7 digits  3.4e38    1 sign, 8 exponent, 23 fraction
//     half       11                 3 digits  65504     1 sign, 5 exponent, 10 fraction
//     bfloat16    8                 2 digits  3.4e38    1 sign, 8 exponent,  7 fraction: a float's first 16 bits
//     int        4 bytes, for token ids
// Kernels do their math in float: read4_as_float reads 4 values as a float4, write4_from_float writes them
// back (end of file).
#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>                  // nv_bfloat16
#include <cuda_fp16.h>                  // half
#include <memory>
#include <type_traits>
#include <vector>
#include "kernels/cuda_common.h"

/// What one value is.
enum class DType { Float32, Float16, BFloat16, Int32 };

// ── one value of each float dtype, as a float ─────────────────────────────────
// to_float is exact; from_float<T> rounds to the nearest T, a tie to the one whose last bit is 0. CPU and GPU.

__host__ __device__ __forceinline__ float to_float(float value) { return value; }
__host__ __device__ __forceinline__ float to_float(half value) { return __half2float(value); }
__host__ __device__ __forceinline__ float to_float(nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> __host__ __device__ __forceinline__ T from_float(float value);
template <> __host__ __device__ __forceinline__ float from_float<float>(float value) { return value; }
template <> __host__ __device__ __forceinline__ half from_float<half>(float value) { return __float2half(value); }
template <> __host__ __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

/// A shape: {2, 4} for a 2x4 matrix, {768} for a bias, {} for a single value.
using Shape = std::vector<size_t>;

class Tensor {
public:
    // ── making tensors ──────────────────────────────────────────────────────────
    Tensor();                                                           // no values yet; give it one later
    static Tensor empty(const Shape& shape, DType dtype = DType::Float32);      // values unset
    static Tensor zeros(const Shape& shape, DType dtype = DType::Float32);      // all 0
    static Tensor ones(const Shape& shape, DType dtype = DType::Float32);       // all 1
    static Tensor randn(const Shape& shape, DType dtype = DType::Float32);      // random, mean 0, std 1
    static Tensor zeros_like(const Tensor& other);                      // all 0, other's shape and dtype
    static Tensor empty_like(const Tensor& other);                      // unset, other's shape and dtype
    static Tensor from_blob(void* data, const Shape& shape, DType dtype = DType::Float32);  // GPU memory you own
    static void manual_seed(uint64_t seed);                             // same seed, same random numbers

    // ── shape and info ──────────────────────────────────────────────────────────
    DType dtype() const;                                                // Float32, Float16, BFloat16 or Int32
    size_t element_size() const;                                        // bytes per value: 4, 2, 2 or 4
    int dim() const;                                                    // how many dims: 2 for a matrix
    size_t numel() const;                                               // how many values
    const Shape& shape() const;                                         // {2, 4}
    const Shape& stride() const;                                        // {4, 1}: all the strides
    size_t size(int d) const;                                           // shape()[d], checked
    size_t stride(int d) const;                                         // stride()[d], checked
    float* data() const;                                                // first value on the GPU (float tensor)
    template <typename T> T* data_ptr() const;                          // as float*, half*, nv_bfloat16* or int*
    void* data_ptr() const;                                             // first value's address, any dtype

    // ── views: the same values, shared, not copied ──────────────────────────────
    Tensor view(const Shape& shape) const;                              // another shape: 2x4 as {8}
    Tensor narrow(int dim, size_t start, size_t length) const;          // some rows (dim 0 only)
    Tensor operator[](size_t i) const;                                  // t[1]: row 1; t[1][2]: one value

    // ── setting values (on the GPU; each returns the tensor) ────────────────────
    const Tensor& zero_() const;                                        // every value 0
    const Tensor& fill_(double value) const;                            // every value `value`
    const Tensor& normal_(float mean, float std) const;                 // random: normal_(0, 0.02)
    const Tensor& copy_(const Tensor& other) const;                     // other's values, GPU to GPU; any float dtype

    // ── copies and the CPU ──────────────────────────────────────────────────────
    Tensor clone() const;                                               // a copy in memory of its own
    Tensor to(DType dtype) const;                                       // the values as another dtype: to(Float16)
    template <typename T> void copy_from_cpu(const T* values) const;    // numel values in, from the CPU
    template <typename T> void copy_to_cpu(T* values) const;            // numel values out, to the CPU
    template <typename T = float> std::vector<T> tolist() const;        // every value, as a CPU vector
    template <typename T = float> T item() const;                       // the one value: t[1][2].item()

private:
    std::shared_ptr<void> memory_;       // frees the GPU memory after the last tensor using it; empty for from_blob
    void* data_;
    Shape shape_;
    Shape strides_;
    size_t numel_;
    DType dtype_;

    Tensor(std::shared_ptr<void> memory, void* data, const Shape& shape, DType dtype);
    size_t bytes() const;                                               // numel x element_size
    void* element(size_t offset) const;                                 // the address of value `offset`
    static size_t numel_of(const Shape& shape);                         // the shape multiplied
    static size_t element_size_of(DType dtype);                         // 4, 2, 2 or 4
    static const char* dtype_name(DType dtype);                         // "float", "half", "bfloat16", "int"
    void check_dim(int d) const;                                        // stops if there is no dim d
    template <typename T> void check_dtype() const;                     // stops if T is not the dtype

    static uint64_t& random_state();                                    // for randn and normal_
    static uint32_t random_u32();
    static float random_uniform();
    static float random_normal();
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── making tensors ────────────────────────────────────────────────────────────

/// Makes a tensor with no values: Tensor t; ... t = Tensor::zeros({4});
inline Tensor::Tensor() : data_(nullptr), numel_(0), dtype_(DType::Float32) {}

/// Makes a tensor in new GPU memory; its values are whatever was there.
///
///   shape, dtype   the new tensor's
inline Tensor Tensor::empty(const Shape& shape, DType dtype) {
    size_t bytes = numel_of(shape) * element_size_of(dtype);
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

/// Makes a random tensor: the same numbers as a float randn after the same seed, rounded to the dtype.
///
///   shape, dtype   the new tensor's
inline Tensor Tensor::randn(const Shape& shape, DType dtype) {
    Tensor tensor = empty(shape, dtype);
    tensor.normal_(0.0f, 1.0f);
    return tensor;
}

inline Tensor Tensor::zeros_like(const Tensor& other) { return zeros(other.shape_, other.dtype_); }

inline Tensor Tensor::empty_like(const Tensor& other) { return empty(other.shape_, other.dtype_); }

/// Makes a tensor on GPU memory that already exists; it neither copies nor frees it.
///
///   data           the first value, in row order
///   shape, dtype   the tensor's
inline Tensor Tensor::from_blob(void* data, const Shape& shape, DType dtype) {
    return Tensor(nullptr, data, shape, dtype);
}

inline void Tensor::manual_seed(uint64_t seed) { random_state() = seed != 0 ? seed : 0x9E3779B97F4A7C15ull; }

// ── shape and info ────────────────────────────────────────────────────────────

inline DType Tensor::dtype() const { return dtype_; }
inline size_t Tensor::element_size() const { return element_size_of(dtype_); }
inline int Tensor::dim() const { return (int)shape_.size(); }
inline size_t Tensor::numel() const { return numel_; }
inline const Shape& Tensor::shape() const { return shape_; }
inline const Shape& Tensor::stride() const { return strides_; }

/// Gives how many values along dim d: for a matrix, dim 0 is its rows and dim 1 its columns.
///
///   d   the dim
inline size_t Tensor::size(int d) const {
    check_dim(d);
    return shape_[d];
}

/// Gives how many values one step along dim d moves in memory: {4, 1} for a 2x4 matrix.
///
///   d   the dim
inline size_t Tensor::stride(int d) const {
    check_dim(d);
    return strides_[d];
}

inline float* Tensor::data() const { return data_ptr<float>(); }

/// Gives the address alone, any dtype: for alignment checks, and for cuBLAS.
inline void* Tensor::data_ptr() const { return data_; }

/// Gives the first value's address as T*; the wrong T stops the program.
///
///   T   float, half, nv_bfloat16 or int: the tensor's dtype
template <typename T>
inline T* Tensor::data_ptr() const {
    check_dtype<T>();
    return (T*)data_;
}

// ── views: the same values, shared, not copied ────────────────────────────────

/// Gives the same values in another shape.
///
///   shape   the same number of values
inline Tensor Tensor::view(const Shape& shape) const {
    if (numel_of(shape) != numel_) {
        fprintf(stderr, "error: cannot view %zu values as a shape of %zu values\n", numel_, numel_of(shape));
        exit(EXIT_FAILURE);
    }
    return Tensor(memory_, data_, shape, dtype_);
}

/// Gives some rows: rows 1 and 2 of a 4x3 tensor are narrow(0, 1, 2), a 2x3 tensor.
///
///   dim             only 0: other dims would not be in row order
///   start, length   the first row and how many
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

/// Gives row i along dim 0, without that dim: t[1] of a 2x4 is 4 values; t[1][2] one value (read it with item()).
///
///   i   the row
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
// const, as in libtorch: they change the values, not the tensor. They return it: t.zero_().fill_(1).

inline const Tensor& Tensor::zero_() const {
    cudaCheck(cudaMemset(data_, 0, bytes()));
    return *this;
}

/// Sets n values to the same bytes.
///
///   data   the values: Word is uint32_t for float and int, uint16_t for half and bfloat16
///   n      how many
///   bits   the value's bytes
template <typename Word>
__global__ void tensor_fill_kernel(Word* data, size_t n, Word bits) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = bits;
}

/// Sets every value; a 2-byte dtype gets it rounded: fill_(0.1) is 0.0999756 in a half, 0.100098 in a bfloat16.
///
///   value   1.5 for floats, 7 for ints
inline const Tensor& Tensor::fill_(double value) const {
    if (numel_ == 0) return *this;
    unsigned blocks = (unsigned)CEIL_DIV(numel_, 256);
    if (element_size() == 2) {
        uint16_t bits;
        if (dtype_ == DType::Float16) {
            half as_half = from_float<half>((float)value);
            std::memcpy(&bits, &as_half, 2);
        } else {
            nv_bfloat16 as_bfloat16 = from_float<nv_bfloat16>((float)value);
            std::memcpy(&bits, &as_bfloat16, 2);
        }
        tensor_fill_kernel<<<blocks, 256>>>((uint16_t*)data_, numel_, bits);
    } else {
        uint32_t bits;
        if (dtype_ == DType::Float32) {
            float as_float = (float)value;
            std::memcpy(&bits, &as_float, 4);
        } else {
            int as_int = (int)value;
            std::memcpy(&bits, &as_int, 4);
        }
        tensor_fill_kernel<<<blocks, 256>>>((uint32_t*)data_, numel_, bits);
    }
    cudaCheck(cudaGetLastError());
    return *this;
}

/// Sets random values, made on the CPU as floats, rounded to the dtype: one seed, the same numbers in every dtype.
///
///   mean, std   normal_(0, 0.02)
inline const Tensor& Tensor::normal_(float mean, float std) const {
    if (dtype_ == DType::Int32) {
        fprintf(stderr, "error: normal_ on an int tensor: it needs float, half or bfloat16\n");
        exit(EXIT_FAILURE);
    }
    std::vector<float> values(numel_);
    for (float& value : values) value = mean + std * random_normal();
    if (dtype_ == DType::Float32) {
        copy_from_cpu(values.data());
    } else if (dtype_ == DType::Float16) {
        std::vector<half> halves(numel_);
        for (size_t i = 0; i < numel_; i++) halves[i] = from_float<half>(values[i]);
        copy_from_cpu(halves.data());
    } else {
        std::vector<nv_bfloat16> bfloat16s(numel_);
        for (size_t i = 0; i < numel_; i++) bfloat16s[i] = from_float<nv_bfloat16>(values[i]);
        copy_from_cpu(bfloat16s.data());
    }
    return *this;
}

/// Converts n values between float dtypes: read as float, rounded to To. A half overflows from 65520.
///
///   to     the result: float, half or nv_bfloat16
///   from   the values: float, half or nv_bfloat16
///   n      how many
template <typename To, typename From>
__global__ void tensor_convert_kernel(To* to, const From* from, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) to[i] = from_float<To>(to_float(from[i]));
}

/// Launches tensor_convert_kernel for the dtype of `from`.
///
///   to           the result
///   from         the values
///   from_dtype   float, half or bfloat16
///   n            how many
template <typename To>
inline void tensor_convert(To* to, const void* from, DType from_dtype, size_t n) {
    unsigned blocks = (unsigned)CEIL_DIV(n, 256);
    if (from_dtype == DType::Float32)
        tensor_convert_kernel<<<blocks, 256>>>(to, (const float*)from, n);
    else if (from_dtype == DType::Float16)
        tensor_convert_kernel<<<blocks, 256>>>(to, (const half*)from, n);
    else
        tensor_convert_kernel<<<blocks, 256>>>(to, (const nv_bfloat16*)from, n);
}

/// Copies other's values in, GPU to GPU: as they are in the same dtype, converted between float dtypes.
/// Ints copy only into ints.
///
///   other   the same numel; the shape may differ
inline const Tensor& Tensor::copy_(const Tensor& other) const {
    if (other.numel_ != numel_) {
        fprintf(stderr, "error: copy_ of %zu values into %zu\n", other.numel_, numel_);
        exit(EXIT_FAILURE);
    }
    if (other.dtype_ == dtype_) {
        cudaCheck(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice));
        return *this;
    }
    if (dtype_ == DType::Int32 || other.dtype_ == DType::Int32) {
        fprintf(stderr, "error: copy_ of a %s tensor into a %s tensor\n", dtype_name(other.dtype_), dtype_name(dtype_));
        exit(EXIT_FAILURE);
    }
    if (numel_ == 0) return *this;
    if (dtype_ == DType::Float32)
        tensor_convert((float*)data_, other.data_, other.dtype_, numel_);
    else if (dtype_ == DType::Float16)
        tensor_convert((half*)data_, other.data_, other.dtype_, numel_);
    else
        tensor_convert((nv_bfloat16*)data_, other.data_, other.dtype_, numel_);
    cudaCheck(cudaGetLastError());
    return *this;
}

// ── copies and the CPU ────────────────────────────────────────────────────────

/// Gives a copy in memory of its own.
inline Tensor Tensor::clone() const {
    Tensor copy = empty(shape_, dtype_);
    copy.copy_(*this);
    return copy;
}

/// Gives the values as another dtype, in memory of their own; its own dtype gives this tensor back, shared.
///
///   dtype   to(DType::BFloat16) rounds each float
inline Tensor Tensor::to(DType dtype) const {
    if (dtype == dtype_) return *this;
    Tensor converted = empty(shape_, dtype);
    converted.copy_(*this);
    return converted;
}

/// Copies numel values in from the CPU.
///
///   values   in row order, of the tensor's dtype
template <typename T>
inline void Tensor::copy_from_cpu(const T* values) const {
    check_dtype<T>();
    cudaCheck(cudaMemcpy(data_, values, bytes(), cudaMemcpyHostToDevice));
}

/// Copies numel values out to the CPU.
///
///   values   room for numel values of the tensor's dtype
template <typename T>
inline void Tensor::copy_to_cpu(T* values) const {
    check_dtype<T>();
    cudaCheck(cudaMemcpy(values, data_, bytes(), cudaMemcpyDeviceToHost));
}

/// Gives every value on the CPU, flat, in row order: a 2x4 gives 8 values.
///
///   T   the tensor's dtype: tolist<nv_bfloat16>() for a bfloat16 one
template <typename T>
inline std::vector<T> Tensor::tolist() const {
    std::vector<T> values(numel_);
    copy_to_cpu(values.data());
    return values;
}

/// Gives the one value of a 1-value tensor.
///
///   T   the tensor's dtype
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

/// Makes a tensor on memory, working out the strides and numel. Shape {} is 1 value with no dims.
///
///   memory         frees the GPU memory; empty for from_blob
///   data           the first value
///   shape, dtype   the tensor's
inline Tensor::Tensor(std::shared_ptr<void> memory, void* data, const Shape& shape, DType dtype)
    : memory_(memory), data_(data), shape_(shape), strides_(shape.size()), numel_(1), dtype_(dtype) {
    for (int d = dim() - 1; d >= 0; d--) {       // from the last dim: its stride is 1
        strides_[d] = numel_;
        numel_ *= shape_[d];
    }
}

inline size_t Tensor::bytes() const { return numel_ * element_size(); }

inline void* Tensor::element(size_t offset) const { return (char*)data_ + offset * element_size(); }

inline size_t Tensor::numel_of(const Shape& shape) {
    size_t numel = 1;
    for (size_t size : shape) numel *= size;
    return numel;
}

inline size_t Tensor::element_size_of(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ? 2 : 4;
}

inline const char* Tensor::dtype_name(DType dtype) {
    return dtype == DType::Float32 ? "float" : dtype == DType::Float16 ? "half" : dtype == DType::BFloat16 ? "bfloat16" : "int";
}

inline void Tensor::check_dim(int d) const {
    if (d < 0 || d >= dim()) {
        fprintf(stderr, "error: a tensor with %d dims has no dim %d\n", dim(), d);
        exit(EXIT_FAILURE);
    }
}

template <typename T>
inline void Tensor::check_dtype() const {
    static_assert(std::is_same<T, float>::value || std::is_same<T, half>::value ||
                      std::is_same<T, nv_bfloat16>::value || std::is_same<T, int>::value,
                  "a Tensor holds float, half, nv_bfloat16 or int");
    DType wanted = std::is_same<T, float>::value         ? DType::Float32
                   : std::is_same<T, half>::value        ? DType::Float16
                   : std::is_same<T, nv_bfloat16>::value ? DType::BFloat16
                                                         : DType::Int32;
    if (wanted != dtype_) {
        fprintf(stderr, "error: a %s tensor read or written as %s\n", dtype_name(dtype_), dtype_name(wanted));
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

/// Gives a uniform number in (0, 1]: never 0, so the log in random_normal is safe.
inline float Tensor::random_uniform() { return ((random_u32() >> 8) + 1) / 16777216.0f; }

/// Gives a normal number, mean 0, std 1, from two uniforms.
inline float Tensor::random_normal() {
    float u1 = random_uniform();
    float u2 = random_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

// ── four values in a kernel, as a float4 ─────────────────────────────────────
// For float one 16-byte load or store; for a 2-byte dtype one 8-byte one, converted on the way.
// `first` must start on 4 values: 16 bytes for float, 8 for the 2-byte dtypes.

#define VALUES_PER_FLOAT4 4   // values read4_as_float, write4_from_float and atomic_add4_from_float move at once

/// Four values of a 2-byte dtype side by side: 8 bytes, read or written as one.
template <typename T>
struct alignas(8) Four {
    T value[4];
};

/// Gives first[0 .. 3] as a float4.
///
///   first   float, half or nv_bfloat16 (T)
template <typename T>
__device__ __forceinline__ float4 read4_as_float(const T* first) {
    if constexpr (std::is_same<T, float>::value) {
        return *reinterpret_cast<const float4*>(first);
    } else {
        Four<T> four = *reinterpret_cast<const Four<T>*>(first);
        return make_float4(to_float(four.value[0]), to_float(four.value[1]), to_float(four.value[2]),
                           to_float(four.value[3]));
    }
}

/// Writes a float4 into first[0 .. 3], each value rounded to T.
///
///   first    float, half or nv_bfloat16 (T)
///   values   the 4 floats
template <typename T>
__device__ __forceinline__ void write4_from_float(T* first, float4 values) {
    if constexpr (std::is_same<T, float>::value) {
        *reinterpret_cast<float4*>(first) = values;
    } else {
        Four<T> four;
        four.value[0] = from_float<T>(values.x);
        four.value[1] = from_float<T>(values.y);
        four.value[2] = from_float<T>(values.z);
        four.value[3] = from_float<T>(values.w);
        *reinterpret_cast<Four<T>*>(first) = four;
    }
}

/// Adds a float4 onto first[0 .. 3] while other threads may add onto the same values. Floats: four atomicAdds.
/// A 2-byte dtype (no atomicAdd on a T4): read the 8 bytes, add in float, round, then write them back with
/// atomicCAS only if they still hold what was read; otherwise go round again.
///
///   first    float, half or nv_bfloat16 (T)
///   values   the 4 floats to add
template <typename T>
__device__ __forceinline__ void atomic_add4_from_float(T* first, float4 values) {
    if constexpr (std::is_same<T, float>::value) {
        atomicAdd(first + 0, values.x);
        atomicAdd(first + 1, values.y);
        atomicAdd(first + 2, values.z);
        atomicAdd(first + 3, values.w);
    } else {
        unsigned long long* word = reinterpret_cast<unsigned long long*>(first);
        unsigned long long seen = *word;
        unsigned long long read;
        do {
            read = seen;
            Four<T> four;
            memcpy(&four, &read, 8);
            four.value[0] = from_float<T>(to_float(four.value[0]) + values.x);
            four.value[1] = from_float<T>(to_float(four.value[1]) + values.y);
            four.value[2] = from_float<T>(to_float(four.value[2]) + values.z);
            four.value[3] = from_float<T>(to_float(four.value[3]) + values.w);
            unsigned long long sum;
            memcpy(&sum, &four, 8);
            seen = atomicCAS(word, read, sum);
        } while (seen != read);
    }
}

#endif // TENSOR_CUH
