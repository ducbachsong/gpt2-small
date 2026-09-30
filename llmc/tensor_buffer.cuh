// tensor_buffer.cuh — many tensors in one GPU allocation, one after the other. The
// trainer keeps its parameters, gradients and AdamW's m and v this way.
//
// The class below lists every function, grouped, one line each. Their code follows
// the class, in the same order and under the same headings.

// Why one allocation: one cudaMalloc and one memset instead of one per tensor, and a
// pass over the whole model (the gradient norm, zeroing the gradients) is one launch
// over flat(). Nothing to free: the memory goes when the last Tensor using it goes.
#ifndef TENSOR_BUFFER_CUH
#define TENSOR_BUFFER_CUH

#include <vector>
#include "tensor.cuh"

class TensorBuffer {
public:
    // ── adding tensors, then one allocation ─────────────────────────────────────
    int add(const Shape& shape);                                        // one tensor; returns its index k
    int add_many(const std::vector<Shape>& shapes);                     // many; returns the first index
    void allocate(DType dtype = DType::Float32);                        // one GPU allocation for all, zeros
    static TensorBuffer zeros_like(const TensorBuffer& other);          // other's tensors, memory of its own

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& tensor(int k) const;                                  // tensor k, with its shape
    const Tensor& flat() const;                                         // every value of every tensor, one flat Tensor
    int num_tensors() const;                                            // how many tensors were added
    size_t numel() const;                                               // how many values, all tensors together

private:
    std::vector<Shape> shapes_;          // each tensor's shape, in the order added
    std::vector<Tensor> tensors_;        // tensor k: a view into flat_, made by allocate()
    Tensor flat_;                        // the one allocation: every value, flat
    bool allocated_ = false;

    void check_index(int k) const;                                      // stops if there is no tensor k
    static size_t numel_of(const Shape& shape);                         // the shape's sizes multiplied
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── adding tensors, then one allocation ───────────────────────────────────────

/// {768, 3072} for a matrix, {768} for a bias. Only before allocate().
inline int TensorBuffer::add(const Shape& shape) {
    if (allocated_) {
        fprintf(stderr, "error: tensor %d added after allocate()\n", num_tensors());
        exit(EXIT_FAILURE);
    }
    shapes_.push_back(shape);
    return num_tensors() - 1;
}

/// add_many({{2, 4}, {12}, {2, 2, 3}}) adds tensors 0, 1 and 2.
inline int TensorBuffer::add_many(const std::vector<Shape>& shapes) {
    int first = num_tensors();
    for (const Shape& shape : shapes) add(shape);
    return first;
}

/// Makes one zeros Tensor of every value, then each tensor as a view of its part:
///
///     flat_      [ 0 .. 7 | 8 .. 19 ]
///     tensor 0    narrow(0, 0, 8).view({2, 4})
///     tensor 1              narrow(0, 8, 12).view({12})
///
/// float4 kernels (adamw.cuh) need every tensor to start at a multiple of 4 values:
/// true when every tensor's numel is a multiple of 4, as all of GPT-2's are.
inline void TensorBuffer::allocate(DType dtype) {
    if (allocated_) {
        fprintf(stderr, "error: allocate() called twice\n");
        exit(EXIT_FAILURE);
    }
    flat_ = Tensor::zeros({numel()}, dtype);
    size_t start = 0;
    for (const Shape& shape : shapes_) {
        size_t size = numel_of(shape);                                  // this tensor's number of values
        tensors_.push_back(flat_.narrow(0, start, size).view(shape));
        start += size;
    }
    allocated_ = true;
}

/// The trainer's gradients and AdamW's m and v are made like its parameters.
inline TensorBuffer TensorBuffer::zeros_like(const TensorBuffer& other) {
    TensorBuffer buffer;
    buffer.add_many(other.shapes_);
    buffer.allocate(other.flat_.dtype());
    return buffer;
}

// ── reading ───────────────────────────────────────────────────────────────────

/// Shares the buffer's memory: writing into tensor(k) writes into flat().
inline const Tensor& TensorBuffer::tensor(int k) const {
    check_index(k);
    if (!allocated_) {
        fprintf(stderr, "error: tensor(%d) before allocate()\n", k);
        exit(EXIT_FAILURE);
    }
    return tensors_[k];
}

inline const Tensor& TensorBuffer::flat() const { return flat_; }
inline int TensorBuffer::num_tensors() const { return (int)shapes_.size(); }
/// The shapes' sizes added up, so it is known before allocate() too.
inline size_t TensorBuffer::numel() const {
    size_t total = 0;
    for (const Shape& shape : shapes_) total += numel_of(shape);
    return total;
}

// ── private helpers ───────────────────────────────────────────────────────────

inline void TensorBuffer::check_index(int k) const {
    if (k < 0 || k >= num_tensors()) {
        fprintf(stderr, "error: no tensor %d (the buffer has %d)\n", k, num_tensors());
        exit(EXIT_FAILURE);
    }
}

inline size_t TensorBuffer::numel_of(const Shape& shape) {
    size_t numel = 1;
    for (size_t size : shape) numel *= size;
    return numel;
}

#endif // TENSOR_BUFFER_CUH
