// tensor_buffer.cuh — many tensors in one GPU allocation, one after the other. The
//
// The matrices (2-D or more) first in each group:
// the first values up to the border then get weight decay.

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
    int add_many(const std::vector<Shape>& tensor_shapes);              // a group of them; returns its index g
    void allocate(DType dtype = DType::Float32);                        // one GPU allocation for all, zeros
    static TensorBuffer zeros_like(const TensorBuffer& other);          // other's tensors and groups, memory of its own

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& tensor(int k) const;                                  // tensor k, with its shape
    const Tensor& flat() const;                                         // every value of every tensor, one flat Tensor
    int num_tensors() const;                                            // how many tensors were added
    size_t numel() const;                                               // how many values, all tensors together

    // ── groups ──────────────────────────────────────────────────────────────────
    const Tensor& group(int g) const;                                   // every value of group g, one flat Tensor
    const Tensor& tensor(int g, int k) const;                           // tensor k of group g, with its shape
    int group_size(int g) const;                                        // how many tensors group g has
    int num_groups() const;                                             // how many groups were added

private:
    /// A group, added by one add_many: tensors first_tensor_index ..
    /// first_tensor_index + num_tensors - 1.
    struct Group {
        int first_tensor_index;          // the index of its first tensor in the buffer
        int num_tensors;                 // how many tensors it has
    };
    std::vector<Shape> tensor_shapes_;   // each tensor's shape, in the order added
    std::vector<Group> groups_;          // each group, in the order added
    std::vector<Tensor> tensors_;        // tensor k: a view into flat_, made by allocate()
    std::vector<Tensor> group_views_;    // group g: a flat view into flat_, made by allocate()
    Tensor flat_;                        // the one allocation: every value, flat
    bool allocated_ = false;

    int next_tensor_index() const;                                      // the index the next tensor added gets
    int next_group_index() const;                                       // the index the next group added gets
    void stop_if_no_tensor(int k) const;                                // k < 0 or k >= num_tensors()
    void stop_if_no_group(int g) const;                                 // g < 0 or g >= num_groups()
    void stop_if_not_allocated(const char* what) const;                 // called before allocate()
    static size_t numel_of(const Shape& shape);                         // the shape's sizes multiplied
};

// ══ the code, in the order of the list above ══════════════════════════════════

// ── adding tensors, then one allocation ───────────────────────────────────────

/// Adds one tensor of this shape to the buffer and returns its index k, for tensor(k)
/// later. {768, 3072} for a matrix, {768} for a bias. Only before allocate().
inline int TensorBuffer::add(const Shape& shape) {
    int tensor_index = next_tensor_index();
    if (allocated_) {
        fprintf(stderr, "error: tensor %d added after allocate()\n", tensor_index);
        exit(EXIT_FAILURE);
    }
    tensor_shapes_.push_back(shape);
    return tensor_index;
}

/// Adds tensors of these shapes, one after the other, as one group, and returns the
/// group's index g, for group(g) and tensor(g, k) later. add_many({{2, 4}, {12}, {2, 2, 3}})
/// on an empty buffer adds tensors 0, 1 and 2 as group 0; a second add_many is group 1.
inline int TensorBuffer::add_many(const std::vector<Shape>& tensor_shapes) {
    if (tensor_shapes.empty()) {
        fprintf(stderr, "error: add_many of no shapes: a group needs at least one tensor\n");
        exit(EXIT_FAILURE);
    }
    int group_index = next_group_index();
    Group group;
    group.first_tensor_index = next_tensor_index();          // its first tensor is the next one added
    group.num_tensors = (int)tensor_shapes.size();
    for (const Shape& shape : tensor_shapes) add(shape);
    groups_.push_back(group);
    return group_index;
}

/// Gets the GPU memory for every tensor added, in one allocation, all zeros; after it,
/// tensor(k) and group(g) can be used and nothing more can be added. Each tensor and each
/// group is a view of its part of the one flat tensor:
///
///     flat_      [ 0 .. 7 | 8 .. 19 ]
///     tensor 0    narrow(0, 0, 8).view({2, 4})
///     tensor 1              narrow(0, 8, 12).view({12})
///     group 0     narrow(0, 0, 20): both tensors, flat
///
/// float4 kernels (adamw.cuh) need every tensor to start at a multiple of 4 values:
/// true when every tensor's numel is a multiple of 4, as all of GPT-2's are.
inline void TensorBuffer::allocate(DType dtype) {
    if (allocated_) {
        fprintf(stderr, "error: allocate() called twice\n");
        exit(EXIT_FAILURE);
    }
    flat_ = Tensor::zeros({numel()}, dtype);
    std::vector<size_t> starts;                                         // where each tensor starts, in values
    size_t start = 0;
    for (const Shape& shape : tensor_shapes_) {
        size_t size = numel_of(shape);                                  // this tensor's number of values
        tensors_.push_back(flat_.narrow(0, start, size).view(shape));
        starts.push_back(start);
        start += size;
    }
    for (const Group& group : groups_) {
        size_t group_numel = 0;                                         // its tensors' values, added up
        for (int k = group.first_tensor_index; k < group.first_tensor_index + group.num_tensors; k++) {
            group_numel += numel_of(tensor_shapes_[k]);
        }
        group_views_.push_back(flat_.narrow(0, starts[group.first_tensor_index], group_numel));
    }
    allocated_ = true;
}

/// Makes a new buffer with the same tensors and groups as `other`, in memory of its own,
/// all zeros: a buffer of gradients laid out like a buffer of parameters.
inline TensorBuffer TensorBuffer::zeros_like(const TensorBuffer& other) {
    TensorBuffer buffer;
    buffer.tensor_shapes_ = other.tensor_shapes_;
    buffer.groups_ = other.groups_;
    buffer.allocate(other.flat_.dtype());
    return buffer;
}

// ── reading ───────────────────────────────────────────────────────────────────

/// Gives tensor k, with its own shape, to read or write. It shares the buffer's memory:
/// writing into tensor(k) writes into flat().
inline const Tensor& TensorBuffer::tensor(int k) const {
    stop_if_no_tensor(k);
    stop_if_not_allocated("tensor()");
    return tensors_[k];
}

/// Gives every value of every tensor as one flat tensor, for a pass over the whole buffer
/// in one launch: flat().zero_() zeroes everything.
inline const Tensor& TensorBuffer::flat() const { return flat_; }

/// Tells how many tensors were added, by add and add_many together.
inline int TensorBuffer::num_tensors() const { return (int)tensor_shapes_.size(); }

/// Tells how many values all the tensors hold together: 8 + 12 = 20 for {2, 4} and {12}.
/// It adds up the shapes, so it is known before allocate() too.
inline size_t TensorBuffer::numel() const {
    size_t total = 0;
    for (const Shape& shape : tensor_shapes_) total += numel_of(shape);
    return total;
}

// ── groups ────────────────────────────────────────────────────────────────────

/// Gives every value of group g as one flat tensor, for a pass over the whole group in one
/// launch, as AdamW's step over all its parameters. group(g).numel() is the group's values
/// added up. It shares the buffer's memory, as tensor(k) does.
inline const Tensor& TensorBuffer::group(int g) const {
    stop_if_no_group(g);
    stop_if_not_allocated("group()");
    return group_views_[g];
}

/// Gives tensor k of group g, counting from the group's start, so the caller needs no
/// index arithmetic: with groups of K, tensor(1, 0) is tensor(K). Stops if k is past the
/// group's end, which would be the next group's tensor.
inline const Tensor& TensorBuffer::tensor(int g, int k) const {
    stop_if_no_group(g);
    if (k < 0 || k >= groups_[g].num_tensors) {
        fprintf(stderr, "error: no tensor %d in group %d (it has %d)\n", k, g, groups_[g].num_tensors);
        exit(EXIT_FAILURE);
    }
    return tensor(groups_[g].first_tensor_index + k);
}

/// Tells how many tensors group g has: the k in tensor(g, k) goes from 0 to this - 1.
inline int TensorBuffer::group_size(int g) const {
    stop_if_no_group(g);
    return groups_[g].num_tensors;
}

/// Tells how many groups were added: one per add_many.
inline int TensorBuffer::num_groups() const { return (int)groups_.size(); }

// ── private helpers ───────────────────────────────────────────────────────────

/// Tells the index the next tensor added will get. Indices start at 0, so with N tensors
/// added (0 .. N-1) the next one gets N.
inline int TensorBuffer::next_tensor_index() const { return num_tensors(); }

/// Tells the index the next group added will get: with G groups added (0 .. G-1), G.
inline int TensorBuffer::next_group_index() const { return num_groups(); }

/// Stops the program with an error if there is no tensor k, before a wrong index reads
/// memory that is not a tensor.
inline void TensorBuffer::stop_if_no_tensor(int k) const {
    if (k < 0 || k >= num_tensors()) {
        fprintf(stderr, "error: no tensor %d (the buffer has %d)\n", k, num_tensors());
        exit(EXIT_FAILURE);
    }
}

/// Stops the program with an error if there is no group g.
inline void TensorBuffer::stop_if_no_group(int g) const {
    if (g < 0 || g >= num_groups()) {
        fprintf(stderr, "error: no group %d (the buffer has %d)\n", g, num_groups());
        exit(EXIT_FAILURE);
    }
}

/// Stops the program with an error if allocate() has not run yet, when there is no memory
/// to read. `what` names the call, for the message: "tensor() before allocate()".
inline void TensorBuffer::stop_if_not_allocated(const char* what) const {
    if (!allocated_) {
        fprintf(stderr, "error: %s before allocate()\n", what);
        exit(EXIT_FAILURE);
    }
}

/// Tells how many values a tensor of this shape holds: its sizes multiplied, {2, 4} -> 8.
inline size_t TensorBuffer::numel_of(const Shape& shape) {
    size_t numel = 1;
    for (size_t size : shape) numel *= size;
    return numel;
}

#endif // TENSOR_BUFFER_CUH
