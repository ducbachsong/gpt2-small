// tensor_buffer.cuh — many tensors in one GPU allocation, one after the other, in groups. One cudaMalloc
// instead of one per tensor, and a pass over everything (the norm, zeroing) is one launch over flat().
// Matrices go first in each group: the values up to the border then get weight decay.
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
    /// A group, added by one add_many.
    struct Group {
        int first_tensor_index;          // its first tensor's index in the buffer
        int num_tensors;                 // how many tensors it has
    };
    std::vector<Shape> tensor_shapes_;   // each tensor's shape, in the order added
    std::vector<Group> groups_;          // each group, in the order added
    std::vector<Tensor> tensors_;        // tensor k: a view into flat_
    std::vector<Tensor> group_views_;    // group g: a flat view into flat_
    Tensor flat_;                        // the one allocation
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

/// Adds one tensor and gives its index k. Only before allocate().
///
///   shape   {768, 3072} for a matrix, {768} for a bias
inline int TensorBuffer::add(const Shape& shape) {
    int tensor_index = next_tensor_index();
    if (allocated_) {
        fprintf(stderr, "error: tensor %d added after allocate()\n", tensor_index);
        exit(EXIT_FAILURE);
    }
    tensor_shapes_.push_back(shape);
    return tensor_index;
}

/// Adds tensors one after the other as one group, and gives the group's index g.
///
///   tensor_shapes   at least one shape
inline int TensorBuffer::add_many(const std::vector<Shape>& tensor_shapes) {
    if (tensor_shapes.empty()) {
        fprintf(stderr, "error: add_many of no shapes: a group needs at least one tensor\n");
        exit(EXIT_FAILURE);
    }
    int group_index = next_group_index();
    Group group;
    group.first_tensor_index = next_tensor_index();
    group.num_tensors = (int)tensor_shapes.size();
    for (const Shape& shape : tensor_shapes) add(shape);
    groups_.push_back(group);
    return group_index;
}

/// Gets one allocation for every tensor added, all zeros; each tensor and group becomes a view of its part.
/// float4 kernels need every tensor's numel a multiple of 4, as all of GPT-2's are.
///
///   dtype   of every value
inline void TensorBuffer::allocate(DType dtype) {
    if (allocated_) {
        fprintf(stderr, "error: allocate() called twice\n");
        exit(EXIT_FAILURE);
    }
    flat_ = Tensor::zeros({numel()}, dtype);
    std::vector<size_t> starts;                                         // where each tensor starts
    size_t start = 0;
    for (const Shape& shape : tensor_shapes_) {
        size_t size = numel_of(shape);
        tensors_.push_back(flat_.narrow(0, start, size).view(shape));
        starts.push_back(start);
        start += size;
    }
    for (const Group& group : groups_) {
        size_t group_numel = 0;
        for (int k = group.first_tensor_index; k < group.first_tensor_index + group.num_tensors; k++) {
            group_numel += numel_of(tensor_shapes_[k]);
        }
        group_views_.push_back(flat_.narrow(0, starts[group.first_tensor_index], group_numel));
    }
    allocated_ = true;
}

/// Makes a buffer with other's tensors and groups, in memory of its own, all zeros: grads laid out as params.
///
///   other   the buffer to copy the layout of
inline TensorBuffer TensorBuffer::zeros_like(const TensorBuffer& other) {
    TensorBuffer buffer;
    buffer.tensor_shapes_ = other.tensor_shapes_;
    buffer.groups_ = other.groups_;
    buffer.allocate(other.flat_.dtype());
    return buffer;
}

// ── reading ───────────────────────────────────────────────────────────────────

/// Gives tensor k, a view sharing the buffer's memory.
///
///   k   from add
inline const Tensor& TensorBuffer::tensor(int k) const {
    stop_if_no_tensor(k);
    stop_if_not_allocated("tensor()");
    return tensors_[k];
}

/// Gives every value as one flat tensor, for one launch over the whole buffer.
inline const Tensor& TensorBuffer::flat() const { return flat_; }

/// Gives how many tensors were added.
inline int TensorBuffer::num_tensors() const { return (int)tensor_shapes_.size(); }

/// Gives how many values all the tensors hold; known before allocate() too.
inline size_t TensorBuffer::numel() const {
    size_t total = 0;
    for (const Shape& shape : tensor_shapes_) total += numel_of(shape);
    return total;
}

// ── groups ────────────────────────────────────────────────────────────────────

/// Gives every value of group g as one flat tensor, for one launch over the group.
///
///   g   from add_many
inline const Tensor& TensorBuffer::group(int g) const {
    stop_if_no_group(g);
    stop_if_not_allocated("group()");
    return group_views_[g];
}

/// Gives tensor k of group g, counting from the group's start.
///
///   g   from add_many
///   k   0 .. group_size(g) - 1
inline const Tensor& TensorBuffer::tensor(int g, int k) const {
    stop_if_no_group(g);
    if (k < 0 || k >= groups_[g].num_tensors) {
        fprintf(stderr, "error: no tensor %d in group %d (it has %d)\n", k, g, groups_[g].num_tensors);
        exit(EXIT_FAILURE);
    }
    return tensor(groups_[g].first_tensor_index + k);
}

/// Gives how many tensors group g has.
///
///   g   from add_many
inline int TensorBuffer::group_size(int g) const {
    stop_if_no_group(g);
    return groups_[g].num_tensors;
}

/// Gives how many groups were added.
inline int TensorBuffer::num_groups() const { return (int)groups_.size(); }

// ── private helpers ───────────────────────────────────────────────────────────

/// Gives the index the next tensor added will get.
inline int TensorBuffer::next_tensor_index() const { return num_tensors(); }

/// Gives the index the next group added will get.
inline int TensorBuffer::next_group_index() const { return num_groups(); }

/// Stops with a message if there is no tensor k.
///
///   k   the index asked for
inline void TensorBuffer::stop_if_no_tensor(int k) const {
    if (k < 0 || k >= num_tensors()) {
        fprintf(stderr, "error: no tensor %d (the buffer has %d)\n", k, num_tensors());
        exit(EXIT_FAILURE);
    }
}

/// Stops with a message if there is no group g.
///
///   g   the index asked for
inline void TensorBuffer::stop_if_no_group(int g) const {
    if (g < 0 || g >= num_groups()) {
        fprintf(stderr, "error: no group %d (the buffer has %d)\n", g, num_groups());
        exit(EXIT_FAILURE);
    }
}

/// Stops with a message if allocate() has not run yet.
///
///   what   the call, for the message: "tensor()"
inline void TensorBuffer::stop_if_not_allocated(const char* what) const {
    if (!allocated_) {
        fprintf(stderr, "error: %s before allocate()\n", what);
        exit(EXIT_FAILURE);
    }
}

/// Gives how many values a shape holds: {2, 4} -> 8.
///
///   shape   the sizes
inline size_t TensorBuffer::numel_of(const Shape& shape) {
    size_t numel = 1;
    for (size_t size : shape) numel *= size;
    return numel;
}

#endif // TENSOR_BUFFER_CUH
