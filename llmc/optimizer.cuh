// optimizer.cuh — the optimizers the trainer can use. Optimizer is the main class: what
// every algorithm needs, the parameters, their gradients, the norm of all the gradients,
// and the step count. Each algorithm is a class made from it (AdamW below; another one,
// SGD or Lion, would be one more) and adds only its own state and its own update.
//
// Everything lives in ONE TensorBuffer, in groups: Optimizer's params and grads, then the
// algorithm's own groups, then grad_norm, last:
//
//     buffer   [ params: K | grads: K | AdamW's m: K | v: K | grad_norm: 1 value ]
//
// grad_norm goes last because it is 1 value: any group after it would start 1 value off
// a multiple of 4, and adamw_update's float4 loads need every group on 16 bytes.
//
// For a model in bfloat16, the params and grads the model uses are bfloat16, in a second
// buffer, and the float buffer keeps a float "master" copy of the params instead of the
// grads:
//
//     model_buffer   [ params: K | grads: K ]                                bfloat16: the model's
//     buffer         [ master params: K | AdamW's m: K | v: K | grad_norm ]  float
//
// Each step's one launch reads the master, writes it back and stores it rounded into the
// model's params (kernels/adamw.cuh): no other pass, and as much memory as in float.
//
// On several GPUs, nanochat's way (ZeRO-2): every GPU keeps the whole model and all its
// grads, but updates only its own slice of the flat params, rank r of world_size:
//
//     all params, flat   [ slice 0 | slice 1 | ... | slice 7 ]   124,475,904 / 8 = 15,559,488 each
//
// The master, m and v are kept for that slice only: 1 / world_size of the memory and of the
// work. The trainer averages the grads into each slice before the step (a reduce-scatter)
// and sends each slice to every GPU after it (an all-gather): distributed.cuh. The
// weight-decay border (the matrices first) falls where it falls, inside a slice or not.
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <algorithm>
#include "tensor_buffer.cuh"
#include "kernels/adamw.cuh"

// ══ Optimizer: what every algorithm needs ═════════════════════════════════════

class Optimizer {
public:
    virtual ~Optimizer() = default;

    // ── a step ──────────────────────────────────────────────────────────────────
    void step();                                                        // step t + 1: the algorithm's update(t)
    void copy_params_to_master();                                       // bfloat16: once, after filling param(k)

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& param(int k) const;                                   // parameter tensor k, as the model reads it
    const Tensor& grad(int k) const;                                    // its gradient, as backward writes it
    const Tensor& master_param(int k) const;                            // the float copy the update changes
    const Tensor& all_params() const;                                   // every param(k), one flat tensor
    const Tensor& all_grads() const;                                    // every grad(k), one flat tensor
    const Tensor& grad_norm() const;                                    // the norm of all the gradients: 1 value
    const TensorBuffer& buffer() const;                                 // the float buffer, all of it in one
    DType dtype() const;                                                // the model's: Float32 or BFloat16
    int num_params() const;                                             // K: how many parameter tensors
    int steps_taken() const;                                            // t: 0 before the first step

    // ── this GPU's slice ────────────────────────────────────────────────────────
    int rank() const;                                                   // this GPU: 0 .. world_size - 1
    int world_size() const;                                             // how many GPUs share the update
    size_t slice_start() const;                                         // where its slice starts in all_params()
    size_t slice_size() const;                                          // how many values it updates
    Tensor master_slice() const;                                        // its float params, flat: for checkpoints

    // ── back from a checkpoint ──────────────────────────────────────────────────
    void set_steps_taken(int steps_taken);                              // the step count it was saved at

protected:
    // ── for the algorithms (the classes made from Optimizer) ────────────────────
    Optimizer(const std::vector<Shape>& param_shapes, DType dtype,     // adds params, grads
              int rank, int world_size);
    virtual void update(int t) = 0;                                     // the algorithm's step t, from the grads
    void allocate();                                                    // adds grad_norm last, then allocates
    std::vector<Shape> slice_shapes() const;                            // the shapes of a state like m: see code
    void stop_unless_one_gpu(const char* what) const;                   // for the per-tensor views of the slice

    TensorBuffer buffer_;                // float: the params (the master in bfloat16), grads in float, ...
    TensorBuffer model_buffer_;          // bfloat16 only: group 0 the model's params, group 1 their grads
    int params_group_index_;             // buffer_'s group of the params: buffer_.group(params_group_index_)
    int grads_group_index_;              // ...of the grads, in float; -1 in bfloat16 (they are in model_buffer_)
    int grad_norm_group_index_;          // ...of grad_norm

private:
    DType dtype_;
    std::vector<Shape> param_shapes_;
    int rank_, world_size_;
    size_t slice_start_, slice_size_;
    int steps_taken_ = 0;
};

// ══ AdamW ═════════════════════════════════════════════════════════════════════
// Each step updates every parameter with ONE launch of adamw_update (kernels/adamw.cuh),
// over whole groups.
//
// The matrices (2-D or more) must come first in param_shapes: they get weight decay, and
// adamw_update decays exactly the first decayed_count values of each group. AdamW works
// that count out from the shapes, and stops if a matrix comes after a 1-D tensor:
//
//     params    [ wte | wpe | qkvw | ... matrices ... | ln1w | ln1b | ... the rest ... ]
//                                                      ↑ decayed_count

/// AdamW's settings. weight_decay is for the matrices only.
struct AdamWConfig {
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    float max_norm = 0.0f;               // clip the gradients to this norm first; 0: no clipping
};

class AdamW : public Optimizer {
public:
    // ── making it ───────────────────────────────────────────────────────────────
    AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config,  // the matrices first
          DType dtype = DType::Float32, int rank = 0, int world_size = 1);

    // ── the learning rate, for a schedule ───────────────────────────────────────
    void set_learning_rate(float learning_rate);                        // used from the next step on
    float learning_rate() const;

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& m(int k) const;                                       // AdamW's m for parameter k
    const Tensor& v(int k) const;                                       // ...and v
    const Tensor& m_slice() const;                                      // m for this GPU's slice, flat
    const Tensor& v_slice() const;                                      // ...and v
    size_t decayed_count() const;                                       // the matrices' values, first in params

protected:
    void update(int t) override;                                        // AdamW's step t: one launch

private:
    int m_group_index_;                  // the buffer's group of m
    int v_group_index_;                  // ...of v
    size_t decayed_count_ = 0;
    AdamWConfig config_;
};

// ══ the code, in the order of the lists above ═════════════════════════════════

// ── Optimizer: a step ─────────────────────────────────────────────────────────

/// Takes one training step with the gradients in grad(k): adds 1 to the step count, then
/// runs the algorithm's update for that step. Here, not in each algorithm, so none of
/// them can forget to count: the bias corrections of AdamW depend on t.
inline void Optimizer::step() {
    steps_taken_++;
    update(steps_taken_);
}

/// Copies the model's params into the float master, once, after the trainer has filled param(k) (with GPT-2's
/// starting values) and before the first step: the update starts from the master. Exact, as every bfloat16 is a
/// float. Only this GPU's slice, the master it keeps. In float the params are the master: nothing to do.
inline void Optimizer::copy_params_to_master() {
    if (dtype_ == DType::BFloat16)
        buffer_.group(params_group_index_).copy_(model_buffer_.group(0).narrow(0, slice_start_, slice_size_));
}

// ── Optimizer: reading ────────────────────────────────────────────────────────

/// Gives parameter tensor k, for the model to read in forward and the update to change: bfloat16 in
/// model_buffer_ for a bfloat16 model, else float in buffer_.
inline const Tensor& Optimizer::param(int k) const {
    return dtype_ == DType::BFloat16 ? model_buffer_.tensor(0, k) : buffer_.tensor(params_group_index_, k);
}

/// Gives the gradient of parameter k, for backward to fill: the same dtype as param(k).
inline const Tensor& Optimizer::grad(int k) const {
    return dtype_ == DType::BFloat16 ? model_buffer_.tensor(1, k) : buffer_.tensor(grads_group_index_, k);
}

/// Gives the float copy of parameter k that the update reads and writes: param(k) itself in float. In
/// bfloat16, one GPU only: on several, the master is kept as one flat slice, which cuts across the tensors.
inline const Tensor& Optimizer::master_param(int k) const {
    if (dtype_ == DType::BFloat16) stop_unless_one_gpu("master_param(k)");
    return buffer_.tensor(params_group_index_, k);
}

/// Gives every param(k) as one flat tensor, one after the other.
inline const Tensor& Optimizer::all_params() const {
    return dtype_ == DType::BFloat16 ? model_buffer_.group(0) : buffer_.group(params_group_index_);
}

/// Gives every grad(k) as one flat tensor, one after the other: one launch zeroes them all, and GradNorm takes
/// the norm of all of them.
inline const Tensor& Optimizer::all_grads() const {
    return dtype_ == DType::BFloat16 ? model_buffer_.group(1) : buffer_.group(grads_group_index_);
}

/// Gives the 1-value tensor that holds the norm of all the gradients, for the norm kernel
/// to write and the update to read when it clips.
inline const Tensor& Optimizer::grad_norm() const { return buffer_.tensor(grad_norm_group_index_, 0); }

/// Gives the whole float buffer, to look at its groups and its numel.
inline const TensorBuffer& Optimizer::buffer() const { return buffer_; }

/// Tells the dtype of the model's params and grads: Float32 or BFloat16.
inline DType Optimizer::dtype() const { return dtype_; }

/// Tells how many parameter tensors there are: K, 148 for GPT-2 small.
inline int Optimizer::num_params() const { return (int)param_shapes_.size(); }

/// Tells how many steps were taken: 0 before the first.
inline int Optimizer::steps_taken() const { return steps_taken_; }

// ── Optimizer: this GPU's slice ───────────────────────────────────────────────

/// Tells which GPU this optimizer is on: 0 .. world_size - 1.
inline int Optimizer::rank() const { return rank_; }

/// Tells how many GPUs share the update, each its own slice.
inline int Optimizer::world_size() const { return world_size_; }

/// Tells where this GPU's slice starts in all_params() and all_grads(): rank x slice_size.
inline size_t Optimizer::slice_start() const { return slice_start_; }

/// Tells how many values this GPU's slice holds: all of them / world_size; all of them on one GPU.
inline size_t Optimizer::slice_size() const { return slice_size_; }

/// Gives the float params of this GPU's slice as one flat tensor, the values the update reads and writes: the
/// master in bfloat16, a view of the params themselves in float. A checkpoint saves it, and writes it back.
inline Tensor Optimizer::master_slice() const {
    if (dtype_ == DType::BFloat16) return buffer_.group(params_group_index_);
    return all_params().narrow(0, slice_start_, slice_size_);
}

// ── Optimizer: back from a checkpoint ─────────────────────────────────────────

/// Sets the step count to the one a checkpoint was saved at, so the next step is steps_taken + 1: AdamW's bias
/// corrections depend on it.
inline void Optimizer::set_steps_taken(int steps_taken) { steps_taken_ = steps_taken; }

// ── Optimizer: for the algorithms ─────────────────────────────────────────────

/// Adds what every algorithm needs: in float, params and grads to the buffer, laid out the
/// same; in bfloat16, the float master params of this GPU's slice to the buffer, and the
/// model's params and grads to model_buffer_. Does not allocate: the algorithm adds its own
/// groups first, then calls allocate(), so the float buffer is one allocation. The values
/// must split into world_size slices of a multiple of 4 (float4 loads): GPT-2 small's
/// 124,475,904 into 8 slices of 15,559,488.
inline Optimizer::Optimizer(const std::vector<Shape>& param_shapes, DType dtype, int rank, int world_size)
    : dtype_(dtype), param_shapes_(param_shapes), rank_(rank), world_size_(world_size) {
    if (dtype != DType::Float32 && dtype != DType::BFloat16) {
        fprintf(stderr, "error: an optimizer's model is float or bfloat16\n");
        exit(EXIT_FAILURE);
    }
    if (world_size < 1 || rank < 0 || rank >= world_size) {
        fprintf(stderr, "error: an optimizer needs 0 <= rank < world_size; got rank %d of %d\n", rank, world_size);
        exit(EXIT_FAILURE);
    }
    size_t total = 0;
    for (const Shape& shape : param_shapes) {
        size_t numel = 1;
        for (size_t size : shape) numel *= size;
        total += numel;
    }
    if (total % (4 * (size_t)world_size) != 0) {
        fprintf(stderr, "error: %zu parameter values do not split into %d slices of a multiple of 4\n", total,
                world_size);
        exit(EXIT_FAILURE);
    }
    slice_size_ = total / world_size;
    slice_start_ = rank * slice_size_;
    params_group_index_ = buffer_.add_many(dtype == DType::Float32 ? param_shapes : slice_shapes());
    grads_group_index_ = -1;
    if (dtype == DType::Float32) {
        grads_group_index_ = buffer_.add_many(param_shapes);
    } else {
        model_buffer_.add_many(param_shapes);                   // group 0: the model's params
        model_buffer_.add_many(param_shapes);                   // group 1: their grads
    }
}

/// Adds grad_norm, then gets the one GPU allocation for every group, all zeros, and in
/// bfloat16 the model's. The algorithm's constructor calls it last, after adding its own
/// groups. grad_norm is 1 value, so it goes after everything: a group behind it would start
/// off 16 bytes.
inline void Optimizer::allocate() {
    grad_norm_group_index_ = buffer_.add_many({Shape{1}});    // a group of one tensor of 1 value
    buffer_.allocate();
    if (dtype_ == DType::BFloat16) model_buffer_.allocate(DType::BFloat16);
}

/// Gives the shapes for a state kept for this GPU's slice only (the master, AdamW's m and v): on one GPU the
/// params' own shapes, so m(k) and master_param(k) are views of tensor k; on several, one flat tensor of
/// slice_size values, which cuts across the tensors.
inline std::vector<Shape> Optimizer::slice_shapes() const {
    if (world_size_ == 1) return param_shapes_;
    return {Shape{slice_size_}};
}

/// Stops with an error if the update is shared by several GPUs: `what` is a per-tensor view of a state that
/// is then kept as one flat slice.
inline void Optimizer::stop_unless_one_gpu(const char* what) const {
    if (world_size_ != 1) {
        fprintf(stderr, "error: %s is for one GPU; on %d, each keeps one flat slice\n", what, world_size_);
        exit(EXIT_FAILURE);
    }
}

// ── AdamW: making it ──────────────────────────────────────────────────────────

/// Makes an AdamW for parameters of these shapes and a model of this dtype, for GPU rank of
/// world_size: Optimizer's params and grads (or master params), then m and v for this GPU's
/// slice, then grad_norm, all in one allocation, all 0. Works out decayed_count, and stops if
/// a matrix comes after a 1-D tensor.
inline AdamW::AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config, DType dtype, int rank,
                    int world_size)
    : Optimizer(param_shapes, dtype, rank, world_size), config_(config) {
    m_group_index_ = buffer_.add_many(slice_shapes());
    v_group_index_ = buffer_.add_many(slice_shapes());
    allocate();

    // The matrices: their values get weight decay, and they must all come first.
    bool past_the_matrices = false;
    for (int k = 0; k < num_params(); k++) {
        if (param(k).dim() < 2) {
            past_the_matrices = true;
            continue;
        }
        if (past_the_matrices) {
            fprintf(stderr, "error: parameter %d is a matrix after a 1-D tensor; give the matrices first\n", k);
            exit(EXIT_FAILURE);
        }
        decayed_count_ += param(k).numel();
    }
}

// ── AdamW: the learning rate ──────────────────────────────────────────────────

/// Sets the learning rate the next steps use: the trainer's schedule calls it before each step.
inline void AdamW::set_learning_rate(float learning_rate) { config_.learning_rate = learning_rate; }

/// Tells the learning rate the next step uses.
inline float AdamW::learning_rate() const { return config_.learning_rate; }

// ── AdamW: reading ────────────────────────────────────────────────────────────

/// Gives AdamW's m for parameter k: the running average of its gradients. One GPU only.
inline const Tensor& AdamW::m(int k) const {
    stop_unless_one_gpu("m(k)");
    return buffer_.tensor(m_group_index_, k);
}

/// Gives AdamW's v for parameter k: the running average of its squared gradients. One GPU only.
inline const Tensor& AdamW::v(int k) const {
    stop_unless_one_gpu("v(k)");
    return buffer_.tensor(v_group_index_, k);
}

/// Gives m for this GPU's slice as one flat tensor (all of m on one GPU): a checkpoint saves it.
inline const Tensor& AdamW::m_slice() const { return buffer_.group(m_group_index_); }

/// Gives v for this GPU's slice as one flat tensor (all of v on one GPU): a checkpoint saves it.
inline const Tensor& AdamW::v_slice() const { return buffer_.group(v_group_index_); }

/// Tells how many values get weight decay: the matrices', first in params.
inline size_t AdamW::decayed_count() const { return decayed_count_; }

// ── AdamW: the update ─────────────────────────────────────────────────────────

/// Does AdamW's step t on this GPU's slice of the parameters (all of them on one GPU) with ONE
/// launch of adamw_update, from the gradients in that slice and grad_norm (read only when
/// config.max_norm > 0: the norm of ALL the gradients, so every GPU clips alike). In bfloat16
/// the launch reads and writes the float master, and stores it rounded into the model's
/// params. Optimizer::step calls it.
///
/// The slice's own decayed count: the values of it before decayed_count. GPT-2 small on 8
/// GPUs, decayed_count 124,354,560: slices 0 .. 6 all decayed, slice 7 (from 108,916,416) its
/// first 15,438,144 of 15,559,488.
inline void AdamW::update(int t) {
    // Each group as one flat block: the slice's values, one after the other.
    const Tensor& all_m = buffer_.group(m_group_index_);
    const Tensor& all_v = buffer_.group(v_group_index_);
    size_t start = slice_start(), n = slice_size();
    size_t decayed = decayed_count_ > start ? std::min(decayed_count_ - start, n) : 0;
    if (dtype() == DType::BFloat16)
        adamw_update(all_params().data_ptr<nv_bfloat16>() + start, all_grads().data_ptr<nv_bfloat16>() + start,
                     all_m.data(), all_v.data(), n, decayed, grad_norm().data(), config_.max_norm,
                     config_.learning_rate, config_.beta1, config_.beta2, config_.eps, config_.weight_decay, t,
                     buffer_.group(params_group_index_).data());
    else
        adamw_update(all_params().data() + start, all_grads().data() + start, all_m.data(), all_v.data(), n,
                     decayed, grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                     config_.beta2, config_.eps, config_.weight_decay, t);
}

#endif // OPTIMIZER_CUH
