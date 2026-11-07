// optimizer.cuh — the optimizers. Optimizer holds what every algorithm needs (params, grads, grad_norm, the
// step count); AdamW adds its m and v and its update. Everything is in one TensorBuffer, grad_norm last (1
// value: anything after it would be off 16 bytes):
//     float model:      buffer         [ params | grads | m | v | grad_norm ]   float: adamw.cuh
//     bfloat16 model:   model_buffer   [ params | grads | m | v ]               bfloat16: adamw_bf16.cuh
//                       buffer         [ master params | grad_norm ]            float
//     ... with 8-bit m and v (AdamWConfig::moments_8bit): model_buffer [ params | grads ], and m and sqrt(v) in
//     tensors of their own, 1 byte a value and a float scale a warp: adamw_8bit.cuh
// On several GPUs (ZeRO-2, nanochat's way) each GPU updates only its slice of the flat params, and keeps the
// master, m and v for that slice only: 124,475,904 / 8 = 15,559,488 values each. distributed.cuh moves the grads
// and params between them.
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <algorithm>
#include "tensor_buffer.cuh"
#include "kernels/adamw.cuh"
#include "kernels/adamw_bf16.cuh"
#include "kernels/adamw_8bit.cuh"

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
    std::vector<Shape> slice_shapes() const;                            // the shapes of a state like m
    void stop_unless_one_gpu(const char* what) const;                   // for the per-tensor views of the slice

    TensorBuffer buffer_;                // float: the params (the master in bfloat16), grads in float, ...
    TensorBuffer model_buffer_;          // bfloat16 only: group 0 the model's params, group 1 their grads
    int params_group_index_;             // buffer_'s group of the params
    int grads_group_index_;              // ...of the grads, in float; -1 in bfloat16
    int grad_norm_group_index_;          // ...of grad_norm

private:
    DType dtype_;
    std::vector<Shape> param_shapes_;
    int rank_;
    int world_size_;
    size_t slice_start_;
    size_t slice_size_;
    int steps_taken_ = 0;
};

// ══ AdamW ═════════════════════════════════════════════════════════════════════
// One launch per step: adamw_update (float model) or adamw_update_bf16 (bfloat16 model, m and v bfloat16).
// The matrices (2-D or more) come first: the first decayed_count values get weight decay.

/// AdamW's settings. weight_decay is for the matrices only.
struct AdamWConfig {
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    float max_norm = 0.0f;               // clip the gradients to this norm first; 0: no clipping
    bool moments_8bit = false;           // bfloat16 model only: m and sqrt(v) in 8 bits (adamw_8bit.cuh)
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
    const Tensor& m(int k) const;                                       // AdamW's m for parameter k: the model's dtype
    const Tensor& v(int k) const;                                       // ...and v
    const Tensor& m_slice() const;                                      // m for this GPU's slice, flat
    const Tensor& v_slice() const;                                      // ...and v
    size_t decayed_count() const;                                       // the matrices' values, first in params

protected:
    void update(int t) override;                                        // AdamW's step t: one launch

private:
    int m_group_index_;                  // moments_buffer()'s group of m; -1 with 8-bit m and v
    int v_group_index_;                  // ...of v
    Tensor m_bytes_;                     // 8-bit only: m / scale, FP8_PER_INT bytes to an int
    Tensor m_scales_;                    // ...one float scale per ADAMW_VALUES_PER_SCALE values
    Tensor sqrt_v_bytes_;                // ...sqrt(v) / scale, the same
    Tensor sqrt_v_scales_;
    const TensorBuffer& moments_buffer() const;                         // where m and v are: the model's dtype
    void stop_if_8bit(const char* what) const;                          // m and v as tensors: not in 8 bits
    size_t decayed_count_ = 0;
    AdamWConfig config_;
};

// ══ the code, in the order of the lists above ═════════════════════════════════

// ── Optimizer: a step ─────────────────────────────────────────────────────────

/// Takes one step with the gradients in grad(k): counts it, then runs the algorithm's update.
inline void Optimizer::step() {
    steps_taken_++;
    update(steps_taken_);
}

/// Copies the model's params into the float master (bfloat16 only), once, after filling param(k) and before
/// the first step. Exact: every bfloat16 is a float.
inline void Optimizer::copy_params_to_master() {
    if (dtype_ == DType::BFloat16)
        buffer_.group(params_group_index_).copy_(model_buffer_.group(0).narrow(0, slice_start_, slice_size_));
}

// ── Optimizer: reading ────────────────────────────────────────────────────────

/// Gives parameter tensor k, in the model's dtype.
///
///   k   0 .. num_params() - 1
inline const Tensor& Optimizer::param(int k) const {
    return dtype_ == DType::BFloat16 ? model_buffer_.tensor(0, k) : buffer_.tensor(params_group_index_, k);
}

/// Gives the gradient of parameter k, in the model's dtype.
///
///   k   0 .. num_params() - 1
inline const Tensor& Optimizer::grad(int k) const {
    return dtype_ == DType::BFloat16 ? model_buffer_.tensor(1, k) : buffer_.tensor(grads_group_index_, k);
}

/// Gives the float copy of parameter k the update changes: param(k) itself in float. One GPU only in bfloat16.
///
///   k   0 .. num_params() - 1
inline const Tensor& Optimizer::master_param(int k) const {
    if (dtype_ == DType::BFloat16) stop_unless_one_gpu("master_param(k)");
    return buffer_.tensor(params_group_index_, k);
}

/// Gives every param(k) as one flat tensor.
inline const Tensor& Optimizer::all_params() const {
    return dtype_ == DType::BFloat16 ? model_buffer_.group(0) : buffer_.group(params_group_index_);
}

/// Gives every grad(k) as one flat tensor: one launch zeroes them all.
inline const Tensor& Optimizer::all_grads() const {
    return dtype_ == DType::BFloat16 ? model_buffer_.group(1) : buffer_.group(grads_group_index_);
}

/// Gives the 1-value tensor holding the norm of all the gradients.
inline const Tensor& Optimizer::grad_norm() const { return buffer_.tensor(grad_norm_group_index_, 0); }

/// Gives the whole float buffer.
inline const TensorBuffer& Optimizer::buffer() const { return buffer_; }

/// Gives the model's dtype: Float32 or BFloat16.
inline DType Optimizer::dtype() const { return dtype_; }

/// Gives how many parameter tensors there are: 148 for GPT-2 small.
inline int Optimizer::num_params() const { return (int)param_shapes_.size(); }

/// Gives how many steps were taken: 0 before the first.
inline int Optimizer::steps_taken() const { return steps_taken_; }

// ── Optimizer: this GPU's slice ───────────────────────────────────────────────

/// Gives which GPU this is: 0 .. world_size - 1.
inline int Optimizer::rank() const { return rank_; }

/// Gives how many GPUs share the update.
inline int Optimizer::world_size() const { return world_size_; }

/// Gives where this GPU's slice starts in all_params(): rank x slice_size.
inline size_t Optimizer::slice_start() const { return slice_start_; }

/// Gives how many values this GPU's slice holds: all / world_size.
inline size_t Optimizer::slice_size() const { return slice_size_; }

/// Gives this slice's float params as one flat tensor (the master in bfloat16): a checkpoint saves it.
inline Tensor Optimizer::master_slice() const {
    if (dtype_ == DType::BFloat16) return buffer_.group(params_group_index_);
    return all_params().narrow(0, slice_start_, slice_size_);
}

// ── Optimizer: back from a checkpoint ─────────────────────────────────────────

/// Sets the step count a checkpoint was saved at: AdamW's bias corrections depend on it.
///
///   steps_taken   the next step is this + 1
inline void Optimizer::set_steps_taken(int steps_taken) { steps_taken_ = steps_taken; }

// ── Optimizer: for the algorithms ─────────────────────────────────────────────

/// Adds params and grads (float), or the slice's master and model_buffer_'s params and grads (bfloat16). Does
/// not allocate: the algorithm adds its groups, then calls allocate().
///
///   param_shapes        the matrices first
///   dtype               the model's: Float32 or BFloat16
///   rank, world_size    this GPU of how many; the values must split into slices of a multiple of 4
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

/// Adds grad_norm last, then allocates the float buffer (and the bfloat16 one), all zeros.
inline void Optimizer::allocate() {
    grad_norm_group_index_ = buffer_.add_many({Shape{1}});
    buffer_.allocate();
    if (dtype_ == DType::BFloat16) model_buffer_.allocate(DType::BFloat16);
}

/// Gives the shapes of a state kept for this slice (master, m, v): the params' shapes on one GPU, one flat
/// tensor of slice_size values on several.
inline std::vector<Shape> Optimizer::slice_shapes() const {
    if (world_size_ == 1) return param_shapes_;
    return {Shape{slice_size_}};
}

/// Stops with a message on several GPUs, where a per-tensor view of a state does not exist.
///
///   what   the call, for the message
inline void Optimizer::stop_unless_one_gpu(const char* what) const {
    if (world_size_ != 1) {
        fprintf(stderr, "error: %s is for one GPU; on %d, each keeps one flat slice\n", what, world_size_);
        exit(EXIT_FAILURE);
    }
}

// ── AdamW: making it ──────────────────────────────────────────────────────────

/// Makes an AdamW: Optimizer's groups, then m and v for this slice in the model's dtype (the float buffer, or
/// the bfloat16 one), or in 8 bits in tensors of their own (config.moments_8bit, bfloat16 only), then grad_norm;
/// all 0. Works out decayed_count; stops if a matrix comes after a 1-D tensor.
///
///   param_shapes        the matrices first
///   config              the settings
///   dtype               the model's: Float32 or BFloat16
///   rank, world_size    this GPU of how many
inline AdamW::AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config, DType dtype, int rank,
                    int world_size)
    : Optimizer(param_shapes, dtype, rank, world_size), config_(config) {
    // 1. m and v: in the model's dtype in a buffer, or in 8 bits with a scale a warp
    if (config.moments_8bit) {
        if (dtype != DType::BFloat16) {
            fprintf(stderr, "error: 8-bit m and v are for a bfloat16 model\n");
            exit(EXIT_FAILURE);
        }
        size_t bytes_as_ints = slice_size() / FP8_PER_INT;                  // whole: slice_size is a multiple of 4
        size_t scales = CEIL_DIV(slice_size(), (size_t)ADAMW_VALUES_PER_SCALE);
        m_bytes_ = Tensor::zeros({bytes_as_ints}, DType::Int32);
        m_scales_ = Tensor::zeros({scales});
        sqrt_v_bytes_ = Tensor::zeros({bytes_as_ints}, DType::Int32);
        sqrt_v_scales_ = Tensor::zeros({scales});
        m_group_index_ = v_group_index_ = -1;
    } else {
        TensorBuffer& moments = dtype == DType::BFloat16 ? model_buffer_ : buffer_;
        m_group_index_ = moments.add_many(slice_shapes());
        v_group_index_ = moments.add_many(slice_shapes());
    }
    allocate();

    // 2. the matrices: they get weight decay, and come first
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

/// Sets the learning rate from the next step on: the schedule calls it before each step.
///
///   learning_rate   the new one
inline void AdamW::set_learning_rate(float learning_rate) { config_.learning_rate = learning_rate; }

/// Gives the learning rate the next step uses.
inline float AdamW::learning_rate() const { return config_.learning_rate; }

// ── AdamW: reading ────────────────────────────────────────────────────────────

/// Gives m for parameter k: the running average of its gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::m(int k) const {
    stop_if_8bit("m(k)");
    stop_unless_one_gpu("m(k)");
    return moments_buffer().tensor(m_group_index_, k);
}

/// Gives v for parameter k: the running average of its squared gradients. One GPU only.
///
///   k   0 .. num_params() - 1
inline const Tensor& AdamW::v(int k) const {
    stop_if_8bit("v(k)");
    stop_unless_one_gpu("v(k)");
    return moments_buffer().tensor(v_group_index_, k);
}

/// Gives m for this slice as one flat tensor: a checkpoint saves it. Not with 8-bit m and v.
inline const Tensor& AdamW::m_slice() const {
    stop_if_8bit("m_slice()");
    return moments_buffer().group(m_group_index_);
}

/// Gives v for this slice as one flat tensor: a checkpoint saves it. Not with 8-bit m and v.
inline const Tensor& AdamW::v_slice() const {
    stop_if_8bit("v_slice()");
    return moments_buffer().group(v_group_index_);
}

/// Gives how many values get weight decay: the matrices', first in params.
inline size_t AdamW::decayed_count() const { return decayed_count_; }

/// Gives the buffer m and v are in: the bfloat16 one for a bfloat16 model, else the float one.
inline const TensorBuffer& AdamW::moments_buffer() const {
    return dtype() == DType::BFloat16 ? model_buffer_ : buffer_;
}

/// Stops with a message when m and v are in 8 bits: there is no float or bfloat16 tensor of them to give
/// (checkpoints do not save 8-bit m and v yet).
///
///   what   the call, for the message
inline void AdamW::stop_if_8bit(const char* what) const {
    if (config_.moments_8bit) {
        fprintf(stderr, "error: %s is not there with 8-bit m and v\n", what);
        exit(EXIT_FAILURE);
    }
}

// ── AdamW: the update ─────────────────────────────────────────────────────────

/// Does AdamW's step t on this slice with one launch of adamw_update, adamw_update_bf16 or adamw_update_8bit.
/// grad_norm is the norm of all the gradients, so every GPU clips alike. The slice's decayed count is its values
/// before decayed_count.
///
///   t   the step number: 1, 2, 3, ...
inline void AdamW::update(int t) {
    size_t start = slice_start();
    size_t params_in_slice = slice_size();
    size_t decayed = decayed_count_ > start ? std::min(decayed_count_ - start, params_in_slice) : 0;
    if (config_.moments_8bit) {
        adamw_update_8bit(all_params().data_ptr<nv_bfloat16>() + start, all_grads().data_ptr<nv_bfloat16>() + start,
                          (__nv_fp8_e4m3*)m_bytes_.data_ptr<int>(), m_scales_.data(),
                          (__nv_fp8_e4m3*)sqrt_v_bytes_.data_ptr<int>(), sqrt_v_scales_.data(), params_in_slice,
                          decayed, grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                          config_.beta2, config_.eps, config_.weight_decay, t, buffer_.group(params_group_index_).data());
        return;
    }
    const Tensor& all_m = m_slice();
    const Tensor& all_v = v_slice();
    if (dtype() == DType::BFloat16)
        adamw_update_bf16(all_params().data_ptr<nv_bfloat16>() + start, all_grads().data_ptr<nv_bfloat16>() + start,
                          all_m.data_ptr<nv_bfloat16>(), all_v.data_ptr<nv_bfloat16>(), params_in_slice, decayed,
                          grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1, config_.beta2,
                          config_.eps, config_.weight_decay, t, buffer_.group(params_group_index_).data());
    else
        adamw_update(all_params().data() + start, all_grads().data() + start, all_m.data(), all_v.data(), params_in_slice,
                     decayed, grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                     config_.beta2, config_.eps, config_.weight_decay, t);
}

#endif // OPTIMIZER_CUH
