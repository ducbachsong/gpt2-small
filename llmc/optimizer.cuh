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
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "tensor_buffer.cuh"
#include "kernels/adamw.cuh"

// ══ Optimizer: what every algorithm needs ═════════════════════════════════════

class Optimizer {
public:
    virtual ~Optimizer() = default;

    // ── a step ──────────────────────────────────────────────────────────────────
    void step();                                                        // step t + 1: the algorithm's update(t)

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& param(int k) const;                                   // parameter tensor k
    const Tensor& grad(int k) const;                                    // its gradient
    const Tensor& grad_norm() const;                                    // the norm of all the gradients: 1 value
    const TensorBuffer& buffer() const;                                 // all of it, in one
    int num_params() const;                                             // K: how many parameter tensors
    int steps_taken() const;                                            // t: 0 before the first step

protected:
    // ── for the algorithms (the classes made from Optimizer) ────────────────────
    explicit Optimizer(const std::vector<Shape>& param_shapes);         // adds params, grads
    virtual void update(int t) = 0;                                     // the algorithm's step t, from the grads
    void allocate();                                                    // adds grad_norm last, then allocates

    TensorBuffer buffer_;
    int params_group_index_;             // the buffer's group of the params: buffer_.group(params_group_index_)
    int grads_group_index_;              // ...of the grads
    int grad_norm_group_index_;          // ...of grad_norm

private:
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
    AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config);  // the matrices first

    // ── reading ─────────────────────────────────────────────────────────────────
    const Tensor& m(int k) const;                                       // AdamW's m for parameter k
    const Tensor& v(int k) const;                                       // ...and v
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

// ── Optimizer: reading ────────────────────────────────────────────────────────

/// Gives parameter tensor k, for the model to read in forward and the update to change.
inline const Tensor& Optimizer::param(int k) const { return buffer_.tensor(params_group_index_, k); }

/// Gives the gradient of parameter k, for backward to fill.
inline const Tensor& Optimizer::grad(int k) const { return buffer_.tensor(grads_group_index_, k); }

/// Gives the 1-value tensor that holds the norm of all the gradients, for the norm kernel
/// to write and the update to read when it clips.
inline const Tensor& Optimizer::grad_norm() const { return buffer_.tensor(grad_norm_group_index_, 0); }

/// Gives the whole buffer, to look at its groups and its numel.
inline const TensorBuffer& Optimizer::buffer() const { return buffer_; }

/// Tells how many parameter tensors there are: K, 148 for GPT-2 small.
inline int Optimizer::num_params() const { return buffer_.group_size(params_group_index_); }

/// Tells how many steps were taken: 0 before the first.
inline int Optimizer::steps_taken() const { return steps_taken_; }

// ── Optimizer: for the algorithms ─────────────────────────────────────────────

/// Adds what every algorithm needs to the buffer: params and grads, laid out the same.
/// Does not allocate: the algorithm adds its own groups first, then calls allocate(), so
/// everything is in one allocation.
inline Optimizer::Optimizer(const std::vector<Shape>& param_shapes) {
    params_group_index_ = buffer_.add_many(param_shapes);
    grads_group_index_ = buffer_.add_many(param_shapes);
}

/// Adds grad_norm, then gets the one GPU allocation for every group, all zeros. The
/// algorithm's constructor calls it last, after adding its own groups. grad_norm is 1
/// value, so it goes after everything: a group behind it would start off 16 bytes.
inline void Optimizer::allocate() {
    grad_norm_group_index_ = buffer_.add_many({Shape{1}});    // a group of one tensor of 1 value
    buffer_.allocate();
}

// ── AdamW: making it ──────────────────────────────────────────────────────────

/// Makes an AdamW for parameters of these shapes: Optimizer's params and grads, then m and
/// v laid out like the params, then grad_norm, all in one allocation, all 0. Works out
/// decayed_count, and stops if a matrix comes after a 1-D tensor.
inline AdamW::AdamW(const std::vector<Shape>& param_shapes, const AdamWConfig& config)
    : Optimizer(param_shapes), config_(config) {
    m_group_index_ = buffer_.add_many(param_shapes);
    v_group_index_ = buffer_.add_many(param_shapes);
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

// ── AdamW: reading ────────────────────────────────────────────────────────────

/// Gives AdamW's m for parameter k: the running average of its gradients.
inline const Tensor& AdamW::m(int k) const { return buffer_.tensor(m_group_index_, k); }

/// Gives AdamW's v for parameter k: the running average of its squared gradients.
inline const Tensor& AdamW::v(int k) const { return buffer_.tensor(v_group_index_, k); }

/// Tells how many values get weight decay: the matrices', first in params.
inline size_t AdamW::decayed_count() const { return decayed_count_; }

// ── AdamW: the update ─────────────────────────────────────────────────────────

/// Does AdamW's step t on every parameter with ONE launch of adamw_update, from the
/// gradients and grad_norm in the buffer (grad_norm read only when config.max_norm > 0).
/// Optimizer::step calls it.
inline void AdamW::update(int t) {
    // Each group as one flat block: all K tensors, one after the other.
    const Tensor& all_params = buffer_.group(params_group_index_);
    const Tensor& all_grads = buffer_.group(grads_group_index_);
    const Tensor& all_m = buffer_.group(m_group_index_);
    const Tensor& all_v = buffer_.group(v_group_index_);
    adamw_update(all_params.data(), all_grads.data(), all_m.data(), all_v.data(), all_params.numel(),
                 decayed_count_, grad_norm().data(), config_.max_norm, config_.learning_rate, config_.beta1,
                 config_.beta2, config_.eps, config_.weight_decay, t);
}

#endif // OPTIMIZER_CUH
