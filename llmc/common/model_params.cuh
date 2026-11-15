// model_params.cuh — what GPT-2 is in every number type: its sizes (GPT2Config), its parameter tensors as views of
// the AdamW's buffer (BlockParams, GPT2Params), their starting values, the mean loss, and what the three AdamWs
// share (AdamWConfig, Optimizer: this GPU's slice, the step count, the learning rate, grad_norm). The models that
// run on them, each with its own AdamW that holds its params and grads: model_fp32.cuh (all float, AdamW),
// model_bf16.cuh (all bfloat16, AdamWBf16), model_8bit.cuh (bfloat16, the products and m and v in 8 bits,
// AdamW8bit). checkpoint.cuh and training.cuh take any of the three AdamWs as an Optimizer&.
//
//     wte              {padded_vocab_size, D}      a row per word, also the head's weight ("tied")
//     wpe              {max_seq_len, D}            a row per position
//     each block       4 matrices, 8 vectors       12 D² + 13 D = 7,087,872 values for D = 768
//                      (linear_bias false: 4 matrices, 4 vectors, the LayerNorms' only: 12 D² + 4 D)
//     lnf_weight, lnf_bias   {D}                   the last LayerNorm
// Every matrix comes first (wte, wpe, each block's 4 matrices), then every vector: AdamW decays the matrices only.
#ifndef MODEL_PARAMS_CUH
#define MODEL_PARAMS_CUH

#include <algorithm>
#include <cmath>
#include <vector>
#include "tensor_buffer.cuh"

/// GPT-2's sizes. The defaults are GPT-2 small's: 124,475,904 parameters with the padded vocabulary.
struct GPT2Config {
    size_t max_seq_len = 1024;           // wpe's rows: the longest seq_len
    size_t vocab_size = 50257;           // the words the tokenizer has
    size_t padded_vocab_size = 50304;    // wte's rows: vocab_size rounded up to a multiple of 128, for the head
    size_t num_layers = 12;
    size_t num_heads = 12;
    size_t embed_dim = 768;
    bool linear_bias = true;             // the 4 linears' biases (qkv, attention_out, mlp_up, mlp_down): GPT-2 has
                                         // them; false: none (the LayerNorms keep theirs), as nanochat and Llama
    size_t attention_window = 0;         // the short layers' sliding window: a token sees itself and this many tokens
                                         // before it (a multiple of 64; model_bf16 and model_8bit, flash attention
                                         // only); 0: every layer sees the whole row, as GPT-2
};

/// Gives a layer's sliding window, in nanochat's pattern SSSL: of every 4 layers the first 3 are short
/// (config.attention_window), the 4th long, and the last layer always long. 0: long, the whole row.
///
///   config   the sizes: attention_window, num_layers
///   layer    which block: 0 .. num_layers - 1
inline size_t gpt2_attention_window(const GPT2Config& config, size_t layer) {
    bool long_val = layer % 4 == 3 || layer + 1 == config.num_layers;
    return long_val ? 0 : config.attention_window;
}

/// A block's 12 parameter tensors, or their gradients: views of the optimizer's buffer, the 4 matrices first.
/// D is embed_dim. Without linear biases (GPT2Config::linear_bias false) it has 8: the 4 linear biases are
/// Tensor(), which the products take as no bias.
struct BlockParams {
    Tensor qkv_weight;                   // {3D, D}
    Tensor attention_out_weight;         // {D, D}
    Tensor mlp_up_weight;                // {4D, D}
    Tensor mlp_down_weight;              // {D, 4D}
    Tensor ln1_weight;                   // {D}
    Tensor ln1_bias;                     // {D}
    Tensor qkv_bias;                     // {3D}
    Tensor attention_out_bias;           // {D}
    Tensor ln2_weight;                   // {D}
    Tensor ln2_bias;                     // {D}
    Tensor mlp_up_bias;                  // {4D}
    Tensor mlp_down_bias;                // {D}

    BlockParams() = default;
    explicit BlockParams(const std::vector<Tensor>& tensors);           // 12 tensors (8 with no linear bias), in shapes()' order
    static std::vector<Shape> shapes(size_t embed_dim, bool linear_bias = true);   // the 12 (or 8) shapes, matrices first
    static size_t vector_count(bool linear_bias);                       // the 1-D tensors after the 4 matrices: 8 or 4
    std::vector<Tensor> tensors() const;                                // the 12 (or 8), in the same order
    bool has_linear_bias() const;                                       // the 4 linear biases are there
};

/// The model's parameter tensors, or their gradients: views of the optimizer's buffer. Every matrix first:
///     wte, wpe, each block's 4 matrices, each block's 8 vectors, lnf_weight, lnf_bias
struct GPT2Params {
    Tensor wte;                          // {padded_vocab_size, D}: a row per word, also the head's weight
    Tensor wpe;                          // {max_seq_len, D}: a row per position
    std::vector<BlockParams> blocks;     // num_layers of them
    Tensor lnf_weight;                   // {D}: the last LayerNorm
    Tensor lnf_bias;                     // {D}: the last LayerNorm

    GPT2Params() = default;
    GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config);  // in shapes()' order
    static std::vector<Shape> shapes(const GPT2Config& config);                // 4 + 12 num_layers shapes (4 + 8 num_layers with no linear bias)
    std::vector<Tensor> tensors() const;                                       // in the same order
};

// ── starting values and the loss ──────────────────────────────────────────────
void block_init_params(const BlockParams& params, int num_layers);
void gpt2_init_params(const GPT2Params& params, const GPT2Config& config);
template <typename ACTIVATIONS>
float gpt2_mean_loss(const ACTIVATIONS& activations);

// ── the params, grads and AdamW: what the three models' AdamWs share ─────────
/// AdamW's settings, the same for all three. weight_decay is for the matrices only.
struct AdamWConfig {
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    float max_norm = 0.0f;               // clip the gradients to this norm first; 0: no clipping
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    // ── a step ──────────────────────────────────────────────────────────────────
    void step();                                                        // step t + 1: the AdamW's update(t)
    virtual void copy_params_to_master() {}                             // bfloat16: once, after filling param(k)

    // ── the learning rate, for a schedule ───────────────────────────────────────
    void set_learning_rate(float learning_rate);                        // used from the next step on
    float learning_rate() const;

    // ── reading: where each AdamW keeps them ────────────────────────────────────
    virtual const Tensor& param(int k) const = 0;                       // parameter tensor k, as the model reads it
    virtual const Tensor& grad(int k) const = 0;                        // its gradient, as backward writes it
    virtual const Tensor& all_params() const = 0;                       // every param(k), one flat tensor
    virtual const Tensor& all_grads() const = 0;                        // every grad(k), one flat tensor
    virtual Tensor master_slice() const = 0;                            // this slice's float params: for checkpoints
    virtual const Tensor& m_slice() const = 0;                          // m for this slice, flat: for checkpoints
    virtual const Tensor& v_slice() const = 0;                          // ...and v

    // ── reading: the same in all three ──────────────────────────────────────────
    const Tensor& grad_norm() const;                                    // the norm of all the gradients: 1 value
    const TensorBuffer& buffer() const;                                 // the float buffer, all of it in one
    DType dtype() const;                                                // the model's params: Float32 or BFloat16
    int num_params() const;                                             // K: how many parameter tensors
    int steps_taken() const;                                            // t: 0 before the first step
    size_t decayed_count() const;                                       // the matrices' values, first in params

    // ── this GPU's slice ────────────────────────────────────────────────────────
    int rank() const;                                                   // this GPU: 0 .. world_size - 1
    int world_size() const;                                             // how many GPUs share the update
    size_t slice_start() const;                                         // where its slice starts in all_params()
    size_t slice_size() const;                                          // how many values it updates

    // ── back from a checkpoint ──────────────────────────────────────────────────
    void set_steps_taken(int steps_taken);                              // the step count it was saved at

protected:
    // ── for the three AdamWs ────────────────────────────────────────────────────
    Optimizer(const std::vector<Shape>& param_shapes, const AdamWConfig& config,   // the slice, decayed_count
              DType dtype, int rank, int world_size);
    virtual void update(int t) = 0;                                     // the AdamW's step t, from the grads
    void allocate();                                                    // adds grad_norm last, then allocates
    const std::vector<Shape>& param_shapes() const;                     // the params' shapes, the matrices first
    std::vector<Shape> slice_shapes() const;                            // the shapes of a state like m
    size_t decayed_in_slice() const;                                    // the slice's values before decayed_count
    void stop_unless_one_gpu(const char* what) const;                   // for the per-tensor views of the slice

    TensorBuffer buffer_;                // float: what each AdamW adds, then grad_norm
    AdamWConfig config_;

private:
    DType dtype_;
    std::vector<Shape> param_shapes_;
    int rank_;
    int world_size_;
    size_t slice_start_;
    size_t slice_size_;
    size_t decayed_count_ = 0;
    int grad_norm_group_index_ = -1;
    int steps_taken_ = 0;
};

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── BlockParams ───────────────────────────────────────────────────────────────

/// Makes the block's views.
///
///   tensors   in shapes()' order: the 4 matrices, then the 8 vectors; or, with no linear bias, the 4 matrices
///             and the LayerNorms' 4 (ln1_weight, ln1_bias, ln2_weight, ln2_bias), the linear biases left Tensor()
inline BlockParams::BlockParams(const std::vector<Tensor>& tensors) {
    if (tensors.size() != 4 + vector_count(true) && tensors.size() != 4 + vector_count(false)) {
        fprintf(stderr, "error: a block has %zu parameter tensors (%zu with no linear bias), got %zu\n",
                4 + vector_count(true), 4 + vector_count(false), tensors.size());
        exit(EXIT_FAILURE);
    }
    qkv_weight = tensors[0], attention_out_weight = tensors[1];
    mlp_up_weight = tensors[2], mlp_down_weight = tensors[3];
    if (tensors.size() == 4 + vector_count(true)) {
        ln1_weight = tensors[4], ln1_bias = tensors[5], qkv_bias = tensors[6], attention_out_bias = tensors[7];
        ln2_weight = tensors[8], ln2_bias = tensors[9], mlp_up_bias = tensors[10], mlp_down_bias = tensors[11];
    } else {
        ln1_weight = tensors[4], ln1_bias = tensors[5], ln2_weight = tensors[6], ln2_bias = tensors[7];
    }
}

/// Gives the block's shapes, matrices first: 12, 12 D² + 13 D = 7,087,872 values for D = 768; with no linear
/// bias 8, 12 D² + 4 D.
///
///   embed_dim     D
///   linear_bias   GPT2Config::linear_bias: the 4 linear biases or not
inline std::vector<Shape> BlockParams::shapes(size_t embed_dim, bool linear_bias) {
    size_t D = embed_dim;
    if (!linear_bias)
        return {{3 * D, D}, {D, D}, {4 * D, D}, {D, 4 * D},   // qkv, attention_out, mlp_up, mlp_down weights
                {D}, {D}, {D}, {D}};                          // ln1, ln2
    return {{3 * D, D}, {D, D}, {4 * D, D}, {D, 4 * D},       // qkv, attention_out, mlp_up, mlp_down weights
            {D}, {D}, {3 * D}, {D}, {D}, {D}, {4 * D}, {D}};  // ln1, qkv b, attention_out b, ln2, mlp_up b, mlp_down b
}

/// Gives how many 1-D tensors a block has after its 4 matrices: 8, or 4 with no linear bias (the LayerNorms').
///
///   linear_bias   GPT2Config::linear_bias
inline size_t BlockParams::vector_count(bool linear_bias) { return linear_bias ? 8 : 4; }

/// Gives the block's tensors in shapes()' order: 12, or 8 with no linear bias.
inline std::vector<Tensor> BlockParams::tensors() const {
    if (!has_linear_bias())
        return {qkv_weight, attention_out_weight, mlp_up_weight, mlp_down_weight, ln1_weight, ln1_bias, ln2_weight, ln2_bias};
    return {qkv_weight, attention_out_weight, mlp_up_weight, mlp_down_weight, ln1_weight, ln1_bias,
            qkv_bias,   attention_out_bias,   ln2_weight,    ln2_bias,        mlp_up_bias, mlp_down_bias};
}

/// Tells if the block has the 4 linear biases (made with GPT2Config::linear_bias true).
inline bool BlockParams::has_linear_bias() const { return qkv_bias.numel() > 0; }

// ── GPT2Params ────────────────────────────────────────────────────────────────

/// Makes the model's views; block l's 4 matrices are tensors 2 + 4l .., its V vectors 2 + 4L + Vl .. (V 8, or 4
/// with no linear bias).
///
///   tensors   4 + (4 + V) num_layers, in shapes()' order
///   config    the sizes, and linear_bias
inline GPT2Params::GPT2Params(const std::vector<Tensor>& tensors, const GPT2Config& config) {
    size_t num_layers = config.num_layers;
    size_t vectors = BlockParams::vector_count(config.linear_bias);      // a block's 1-D tensors
    size_t expected = 4 + (4 + vectors) * num_layers;
    if (tensors.size() != expected) {
        fprintf(stderr, "error: a GPT-2 of %zu layers%s has %zu parameter tensors, got %zu\n", num_layers,
                config.linear_bias ? "" : " with no linear bias", expected, tensors.size());
        exit(EXIT_FAILURE);
    }
    wte = tensors[0], wpe = tensors[1];
    size_t first_vector = 2 + 4 * num_layers;               // block 0's first vector
    for (size_t l = 0; l < num_layers; l++) {
        std::vector<Tensor> block(tensors.begin() + 2 + 4 * l, tensors.begin() + 2 + 4 * l + 4);
        block.insert(block.end(), tensors.begin() + first_vector + vectors * l, tensors.begin() + first_vector + vectors * (l + 1));
        blocks.push_back(BlockParams(block));
    }
    lnf_weight = tensors[expected - 2], lnf_bias = tensors[expected - 1];
}

/// Gives the model's shapes, every matrix first: 148 shapes, 124,475,904 values for GPT-2 small (100 shapes,
/// 124,392,960 values with no linear bias: 12 x 9 x 768 fewer).
///
///   config   the sizes, and linear_bias
inline std::vector<Shape> GPT2Params::shapes(const GPT2Config& config) {
    size_t D = config.embed_dim;
    std::vector<Shape> block = BlockParams::shapes(D, config.linear_bias);
    std::vector<Shape> shapes = {{config.padded_vocab_size, D}, {config.max_seq_len, D}};
    for (size_t l = 0; l < config.num_layers; l++) shapes.insert(shapes.end(), block.begin(), block.begin() + 4);
    for (size_t l = 0; l < config.num_layers; l++) shapes.insert(shapes.end(), block.begin() + 4, block.end());
    shapes.push_back({D});                                  // lnf_weight
    shapes.push_back({D});                                  // lnf_bias
    return shapes;
}

/// Gives the model's tensors in shapes()' order.
inline std::vector<Tensor> GPT2Params::tensors() const {
    std::vector<Tensor> all = {wte, wpe};
    for (const BlockParams& block : blocks) {
        std::vector<Tensor> its = block.tensors();
        all.insert(all.end(), its.begin(), its.begin() + 4);
    }
    for (const BlockParams& block : blocks) {
        std::vector<Tensor> its = block.tensors();
        all.insert(all.end(), its.begin() + 4, its.end());
    }
    all.push_back(lnf_weight);
    all.push_back(lnf_bias);
    return all;
}

// ── starting values and the loss ──────────────────────────────────────────────

/// Sets a block's GPT-2 starting values: matrices normal std 0.02, the two projections std 0.02 / sqrt(2
/// num_layers), biases 0 (the linear ones if it has them), LayerNorm weights 1.
///
///   params       the block's
///   num_layers   12 for GPT-2 small
inline void block_init_params(const BlockParams& params, int num_layers) {
    float projection_std = 0.02f / sqrtf(2.0f * num_layers);
    params.qkv_weight.normal_(0.0f, 0.02f);
    params.attention_out_weight.normal_(0.0f, projection_std);
    params.mlp_up_weight.normal_(0.0f, 0.02f);
    params.mlp_down_weight.normal_(0.0f, projection_std);
    params.ln1_weight.fill_(1.0), params.ln2_weight.fill_(1.0);
    params.ln1_bias.zero_(), params.ln2_bias.zero_();
    if (params.has_linear_bias()) {
        params.qkv_bias.zero_(), params.attention_out_bias.zero_();
        params.mlp_up_bias.zero_(), params.mlp_down_bias.zero_();
    }
}

/// Sets the whole model's GPT-2 starting values: wte and wpe normal std 0.02 (wte's padding rows 0, and they
/// stay 0), each block as block_init_params, lnf_weight 1, lnf_bias 0.
///
///   params   the model's
///   config   the sizes
inline void gpt2_init_params(const GPT2Params& params, const GPT2Config& config) {
    params.wte.normal_(0.0f, 0.02f);
    if (config.padded_vocab_size > config.vocab_size)
        params.wte.narrow(0, config.vocab_size, config.padded_vocab_size - config.vocab_size).zero_();
    params.wpe.normal_(0.0f, 0.02f);
    for (const BlockParams& block : params.blocks) block_init_params(block, (int)config.num_layers);
    params.lnf_weight.fill_(1.0), params.lnf_bias.zero_();
}

/// Gives the mean of the row losses the last forward wrote, for any of the three models. Waits for the GPU.
///
///   activations   the forward's: GPT2Activations, GPT2ActivationsBf16 or GPT2Activations8bit
template <typename ACTIVATIONS>
inline float gpt2_mean_loss(const ACTIVATIONS& activations) {
    std::vector<float> losses = activations.losses.tolist();
    double sum = 0.0;
    for (float loss : losses) sum += loss;
    return (float)(sum / losses.size());
}

// ── Optimizer: a step ────────────────────────────────────────────────────────

/// Takes one step with the gradients in grad(k): counts it, then runs the AdamW's update.
inline void Optimizer::step() {
    steps_taken_++;
    update(steps_taken_);
}

// ── Optimizer: the learning rate ─────────────────────────────────────────────

/// Sets the learning rate from the next step on: the schedule calls it before each step.
///
///   learning_rate   the new one
inline void Optimizer::set_learning_rate(float learning_rate) { config_.learning_rate = learning_rate; }

/// Gives the learning rate the next step uses.
inline float Optimizer::learning_rate() const { return config_.learning_rate; }

// ── Optimizer: reading: the same in all three ────────────────────────────────

/// Gives the 1-value tensor holding the norm of all the gradients.
inline const Tensor& Optimizer::grad_norm() const { return buffer_.tensor(grad_norm_group_index_, 0); }

/// Gives the whole float buffer.
inline const TensorBuffer& Optimizer::buffer() const { return buffer_; }

/// Gives the dtype of the model's params: Float32 (AdamW) or BFloat16 (AdamWBf16, AdamW8bit).
inline DType Optimizer::dtype() const { return dtype_; }

/// Gives how many parameter tensors there are: 148 for GPT-2 small.
inline int Optimizer::num_params() const { return (int)param_shapes_.size(); }

/// Gives how many steps were taken: 0 before the first.
inline int Optimizer::steps_taken() const { return steps_taken_; }

/// Gives how many values get weight decay: the matrices', first in params.
inline size_t Optimizer::decayed_count() const { return decayed_count_; }

// ── Optimizer: this GPU's slice ──────────────────────────────────────────────

/// Gives which GPU this is: 0 .. world_size - 1.
inline int Optimizer::rank() const { return rank_; }

/// Gives how many GPUs share the update.
inline int Optimizer::world_size() const { return world_size_; }

/// Gives where this GPU's slice starts in all_params(): rank x slice_size.
inline size_t Optimizer::slice_start() const { return slice_start_; }

/// Gives how many values this GPU's slice holds: all / world_size.
inline size_t Optimizer::slice_size() const { return slice_size_; }

// ── Optimizer: back from a checkpoint ────────────────────────────────────────

/// Sets the step count a checkpoint was saved at: AdamW's bias corrections depend on it.
///
///   steps_taken   the next step is this + 1
inline void Optimizer::set_steps_taken(int steps_taken) { steps_taken_ = steps_taken; }

// ── Optimizer: for the three AdamWs ──────────────────────────────────────────

/// Works out this GPU's slice and decayed_count; adds nothing to the buffer: the AdamW adds its groups, then
/// calls allocate(). Stops if rank is not one of world_size, if the values do not split into slices of a
/// multiple of 4, or if a matrix comes after a 1-D tensor.
///
///   param_shapes        the matrices first
///   config              the settings
///   dtype               the model's params: Float32 or BFloat16
///   rank, world_size    this GPU of how many
inline Optimizer::Optimizer(const std::vector<Shape>& param_shapes, const AdamWConfig& config, DType dtype, int rank,
                            int world_size)
    : config_(config), dtype_(dtype), param_shapes_(param_shapes), rank_(rank), world_size_(world_size) {
    // 1. this GPU's slice
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

    // 2. the matrices: they get weight decay, and come first
    bool past_the_matrices = false;
    for (int k = 0; k < (int)param_shapes.size(); k++) {
        if (param_shapes[k].size() < 2) {
            past_the_matrices = true;
            continue;
        }
        if (past_the_matrices) {
            fprintf(stderr, "error: parameter %d is a matrix after a 1-D tensor; give the matrices first\n", k);
            exit(EXIT_FAILURE);
        }
        size_t numel = 1;
        for (size_t size : param_shapes[k]) numel *= size;
        decayed_count_ += numel;
    }
}

/// Adds grad_norm last, then allocates the float buffer, all zeros.
inline void Optimizer::allocate() {
    grad_norm_group_index_ = buffer_.add_many({Shape{1}});
    buffer_.allocate();
}

/// Gives the params' shapes, the matrices first.
inline const std::vector<Shape>& Optimizer::param_shapes() const { return param_shapes_; }

/// Gives the shapes of a state kept for this slice (master, m, v): the params' shapes on one GPU, one flat
/// tensor of slice_size values on several.
inline std::vector<Shape> Optimizer::slice_shapes() const {
    if (world_size_ == 1) return param_shapes_;
    return {Shape{slice_size_}};
}

/// Gives how many of this slice's values get weight decay: its values before decayed_count.
inline size_t Optimizer::decayed_in_slice() const {
    return decayed_count_ > slice_start_ ? std::min(decayed_count_ - slice_start_, slice_size_) : 0;
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

#endif // MODEL_PARAMS_CUH
