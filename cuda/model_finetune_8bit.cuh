// model_finetune_8bit.cuh — model_8bit.cuh's model for fine-tuning on chats: the loss leaves out every row whose
// target is -1.
#ifndef MODEL_FINETUNE_8BIT_CUH
#define MODEL_FINETUNE_8BIT_CUH

#include "model_8bit.cuh"                                  // the 8-bit model: blocks, head, AdamW8bit, the activations
#include "kernels/finetune/softmax_loss_finetune_bf16.cuh"  // the loss with the -1 targets left out

/// Runs the whole forward as gpt2_8bit_forward does, the loss leaving out the rows with target -1.
///
///   activations          written; the products' scratch used
///   token_ids, targets   {B, T} ints; a target -1: not graded
///   params               the model's, bfloat16
///   config               the sizes
inline void gpt2_8bit_finetune_forward(const GPT2Activations8bit& activations, const Tensor& token_ids, const Tensor& targets,
                                       const GPT2Params& params, const GPT2Config& config) {
    // Activations.losses holds each row's loss, 0 for a row left out.
    size_t num_layers = params.blocks.size();
    // 1. the weights' 8-bit copies, the first forward after they changed
    if (!activations.weight_copies_made) {
        gpt2_8bit_round_weights(activations, params);
        activations.weight_copies_made = true;
    }
    // 2. the encoder and the blocks
    encoder_bf16_forward(activations.encoded, token_ids, params.wte, params.wpe);
    for (size_t l = 0; l < num_layers; l++) {
        const Tensor& input = l == 0 ? activations.encoded : activations.block_outputs[l - 1];
        block_8bit_forward(activations.block_outputs[l], input, params.blocks[l], activations, l);
    }
    // 3. LayerNorm f straight in 8 bits, then the head (no bias)
    head_8bit_rounded(activations.logits,
                      layer_norm_copies_8bit(activations.head_copies_scratch, activations.lnf_turned_8bit, activations.lnf_scale,
                                             activations.lnf_mean, activations.lnf_rstd, activations.block_outputs[num_layers - 1],
                                             params.lnf_weight, params.lnf_bias, activations.wte_copies,
                                             activations.head_history(MODEL_8BIT_HISTORY_HEAD_INPUT)));
    // 4. the loss, the rows with target -1 left out
    softmax_loss_finetune_bf16_forward(activations.losses, activations.max_of_each_row, activations.sum_of_each_row,
                                       activations.logits, targets, config.vocab_size,
                                       SOFTMAX_LOSS_FINETUNE_BF16_THREADS_PER_BLOCK, config.logit_softcap);
}

/// Adds every parameter's gradient onto param_grads as gpt2_8bit_backward does, rows with target -1 left out.
///
///   param_grads                   added onto: 0 before a step's first backward
///   token_ids, targets            the forward's
///   params, activations           the forward's; activations' scratch used
///   config                        the sizes
///   loss_grad_val                 d(loss) / d(each graded row's loss): GPUs / (the step's graded targets, every GPU's),
///                                 the GPUs since the grads are averaged over them after
inline void gpt2_8bit_finetune_backward(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                        const GPT2Params& params, const GPT2Activations8bit& activations,
                                        const GPT2Config& config, float loss_grad_val) {
    // Adds every parameter's gradient onto param_grads as gpt2_8bit_backward does, the loss's gradient each graded row's
    // loss_grad and 0 for a row left out.
    size_t num_layers = params.blocks.size();
    // 1. logits_grad written over the logits, the rows left out 0; then the head's backward: lnf_output_grad, and its
    //    part added onto wte_grad
    softmax_loss_finetune_bf16_backward(activations.logits, activations.logits, targets, activations.max_of_each_row,
                                        activations.sum_of_each_row, config.vocab_size, loss_grad_val,
                                        SOFTMAX_LOSS_FINETUNE_BF16_THREADS_PER_BLOCK, config.logit_softcap);
    HeadWeightGrad8bitCopies head_weight_grad_copies =
        weight_grad_copies_8bit(activations.head_input_grad_copies_scratch, activations.head_weight_grad_copies_scratch, activations.logits,
                                activations.head_history(MODEL_8BIT_HISTORY_HEAD_GRAD), activations.lnf_turned_8bit,
                                activations.lnf_scale);
    head_input_grad_8bit_rounded(activations.lnf_output_grad_scratch,
                                 input_grad_copies_8bit(activations.head_input_grad_copies_scratch, activations.wte_copies));
    head_weight_grad_8bit_rounded(param_grads.wte, head_weight_grad_copies);
    // 2. LayerNorm f, the blocks the other way, the encoder
    Tensor output_grad = activations.residual_grads_scratch[0];
    Tensor input_grad = activations.residual_grads_scratch[1];
    output_grad.zero_();                                    // LayerNorm's backward adds onto it
    layer_norm_bf16_backward(output_grad, param_grads.lnf_weight, param_grads.lnf_bias, activations.lnf_output_grad_scratch,
                             activations.block_outputs[num_layers - 1], params.lnf_weight, activations.lnf_mean, activations.lnf_rstd);
    for (size_t l = num_layers; l-- > 0;) {                 // l = num_layers - 1, ..., 0
        const Tensor& input = l == 0 ? activations.encoded : activations.block_outputs[l - 1];
        block_8bit_backward(input_grad, param_grads.blocks[l], output_grad, input, params.blocks[l], activations, l);
        std::swap(output_grad, input_grad);                 // block l - 1's output_grad
    }
    encoder_bf16_backward(param_grads.wte, param_grads.wpe, output_grad, token_ids);
}

/// Gives the sum of the row losses the last forward wrote (the rows left out add 0). Waits for the GPU.
///
///   activations   the forward's
inline double gpt2_8bit_finetune_loss_sum(const GPT2Activations8bit& activations) {
    std::vector<float> losses = activations.losses.tolist();
    double sum = 0.0;
    for (float loss : losses) sum += loss;
    return sum;
}

/// Runs one micro-batch of a fine-tune step and gives its loss sum over its graded rows.
///
///   param_grads, token_ids, targets, params, activations, config, loss_grad_val   as in gpt2_8bit_finetune_backward
inline double gpt2_8bit_finetune_train_step(const GPT2Params& param_grads, const Tensor& token_ids, const Tensor& targets,
                                            const GPT2Params& params, const GPT2Activations8bit& activations,
                                            const GPT2Config& config, float loss_grad_val) {
    // Runs one micro-batch of a fine-tune step: the forward, then the backward adding every parameter's gradient onto
    // param_grads. Gives the micro-batch's loss SUM over its graded rows, so waits for the GPU.
    gpt2_8bit_finetune_forward(activations, token_ids, targets, params, config);
    gpt2_8bit_finetune_backward(param_grads, token_ids, targets, params, activations, config, loss_grad_val);
    return gpt2_8bit_finetune_loss_sum(activations);
}

#endif // MODEL_FINETUNE_8BIT_CUH
