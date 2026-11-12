// train_8bit.cu — trains GPT-2 small from scratch on the FineWeb-Edu token files with its products in 8 bits
// (model_8bit.cuh), on one H100 or on several of one machine: the params, grads and activations bfloat16,
// each product's two sides rounded to FP8 E4M3 just before it; AdamW8bit keeps float master params and m and
// sqrt(v) in 8 bits for this GPU's slice. No checkpoints yet. The other two: train_fp32.cu and train_bf16.cu.
// The defaults are llm.c's GPT-2 (124M) run: 524,288 tokens a step, one pass over the train files (10B tokens:
// 19,073 steps), warmup 700 steps, learning rate 6e-4 down to 0.
//
//     nvcc -O3 -std=c++17 -arch=sm_90a train_8bit.cu -o train_8bit -lnccl   # an H100: WGMMA and TMA
//     torchrun --standalone --nproc_per_node=8 --no-python ./train_8bit   # 8 GPUs (distributed.cuh)
//     ./train_8bit                                                        # one GPU: the same steps, 8x the time
//     ./train_8bit --steps 20 --tokens-per-step 32768 --val-every 10      # a quick try
//
// One step trains on tokens_per_step tokens, all GPUs together: each GPU runs micro_batches batches of
// {16, 1024}, 524,288 / (GPUs x 16,384) of them: 4 on 8 GPUs, 32 on one. 16 rows, as train_bf16.cu runs.
//
//     zero the grads
//     micro_batches times:   next batch → gpt2_8bit_train_step: forward, backward (each row's loss
//                            1 / (micro_batches x 16,384), the grads adding up)
//     average the grads over the GPUs, each GPU its own slice (on one GPU: nothing to do)
//     the norm of all the grads → AdamW on this GPU's slice, at this step's learning rate, clipped to norm 1
//     every GPU's slice of the new params to every GPU (on one GPU: nothing to do)
//
// The learning rate, the val loss, the progress lines and the flags are train_bf16.cu's; the settings, flags
// and printing all three programs share: common/training.cuh.
#include <chrono>
#include "llmc/common/training.cuh"                 // TrainConfig, parse_args, learning_rate_at, print0, ...
#include "llmc/model_8bit.cuh"                      // GPT2Activations8bit, gpt2_8bit_train_step, AdamW8bit
#include "llmc/kernels/bf16/grad_norm_bf16.cuh"     // GradNormBf16: the grads stay bfloat16

/// Gives the val loss: the mean loss of the first val_batches batches of each GPU's turn of the val files,
/// averaged over the GPUs; from the start every time, so every val loss of a run is on the same tokens.
///
///   val_loader                       read from its start
///   token_ids, targets               filled with each batch
///   params, activations, config      the model's
///   val_batches                      on each GPU
///   distributed                      for the mean over the GPUs
static float val_loss(DataLoader& val_loader, const Tensor& token_ids, const Tensor& targets, const GPT2Params& params,
                      const GPT2Activations8bit& activations, const GPT2Config& config, int val_batches,
                      const Distributed& distributed) {
    val_loader.reset();
    double sum = 0.0;
    for (int batch = 0; batch < val_batches; batch++) {
        val_loader.next_batch(token_ids, targets);
        gpt2_8bit_forward(activations, token_ids, targets, params, config);
        sum += gpt2_mean_loss(activations);
    }
    return distributed.mean_of_cpu_value((float)(sum / val_batches));
}

int main(int argc, char** argv) {
    // 1. the settings (no checkpoints yet), the GPUs and the data: every file's header checked before anything
    // else
    TrainConfig c = parse_args(argc, argv);
    if (!c.checkpoint_dir.empty() || !c.resume.empty()) {
        fprintf(stderr, "error: train_8bit has no checkpoints yet (no --checkpoint-dir or --resume): its m and v "
                        "are 8 bits\n");
        exit(EXIT_FAILURE);
    }
    Distributed distributed(c.nccl_id_file);                            // first: it picks this process's GPU
    int micro_batches = micro_batches_of(c, distributed);
    DataLoader train_loader(c.train_files, c.batch_size, c.seq_len, distributed.rank(), distributed.world_size());
    DataLoader val_loader(c.val_files, c.batch_size, c.seq_len, distributed.rank(), distributed.world_size());
    set_steps(c, train_loader, distributed);

    // 2. the model: its params and grads live in AdamW8bit, the model reads and writes views of them. Every rank
    // makes the same starting values from the same seed, and keeps the master of its own slice.
    GPT2Config model_config;                                            // GPT-2 small
    AdamW8bit optimizer(GPT2Params::shapes(model_config), adamw_config_of(c), distributed.rank(),
                        distributed.world_size());
    GPT2Params params = params_of(optimizer, model_config);
    GPT2Params grads = grads_of(optimizer, model_config);
    Tensor::manual_seed(c.seed);
    gpt2_init_params(params, model_config);
    optimizer.copy_params_to_master();
    GPT2Activations8bit activations(c.batch_size, c.seq_len, model_config);
    Tensor token_ids = Tensor::zeros({c.batch_size, c.seq_len}, DType::Int32);
    Tensor targets = Tensor::zeros({c.batch_size, c.seq_len}, DType::Int32);
    Tensor grads_slice = optimizer.all_grads().narrow(0, optimizer.slice_start(), optimizer.slice_size());
    Tensor grad_norm_block_sums = Tensor::zeros({2, GRAD_NORM_BF16_BLOCKS});
    GradNormBf16 grad_norm(grad_norm_block_sums);
    print_run("8 bits", c, optimizer, micro_batches, train_loader, val_loader, distributed);

    int first_step = 0;                                                 // no checkpoints: always from scratch
    print0(distributed, "\n");

    // 4. the steps; the val loss before the first, then after every val_every steps and after the last
    float loss_before = val_loss(val_loader, token_ids, targets, params, activations, model_config, c.val_batches,
                                 distributed);
    print0(distributed, "step %5d | val loss %.4f\n", first_step, loss_before);
    auto run_started = std::chrono::steady_clock::now();
    for (int step = first_step; step < c.steps; step++) {
        auto started = std::chrono::steady_clock::now();
        optimizer.all_grads().zero_();
        double loss_sum = 0.0;
        for (int micro_batch = 0; micro_batch < micro_batches; micro_batch++) {
            train_loader.next_batch(token_ids, targets);
            loss_sum += gpt2_8bit_train_step(grads, token_ids, targets, params, activations, model_config,
                                             micro_batches);
        }
        distributed.reduce_scatter_mean(optimizer.all_grads());        // this rank's slice: averaged over all
        grad_norm.compute(grads_slice, optimizer.grad_norm());         // the norm of this rank's slice...
        distributed.slice_norms_to_norm(optimizer.grad_norm());        // ...then of all of them
        float learning_rate = learning_rate_at(step, c);
        optimizer.set_learning_rate(learning_rate);
        optimizer.step();                                              // this rank's slice
        distributed.all_gather(optimizer.all_params());                // every slice to every rank
        float norm = optimizer.grad_norm().item<float>();              // waits for the GPU: the step is done
        float loss = distributed.mean_of_cpu_value((float)(loss_sum / micro_batches));
        double seconds = seconds_since(started);
        print0(distributed, "step %5d/%d | loss %.4f | norm %7.4f | lr %.2e | %6.0f ms | %7.0f tokens/s\n", step + 1,
               c.steps, loss, norm, learning_rate, seconds * 1000.0, c.tokens_per_step / seconds);

        int steps_done = step + 1;
        if (steps_done % c.val_every == 0 || steps_done == c.steps) {
            float loss_now = val_loss(val_loader, token_ids, targets, params, activations, model_config, c.val_batches,
                                      distributed);
            print0(distributed, "step %5d | val loss %.4f\n", steps_done, loss_now);
        }
        print_progress(steps_done, first_step, c, seconds_since(run_started), distributed);
    }
    print0(distributed, "done: steps %d .. %d in %s\n", first_step + 1, c.steps,
           duration_text(seconds_since(run_started)).c_str());
    return 0;
}
