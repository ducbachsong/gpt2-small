// finetune_8bit.cu — fine-tunes the pretrained d24 model into a chat model (nanochat's SFT, scripts/chat_sft.py) with
// train_8bit.cu's 8-bit model: the same sizes, kernels, AdamW8bit and steps, on python/chat_sft.py's rows (SmolTalk,
// MMLU x 3, GSM8K x 4 in nanochat's chat format), the loss over the graded targets only (model_finetune_8bit.cuh:
// the user's words, special tokens, tool outputs and padding left out).
//
// It starts from the pretrained weights (--init-from: python/finetune_init.py's file, made from the exported
// nanochat-d24 model) with AdamW's m and v at 0, or goes on from its own checkpoint (--resume). It only reads the
// pretrained model: its checkpoints go to their own folder (--checkpoint-dir, default ckpt_sft), and it stops if that
// folder holds the --init-from file. On the Hub, the fine-tune keeps its own repos (d24-sft-checkpoints,
// nanochat-d24-sft), never the pretraining ones.
//
//     nvcc -O3 -std=c++20 -gencode arch=compute_90a,code=sm_90a finetune_8bit.cu -o finetune_8bit -lnccl
//     python python/finetune_init.py && python python/chat_sft.py                # the weights and the chat rows
//     ./finetune_8bit                                                       # one GPU: one pass over the rows
//     RANK=r LOCAL_RANK=r WORLD_SIZE=8 ./finetune_8bit                       # 8 GPUs (one process each, as train_8bit)
//
// The learning rate (nanochat's SFT): max 0.8 x the pretraining's 3e-4, a short warmup (FINETUNE_8BIT_WARMUP_STEPS:
// nanochat loads the base run's optimizer state, here m and v start at 0), flat, then a straight line down to 0 over
// the last FINETUNE_8BIT_WARMDOWN_SHARE of the steps; weight decay 0.
//
//     zero the grads
//     graded = the step's graded targets, every GPU's (ChatLoader::graded_ahead: no talk between GPUs)
//     micro_batches times:   next batch → forward, backward (each graded row's loss_grad GPUs / graded)
//     average the grads over the GPUs, clip to norm 1, AdamW on this GPU's slice, the new params to every GPU
//     loss = the sum of the graded rows' losses, every GPU's / graded
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#include "common/training.cuh"                 // TrainConfig, parse_args, print0, micro_batches_of, ...
#include "model_finetune_8bit.cuh"             // the 8-bit model's fine-tune step: the loss without the -1 targets
#include "helper/chat_loader.cuh"              // ChatLoader: python/chat_sft.py's rows
#include "helper/checkpoint_8bit.cuh"          // the 8-bit run's checkpoints, with the chat loader's place
#include "kernels/bf16/grad_norm_bf16.cuh"     // GradNormBf16: the grads stay bfloat16

// The model: train_8bit.cu's d24 (TRAIN_8BIT_*), which the --init-from weights must be
#define FINETUNE_8BIT_NUM_LAYERS 24
#define FINETUNE_8BIT_EMBED_DIM 1536
#define FINETUNE_8BIT_NUM_HEADS 12
#define FINETUNE_8BIT_VOCAB_SIZE 32768                  // nanochat's tokenizer, its 9 chat tokens among the ids
#define FINETUNE_8BIT_ATTENTION_WINDOW 512
#define FINETUNE_8BIT_MAX_SEQ_LEN 2048
#define FINETUNE_8BIT_QK_NORM 1
#define FINETUNE_8BIT_LOGIT_SOFTCAP 15.0f
// The run
#define FINETUNE_8BIT_TRAIN_FILE "python/tokenizer/data/chat_sft/chat_sft_train.bin"     // python/chat_sft.py's
#define FINETUNE_8BIT_VAL_FILE "python/tokenizer/data/chat_sft/chat_sft_val.bin"
#define FINETUNE_8BIT_INIT_FROM "python/tokenizer/data/finetune_init/nanochat-d24_params.bin"   // python/finetune_init.py's
#define FINETUNE_8BIT_CHECKPOINT_DIR "ckpt_sft"            // never the pretraining's ckpt
#define FINETUNE_8BIT_BATCH_SIZE 16                     // rows a micro-batch (train_8bit's d24 runs: 16, 42.4 GB)
#define FINETUNE_8BIT_MAX_LEARNING_RATE 2.4e-4f         // 0.8 x the pretraining's 3e-4 (nanochat's --init-lr-frac 0.8)
#define FINETUNE_8BIT_WARMUP_STEPS 20                   // m and v start at 0 (nanochat: 0, with the base run's momentum)
#define FINETUNE_8BIT_WARMDOWN_SHARE 0.5f               // the last half of the steps: a straight line down to 0
#define FINETUNE_8BIT_WEIGHT_DECAY 0.0f                 // nanochat's SFT
#define FINETUNE_8BIT_VAL_EVERY 100
#define FINETUNE_8BIT_VAL_BATCHES 40                    // on each GPU: 40 x 16 rows of the val file
#define FINETUNE_8BIT_INIT_MAGIC 20261120               // python/finetune_init.py's header[0]
#define FINETUNE_8BIT_INIT_HEADER_INT64S 8

/// Gives the model this program fine-tunes: train_8bit's d24, no linear biases, QK-norm, softcap 15.
static GPT2Config model_config_of(void) {
    GPT2Config config;
    config.linear_bias = false;
    config.num_layers = FINETUNE_8BIT_NUM_LAYERS;
    config.embed_dim = FINETUNE_8BIT_EMBED_DIM;
    config.num_heads = FINETUNE_8BIT_NUM_HEADS;
    config.max_seq_len = FINETUNE_8BIT_MAX_SEQ_LEN;
    config.vocab_size = FINETUNE_8BIT_VOCAB_SIZE;
    config.padded_vocab_size = FINETUNE_8BIT_VOCAB_SIZE;
    config.attention_window = FINETUNE_8BIT_ATTENTION_WINDOW;
    config.qk_norm = FINETUNE_8BIT_QK_NORM != 0;
    config.logit_softcap = FINETUNE_8BIT_LOGIT_SOFTCAP;
    return config;
}

/// Gives this program's run defaults: the chat files, the SFT learning rate, no weight decay, its own checkpoint dir.
static TrainConfig finetune_defaults_of(void) {
    TrainConfig defaults;
    defaults.linear_bias = false;
    defaults.seq_len = FINETUNE_8BIT_MAX_SEQ_LEN;
    defaults.batch_size = FINETUNE_8BIT_BATCH_SIZE;
    defaults.train_files = FINETUNE_8BIT_TRAIN_FILE;
    defaults.val_files = FINETUNE_8BIT_VAL_FILE;
    defaults.max_learning_rate = FINETUNE_8BIT_MAX_LEARNING_RATE;
    defaults.warmup_steps = FINETUNE_8BIT_WARMUP_STEPS;
    defaults.weight_decay = FINETUNE_8BIT_WEIGHT_DECAY;
    defaults.val_every = FINETUNE_8BIT_VAL_EVERY;
    defaults.val_batches = FINETUNE_8BIT_VAL_BATCHES;
    defaults.checkpoint_dir = FINETUNE_8BIT_CHECKPOINT_DIR;
    defaults.data_ready = true;
    return defaults;
}

/// Gives the learning rate of a fine-tune step: warmup, flat at max, then linear down to min.
///
///   step   0 .. c.steps - 1
///   c      the run's settings
static float finetune_learning_rate_at(int step, const TrainConfig& c) {
    // Gives the learning rate of step `step` (0 for the first), nanochat's SFT shape: up over the warmup, flat at max,
    // then a straight line from max down to min over the last FINETUNE_8BIT_WARMDOWN_SHARE of the steps.
    if (step < c.warmup_steps) return c.max_learning_rate * (float)(step + 1) / (float)c.warmup_steps;
    int warmdown_steps = (int)(c.steps * FINETUNE_8BIT_WARMDOWN_SHARE);
    int warmdown_start = c.steps - warmdown_steps;
    if (step < warmdown_start || warmdown_steps < 1) return c.max_learning_rate;
    float progress = (float)(step - warmdown_start) / (float)warmdown_steps;   // 0 .. just under 1
    return c.max_learning_rate + progress * (c.min_learning_rate - c.max_learning_rate);
}

/// Takes --init-from PATH out of argv (parse_args does not know it) and gives PATH, or the default.
///
///   argc, argv   main's; the pair removed
static std::string take_init_from(int& argc, char** argv) {
    std::string init_from = FINETUNE_8BIT_INIT_FROM;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--init-from") != 0) continue;
        init_from = argv[i + 1];
        for (int k = i; k + 2 <= argc; k++) argv[k] = argv[k + 2];
        argc -= 2;
        break;
    }
    return init_from;
}

/// Fills the params and this slice's master with the pretrained weights from python/finetune_init.py's file.
///
///   path          the file
///   optimizer     its params and master filled
///   distributed   for printing on rank 0
static void load_init_params(const std::string& path, AdamW8bit& optimizer, const Distributed& distributed) {
    // Fills the params with the pretrained weights from python/finetune_init.py's file (every value in GPT2Params::shapes'
    // order, float), then this slice's master from them; stops if the file is not that or not this model's size.
    FILE* file = fopen(path.c_str(), "rb");
    int64_t header[FINETUNE_8BIT_INIT_HEADER_INT64S] = {};
    size_t count = optimizer.all_params().numel();
    if (file == nullptr || fread(header, sizeof(int64_t), FINETUNE_8BIT_INIT_HEADER_INT64S, file) != FINETUNE_8BIT_INIT_HEADER_INT64S ||
        header[0] != FINETUNE_8BIT_INIT_MAGIC || (size_t)header[2] != count) {
        print0(distributed, "error: %s is not python/finetune_init.py's file for this model (%zu values; it says %lld)\n",
               path.c_str(), count, (long long)header[2]);
        exit(EXIT_FAILURE);
    }
    std::vector<float> values(count);
    size_t read_values = fread(values.data(), sizeof(float), count, file);
    fclose(file);
    if (read_values != count) {
        print0(distributed, "error: %s is cut short: %zu of %zu values\n", path.c_str(), read_values, count);
        exit(EXIT_FAILURE);
    }
    Tensor floats = Tensor::empty({count});
    floats.copy_from_cpu(values.data());
    optimizer.all_params().copy_(floats);                               // to bfloat16: the file's values are bfloat16's
    optimizer.copy_params_to_master();
}

/// Gives the val loss over the first val_batches rounds of the val rows, every GPU's.
///
///   val_loader                       read from its start
///   token_ids, targets               filled with each batch
///   params, activations, config      the model's
///   val_batches                      rounds
///   distributed                      for the sum over the GPUs
static float val_loss(ChatLoader& val_loader, const Tensor& token_ids, const Tensor& targets, const GPT2Params& params,
                      const GPT2Activations8bit& activations, const GPT2Config& config, int val_batches,
                      const Distributed& distributed) {
    // Gives the val loss: the sum of the graded rows' losses over the first val_batches rounds of the val rows, every
    // GPU's, / their graded targets; from the start every time, so every val loss is on the same targets.
    val_loader.reset();
    size_t graded = val_loader.graded_ahead(val_batches);
    double sum = 0.0;
    for (int batch = 0; batch < val_batches; batch++) {
        val_loader.next_batch(token_ids, targets);
        gpt2_8bit_finetune_forward(activations, token_ids, targets, params, config);
        sum += gpt2_8bit_finetune_loss_sum(activations);
    }
    double all_sum = (double)distributed.mean_of_cpu_value((float)sum) * distributed.world_size();
    return graded > 0 ? (float)(all_sum / graded) : 0.0f;
}

int main(int argc, char** argv) {
    // 1. the settings, the GPUs and the chat rows
    std::string init_from = take_init_from(argc, argv);
    TrainConfig c = parse_args(argc, argv, finetune_defaults_of());
    Distributed distributed(c.nccl_id_file);
    int micro_batches = micro_batches_of(c, distributed);
    ChatLoader train_loader(c.train_files, c.batch_size, c.seq_len, distributed.rank(), distributed.world_size());
    ChatLoader val_loader(c.val_files, c.batch_size, c.seq_len, distributed.rank(), distributed.world_size());
    if (c.steps == -1) c.steps = (int)(train_loader.rounds_per_epoch() / micro_batches);   // one pass
    if (c.steps < 1) {
        print0(distributed, "error: %s holds %zu rows, less than one step\n", c.train_files.c_str(), train_loader.total_rows());
        exit(EXIT_FAILURE);
    }
    int val_batches = (int)std::min((size_t)c.val_batches, val_loader.rounds_per_epoch());
    // the pretrained weights are only read: the fine-tune's checkpoints never go where they are
    if (!c.checkpoint_dir.empty() && c.resume.empty() &&
        init_from.compare(0, c.checkpoint_dir.size() + 1, c.checkpoint_dir + "/") == 0) {
        print0(distributed, "error: --init-from %s is inside --checkpoint-dir %s: the fine-tune would save over the "
                            "pretrained weights; give it its own folder\n", init_from.c_str(), c.checkpoint_dir.c_str());
        exit(EXIT_FAILURE);
    }

    // 2. the model: the params and grads live in AdamW8bit; the pretrained weights in, m and v at 0
    GPT2Config model_config = model_config_of();
    AdamWConfig adamw_config = adamw_config_of(c);
    adamw_config.learning_rate = finetune_learning_rate_at(0, c);
    AdamW8bit optimizer(GPT2Params::shapes(model_config), adamw_config, distributed.rank(), distributed.world_size());
    GPT2Params params = params_of(optimizer, model_config);
    GPT2Params grads = grads_of(optimizer, model_config);
    if (c.resume.empty()) load_init_params(init_from, optimizer, distributed);
    GPT2Activations8bit activations(c.batch_size, c.seq_len, model_config);
    Tensor token_ids = Tensor::zeros({c.batch_size, c.seq_len}, DType::Int32);
    Tensor targets = Tensor::zeros({c.batch_size, c.seq_len}, DType::Int32);
    Tensor grads_slice = optimizer.all_grads().narrow(0, optimizer.slice_start(), optimizer.slice_size());
    Tensor grad_norm_block_sums = Tensor::zeros({2, GRAD_NORM_BF16_BLOCKS});
    GradNormBf16 grad_norm(grad_norm_block_sums);
    size_t free_bytes, total_bytes;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    print0(distributed, "fine-tune d24 (8 bits, QK-norm, logit softcap %g): %zu parameters, on %d GPU(s), from %s\n",
           model_config.logit_softcap, optimizer.all_params().numel(), distributed.world_size(),
           c.resume.empty() ? init_from.c_str() : c.resume.c_str());
    print0(distributed, "train: %s, %zu rows, %zu graded targets (%.1f%% of the places); val: %s, %zu rows, %d rounds of "
                        "it each time\n", c.train_files.c_str(), train_loader.total_rows(), train_loader.graded_targets(),
           100.0 * train_loader.graded_targets() / train_loader.total_tokens(), c.val_files.c_str(),
           val_loader.total_rows(), val_batches);
    print0(distributed, "%d steps of %d GPU(s) x %d x {%zu, %zu} = %zu tokens (%.2f epochs)\n", c.steps,
           distributed.world_size(), micro_batches, c.batch_size, c.seq_len, c.tokens_per_step,
           (double)c.steps * micro_batches / train_loader.rounds_per_epoch());
    print0(distributed, "learning rate: %g, warmup %d steps, flat, down to %g over the last %.0f%%; weight decay %g, clip at %g\n",
           c.max_learning_rate, c.warmup_steps, c.min_learning_rate, 100.0f * FINETUNE_8BIT_WARMDOWN_SHARE,
           c.weight_decay, c.max_norm);
    print0(distributed, "GPU memory, rank 0: %.1f GB used of %.1f GB\n", (total_bytes - free_bytes) / 1e9, total_bytes / 1e9);
    print0(distributed, "checkpoints: %s\n", c.checkpoint_dir.empty() ? "none" : (c.checkpoint_dir + ", the last step's").c_str());

    // 3. from its own checkpoint: master, m, sqrt(v), step count, rounding histories, the loader's place
    int first_step = 0;
    if (!c.resume.empty()) {
        first_step = checkpoint_8bit_load(c.resume, c.steps, optimizer, activations.round_histories, train_loader,
                                          distributed);
        gpt2_8bit_weights_changed(activations);
        print0(distributed, "resumed from %s: %d steps done\n", c.resume.c_str(), first_step);
    }
    print0(distributed, "\n");

    // 4. the steps; the val loss before the first, every val_every steps and after the last
    float loss_before = val_loss(val_loader, token_ids, targets, params, activations, model_config, val_batches, distributed);
    print0(distributed, "step %5d | val loss %.4f\n", first_step, loss_before);
    auto run_started = std::chrono::steady_clock::now();
    for (int step = first_step; step < c.steps; step++) {
        auto started = std::chrono::steady_clock::now();
        optimizer.all_grads().zero_();
        size_t graded = train_loader.graded_ahead(micro_batches);       // every GPU's, this step
        float loss_grad_val = graded > 0 ? (float)distributed.world_size() / (float)graded : 0.0f;
        double loss_sum = 0.0;
        for (int micro_batch = 0; micro_batch < micro_batches; micro_batch++) {
            train_loader.next_batch(token_ids, targets);
            loss_sum += gpt2_8bit_finetune_train_step(grads, token_ids, targets, params, activations, model_config,
                                                      loss_grad_val);
        }
        distributed.reduce_scatter_mean(optimizer.all_grads());
        grad_norm.compute(grads_slice, optimizer.grad_norm());
        distributed.slice_norms_to_norm(optimizer.grad_norm());
        float learning_rate = finetune_learning_rate_at(step, c);
        optimizer.set_learning_rate(learning_rate);
        optimizer.step();
        distributed.all_gather(optimizer.all_params());
        gpt2_8bit_weights_changed(activations);
        float norm = optimizer.grad_norm().item<float>();
        double all_loss_sum = (double)distributed.mean_of_cpu_value((float)loss_sum) * distributed.world_size();
        float loss = graded > 0 ? (float)(all_loss_sum / graded) : 0.0f;
        double seconds = seconds_since(started);
        print0(distributed, "step %5d/%d | loss %.4f | norm %7.4f | lr %.2e | graded %5.1f%% | %6.0f ms | %7.0f tokens/s\n",
               step + 1, c.steps, loss, norm, learning_rate, 100.0 * graded / c.tokens_per_step, seconds * 1000.0,
               c.tokens_per_step / seconds);

        int steps_done = step + 1;
        if (steps_done % c.val_every == 0 || steps_done == c.steps) {
            float loss_now = val_loss(val_loader, token_ids, targets, params, activations, model_config, val_batches, distributed);
            print0(distributed, "step %5d | val loss %.4f\n", steps_done, loss_now);
        }
        if (!c.checkpoint_dir.empty() && (steps_done % c.checkpoint_every == 0 || steps_done == c.steps)) {
            auto saving_started = std::chrono::steady_clock::now();
            checkpoint_8bit_save(c.checkpoint_dir, steps_done, c.steps, optimizer, activations.round_histories,
                                 train_loader, distributed, c.keep_checkpoints);
            print0(distributed, "step %5d | checkpoint %s/%s (%s)\n", steps_done, c.checkpoint_dir.c_str(),
                   checkpoint_folder_name(steps_done).c_str(), duration_text(seconds_since(saving_started)).c_str());
        }
        print_progress(steps_done, first_step, c, seconds_since(run_started), distributed);
    }
    print0(distributed, "done: steps %d .. %d in %s\n", first_step + 1, c.steps,
           duration_text(seconds_since(run_started)).c_str());
    return 0;
}
