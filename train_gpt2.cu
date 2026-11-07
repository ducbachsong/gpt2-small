// train_gpt2.cu — trains GPT-2 small from scratch on the FineWeb-Edu token files, in bfloat16, on one GPU or
// on several GPUs of one machine. The defaults are llm.c's GPT-2 (124M) run: 524,288 tokens a step, one pass
// over the train files (10B tokens: 19,073 steps), warmup 700 steps, learning rate 6e-4 down to 0.
//
//     python data/fineweb.py --train-tokens 10000M                        # once: the token files, 10B tokens
//     ./train_gpt2 --steps 19073 ...                                      # ...or at the same time: dataloader.cuh
//     nvcc -O3 -std=c++17 -arch=native train_gpt2.cu -o train_gpt2 -lcublas -lcublasLt -lnccl
//     torchrun --standalone --nproc_per_node=8 --no-python ./train_gpt2   # 8 GPUs (distributed.cuh)
//     ./train_gpt2                                                        # one GPU: the same steps, 8x the time
//     ./train_gpt2 --steps 20 --tokens-per-step 32768 --val-every 10      # a quick try
//     ./train_gpt2 --checkpoint-dir ckpt                                  # a checkpoint every 1000 steps
//     ./train_gpt2 --checkpoint-dir ckpt --resume ckpt                    # on from the newest (checkpoint.cuh)
//
// One step trains on tokens_per_step tokens, all GPUs together: each GPU runs micro_batches batches of
// {16, 1024}, 524,288 / (GPUs x 16,384) of them: 4 on 8 GPUs, 32 on one. 16 rows, not 4: bigger products
// for the tensor cores and 4 times fewer launches per step; with flash attention a block keeps 94 MB per 4
// rows, so 16 rows take about 6 GB of an 80 GB GPU (T4: --batch-size 4).
//
//     zero the grads
//     micro_batches times:   next batch → gpt2_forward → gpt2_backward (each row's loss 1 / (micro_batches x
//                            16,384), the grads adding up)
//     average the grads over the GPUs, each GPU its own slice (on one GPU: nothing to do)
//     the norm of all the grads → AdamW on this GPU's slice, at this step's learning rate, clipped to norm 1
//     every GPU's slice of the new params to every GPU (on one GPU: nothing to do)
//
// The learning rate rises in a straight line over the first 700 steps, to 6e-4, then follows half a cosine
// down to 0 at the last step. Before the first step, after every 250 and after the last, the val loss: the
// mean loss of the val file's first 20 batches on each GPU, the same tokens every time.
//
// The model's params, grads and activations are bfloat16; AdamW keeps float master params, m and v for this
// GPU's slice (optimizer.cuh). --dtype float trains in float throughout. Only rank 0 prints. Every 200 steps a
// progress line: the time so far, the time left at this run's pace, and the clock time it will be done.
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <string>
#include "llmc/tensor.cuh"
#include "llmc/distributed.cuh"
#include "llmc/gpt2.cuh"
#include "llmc/dataloader.cuh"
#include "llmc/optimizer.cuh"
#include "llmc/checkpoint.cuh"
#include "llmc/kernels/grad_norm.cuh"

/// The run's settings; each can be changed with a --kebab-case flag of the same name.
struct TrainConfig {
    std::string train_files = "data/fineweb10B/fineweb_train_%06d.bin";
    std::string val_files = "data/fineweb10B/fineweb_val_%06d.bin";
    size_t batch_size = 16;                  // rows per micro-batch, on each GPU
    size_t seq_len = 1024;                   // tokens per row
    size_t tokens_per_step = 524288;         // all GPUs together: 2^19, GPT-2 small's ~0.5M (GPT-3 paper, llm.c)
    int steps = -1;                          // -1: one pass over the train files, 19,073 for 10B tokens
    float max_learning_rate = 6e-4f;
    float min_learning_rate = 0.0f;          // where the cosine ends
    int warmup_steps = 700;
    float weight_decay = 0.1f;
    float max_norm = 1.0f;                   // clip the grads to this norm; 0: no clipping
    int val_every = 250;
    int val_batches = 20;                    // on each GPU
    std::string dtype = "bf16";              // bf16 or float
    std::string moments = "bf16";            // AdamW's m and v with a bf16 model: bf16, or 8bit (m and sqrt(v))
    unsigned long long seed = 42;
    std::string nccl_id_file;                // "": /tmp/gpt2_nccl_id_<TORCHELASTIC_RUN_ID>.bin
    int progress_every = 200;                // a progress line, with the time left
    std::string checkpoint_dir;              // "": no checkpoints
    int checkpoint_every = 1000;
    int keep_checkpoints = 2;                // the newest ones; older ones are deleted
    std::string resume;                      // a checkpoint_dir (its latest) or a step folder; "": from scratch
};

// ── settings ──────────────────────────────────────────────────────────────────

/// Prints the flags and their defaults, then stops: for --help and for a flag it does not know.
static void print_usage_and_stop(const char* program, int exit_code) {
    TrainConfig d;
    printf("usage: %s [--flag value] ...\n", program);
    printf("  --train-files PATTERN         %s\n", d.train_files.c_str());
    printf("  --val-files PATTERN           %s\n", d.val_files.c_str());
    printf("  --batch-size N                %zu   rows per micro-batch\n", d.batch_size);
    printf("  --seq-len N                   %zu\n", d.seq_len);
    printf("  --tokens-per-step N           %zu   all GPUs together\n", d.tokens_per_step);
    printf("  --steps N                     one pass over the train files\n");
    printf("  --max-learning-rate X         %g\n", d.max_learning_rate);
    printf("  --min-learning-rate X         %g\n", d.min_learning_rate);
    printf("  --warmup-steps N              %d\n", d.warmup_steps);
    printf("  --weight-decay X              %g\n", d.weight_decay);
    printf("  --max-norm X                  %g   0: no clipping\n", d.max_norm);
    printf("  --val-every N                 %d\n", d.val_every);
    printf("  --val-batches N               %d   on each GPU\n", d.val_batches);
    printf("  --dtype bf16|float            %s\n", d.dtype.c_str());
    printf("  --moments bf16|8bit           %s   AdamW's m and v, with --dtype bf16; 8bit: no checkpoints yet\n",
           d.moments.c_str());
    printf("  --seed N                      %llu\n", d.seed);
    printf("  --nccl-id-file PATH           /tmp/gpt2_nccl_id_<TORCHELASTIC_RUN_ID>.bin; several GPUs only\n");
    printf("  --progress-every N            %d   a line with the time left\n", d.progress_every);
    printf("  --checkpoint-dir DIR          none: no checkpoints\n");
    printf("  --checkpoint-every N          %d\n", d.checkpoint_every);
    printf("  --keep-checkpoints N          %d   the newest; older ones are deleted\n", d.keep_checkpoints);
    printf("  --resume PATH                 a checkpoint dir (its latest) or one step folder of it\n");
    exit(exit_code);
}

/// Reads the --flag value pairs into a TrainConfig, starting from the defaults; stops on a flag it does not
/// know or a flag with no value.
static TrainConfig parse_args(int argc, char** argv) {
    TrainConfig c;
    for (int i = 1; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") print_usage_and_stop(argv[0], EXIT_SUCCESS);
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s needs a value\n", flag.c_str());
            print_usage_and_stop(argv[0], EXIT_FAILURE);
        }
        const char* value = argv[++i];
        if (flag == "--train-files") c.train_files = value;
        else if (flag == "--val-files") c.val_files = value;
        else if (flag == "--batch-size") c.batch_size = (size_t)atoll(value);
        else if (flag == "--seq-len") c.seq_len = (size_t)atoll(value);
        else if (flag == "--tokens-per-step") c.tokens_per_step = (size_t)atoll(value);
        else if (flag == "--steps") c.steps = atoi(value);
        else if (flag == "--max-learning-rate") c.max_learning_rate = (float)atof(value);
        else if (flag == "--min-learning-rate") c.min_learning_rate = (float)atof(value);
        else if (flag == "--warmup-steps") c.warmup_steps = atoi(value);
        else if (flag == "--weight-decay") c.weight_decay = (float)atof(value);
        else if (flag == "--max-norm") c.max_norm = (float)atof(value);
        else if (flag == "--val-every") c.val_every = atoi(value);
        else if (flag == "--val-batches") c.val_batches = atoi(value);
        else if (flag == "--dtype") c.dtype = value;
        else if (flag == "--moments") c.moments = value;
        else if (flag == "--seed") c.seed = strtoull(value, nullptr, 10);
        else if (flag == "--nccl-id-file") c.nccl_id_file = value;
        else if (flag == "--progress-every") c.progress_every = atoi(value);
        else if (flag == "--checkpoint-dir") c.checkpoint_dir = value;
        else if (flag == "--checkpoint-every") c.checkpoint_every = atoi(value);
        else if (flag == "--keep-checkpoints") c.keep_checkpoints = atoi(value);
        else if (flag == "--resume") c.resume = value;
        else {
            fprintf(stderr, "error: no flag %s\n", flag.c_str());
            print_usage_and_stop(argv[0], EXIT_FAILURE);
        }
    }
    if (c.dtype != "bf16" && c.dtype != "float") {
        fprintf(stderr, "error: --dtype is bf16 or float, not %s\n", c.dtype.c_str());
        exit(EXIT_FAILURE);
    }
    if (c.moments != "bf16" && c.moments != "8bit") {
        fprintf(stderr, "error: --moments is bf16 or 8bit, not %s\n", c.moments.c_str());
        exit(EXIT_FAILURE);
    }
    if (c.moments == "8bit" && (c.dtype != "bf16" || !c.checkpoint_dir.empty() || !c.resume.empty())) {
        fprintf(stderr, "error: --moments 8bit needs --dtype bf16, and has no checkpoints yet (no --checkpoint-dir "
                        "or --resume)\n");
        exit(EXIT_FAILURE);
    }
    if ((c.steps < 1 && c.steps != -1) || c.tokens_per_step < 1 || c.val_every < 1 || c.val_batches < 1 ||
        c.progress_every < 1 || c.checkpoint_every < 1 || c.keep_checkpoints < 1) {
        fprintf(stderr, "error: --steps (or -1), --tokens-per-step, --val-every, --val-batches, --progress-every, "
                        "--checkpoint-every and --keep-checkpoints must be at least 1\n");
        exit(EXIT_FAILURE);
    }
    if (c.warmup_steps < 0) {
        fprintf(stderr, "error: --warmup-steps must be at least 0\n");
        exit(EXIT_FAILURE);
    }
    if (c.nccl_id_file.empty()) {
        const char* run_id = getenv("TORCHELASTIC_RUN_ID");                // torchrun: new for every run
        c.nccl_id_file = std::string("/tmp/gpt2_nccl_id_") + (run_id != nullptr ? run_id : "no_run_id") + ".bin";
    }
    return c;
}

/// Prints like printf, on rank 0 only, and flushes: the other ranks train without a word.
static void print0(const Distributed& distributed, const char* format, ...) {
    if (distributed.rank() != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

// ── the time left ─────────────────────────────────────────────────────────────

/// Gives a length of time in the unit that reads best: "45 s", "12.5 min", "6.4 h", "9.9 days".
static std::string duration_text(double seconds) {
    char text[32];
    if (seconds < 90) snprintf(text, sizeof(text), "%.0f s", seconds);
    else if (seconds < 90 * 60) snprintf(text, sizeof(text), "%.1f min", seconds / 60);
    else if (seconds < 48 * 3600) snprintf(text, sizeof(text), "%.1f h", seconds / 3600);
    else snprintf(text, sizeof(text), "%.1f days", seconds / 86400);
    return text;
}

/// Gives the clock time `seconds` from now on this machine, as "Nov 12 14:20": when the run will be done.
static std::string clock_text_in(double seconds) {
    std::time_t then = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() +
                                                            std::chrono::seconds((long long)seconds));
    char text[32];
    std::strftime(text, sizeof(text), "%b %d %H:%M", std::localtime(&then));
    return text;
}

// ── the learning rate and the val loss ────────────────────────────────────────

/// Gives the learning rate of step `step` (0 for the first): a straight line up over the warmup steps, from
/// max / warmup_steps to max, then half a cosine from max down to min at the last step. With the defaults and
/// 10B train tokens (19,073 steps):
///
///     step       0          699      700      9886            19072
///     lr         8.6e-7     6e-4     6e-4     3e-4 (half)     0
static float learning_rate_at(int step, const TrainConfig& c) {
    if (step < c.warmup_steps) return c.max_learning_rate * (float)(step + 1) / (float)c.warmup_steps;
    int decay_steps = c.steps - 1 - c.warmup_steps;                    // from the end of warmup to the last step
    float progress = decay_steps > 0 ? (float)(step - c.warmup_steps) / (float)decay_steps : 1.0f;   // 0 .. 1
    float cosine = 0.5f * (1.0f + cosf(3.14159265f * progress));       // 1 .. 0
    return c.min_learning_rate + cosine * (c.max_learning_rate - c.min_learning_rate);
}

/// Gives the val loss: the mean loss of the first val_batches batches of each GPU's turn of the val files,
/// averaged over the GPUs; from the start every time, so every val loss of a run is on the same tokens.
static float val_loss(DataLoader& val_loader, const Tensor& token_ids, const Tensor& targets, const GPT2Params& params,
                      const GPT2Activations& activations, const GPT2Scratch& scratch, const GPT2Config& model_config,
                      int val_batches, const Distributed& distributed) {
    val_loader.reset();
    double sum = 0.0;
    for (int batch = 0; batch < val_batches; batch++) {
        val_loader.next_batch(token_ids, targets);
        gpt2_forward(activations, token_ids, targets, params, scratch, model_config);
        sum += gpt2_mean_loss(activations);
    }
    return distributed.mean_of_cpu_value((float)(sum / val_batches));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    TrainConfig c = parse_args(argc, argv);
    Distributed distributed(c.nccl_id_file);                            // first: it picks this process's GPU
    int rank = distributed.rank();
    int world_size = distributed.world_size();
    GPT2Config model_config;                                            // GPT-2 small
    DType dtype = c.dtype == "bf16" ? DType::BFloat16 : DType::Float32;
    size_t B = c.batch_size;
    size_t T = c.seq_len;
    size_t tokens_per_step = c.tokens_per_step;
    size_t tokens_per_round = (size_t)world_size * B * T;              // one micro-batch on every GPU
    if (tokens_per_step % tokens_per_round != 0) {
        print0(distributed, "error: --tokens-per-step %zu is not a multiple of %d GPU(s) x %zu x %zu = %zu\n",
               tokens_per_step, world_size, B, T, tokens_per_round);
        exit(EXIT_FAILURE);
    }
    int micro_batches = (int)(tokens_per_step / tokens_per_round);     // per GPU: 4 on 8 GPUs, 32 on one

    // the data: checks every file's header before anything else; each rank reads its own turns
    DataLoader train_loader(c.train_files, B, T, rank, world_size);
    DataLoader val_loader(c.val_files, B, T, rank, world_size);
    if (c.steps == -1) {                                                // one pass over the train files
        if (train_loader.more_files_coming()) {
            print0(distributed, "error: the train files are still being written (data/fineweb.py), so one pass is "
                                "not known yet: pass --steps (19,073 for 10B tokens)\n");
            exit(EXIT_FAILURE);
        }
        c.steps = (int)(train_loader.total_tokens() / tokens_per_step);
    }
    if (c.steps < 1) {
        print0(distributed, "error: the train files hold %zu tokens, less than one step of %zu\n",
               train_loader.total_tokens(), tokens_per_step);
        exit(EXIT_FAILURE);
    }

    // the parameters and their grads live in the optimizer; the model reads and writes views of them. Every
    // rank makes the same starting values from the same seed, and keeps the master of its own slice.
    AdamWConfig adamw_config;
    adamw_config.learning_rate = learning_rate_at(0, c);
    adamw_config.weight_decay = c.weight_decay;
    adamw_config.max_norm = c.max_norm;
    adamw_config.moments_8bit = c.moments == "8bit";
    AdamW optimizer(GPT2Params::shapes(model_config), adamw_config, dtype, rank, world_size);
    std::vector<Tensor> param_tensors;
    std::vector<Tensor> grad_tensors;
    for (int k = 0; k < optimizer.num_params(); k++) {
        param_tensors.push_back(optimizer.param(k));
        grad_tensors.push_back(optimizer.grad(k));
    }
    GPT2Params params(param_tensors, model_config), grads(grad_tensors, model_config);
    Tensor::manual_seed(c.seed);
    gpt2_init_params(params, model_config);
    optimizer.copy_params_to_master();
    Tensor grads_slice = optimizer.all_grads().narrow(0, optimizer.slice_start(), optimizer.slice_size());

    GPT2Activations activations(B, T, model_config, dtype);
    GPT2Scratch scratch(B, T, model_config, dtype);
    Tensor token_ids = Tensor::zeros({B, T}, DType::Int32);
    Tensor targets = Tensor::zeros({B, T}, DType::Int32);
    Tensor grad_norm_block_sums = Tensor::zeros({2, GRAD_NORM_BLOCKS});
    GradNorm grad_norm(grad_norm_block_sums);

    size_t free_bytes;
    size_t total_bytes;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    print0(distributed, "GPT-2 small: %zu parameters, %s, AdamW m and v in %s, on %d GPU(s)\n",
           optimizer.all_params().numel(), c.dtype.c_str(), c.dtype == "bf16" ? c.moments.c_str() : "float", world_size);
    print0(distributed, "train: %d file(s), %zu tokens%s; val: %zu tokens, %d batches of it on each GPU each time\n",
           train_loader.num_files(), train_loader.total_tokens(),
           train_loader.more_files_coming() ? " so far (more being written)" : "", val_loader.total_tokens(),
           c.val_batches);
    print0(distributed, "%d steps of %d GPU(s) x %d x {%zu, %zu} = %zu tokens: %zu tokens, %.2f epochs\n", c.steps,
           world_size, micro_batches, B, T, tokens_per_step, tokens_per_step * c.steps,
           (double)tokens_per_step * c.steps / train_loader.total_tokens());
    print0(distributed, "learning rate: %g, warmup %d steps, cosine down to %g; weight decay %g, clip at %g\n",
           c.max_learning_rate, c.warmup_steps, c.min_learning_rate, c.weight_decay, c.max_norm);
    print0(distributed, "GPU memory, rank 0: %.1f GB used of %.1f GB\n", (total_bytes - free_bytes) / 1e9,
           total_bytes / 1e9);
    if (c.checkpoint_dir.empty()) print0(distributed, "checkpoints: none\n");
    else print0(distributed, "checkpoints: every %d steps into %s, the newest %d kept\n", c.checkpoint_every,
                c.checkpoint_dir.c_str(), c.keep_checkpoints);

    // from a checkpoint: its master, m, v, step count, params and the train loader's place
    int first_step = 0;
    if (!c.resume.empty()) {
        first_step = checkpoint_load(c.resume, c.steps, optimizer, train_loader, distributed);
        print0(distributed, "resumed from %s: %d steps done\n", c.resume.c_str(), first_step);
    }
    print0(distributed, "\n");

    // the val loss before the first step, then after every val_every steps and after the last
    float loss_before = val_loss(val_loader, token_ids, targets, params, activations, scratch, model_config,
                                 c.val_batches, distributed);
    print0(distributed, "step %5d | val loss %.4f\n", first_step, loss_before);

    auto run_started = std::chrono::steady_clock::now();
    for (int step = first_step; step < c.steps; step++) {
        auto started = std::chrono::steady_clock::now();
        optimizer.all_grads().zero_();
        double loss_sum = 0.0;
        for (int micro_batch = 0; micro_batch < micro_batches; micro_batch++) {
            train_loader.next_batch(token_ids, targets);
            gpt2_forward(activations, token_ids, targets, params, scratch, model_config);
            gpt2_backward(grads, token_ids, targets, params, activations, scratch, model_config, micro_batches);
            loss_sum += gpt2_mean_loss(activations);
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
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        print0(distributed, "step %5d/%d | loss %.4f | norm %7.4f | lr %.2e | %6.0f ms | %7.0f tokens/s\n", step + 1,
               c.steps, loss, norm, learning_rate, seconds * 1000.0, tokens_per_step / seconds);

        int steps_done = step + 1;
        if (steps_done % c.val_every == 0 || steps_done == c.steps) {
            float loss_now = val_loss(val_loader, token_ids, targets, params, activations, scratch, model_config,
                                      c.val_batches, distributed);
            print0(distributed, "step %5d | val loss %.4f\n", steps_done, loss_now);
        }
        if (!c.checkpoint_dir.empty() && (steps_done % c.checkpoint_every == 0 || steps_done == c.steps)) {
            auto saving_started = std::chrono::steady_clock::now();
            checkpoint_save(c.checkpoint_dir, steps_done, c.steps, optimizer, train_loader, distributed,
                            c.keep_checkpoints);
            double saving_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - saving_started).count();
            print0(distributed, "step %5d | checkpoint %s/%s (%s)\n", steps_done, c.checkpoint_dir.c_str(),
                   checkpoint_folder_name(steps_done).c_str(), duration_text(saving_seconds).c_str());
        }
        // the time left: this run's time per step so far (val losses and checkpoints included), times the
        // steps left
        if (steps_done % c.progress_every == 0 && steps_done < c.steps) {
            double so_far = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_started).count();
            double left = so_far / (steps_done - first_step) * (c.steps - steps_done);
            print0(distributed, "progress: step %d/%d (%.1f%%) | %s so far | %s left | done ~%s\n", steps_done, c.steps,
                   100.0 * steps_done / c.steps, duration_text(so_far).c_str(), duration_text(left).c_str(),
                   clock_text_in(left).c_str());
        }
    }
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_started).count();
    print0(distributed, "done: steps %d .. %d in %s\n", first_step + 1, c.steps, duration_text(total).c_str());
    return 0;
}
