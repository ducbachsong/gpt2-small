// training.cuh — what the three train programs share: train_fp32.cu (model_fp32.cuh), train_bf16.cu
// (model_bf16.cuh) and train_8bit.cu (model_8bit.cuh). The run's settings and their flags (TrainConfig,
// parse_args), the learning rate schedule, how many micro-batches a step takes, the model's params and grads as
// views of the AdamW's buffer, and the lines rank 0 prints. Each program has its own main: the same steps, with
// its model's activations, train step and AdamW.
#ifndef TRAINING_CUH
#define TRAINING_CUH

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <ctime>
#include <string>
#include <vector>
#include "model_params.cuh"                 // GPT2Config, GPT2Params, AdamWConfig, Optimizer
#include "../helper/distributed.cuh"
#include "../helper/dataloader.cuh"

/// The run's settings; each can be changed with a --kebab-case flag of the same name.
struct TrainConfig {
    std::string train_files = "data/fineweb10B/fineweb_train_%06d.bin";
    std::string val_files = "data/fineweb10B/fineweb_val_%06d.bin";
    size_t batch_size = 64;                  // rows per micro-batch, on each GPU: tune_model's best MFU on an H100 NVL
    size_t seq_len = 1024;                   // tokens per row
    size_t tokens_per_step = 524288;         // all GPUs together: 2^19, GPT-2 small's ~0.5M (GPT-3 paper, llm.c)
    bool linear_bias = true;                 // the model's 4 linears have biases (GPT2Config::linear_bias); train_bf16 and train_8bit only
    int steps = -1;                          // -1: one pass over the train files, 19,073 for 10B tokens
    float max_learning_rate = 6e-4f;
    float min_learning_rate = 0.0f;          // where the cosine ends
    int warmup_steps = 700;
    float weight_decay = 0.1f;
    float max_norm = 1.0f;                   // clip the grads to this norm; 0: no clipping
    int val_every = 250;
    int val_batches = 20;                    // on each GPU
    unsigned long long seed = 42;
    std::string nccl_id_file;                // "": /tmp/gpt2_nccl_id_<TORCHELASTIC_RUN_ID>.bin
    int progress_every = 200;                // a progress line, with the time left
    std::string checkpoint_dir;              // "": no checkpoints
    int checkpoint_every = 1000;
    int keep_checkpoints = 2;                // the newest ones; older ones are deleted
    std::string resume;                      // a checkpoint_dir (its latest) or a step folder; "": from scratch
};

// ── settings ──────────────────────────────────────────────────────────────────
void print_usage_and_stop(const char* program, int exit_code, const TrainConfig& defaults = TrainConfig());
TrainConfig parse_args(int argc, char** argv, const TrainConfig& defaults = TrainConfig());
float learning_rate_at(int step, const TrainConfig& c);
AdamWConfig adamw_config_of(const TrainConfig& c);

// ── the run's size ────────────────────────────────────────────────────────────
int micro_batches_of(const TrainConfig& c, const Distributed& distributed);
void set_steps(TrainConfig& c, const DataLoader& train_loader, const Distributed& distributed);

// ── the params and grads ──────────────────────────────────────────────────────
GPT2Params params_of(const Optimizer& optimizer, const GPT2Config& config);
GPT2Params grads_of(const Optimizer& optimizer, const GPT2Config& config);

// ── printing ──────────────────────────────────────────────────────────────────
void print0(const Distributed& distributed, const char* format, ...);
void print_run(const char* model_name, const TrainConfig& c, const Optimizer& optimizer, int micro_batches,
               const DataLoader& train_loader, const DataLoader& val_loader, const Distributed& distributed);
void print_progress(int steps_done, int first_step, const TrainConfig& c, double seconds_so_far,
                    const Distributed& distributed);
std::string duration_text(double seconds);
std::string clock_text_in(double seconds);
double seconds_since(std::chrono::steady_clock::time_point started);

// ══ the code, in the order of the lists above ══════════════════════════════════

// ── settings ──────────────────────────────────────────────────────────────────

/// Prints the flags and their defaults, then stops: for --help and for a flag it does not know.
///
///   program     argv[0], for the usage line
///   exit_code   EXIT_SUCCESS for --help, EXIT_FAILURE for a mistake
///   defaults    the trainer's defaults; not given: TrainConfig's
inline void print_usage_and_stop(const char* program, int exit_code, const TrainConfig& defaults) {
    const TrainConfig& d = defaults;
    printf("usage: %s [--flag value] ...\n", program);
    printf("  --train-files PATTERN         %s\n", d.train_files.c_str());
    printf("  --val-files PATTERN           %s\n", d.val_files.c_str());
    printf("  --batch-size N                %zu   rows per micro-batch\n", d.batch_size);
    printf("  --seq-len N                   %zu\n", d.seq_len);
    printf("  --tokens-per-step N           %zu   all GPUs together\n", d.tokens_per_step);
    printf("  --linear-bias 0|1             %d   the linears' biases (the LayerNorms keep theirs); train_bf16, train_8bit\n",
           d.linear_bias ? 1 : 0);
    printf("  --steps N                     one pass over the train files\n");
    printf("  --max-learning-rate X         %g\n", d.max_learning_rate);
    printf("  --min-learning-rate X         %g\n", d.min_learning_rate);
    printf("  --warmup-steps N              %d\n", d.warmup_steps);
    printf("  --weight-decay X              %g\n", d.weight_decay);
    printf("  --max-norm X                  %g   0: no clipping\n", d.max_norm);
    printf("  --val-every N                 %d\n", d.val_every);
    printf("  --val-batches N               %d   on each GPU\n", d.val_batches);
    printf("  --seed N                      %llu\n", d.seed);
    printf("  --nccl-id-file PATH           /tmp/gpt2_nccl_id_<TORCHELASTIC_RUN_ID>.bin; several GPUs only\n");
    printf("  --progress-every N            %d   a line with the time left\n", d.progress_every);
    printf("  --checkpoint-dir DIR          none: no checkpoints (train_8bit: none yet)\n");
    printf("  --checkpoint-every N          %d\n", d.checkpoint_every);
    printf("  --keep-checkpoints N          %d   the newest; older ones are deleted\n", d.keep_checkpoints);
    printf("  --resume PATH                 a checkpoint dir (its latest) or one step folder of it\n");
    exit(exit_code);
}

/// Reads the --flag value pairs into a TrainConfig, starting from the defaults; stops on a flag it does not
/// know or a flag with no value.
///
///   argc, argv   main's
///   defaults     the trainer's defaults (a model of another size: its seq_len and rows); not given: TrainConfig's
inline TrainConfig parse_args(int argc, char** argv, const TrainConfig& defaults) {
    TrainConfig c = defaults;
    for (int i = 1; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") print_usage_and_stop(argv[0], EXIT_SUCCESS, defaults);
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s needs a value\n", flag.c_str());
            print_usage_and_stop(argv[0], EXIT_FAILURE, defaults);
        }
        const char* value = argv[++i];
        if (flag == "--train-files") c.train_files = value;
        else if (flag == "--val-files") c.val_files = value;
        else if (flag == "--batch-size") c.batch_size = (size_t)atoll(value);
        else if (flag == "--seq-len") c.seq_len = (size_t)atoll(value);
        else if (flag == "--tokens-per-step") c.tokens_per_step = (size_t)atoll(value);
        else if (flag == "--linear-bias") c.linear_bias = atoi(value) != 0;
        else if (flag == "--steps") c.steps = atoi(value);
        else if (flag == "--max-learning-rate") c.max_learning_rate = (float)atof(value);
        else if (flag == "--min-learning-rate") c.min_learning_rate = (float)atof(value);
        else if (flag == "--warmup-steps") c.warmup_steps = atoi(value);
        else if (flag == "--weight-decay") c.weight_decay = (float)atof(value);
        else if (flag == "--max-norm") c.max_norm = (float)atof(value);
        else if (flag == "--val-every") c.val_every = atoi(value);
        else if (flag == "--val-batches") c.val_batches = atoi(value);
        else if (flag == "--seed") c.seed = strtoull(value, nullptr, 10);
        else if (flag == "--nccl-id-file") c.nccl_id_file = value;
        else if (flag == "--progress-every") c.progress_every = atoi(value);
        else if (flag == "--checkpoint-dir") c.checkpoint_dir = value;
        else if (flag == "--checkpoint-every") c.checkpoint_every = atoi(value);
        else if (flag == "--keep-checkpoints") c.keep_checkpoints = atoi(value);
        else if (flag == "--resume") c.resume = value;
        else {
            fprintf(stderr, "error: no flag %s\n", flag.c_str());
            print_usage_and_stop(argv[0], EXIT_FAILURE, defaults);
        }
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

/// Gives the learning rate of step `step` (0 for the first): a straight line up over the warmup steps, from
/// max / warmup_steps to max, then half a cosine from max down to min at the last step. With the defaults and
/// 10B train tokens (19,073 steps):
///
///     step       0          699      700      9886            19072
///     lr         8.6e-7     6e-4     6e-4     3e-4 (half)     0
///
///   step   0 .. c.steps - 1
///   c      the run's settings
inline float learning_rate_at(int step, const TrainConfig& c) {
    if (step < c.warmup_steps) return c.max_learning_rate * (float)(step + 1) / (float)c.warmup_steps;
    int decay_steps = c.steps - 1 - c.warmup_steps;                    // from the end of warmup to the last step
    float progress = decay_steps > 0 ? (float)(step - c.warmup_steps) / (float)decay_steps : 1.0f;   // 0 .. 1
    float cosine = 0.5f * (1.0f + cosf(3.14159265f * progress));       // 1 .. 0
    return c.min_learning_rate + cosine * (c.max_learning_rate - c.min_learning_rate);
}

/// Gives AdamW's settings for this run: the first step's learning rate, the weight decay and the clipping.
///
///   c   the run's settings
inline AdamWConfig adamw_config_of(const TrainConfig& c) {
    AdamWConfig adamw_config;
    adamw_config.learning_rate = learning_rate_at(0, c);
    adamw_config.weight_decay = c.weight_decay;
    adamw_config.max_norm = c.max_norm;
    return adamw_config;
}

// ── the run's size ────────────────────────────────────────────────────────────

/// Gives how many micro-batches of {batch_size, seq_len} each GPU runs a step: tokens_per_step / (GPUs x B x T),
/// 4 on 8 GPUs, 32 on one. Stops if it does not divide.
///
///   c             the run's settings
///   distributed   for the number of GPUs
inline int micro_batches_of(const TrainConfig& c, const Distributed& distributed) {
    size_t tokens_per_round = (size_t)distributed.world_size() * c.batch_size * c.seq_len;   // one on every GPU
    if (c.tokens_per_step % tokens_per_round != 0) {
        print0(distributed, "error: --tokens-per-step %zu is not a multiple of %d GPU(s) x %zu x %zu = %zu\n",
               c.tokens_per_step, distributed.world_size(), c.batch_size, c.seq_len, tokens_per_round);
        exit(EXIT_FAILURE);
    }
    return (int)(c.tokens_per_step / tokens_per_round);
}

/// Sets c.steps to one pass over the train files when it is -1; stops if the files are still being written
/// (one pass is not known yet) or hold less than one step.
///
///   c              c.steps changed
///   train_loader   the train files
///   distributed    for printing on rank 0
inline void set_steps(TrainConfig& c, const DataLoader& train_loader, const Distributed& distributed) {
    if (c.steps == -1) {
        if (train_loader.more_files_coming()) {
            print0(distributed, "error: the train files are still being written (data/fineweb.py), so one pass is "
                                "not known yet: pass --steps (19,073 for 10B tokens)\n");
            exit(EXIT_FAILURE);
        }
        c.steps = (int)(train_loader.total_tokens() / c.tokens_per_step);
    }
    if (c.steps < 1) {
        print0(distributed, "error: the train files hold %zu tokens, less than one step of %zu\n",
               train_loader.total_tokens(), c.tokens_per_step);
        exit(EXIT_FAILURE);
    }
}

// ── the params and grads ──────────────────────────────────────────────────────

/// Gives the model's params as views of the AdamW's buffer: the model reads them, the AdamW updates them.
///
///   optimizer   any of the three AdamWs
///   config      the sizes
inline GPT2Params params_of(const Optimizer& optimizer, const GPT2Config& config) {
    std::vector<Tensor> tensors;
    for (int k = 0; k < optimizer.num_params(); k++) tensors.push_back(optimizer.param(k));
    return GPT2Params(tensors, config);
}

/// Gives the params' gradients as views of the AdamW's buffer: the backward adds onto them.
///
///   optimizer, config   as in params_of
inline GPT2Params grads_of(const Optimizer& optimizer, const GPT2Config& config) {
    std::vector<Tensor> tensors;
    for (int k = 0; k < optimizer.num_params(); k++) tensors.push_back(optimizer.grad(k));
    return GPT2Params(tensors, config);
}

// ── printing ──────────────────────────────────────────────────────────────────

/// Prints like printf, on rank 0 only, and flushes: the other ranks train without a word.
///
///   distributed        for the rank
///   format, ...        as printf's
inline void print0(const Distributed& distributed, const char* format, ...) {
    if (distributed.rank() != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

/// Prints what this run is before it starts: the model, the data, the steps, the learning rate, the GPU memory
/// and the checkpoints.
///
///   model_name                 "float", "bfloat16" or "8 bits", for the first line
///   c                          the run's settings
///   optimizer                  for the number of parameters
///   micro_batches              per GPU, per step
///   train_loader, val_loader   for their tokens
///   distributed                for the GPUs and printing on rank 0
inline void print_run(const char* model_name, const TrainConfig& c, const Optimizer& optimizer, int micro_batches,
                      const DataLoader& train_loader, const DataLoader& val_loader, const Distributed& distributed) {
    size_t free_bytes;
    size_t total_bytes;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    print0(distributed, "GPT-2 small: %zu parameters, %s, on %d GPU(s)\n", optimizer.all_params().numel(), model_name,
           distributed.world_size());
    print0(distributed, "train: %d file(s), %zu tokens%s; val: %zu tokens, %d batches of it on each GPU each time\n",
           train_loader.num_files(), train_loader.total_tokens(),
           train_loader.more_files_coming() ? " so far (more being written)" : "", val_loader.total_tokens(),
           c.val_batches);
    print0(distributed, "%d steps of %d GPU(s) x %d x {%zu, %zu} = %zu tokens: %zu tokens, %.2f epochs\n", c.steps,
           distributed.world_size(), micro_batches, c.batch_size, c.seq_len, c.tokens_per_step,
           c.tokens_per_step * c.steps, (double)c.tokens_per_step * c.steps / train_loader.total_tokens());
    print0(distributed, "learning rate: %g, warmup %d steps, cosine down to %g; weight decay %g, clip at %g\n",
           c.max_learning_rate, c.warmup_steps, c.min_learning_rate, c.weight_decay, c.max_norm);
    print0(distributed, "GPU memory, rank 0: %.1f GB used of %.1f GB\n", (total_bytes - free_bytes) / 1e9,
           total_bytes / 1e9);
    if (c.checkpoint_dir.empty()) print0(distributed, "checkpoints: none\n");
    else print0(distributed, "checkpoints: every %d steps into %s, the newest %d kept\n", c.checkpoint_every,
                c.checkpoint_dir.c_str(), c.keep_checkpoints);
}

/// Prints a progress line every c.progress_every steps (not after the last): the time so far, the time left at
/// this run's pace (val losses and checkpoints included), and the clock time it will be done.
///
///   steps_done, first_step   where the run is, and where this process started (0, or the checkpoint's step)
///   c                        the run's settings
///   seconds_so_far           since this process's first step
///   distributed              for printing on rank 0
inline void print_progress(int steps_done, int first_step, const TrainConfig& c, double seconds_so_far,
                           const Distributed& distributed) {
    if (steps_done % c.progress_every != 0 || steps_done >= c.steps) return;
    double left = seconds_so_far / (steps_done - first_step) * (c.steps - steps_done);
    print0(distributed, "progress: step %d/%d (%.1f%%) | %s so far | %s left | done ~%s\n", steps_done, c.steps,
           100.0 * steps_done / c.steps, duration_text(seconds_so_far).c_str(), duration_text(left).c_str(),
           clock_text_in(left).c_str());
}

/// Gives a length of time in the unit that reads best: "45 s", "12.5 min", "6.4 h", "9.9 days".
///
///   seconds   the length
inline std::string duration_text(double seconds) {
    char text[32];
    if (seconds < 90) snprintf(text, sizeof(text), "%.0f s", seconds);
    else if (seconds < 90 * 60) snprintf(text, sizeof(text), "%.1f min", seconds / 60);
    else if (seconds < 48 * 3600) snprintf(text, sizeof(text), "%.1f h", seconds / 3600);
    else snprintf(text, sizeof(text), "%.1f days", seconds / 86400);
    return text;
}

/// Gives the clock time `seconds` from now on this machine, as "Nov 12 14:20": when the run will be done.
///
///   seconds   from now
inline std::string clock_text_in(double seconds) {
    std::time_t then = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() +
                                                            std::chrono::seconds((long long)seconds));
    char text[32];
    std::strftime(text, sizeof(text), "%b %d %H:%M", std::localtime(&then));
    return text;
}

/// Gives the seconds from `started` to now.
///
///   started   a steady_clock::now() taken earlier
inline double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

#endif // TRAINING_CUH
