"""GPT-2 training run on FineWeb-Edu, built from the common modules.

    python traingpt2.py          # on Colab: !python traingpt2.py

Not a check: it trains a GPT-2 small (124M parameters) on
FineWeb-Edu's 10B-token sample. The whole pipe is the modules in common/:

    ParquetPool   downloads the parquet files into RAM
    TokenPool     tokenizes them into (BATCH_SIZE, SEQUENCE_LENGTH) batches
    GPT-2         trains on each batch
    CsvWriter     logs every step, reading the GPU only once a flush
    csvexplorer   draws the loss chart at the end (or the live page, MONITOR)

The first EVAL_BATCHES batches are held back and never trained on: the loss
on them is val_loss. Checkpoints are saved every SAVE_EVERY steps, at the
end, and on Ctrl+C. Everything goes to OUTPUT_DIR. Needs a GPU to finish in
reasonable time, and the network.

"""
import json
import math
import os
import sys
import time

# Before torch and the hub are imported, or they are ignored.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import torch

# The modules live in common/, next to this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

from csvexplorer import chart_image, line_chart, load_csvs, open_dashboard, plot_data
from csvwriter import CsvWriter, full_path
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

# ── the data: 14 files, ~2.0 GB each, ~10B tokens in all ────────────────────
DATASET = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = "sample/10BT/*.parquet"
# RAM on Colab's 12.7 GB: the pool holds PARQUET_RAM_LIMIT // 2 GB = 1 file,
# and each tokenize thread holds the file it is working on besides, so
# (1 + TOKENIZE_THREAD_COUNT) x 2 GB + TOKEN_RAM_LIMIT = ~7 GB at most.
PARQUET_RAM_LIMIT = 3 * GB
TOKENIZE_THREAD_COUNT = 2
TOKEN_RAM_LIMIT = 1 * GB
TEXT_COLUMN = "text"
TOKENIZER = "gpt2"
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 4                  # fits a 15 GB T4; 8 with GRAD_ACCUM_STEPS = 4 on an A100

# ── the model: GPT-2 small, from scratch ────────────────────────────────────
MODEL_NAME = ""                 # "gpt2" starts from OpenAI's weights instead
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# ── the optimiser: 4 x 1024 x 8 = 32,768 tokens per step ────────────────────
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 6e-4            # peak, after WARMUP_STEPS
MIN_LEARNING_RATE = 6e-5        # where the cosine ends, at MAX_STEPS
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_STEPS = 3000                # ~98M tokens

# ── watching and saving ─────────────────────────────────────────────────────
EVAL_BATCHES = 20
EVAL_EVERY = 250
PRINT_EVERY = 25
SAVE_EVERY = 500
OUTPUT_DIR = "runs"
RUN_NAME = "fineweb-edu-gpt2-{time}"    # {time} is when this run started
MONITOR = False                 # True opens the live csvexplorer page; not on Colab
PROMPTS = ["The theory of evolution explains",
           "In mathematics, a prime number is",
           "Photosynthesis is the process"]

# ── the machine ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bf16 only where the hardware has it (A100, L4, ...: compute capability 8+).
# A T4 is 7.5: torch emulates bf16 there, slowly, so it gets fp16.
AMP_DTYPE = (None if DEVICE.type != "cuda" else
             torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
COMPILE = False                 # True is faster on A100 after a slow start
SEED = 0

LOG_COLUMNS = ("step", "tokens", "loss", "val_loss", "lr", "grad_norm",
               "tokens_per_second")
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH * GRAD_ACCUM_STEPS


# ── the pieces ──────────────────────────────────────────────────────────────
def learning_rate_at(step):
    """Linear warmup to LEARNING_RATE, then a cosine down to MIN_LEARNING_RATE."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS))
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * \
        0.5 * (1 + math.cos(math.pi * progress))


def on_device(batch):
    """A token batch from the pool as int64 ids on the GPU."""
    input_ids = torch.from_numpy(batch).long()
    if DEVICE.type == "cuda":
        return input_ids.pin_memory().to(DEVICE, non_blocking=True)
    return input_ids


def build_model(tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    if MODEL_NAME:
        return GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=SEQUENCE_LENGTH,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id))


def build_optimizer(model):
    """AdamW, with weight decay on the weight matrices only."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in parameters if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
              {"params": [p for p in parameters if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=LEARNING_RATE, betas=(0.9, 0.95),
                             fused=DEVICE.type == "cuda")


def evaluate(model, eval_batches):
    """Mean loss over the held-back batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_batches:
            input_ids = on_device(batch)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
                losses.append(model(input_ids=input_ids, labels=input_ids).loss.float())
    model.train()
    return torch.stack(losses).mean().item()


def save_checkpoint(model, tokenizer, step, folder):
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    with open(os.path.join(folder, "training.json"), "w", encoding="utf-8") as state:
        json.dump({"step": step, "tokens": step * TOKENS_PER_STEP,
                   "settings": {name: value for name, value in globals().items()
                                if name.isupper() and isinstance(value, (int, float, str))}},
                  state, indent=2)
    print(f"saved step {step} to {folder}", flush=True)
    return folder


def time_left(step, seconds_so_far):
    seconds = int(seconds_so_far / step * (MAX_STEPS - step))
    hours, minutes = seconds // 3600, seconds % 3600 // 60
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds % 60:02d}s"


# ── the run ─────────────────────────────────────────────────────────────────
def train(run_folder):
    """Train for MAX_STEPS steps, or until the data runs out or Ctrl+C.

    Out: (steps taken, final val_loss, log path, final checkpoint folder)
    """
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                               ram_limit=PARQUET_RAM_LIMIT)
    token_pool = TokenPool(parquet_pool, tokenizer=TOKENIZER, text_column=TEXT_COLUMN,
                           sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
                           token_ram_limit=TOKEN_RAM_LIMIT,
                           tokenize_thread_count=TOKENIZE_THREAD_COUNT)
    log_path = os.path.join(run_folder, "log.csv")
    log = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        eval_batches = [token_pool.get_token_batch() for _ in range(EVAL_BATCHES)]
        model = build_model(tokenizer).to(DEVICE)
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=AMP_DTYPE == torch.float16)
        step_model = torch.compile(model) if COMPILE else model
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters "
              f"on {DEVICE}{f', {AMP_DTYPE}' if AMP_DTYPE else ''}; "
              f"{TOKENS_PER_STEP:,} tokens per step", flush=True)

        # Every column with the first row, so the file is never rewritten
        # for a new column while the live page reads it.
        val_loss = evaluate(step_model, eval_batches)
        log = CsvWriter(log_path, mode="overwrite", float_format=".6g")
        log.write({**dict.fromkeys(LOG_COLUMNS), "step": 0, "tokens": 0,
                   "val_loss": val_loss, "lr": learning_rate_at(0)})
        log.flush()
        print(f"step      0 | val_loss {val_loss:.4f}", flush=True)
        if MONITOR:
            open_dashboard(log_path, wait=False)

        step = evaluated_at = 0
        run_started = window_started = time.perf_counter()
        window_steps = 0
        step_model.train()
        try:
            while step < MAX_STEPS:
                learning_rate = learning_rate_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                # The loss stays a tensor on the GPU: nothing here waits for
                # it, so the GPU runs ahead while the CPU queues the next step.
                step_loss = torch.zeros((), device=DEVICE)
                for _ in range(GRAD_ACCUM_STEPS):
                    batch = token_pool.get_token_batch()
                    if batch is None:
                        raise StopIteration
                    input_ids = on_device(batch)
                    with torch.autocast(DEVICE.type, dtype=AMP_DTYPE,
                                        enabled=AMP_DTYPE is not None):
                        loss = step_model(input_ids=input_ids, labels=input_ids).loss
                    loss = loss / GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                    step_loss += loss.detach()

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_steps += 1

                row = {"step": step, "tokens": step * TOKENS_PER_STEP,
                       "loss": step_loss, "lr": learning_rate, "grad_norm": grad_norm}
                if step % PRINT_EVERY == 0:
                    now = time.perf_counter()
                    loss_now = step_loss.item()     # waits for the GPU, once per print
                    row["tokens_per_second"] = window_steps * TOKENS_PER_STEP / (now - window_started)
                    print(f"step {step:>6} | loss {loss_now:.4f} | lr {learning_rate:.2e} | "
                          f"{row['tokens_per_second']:,.0f} tok/s | "
                          f"{time_left(step, now - run_started)} left", flush=True)
                    window_started, window_steps = now, 0
                if step % EVAL_EVERY == 0:
                    row["val_loss"] = val_loss = evaluate(step_model, eval_batches)
                    evaluated_at = step
                    print(f"step {step:>6} | val_loss {val_loss:.4f}", flush=True)
                log.write(row)
                if step % SAVE_EVERY == 0:
                    save_checkpoint(model, tokenizer, step,
                                    os.path.join(run_folder, "checkpoints", f"step-{step}"))
        except StopIteration:
            print(f"the dataset ran out at step {step}", flush=True)
        except KeyboardInterrupt:
            print(f"\nstopped by Ctrl+C at step {step}; saving it", flush=True)

        if evaluated_at != step:
            val_loss = evaluate(step_model, eval_batches)
            log.write(step=step, val_loss=val_loss)
            print(f"step {step:>6} | val_loss {val_loss:.4f}", flush=True)
        final = save_checkpoint(model, tokenizer, step,
                                os.path.join(run_folder, "checkpoints", "final"))
        return step, val_loss, log_path, final
    finally:
        if log is not None:
            log.close()
        close_token_pool()
        close_parquet_pool()


def save_loss_chart(log_path, run_folder):
    """loss and val_loss against step, into loss.png."""
    load_csvs(log_path)
    chart = line_chart(plot_data("step", ["loss", "val_loss"], smoothing=20),
                       title="GPT-2 on FineWeb-Edu", y_title="loss")
    path = os.path.join(run_folder, "loss.png")
    try:
        with open(path, "wb") as image_file:
            image_file.write(chart_image(chart, "png"))
        print(f"loss chart: {path}")
    except Exception as error:          # PNG export needs vl-convert-python
        print(f"no loss chart ({error!r}); pip install vl-convert-python")


def save_samples(checkpoint, run_folder):
    """The trained model continues each prompt, into samples.txt."""
    from transformers import AutoTokenizer, GPT2LMHeadModel
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(DEVICE).eval()
    lines = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                                    top_k=50, temperature=0.8,
                                    pad_token_id=tokenizer.eos_token_id)
        lines.append(tokenizer.decode(output[0], skip_special_tokens=True))
    text = "\n\n".join(lines)
    with open(os.path.join(run_folder, "samples.txt"), "w", encoding="utf-8") as samples:
        samples.write(text + "\n")
    print(f"\n── samples ──\n{text}\n")


def main():
    run_folder = full_path(os.path.join(OUTPUT_DIR, RUN_NAME))
    os.makedirs(run_folder, exist_ok=True)
    started = time.perf_counter()
    steps, val_loss, log_path, checkpoint = train(run_folder)
    print(f"\n{steps} steps in {(time.perf_counter() - started) / 3600:.2f} h, "
          f"final val_loss {val_loss:.4f}")
    print(f"log:        {log_path}")
    print(f"checkpoint: {checkpoint}")
    save_loss_chart(log_path, run_folder)
    save_samples(checkpoint, run_folder)


if __name__ == "__main__":
    main()
