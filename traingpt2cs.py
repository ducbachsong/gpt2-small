"""GPT-2 training run on FineWeb-Edu: Python feeds the data, C# trains the model.

    python traingpt2cs.py          # on Colab: !bash colab_setup.sh, then !python traingpt2cs.py

The same run as traingpt2.py, without a Python model library: GPT-2, AdamW
and the training loop are written by hand in C# (csharp/Gpt2Trainer, on
TorchSharp, the .NET bindings of LibTorch). Everything else is the Python
modules in common/, unchanged:

    ParquetPool   downloads the parquet files into RAM                  Python
    TokenPool     tokenizes them into (SHARD_ROWS, SEQUENCE_LENGTH)     Python
    shard files   one batch per file in <run>/pool, SHARDS_AHEAD ahead  Python -> C#
    Gpt2Trainer   trains GPT-2, prints one line per step                C#
    CsvWriter     turns those lines into log.csv                        Python
    csvexplorer   draws the loss chart at the end                       Python

The two sides share only files and the trainer's console, as in lab06: no
sockets, no bindings. The first shard is held back as heldout.bin and never
trained on: its first EVAL_ROWS rows give val_loss. The model trains in
fp32 and is not saved: at the end it continues PROMPTS into samples.txt, and
the run is over. Everything goes to OUTPUT_DIR. Needs .NET 8 (colab_setup.sh installs it), a GPU to finish in
reasonable time, and the network.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

os.environ.setdefault("DOTNET_CLI_TELEMETRY_OPTOUT", "1")
os.environ.setdefault("DOTNET_NOLOGO", "1")

# The modules live in common/, next to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

from csvexplorer import chart_image, line_chart, load_csvs, open_dashboard, plot_data
from csvwriter import CsvWriter, full_path
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

# ── the data: 14 files, ~2.0 GB each, ~10B tokens in all ────────────────────
DATASET = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = "sample/10BT/*.parquet"
PARQUET_RAM_LIMIT = 3 * GB      # sized for Colab's 12.7 GB, as in traingpt2.py
TOKENIZE_THREAD_COUNT = 2
TOKEN_RAM_LIMIT = 1 * GB
TEXT_COLUMN = "text"
TOKENIZER = "gpt2"              # the C# model's vocabulary is GPT-2's 50,257 ids
SEQUENCE_LENGTH = 1024
SHARD_ROWS = 256                # rows per shard file: 256 x 1024 x 2 bytes = 512 KB
SHARDS_AHEAD = 16               # shard files kept ready for the trainer

# ── the model: GPT-2 small, from scratch ────────────────────────────────────
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# ── the optimiser: 32,768 tokens per step, as in traingpt2.py ───────────────
TOKENS_PER_STEP = 32768         # MICRO_BATCH x SEQUENCE_LENGTH x grad accumulation
MICRO_BATCH = None              # rows per forward pass; None picks from GPU memory
LEARNING_RATE = 6e-4            # peak, after WARMUP_STEPS
MIN_LEARNING_RATE = 6e-5        # where the cosine ends, at MAX_STEPS
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_STEPS = 3000                # ~98M tokens; ~305,000 steps is the whole 10B

# ── watching and saving ─────────────────────────────────────────────────────
EVAL_ROWS = 80                  # held-out rows (x 1024 tokens) for val_loss
EVAL_EVERY = 250
PRINT_EVERY = 25                # the trainer reads the GPU once per this many steps
OUTPUT_DIR = "runs"
RUN_NAME = "fineweb-edu-gpt2cs-{time}"  # {time} is when this run started
MONITOR = False                 # True opens the live csvexplorer page; not on Colab
PROMPTS = ["The theory of evolution explains",
           "In mathematics, a prime number is",
           "Photosynthesis is the process"]

# ── the machine ─────────────────────────────────────────────────────────────
SEED = 0
DOTNET = "dotnet"               # found on PATH, else in ~/.dotnet
CSHARP_PROJECT = os.path.join(HERE, "csharp", "Gpt2Trainer")

LOG_COLUMNS = ("step", "tokens", "loss", "val_loss", "lr", "grad_norm",
               "tokens_per_second")
SHARD_MAGIC = 0x32545047        # "GPT2"; must match csharp/Gpt2Trainer/Shards.cs


# ── the machine, found out ──────────────────────────────────────────────────
def gpu_info():
    """(compute capability, memory in GB) of GPU 0, or None without one."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    capability, memory_mb = output.splitlines()[0].split(",")
    return float(capability), float(memory_mb) / 1024


def pick_micro_batch(gpu):
    """Rows per forward pass that fit: 4 on a 16 GB T4, 8 on a 40 GB A100, 16 on 80 GB.

    A power of two, so it divides TOKENS_PER_STEP. Raise MICRO_BATCH by hand
    if nvidia-smi shows room to spare: fewer, larger passes are faster.
    """
    if MICRO_BATCH:
        return MICRO_BATCH
    if gpu is None:
        return 4
    return 16 if gpu[1] > 70 else 8 if gpu[1] > 35 else 4


def find_dotnet():
    for candidate in (DOTNET, os.path.expanduser("~/.dotnet/dotnet")):
        if shutil.which(candidate):
            return shutil.which(candidate)
    raise RuntimeError("no .NET SDK found: run colab_setup.sh, or install .NET 8")


def build_trainer(dotnet, gpu):
    """Builds csharp/Gpt2Trainer against the right LibTorch; returns its dll."""
    backend = ("cpu" if gpu is None else
               "cuda-windows" if os.name == "nt" else "cuda-linux")
    output = os.path.join(CSHARP_PROJECT, "bin", backend)
    print(f"building the C# trainer ({backend}); the first build downloads LibTorch", flush=True)
    subprocess.run([dotnet, "build", CSHARP_PROJECT, "-c", "Release", "-v", "quiet", "-nologo",
                    f"-p:TorchBackend={backend}", "-o", output], check=True)
    return os.path.join(output, "Gpt2Trainer.dll")


# ── Python -> C#: token shards ──────────────────────────────────────────────
def write_shard(path, batch):
    """One (rows, SEQUENCE_LENGTH) batch as a shard; renamed into place when complete."""
    header = np.array([SHARD_MAGIC, batch.shape[0], batch.shape[1]], dtype=np.uint32)
    with open(path + ".tmp", "wb") as shard:
        shard.write(header.tobytes())
        shard.write(batch.astype(np.uint16).tobytes())      # GPT-2 ids < 65,536
    os.replace(path + ".tmp", path)


def read_consumed(pool_dir):
    """The highest shard index the trainer has already loaded, or -1."""
    try:
        with open(os.path.join(pool_dir, "consumed.txt"), encoding="utf-8") as consumed:
            return int(consumed.read().strip())
    except (OSError, ValueError):
        return -1


def feed_trainer(token_pool, pool_dir, errors):
    """Thread: keeps SHARDS_AHEAD shards ready and deletes the ones the trainer took.

    heldout.bin first, then shard_000000.bin, shard_000001.bin, ... Writes
    STOP when the dataset runs out; returns when the trainer writes STOP.
    """
    stop_path = os.path.join(pool_dir, "STOP")
    try:
        write_shard(os.path.join(pool_dir, "heldout.bin"), token_pool.get_token_batch())
        index = deleted = 0
        while not os.path.exists(stop_path):
            consumed = read_consumed(pool_dir)
            for old in range(deleted, consumed + 1):
                try:
                    os.remove(os.path.join(pool_dir, f"shard_{old:06d}.bin"))
                except OSError:
                    pass
            deleted = max(deleted, consumed + 1)
            if index - consumed - 1 >= SHARDS_AHEAD:
                time.sleep(0.05)
                continue
            try:
                batch = token_pool.get_token_batch(timeout=1.0)
            except TimeoutError:
                continue                    # look for STOP again, then keep waiting
            if batch is None:
                print("the dataset ran out; the trainer finishes what it has", flush=True)
                with open(stop_path, "w", encoding="utf-8") as stop:
                    stop.write("no more data\n")
                return
            write_shard(os.path.join(pool_dir, f"shard_{index:06d}.bin"), batch)
            index += 1
    except Exception as error:              # noqa: BLE001 — reported by the main thread
        errors.append(error)
        with open(stop_path, "w", encoding="utf-8") as stop:
            stop.write(f"feeder failed: {error!r}\n")


# ── C# -> Python: the trainer's console ─────────────────────────────────────
LINE = re.compile(r"^\[(\w+)\] (.*)$")


def parse_line(line):
    """ "[Step] step=25 loss=7.01" -> ("Step", {"step": "25", "loss": "7.01"}), else None."""
    match = LINE.match(line)
    if not match:
        return None
    fields = dict(part.split("=", 1) for part in match.group(2).split() if "=" in part)
    return match.group(1), fields


def trainer_arguments(pool_dir, gpu, tokenizer):
    micro_batch = pick_micro_batch(gpu)
    grad_accum = max(1, TOKENS_PER_STEP // (micro_batch * SEQUENCE_LENGTH))
    prompts = ";".join(",".join(str(i) for i in tokenizer.encode(prompt)) for prompt in PROMPTS)
    settings = {
        "data": pool_dir,
        "n-layer": N_LAYER, "n-head": N_HEAD, "n-embd": N_EMBD,
        "sequence-length": SEQUENCE_LENGTH, "micro-batch": micro_batch,
        "grad-accum-steps": grad_accum, "learning-rate": LEARNING_RATE,
        "min-learning-rate": MIN_LEARNING_RATE, "warmup-steps": WARMUP_STEPS,
        "weight-decay": WEIGHT_DECAY, "grad-clip": GRAD_CLIP, "max-steps": MAX_STEPS,
        "eval-rows": EVAL_ROWS, "eval-every": EVAL_EVERY, "print-every": PRINT_EVERY,
        "prompts": prompts, "max-new-tokens": 60, "temperature": 0.8, "top-k": 50,
        "device": "cpu" if gpu is None else "cuda", "seed": SEED,
    }
    arguments = []
    for name, value in settings.items():
        if value != "":
            arguments += [f"--{name}", str(value)]
    return arguments


# ── the run ─────────────────────────────────────────────────────────────────
def train(run_folder, dotnet, trainer_dll, gpu):
    """Runs the feeder thread and the C# trainer until the trainer is done.

    Out: (steps taken, final val_loss, log path, samples as text)
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    pool_dir = os.path.join(run_folder, "pool")
    shutil.rmtree(pool_dir, ignore_errors=True)
    os.makedirs(pool_dir)
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                               ram_limit=PARQUET_RAM_LIMIT)
    token_pool = TokenPool(parquet_pool, tokenizer=TOKENIZER, text_column=TEXT_COLUMN,
                           sequence_length=SEQUENCE_LENGTH, batch_size=SHARD_ROWS,
                           token_ram_limit=TOKEN_RAM_LIMIT,
                           tokenize_thread_count=TOKENIZE_THREAD_COUNT)
    errors = []
    feeder = threading.Thread(target=feed_trainer, args=(token_pool, pool_dir, errors),
                              name="feed-trainer", daemon=True)
    log_path = os.path.join(run_folder, "log.csv")
    log = trainer = None
    step, val_loss, samples = 0, float("nan"), []
    tokens_per_step = TOKENS_PER_STEP       # replaced by what the trainer reports
    try:
        feeder.start()
        trainer = subprocess.Popen(
            [dotnet, trainer_dll, *trainer_arguments(pool_dir, gpu, tokenizer)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace")
        while True:
            try:
                for line in trainer.stdout:
                    line = line.rstrip()
                    print(line, flush=True)
                    parsed = parse_line(line)
                    if parsed is None:
                        continue
                    kind, fields = parsed
                    if kind == "Config":
                        tokens_per_step = int(fields["tokens_per_step"])
                    elif kind == "Step":
                        row = {name: float(fields[name]) for name in LOG_COLUMNS if name in fields}
                        step = int(row["step"])
                        log.write(row)
                    elif kind == "Eval":
                        step, val_loss = int(fields["step"]), float(fields["val_loss"])
                        row = {"step": step, "tokens": step * tokens_per_step, "val_loss": val_loss}
                        if log is None:
                            # Every column with the first row, so the file is never
                            # rewritten for a new column while the live page reads it.
                            log = CsvWriter(log_path, mode="overwrite", float_format=".6g")
                            row = {**dict.fromkeys(LOG_COLUMNS), **row}
                            log.write(row)
                            log.flush()
                            if MONITOR:
                                open_dashboard(log_path, wait=False)
                        else:
                            log.write(row)
                    elif kind == "Sample":
                        ids = [int(i) for i in fields["ids"].split(",")]
                        samples.append(tokenizer.decode(ids))
                break
            except KeyboardInterrupt:
                # The trainer got the Ctrl+C too: it finishes its step, then samples.
                print("\nCtrl+C: waiting for the trainer to finish", flush=True)
        if trainer.wait() != 0:
            raise RuntimeError(f"the C# trainer failed (exit code {trainer.returncode})")
        if errors:
            raise RuntimeError("feeding the trainer failed") from errors[0]
        return step, val_loss, log_path, samples
    finally:
        if trainer is not None and trainer.poll() is None:
            trainer.kill()
        with open(os.path.join(pool_dir, "STOP"), "w", encoding="utf-8") as stop:
            stop.write("done\n")                # ends the feeder if it still runs
        feeder.join(timeout=10)
        if log is not None:
            log.close()
        close_token_pool()
        close_parquet_pool()
        shutil.rmtree(pool_dir, ignore_errors=True)


def save_loss_chart(log_path, run_folder):
    """loss and val_loss against step, into loss.png."""
    load_csvs(log_path)
    chart = line_chart(plot_data("step", ["loss", "val_loss"], smoothing=20),
                       title="GPT-2 (C#) on FineWeb-Edu", y_title="loss")
    path = os.path.join(run_folder, "loss.png")
    try:
        with open(path, "wb") as image_file:
            image_file.write(chart_image(chart, "png"))
        print(f"loss chart: {path}")
    except Exception as error:          # PNG export needs vl-convert-python
        print(f"no loss chart ({error!r}); pip install vl-convert-python")


def save_samples(samples, run_folder):
    """The trained model's continuation of each prompt, into samples.txt."""
    text = "\n\n".join(samples)
    with open(os.path.join(run_folder, "samples.txt"), "w", encoding="utf-8") as samples_file:
        samples_file.write(text + "\n")
    print(f"\n── samples ──\n{text}\n")


def main():
    run_folder = full_path(os.path.join(OUTPUT_DIR, RUN_NAME))
    os.makedirs(run_folder, exist_ok=True)
    gpu = gpu_info()
    print("GPU: " + (f"compute capability {gpu[0]}, {gpu[1]:.0f} GB" if gpu else "none, training on the CPU"))
    dotnet = find_dotnet()
    trainer_dll = build_trainer(dotnet, gpu)
    started = time.perf_counter()
    steps, val_loss, log_path, samples = train(run_folder, dotnet, trainer_dll, gpu)
    print(f"\n{steps} steps in {(time.perf_counter() - started) / 3600:.2f} h, "
          f"final val_loss {val_loss:.4f}")
    print(f"log: {log_path}")
    save_loss_chart(log_path, run_folder)
    save_samples(samples, run_folder)


if __name__ == "__main__":
    main()
