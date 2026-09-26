"""GPT-2 training run on FineWeb-Edu: Python feeds the data, C# trains the model.

    pip install -r requirements.txt && python traingpt2cs.py     # on Colab too

The same run as main's traingpt2.py, without a Python model library: GPT-2, AdamW
and the training loop are written by hand in C# (src/*.cs, on
TorchSharp, the .NET bindings of LibTorch). Everything else is the Python
modules in src/common/, unchanged:

    ParquetPool   downloads the parquet files into RAM                  Python
    TokenPool     tokenizes them into (BATCH_ROWS, SEQUENCE_LENGTH)     Python
    next_batch    token_pool.get_token_batch(), called by C#            Python -> C#
    Gpt2Trainer   trains GPT-2, reports one line per step               C#
    CsvWriter     turns those lines into log.csv                        Python
    csvexplorer   draws the loss chart at the end                       Python

One process: pythonnet loads the .NET runtime and the trainer dll into this
Python, and the trainer takes each batch straight from TokenPool, as
main's traingpt2.py does, with no files in between. It trains on a .NET thread, so
the tokenizer threads keep running. The first batch is held out and never
trained on: its first EVAL_ROWS rows give val_loss. The model trains in
fp32 and is not saved: at the end it continues PROMPTS into samples.txt, and
the run is over. Everything goes to OUTPUT_DIR. Needs the .NET 8 SDK (installed into ~/.dotnet when missing, as on Colab),
pythonnet (requirements.txt), a GPU to finish in
reasonable time, and the network.
"""
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import urllib.request

# TorchSharp brings its own LibTorch into this process; PyTorch's must stay out,
# or the two clash. transformers imports torch when it can; this makes it use
# only its tokenizers, which is all TokenPool needs.
sys.modules["torch"] = None

import numpy as np

os.environ.setdefault("DOTNET_CLI_TELEMETRY_OPTOUT", "1")
os.environ.setdefault("DOTNET_NOLOGO", "1")

# The modules live in src/common/.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src", "common"))

from csvexplorer import chart_image, line_chart, load_csvs, open_dashboard, plot_data
from csvwriter import CsvWriter, full_path
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

# ── the data: 14 files, ~2.0 GB each, ~10B tokens in all ────────────────────
DATASET = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = "sample/10BT/*.parquet"
PARQUET_RAM_LIMIT = 3 * GB      # sized for Colab's 12.7 GB, as on main
TOKENIZE_THREAD_COUNT = 2
TOKEN_RAM_LIMIT = 1 * GB
TEXT_COLUMN = "text"
TOKENIZER = "gpt2"              # the C# model's vocabulary is GPT-2's 50,257 ids
SEQUENCE_LENGTH = 1024
BATCH_ROWS = 256                # rows per TokenPool batch: one copy to the GPU per batch

# ── the model: GPT-2 small, from scratch ────────────────────────────────────
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# ── the optimiser: 32,768 tokens per step, as on main ────────────────────────
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
DOTNET = "dotnet"               # found on PATH, else in ~/.dotnet, else installed there
DOTNET_CHANNEL = "8.0"
CSHARP_PROJECT = os.path.join(HERE, "Gpt2Trainer.csproj")

LOG_COLUMNS = ("step", "tokens", "loss", "val_loss", "lr", "grad_norm",
               "tokens_per_second")


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
    """The dotnet executable; installs the .NET 8 SDK into ~/.dotnet when there is none."""
    install_dir = os.path.expanduser("~/.dotnet")
    local_dotnet = os.path.join(install_dir, "dotnet.exe" if os.name == "nt" else "dotnet")
    for candidate in (DOTNET, local_dotnet):
        if shutil.which(candidate):
            return shutil.which(candidate)
    print(f"no .NET SDK found: installing .NET {DOTNET_CHANNEL} into {install_dir}", flush=True)
    script = "dotnet-install.ps1" if os.name == "nt" else "dotnet-install.sh"
    script_path = os.path.join(install_dir, script)
    os.makedirs(install_dir, exist_ok=True)
    urllib.request.urlretrieve(f"https://dot.net/v1/{script}", script_path)
    if os.name == "nt":
        command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path,
                   "-Channel", DOTNET_CHANNEL, "-InstallDir", install_dir]
    else:
        command = ["bash", script_path, "--channel", DOTNET_CHANNEL, "--install-dir", install_dir]
    # On Windows a virus scanner can hold a file the installer just unpacked; a
    # second go skips what is already there and finishes.
    for _ in range(3):
        if subprocess.run(command).returncode == 0:
            return local_dotnet
        print("the .NET install failed; trying again", flush=True)
    raise RuntimeError(f"could not install .NET {DOTNET_CHANNEL}: delete {install_dir} "
                       f"and run again, or install the .NET {DOTNET_CHANNEL} SDK yourself")


def build_trainer(dotnet, gpu):
    """Builds Gpt2Trainer.csproj against the right LibTorch; returns its folder."""
    backend = ("cpu" if gpu is None else
               "cuda-windows" if os.name == "nt" else "cuda-linux")
    runtime = "win-x64" if os.name == "nt" else "linux-x64"   # LibTorch's files next to the dll
    trainer_dir = os.path.join(HERE, "bin", backend)
    print(f"building the C# trainer ({backend}); the first build downloads LibTorch", flush=True)
    subprocess.run([dotnet, "build", CSHARP_PROJECT, "-c", "Release", "-v", "quiet", "-nologo",
                    f"-p:TorchBackend={backend}", "-r", runtime, "--no-self-contained",
                    "-o", trainer_dir], check=True)
    return trainer_dir


def load_trainer(dotnet, trainer_dir):
    """Starts .NET inside this process and loads the trainer dll; returns its C# namespace."""
    from pythonnet import load
    load("coreclr", runtime_config=os.path.join(trainer_dir, "Gpt2Trainer.runtimeconfig.json"),
         dotnet_root=os.path.dirname(os.path.realpath(dotnet)))
    import clr
    sys.path.append(trainer_dir)
    clr.AddReference("Gpt2Trainer")
    import Gpt2Trainer as csharp
    return csharp


# ── Python -> C#: token batches ─────────────────────────────────────────────
def make_next_batch(token_pool):
    """The trainer's NextBatch: the address of the next int32 batch, 0 at the end.

    C# copies a batch to the GPU before it asks again, so only the batch
    handed out last has to stay alive.
    """
    batch_in_use = None

    def next_batch():
        nonlocal batch_in_use
        batch = token_pool.get_token_batch()    # waits, without holding the GIL
        if batch is None:
            batch_in_use = None
            return 0
        batch_in_use = np.ascontiguousarray(batch, dtype=np.int32)
        if batch_in_use.shape != (BATCH_ROWS, SEQUENCE_LENGTH):
            raise ValueError(f"TokenPool gave a {batch_in_use.shape} batch, "
                             f"not ({BATCH_ROWS}, {SEQUENCE_LENGTH})")
        return batch_in_use.ctypes.data

    return next_batch


# ── C# -> Python: the trainer's report lines ───────────────────────────────
REPORT_LINE = re.compile(r"^\[(\w+)\] (.*)$")


def parse_report_line(line):
    """ "[Step] step=25 loss=7.01" -> ("Step", {"step": "25", "loss": "7.01"}), else None."""
    match = REPORT_LINE.match(line)
    if not match:
        return None
    fields = dict(part.split("=", 1) for part in match.group(2).split() if "=" in part)
    return match.group(1), fields


def trainer_arguments(gpu, tokenizer):
    micro_batch = pick_micro_batch(gpu)
    grad_accum = max(1, TOKENS_PER_STEP // (micro_batch * SEQUENCE_LENGTH))
    prompts = ";".join(",".join(str(token_id) for token_id in tokenizer.encode(prompt)) for prompt in PROMPTS)
    settings = {
        "batch-rows": BATCH_ROWS,
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
def train(run_folder, csharp, gpu):
    """Runs the C# trainer on TokenPool's batches until it is done.

    Out: (steps taken, final val_loss, log path, samples as text)
    """
    from System import Action, Array, String
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                               ram_limit=PARQUET_RAM_LIMIT)
    token_pool = TokenPool(parquet_pool, tokenizer=TOKENIZER, text_column=TEXT_COLUMN,
                           sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_ROWS,
                           token_ram_limit=TOKEN_RAM_LIMIT,
                           tokenize_thread_count=TOKENIZE_THREAD_COUNT)
    log_path = os.path.join(run_folder, "log.csv")
    log_writer = None
    step, val_loss, samples = 0, float("nan"), []
    tokens_per_step = TOKENS_PER_STEP       # replaced by what the trainer reports
    report_lines = queue.Queue()            # the trainer's thread puts, this one reads
    stop_requested = False
    try:
        trainer = csharp.Trainer.Start(Array[String](trainer_arguments(gpu, tokenizer)),
                                        csharp.NextBatch(make_next_batch(token_pool)),
                                        Action[String](report_lines.put))
        while True:
            try:
                try:
                    line = report_lines.get(timeout=0.2)
                except queue.Empty:
                    if trainer.Finished and report_lines.empty():
                        break
                    continue
                print(line, flush=True)
                parsed = parse_report_line(line)
                if parsed is None:
                    continue
                kind, fields = parsed
                if kind == "Config":
                    tokens_per_step = int(fields["tokens_per_step"])
                elif kind == "Step":
                    row = {name: float(fields[name]) for name in LOG_COLUMNS if name in fields}
                    step = int(row["step"])
                    log_writer.write(row)
                elif kind == "Eval":
                    step, val_loss = int(fields["step"]), float(fields["val_loss"])
                    row = {"step": step, "tokens": step * tokens_per_step, "val_loss": val_loss}
                    if log_writer is None:
                        # Every column with the first row, so the file is never
                        # rewritten for a new column while the live page reads it.
                        log_writer = CsvWriter(log_path, mode="overwrite", float_format=".6g")
                        row = {**dict.fromkeys(LOG_COLUMNS), **row}
                        log_writer.write(row)
                        log_writer.flush()
                        if MONITOR:
                            open_dashboard(log_path, wait=False)
                    else:
                        log_writer.write(row)
                elif kind == "Sample":
                    sample_ids = [int(token_id) for token_id in fields["ids"].split(",")]
                    samples.append(tokenizer.decode(sample_ids))
            except KeyboardInterrupt:
                if stop_requested:
                    raise                   # a second Ctrl+C ends the run at once
                # The trainer finishes its step, evaluates and samples; the lines keep coming.
                stop_requested = True
                trainer.RequestStop()
        if trainer.Error is not None:
            raise RuntimeError(f"the C# trainer failed:\n{trainer.Error}")
        return step, val_loss, log_path, samples
    finally:
        if log_writer is not None:
            log_writer.close()
        close_token_pool()                  # a trainer still waiting for a batch gets an error
        close_parquet_pool()


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
    csharp = load_trainer(dotnet, build_trainer(dotnet, gpu))
    started = time.perf_counter()
    steps, val_loss, log_path, samples = train(run_folder, csharp, gpu)
    print(f"\n{steps} steps in {(time.perf_counter() - started) / 3600:.2f} h, "
          f"final val_loss {val_loss:.4f}")
    print(f"log: {log_path}")
    save_loss_chart(log_path, run_folder)
    save_samples(samples, run_folder)


if __name__ == "__main__":
    main()
