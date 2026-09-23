# GPT-2 Small on FineWeb-Edu 10B — C# trainer

Pre-train a **GPT-2 small (124M parameters)** from scratch on the
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample/10BT`
split (~10 billion tokens of educational web text), as fast as a single GPU allows.

> **This is the `csharp-trainer` branch.** No Python model library is used:
> the GPT-2 model, the AdamW optimizer and the training loop are
> written by hand in **C#** on [TorchSharp](https://github.com/dotnet/TorchSharp)
> (the .NET bindings of LibTorch). The Python data pipeline, logging and charts
> from `main` are reused unchanged. The Hugging Face `transformers` version of the
> trainer is on [`main`](https://github.com/ducbachsong/gpt2-small/tree/main)
> (`traingpt2.py`, still here too).

![Training pipeline](docs/pipeline-csharp.svg)

## How the two sides work together

Python and C# run as two processes and share only **files** and the trainer's
**console**: no sockets, no language bindings.

| Direction | Channel | What |
|---|---|---|
| Python → C# | `<run>/pool/heldout.bin` | first batch, held out for `val_loss`, never trained on |
| Python → C# | `<run>/pool/shard_NNNNNN.bin` | one token batch per file: `[u32 'GPT2'][u32 rows][u32 seq_len][u16 ids…]`, written to `.tmp` then renamed |
| C# → Python | `<run>/pool/consumed.txt` | highest shard already on the GPU; Python deletes those files and writes more (16 kept ready) |
| either | `<run>/pool/STOP` | Python: dataset ran out · C#: training finished |
| C# → Python | stdout | `[Config]`, `[Step]`, `[Eval]`, `[Done]` lines, parsed into `log.csv` |
| Python → C# → Python | `--prompts` / `[Sample]` | prompts go in as token ids; generated ids come back and Python decodes them |

## What makes it fast

**Data (Python, from `main`)**
- Parquet files are downloaded **straight into RAM** in the background, with a RAM cap.
- **Tokenizer threads** fill a buffer of ready batches while the GPU trains.
- Each **shard is copied to the GPU whole** when it is read. Micro-batches are views of it,
  so the training loop never waits on a file or a host→device copy.

**Model and training (C#)**
- **Fused causal attention** (`scaled_dot_product_attention`), which never builds the T×T matrix.
- **Plain fp32** everywhere, like lab06. On Ampere and newer GPUs (A100, L4) the matmuls use TF32
  tensor cores. That's one switch, not mixed precision; on a T4 it has no effect.
- **Few CPU↔GPU syncs**: the loss and grad norm of each step go into GPU buffers that are read
  once every `PRINT_EVERY` steps. Gradient clipping is written to stay on the GPU
  (TorchSharp's `clip_grad_norm_` returns a `double`, which would sync every step).
- **Vocabulary padded to 50,304** (a multiple of 64) for faster matmuls. Padding ids are never sampled.
- **Loss on T−1 positions only**: the last hidden state is dropped *before* the 50k-wide head matmul.
- AdamW with decoupled weight decay on matrices only, β = (0.9, 0.95), linear warmup + cosine LR,
  gradient clipping at 1.0, and GPT-2's scaled init for residual projections.

## Project layout

```
gpt2-small/
├── traingpt2cs.py            # the run: settings, feeds shards, runs the C# trainer, logs
├── colab_setup.sh            # installs .NET 8 + Python packages (Colab / Linux)
├── csharp/Gpt2Trainer/       # the C# trainer (TorchSharp)
│   ├── Model.cs              # GPT-2: attention, MLP, blocks, tied head, sampling
│   ├── AdamW.cs              # AdamW from scratch, LR schedule, on-GPU grad clipping
│   ├── Trainer.cs            # training loop, eval, samples, console protocol
│   ├── Shards.cs             # reads the token shards Python writes
│   ├── Config.cs             # every setting, overridable as --kebab-case flags
│   └── Program.cs            # entry point: device, seed, TF32
├── common/                   # reused Python modules (unchanged from main)
│   ├── parquetpool.py        # parquet files from the HF Hub into RAM, thread-safe
│   ├── tokenpool.py          # background tokenization into ready batches
│   ├── csvwriter.py          # low-overhead CSV logger
│   └── csvexplorer.py        # DuckDB + Altair charts / live dashboard
├── traingpt2.py              # the transformers-based trainer from main, for comparison
└── docs/                     # diagrams
```

## Quick start

### Google Colab (GPU)

```python
!git clone -b csharp-trainer https://github.com/ducbachsong/gpt2-small.git
%cd gpt2-small
!bash colab_setup.sh          # .NET 8 into ~/.dotnet + pip install -r requirements.txt
!python traingpt2cs.py
```

The first run builds the C# project, and NuGet downloads LibTorch 2.10 with CUDA 12.8 (~2 GB).

### Locally

Requirements: Python 3.10+, the [.NET 8 SDK](https://dotnet.microsoft.com/download), and an
NVIDIA GPU (the CPU works for testing only).

```bash
git clone -b csharp-trainer https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small
pip install -r requirements.txt
python traingpt2cs.py
```

`traingpt2cs.py` detects the GPU with `nvidia-smi` and then picks the LibTorch build
(`cpu`, `cuda-linux` or `cuda-windows`) and the micro-batch size, based on GPU memory.

### Running the C# trainer by itself

```bash
cd csharp/Gpt2Trainer
dotnet build -c Release -p:TorchBackend=cuda-linux      # or cpu / cuda-windows
dotnet bin/Release/net8.0/Gpt2Trainer.dll --data <pool dir> \
    --micro-batch 8 --grad-accum-steps 4 --max-steps 3000 --prompts "464,3290;818,19473"
```

Every property in `Config.cs` is a `--kebab-case` flag.

## Configuration

All settings are at the top of `traingpt2cs.py`, which passes them to the C# trainer:

| Setting | Default | Notes |
|---|---|---|
| `DATASET` / `ALLOW_PATTERNS` | `HuggingFaceFW/fineweb-edu` / `sample/10BT/*.parquet` | 14 files, ~2 GB each |
| `SEQUENCE_LENGTH` | 1024 | context length |
| `SHARD_ROWS` / `SHARDS_AHEAD` | 256 / 16 | rows per shard file, files kept ready |
| `N_LAYER` / `N_HEAD` / `N_EMBD` | 12 / 12 / 768 | GPT-2 small |
| `TOKENS_PER_STEP` | 32,768 | micro-batch × 1024 × gradient accumulation |
| `MICRO_BATCH` | `None` = auto | 4 on 16 GB (T4), 8 on 40 GB (A100), 16 on 80 GB |
| `LEARNING_RATE` → `MIN_LEARNING_RATE` | 6e-4 → 6e-5 | cosine after `WARMUP_STEPS = 300` |
| `WEIGHT_DECAY` / `GRAD_CLIP` | 0.1 / 1.0 | |
| `MAX_STEPS` | 3000 | ~98M tokens; ~305,000 steps for the full 10B |
| `PRINT_EVERY` | 25 | also how often the trainer waits for the GPU |
| `PROMPTS` | 3 prompts | continued by the model at the end, into `samples.txt` |
| `MONITOR` | `False` | `True` opens a live loss dashboard (not on Colab) |

## Outputs

Everything goes to `runs/fineweb-edu-gpt2cs-<time>/`:

- `log.csv`: step, tokens, loss, val_loss, lr, grad_norm, tokens/second
- `loss.png`: the training and validation loss curves
- `samples.txt`: the trained model's continuation of each prompt

The model is not saved: when training ends, the trainer evaluates, writes the samples, and exits.
Press **Ctrl+C** to end early. The trainer finishes its step, evaluates and writes the samples.

## Acknowledgements

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by Hugging Face
- GPT-2 by OpenAI
- [TorchSharp](https://github.com/dotnet/TorchSharp) by the .NET Foundation

## License

MIT
