# nanochat

nanochat is a from-scratch C/CUDA trainer for nanochat-sized LLMs, with every kernel written by hand: no PyTorch, no cuBLAS, no cuDNN at training time. It started as llm.c's GPT-2 (124M) and grew to the size and recipe of [karpathy/nanochat](https://github.com/karpathy/nanochat)'s d24 model: 24 layers of 1,536, 12 heads of 128, 2,048 tokens, about 0.76B parameters, ClimbMix data, nanochat's 32,768-id tokenizer and sliding-window attention. The matrix products run in FP8 (E4M3) on the H100's tensor cores through WGMMA and TMA, one kernel for each of the model's 15 products, each one tuned on its own. Everything is small enough to read: one `.cuh` file per kernel, one test and one tune next to it, and three trainers (`train_fp32.cu`, `train_bf16.cu`, `train_8bit.cu`) that differ only in their number type.

## Speed

`train_8bit` on the d24 model (24 × 1,536, 2,048 tokens a row), tokens per second:

| GPUs | Tokens/s | Step (524,288 tokens) | Date | Commit |
|---|---|---|---|---|
| 1× H100 SXM | 112,426 | 4.66 s | 2026-11-16 | e4c6ab7 |
| 2× H100 PCIe (NVLink) | 152,283 | 3.44 s | 2026-11-16 | e4c6ab7 |
| 1× H100 PCIe | 76,417 | 6.86 s | 2026-11-16 | e4c6ab7 |

Against PyTorch on the same H100 SXM (`tune_model`: one micro-batch as the training loop runs it; PyTorch uses cuBLAS for every product and its own flash attention):

| Rows | Ours bf16 | Ours 8-bit | PyTorch bf16 | 8-bit / PyTorch |
|---|---|---|---|---|
| 4 | 94,010 | 101,255 | 101,116 | 1.00× |
| 8 | 99,335 | 107,233 | 103,789 | 1.03× |
| 16 | 103,957 | 111,049 | 104,203 | 1.07× |
At its best our 8-bit step reaches 112,678 tokens/s at 24 rows (MFU 50.5%), and our bf16 one 106,796 at 26 rows (MFU 47.8%); PyTorch runs out of memory at 26. PyTorch's attention has no sliding window, so it does a little more work: in ours, 18 of the 24 layers see only the last 512 tokens. 13 of the 15 8-bit products beat cuBLAS's bfloat16 on their own (1.11× to 1.56×). The two that don't, `mlp_up` and `mlp_down_input_grad`, do GELU inside them and cuBLAS doesn't; `mlp_up` is now 0.97× cuBLAS on an H100 PCIe (b7101a5). Every run's full log is in [tests/results/H100/](tests/results/H100/).

## Getting started

### Setup

An H100 with CUDA 12.8 (the products need `-arch=sm_90a`) and NCCL. The trainers themselves need nothing else; the data scripts and the tests need Python, and the tests check against libtorch:

```bash
sudo apt install cuda-toolkit-12-8
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # the tests only
```

### Train

The GPUs should never wait for data, so the tokens come first. Tokenize once, on any machine (no GPU needed; about 1.3M tokens a second with 16 threads), and keep the files in a dataset on the Hugging Face Hub. Ours is public, [ducbachsong/climbmix-tokens-32768](https://huggingface.co/datasets/ducbachsong/climbmix-tokens-32768) (CC BY-NC 4.0, as ClimbMix), so you can skip this step and just download it:

```bash
python data/climbmix.py --train-tokens 6000M --threads 16
python data/token_hub.py upload --repo <user>/climbmix-tokens-32768
```

Then on each GPU machine download them, a few minutes, and train. `--data-ready 1` starts only when every token of the run is on disk:

```bash
python data/token_hub.py download --repo <user>/climbmix-tokens-32768
nvcc -ccbin g++ -O3 -std=c++17 -arch=sm_90a train_8bit.cu -o train_8bit -lnccl
torchrun --standalone --nproc_per_node=8 --no-python ./train_8bit --data-ready 1 --steps 11444   # 8 GPUs
```

One GPU runs the same steps, just 8× slower: `./train_8bit`. For a quick try you can also train while the files are still being written or downloaded: the trainer waits for a lead of `--data-lead-tokens` (500M) before the first step, each step's line shows its `data` milliseconds, and the loader says so whenever the GPUs had to wait for a file:

```bash
python data/climbmix.py --train-tokens 400M --threads 16 &
./train_8bit --steps 20 --tokens-per-step 32768 --val-every 10
```

Each step is 524,288 tokens, all GPUs together. `--batch-size` sets the rows per micro-batch; `tune_model` finds the best one for your GPU (16 to 24 on an H100 at 80 GB). `./train_8bit --help` lists the other flags (learning rate, warmup, val, checkpoints).

## Precision

| Trainer | Params, grads, activations | Matrix products | AdamW state |
|---|---|---|---|
| `train_fp32.cu` | float | float | float |
| `train_bf16.cu` | bfloat16 | bfloat16 on WGMMA | float master params |
| `train_8bit.cu` | bfloat16 | FP8 E4M3 on WGMMA, float sums | float master params, m and √v in 8 bits |

In `train_8bit` each product's two sides are rounded to E4M3 just before it, with one scale per tensor taken from its largest values over the last steps (delayed scaling, as Transformer Engine does). Where it can, the rounding happens inside the kernel that writes the tensor: LayerNorm writes its output's 8-bit copy, and `mlp_up` writes GELU's output straight in 8 bits, both the copy the forward reads and the turned copy the backward reads.

## Tests and tunes

Every kernel has a **test** (`test_<kernel>.cu`: checks its numbers against PyTorch, passes or fails) and a **tune** (`tune_<kernel>.cu`: times its settings on this GPU and against cuBLAS). They sit side by side in `tests/fp32/`, `tests/bf16/` and `tests/8bit/`; [tests/README.md](tests/README.md) lists them all. Each kernel's settings are `#define`s at the top of its file, set from a tune on the GPU it will train on.

## File structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── train_fp32.cu                   # the trainer in float, any GPU
├── train_bf16.cu                   # the trainer in bfloat16, an H100
├── train_8bit.cu                   # the trainer with FP8 products, an H100
├── data
│   ├── climbmix.py                 # ClimbMix into token files, trains nanochat's tokenizer
│   ├── fineweb.py                  # FineWeb-Edu into llm.c's token files (GPT-2's tokenizer)
│   ├── token_hub.py                # uploads and downloads the token files: tokenize once, keep them on the Hub
│   ├── parquetpool.py              # downloads the parquet shards into RAM in the background
│   ├── tokenpool.py                # tokenizes in threads
│   └── data_common.py              # the token-file and tokenizer-file formats
├── llmc
│   ├── model_fp32.cuh              # the model's forward and backward, one per number type
│   ├── model_bf16.cuh
│   ├── model_8bit.cuh
│   ├── common                      # tensors, the model's params, the training settings and flags
│   ├── helper                      # the dataloader, checkpoints, NCCL across GPUs
│   └── kernels
│       ├── cuda_common.h           # warp and block reductions, the GPU's fixed numbers
│       ├── fp32                    # one file per kernel: attention, layer_norm, gelu, adamw, ...
│       ├── bf16                    # the same in bfloat16, plus WGMMA products
│       │   └── prod                # one kernel per model product, each tuned on its own
│       └── 8bit                    # FP8 rounding, 8-bit AdamW, FP8 products
│           └── prod                # the 15 FP8 product kernels: qkv, mlp_up, head, their grads, ...
└── tests
    ├── test_helpers.h              # ASSERT_EQ, ASSERT_NEAR
    ├── tune_helpers.h              # GPU timing, GB/s and MBU
    ├── test_model.cu, tune_model.cu
    ├── fp32, bf16, 8bit            # test_<kernel>.cu and tune_<kernel>.cu for every kernel
    └── results                     # what each tune printed, per GPU and per day
```

## Contributing

The goal is a trainer you can read end to end and still be fast: every kernel in its own file, every number in a kernel a named `#define` that a tune can change, and every change measured. A change to a kernel comes with its test passing and its tune's numbers, on the GPU it targets.

## Acknowledgements

- [karpathy/llm.c](https://github.com/karpathy/llm.c), where this started: GPT-2 in plain C and CUDA.
- [karpathy/nanochat](https://github.com/karpathy/nanochat) for the model size, the recipe, the tokenizer and the name.
- NVIDIA's [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) data, and CUTLASS and Transformer Engine, whose ping-pong products and delayed FP8 scaling are copied here.

## License

MIT
