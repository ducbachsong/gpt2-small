# Tunes

Timing every kernel to tune it for each GPU: which launch (block size, blocks, work per thread) is fastest
there, and how near it comes to the GPU's memory speed. Kept apart from `tests/`: the tests check numbers and
pass or fail; the tunes time and print tables. The same code runs on any GPU (it reads the SM count and the
peak memory bandwidth from the GPU itself); only the results are kept per GPU.

```
tunes/
  tune_helpers.h          gpu_milliseconds, warm_up_gpu, print_gpu, GB/s and MBU
  tune_<kernel>.cu        the timings of tests/test_<kernel>.cu
  results/
    T4/  L4/  A100/  H100/  H200/  B200/
      <kernel>_<date>_<time>.txt   what one run of tune_<kernel> printed on that GPU
```

| File | What it times |
|---|---|
| `tune_adamw.cu` | float AdamW against PyTorch's fused AdamW; other block and thread counts |
| `tune_adamw_bf16.cu` | bfloat16 AdamW with other block and thread counts |
| `tune_adamw_8bit.cu` | the three AdamWs side by side with GB/s and MBU; other block and thread counts, with GB/s and MBU |
| `tune_attention.cu` | attention against PyTorch; flash attention against the steps and PyTorch; the softmax kernels' launches |
| `tune_block.cu` | one transformer block, forward and backward, bfloat16 |
| `tune_classifier.cu` | the classifier against PyTorch; other launches |
| `tune_encoder.cu` | the encoder against PyTorch; other launches; the float4 wte backward and the finer forward sweep |
| `tune_gelu.cu` | GELU against PyTorch; other launches |
| `tune_gpt2.cu` | the whole GPT-2 small step, bfloat16; the micro-batch size (`--batch-size`) with the best MFU |
| `tune_grad_norm.cu` | the grad norm against PyTorch; other launches |
| `tune_layer_norm.cu` | layer norm against PyTorch; other launches |
| `tune_linear.cu`, `tune_linear_cublas.cu` | GPT-2 small's 5 linear layers against PyTorch |
| `tune_residual.cu` | the residual add against PyTorch; other launches |

## MBU

Most kernels here do only a few flops for every byte they read or write, so memory speed is what limits them.
MBU (memory bandwidth utilization) says how much of that speed a kernel uses:

```
GB/s = bytes read and written / time          MBU = GB/s / the GPU's peak GB/s
```

For example, 8-bit AdamW moves 16 bytes per value: it reads 4 + 2 + 1 + 1 and writes 4 + 2 + 1 + 1. On GPT-2
small's 124,439,808 values that is 1.99 GB. At an H100 PCIe's 2,039 GB/s, the fastest it could ever run is
0.98 ms. If it takes 1.20 ms, it reaches 1,659 GB/s, an MBU of 81%.

## MFU

The whole model is limited by the tensor cores instead, and MFU (model flops utilization) says how much of
them training uses:

```
flops a token = 6 x parameters + 12 x layers x embed_dim x seq_len      (forward and backward)
MFU = tokens/s x flops a token / the GPU's peak bfloat16 flops a second
```

GPT-2 small at T = 1024 takes 6 x 124,475,904 + 12 x 12 x 768 x 1024 = 860.1M flops a token. The H100 PCIe
run of 2026-11-04 trained at 186,000 tokens/s: 160 TFLOPS of the GPU's 820, an MFU of 20%, with `--batch-size
4`, which used 6 GB of its 85 GB. `tune_gpt2.cu`'s `find_the_micro_batch_with_the_best_mfu` tries 1, 2, 4, ...
rows a micro-batch while they fit in memory, and prints the MFU of each and the best `--batch-size`.

## Running one

Build a tune the way its test is built, with `tunes/tune_<kernel>.cu` in place of
`tests/test_<kernel>.cu`. On Colab, with the libtorch that comes with PyTorch:

```
TORCH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
nvcc -O3 -std=c++17 -arch=native tunes/tune_gelu.cu -o tune_gelu \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
    -L$TORCH/lib -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 \
    -Xlinker -rpath=$TORCH/lib
./tune_gelu
```

Run it from the repo's folder. Everything it prints shows on the console and also goes into a new file for
that run, in the folder of the GPU it ran on (picked from the GPU's name):

```
log: tunes/results/T4/gelu_2026-11-07_153012.txt
GPU: Tesla T4   compute 7.5   40 SMs ...
```

The tests do the same into `logs/tests/` (`logs/` is not kept in git), e.g.
`logs/tests/test_gelu_2026-11-07_153012.log`. Under torchrun each rank writes its own, `_rank0`, `_rank1`, ...
This needs `tee`, so it works on Linux and Colab; on Windows the output only goes to the console.

A tune needs `tunes/tune_<kernel>.cu`, `tunes/tune_helpers.h`, `tests/test_helpers.h` and
the `llmc/` files its test needs. The ones that call cuBLAS (attention, linear_cublas, block, gpt2) also link
PyTorch's cuBLAS, the same way as their tests.

Run one tune at a time, never two together: they would share the GPU and slow each other down. Each
result file starts with the GPU line `print_gpu` prints, so you can always tell where the numbers came from:

```
GPU: NVIDIA H100 PCIe   compute 9.0   114 SMs   85.5 GB   peak memory bandwidth 2039 GB/s   peak bfloat16 820 TFLOPS   CUDA 12.8
```

## Results

| Folder | GPU | SMs | Peak GB/s (spec) | What is there |
|---|---|---|---|---|
| `T4/` | Tesla T4 (Colab) | 40 | 320 | nothing yet |
| `L4/` | L4 (Colab) | 58 | 300 | nothing yet |
| `A100/` | A100 | 108 | 1,555 (40 GB) / 2,039 (80 GB) | nothing yet |
| `H100/` | H100 PCIe | 114 | 2,039 | the run of 2026-11-04, taken from the test logs before this split (no flash attention yet, no MBU) |
| `H200/` | H200 | 132 | 4,800 | nothing yet |
| `B200/` | B200 | 148 | 8,000 | nothing yet |

An H100 SXM has 132 SMs and 3,350 GB/s, so it is not the same as the PCIe card. If you run on one, say which
card it was in the file. The `print_gpu` line at the top does this for you.
