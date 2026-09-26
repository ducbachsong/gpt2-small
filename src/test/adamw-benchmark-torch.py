"""Python PyTorch's AdamW in its three modes, timed the way the C# benchmark times ours.

    python src/test/adamw-benchmark-torch.py        # on Colab: PyTorch is already installed

Compare with BenchmarkOursAgainstTheBuiltInAdamW (AdamWTests.cs), run on the same GPU:

    dotnet test Gpt2Trainer.sln -p:TorchBackend=cuda-linux --filter Category=Benchmark --logger "console;verbosity=detailed"

TorchSharp's built-in AdamW (what the C# benchmark compares against) only loops over the
parameters. Python PyTorch has two faster modes, and this times all three:

    for-loop   foreach=False   one set of operations per parameter tensor (like TorchSharp's)
    foreach    foreach=True    each operation runs over all parameters in one batched call
    fused      fused=True      the whole AdamW step in one CUDA kernel per group of parameters

The parameters are the same as in the C# benchmark: GPT-2's 148 parameter tensors,
narrowed to 128 wide (8.9M values), so the numbers line up.
"""
import time

import torch

WIDTH = 128               # NEmbd in the C# benchmark
LAYERS = 12
PADDED_VOCAB = 50304
SEQUENCE_LENGTH = 1024
WARMUP_STEPS = 3          # untimed: the first steps allocate memory and warm caches
TIMED_STEPS = 20
LR = 1e-3


def gpt2_shapes(width, layers):
    """The shapes of GPT-2's parameters, in the order Gpt2 (Model.cs) creates them."""
    shapes = [(PADDED_VOCAB, width), (SEQUENCE_LENGTH, width)]                # wte, wpe
    for _ in range(layers):
        shapes += [(width,), (width,)]                                        # ln_1 weight, bias
        shapes += [(3 * width, width), (3 * width,)]                          # attn.c_attn
        shapes += [(width, width), (width,)]                                  # attn.c_proj
        shapes += [(width,), (width,)]                                        # ln_2 weight, bias
        shapes += [(4 * width, width), (4 * width,)]                          # mlp.c_fc
        shapes += [(width, 4 * width), (width,)]                              # mlp.c_proj
    shapes += [(width,), (width,)]                                            # ln_f weight, bias
    return shapes


def milliseconds_per_step(mode, device):
    """Times step() + zero_grad() of torch.optim.AdamW in one mode; backward is outside the clock."""
    torch.manual_seed(0)
    parameters = [torch.nn.Parameter(torch.randn(shape, device=device)) for shape in gpt2_shapes(WIDTH, LAYERS)]
    # One random gradient per parameter, made once and reused every step.
    raw_gradients = [torch.randn_like(parameter) for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                                  foreach=(mode == "foreach"), fused=(mode == "fused"))

    def wait_for_device():
        if device.type == "cuda":
            torch.cuda.synchronize()

    total = 0.0
    for t in range(1, WARMUP_STEPS + TIMED_STEPS + 1):
        # The gradients, set outside the clock (the C# benchmark runs backward() here).
        for parameter, gradient in zip(parameters, raw_gradients):
            parameter.grad = gradient.clone()
        wait_for_device()
        started = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        wait_for_device()
        if t > WARMUP_STEPS:
            total += time.perf_counter() - started
    count = sum(parameter.numel() for parameter in parameters)
    return total / TIMED_STEPS * 1000, len(parameters), count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    print(f"device {name}, PyTorch {torch.__version__}")
    modes = ["for-loop", "foreach"] + (["fused"] if device.type == "cuda" else [])   # fused needs a GPU
    for mode in modes:
        ms, tensors, values = milliseconds_per_step(mode, device)
        print(f"{mode:9} {ms:8.2f} ms per step   ({tensors} parameter tensors, {values / 1e6:.1f}M values)")


if __name__ == "__main__":
    main()
