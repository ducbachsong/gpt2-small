"""tune_model_torchao.py — PyTorch's own speed on the model tune_model.cu times (nanochat's d24: 24 layers x 1,536,
12 heads of 128, rows of 2,048 tokens, 32,768 ids, no linear bias), the way PyTorch trains it at its fastest: compiled
(torch.compile), in bfloat16, and with torchao's float8 training (convert_to_float8_training, its tensorwise recipe:
the 4 linears and the head in FP8, one scale for each tensor, the casts fused by the compiler), as nanochat does.
tune_model.cu cannot do either from C++ (libtorch has no torch.compile): its PyTorch FP8 model runs the casts eagerly.

Each micro-batch as tune_model.cu's: gradients set to none, the loss, its backward, the loss read back; warmed up (the
compile happens there), then the average of TIMED_STEPS. Attention: PyTorch's flash attention, causal, no window
(every layer the whole row; ours 18 of 24 layers a window of 512 tokens), as tune_model.cu's PyTorch model.

    pip install torchao
    python3 tests/tune_model_torchao.py                 # rows 1, 2, 4, 8, 16, then the most that fit
    python3 tests/tune_model_torchao.py --rows 8 16
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LAYERS = 24        # tune_model.cu's MODEL_NUM_LAYERS
EMBED_DIM = 1536       # MODEL_EMBED_DIM
NUM_HEADS = 12         # MODEL_NUM_HEADS
VOCAB_SIZE = 32768     # MODEL_VOCAB_SIZE
SEQ_LEN = 2048         # SEQ_LEN
TIMED_STEPS = 20       # tune_helpers.h's
WARM_UP_STEPS = 3      # micro-batches before the timing: the compile, then two more
ROWS_LIST = [1, 2, 4, 8, 16]
MOST_ROWS_STEP = 2     # after the list, the most rows that fit, a multiple of this, tried down from the next power of 2


class Block(nn.Module):
    """One block, as tune_model.cu's pytorch_block_forward: LayerNorm 1, qkv, causal attention, its projection, the
    residual; LayerNorm 2, the MLP (tanh GELU), the residual. No linear bias; the LayerNorms with theirs."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.attention_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp_up = nn.Linear(EMBED_DIM, 4 * EMBED_DIM, bias=False)
        self.mlp_down = nn.Linear(4 * EMBED_DIM, EMBED_DIM, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        query, key, value = self.qkv(self.ln1(x)).split(EMBED_DIM, dim=2)
        heads = lambda part: part.view(batch_size, seq_len, NUM_HEADS, EMBED_DIM // NUM_HEADS).transpose(1, 2)
        y = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
        x = x + self.attention_out(y.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM))
        return x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x)), approximate="tanh"))


class Model(nn.Module):
    """The whole model, as tune_model.cu's pytorch_gpt2_logits: token and position embeddings, the blocks, LayerNorm
    f, and the head sharing the token embedding (no bias); gives the mean cross-entropy loss."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.wpe = nn.Embedding(SEQ_LEN, EMBED_DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.lnf = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.wte.weight                      # tied, as ours

    def forward(self, token_ids, targets):
        x = self.wte(token_ids) + self.wpe(torch.arange(token_ids.shape[1], device=token_ids.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.lnf(x))
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))


def make_model(fp8):
    """Gives the model in bfloat16 on the GPU, GPT-2's small random starting values, compiled; with fp8 its linears
    (the 4 a block and the head) made torchao's Float8Linear first, its tensorwise recipe."""
    torch.manual_seed(0)
    model = Model()
    for name, param in model.named_parameters():
        if param.dim() == 2:
            nn.init.normal_(param, std=0.02)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    if fp8:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"),
                                   module_filter_fn=lambda module, fqn: isinstance(module, nn.Linear))
    return torch.compile(model)


def tokens_per_second(model, rows):
    """Gives the model's tokens/s at `rows` rows of SEQ_LEN, a micro-batch as tune_model.cu's, warmed up then averaged
    over TIMED_STEPS; 0 if it runs out of memory (its cached memory then given back)."""
    try:
        token_ids = (torch.arange(rows * SEQ_LEN, device="cuda") * 7919 % VOCAB_SIZE).view(rows, SEQ_LEN)   # as fill_ids
        targets = token_ids.flatten().roll(-1).view(rows, SEQ_LEN)
        def micro_batch():
            for param in model.parameters():
                param.grad = None
            loss = model(token_ids, targets)
            loss.backward()
            loss.item()                                         # read back, as the loop does
        for _ in range(WARM_UP_STEPS):
            micro_batch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED_STEPS):
            micro_batch()
        end.record()
        torch.cuda.synchronize()
        return rows * SEQ_LEN / (start.elapsed_time(end) / TIMED_STEPS / 1000.0)
    except (torch.OutOfMemoryError, RuntimeError) as error:   # the compiler's own can come as a RuntimeError
        if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
        for param in model.parameters():
            param.grad = None
        torch.cuda.empty_cache()
        return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS_LIST, help="rows a micro-batch, each timed")
    parser.add_argument("--no-most", action="store_true", help="skip the most rows that fit")
    args = parser.parse_args()
    torch._dynamo.config.cache_size_limit = 64                  # a compile for each row count
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name()}   torch {torch.__version__}", end="")
    try:
        import torchao
        print(f"   torchao {torchao.__version__}")
    except ImportError:
        print("   torchao: not installed (pip install torchao)")
    print("tokens/s, a micro-batch as tune_model.cu runs it; both compiled; attention PyTorch's flash, no window")
    print("    rows   PyTorch bf16 (compiled)   PyTorch FP8 (torchao tensorwise, compiled)   FP8 / bf16")
    results = {}
    for fp8 in (False, True):
        model = make_model(fp8)
        rows_list = list(args.rows)
        for rows in rows_list:
            results[(fp8, rows)] = tokens_per_second(model, rows)
        if not args.no_most:                                    # the most that fit: down from twice the largest tried
            rows = 2 * max(rows_list)
            while rows > max(rows_list):
                value = tokens_per_second(model, rows)
                if value > 0.0:
                    results[(fp8, rows)] = value
                    break
                rows -= MOST_ROWS_STEP
        del model
        torch.cuda.empty_cache()
    for rows in sorted({rows for _, rows in results}):
        bf16, fp8 = results.get((False, rows), 0.0), results.get((True, rows), 0.0)
        cell = lambda value: f"{value:9.0f}" if value > 0.0 else f"{'-':>9}"
        ratio = f"{fp8 / bf16:9.2f}x" if bf16 > 0.0 and fp8 > 0.0 else f"{'-':>10}"
        print(f"    {rows:4d}   {cell(bf16):>23}   {cell(fp8):>42}   {ratio}")
    print("\nall tunes done")


if __name__ == "__main__":
    main()
