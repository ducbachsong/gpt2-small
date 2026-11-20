"""safetensors_model.py — the model in plain PyTorch, read from an export_safetensors.py folder (model.safetensors +
config.json): what anyone who downloads the model runs it with, since transformers' GPT-2 cannot. It also checks an
export: the val loss on the val token files, the same tokens as the trainer's `val loss` line, so the two can be put
side by side. A weight cut, named or turned wrongly gives a loss far off (a model that knows nothing: about 10.4).

    python python/safetensors_model.py --model d24-8bit --batch-size 16 --batches 10
                                    # the val loss, then the val bpb (nanochat's number)

--batches is the trainer's --val-batches times its GPUs: the trainer's GPUs take turns batch by batch from the val
file's start (dataloader.cuh), and its val loss is the mean of all those batches' mean losses, as here.

It also gives the val bpb, bits per byte: the number nanochat reports (its `val bpb`), which does not depend on the
tokenizer, so it can be set beside nanochat's. The val loss is nats per token. bpb turns it into bits per byte of
text, as nanochat's evaluate_bpb does:
    bpb = sum of the target tokens' losses / ln 2 / sum of their lengths in bytes
with a special token (<|bos|>, the chat ones) counted as 0 bytes and its loss left out, since it stands for no text.
The lengths come from the folder's tokenizer.pkl (or --tokenizer); without one, no bpb.

The model, as cuda/model_bf16.cuh and model_8bit.cuh run it (computed here in float32):
    x = wte[tokens] + wpe[positions]
    each layer:   x = x + proj(attention(qkv(ln_1(x))))       qkv rows q, k, v; heads of head_size side by side
                  x = x + down(gelu_tanh(up(ln_2(x))))
    logits = ln_f(x) @ wte^T                                    no head bias
With config.json's qk_norm, each head row of q and k is gain * x / sqrt(mean(x^2) + eps) before attention (no
weight); with logit_softcap c > 0, logits = c * tanh(logits / c). Folders made before these keys: neither.
Attention is causal; a short layer's query sees its own key and the attention_window keys before it (of every 4
layers the first 3 are short, the last layer long). LayerNorm's epsilon is config.json's.
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HEADER_INTS = 256                        # a token file's header: 256 int32s ([2] the tokens), then uint16 tokens


def layer_window(config, layer):
    """Gives a layer's sliding window: config's attention_window for a short layer, 0 (the whole row) for a long one:
    of every 4 layers the 4th is long, and the last layer always (gpt2_attention_window in model_params.cuh)."""
    long_layer = layer % 4 == 3 or layer + 1 == config["n_layer"]
    return 0 if long_layer else config["attention_window"]


class Model:
    """The model's weights from model.safetensors, and its forward: logits for a batch of token ids."""

    def __init__(self, folder, device):
        """Reads config.json and model.safetensors from `folder` onto `device`, in float32."""
        with open(os.path.join(folder, "config.json")) as config_file:
            self.config = json.load(config_file)
        self.weights = {name: tensor.to(device=device, dtype=torch.float32)
                        for name, tensor in load_file(os.path.join(folder, "model.safetensors")).items()}
        self.device = device
        self.masks = {}                  # (T, window): the attention mask, made once

    def mask(self, seq_len, window):
        """Gives the {T, T} mask of which keys each query sees: those before it and its own, and with a window only
        the `window` keys before it."""
        if (seq_len, window) not in self.masks:
            positions = torch.arange(seq_len, device=self.device)
            distance = positions[:, None] - positions[None, :]           # query - key
            self.masks[(seq_len, window)] = (distance >= 0) & (distance <= window)
        return self.masks[(seq_len, window)]

    def layer_norm(self, x, name):
        """Gives LayerNorm `name` of x: its .weight and .bias, config's epsilon."""
        w = self.weights
        return F.layer_norm(x, (x.shape[-1],), w[f"{name}.weight"], w[f"{name}.bias"], self.config["layer_norm_epsilon"])

    def linear(self, x, name):
        """Gives x times linear `name`'s weight {out, in} turned, plus its bias if the model has linear biases."""
        return F.linear(x, self.weights[f"{name}.weight"], self.weights.get(f"{name}.bias"))

    def attention(self, x, layer):
        """Gives layer `layer`'s attention branch of x {B, T, D} (before the residual add)."""
        B, T, D = x.shape
        NH, HS = self.config["n_head"], self.config["head_size"]
        qkv = self.linear(self.layer_norm(x, f"h.{layer}.ln_1"), f"h.{layer}.attn.qkv")
        q, k, v = (part.view(B, T, NH, HS).transpose(1, 2) for part in qkv.split(D, dim=-1))
        if self.config.get("qk_norm"):                    # QK-norm: each head row to RMS gain, no weight
            gain, eps = self.config["qk_norm_gain"], self.config["qk_norm_epsilon"]
            q = gain * q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
            k = gain * k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
        window = layer_window(self.config, layer)
        if window == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(T, window))
        return self.linear(y.transpose(1, 2).reshape(B, T, D), f"h.{layer}.attn.proj")

    def mlp(self, x, layer):
        """Gives layer `layer`'s MLP branch of x (before the residual add)."""
        up = self.linear(self.layer_norm(x, f"h.{layer}.ln_2"), f"h.{layer}.mlp.up")
        return self.linear(F.gelu(up, approximate="tanh"), f"h.{layer}.mlp.down")

    @torch.no_grad()
    def logits(self, token_ids):
        """Gives the logits {B, T, padded_vocab} for token ids {B, T}."""
        w = self.weights
        T = token_ids.shape[1]
        x = w["wte.weight"][token_ids] + w["wpe.weight"][:T]
        for layer in range(self.config["n_layer"]):
            x = x + self.attention(x, layer)
            x = x + self.mlp(x, layer)
        logits = self.layer_norm(x, "ln_f") @ w["wte.weight"].T
        softcap = self.config.get("logit_softcap", 0)
        return softcap * torch.tanh(logits / softcap) if softcap > 0 else logits


def token_byte_counts(tokenizer_pickle, vocab_size, device):
    """Gives each token id's length in bytes of text {vocab_size}, for the val bpb: as nanochat's token_bytes, a
    special token counts 0 bytes, and so do the ids past the tokenizer's (padded rows, if any).

    In : tokenizer_pickle  str     a pickled tiktoken encoding, e.g. "d24-8bit/tokenizer.pkl" (python/climbmix.py's)
         vocab_size        int     the model's rows of wte
    Out: int64 tensor on `device`"""
    with open(tokenizer_pickle, "rb") as pickle_file:
        encoding = pickle.load(pickle_file)
    special_ids = set(encoding._special_tokens.values())
    counts = [0 if token_id in special_ids else len(encoding.decode_single_token_bytes(token_id))
              for token_id in range(encoding.n_vocab)]
    counts += [0] * (vocab_size - len(counts))
    return torch.tensor(counts, dtype=torch.int64, device=device)


def val_loss(model, val_file, batch_size, seq_len, batches, byte_counts=None):
    """Gives the val loss and, with byte_counts, the val bpb: the loss is the mean of `batches` batches' mean losses,
    the batches one after another from the val file's start, each B x T + 1 tokens (the targets the next tokens), as
    the trainer's GPUs read them; the bpb is those targets' losses in bits over their bytes of text (top of the file).

    Out: (val loss, each batch's loss, val bpb or None, the targets' bytes of text, the targets)"""
    header = np.fromfile(val_file, dtype=np.int32, count=HEADER_INTS)
    batch_tokens = batch_size * seq_len
    if int(header[2]) < batches * batch_tokens + 1:
        raise SystemExit(f"{val_file} holds {int(header[2]):,} tokens, {batches} batches need {batches * batch_tokens + 1:,}")
    tokens = np.fromfile(val_file, dtype=np.uint16, count=batches * batch_tokens + 1, offset=HEADER_INTS * 4)
    losses = []
    text_nats, text_bytes = 0.0, 0                  # summed over the targets that are text (bytes > 0)
    for batch in range(batches):
        chunk = torch.from_numpy(tokens[batch * batch_tokens:(batch + 1) * batch_tokens + 1].astype(np.int64)).to(model.device)
        token_ids, targets = chunk[:-1].view(batch_size, seq_len), chunk[1:].view(batch_size, seq_len)
        logits = model.logits(token_ids)
        # 1. each target's loss in nats: their mean is the batch's loss, as the trainer's
        token_losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        losses.append(token_losses.mean().item())
        # 2. for bpb: the text targets' losses and their bytes; special tokens (0 bytes) left out of both
        if byte_counts is not None:
            target_bytes = byte_counts[targets.reshape(-1)]
            text_nats += token_losses[target_bytes > 0].double().sum().item()
            text_bytes += int(target_bytes.sum().item())
    bpb = text_nats / math.log(2) / text_bytes if byte_counts is not None else None
    return sum(losses) / len(losses), losses, bpb, text_bytes, batches * batch_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="an export_safetensors.py folder")
    parser.add_argument("--val-files", default="python/tokenizer/data/climbmix/climbmix_val_%06d.bin", help="the trainer's --val-files")
    parser.add_argument("--batch-size", type=int, default=8, help="the trainer's --batch-size (rows)")
    parser.add_argument("--seq-len", type=int, help="the trainer's --seq-len; default the model's context")
    parser.add_argument("--batches", type=int, default=20, help="the trainer's --val-batches x its GPUs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer", help="a pickled tiktoken encoding, for the val bpb; default the folder's tokenizer.pkl")
    args = parser.parse_args()

    model = Model(args.model, args.device)
    config = model.config
    params = sum(tensor.numel() for tensor in model.weights.values())
    print(f"{args.model}: {len(model.weights)} tensors, {params:,} params, {config['n_layer']} layers x {config['n_embd']}, "
          f"{config.get('training', {})}")
    seq_len = args.seq_len or config["n_positions"]
    tokenizer_pickle = args.tokenizer or os.path.join(args.model, "tokenizer.pkl")
    byte_counts = None
    if os.path.exists(tokenizer_pickle):
        byte_counts = token_byte_counts(tokenizer_pickle, model.weights["wte.weight"].shape[0], args.device)
    else:
        print(f"no {tokenizer_pickle}: no val bpb (--tokenizer)")
    mean, losses, bpb, text_bytes, targets = val_loss(model, args.val_files % 0, args.batch_size, seq_len, args.batches,
                                                      byte_counts)
    print("batch losses: " + " ".join(f"{loss:.4f}" for loss in losses))
    print(f"val loss {mean:.4f}")
    if bpb is not None:
        print(f"val bpb {bpb:.4f}   ({targets:,} targets, {text_bytes:,} bytes of text: "
              f"{text_bytes / targets:.2f} bytes per token)")


if __name__ == "__main__":
    sys.exit(main())
