"""export_safetensors.py — turns a training checkpoint into a model folder as Hugging Face models are kept: the weights
in one model.safetensors (bfloat16, by name), a config.json with the sizes, the tokenizer, and a README.md. That is
the model to share and use; the checkpoint (master, m, v per GPU) is only for going on with training.

    python data/export_safetensors.py --checkpoint ckpt_8bit --out d24-8bit             # the newest checkpoint
    python data/export_safetensors.py --checkpoint ckpt_8bit/step_00010440 --out d24-8bit --repo <user>/d24-8bit

No GPU: it reads the float master params out of the rank files (llmc/helper/checkpoint.cuh's or checkpoint_8bit.cuh's,
either), joins the ranks' slices in order, which gives every parameter in GPT2Params::shapes' order, cuts them back
into tensors and names them. A rank file is read a tensor at a time (memory-mapped): the 0.76B-parameter d24 model
needs no more memory than its largest tensor. The sizes default to train_8bit's and train_bf16's d24 model; a model of
other sizes needs them given, and a wrong size stops it (the values would not add up).

model.safetensors (the safetensors format: an 8-byte header length, a JSON header naming each tensor's dtype, shape and
byte range, then the raw bytes; no Python pickle, so opening it runs no code):
    wte.weight                 {padded_vocab, D}   the words' rows; also the head's weight (tied, no head bias)
    wpe.weight                 {max_seq_len, D}    the positions' rows
    h.<l>.ln_1.weight, .bias   {D}
    h.<l>.attn.qkv.weight      {3D, D}             as torch.nn.Linear: {out, in}; rows q, then k, then v
    h.<l>.attn.proj.weight     {D, D}
    h.<l>.ln_2.weight, .bias   {D}
    h.<l>.mlp.up.weight        {4D, D}             then GELU (tanh form)
    h.<l>.mlp.down.weight      {D, 4D}
    ln_f.weight, .bias         {D}
(the 4 linears' .bias too, if the model has them). Read it with `safetensors.torch.load_file`. It is not a
transformers GPT-2 (no linear biases, the sliding windows, nn.Linear's {out, in} layout), so transformers cannot load
it without this model's own code; the file itself is standard.

--repo sends the folder to a private model repo on the Hub (`hf auth login` once, or HF_TOKEN); --public makes a new
one public.
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER_INT64S = 32                         # a checkpoint file's header (checkpoint.cuh, checkpoint_8bit.cuh)
HEADER_BYTES = HEADER_INT64S * 8
FLOAT_MAGIC, EIGHT_BIT_MAGIC = 20241103, 20261118
SLICE_INDEXES = {FLOAT_MAGIC: (7, 8), EIGHT_BIT_MAGIC: (6, 7)}   # header[...]: slice start, slice size
KINDS = {FLOAT_MAGIC: "float or bfloat16 run", EIGHT_BIT_MAGIC: "8-bit run"}
STEPS_DONE_INDEX, WORLD_SIZE_INDEX, RANK_INDEX = 2, 4, 5
LAYER_NORM_EPS = 1e-5                      # LAYER_NORM_BF16_EPS: added to the variance, as GPT-2
QK_NORM_GAIN = 1.2                         # QK_NORM_8BIT_GAIN: query's and key's head rows' RMS after QK-norm
QK_NORM_EPS = 1e-6                         # QK_NORM_8BIT_EPS: added to mean(x²)


def step_folder_of(path):
    """Gives the step folder to read: `path` itself, or the one its `latest` names if it is a checkpoint dir."""
    latest = os.path.join(path, "latest")
    if os.path.exists(latest):
        with open(latest) as latest_file:
            return os.path.join(path, latest_file.read().strip())
    return path


def rank_slices_of(step_folder):
    """Opens every rank file's master params (memory-mapped floats) and gives [(slice start, values)] in rank order,
    with the run's kind and step; stops unless the slices follow one another from 0 with no gap."""
    names = [name for name in os.listdir(step_folder) if name.startswith("rank_") and name.endswith(".bin")]
    names.sort(key=lambda name: int(name[len("rank_"):-len(".bin")]))
    if not names:
        raise SystemExit(f"{step_folder} has no rank files")
    slices, kinds, steps = [], set(), set()
    for rank, name in enumerate(names):
        path = os.path.join(step_folder, name)
        header = np.fromfile(path, dtype=np.int64, count=HEADER_INT64S)
        magic = int(header[0]) if len(header) == HEADER_INT64S else 0
        if magic not in SLICE_INDEXES:
            raise SystemExit(f"{path} is not a checkpoint file")
        start_index, size_index = SLICE_INDEXES[magic]
        start, size = int(header[start_index]), int(header[size_index])
        if int(header[RANK_INDEX]) != rank or int(header[WORLD_SIZE_INDEX]) != len(names):
            raise SystemExit(f"{path}: rank {int(header[RANK_INDEX])} of {int(header[WORLD_SIZE_INDEX])}, but it is "
                             f"file {rank} of {len(names)}")
        expected_start = slices[-1][0] + len(slices[-1][1]) if slices else 0
        if start != expected_start:
            raise SystemExit(f"{path}: its slice starts at {start:,}, the one before ends at {expected_start:,}")
        slices.append((start, np.memmap(path, dtype=np.float32, mode="r", offset=HEADER_BYTES, shape=(size,))))
        kinds.add(KINDS[magic])
        steps.add(int(header[STEPS_DONE_INDEX]))
    if len(kinds) != 1 or len(steps) != 1:
        raise SystemExit(f"{step_folder}: its rank files are of more than one run ({kinds}, steps {steps})")
    return slices, kinds.pop(), steps.pop()


def values_at(slices, start, count):
    """Gives `count` master values from position `start` of all the params, taken from whichever ranks' slices hold
    them (a tensor can sit across two ranks)."""
    parts = []
    for slice_start, values in slices:
        first, last = max(start, slice_start), min(start + count, slice_start + len(values))
        if first < last:
            parts.append(values[first - slice_start:last - slice_start])
    return np.concatenate(parts) if len(parts) > 1 else np.array(parts[0])


def named_shapes(config):
    """Gives the model's (name, shape) list in GPT2Params::shapes' order: wte, wpe, every block's 4 matrices, every
    block's vectors, ln_f."""
    D, L = config["n_embd"], config["n_layer"]
    shapes = [("wte.weight", (config["padded_vocab_size"], D)), ("wpe.weight", (config["n_positions"], D))]
    for l in range(L):
        shapes += [(f"h.{l}.attn.qkv.weight", (3 * D, D)), (f"h.{l}.attn.proj.weight", (D, D)),
                   (f"h.{l}.mlp.up.weight", (4 * D, D)), (f"h.{l}.mlp.down.weight", (D, 4 * D))]
    for l in range(L):
        if config["linear_bias"]:
            shapes += [(f"h.{l}.ln_1.weight", (D,)), (f"h.{l}.ln_1.bias", (D,)),
                       (f"h.{l}.attn.qkv.bias", (3 * D,)), (f"h.{l}.attn.proj.bias", (D,)),
                       (f"h.{l}.ln_2.weight", (D,)), (f"h.{l}.ln_2.bias", (D,)),
                       (f"h.{l}.mlp.up.bias", (4 * D,)), (f"h.{l}.mlp.down.bias", (D,))]
        else:
            shapes += [(f"h.{l}.ln_1.weight", (D,)), (f"h.{l}.ln_1.bias", (D,)),
                       (f"h.{l}.ln_2.weight", (D,)), (f"h.{l}.ln_2.bias", (D,))]
    shapes += [("ln_f.weight", (D,)), ("ln_f.bias", (D,))]
    return shapes


def to_bfloat16_bits(values):
    """Rounds floats to bfloat16, to the nearest (ties to even, as the GPU's __float2bfloat16): gives their 16-bit
    patterns, the top half of each float's bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def write_safetensors(path, slices, shapes, dtype):
    """Writes every named tensor into one safetensors file, a tensor at a time: the header (each tensor's dtype,
    shape and byte range), then their bytes in the same order. Gives the values written."""
    # 1. the header: byte ranges one after another, the JSON padded with spaces to a multiple of 8 bytes
    value_bytes = 2 if dtype == "BF16" else 4
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, shape in shapes:
        size = int(np.prod(shape)) * value_bytes
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header_json = json.dumps(header, separators=(",", ":")).encode()
    header_json += b" " * (-len(header_json) % 8)

    # 2. the tensors, from the ranks' slices in order
    position = 0
    with open(path + ".partial", "wb") as out_file:
        out_file.write(struct.pack("<Q", len(header_json)))
        out_file.write(header_json)
        for name, shape in shapes:
            count = int(np.prod(shape))
            values = values_at(slices, position, count)
            out_file.write((to_bfloat16_bits(values) if dtype == "BF16" else values.astype(np.float32)).tobytes())
            position += count
    os.replace(path + ".partial", path)
    return position


def model_card(config, repo):
    """Gives a short README.md for the folder: what the model is, and how to read its weights."""
    return f"""---
library_name: safetensors
tags: [gpt2, llm.c, fp8]
---
# {repo or 'GPT-2 d' + str(config['n_layer'])}

A GPT-2-style model of {config['n_layer']} layers x {config['n_embd']} ({config['n_head']} heads of
{config['n_embd'] // config['n_head']}), {config['n_positions']:,} tokens of context and a {config['vocab_size']:,}-word tokenizer,
trained from scratch in CUDA/C++ (github.com/ducbachsong/nanochat) on NVIDIA ClimbMix: {config['training']['steps']:,}
steps of {config['training']['run']}.

`config.json` has the sizes and how the layers work (sliding windows, no linear biases, GELU tanh, the head tied to
`wte`). The weights:

```python
from safetensors.torch import load_file
weights = load_file("model.safetensors")       # {{"wte.weight": tensor, "h.0.attn.qkv.weight": ..., ...}}
```

Not loadable by `transformers`' GPT-2 classes: the layout is this model's own (see `config.json`).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="a checkpoint dir (its latest) or one step folder")
    parser.add_argument("--out", required=True, help="the model folder to write")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--padded-vocab-size", type=int, help="wte's rows; default --vocab-size")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--attention-window", type=int, default=512, help="the short layers' window; 0: none")
    parser.add_argument("--linear-bias", type=int, default=0, help="1: the 4 linears have biases")
    parser.add_argument("--qk-norm", type=int, default=1, help="1: QK-norm before attention (train_8bit's TRAIN_8BIT_QK_NORM)")
    parser.add_argument("--logit-softcap", type=float, default=15.0,
                        help="logits capped to c tanh(logit / c) (train_8bit's TRAIN_8BIT_LOGIT_SOFTCAP); 0: none")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16", help="the weights' dtype in the file")
    parser.add_argument("--tokenizer", default=os.path.join(HERE, "climbmix", "tokenizer.pkl"),
                        help="copied into the folder if it is there")
    parser.add_argument("--repo", help="send the folder to this model repo on the Hub: <user>/<name>")
    parser.add_argument("--public", action="store_true", help="--repo: make a new repo public")
    args = parser.parse_args()

    # 1. the checkpoint's master params, and the model's tensors
    step_folder = step_folder_of(args.checkpoint)
    slices, kind, step = rank_slices_of(step_folder)
    config = {
        "architecture": "llmc-gpt2", "n_layer": args.num_layers, "n_embd": args.embed_dim, "n_head": args.num_heads,
        "head_size": args.embed_dim // args.num_heads, "vocab_size": args.vocab_size,
        "padded_vocab_size": args.padded_vocab_size or args.vocab_size, "n_positions": args.max_seq_len,
        "attention_window": args.attention_window,
        "window_pattern": "of every 4 layers the first 3 see attention_window tokens back, the 4th all; the last layer all",
        "linear_bias": bool(args.linear_bias), "layer_norm_epsilon": LAYER_NORM_EPS, "activation": "gelu_tanh",
        "qk_norm": bool(args.qk_norm), "qk_norm_gain": QK_NORM_GAIN, "qk_norm_epsilon": QK_NORM_EPS,
        "qk_norm_rule": "q, k = gain * x / sqrt(mean(x^2) + eps) per head row, no weight, before attention",
        "logit_softcap": args.logit_softcap, "logit_softcap_rule": "logits = c * tanh(logits / c) before the softmax; 0: none",
        "tied_head": "logits = ln_f(x) @ wte.weight^T, no bias", "position": "learned wpe added to wte",
        "weight_layout": "torch.nn.Linear {out, in}; qkv rows: q, k, v, each heads side by side",
        "torch_dtype": "bfloat16" if args.dtype == "bf16" else "float32",
        "training": {"run": kind, "steps": step, "gpus": len(slices), "checkpoint": os.path.basename(step_folder)},
    }
    shapes = named_shapes(config)
    expected = sum(int(np.prod(shape)) for _, shape in shapes)
    held = slices[-1][0] + len(slices[-1][1])
    if held != expected:
        raise SystemExit(f"{step_folder} holds {held:,} params, a model of these sizes has {expected:,}: give its sizes")

    # 2. the folder: the weights, config.json, the tokenizer, README.md
    os.makedirs(args.out, exist_ok=True)
    print(f"{step_folder}: step {step:,} ({kind}), {len(slices)} rank file(s), {held:,} params")
    written = write_safetensors(os.path.join(args.out, "model.safetensors"), slices, shapes, args.dtype.upper())
    with open(os.path.join(args.out, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)
    if os.path.exists(args.tokenizer):
        shutil.copy(args.tokenizer, os.path.join(args.out, os.path.basename(args.tokenizer)))
    with open(os.path.join(args.out, "README.md"), "w") as card_file:
        card_file.write(model_card(config, args.repo))
    size = os.path.getsize(os.path.join(args.out, "model.safetensors"))
    print(f"{args.out}/model.safetensors: {len(shapes)} tensors, {written:,} values, {size / 1e9:.2f} GB ({args.dtype})")

    # 3. the Hub, if asked
    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=args.out,
                          commit_message=f"{kind}, step {step:,}")
        print(f"up: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())
