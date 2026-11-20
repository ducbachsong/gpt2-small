"""finetune_init.py — the fine-tune's starting weights (finetune_8bit --init-from): the pretrained model folder that
export_safetensors.py made (by default downloaded from the private Hub repo ducbachsong/nanochat-d24), its tensors
written as one float32 list in GPT2Params::shapes' order (export_safetensors.named_shapes). The pretrained model is
only read: nothing is written to its repo or folder.

    python python/finetune_init.py                                     # ducbachsong/nanochat-d24 -> the default file
    python python/finetune_init.py --model nanochat-d24                # a folder on disk instead of the Hub
    python python/finetune_init.py --repo <user>/nanochat-d24 --out python/tokenizer/data/finetune_init/nanochat-d24_params.bin

The Hub repo is private: `hf auth login` once, or HF_TOKEN. The model must be finetune_8bit's: d24 (24 x 1,536, 12
heads, 32,768 words, 2,048 positions), no linear biases, QK-norm, softcap 15, window 512; anything else stops it.

File: 8 int64s ([0] magic 20261120, [1] version 1, [2] the values), then the values, float32. safetensors' bfloat16
turns into float exactly (its 16 bits on top), so finetune_8bit's bfloat16 params are the model's to the bit.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tokenizer", "data")   # what the scripts make and read: python/tokenizer/data/ (tokenizer.DATA)
sys.path.insert(0, HERE)
from export_safetensors import named_shapes  # noqa: E402  (the model's names in GPT2Params::shapes' order)

MAGIC, VERSION, HEADER_INT64S = 20261120, 1, 8
EXPECTED = {"n_layer": 24, "n_embd": 1536, "n_head": 12, "vocab_size": 32768, "padded_vocab_size": 32768,
            "n_positions": 2048, "attention_window": 512, "linear_bias": False, "qk_norm": True, "logit_softcap": 15.0}


def read_safetensors_header(path):
    """Gives a safetensors file's header (each tensor's dtype, shape and byte range) and where its bytes start."""
    with open(path, "rb") as model_file:
        header_length = struct.unpack("<Q", model_file.read(8))[0]
        header = json.loads(model_file.read(header_length))
    return header, 8 + header_length


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default="ducbachsong/nanochat-d24", help="the pretrained model's Hub repo (read only)")
    parser.add_argument("--model", help="a model folder on disk instead of --repo")
    parser.add_argument("--out", default=os.path.join(DATA, "finetune_init", "nanochat-d24_params.bin"))
    args = parser.parse_args()

    # 1. the model folder: on disk, or the Hub's two files
    folder = args.model
    if folder is None:
        from huggingface_hub import hf_hub_download
        folder = os.path.join(os.path.dirname(args.out), args.repo.split("/")[-1])
        for name in ["config.json", "model.safetensors"]:
            hf_hub_download(args.repo, name, repo_type="model", local_dir=folder)
    with open(os.path.join(folder, "config.json")) as config_file:
        config = json.load(config_file)
    wrong = {key: config.get(key) for key, value in EXPECTED.items() if config.get(key) != value}
    if wrong:
        raise SystemExit(f"{folder} is not finetune_8bit's model: {wrong} (expected {EXPECTED})")

    # 2. every tensor in GPT2Params::shapes' order, bfloat16 or float32 bits to float32
    model_path = os.path.join(folder, "model.safetensors")
    header, data_start = read_safetensors_header(model_path)
    data = np.memmap(model_path, dtype=np.uint8, mode="r")
    shapes = named_shapes(config)
    total = sum(int(np.prod(shape)) for _, shape in shapes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".partial", "wb") as out_file:
        out_file.write(np.array([MAGIC, VERSION, total] + [0] * (HEADER_INT64S - 3), dtype=np.int64).tobytes())
        for name, shape in shapes:
            info = header[name]
            if tuple(info["shape"]) != tuple(shape):
                raise SystemExit(f"{name}: shape {info['shape']} in the file, {list(shape)} expected")
            start, end = info["data_offsets"]
            raw = np.asarray(data[data_start + start:data_start + end])
            if info["dtype"] == "BF16":
                values = (raw.view(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
            elif info["dtype"] == "F32":
                values = raw.view(np.float32)
            else:
                raise SystemExit(f"{name}: dtype {info['dtype']}, expected BF16 or F32")
            out_file.write(np.ascontiguousarray(values).tobytes())
    os.replace(args.out + ".partial", args.out)
    trained = config.get("training", {})
    print(f"{folder}: {len(shapes)} tensors, {total:,} values (trained: {trained}) -> {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
