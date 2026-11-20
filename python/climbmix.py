"""climbmix.py — NVIDIA ClimbMix (nanochat's training data) into token files for the C/CUDA trainers, with
nanochat's tokenizer; fineweb.py's design, only the data and the tokenizer changed.

    python python/climbmix.py                              # 10M val + 200M train tokens
    python python/climbmix.py --train-tokens 6000M --threads 16 &   # in the background, then at once:
    ./train_8bit --steps 11444                           # trains while the files come

Writes, in python/tokenizer/data/climbmix/:

    climbmix_val_000000.bin      the first --val-tokens tokens: held out, never trained on
    climbmix_train_000001.bin    then up to 100M tokens per file (200 MB)
    climbmix_train_000002.bin    ...

with a file `writing` in the folder while it runs, so training can start at the same time (train_8bit --steps N):
each file appears whole, and cuda/helper/dataloader.cuh waits for the next one while `writing` is there. As in
fineweb.py the documents come through parquetpool.py (downloads into RAM in the background) and tokenpool.py
(tokenizes in threads), so downloading, tokenizing and training all run at once and the GPU waits only for the
first train file.

The data: karpathy/climbmix-400b-shuffle on the Hugging Face Hub, ClimbMix's documents already shuffled into 6,543
parquet shards; --shards picks the first N (enough for --train-tokens: about 60M tokens a shard).

The tokenizer: nanochat's (python/tokenizer/tokenizer.py), trained once before the tokens can start, then kept:
rustbpe on the first 2B characters of the first --tokenizer-shards shards in name order. Saved as
python/tokenizer/data/climbmix/tokenizer.pkl (TokenPool loads it) and climbmix_tokenizer.bin next to it (for C). Its few
minutes need no GPU: `--tokenizer-only` on any machine, then copy tokenizer.pkl to the GPU's
python/tokenizer/data/climbmix/, and the GPU waits for nothing but the first file. <|bos|> (32,759) goes after each
document, where fineweb.py puts <|endoftext|>: it begins the next one, as nanochat puts it before each.
"""
import argparse
import os
import time

import numpy as np

from data_common import read_token_file, write_token_file
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenizer.tokenizer import (DATA, TOKENIZER_PICKLE, VOCAB_SIZE, load_tokenizer, train_tokenizer,
                                 write_tokenizer_bin)
from tokenpool import TokenPool, close_token_pool

DATASET = "karpathy/climbmix-400b-shuffle"
SHARD_COUNT = 6543                         # shard_00000.parquet .. shard_06542.parquet
OUTPUT_DIR = os.path.join(DATA, "climbmix")
SHARD_TOKENS = 100_000_000                 # tokens per train file, as in llm.c
ROW_TOKENS = 1024                          # TokenPool's batches: 64 rows of 1024 tokens
ROWS_PER_BATCH = 64


def millions(text):
    """ "200M" -> 200_000_000, "10000M" -> 10_000_000_000, "5000" -> 5000."""
    return int(float(text[:-1]) * 1e6) if text.upper().endswith("M") else int(text)


def shard_names(count):
    """The first `count` shards' names in the repo, e.g. ["shard_00000.parquet", ...]."""
    return [f"shard_{index:05d}.parquet" for index in range(min(count, SHARD_COUNT))]


def shard_path(index):
    """Gives token file `index`'s path: file 0 is the val file, the rest train files."""
    kind = "val" if index == 0 else "train"
    return os.path.join(OUTPUT_DIR, f"climbmix_{kind}_{index:06d}.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--val-tokens", type=millions, default="10M")
    parser.add_argument("--train-tokens", type=millions, default="200M")
    parser.add_argument("--shards", type=int, default=0, help="shards to read; 0: enough for the tokens (60M a shard, +2)")
    parser.add_argument("--threads", type=int, default=4, help="tokenizer threads; each holds a parquet file in RAM")
    parser.add_argument("--parquet-ram", type=float, default=3, help="GB of parquet files in RAM")
    parser.add_argument("--tokenizer-shards", type=int, default=10, help="shards the tokenizer trains on (2B characters)")
    parser.add_argument("--tokenizer-only", action="store_true", help="train the tokenizer and stop (no GPU needed)")
    args = parser.parse_args()
    shards = args.shards or (args.val_tokens + args.train_tokens) // 60_000_000 + 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # `writing` is there while this runs: a trainer started at the same time waits for the files still to come
    # instead of starting its next epoch (cuda/helper/dataloader.cuh), even while the tokenizer trains. Deleted at
    # the end, whatever happens; it holds this process's id, so a trainer that waits can tell a writer that was
    # killed (its `writing` left behind) from one still working.
    writing_marker = os.path.join(OUTPUT_DIR, "writing")
    if not args.tokenizer_only:
        with open(writing_marker, "w") as marker_file:
            marker_file.write(str(os.getpid()))
    try:
        # 1. the tokenizer (python/tokenizer/): trained once, then kept
        if os.path.exists(TOKENIZER_PICKLE):
            encoding = load_tokenizer()
        else:
            encoding = train_tokenizer(DATASET, shard_names(args.tokenizer_shards), args.parquet_ram)
        if encoding.n_vocab != VOCAB_SIZE:
            raise SystemExit(f"the tokenizer has {encoding.n_vocab:,} ids, the trainers' models {VOCAB_SIZE:,}")
        write_tokenizer_bin(encoding)
        if args.tokenizer_only:
            return

        # 2. the tokens, as fineweb.py: the pools download and tokenize in the background; batches of 64 x 1024
        #    collect in `pending` until there are enough for the next file, the rest waits for the one after
        parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=shard_names(shards),
                                   ram_limit=int(args.parquet_ram * GB))
        try:
            token_pool = TokenPool(parquet_pool, tokenizer=TOKENIZER_PICKLE, text_column="text",
                                   sequence_length=ROW_TOKENS, batch_size=ROWS_PER_BATCH,
                                   token_ram_limit=1 * GB, tokenize_thread_count=args.threads)
            started = time.time()
            shard_index = 0
            shard_size = args.val_tokens
            train_tokens_written = 0
            pending, pending_count = [], 0
            while shard_size > 0:
                batch = token_pool.get_token_batch()
                if batch is None:                      # the shards ran out
                    break
                pending.append(batch.reshape(-1).astype(np.uint16))
                pending_count += batch.size
                if pending_count < shard_size:
                    continue
                tokens = np.concatenate(pending)
                write_token_file(shard_path(shard_index), tokens[:shard_size])
                # a copy of the rest, so the joined array is freed rather than kept by a view
                pending, pending_count = [tokens[shard_size:].copy()], len(tokens) - shard_size
                if shard_index > 0:
                    train_tokens_written += shard_size
                shard_index += 1
                shard_size = min(SHARD_TOKENS, args.train_tokens - train_tokens_written)
                print(f"  {time.time() - started:.0f}s: {train_tokens_written:,} train tokens so far", flush=True)
            if shard_size > 0 and pending_count > 0:   # ran out mid-file: keep what there is
                write_token_file(shard_path(shard_index), np.concatenate(pending))
        finally:
            close_token_pool()
            close_parquet_pool()
    finally:
        if os.path.exists(writing_marker):
            os.remove(writing_marker)

    # Check: the first whole document of the val file decodes and tokenizes again to the very same ids
    val_tokens = read_token_file(shard_path(0))
    bos = encoding.encode_single_token("<|bos|>")
    first_end = int(np.argmax(val_tokens == bos))
    document = val_tokens[:first_end].tolist()
    text = encoding.decode(document)
    if encoding.encode_ordinary(text) != document:
        raise SystemExit("the first document does not decode and re-encode to the same ids")
    print(f"check: document 1 ({len(document)} tokens) decodes and re-encodes exactly; it starts:\n{text[:300]}")


if __name__ == "__main__":
    main()
