"""fineweb.py — FineWeb-Edu into token files for the C/CUDA trainers.

    python data/fineweb.py                          # 10M val + 200M train tokens
    python data/fineweb.py --train-tokens 10000M    # the whole sample/10BT split (~20 GB)

Writes, in data/fineweb10B/:

    fineweb_val_000000.bin      the first --val-tokens tokens: held out, never trained on
    fineweb_train_000001.bin    then up to 100M tokens per file (200 MB)
    fineweb_train_000002.bin    ...

and data/gpt2_tokenizer.bin, with a file `writing` in the folder while it runs, so training can start at
the same time (train_gpt2 --steps N): each file appears whole. llmc/dataloader.cuh reads them: the train files with
the pattern data/fineweb10B/fineweb_train_%06d.bin, the val file on its own.

The documents come through parquetpool.py and tokenpool.py, next to this file,
unchanged from main: ParquetPool downloads the parquet files into RAM in the
background, and TokenPool tokenizes them with GPT-2's tokenizer, <|endoftext|> (50256) after
each document. One tokenizer thread by default, so every run writes the same
files; --threads 4 is faster, and then the order of documents varies.
"""
import argparse
import os
import time

import numpy as np

from data_common import (EOT_TOKEN, decode, gpt2_tokenizer, read_token_file,
                         write_token_file, write_tokenizer_file)
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

HERE = os.path.dirname(os.path.abspath(__file__))

DATASET = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = "sample/10BT/*.parquet"   # 14 files, ~2 GB each, ~10B tokens in all
OUTPUT_DIR = os.path.join(HERE, "fineweb10B")
SHARD_TOKENS = 100_000_000                 # tokens per train file, as in llm.c
ROW_TOKENS = 1024                          # TokenPool's batches: 64 rows of 1024 tokens
ROWS_PER_BATCH = 64


def millions(text):
    """ "200M" -> 200_000_000, "10000M" -> 10_000_000_000, "5000" -> 5000."""
    return int(float(text[:-1]) * 1e6) if text.upper().endswith("M") else int(text)


def shard_path(index):
    kind = "val" if index == 0 else "train"
    return os.path.join(OUTPUT_DIR, f"fineweb_{kind}_{index:06d}.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--val-tokens", type=millions, default="10M")
    parser.add_argument("--train-tokens", type=millions, default="200M")
    parser.add_argument("--threads", type=int, default=1, help="tokenizer threads; each holds a ~2 GB parquet file in RAM")
    parser.add_argument("--parquet-ram", type=float, default=3, help="GB of parquet files in RAM")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # `writing` is there while this runs: a trainer started at the same time waits for the files still
    # to come instead of starting its next epoch (llmc/dataloader.cuh). Deleted at the end, whatever happens.
    writing_marker = os.path.join(OUTPUT_DIR, "writing")
    open(writing_marker, "w").close()
    # ParquetPool first: it turns on huggingface_hub's faster downloads, which
    # only works before transformers imports huggingface_hub.
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                               ram_limit=int(args.parquet_ram * GB))
    try:                               # from here on, the pools are closed whatever happens
        from transformers.utils import logging
        logging.set_verbosity_error()  # documents longer than 1024 tokens are fine here
        write_tokenizer_file()
        token_pool = TokenPool(parquet_pool, tokenizer="gpt2", text_column="text",
                               sequence_length=ROW_TOKENS, batch_size=ROWS_PER_BATCH,
                               token_ram_limit=1 * GB, tokenize_thread_count=args.threads)
        started = time.time()
        # Tokens arrive in batches of 64 x 1024. They collect in `pending` until
        # there are enough for the next file; the rest waits for the one after.
        # Kept as uint16, as the file stores them: 200 MB for a file of 100M, not
        # the 400 MB of TokenPool's int32s.
        shard_index = 0
        shard_size = args.val_tokens
        train_tokens_written = 0
        pending, pending_count = [], 0
        while shard_size > 0:
            batch = token_pool.get_token_batch()
            if batch is None:                      # the dataset ran out
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
            print(f"  {time.time() - started:.0f}s: {train_tokens_written:,} train tokens so far")
        if shard_size > 0 and pending_count > 0:   # ran out mid-file: keep what there is
            write_token_file(shard_path(shard_index), np.concatenate(pending))
    finally:
        close_token_pool()
        close_parquet_pool()
        os.remove(writing_marker)

    # Check: the first whole document of the val file, decoded with the tokenizer
    # file alone and tokenized again, gives the very same ids.
    val_tokens = read_token_file(shard_path(0))
    first_end = int(np.argmax(val_tokens == EOT_TOKEN))
    document = val_tokens[:first_end]
    text = decode(document)
    if gpt2_tokenizer().encode(text) != document.tolist():
        raise SystemExit("the first document does not decode and re-encode to the same ids")
    print(f"check: document 1 ({len(document)} tokens) decodes and re-encodes exactly; it starts:\n"
          f"{text[:300]}")


if __name__ == "__main__":
    main()
