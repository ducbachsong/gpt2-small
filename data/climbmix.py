"""climbmix.py — NVIDIA ClimbMix (nanochat's training data) into token files for the C/CUDA trainers, with
nanochat's tokenizer; fineweb.py's design, only the data and the tokenizer changed.

    python data/climbmix.py                              # 10M val + 200M train tokens
    python data/climbmix.py --train-tokens 6000M --threads 16 &   # in the background, then at once:
    ./train_8bit --steps 11444                           # trains while the files come

Writes, in data/climbmix/:

    climbmix_val_000000.bin      the first --val-tokens tokens: held out, never trained on
    climbmix_train_000001.bin    then up to 100M tokens per file (200 MB)
    climbmix_train_000002.bin    ...

with a file `writing` in the folder while it runs, so training can start at the same time (train_8bit --steps N):
each file appears whole, and llmc/helper/dataloader.cuh waits for the next one while `writing` is there. As in
fineweb.py the documents come through parquetpool.py (downloads into RAM in the background) and tokenpool.py
(tokenizes in threads), so downloading, tokenizing and training all run at once and the GPU waits only for the
first train file.

The data: karpathy/climbmix-400b-shuffle on the Hugging Face Hub, ClimbMix's documents already shuffled into 6,543
parquet shards; --shards picks the first N (enough for --train-tokens: about 60M tokens a shard).

The tokenizer: nanochat's (its scripts/tok_train.py), trained once before the tokens can start, then kept:
rustbpe on the first 2B characters of the first --tokenizer-shards shards in name order (each document cut at
10,000 characters), 32,768 ids = 32,759 merges + 9 special tokens, GPT-4's split pattern with numbers of 1-2
digits. Saved as data/climbmix/tokenizer.pkl (the tiktoken encoding, as nanochat saves it; TokenPool loads it) and
data/climbmix/climbmix_tokenizer.bin (data_common's tokenizer file, for C). Its few minutes need no GPU:
`--tokenizer-only` on any machine, then copy tokenizer.pkl to the GPU's data/climbmix/, and the GPU waits for
nothing but the first file. <|bos|> (32,759) goes after each document, where fineweb.py puts <|endoftext|>: it
begins the next one, as nanochat puts it before each.
"""
import argparse
import os
import pickle
import time

import numpy as np

from data_common import HEADER_INTS, TOKENIZER_FILE_MAGIC, TOKENIZER_FILE_VERSION, read_token_file, write_token_file
from parquetpool import GB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

HERE = os.path.dirname(os.path.abspath(__file__))

DATASET = "karpathy/climbmix-400b-shuffle"
SHARD_COUNT = 6543                         # shard_00000.parquet .. shard_06542.parquet
OUTPUT_DIR = os.path.join(HERE, "climbmix")
TOKENIZER_PICKLE = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
TOKENIZER_BIN = os.path.join(OUTPUT_DIR, "climbmix_tokenizer.bin")
SHARD_TOKENS = 100_000_000                 # tokens per train file, as in llm.c
ROW_TOKENS = 1024                          # TokenPool's batches: 64 rows of 1024 tokens
ROWS_PER_BATCH = 64
VOCAB_SIZE = 32768                         # nanochat's: 2^15
TOKENIZER_MAX_CHARS = 2_000_000_000        # nanochat's tok_train defaults
TOKENIZER_DOC_CAP = 10_000
# nanochat/tokenizer.py's, to the letter
SPECIAL_TOKENS = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                  "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def millions(text):
    """ "200M" -> 200_000_000, "10000M" -> 10_000_000_000, "5000" -> 5000."""
    return int(float(text[:-1]) * 1e6) if text.upper().endswith("M") else int(text)


def shard_names(count):
    """The first `count` shards' names in the repo, e.g. ["shard_00000.parquet", ...]."""
    return [f"shard_{index:05d}.parquet" for index in range(min(count, SHARD_COUNT))]


def shard_path(index):
    kind = "val" if index == 0 else "train"
    return os.path.join(OUTPUT_DIR, f"climbmix_{kind}_{index:06d}.bin")


def train_tokenizer(shards, parquet_ram):
    """Trains nanochat's tokenizer on the first TOKENIZER_MAX_CHARS characters of the first `shards` shards, read in
    name order as nanochat reads them, and pickles it to TOKENIZER_PICKLE.

    In : shards       int    shards to train on (2B characters take about 8)
         parquet_ram  float  GB of parquet files the pool may hold
    Out: the tiktoken.Encoding
    """
    import pyarrow
    import pyarrow.parquet as pq
    import rustbpe
    import tiktoken

    # 1. the shards, into RAM by ParquetPool; they come in whatever order they finish, so kept and read by name
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=shard_names(shards), ram_limit=int(parquet_ram * GB))
    files = {}
    try:
        for file_name, file_bytes in parquet_pool:
            files[file_name] = file_bytes
    finally:
        close_parquet_pool()

    # 2. the documents in order, each cut at TOKENIZER_DOC_CAP, until TOKENIZER_MAX_CHARS
    def texts():
        characters = 0
        for file_name in sorted(files):
            parquet_reader = pq.ParquetFile(pyarrow.BufferReader(files[file_name]))
            for chunk in parquet_reader.iter_batches(batch_size=1000, columns=["text"]):
                for document in chunk.column(0).to_pylist():
                    document = (document or "")[:TOKENIZER_DOC_CAP]
                    characters += len(document)
                    yield document
                    if characters > TOKENIZER_MAX_CHARS:
                        return
        print(f"note: the {len(files)} shards hold {characters:,} characters, fewer than {TOKENIZER_MAX_CHARS:,}: "
              f"--tokenizer-shards more for nanochat's vocabulary")

    # 3. rustbpe's merges, then the tiktoken encoding with the special tokens after them
    started = time.time()
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(texts(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    mergeable_ranks = {bytes(piece): rank for piece, rank in tokenizer.get_mergeable_ranks()}
    special_tokens = {name: len(mergeable_ranks) + offset for offset, name in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(name="rustbpe", pat_str=tokenizer.get_pattern(), mergeable_ranks=mergeable_ranks,
                                 special_tokens=special_tokens)
    with open(TOKENIZER_PICKLE, "wb") as pickle_file:
        pickle.dump(encoding, pickle_file)
    print(f"trained the tokenizer in {time.time() - started:.0f}s: {encoding.n_vocab:,} ids, saved to {TOKENIZER_PICKLE}")
    return encoding


def write_tokenizer_bin(encoding):
    """Writes data_common's tokenizer file for this encoding, once: each id's bytes (a special token's: its name),
    and <|bos|> where GPT-2's file has <|endoftext|>."""
    if os.path.exists(TOKENIZER_BIN):
        return
    special_names = {token_id: name for name, token_id in encoding._special_tokens.items()}
    pieces = [special_names[token_id].encode() if token_id in special_names else encoding.decode_single_token_bytes(token_id)
              for token_id in range(encoding.n_vocab)]
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKENIZER_FILE_MAGIC
    header[1] = TOKENIZER_FILE_VERSION
    header[2] = len(pieces)
    header[3] = encoding.encode_single_token("<|bos|>")
    with open(TOKENIZER_BIN, "wb") as tokenizer_file:
        tokenizer_file.write(header.tobytes())
        for piece in pieces:
            if not 0 < len(piece) < 256:
                raise ValueError(f"a token of {len(piece)} bytes does not fit the file format")
            tokenizer_file.write(bytes([len(piece)]) + piece)
    print(f"wrote {len(pieces):,} tokens to {TOKENIZER_BIN}")


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
    # instead of starting its next epoch (llmc/helper/dataloader.cuh), even while the tokenizer trains. Deleted at
    # the end, whatever happens.
    writing_marker = os.path.join(OUTPUT_DIR, "writing")
    if not args.tokenizer_only:
        open(writing_marker, "w").close()
    try:
        # 1. the tokenizer: trained once, then kept
        if os.path.exists(TOKENIZER_PICKLE):
            with open(TOKENIZER_PICKLE, "rb") as pickle_file:
                encoding = pickle.load(pickle_file)
        else:
            encoding = train_tokenizer(args.tokenizer_shards, args.parquet_ram)
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
