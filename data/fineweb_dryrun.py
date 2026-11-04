"""fineweb_dryrun.py — how many tokens FineWeb-Edu holds, how many steps that is, and how long training takes,
before downloading it.

    python data/fineweb_dryrun.py                     # an estimate, in seconds
    python data/fineweb_dryrun.py --exact             # the exact count, ~10,000 small reads, a minute or two
    python data/fineweb_dryrun.py --tokens-per-step 524288 --tokens-per-second 2830,5632,400000

It reads only what it needs, over HTTP, from the Hugging Face Hub. A file of sample/10BT holds 726,000
documents in 726 row groups of 1,000; each row group's text is ~2.9 MB, its token_count column 4.4 KB:

    1. the list of parquet files, and each one's footer (a few hundred KB): its documents, and every row
       group's text size in bytes
    2. 500 documents from 5 places across the dataset, from the Hub's dataset viewer, tokenized here with
       GPT-2's tokenizer: tokens per byte of text, and a check that token_count counts GPT-2's tokens
    3. the tokens: text bytes x tokens per byte, within about 1%; or with --exact, the sum of every
       document's token_count, read row group by row group (only those 4.4 KB each, 32 at a time)

fineweb.py writes every document's tokens and one <|endoftext|> after each, so

    total tokens = the documents' tokens + documents

The first --val-tokens are the val file; the rest is for training. steps = train tokens / tokens per step, and
the time is that at each --tokens-per-second given: measured 2,830 on 1 T4 and 5,632 on 2 (train_gpt2.cu).
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

from data_common import EOT_TOKEN, gpt2_tokenizer

DATASET = "HuggingFaceFW/fineweb-edu"
PATTERN = "sample/10BT/*.parquet"
VIEWER_CONFIG = "sample-10BT"               # the same files, as the dataset viewer names them
TOKENIZE_TOKENS_PER_SECOND = 390_000       # fineweb.py with one thread, measured on Colab
SAMPLE_PLACES = 5                           # where across the dataset the sample documents come from
ROW_GROUPS_PER_READ = 25                    # --exact: one task reads 25 row groups' token_count


def millions(text):
    """ "10M" -> 10_000_000, "5000" -> 5000."""
    return int(float(text[:-1]) * 1e6) if text.upper().endswith("M") else int(text)


def duration_text(seconds):
    """A length of time in the unit that reads best: "45 s", "12.5 min", "6.4 h", "9.9 days"."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def open_exactly(filesystem, path):
    """Opens a file on the Hub that reads exactly the bytes asked for: without it, each small read fetches a
    block of several MB, and the 4.4 KB token_count reads would bring in the text around them."""
    return filesystem.open(path, "rb", cache_type="none")


def read_footer(filesystem, path):
    """One parquet file's footer: its metadata (documents, row groups, each column's size), a few hundred KB."""
    with open_exactly(filesystem, path) as remote_file:
        return pq.ParquetFile(remote_file).metadata


def text_bytes(metadata):
    """The size of a file's text column as the reader decodes it: every row group's, uncompressed, in bytes.
    Each document is its UTF-8 bytes and a 4-byte length."""
    names = [metadata.schema.column(k).name for k in range(metadata.num_columns)]
    text = names.index("text")
    return sum(metadata.row_group(g).column(text).total_uncompressed_size for g in range(metadata.num_row_groups))


def sample_documents(documents, sample_docs):
    """Gives sample_docs documents (text and token_count) from SAMPLE_PLACES places spread across the dataset,
    from the Hub's dataset viewer API: up to 100 rows per request, as JSON, no parquet needed."""
    import requests
    rows, per_place = [], sample_docs // SAMPLE_PLACES
    for place in range(SAMPLE_PLACES):
        start = place * documents // SAMPLE_PLACES
        for offset in range(start, start + per_place, 100):
            response = requests.get("https://datasets-server.huggingface.co/rows", timeout=60,
                                    params={"dataset": DATASET, "config": VIEWER_CONFIG, "split": "train",
                                            "offset": offset, "length": min(100, start + per_place - offset)})
            response.raise_for_status()
            rows += [item["row"] for item in response.json()["rows"]]
    return rows


def exact_token_count(filesystem, paths, footers, workers):
    """The sum of every document's token_count in every file, reading only that column: each task opens its
    file and reads ROW_GROUPS_PER_READ row groups' token_count, workers tasks at a time."""
    def count(task):
        path, metadata, groups = task
        with open_exactly(filesystem, path) as remote_file:
            table = pq.ParquetFile(remote_file, metadata=metadata).read_row_groups(groups, columns=["token_count"])
        return int(table.column("token_count").to_numpy().sum())

    tasks = []
    for path, metadata in zip(paths, footers):
        groups = list(range(metadata.num_row_groups))
        tasks += [(path, metadata, groups[k:k + ROW_GROUPS_PER_READ])
                  for k in range(0, len(groups), ROW_GROUPS_PER_READ)]
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(count, tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--val-tokens", type=millions, default="10M", help="as fineweb.py's")
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--tokens-per-second", default="2830,5632",
                        help="speeds to show the time at, comma-separated")
    parser.add_argument("--sample-docs", type=int, default=500, help="documents to measure and check on")
    parser.add_argument("--exact", action="store_true", help="sum every token_count: ~10,000 small reads")
    parser.add_argument("--workers", type=int, default=32, help="--exact: reads at once")
    args = parser.parse_args()

    from transformers.utils import logging
    logging.set_verbosity_error()                         # documents longer than 1024 tokens are fine here
    from huggingface_hub import HfFileSystem
    filesystem = HfFileSystem()
    started = time.time()

    # 1. the files and their footers
    paths = sorted(filesystem.glob(f"datasets/{DATASET}/{PATTERN}"))
    if not paths:
        raise SystemExit(f"no files match {DATASET}/{PATTERN}")
    parquet_bytes = sum(filesystem.info(path)["size"] for path in paths)
    print(f"{DATASET} {PATTERN}: {len(paths)} files, {parquet_bytes / 1e9:.1f} GB of parquet")
    with ThreadPoolExecutor(len(paths)) as pool:
        footers = list(pool.map(lambda path: read_footer(filesystem, path), paths))
    documents = sum(metadata.num_rows for metadata in footers)
    all_text_bytes = sum(text_bytes(metadata) for metadata in footers)
    for path, metadata in zip(paths, footers):
        print(f"  {path.rsplit('/', 1)[-1]}   {metadata.num_rows:>9,} documents   {metadata.num_row_groups:>5} "
              f"row groups   {text_bytes(metadata) / 1e9:5.2f} GB of text")

    # 2. the sample: tokens per byte, and does token_count count GPT-2's tokens?
    rows = sample_documents(documents, args.sample_docs)
    texts = [row["text"] for row in rows]
    tokenizer = gpt2_tokenizer()
    our_counts = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    their_counts = [row["token_count"] for row in rows]
    sample_bytes = sum(len(text.encode("utf-8")) + 4 for text in texts)     # as text_bytes counts them
    tokens_per_byte = sum(our_counts) / sample_bytes
    matching = sum(ours == theirs for ours, theirs in zip(our_counts, their_counts))
    print(f"\nsample: {len(rows)} documents from {SAMPLE_PLACES} places, tokenized here: "
          f"{sum(our_counts):,} tokens in {sample_bytes:,} bytes, {1 / tokens_per_byte:.2f} bytes per token")
    print(f"check: {matching} of {len(rows)} match token_count ({sum(our_counts):,} tokens here, "
          f"{sum(their_counts):,} in token_count)")

    # 3. the tokens
    if args.exact:
        token_count_sum = exact_token_count(filesystem, paths, footers, args.workers)
        if sum(our_counts) == sum(their_counts):
            content_tokens, how = token_count_sum, "exact: the sum of token_count, GPT-2's count"
        else:
            ratio = sum(our_counts) / sum(their_counts)
            content_tokens = round(token_count_sum * ratio)
            how = f"the sum of token_count x {ratio:.4f}: it is not quite GPT-2's count"
    else:
        content_tokens = round(all_text_bytes * tokens_per_byte)
        how = f"estimated: {all_text_bytes / 1e9:.1f} GB of text x the sample's tokens per byte (--exact: exact)"
    total_tokens = content_tokens + documents             # + one <|endoftext|> (EOT_TOKEN) per document
    train_tokens = total_tokens - args.val_tokens
    steps = train_tokens // args.tokens_per_step

    print(f"\ndocuments                {documents:>16,}")
    print(f"tokens in the text       {content_tokens:>16,}   {how}")
    print(f"+ <|endoftext|> ({EOT_TOKEN}) {documents:>14,}   one after each document")
    print(f"= tokens                 {total_tokens:>16,}   {total_tokens / documents:,.0f} per document")
    print(f"  val (the first)        {args.val_tokens:>16,}")
    print(f"  train                  {train_tokens:>16,}   {2 * train_tokens / 1e9:.1f} GB as token files")
    print(f"steps of {args.tokens_per_step:,} tokens    {steps:>14,}   one epoch")
    print(f"tokenizing it all (fineweb.py, 1 thread, ~{TOKENIZE_TOKENS_PER_SECOND:,} tokens/s): "
          f"{duration_text(total_tokens / TOKENIZE_TOKENS_PER_SECOND)}")
    print("\ntraining one epoch at:")
    for speed in (float(text) for text in args.tokens_per_second.split(",")):
        print(f"  {speed:>12,.0f} tokens/s   {duration_text(steps * args.tokens_per_step / speed):>12}")
    print(f"\n(read in {time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
