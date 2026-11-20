"""chat_sft.py — the chat data the fine-tune (finetune_8bit.cu) trains on, as nanochat's SFT (scripts/chat_sft.py):
SmolTalk's conversations once, MMLU's auxiliary_train 3 times, GSM8K's train 4 times, shuffled; each conversation in
nanochat's chat format with its loss mask, packed best-fit into rows of seq_len + 1 tokens, and written as two int32
lists a row: the input tokens and the targets (-1 where the mask is 0: not graded).

    python python/chat_sft.py                          # python/tokenizer/data/chat_sft/chat_sft_train.bin and chat_sft_val.bin
    python python/chat_sft.py --workers 16 --seq-len 2048

A conversation, as nanochat's tokenizer.render_conversation writes it:
    <|bos|> <|user_start|> question <|user_end|> <|assistant_start|> answer <|assistant_end|> <|user_start|> ...
mask (1: graded, the model learns to write it): the assistant's answer and its <|assistant_end|>; never <|bos|>, the
user's words, <|user_start|>, <|user_end|>, <|assistant_start|>, or a tool's output. GSM8K's calculator steps
<<48/2=24>> become <|python_start|>48/2<|python_end|> (graded: the model writes the call) then
<|output_start|>24<|output_end|> (not graded: the calculator writes it). A system message goes in front of the first
user message. A conversation longer than a row is cut to the row.

The rows, nanochat's best-fit packing: from a buffer of the next 100 conversations the largest that fits the room
left goes in, again and again, until none fits; the rest of the row is <|bos|> padding, not graded. A conversation is
never split across two rows. Row r of the file: tokens[t] = row[t], targets[t] = row[t + 1] if graded else -1,
t = 0 .. seq_len - 1.

The data, from the Hugging Face Hub (public datasets, no token needed):
    train   HuggingFaceTB/smol-smoltalk train (~460K), cais/mmlu all/auxiliary_train (~100K) x 3, openai/gsm8k main
            train (~7.5K) x 4
    val     smol-smoltalk test, MMLU test's first 5,200, GSM8K test's first 420 (nanochat's val mixture)

File (cuda/helper/chat_loader.cuh reads it): a header of 256 int32s ([0] magic 20261119, [1] version 1, [2] rows,
[3] seq_len, [4] graded targets, [5] conversations), then each row's seq_len tokens and seq_len targets, int32.
"""
import argparse
import glob
import multiprocessing
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # tokenizer/, also when imported from python/tests/
from tokenizer.tokenizer import DATA, TOKENIZER_PICKLE, load_tokenizer  # noqa: E402

MAGIC, VERSION, HEADER_INTS = 20261119, 1, 256
IGNORE_TARGET = -1                       # SOFTMAX_LOSS_FINETUNE_BF16_IGNORE_TARGET: a target the loss leaves out
PACKING_BUFFER = 100                     # nanochat's best-fit buffer: conversations looked at for each place
SEED = 42                                # the mixture's shuffle
MMLU_EPOCHS, GSM8K_EPOCHS = 3, 4         # nanochat's chat_sft.py defaults (--mmlu-epochs, --gsm8k-epochs)
VAL_MMLU_ROWS, VAL_GSM8K_ROWS = 5200, 420
MMLU_LETTERS = "ABCD"
DATASETS = {                             # repo: the parquet files to fetch
    "HuggingFaceTB/smol-smoltalk": ["data/*.parquet"],
    "cais/mmlu": ["all/auxiliary_train-*.parquet", "all/test-*.parquet"],
    "openai/gsm8k": ["main/*.parquet"],
}

# ── the conversations ─────────────────────────────────────────────────────────


def read_parquet(folder, pattern):
    """Gives every row of the parquet files under `folder` matching `pattern`, as dicts, files in name order; stops if
    there is none (the dataset's layout changed)."""
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f"no files {pattern} in {folder}")
    rows = []
    for path in paths:
        rows += pq.read_table(path).to_pylist()
    return rows


def smoltalk_conversations(rows):
    """Gives SmolTalk's rows as conversations: their messages, a system message first or none, then user and
    assistant turns one after the other (rows that do not: left out, as nanochat's checks would stop on them)."""
    conversations = []
    for row in rows:
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        turns = messages[1:] if messages and messages[0]["role"] == "system" else messages
        roles_ok = len(turns) >= 2 and all(m["role"] == ("user" if k % 2 == 0 else "assistant") for k, m in enumerate(turns))
        if roles_ok:
            conversations.append(messages)
    return conversations


def mmlu_conversations(rows):
    """Gives MMLU's rows as one-turn conversations, nanochat's tasks/mmlu.py: the question with its choices as
    `- choice=letter` lines, then the answer's letter alone. auxiliary_train's rows may hold their fields in a `train`
    column: taken from it."""
    conversations = []
    for row in rows:
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        query = f"Multiple Choice question: {row['question']}\n"
        query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(MMLU_LETTERS, row["choices"]))
        query += "\nRespond only with the letter of the correct answer."
        conversations.append([{"role": "user", "content": query},
                              {"role": "assistant", "content": MMLU_LETTERS[int(row["answer"])]}])
    return conversations


def gsm8k_conversations(rows):
    """Gives GSM8K's rows as one-turn conversations, nanochat's tasks/gsm8k.py: the answer cut into text parts and
    calculator steps, each <<expression=result>> a python part (the expression) and its python_output (the result)."""
    conversations = []
    for row in rows:
        parts = []
        for piece in re.split(r"(<<[^>]+>>)", row["answer"]):
            if piece.startswith("<<") and piece.endswith(">>"):
                expression, _, result = piece[2:-2].rpartition("=") if "=" in piece else (piece[2:-2], "", "")
                parts.append({"type": "python", "text": expression})
                parts.append({"type": "python_output", "text": result})
            elif piece:
                parts.append({"type": "text", "text": piece})
        conversations.append([{"role": "user", "content": row["question"]}, {"role": "assistant", "content": parts}])
    return conversations


# ── tokens and mask ───────────────────────────────────────────────────────────

ENCODING, SPECIAL, CAPACITY = None, None, None


def start_worker(tokenizer_path, capacity):
    """Loads the tokenizer once in each worker process, with the chat tokens' ids and a row's length."""
    global ENCODING, SPECIAL, CAPACITY
    ENCODING = load_tokenizer(tokenizer_path)
    SPECIAL = {name.strip("<|>"): token_id for name, token_id in ENCODING._special_tokens.items()}
    CAPACITY = capacity


def render(messages):
    """Gives a conversation's (token ids, mask) as int32 and uint8 arrays, nanochat's render_conversation: a system
    message put in front of the first user message; user turns not graded; an assistant's <|assistant_start|> not
    graded, its words, python calls and <|assistant_end|> graded, a python call's output not; cut to a row."""
    ids, mask = [SPECIAL["bos"]], [0]

    def add(token_ids, graded):
        ids.extend(token_ids)
        mask.extend([graded] * len(token_ids))

    if messages[0]["role"] == "system":
        messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            add([SPECIAL["user_start"]] + ENCODING.encode_ordinary(message["content"]) + [SPECIAL["user_end"]], 0)
        else:
            add([SPECIAL["assistant_start"]], 0)
            content = message["content"]
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for part in parts:
                text_ids = ENCODING.encode_ordinary(part["text"])
                if part["type"] == "text":
                    add(text_ids, 1)
                elif part["type"] == "python":
                    add([SPECIAL["python_start"]] + text_ids + [SPECIAL["python_end"]], 1)
                else:                                    # python_output: the calculator's, not the model's
                    add([SPECIAL["output_start"]] + text_ids + [SPECIAL["output_end"]], 0)
            add([SPECIAL["assistant_end"]], 1)
    return np.array(ids[:CAPACITY], dtype=np.int32), np.array(mask[:CAPACITY], dtype=np.uint8)


# ── rows ──────────────────────────────────────────────────────────────────────


def write_rows(path, rendered, seq_len, bos_id):
    """Packs the rendered conversations best-fit into rows of seq_len + 1 tokens, in their order, and writes the file.
    Gives (rows, tokens of the conversations, graded targets)."""
    capacity = seq_len + 1
    rows = graded_total = conversation_tokens = 0
    buffer, upcoming = [], iter(rendered)
    with open(path + ".partial", "wb") as out_file:
        out_file.write(np.zeros(HEADER_INTS, dtype=np.int32).tobytes())
        while True:
            # 1. a row: the largest conversation of the buffer that fits the room left, until none fits
            while len(buffer) < PACKING_BUFFER and (item := next(upcoming, None)) is not None:
                buffer.append(item)
            if not buffer:
                break
            row_ids, row_mask, room = [], [], capacity
            while True:
                best_index, best_length = -1, 0
                for index, (ids, _) in enumerate(buffer):
                    if best_length < len(ids) <= room:
                        best_index, best_length = index, len(ids)
                if best_index < 0:
                    break
                ids, mask = buffer.pop(best_index)
                row_ids.append(ids), row_mask.append(mask)
                room -= len(ids)
                conversation_tokens += len(ids)
                if (item := next(upcoming, None)) is not None:
                    buffer.append(item)
            # 2. the rest <|bos|>, not graded; the targets: the next token where graded, else -1
            row_ids.append(np.full(room, bos_id, dtype=np.int32)), row_mask.append(np.zeros(room, dtype=np.uint8))
            row = np.concatenate(row_ids)
            graded = np.concatenate(row_mask)[1:] == 1
            targets = np.where(graded, row[1:], IGNORE_TARGET).astype(np.int32)
            out_file.write(row[:seq_len].tobytes())
            out_file.write(targets.tobytes())
            rows += 1
            graded_total += int(graded.sum())
        # 3. the header, now the counts are known
        if graded_total >= 2 ** 31:
            raise SystemExit("more graded targets than an int32 holds: the header needs another field")
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[:6] = [MAGIC, VERSION, rows, seq_len, graded_total, len(rendered)]
        out_file.seek(0)
        out_file.write(header.tobytes())
    os.replace(path + ".partial", path)
    return rows, conversation_tokens, graded_total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(DATA, "chat_sft"), help="the folder for the two files")
    parser.add_argument("--seq-len", type=int, default=2048, help="the trainer's --seq-len: tokens a row")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="tokenizer processes")
    parser.add_argument("--tokenizer", default=TOKENIZER_PICKLE)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    # 1. the datasets' parquet files
    folders = {}
    for repo, patterns in DATASETS.items():
        folders[repo] = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                                          local_dir=os.path.join(args.out, "hub", repo.replace("/", "__")))
    print(f"downloaded in {time.time() - started:.0f} s", flush=True)

    # 2. the conversations: train (shuffled) and val
    smoltalk = folders["HuggingFaceTB/smol-smoltalk"]
    mmlu, gsm8k = folders["cais/mmlu"], folders["openai/gsm8k"]
    parts = {
        "train": [("smoltalk", smoltalk_conversations(read_parquet(smoltalk, "data/train-*.parquet")), 1),
                  ("mmlu auxiliary_train", mmlu_conversations(read_parquet(mmlu, "all/auxiliary_train-*.parquet")), MMLU_EPOCHS),
                  ("gsm8k train", gsm8k_conversations(read_parquet(gsm8k, "main/train-*.parquet")), GSM8K_EPOCHS)],
        "val": [("smoltalk test", smoltalk_conversations(read_parquet(smoltalk, "data/test-*.parquet")), 1),
                ("mmlu test", mmlu_conversations(read_parquet(mmlu, "all/test-*.parquet"))[:VAL_MMLU_ROWS], 1),
                ("gsm8k test", gsm8k_conversations(read_parquet(gsm8k, "main/test-*.parquet"))[:VAL_GSM8K_ROWS], 1)],
    }

    # 3. each split: tokens and mask in parallel, then the rows
    bos_id = load_tokenizer(args.tokenizer).encode_single_token("<|bos|>")
    with multiprocessing.Pool(args.workers, initializer=start_worker, initargs=(args.tokenizer, args.seq_len + 1)) as pool:
        for split, split_parts in parts.items():
            conversations = []
            for name, part, epochs in split_parts:
                print(f"{split}: {name} {len(part):,} conversations x {epochs}", flush=True)
                conversations += part * epochs
            random.Random(SEED).shuffle(conversations)
            rendered = pool.map(render, conversations, chunksize=1000)
            path = os.path.join(args.out, f"chat_sft_{split}.bin")
            rows, tokens, graded = write_rows(path, rendered, args.seq_len, bos_id)
            print(f"{split}: {len(conversations):,} conversations, {tokens:,} tokens in {rows:,} rows of {args.seq_len} "
                  f"({rows * args.seq_len:,} places, {100.0 * tokens / (rows * (args.seq_len + 1)):.1f}% used), "
                  f"{graded:,} graded targets ({100.0 * graded / (rows * args.seq_len):.1f}%) -> {path} "
                  f"({time.time() - started:.0f} s)", flush=True)
            if split == "train":
                rows_per_step = 524288 // args.seq_len
                print(f"train: one pass = {rows // rows_per_step:,} steps of 524,288 tokens ({rows_per_step} rows)", flush=True)


if __name__ == "__main__":
    main()
