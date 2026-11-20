"""test_chat_model.py — tests a chat model folder (python/export_safetensors.py's, e.g. nanochat-d24-sft) in plain
PyTorch (safetensors_model.Model): that it answers in nanochat's chat format and stops, gets two easy facts right, and
beats chance on MMLU. Every check prints its test case, input, expected and actual value, and ok; the first FAIL
stops the program.

    python python/tests/test_chat_model.py --model nanochat-d24-sft
    python python/tests/test_chat_model.py --model python/tokenizer/data/finetune_init/nanochat-d24     # the base model: FAILs part 1

Part 1 checks the chat format: each question, asked as
    <|bos|><|user_start|>question<|user_end|><|assistant_start|>
gets a greedy answer (always the most likely token) that ends with <|assistant_end|> within MAX_ANSWER_TOKENS and does
not just repeat the question. Part 2 checks two facts any chat model of this size should know. Part 3 checks MMLU: the
last MMLU_QUESTIONS questions of the test split (its first 5,200 are the fine-tune's val rows; neither part is trained
on), right when the highest of the letters A B C D's logits after <|assistant_start|> is the answer's.

The numbers to expect, from the first fine-tune (2026-11-19): nanochat-d24-sft passes every check, MMLU 35.6%; the
base nanochat-d24 never stops (it repeats the question or one sentence until the limit) and scores 25.3% (chance).
"""
import argparse
import functools
import glob
import inspect
import os
import sys

import torch

# The modules under test live one folder up, next to this test folder.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from chat_sft import mmlu_conversations  # noqa: E402  (MMLU's questions, written as the fine-tune saw them)
from tokenizer.tokenizer import DATA, TOKENIZER_PICKLE, load_tokenizer  # noqa: E402
from safetensors_model import Model  # noqa: E402

MAX_ANSWER_TOKENS = 200             # an answer longer than this has not stopped
MMLU_QUESTIONS = 1000               # from the end of the test split
MMLU_SMALLEST_ACCURACY = 0.30       # chance is 0.25; the first fine-tune got 0.356
MMLU_LETTERS = "ABCD"
QUESTIONS = [
    "Hi! Who are you?",
    "Why is the sky blue?",
    "Write a short poem about the sea.",
    "What is the capital of France, and what is it famous for?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips "
    "did Natalia sell altogether in April and May?",
]

MODEL, ENCODING, SPECIAL = None, None, None     # main() loads them once; every test reads them

# ── ASSERT_EQ: print every check, stop on a FAIL ─────────────────────────────

LAST_TEST = [None]


def print_test_name(test):
    """Prints the test case's name, once, before its first check."""
    if test != LAST_TEST[0]:
        print(f"\n{test}", flush=True)
    LAST_TEST[0] = test


def ASSERT_EQ(actual, expected):
    """Checks one value, as cuda/tests/common/test_helpers.h's ASSERT_EQ: prints the test case (the calling function), the checked
    expression (from the call's line), the expected and the actual value, ok or FAIL; stops the program on a FAIL."""
    caller = inspect.stack()[1]
    print_test_name(caller.function)
    line = caller.code_context[0].strip() if caller.code_context else "?"
    expression = line[line.find("(") + 1:line.rfind(",")] if line.startswith("ASSERT_EQ(") else line
    equal = actual == expected
    print(f"    input: {expression:<36} expected: {expected}   actual: {actual}   {'ok' if equal else 'FAIL'}  "
          f"(line {caller.lineno})", flush=True)
    assert equal, "the values are printed above"


# ── helpers ───────────────────────────────────────────────────────────────────


def prompt_ids(question):
    """Gives the token ids of a one-question chat, up to the assistant's start."""
    return ([SPECIAL["<|bos|>"], SPECIAL["<|user_start|>"]] + ENCODING.encode_ordinary(question) +
            [SPECIAL["<|user_end|>"], SPECIAL["<|assistant_start|>"]])


@functools.lru_cache(maxsize=None)
def answer_of(question):
    """Gives the model's greedy answer to `question` as (text, stopped): its tokens until <|assistant_end|> (stopped
    True) or MAX_ANSWER_TOKENS of them (stopped False); a special token inside shows as its name. Worked out once per
    question, for every test that reads it, and printed under the first test's name."""
    print_test_name(inspect.stack()[1].function)
    ids, out, stopped = prompt_ids(question), [], False
    with torch.no_grad():
        for _ in range(MAX_ANSWER_TOKENS):
            context = ids[-MODEL.config["n_positions"]:]
            logits = MODEL.logits(torch.tensor([context], device=MODEL.device))[0, -1, :MODEL.config["vocab_size"]]
            next_id = int(logits.argmax())
            if next_id == SPECIAL["<|assistant_end|>"]:
                stopped = True
                break
            ids.append(next_id)
            out.append(next_id)
    names = {token_id: name for name, token_id in SPECIAL.items()}
    text = "".join(names[token_id] if token_id in names else ENCODING.decode([token_id]) for token_id in out)
    print(f"    answer to {question[:40]!r}: {text[:150]!r}{' ...' if len(text) > 150 else ''}", flush=True)
    return text, stopped


def mmlu_test_parquet():
    """Gives MMLU's all/test parquet: the one python/chat_sft.py downloaded, or downloads it the same way."""
    folder = os.path.join(DATA, "chat_sft", "hub", "cais__mmlu")
    paths = sorted(glob.glob(os.path.join(folder, "all", "test-*.parquet")))
    if not paths:
        from huggingface_hub import snapshot_download
        snapshot_download("cais/mmlu", repo_type="dataset", allow_patterns=["all/test-*.parquet"], local_dir=folder)
        paths = sorted(glob.glob(os.path.join(folder, "all", "test-*.parquet")))
    return paths[0]


# ══ part 1: the chat format ══════════════════════════════════════════════════


def every_answer_stops():
    # Expect: each of the 5 questions gets an answer that ends with <|assistant_end|> within 200 tokens. The base model
    # FAILs here: it never writes <|assistant_end|>.
    for question in QUESTIONS:
        text, stopped = answer_of(question)
        ASSERT_EQ(stopped, True)


def no_answer_repeats_its_question():
    # Expect: no answer starts by repeating its question (what the base model does), and none is empty.
    for question in QUESTIONS:
        text, stopped = answer_of(question)
        ASSERT_EQ(text.strip().startswith(question[:30]), False)
        ASSERT_EQ(len(text.strip()) > 0, True)


# ══ part 2: two easy facts ═══════════════════════════════════════════════════


def capital_of_france_is_paris():
    # Expect: the answer to "What is the capital of France, ..." names Paris.
    text, stopped = answer_of(QUESTIONS[3])
    ASSERT_EQ("Paris" in text, True)


def sky_is_blue_from_rayleigh_scattering():
    # Expect: the answer to "Why is the sky blue?" names Rayleigh scattering.
    text, stopped = answer_of(QUESTIONS[1])
    ASSERT_EQ("Rayleigh" in text, True)


# ══ part 3: MMLU ═════════════════════════════════════════════════════════════


def mmlu_beats_chance():
    # Expect: of MMLU test's last 1,000 questions, more than 30% right (chance 25%; the first fine-tune 35.6%, the base
    # model 25.3%), each the letter of A B C D with the highest logit right after <|assistant_start|>.
    import pyarrow.parquet as pq
    rows = pq.read_table(mmlu_test_parquet()).to_pylist()[-MMLU_QUESTIONS:]
    letter_ids = [ENCODING.encode_ordinary(letter)[0] for letter in MMLU_LETTERS]
    right = 0
    with torch.no_grad():
        for row, conversation in zip(rows, mmlu_conversations(rows)):
            ids = prompt_ids(conversation[0]["content"])[-MODEL.config["n_positions"]:]
            logits = MODEL.logits(torch.tensor([ids], device=MODEL.device))[0, -1]
            right += int(logits[letter_ids].argmax()) == int(row["answer"])
    accuracy = right / len(rows)
    ASSERT_EQ(len(rows), MMLU_QUESTIONS)
    print(f"    {right} of {len(rows)} right: {100.0 * accuracy:.1f}%", flush=True)
    ASSERT_EQ(accuracy > MMLU_SMALLEST_ACCURACY, True)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    global MODEL, ENCODING, SPECIAL
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="a python/export_safetensors.py folder")
    parser.add_argument("--tokenizer", default=TOKENIZER_PICKLE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ENCODING = load_tokenizer(args.tokenizer)
    SPECIAL = dict(ENCODING._special_tokens)
    MODEL = Model(args.model, args.device)
    print(f"model: {args.model} ({MODEL.config.get('training', {})})")

    every_answer_stops()
    no_answer_repeats_its_question()

    capital_of_france_is_paris()
    sky_is_blue_from_rayleigh_scattering()

    mmlu_beats_chance()

    print("\nall tests passed")


if __name__ == "__main__":
    main()
