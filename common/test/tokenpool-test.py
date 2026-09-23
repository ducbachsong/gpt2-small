"""test — run tokenpool on real data, with many threads, and print what happened.

    python test/tokenpool-test.py

It downloads two small parquet files (~69 MB, babylm's test and validation
splits) into RAM through a parquet pool, and tokenizes them with the real gpt2
tokenizer, so it needs a working network connection the first time. Nothing
is mocked: if this prints "all good", the batches are real and a model can
train on them.

The parquet pool is created here by hand, then handed to the token pool —
tokenpool never creates one itself.

Each case is one function, run in order by main(): the pools the second one
creates are the pools the next ones use. Every case prints what it found, so
a failure shows you where it stopped instead of just saying False.
"""
import os
import sys
import threading
import time

import numpy as np
import torch

# The modules under test live one folder up, next to this test folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parquetpool import MB, ParquetPool, close_parquet_pool
from tokenpool import TokenPool, close_token_pool

DATASET = "nilq/babylm-100M"
ALLOW_PATTERNS = ["data/test-*.parquet", "data/validation-*.parquet"]
PARQUET_RAM_LIMIT = 80 * MB     # room for both files (35.9 + 33.0 MB) at once
TOKENIZER = "gpt2"
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4
EXPECTED_MAX_BATCHES_IN_RAM = 8     # small, so the pool fills up;
TOKEN_RAM_LIMIT = EXPECTED_MAX_BATCHES_IN_RAM * BATCH_SIZE * SEQUENCE_LENGTH * 4
                                    # the limit that holds exactly that many:
                                    # 8 x 2,048 bytes
TOKENIZE_THREAD_COUNT = 4
THREAD_COUNT = 8
END_OF_TEXT_ID = 50256              # gpt2's token between documents


# ── helpers ─────────────────────────────────────────────────────────────────
def token_batches_in_ram(token_pool):
    """Token batches the pool holds ready right now."""
    with token_pool._token_cond:
        return len(token_pool._ready_token_batches)


def create_token_pool(parquet_pool):
    """TokenPool reading from `parquet_pool`, with this test's settings."""
    return TokenPool(parquet_pool, tokenizer=TOKENIZER,
                     sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
                     token_ram_limit=TOKEN_RAM_LIMIT,
                     tokenize_thread_count=TOKENIZE_THREAD_COUNT)


# ── the cases ───────────────────────────────────────────────────────────────

def test_forgot_to_create_raises():
    """TokenPool() and ParquetPool() before creating them raise ValueError."""
    print("\n=== 1. both pools used before creating them ===")
    for pool_call, use_pool in (("ParquetPool()", ParquetPool),
                                ("TokenPool()", TokenPool)):
        try:
            use_pool()
            raise AssertionError(f"{pool_call} worked without creating it first")
        except ValueError as error:
            print(f"    {pool_call} raises ValueError: {error}")


def test_only_one_token_pool():
    """The parquet pool is created by hand; then many threads create the token
    pool at once, and they must all get the same one.

    Out: (parquet_pool, token_pool), for the cases after this one.
    """
    print(f"\n=== 2. parquet pool first, then {THREAD_COUNT} threads create the "
          f"token pool at once ===")
    parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                               ram_limit=PARQUET_RAM_LIMIT)
    created_pools = []
    start_line = threading.Barrier(THREAD_COUNT)

    def creator():
        start_line.wait()                   # all of them go at once
        created_pools.append(create_token_pool(parquet_pool))

    threads = [threading.Thread(target=creator) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    token_pool = created_pools[0]
    assert len(created_pools) == THREAD_COUNT, "a creator hung"
    assert all(created is token_pool for created in created_pools),"more than one token pool was created"
    assert token_pool.parquet_pool is parquet_pool,"the token pool is not reading the parquet pool it was given"
    assert token_pool.max_batches_in_ram == EXPECTED_MAX_BATCHES_IN_RAM, f"a {TOKEN_RAM_LIMIT:,}-byte limit gave max_batches_in_ram="f"{token_pool.max_batches_in_ram}, not {EXPECTED_MAX_BATCHES_IN_RAM}"

    asked_again = TokenPool(sequence_length=99)     # no parquet pool needed now
    assert asked_again is token_pool, "asking again gave a new token pool"
    assert token_pool.sequence_length == SEQUENCE_LENGTH,"asking again changed the token pool's settings"
    return parquet_pool, token_pool


def test_token_batch_is_real_tokens():
    """One token batch has the right shape and type, and decodes back to text.

    Out: the batch, for the training case.
    """
    print("\n=== 3. one token batch, looked at closely ===")
    batch = TokenPool().get_token_batch(timeout=600)

    assert batch is not None, "the token pool ended before giving a single batch"
    assert batch.shape == (BATCH_SIZE, SEQUENCE_LENGTH), f"wrong shape {batch.shape}"
    assert batch.dtype == np.int32, f"wrong dtype {batch.dtype}"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    assert batch.min() >= 0, "a negative token id"
    assert batch.max() < len(tokenizer), "a token id outside the vocabulary"
    first_row_text = tokenizer.decode(batch[0])
    assert first_row_text.strip(), "row 0 decodes to nothing"
    return batch


def test_model_trains_on_token_batch(batch):
    """A small gpt2-shaped model takes the batch as it is and learns from it.

    The model is built from a config, not downloaded: only the batch is real.
    """
    print("\n=== 4. a model trains on that token batch ===")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model_config = GPT2Config(vocab_size=50257, n_positions=SEQUENCE_LENGTH,
                              n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.from_numpy(batch).long()
    losses = []
    for _ in range(5):                      # the same batch, five steps
        loss = model(input_ids=input_ids, labels=input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), "the loss is not a number"
    assert losses[-1] < losses[0], "the loss did not go down"


def test_threads_never_share_token_batches(parquet_pool, token_pool):
    """Many threads call get_token_batch() until it is empty; no batch goes twice."""
    print(f"\n=== 5. {THREAD_COUNT} threads call get_token_batch() until it is empty ===")
    most_batches_in_ram = 0
    still_watching = True

    def watch_ram():
        nonlocal most_batches_in_ram
        while still_watching:
            most_batches_in_ram = max(most_batches_in_ram,
                                      token_batches_in_ram(token_pool))
            time.sleep(0.001)

    watcher = threading.Thread(target=watch_ram, daemon=True)
    watcher.start()

    received_batches = []                   # kept alive, so ids stay unique
    received_lock = threading.Lock()

    def trainer():
        while (batch := TokenPool().get_token_batch(timeout=600)) is not None:
            with received_lock:
                received_batches.append(batch)

    threads = [threading.Thread(target=trainer, name=f"trainer-{thread_number}")
               for thread_number in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1800)
    still_watching = False
    end_of_text_count = sum(int((batch == END_OF_TEXT_ID).sum())
                            for batch in received_batches)

    assert not any(thread.is_alive() for thread in threads), "a trainer hung"
    assert received_batches, "no token batches came out"
    assert len({id(batch) for batch in received_batches}) == len(received_batches), \
        "two threads got the same token batch"
    assert all(batch.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
               for batch in received_batches), "a token batch has the wrong shape"
    assert end_of_text_count > 0, "no end-of-text token between documents"
    assert parquet_pool.skipped_files == [], "a parquet file failed to download"
    assert token_pool.untokenized_files == [], "a parquet file failed to tokenize"
    assert most_batches_in_ram <= EXPECTED_MAX_BATCHES_IN_RAM, \
        f"held {most_batches_in_ram} token batches, more than max_batches_in_ram"


def test_empty_pools_stay_empty(parquet_pool):
    """After the last token batch, both pools keep returning None."""
    print("\n=== 6. after the last token batch ===")
    assert TokenPool().get_token_batch() is None, "an empty token pool gave another batch"
    assert parquet_pool.get_parquet_file() is None, "the parquet pool still had a file"


def test_close_token_pool_starts_fresh(parquet_pool, token_pool):
    """close_token_pool() ends it: taking raises, the next TokenPool() is new."""
    print("\n=== 7. close_token_pool(), then create again ===")
    token_pool.close_token_pool()
    try:
        token_pool.get_token_batch()
        raise AssertionError("get_token_batch() on a closed pool did not raise")
    except RuntimeError:
        print("    get_token_batch() on the closed pool raises RuntimeError")
    assert not any(thread.is_alive() for thread in token_pool._tokenize_threads), \
        "close_token_pool() returned with tokenize threads still running"
    try:
        TokenPool()
        raise AssertionError("TokenPool() after closing found a pool")
    except ValueError:
        print("    TokenPool() after closing raises ValueError: create again")

    # The parquet pool is empty, so the new pool ends at once: no download.
    new_token_pool = create_token_pool(parquet_pool)

    assert new_token_pool is not token_pool, "close_token_pool() did not let a new pool start"
    assert new_token_pool.get_token_batch(timeout=60) is None, \
        "the new token pool found data to read"
    return new_token_pool


def test_idle_pools_stay_alive(token_pool, idle_seconds=10):
    """Nobody takes a batch for a while; both pools must still be there after.

    Run while the token pool is full, so its threads sit waiting for room.
    """
    print(f"\n=== idle: nobody takes a batch for {idle_seconds}s ===")
    time.sleep(idle_seconds)
    alive_thread_count = sum(thread.is_alive() for thread in token_pool._tokenize_threads)
    print(f"    {alive_thread_count} of {len(token_pool._tokenize_threads)} "
          f"tokenize threads still alive")
    assert alive_thread_count, "the tokenize threads ended while the pool sat idle"
    assert TokenPool().get_token_batch(timeout=60) is not None, \
        "the pool gave nothing after sitting idle"


def test_close_token_pool_only(parquet_pool, token_pool):
    """close_token_pool() ends the token pool and its threads, and nothing else."""
    print("\n=== 8. close_token_pool() ===")
    close_token_pool()
    assert not any(thread.is_alive() for thread in token_pool._tokenize_threads), \
        "tokenize threads still running after close_token_pool()"
    try:
        token_pool.get_token_batch()
        raise AssertionError("the token pool still works after close_token_pool()")
    except RuntimeError:
        pass
    # Still open: an empty parquet pool answers None, a closed one would raise.
    assert parquet_pool.get_parquet_file() is None, \
        "close_token_pool() closed the parquet pool"


def test_close_parquet_pool(parquet_pool):
    """close_parquet_pool() ends the parquet pool."""
    print("\n=== 9. close_parquet_pool() ===")
    close_parquet_pool()
    try:
        parquet_pool.get_parquet_file()
        raise AssertionError("the parquet pool still works after close_parquet_pool()")
    except RuntimeError:
        print("    parquet pool closed")


def main():
    test_forgot_to_create_raises()
    parquet_pool, token_pool = test_only_one_token_pool()
    batch = test_token_batch_is_real_tokens()
    test_model_trains_on_token_batch(batch)
    test_idle_pools_stay_alive(token_pool)
    test_threads_never_share_token_batches(parquet_pool, token_pool)
    test_empty_pools_stay_empty(parquet_pool)
    new_token_pool = test_close_token_pool_starts_fresh(parquet_pool, token_pool)
    test_close_token_pool_only(parquet_pool, new_token_pool)
    test_close_parquet_pool(parquet_pool)

    print("\nall good\n")


if __name__ == "__main__":
    main()
