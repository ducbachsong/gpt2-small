"""test — run parquetpool against the real Hub, with many threads, and print what happened.

    python test/parquetpool-test.py

It downloads two small parquet files (~69 MB, babylm's test and validation
splits) into RAM, so it needs a working network connection. Nothing is
mocked: if this prints "all good", the pool works against the live Hub.

Each case is one function, run in order by main(): the pool the first one
creates is the pool the next ones use. Every case prints what it found, so a
failure shows you where it stopped instead of just saying False.
"""
import glob
import os
import sys
import tempfile
import threading
import time

import pyarrow
import pyarrow.parquet as pq

# The module under test lives one folder up, next to this test folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parquetpool import MB, ParquetPool

DATASET = "nilq/babylm-100M"
ALLOW_PATTERNS = ["data/test-*.parquet", "data/validation-*.parquet"]
RAM_LIMIT = 50 * MB                 # the largest file is 35.9 MB, so this fits one:
EXPECTED_MAX_FILES_IN_RAM = 1       # every get_parquet_file() refills
THREAD_COUNT = 8

EXPECTED_FILES = ["data/test-00000-of-00001.parquet",
                  "data/validation-00000-of-00001.parquet"]


def temp_folders():
    """The pool's temporary download folders that exist right now."""
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "parquetpool-*")))


# ── the cases ───────────────────────────────────────────────────────────────
def test_forgot_to_create_raises():
    """ParquetPool() before any pool was created raises, and creates nothing."""
    print("\n=== 1. ParquetPool() before creating one ===")
    try:
        ParquetPool()
        raise AssertionError("ParquetPool() worked without creating a pool first")
    except ValueError as error:
        print(f"    raises ValueError: {error}")


def test_only_one_pool():
    """Many threads create the pool at once; they must all get the same one.

    Out: the pool, for the cases after this one.
    """
    created_pools = []
    start_line = threading.Barrier(THREAD_COUNT)

    def creator():
        start_line.wait()                   # all of them go at once
        created_pools.append(ParquetPool(dataset=DATASET,
                                         allow_patterns=ALLOW_PATTERNS,
                                         ram_limit=RAM_LIMIT))

    threads = [threading.Thread(target=creator) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    parquet_pool = created_pools[0]

    assert len(created_pools) == THREAD_COUNT, "a creator hung"
    assert all(created is parquet_pool for created in created_pools), \
        "more than one pool was created"

    assert parquet_pool.max_files_in_ram == EXPECTED_MAX_FILES_IN_RAM, \
        f"a {RAM_LIMIT // MB} MB limit gave max_files_in_ram=" \
        f"{parquet_pool.max_files_in_ram}, not {EXPECTED_MAX_FILES_IN_RAM}"

    asked_again = ParquetPool(ram_limit=99 * MB)  # settings are ignored now
    assert asked_again is parquet_pool, "asking again gave a new pool"
    assert parquet_pool.ram_limit == RAM_LIMIT, "asking again changed the RAM limit"
    return parquet_pool


def test_threads_never_share(parquet_pool):
    """Many threads call get_parquet_file() until it is empty; no file goes twice."""
    most_files_in_ram = 0
    still_watching = True

    def watch_ram():
        nonlocal most_files_in_ram
        while still_watching:
            with parquet_pool._parquet_cond:
                files_in_ram = (parquet_pool._files_not_handed_out
                                - len(parquet_pool._files_to_download))
            most_files_in_ram = max(most_files_in_ram, files_in_ram)
            time.sleep(0.001)

    watcher = threading.Thread(target=watch_ram, daemon=True)
    watcher.start()

    received_files = []
    received_lock = threading.Lock()

    def reader():
        while (parquet_file := ParquetPool().get_parquet_file(timeout=600)) is not None:
            file_name, file_bytes = parquet_file
            table = pq.read_table(pyarrow.BufferReader(file_bytes))
            print(f"    {threading.current_thread().name} got {file_name}: "
                  f"{table.num_rows:,} rows")
            assert table.num_rows > 0, f"{file_name} is empty"
            with received_lock:
                received_files.append(file_name)

    threads = [threading.Thread(target=reader, name=f"reader-{thread_number}")
               for thread_number in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=900)
    still_watching = False

    assert not any(thread.is_alive() for thread in threads), "a reader hung"
    assert len(received_files) == len(set(received_files)), \
        "two threads got the same file"
    assert sorted(received_files) == EXPECTED_FILES, \
        f"expected {EXPECTED_FILES}, got {sorted(received_files)}"
    assert parquet_pool.skipped_files == [], \
        f"files were skipped: {parquet_pool.skipped_files}"
    assert most_files_in_ram <= EXPECTED_MAX_FILES_IN_RAM, \
        f"held {most_files_in_ram} files, more than max_files_in_ram"


def test_empty_pool_leaves_nothing(folders_before):
    """After the last file, get_parquet_file() keeps returning None, and no
    temporary folder is left on disk.

    In : folders_before  set[str]   temp_folders() from before the pool existed
    """
    assert ParquetPool().get_parquet_file() is None, \
        "an empty pool handed out another file"
    folders_left_behind = temp_folders() - folders_before
    assert not folders_left_behind, \
        f"temporary folders left behind: {folders_left_behind}"


def test_close_starts_fresh(parquet_pool):
    """close_parquet_pool() ends the pool: taking raises, the next one is new."""
    print("\n=== 4. close_parquet_pool(), then ask again ===")
    parquet_pool.close_parquet_pool()
    try:
        parquet_pool.get_parquet_file()
        raise AssertionError("get_parquet_file() on a closed pool did not raise")
    except RuntimeError:
        print("    get_parquet_file() on the closed pool raises RuntimeError")

    try:
        ParquetPool()
        raise AssertionError("ParquetPool() after closing found a pool")
    except ValueError:
        print("    ParquetPool() after closing raises ValueError: create again")

    new_parquet_pool = ParquetPool(dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                                   ram_limit=RAM_LIMIT)
    print(f"    ParquetPool(dataset=...) after closing is a new pool: "
          f"{new_parquet_pool is not parquet_pool}")
    assert new_parquet_pool is not parquet_pool, "closing did not let a new pool start"
    new_parquet_pool.close_parquet_pool()


def main():
    folders_before = temp_folders()

    test_forgot_to_create_raises()
    parquet_pool = test_only_one_pool()
    test_threads_never_share(parquet_pool)
    test_empty_pool_leaves_nothing(folders_before)
    test_close_starts_fresh(parquet_pool)

    print("\nall good\n")


if __name__ == "__main__":
    main()
