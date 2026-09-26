"""test — write CSVs with csvwriter into a temporary folder, and print what happened.

    python test/csvwriter-test.py

Everything is local and nothing is left behind. The rows are shop sales,
not training logs: the module must not care what the numbers mean.

Each case is one function, run in order by main(). Every case prints what it
found, so a failure shows you where it stopped instead of just saying False.
"""
import csv
import importlib
import os
import shutil
import sys
import tempfile
import threading
import time

# The module under test lives one folder up, next to this test folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csvwriter import CsvWriter, close_all_csv, write_csv

THREAD_COUNT = 8
ROWS_PER_THREAD = 2000


def read_back(path):
    """The file as (header, rows)."""
    with open(path, newline="", encoding="utf-8") as written:
        lines = list(csv.reader(written))
    return lines[0], lines[1:]


def check(label, ok, found):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {found}")
    return ok


def case_header_and_rows(folder):
    print("header from the first row, blanks for missing columns")
    path = os.path.join(folder, "deep", "sales.csv")
    with CsvWriter(path) as log:
        log.write(day=1, shop="north", sold=12)
        log.write({"day": 2, "shop": "south"})
    header, rows = read_back(path)
    return all([check("header", header == ["day", "shop", "sold"], header),
                check("rows", rows == [["1", "north", "12"], ["2", "south", ""]], rows)])


def case_new_column(folder):
    print("a new column widens the header and pads old rows")
    path = os.path.join(folder, "grow.csv")
    with CsvWriter(path) as log:
        log.write(day=1, sold=5)
        log.write(day=2, sold=6, returned=1)
        log.write(day=3, sold=7)
    header, rows = read_back(path)
    return all([check("header", header == ["day", "sold", "returned"], header),
                check("rows", rows == [["1", "5", ""], ["2", "6", "1"], ["3", "7", ""]], rows)])


def case_append_and_overwrite(folder):
    print("append keeps the file, overwrite empties it")
    path = os.path.join(folder, "again.csv")
    with CsvWriter(path) as log:
        log.write(day=1, sold=1)
    with CsvWriter(path) as log:                   # appends by default
        log.write(sold=2, day=2)                   # order follows the file's header
    _, appended = read_back(path)
    with CsvWriter(path, mode="overwrite") as log:
        log.write(day=9, sold=9)
    _, overwritten = read_back(path)
    return all([check("appended", appended == [["1", "1"], ["2", "2"]], appended),
                check("overwritten", overwritten == [["9", "9"]], overwritten)])


def case_values(folder):
    print("None, float format, numpy scalars, time column")
    path = os.path.join(folder, "values.csv")
    with CsvWriter(path, float_format=".3g", time_column="time") as log:
        values = {"price": 1 / 3, "note": None}
        try:
            import numpy
            values["count"] = numpy.int64(4)
        except ImportError:
            values["count"] = 4
        log.write(values)
    header, rows = read_back(path)
    row = dict(zip(header, rows[0]))
    return all([check("price", row["price"] == "0.333", row["price"]),
                check("note", row["note"] == "", row["note"]),
                check("count", row["count"] == "4", row["count"]),
                check("time", len(row["time"]) == 23, row["time"])])


def case_threads(folder):
    print(f"{THREAD_COUNT} threads x {ROWS_PER_THREAD} rows into one writer")
    path = os.path.join(folder, "threads.csv")
    log = CsvWriter(path)

    def work(thread_number):
        for i in range(ROWS_PER_THREAD):
            log.write(thread=thread_number, i=i)

    started = time.perf_counter()
    threads = [threading.Thread(target=work, args=(n,)) for n in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.close()
    seconds = time.perf_counter() - started
    _, rows = read_back(path)
    total = THREAD_COUNT * ROWS_PER_THREAD
    print(f"       {total:,} rows in {seconds:.2f} s ({total / seconds:,.0f} rows/s)")
    return all([check("row count", len(rows) == total, len(rows)),
                check("unique rows", len(set(map(tuple, rows))) == total, len(set(map(tuple, rows))))])


def case_write_csv(folder):
    print("write_csv() reuses one writer per path")
    path = os.path.join(folder, "quick.csv")
    first = write_csv(path, day=1, sold=3)
    second = write_csv(path, day=2, sold=4)
    close_all_csv()
    _, rows = read_back(path)
    return all([check("same writer", first is second, first is second),
                check("rows", rows == [["1", "3"], ["2", "4"]], rows)])


class FakeTensor:
    """Stands in for a GPU tensor: counts detach() and item(), like torch has them."""
    item_calls = 0

    def __init__(self, number, detached=False):
        self.number = number
        self.detached = detached

    def detach(self):
        return FakeTensor(self.number, detached=True)

    def size(self):                 # a method, as in torch, not a number
        return ()

    def item(self):
        FakeTensor.item_calls += 1
        assert self.detached, "item() on a tensor still holding its graph"
        return self.number


def case_rows_wait_in_ram(folder):
    print("rows wait in RAM; tensors are read only at flush")
    path = os.path.join(folder, "ram.csv")
    FakeTensor.item_calls = 0
    log = CsvWriter(path, flush_every_seconds=3600)
    for step in range(100):
        log.write(step=step, loss=FakeTensor(step / 10))
    items_before_flush = FakeTensor.item_calls
    size_before_flush = os.path.getsize(path)
    log.flush()
    items_after_flush = FakeTensor.item_calls
    log.close()
    header, rows = read_back(path)
    return all([check("item() before flush", items_before_flush == 0, items_before_flush),
                check("file before flush", size_before_flush == 0, size_before_flush),
                check("item() at flush", items_after_flush == 100, items_after_flush),
                check("header", header == ["step", "loss"], header),
                check("last row", rows[-1] == ["99", "9.9"], rows[-1])])


def case_log(folder):
    print("log() writes to LOG_PATH with nothing set up")
    csvwriter = importlib.import_module("csvwriter")
    path = os.path.join(folder, "logged.csv")
    csvwriter.LOG_PATH = path
    from csvwriter import log
    log(step=0, loss=FakeTensor(2.5))
    log(step=1, loss=FakeTensor(1.5))
    close_all_csv()
    header, rows = read_back(path)
    return all([check("header", header == ["step", "loss"], header),
                check("rows", rows == [["0", "2.5"], ["1", "1.5"]], rows)])


def case_time_in_name(folder):
    print("{time} in a path becomes the run's start time, the same every call")
    csvwriter = importlib.import_module("csvwriter")
    csvwriter.LOG_PATH = os.path.join(folder, "log_{time}.csv")
    first = csvwriter.log(step=0)
    time.sleep(1.1)                 # a later call must still land in the same file
    second = csvwriter.log(step=1)
    close_all_csv()
    name = os.path.basename(first.path)
    stamp = csvwriter.RUN_STARTED.strftime(csvwriter.TIME_FORMAT)
    _, rows = read_back(first.path)
    return all([check("name", name == f"log_{stamp}.csv", name),
                check("same file", first.path == second.path, second.path == first.path),
                check("rows", rows == [["0"], ["1"]], rows)])


def main():
    folder = tempfile.mkdtemp(prefix="csvwriter-test-")
    try:
        results = [case(folder) for case in (case_header_and_rows, case_new_column,
                                             case_append_and_overwrite, case_values,
                                             case_threads, case_write_csv,
                                             case_rows_wait_in_ram, case_log,
                                             case_time_in_name)]
    finally:
        close_all_csv()
        shutil.rmtree(folder, ignore_errors=True)
    print("all good" if all(results) else "SOMETHING FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
