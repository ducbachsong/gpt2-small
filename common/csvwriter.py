"""csvwriter — append rows to a CSV file, one line of code per row.

    from csvwriter import log
    log(step=1, loss=2.31, lr=3e-4)            # nothing to set up: goes to LOG_PATH,
                                               # e.g. logs/log_2026-09-23_16-35-02.csv

    writer = CsvWriter("runs/loss.csv")       # opens, or creates, the file
    writer.write(step=1, loss=2.31, lr=3e-4)   # one row; the header is made for you
    writer.write({"step": 2, "loss": 2.10})    # a dict works too
    writer.close()                             # or `with CsvWriter(...) as writer:`

    write_csv("runs/loss.csv", step=3, loss=1.9)   # no object to keep: one call, one row

Nothing here knows about training: a row is any set of named values. The
header comes from the first row. A column missing from a row is left blank;
a column never seen before is added to the header, and the rows already
written get a blank in it (the file is rewritten once, so keep that rare).
Tensors and numpy scalars are written as plain numbers. "{time}" in any
path becomes the time this run started, e.g. "2026-09-23_16-35-02": the same
for every file of one run, new for the next run.

Writing is fast: write() only puts the row in a list in RAM. Every
FLUSH_EVERY_SECONDS, on flush(), on close() and when Python exits, the list
is turned into text and written to the file. Tensors are detached when
written but read (.item()) only then, so logging a GPU loss every step makes
the CPU wait for the GPU once per flush, not once per step. Any number of
threads may write to the same writer at once. The output is what
csvexplorer reads. Settings are at the top of this file.
"""
import atexit
import csv
import datetime
import os
import threading
import time

__all__ = ["CsvWriter", "log", "write_csv", "close_all_csv", "full_path"]

# ── the settings ────────────────────────────────────────────────────────────
# Edit these for good. Each one but LOG_PATH is also a keyword argument of
# CsvWriter(), which changes it for that one writer and leaves the global alone.

LOG_PATH = "logs/log_{time}.csv"
                            # where log() writes; relative to the folder you
                            # run Python from. {time} is filled with the run's
                            # start time. Change it for good here, or for one
                            # run with csvwriter.LOG_PATH = "runs/exp1_{time}.csv"
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
                            # how {time} is written; no ":" so Windows accepts it
MODE = "append"             # "append" keeps what the file has and adds to it;
                            # "overwrite" starts the file empty
FLUSH_EVERY_SECONDS = 1.0   # most time a written row may wait in RAM before
                            # it reaches the disk; each flush reads the waiting
                            # GPU tensors in one go. 0 flushes every row
TIME_COLUMN = ""            # name of a column filled with the wall-clock time
                            # of each row, e.g. "time"; "" adds no such column
FLOAT_FORMAT = ""           # how floats are written, e.g. ".6g"; "" writes
                            # them in full, e.g. 0.30000000000000004
DELIMITER = ","
ENCODING = "utf-8"


class CsvWriter:
    """One CSV file, written a row at a time.

        with CsvWriter("results/sensor.csv", time_column="time") as log:
            for reading in readings:
                log.write(sensor=reading.name, celsius=reading.value)

    In appending mode an existing file keeps its header and rows, and new
    rows follow them; a file that does not exist yet is created, along with
    the folders above it.
    """

    def __init__(self, path, mode=None, flush_every_seconds=None,
                 time_column=None, float_format=None, delimiter=None,
                 encoding=None):
        """Open `path` for writing.

        In : path                 str     e.g. "runs/exp1/loss.csv"
             mode                 str     "append" | "overwrite"; None means MODE
             the rest             None means the setting of the same name
        Out: CsvWriter, open. Raises ValueError for an unknown mode.
        """
        self.path = full_path(path)
        self.mode = MODE if mode is None else mode
        self.flush_every_seconds = (FLUSH_EVERY_SECONDS if flush_every_seconds is None
                                    else flush_every_seconds)
        self.time_column = TIME_COLUMN if time_column is None else time_column
        self.float_format = FLOAT_FORMAT if float_format is None else float_format
        self.delimiter = DELIMITER if delimiter is None else delimiter
        self.encoding = ENCODING if encoding is None else encoding
        if self.mode not in ("append", "overwrite"):
            raise ValueError(f'mode must be "append" or "overwrite", not {self.mode!r}')

        self.columns = []           # the header, in order
        self.rows_written = 0       # rows this writer wrote, not the file's total
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._pending = []          # rows written but not yet in the file
        self._last_flush = time.monotonic()

        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if self.mode == "append":
            self.columns = self._header_on_disk()
        self._open("a" if self.mode == "append" else "w")
        _open_writers.add(self)

    # ── writing ─────────────────────────────────────────────────────────────
    def write(self, row=None, **values):
        """Write one row.

        In : row     dict | None   e.g. {"step": 1, "loss": 2.31}
             values  named values, e.g. step=1, loss=2.31; merged over `row`
        Out: None. A column not in the header yet is added to it.
        """
        merged = dict(row) if row else {}
        merged.update(values)
        self.write_many([merged])

    def write_many(self, rows):
        """Write several rows at once, cheaper than calling write() for each.

        In : rows  iterable of dict   e.g. [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
        Out: None.
        """
        stamp = datetime.datetime.now().isoformat(timespec="milliseconds")
        prepared = []
        for row in rows:
            cells = {str(name): _detached(value) for name, value in row.items()}
            if self.time_column and self.time_column not in cells:
                cells[self.time_column] = stamp
            prepared.append(cells)
        if not prepared:
            return

        with self._lock:
            if self._file is None:
                raise ValueError(f"{self.path} is closed")
            self._pending.extend(prepared)
            self.rows_written += len(prepared)
            if time.monotonic() - self._last_flush >= self.flush_every_seconds:
                self._flush_locked()

    def flush(self):
        """Push every buffered row to the disk now.

        In : nothing.  Out: None.
        """
        with self._lock:
            if self._file is not None:
                self._flush_locked()

    def close(self):
        """Flush and close the file. Closing twice does nothing.

        In : nothing.  Out: None.
        """
        with self._lock:
            if self._file is not None:
                self._flush_locked()
                self._file.close()
                self._file = None
        _open_writers.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __repr__(self):
        state = "closed" if self._file is None else "open"
        return f"CsvWriter({self.path!r}, {state}, {len(self.columns)} columns)"

    # ── inside ──────────────────────────────────────────────────────────────
    def _cell(self, value):
        """One value as the text or number csv should write."""
        if value is None:
            return ""
        # Tensors and numpy scalars: a one-element one becomes its plain number;
        # a bigger one refuses .item() and is written as its text.
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except (ValueError, RuntimeError, TypeError):
                pass
        if isinstance(value, float) and self.float_format:
            return format(value, self.float_format)
        return value

    def _header_on_disk(self):
        """The header of the file as it is now; [] when it is missing or empty."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="", encoding=self.encoding) as existing:
            return next(csv.reader(existing, delimiter=self.delimiter), [])

    def _open(self, file_mode):
        """Open the file and write the header if the file is empty."""
        self._file = open(self.path, file_mode, newline="", encoding=self.encoding)
        self._writer = csv.writer(self._file, delimiter=self.delimiter)
        if self._file.tell() == 0 and self.columns:
            self._writer.writerow(self.columns)

    def _add_columns(self, new_columns):
        """Widen the header. The file is rewritten when it already has rows."""
        self._file.flush()
        empty = self._file.tell() == 0
        self.columns = self.columns + new_columns
        if empty:
            self._writer.writerow(self.columns)
            return

        # Rows already on disk get a blank in each new column: read them all,
        # write them again under the wider header, and swap the files at once.
        self._file.close()
        with open(self.path, newline="", encoding=self.encoding) as old:
            old_rows = list(csv.reader(old, delimiter=self.delimiter))[1:]
        padding = [""] * len(new_columns)
        temp_path = self.path + ".tmp"
        with open(temp_path, "w", newline="", encoding=self.encoding) as new:
            rewriter = csv.writer(new, delimiter=self.delimiter)
            rewriter.writerow(self.columns)
            width = len(self.columns) - len(new_columns)
            for old_row in old_rows:
                rewriter.writerow((old_row + [""] * width)[:width] + padding)
        os.replace(temp_path, self.path)
        self._open("a")

    def _flush_locked(self):
        """Turn the waiting rows into text, write them, and push them to disk."""
        if self._pending:
            pending, self._pending = self._pending, []
            # The first .item() on a GPU tensor waits for the GPU; the rest are
            # then already computed, so a whole second of rows costs one wait.
            rows = [{name: self._cell(value) for name, value in cells.items()}
                    for cells in pending]
            new_columns = []
            for cells in rows:
                for name in cells:
                    if name not in self.columns and name not in new_columns:
                        new_columns.append(name)
            if new_columns:
                self._add_columns(new_columns)
            self._writer.writerows([cells.get(name, "") for name in self.columns]
                                   for cells in rows)
        self._file.flush()
        self._last_flush = time.monotonic()


def _detached(value):
    """A tensor cut from its autograd graph, so keeping it does not keep the
    graph's memory alive; anything else as it is. Does not wait for the GPU."""
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


def full_path(path):
    """`path` made absolute, with "{time}" filled with the run's start time.

    In : path  str   e.g. "logs/log_{time}.csv"
    Out: str         e.g. "C:/lab/logs/log_2026-09-23_16-35-02.csv"
    """
    path = path.replace("{time}", RUN_STARTED.strftime(TIME_FORMAT))
    return os.path.abspath(os.path.expanduser(path))


# ── one call, one row ───────────────────────────────────────────────────────
RUN_STARTED = datetime.datetime.now()   # when this run imported csvwriter
_open_writers = set()               # every open CsvWriter, closed at exit
_writers_by_path = {}               # the writers write_csv() keeps open
_writers_by_path_lock = threading.Lock()


def write_csv(path, row=None, **values):
    """Write one row to `path` without keeping a writer yourself.

    The first call for a path opens a CsvWriter with the settings, and later
    calls reuse it, so this is as fast as holding the writer.

    In : path    str          e.g. "runs/loss.csv"
         row     dict | None  e.g. {"step": 1}
         values  named values, e.g. loss=2.31
    Out: CsvWriter, the one used, e.g. to flush() it.
    """
    key = full_path(path)
    with _writers_by_path_lock:
        writer = _writers_by_path.get(key)
        if writer is None or writer._file is None:
            writer = _writers_by_path[key] = CsvWriter(key)
    writer.write(row, **values)
    return writer


def log(row=None, **values):
    """Write one row to LOG_PATH: import it and call it, nothing to set up.

        from csvwriter import log
        log(step=step, loss=loss, lr=scheduler.get_last_lr()[0])

    In : row     dict | None  e.g. {"step": 1}
         values  named values, e.g. loss=2.31
    Out: CsvWriter, the one used. LOG_PATH is read on every call, so setting
         csvwriter.LOG_PATH sends the next rows to the new file.
    """
    return write_csv(LOG_PATH, row, **values)


def close_all_csv():
    """Flush and close every open CsvWriter. Runs by itself when Python exits.

    In : nothing.  Out: None.
    """
    for writer in list(_open_writers):
        writer.close()
    with _writers_by_path_lock:
        _writers_by_path.clear()


atexit.register(close_all_csv)
