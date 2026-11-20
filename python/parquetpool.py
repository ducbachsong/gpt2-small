"""parquetpool — parquet files downloaded into RAM, shared safely between threads.

    ParquetPool(dataset="nilq/babylm-100M", ram_limit=20 * GB)   # create first
    file_name, file_bytes = ParquetPool().get_parquet_file()     # any thread; None at the end
    close_parquet_pool()                                         # the only thing that ends it

One pool per process: ParquetPool(dataset=...) creates it, every later
ParquetPool() returns it, and ParquetPool() before it was created raises
ValueError. It holds RAM_LIMIT // the largest matching file at once. Each
get_parquet_file() hands a file to exactly one caller and starts the next
download. Failed downloads are retried after DOWNLOAD_RETRY_DELAYS, then
skipped. Settings are at the top of this file.
"""
import collections
import os
import shutil
import sys
import tempfile
import threading

__all__ = ["ParquetPool", "close_parquet_pool", "format_bytes", "MB", "GB"]

MB = 1024 ** 2
GB = 1024 ** 3

# ── the settings ────────────────────────────────────────────────────────────
# Edit these for good. Each one is also a keyword argument of ParquetPool(),
# which changes it for that one pool and leaves the global alone. The dataset
# has no setting here: it is always given when creating the pool, which is how
# ParquetPool(dataset=...) creating it differs from ParquetPool() finding it.

# ── which files ─────────────────────────────────────────────────────────────
ALLOW_PATTERNS = None               # globs picking the files to download, e.g.
                                    # ["20231101.vi/*.parquet"], or a lone
                                    # "data/train-*.parquet". None means every
                                    # parquet file in the repo. Same name and
                                    # meaning as in snapshot_download.
REPO_TYPE = "dataset"
REVISION = ""                       # "" means the default branch; use
                                    # "refs/convert/parquet" for repos whose
                                    # main branch is not parquet
HF_TOKEN = ""                       # Hugging Face access token, e.g. "hf_xxx";
                                    # "" lets huggingface_hub find your login

# ── how to download ─────────────────────────────────────────────────────────
RAM_LIMIT = 20 * GB                 # most RAM the parquet files in the pool
                                    # may take; how many files the pool holds
                                    # is worked out from it and the largest
                                    # file listed
DOWNLOAD_RETRY_DELAYS = (3, 6, 12)  # seconds before each retry of a failed
                                    # download; after the last the file is
                                    # skipped
XET_HIGH_PERFORMANCE = True         # more parallel chunk fetches within one file
                                    # on Xet repos; only works if nothing has
                                    # imported huggingface_hub yet


def format_bytes(byte_count):
    """Render a byte count as a short human-readable string.

    In : byte_count  int|float   e.g. 429632933        a rate works too
    Out: str                     e.g. "409.7 MB"       also "888 B", "1.2 GB"
                                 Binary units: 1 KB = 1024 B.
    """
    units = ("B", "KB", "MB", "GB", "TB")

    for unit in units:
        last_unit = unit == units[-1]
        # Stop at the first unit the number fits in; TB is the end of the road,
        # so anything that big is printed in TB however large it is.
        if abs(byte_count) < 1024 or last_unit:
            if unit == "B":
                return f"{byte_count:,.0f} B"   # whole bytes: "888 B", not "888.0 B"
            return f"{byte_count:,.1f} {unit}"
        byte_count /= 1024.0


def close_parquet_pool():
    """Close the parquet pool, the one command that ends it.

    Nothing else closes the pool: it stays alive while idle, however long
    nobody takes a file. Does nothing when no parquet pool exists. Only the
    parquet pool is closed; a token pool reading from it notices, and ends
    once its ready batches are taken.

    In : nothing.
    Out: None. A download already under way finishes in the background and
         is thrown away.
    """
    with ParquetPool._shared_pool_lock:
        parquet_pool = ParquetPool.__dict__.get("_shared_pool")
    if parquet_pool is None or not parquet_pool._setup_done:
        return
    parquet_pool.close_parquet_pool()


class ParquetPool:
    """The one pool of parquet files held in RAM, handed out one owner each.

    The pool keeps up to `max_files_in_ram` files downloaded and waiting in
    memory. get_parquet_file() takes one out; its place is freed and a
    background thread downloads the next file to fill it. Any number of
    threads may call get_parquet_file() at once: each file goes to exactly one
    caller, never two, and no file is downloaded twice. When every file has
    been handed out, get_parquet_file() returns None.

    max_files_in_ram is not set by hand. The pool lists the matching files
    first and works it out from ram_limit and the largest file:

        max_files_in_ram = ram_limit // largest file, at most the file count

    so even a pool full of the largest file stays within the limit, e.g. a
    20 GB limit over files of up to 1.5 GB holds 13. The limit covers the
    files the pool holds, ready or downloading; a file stops counting once
    get_parquet_file() hands it out, and is then the caller's to free.

        parquet_pool = ParquetPool(dataset="HuggingFaceFW/fineweb-edu",
                                   allow_patterns="data/CC-MAIN-2013-20/*.parquet",
                                   ram_limit=20 * GB)
        parquet_pool.max_files_in_ram               # e.g. 13, worked out
        ParquetPool() is parquet_pool               # True: the same pool
        for file_name, file_bytes in parquet_pool:  # or get_parquet_file() in threads
            table = pyarrow.parquet.read_table(pyarrow.BufferReader(file_bytes))

    ParquetPool(dataset=...) creates the pool; every later ParquetPool()
    returns it, and ignores any settings it was given. ParquetPool() with no
    dataset before a pool exists raises ValueError: create it first. After
    close_parquet_pool(), the next ParquetPool(dataset=...) creates a new one.

    Files come from the Hub straight into memory: each one passes through a
    temporary folder only while it downloads, and the cache is not used.

    A download that fails is tried again after each of DOWNLOAD_RETRY_DELAYS
    — 3 s, 6 s, then 12 s; once those run out the file is skipped,
    get_parquet_file() moves on to the next one, and the skipped name is added
    to parquet_pool.skipped_files.

    In : dataset  str   e.g. "wikimedia/wikipedia", needed to create the pool;
                        leave it out to get the pool that already exists.
         every setting at the top of the file, by keyword; anything not
         given keeps the global, e.g.
             allow_patterns   list[str] | str | None
                                    e.g. ["20231101.vi/*.parquet"]
             repo_type        str   e.g. "dataset"
             revision         str   e.g. "refs/convert/parquet"; "" is default
             hf_token         str   e.g. "hf_xxx"; "" finds your own login
             ram_limit        int   bytes, e.g. 20 * GB
             xet_high_performance  bool  e.g. True
    Out: the ParquetPool, already filling. Calls sys.exit when no parquet file
         matches, or when the largest one alone is bigger than ram_limit.
         Raises ValueError when there is no pool yet and no dataset was
         given, or for a setting that cannot work, such as ram_limit=0.
    """

    _shared_pool = None                 # the one pool, once created
    _shared_pool_lock = threading.Lock()  # so two threads cannot both create it

    def __new__(cls, *args, **kwargs):
        with cls._shared_pool_lock:
            # Looked up on cls itself, so a subclass keeps a pool of its own
            # instead of inheriting ParquetPool's.
            if cls.__dict__.get("_shared_pool") is None:
                new_pool = super().__new__(cls)
                new_pool._setup_done = False    # __init__ has not finished yet
                cls._shared_pool = new_pool
            return cls._shared_pool

    def __init__(self, dataset=None, allow_patterns=ALLOW_PATTERNS,
                 repo_type=REPO_TYPE, revision=REVISION, hf_token=HF_TOKEN,
                 ram_limit=RAM_LIMIT,
                 xet_high_performance=XET_HIGH_PERFORMANCE):
        # Python runs __init__ on every ParquetPool(), even when __new__ gave
        # back the pool that already exists. Only the first one sets it up;
        # the lock makes the others wait until it is ready, and if setting it
        # up fails, the next ParquetPool() tries again.
        with self._shared_pool_lock:
            if self._setup_done:
                return
            self._start_parquet_pool(dataset, allow_patterns, repo_type,
                                     revision, hf_token, ram_limit,
                                     xet_high_performance)
            self._setup_done = True

    def _start_parquet_pool(self, dataset, allow_patterns, repo_type, revision,
                            hf_token, ram_limit, xet_high_performance):
        """Check the settings, list the files, size the pool, start threads."""
        if not dataset:
            raise ValueError("[parquetpool] no parquet pool yet: create it "
                             "first with ParquetPool(dataset=...)")
        if ram_limit < 1:
            raise ValueError("ram_limit must be >= 1 byte")

        self.dataset = dataset
        self.allow_patterns = allow_patterns
        self.repo_type = repo_type
        self.revision = revision
        self.hf_token = hf_token
        self.ram_limit = ram_limit
        self.xet_high_performance = xet_high_performance
        self.skipped_files = []             # names given up on, in that order

        # Before the listing, which is the first to import the hub.
        self._enable_xet_high_performance()

        file_sizes = self._list_parquet_files()
        if not file_sizes:
            sys.exit(f"[parquetpool] no parquet files matched "
                     f"{allow_patterns!r} in {dataset}")
        file_names = list(file_sizes)
        total_bytes = sum(file_sizes.values())
        largest_file_bytes = max(file_sizes.values())

        max_files_in_ram = self._max_files_for_ram_limit(len(file_names),
                                                         largest_file_bytes)
        self.max_files_in_ram = max_files_in_ram

        print(f"[parquetpool] {len(file_names)} file(s), "
              f"{format_bytes(total_bytes)} total, largest "
              f"{format_bytes(largest_file_bytes)}; {format_bytes(ram_limit)} "
              f"RAM limit -> {max_files_in_ram} in RAM at a time")

        # One lock guards all the shared state below. A name leaves
        # _files_to_download under it, so exactly one thread downloads that
        # file; its bytes leave _ready_files under it, so exactly one caller
        # gets them.
        self._parquet_cond = threading.Condition()
        self._files_to_download = collections.deque(file_names)  # not started yet
        self._ready_files = collections.deque()     # (name, bytes) waiting
        self._files_not_handed_out = len(file_names)  # the pool ends at 0
        self._parquet_pool_closed = False

        # One slot per file RAM may hold. A thread takes one before
        # downloading and get_parquet_file() gives it back, so ready +
        # downloading files never exceed max_files_in_ram.
        self._ram_slots = threading.Semaphore(max_files_in_ram)

        self._download_threads = []
        for thread_number in range(max_files_in_ram):
            thread = threading.Thread(target=self._download_worker, daemon=True,
                                      name=f"parquetpool-{thread_number}")
            thread.start()
            self._download_threads.append(thread)

    def _max_files_for_ram_limit(self, file_count, largest_file_bytes):
        """How many files the pool may hold at once within self.ram_limit.

        In : file_count          int   e.g. 14, matching files listed
             largest_file_bytes  int   e.g. 1610612736, the largest of them
        Out: int   e.g. 13 for a 20 GB limit over files of up to 1.5 GB;
                   never more than file_count, and never below 1.
             Calls sys.exit when the largest file alone exceeds the limit.
        """
        if largest_file_bytes == 0:
            # The Hub gave no sizes, so there is nothing to divide by; one
            # file at a time is the safest guess.
            print("[parquetpool] the Hub listed no file sizes; holding one "
                  "file at a time")
            return 1
        if largest_file_bytes > self.ram_limit:
            sys.exit(f"[parquetpool] the largest file is "
                     f"{format_bytes(largest_file_bytes)}, more than the "
                     f"{format_bytes(self.ram_limit)} RAM limit — raise "
                     f"ram_limit or narrow allow_patterns")
        return min(file_count, self.ram_limit // largest_file_bytes)

    def __repr__(self):
        return (f"ParquetPool(dataset={self.dataset!r}, "
                f"allow_patterns={self.allow_patterns!r}, "
                f"ram_limit={format_bytes(self.ram_limit)}, "
                f"max_files_in_ram={self.max_files_in_ram})")

    def __iter__(self):
        """Yield (file_name, file_bytes) until every file has been handed out."""
        while True:
            parquet_file = self.get_parquet_file()
            if parquet_file is None:
                return
            yield parquet_file

    # ── taking files out ────────────────────────────────────────────────────
    def get_parquet_file(self, timeout=None):
        """Take one parquet file out of the pool; a new one starts downloading.

        Blocks until a file is ready. Safe to call from many threads at once.

        In : timeout  float | None   e.g. 60.0 seconds; None waits for ever
        Out: (file_name, file_bytes)
               file_name   str    e.g. "data/train-00000-of-00002.parquet"
               file_bytes  bytes  the whole file, e.g. read it with
                     pyarrow.parquet.read_table(pyarrow.BufferReader(file_bytes))
                     or pandas.read_parquet(io.BytesIO(file_bytes))
             None once every file has been handed out or skipped.
             Raises TimeoutError when nothing arrived in time, RuntimeError
             when the pool is closed.
        """
        with self._parquet_cond:
            arrived = self._parquet_cond.wait_for(
                lambda: (self._ready_files
                         or self._files_not_handed_out == 0
                         or self._parquet_pool_closed),
                timeout)
            if self._parquet_pool_closed:
                raise RuntimeError("[parquetpool] the parquet pool is closed")
            if not arrived:
                raise TimeoutError(f"[parquetpool] no parquet file ready "
                                   f"within {timeout}s")
            if not self._ready_files:       # woken because nothing is left
                return None

            file_name, file_bytes = self._ready_files.popleft()
            self._count_file_finished()

        self._ram_slots.release()           # room in RAM: download the next
        return file_name, file_bytes

    def close_parquet_pool(self):
        """Stop downloading and drop what is held in RAM, for everyone.

        The pool is shared, so this ends it for every thread using it: their
        waiting get_parquet_file() calls raise RuntimeError. A download
        already under way finishes and is then thrown away. The next
        ParquetPool(dataset=...) creates a new pool.

        In : nothing.
        Out: None.
        """
        with self._parquet_cond:
            self._parquet_pool_closed = True
            self._ready_files.clear()
            self._parquet_cond.notify_all()

        # Closing twice is harmless: after the first close this pool is no
        # longer the shared one, and extra slots only wake threads that have
        # already left.
        with self._shared_pool_lock:
            if type(self)._shared_pool is self:
                type(self)._shared_pool = None  # the next one starts afresh

        # Wake every thread waiting for a slot, so it sees the pool is closed
        # and leaves.
        for _ in self._download_threads:
            self._ram_slots.release()

    # ── filling the pool ────────────────────────────────────────────────────
    def _download_worker(self):
        """One background thread: take a slot, download a file, repeat."""
        while True:
            self._ram_slots.acquire()       # wait for room in RAM

            with self._parquet_cond:
                if self._parquet_pool_closed or not self._files_to_download:
                    # Pass the slot on, so a thread still waiting on one wakes
                    # up and finds the same thing.
                    self._ram_slots.release()
                    return
                file_name = self._files_to_download.popleft()  # ours alone

            file_bytes = self._download_with_retries(file_name)  # None: gave up

            with self._parquet_cond:
                if self._parquet_pool_closed:
                    return                  # nobody will take it; drop it
                if file_bytes is None:
                    self.skipped_files.append(file_name)
                    self._count_file_finished()  # it will never be handed out
                else:
                    self._ready_files.append((file_name, file_bytes))
                    self._parquet_cond.notify()  # wake one waiting caller
                    continue

            self._ram_slots.release()       # skipped: the slot holds nothing

    def _count_file_finished(self):
        """Count one file as handed out or skipped. Call with _parquet_cond held."""
        self._files_not_handed_out -= 1
        if self._files_not_handed_out == 0:
            # Other callers may be waiting for a file that will never come.
            self._parquet_cond.notify_all()

    def _download_with_retries(self, file_name):
        """Download one file into RAM, retrying after each DOWNLOAD_RETRY_DELAYS.

        In : file_name  str   repo-relative, e.g. "data/train-00000-of-00002.parquet"
        Out: bytes      the whole file
             None       every try failed, or the pool closed while waiting to
                        retry
        """
        from huggingface_hub import hf_hub_download

        total_tries = len(DOWNLOAD_RETRY_DELAYS) + 1
        for try_number in range(total_tries):
            # A folder of its own per try, so parallel downloads never share a
            # path, and removing it leaves nothing behind on disk.
            temp_folder = tempfile.mkdtemp(prefix="parquetpool-")
            try:
                downloaded_path = hf_hub_download(
                    self.dataset,
                    file_name,
                    repo_type=self.repo_type,
                    revision=self.revision or None,  # "" means the default branch
                    local_dir=temp_folder,
                    token=self.hf_token or None,     # "" lets the Hub find your login
                )
                with open(downloaded_path, "rb") as downloaded_file:
                    file_bytes = downloaded_file.read()
                print(f"[parquetpool] fetched {os.path.basename(file_name)}, "
                      f"{format_bytes(len(file_bytes))}")
                return file_bytes
            except Exception as error:
                if try_number == total_tries - 1:
                    print(f"[parquetpool] skipping {file_name} after "
                          f"{total_tries} tries: {error!r}")
                    return None
                wait_seconds = DOWNLOAD_RETRY_DELAYS[try_number]
                print(f"[parquetpool] {file_name} failed ({error!r}), retry "
                      f"{try_number + 1}/{total_tries - 1} in {wait_seconds}s")
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)

            # Sleep, but wake at once if the pool is closed meanwhile.
            with self._parquet_cond:
                if self._parquet_cond.wait_for(
                        lambda: self._parquet_pool_closed, wait_seconds):
                    return None

    # ── talking to the Hub ──────────────────────────────────────────────────
    def _enable_xet_high_performance(self):
        """Ask huggingface_hub for parallel chunk fetches, if the setting says so.

        In : nothing.
        Out: None. Prints a note when it is too late for it to work.
        """
        # huggingface_hub reads HF_XET_HIGH_PERFORMANCE once, when it is
        # imported, so setting it afterwards changes nothing. That is why the
        # hub is imported inside the methods, and why this checks the
        # constant rather than assuming the write worked.
        if self.xet_high_performance:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
            import huggingface_hub.constants as hub_constants
            if not hub_constants.HF_XET_HIGH_PERFORMANCE:
                print("[parquetpool] xet_high_performance ignored: "
                      "huggingface_hub was already imported. Set "
                      "HF_XET_HIGH_PERFORMANCE=1 in the shell, or import "
                      "parquetpool before transformers.")

    def _list_parquet_files(self):
        """List the matching parquet files on the Hub, without downloading.

        In : nothing.
        Out: dict[str, int]   repo-relative name -> bytes, sorted by name, e.g.
               {"data/train-00000-of-00002.parquet": 152790334,
                "data/train-00001-of-00002.parquet": 204588835}
             0 for a file the Hub gave no size for.
        """
        from huggingface_hub import snapshot_download

        # The settings spell "not set" as "", the Hub spells it None and reads
        # it as "you decide" — default branch, find your own login.
        remote_files = snapshot_download(
            self.dataset,
            repo_type=self.repo_type,
            revision=self.revision or None,
            allow_patterns=self.allow_patterns,
            token=self.hf_token or None,
            dry_run=True,                   # metadata only: nothing is fetched
        )

        file_sizes = {}
        for remote_file in remote_files:
            if not remote_file.filename.endswith(".parquet"):
                continue
            # The Hub leaves the size None sometimes.
            file_sizes[remote_file.filename] = remote_file.file_size or 0

        return dict(sorted(file_sizes.items()))
