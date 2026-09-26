"""tokenpool — tokenized training batches kept ready in RAM, shared by threads.

    TokenPool(parquet_pool, sequence_length=512)   # create first, from your parquet pool
    batch = TokenPool().get_token_batch()          # any thread; int32 (BATCH_SIZE, SEQUENCE_LENGTH)
    close_token_pool()                             # the only thing that ends it

    input_ids = torch.from_numpy(batch).long()     # ready for a causal LM as it is
    loss = model(input_ids=input_ids, labels=input_ids).loss

One pool per process: TokenPool(parquet_pool) creates it, every later
TokenPool() returns it, and TokenPool() before it was created raises
ValueError. This module never imports parquetpool. Threads tokenize parquet
files into rows of SEQUENCE_LENGTH tokens, batches of BATCH_SIZE rows; the
pool holds TOKEN_RAM_LIMIT // (BATCH_SIZE * SEQUENCE_LENGTH * 4) of them. Each
batch goes to exactly one caller. Settings are at the top of this file.
"""
import collections
import threading

import numpy as np
import pyarrow
import pyarrow.parquet as pq

__all__ = ["TokenPool", "close_token_pool", "MB", "GB"]

MB = 1024 ** 2
GB = 1024 ** 3
TOKEN_DTYPE = np.int32      # every token id in a batch; 4 bytes each

# ── the settings ────────────────────────────────────────────────────────────
# Edit these for good. Each one is also a keyword argument of TokenPool(),
# which changes it for that one pool and leaves the global alone.
TOKENIZER = "gpt2"              # any Hugging Face tokenizer name or local path
TEXT_COLUMN = "text"            # the parquet column holding the documents
SEQUENCE_LENGTH = 1024          # tokens in one row of a batch
BATCH_SIZE = 8                  # rows in one batch
TOKEN_RAM_LIMIT = 20 * GB       # most RAM the ready token batches may take;
                                # how many batches the pool holds is worked
                                # out from it
TOKENIZE_THREAD_COUNT = 4       # threads reading and tokenizing at once
DOCUMENTS_PER_CHUNK = 1000      # documents tokenized per call; bounds the
                                # memory one thread uses on a large file


def close_token_pool():
    """Close the token pool, the one command that ends it.

    Nothing else closes the pool: it stays alive while idle, however long
    nobody takes a batch. Does nothing when no token pool exists. Only the
    token pool is closed; the parquet pool it reads keeps running until
    parquetpool's close_parquet_pool() is called.

    In : nothing.
    Out: None, once every tokenize thread has ended.
    """
    with TokenPool._shared_pool_lock:
        token_pool = TokenPool.__dict__.get("_shared_pool")
    if token_pool is None or not token_pool._setup_done:
        return
    token_pool.close_token_pool()


class TokenPool:
    """The one pool of tokenized training batches, handed out one owner each.

    TOKENIZE_THREAD_COUNT threads each loop: take a parquet file with
    parquet_pool.get_parquet_file(), read TEXT_COLUMN in chunks of
    DOCUMENTS_PER_CHUNK documents, tokenize, and cut the tokens into batches.
    A thread keeps the tokens left over from one file and carries them into
    its next, so nothing is cut short at a file boundary; only the last few
    tokens that never fill a whole batch are dropped.

        parquet_pool = ParquetPool(dataset="nilq/babylm-100M")
        token_pool = TokenPool(parquet_pool, sequence_length=512)
        TokenPool() is token_pool                   # True: the same pool
        for batch in token_pool:                    # or get_token_batch() in threads
            train_step(batch)

    token_pool.max_batches_in_ram is not set by hand. Every batch takes
    exactly batch_size x sequence_length x 4 bytes, so it is worked out from
    token_ram_limit:

        max_batches_in_ram = token_ram_limit // (batch_size * sequence_length * 4)

    e.g. a 2 GB limit with 8 x 1024-token batches (31.2 KB each) holds 65,536
    batches. When a thread fills the pool it waits, so the ready batches stay
    within the limit. Not counted: what each thread is still working on (one
    parquet file, and the tokens not yet a full batch), and batches once
    get_token_batch() has handed them out. A file that cannot be read or
    tokenized is skipped and its name added to token_pool.untokenized_files.

    TokenPool(parquet_pool) creates the pool; every later TokenPool() returns
    it, and ignores any settings it was given. TokenPool() with no parquet
    pool before a pool exists raises ValueError: create it first. After
    close_token_pool(), the next TokenPool(parquet_pool) creates a new one.

    In : parquet_pool  the ParquetPool to read from, created by you first.
                       Anything whose get_parquet_file(timeout) gives
                       (file_name, file_bytes), None at the end, raises
                       TimeoutError when nothing came in time and RuntimeError
                       once closed, will do.
                       Needed only to create the pool; later calls leave it out.
         every setting at the top of the file, by keyword; anything not
         given keeps the global, e.g.
             tokenizer              str   e.g. "gpt2"
             text_column            str   e.g. "text"
             sequence_length        int   e.g. 1024
             batch_size             int   e.g. 8
             token_ram_limit        int   bytes, e.g. 2 * GB
             tokenize_thread_count  int   e.g. 4
             documents_per_chunk    int   e.g. 1000
    Out: the TokenPool, already filling. Raises ValueError when there is no
         token pool yet and no parquet_pool was given, for a setting that
         cannot work, such as sequence_length=0, when one batch alone is
         bigger than token_ram_limit, and whatever transformers raises for a
         tokenizer it cannot load.
    """

    _shared_pool = None                 # the one pool, once created
    _shared_pool_lock = threading.Lock()  # so two threads cannot both create it

    def __new__(cls, *args, **kwargs):
        with cls._shared_pool_lock:
            # Looked up on cls itself, so a subclass keeps a pool of its own
            # instead of inheriting TokenPool's.
            if cls.__dict__.get("_shared_pool") is None:
                new_pool = super().__new__(cls)
                new_pool._setup_done = False    # __init__ has not finished yet
                cls._shared_pool = new_pool
            return cls._shared_pool

    def __init__(self, parquet_pool=None, tokenizer=TOKENIZER,
                 text_column=TEXT_COLUMN, sequence_length=SEQUENCE_LENGTH,
                 batch_size=BATCH_SIZE, token_ram_limit=TOKEN_RAM_LIMIT,
                 tokenize_thread_count=TOKENIZE_THREAD_COUNT,
                 documents_per_chunk=DOCUMENTS_PER_CHUNK):
        # Python runs __init__ on every TokenPool(), even when __new__ gave
        # back the pool that already exists. Only the first one sets it up;
        # the lock makes the others wait until it is ready, and if setting it
        # up fails, the next TokenPool() tries again.
        with self._shared_pool_lock:
            if self._setup_done:
                return
            self._start_token_pool(parquet_pool, tokenizer, text_column,
                                   sequence_length, batch_size,
                                   token_ram_limit, tokenize_thread_count,
                                   documents_per_chunk)
            self._setup_done = True

    def _start_token_pool(self, parquet_pool, tokenizer, text_column,
                          sequence_length, batch_size, token_ram_limit,
                          tokenize_thread_count, documents_per_chunk):
        """Check the settings, size the pool, load the tokenizer, start threads."""
        if parquet_pool is None:
            raise ValueError("[tokenpool] no token pool yet: create the "
                             "parquet pool first, then "
                             "TokenPool(parquet_pool)")
        for setting_name, setting_value in (
                ("sequence_length", sequence_length),
                ("batch_size", batch_size),
                ("token_ram_limit", token_ram_limit),
                ("tokenize_thread_count", tokenize_thread_count),
                ("documents_per_chunk", documents_per_chunk)):
            if setting_value < 1:
                raise ValueError(f"{setting_name} must be >= 1")

        self.parquet_pool = parquet_pool
        self.tokenizer_name = tokenizer
        self.text_column = text_column
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.token_ram_limit = token_ram_limit
        self.max_batches_in_ram = self._max_batches_for_ram_limit()
        self.tokenize_thread_count = tokenize_thread_count
        self.documents_per_chunk = documents_per_chunk
        self.untokenized_files = []         # parquet names given up on

        # Loaded here once so a wrong name fails in the caller, not silently
        # in every thread. Each thread then loads its own copy: a tokenizer
        # shared between threads can raise "Already borrowed".
        self._load_tokenizer()

        # One lock guards the batches waiting and the count of live threads.
        self._token_cond = threading.Condition()
        self._ready_token_batches = collections.deque()
        self._running_tokenize_threads = tokenize_thread_count
        self._token_pool_closed = False

        self._tokenize_threads = []
        for thread_number in range(tokenize_thread_count):
            thread = threading.Thread(target=self._tokenize_worker, daemon=True,
                                      name=f"tokenpool-{thread_number}")
            thread.start()
            self._tokenize_threads.append(thread)

    def _max_batches_for_ram_limit(self):
        """How many token batches may wait in RAM within self.token_ram_limit.

        In : nothing; uses batch_size, sequence_length and token_ram_limit.
        Out: int   e.g. 65536 for a 2 GB limit and 8 x 1024-token batches,
                   which take 8 * 1024 * 4 = 32,768 bytes each.
             Raises ValueError when one batch alone exceeds the limit.
        """
        bytes_per_batch = (self.batch_size * self.sequence_length
                           * np.dtype(TOKEN_DTYPE).itemsize)
        if bytes_per_batch > self.token_ram_limit:
            raise ValueError(f"[tokenpool] one batch of {self.batch_size} x "
                             f"{self.sequence_length} tokens takes "
                             f"{bytes_per_batch:,} bytes, more than "
                             f"token_ram_limit={self.token_ram_limit:,} — "
                             f"raise the limit or make batches smaller")
        return self.token_ram_limit // bytes_per_batch

    def __repr__(self):
        return (f"TokenPool(tokenizer={self.tokenizer_name!r}, "
                f"sequence_length={self.sequence_length}, "
                f"batch_size={self.batch_size}, "
                f"max_batches_in_ram={self.max_batches_in_ram})")

    def __iter__(self):
        """Yield token batches until every file has been tokenized and handed out."""
        while True:
            batch = self.get_token_batch()
            if batch is None:
                return
            yield batch

    # ── taking batches out ──────────────────────────────────────────────────
    def get_token_batch(self, timeout=None):
        """Take one token batch out of the pool; the threads make another.

        Blocks until a batch is ready. Safe to call from many threads at once.

        In : timeout  float | None   e.g. 60.0 seconds; None waits for ever
        Out: np.ndarray int32, shape (batch_size, sequence_length), e.g.
               array([[  464,  3290,   318, ...],
                      [ 1110,    13, 50256, ...]], dtype=int32)
               50256 is gpt2's end-of-text token, between documents.
             None once every parquet file has been tokenized and handed out.
             Raises TimeoutError when nothing arrived in time, RuntimeError
             when the token pool is closed.
        """
        with self._token_cond:
            arrived = self._token_cond.wait_for(
                lambda: (self._ready_token_batches
                         or self._running_tokenize_threads == 0
                         or self._token_pool_closed),
                timeout)
            if self._token_pool_closed:
                raise RuntimeError("[tokenpool] the token pool is closed")
            if not arrived:
                raise TimeoutError(f"[tokenpool] no token batch ready within "
                                   f"{timeout}s")
            if not self._ready_token_batches:   # woken because nothing is left
                return None

            batch = self._ready_token_batches.popleft()
            self._token_cond.notify_all()       # a thread may wait for room
        return batch

    def close_token_pool(self):
        """Stop tokenizing, drop the batches held in RAM, and wait for the
        tokenize threads to end — for everyone.

        Nothing else ever closes the token pool: it stays alive while idle,
        however long nobody asks for a batch. The pool is shared, so this ends
        it for every thread using it: their waiting get_token_batch() calls
        raise RuntimeError. The parquet pool is left running; use
        parquetpool's close_parquet_pool() to end that one. The next
        TokenPool(parquet_pool) creates a new token pool.

        In : nothing.
        Out: None, once every tokenize thread has ended — within about a
             second, or as long as the tokenizer takes on one chunk.
        """
        with self._token_cond:
            self._token_pool_closed = True
            self._ready_token_batches.clear()
            self._token_cond.notify_all()

        # Closing twice is harmless: after the first close this pool is no
        # longer the shared one, so nothing here changes the second time.
        with self._shared_pool_lock:
            if type(self)._shared_pool is self:
                type(self)._shared_pool = None  # the next one starts afresh

        for thread in self._tokenize_threads:
            if thread is not threading.current_thread():
                thread.join()

    # ── filling the pool ────────────────────────────────────────────────────
    def _tokenize_worker(self):
        """One background thread: take a parquet file, tokenize it, repeat."""
        try:
            tokenizer = self._load_tokenizer()
            end_of_text_id = tokenizer.eos_token_id
            if end_of_text_id is None:
                document_separator = []
            else:
                document_separator = [end_of_text_id]

            # Tokens not yet a full row, and rows not yet a full batch. They
            # carry over from one file to the next.
            leftover_tokens = np.empty(0, dtype=TOKEN_DTYPE)
            leftover_rows = np.empty((0, self.sequence_length), dtype=TOKEN_DTYPE)

            while not self._token_pool_closed:
                try:
                    # A short wait, looped, rather than one endless one: an
                    # idle pool stays alive for as long as it takes, yet still
                    # notices close_token_pool() within a second.
                    parquet_file = self.parquet_pool.get_parquet_file(timeout=1.0)
                except TimeoutError:        # no file yet; look again
                    continue
                except RuntimeError:        # the parquet pool was closed
                    return
                if parquet_file is None:    # every file has been taken
                    return
                file_name, file_bytes = parquet_file

                try:
                    parquet_reader = pq.ParquetFile(pyarrow.BufferReader(file_bytes))
                    document_chunks = parquet_reader.iter_batches(
                        batch_size=self.documents_per_chunk,
                        columns=[self.text_column])
                    for document_chunk in document_chunks:
                        if self._token_pool_closed:
                            return          # no need to finish this file
                        documents = []
                        for document_text in document_chunk.column(0).to_pylist():
                            if document_text:   # None and "" add nothing
                                documents.append(document_text)
                        if not documents:
                            continue

                        token_ids_per_document = tokenizer(
                            documents, add_special_tokens=False)["input_ids"]
                        token_stream = []
                        for document_token_ids in token_ids_per_document:
                            token_stream.extend(document_token_ids)
                            token_stream.extend(document_separator)
                        leftover_tokens = np.concatenate(
                            [leftover_tokens,
                             np.asarray(token_stream, dtype=TOKEN_DTYPE)])

                        # Cut whole rows off the front of the tokens...
                        full_row_count = len(leftover_tokens) // self.sequence_length
                        if full_row_count:
                            tokens_in_full_rows = full_row_count * self.sequence_length
                            new_rows = leftover_tokens[:tokens_in_full_rows].reshape(
                                full_row_count, self.sequence_length)
                            leftover_rows = np.concatenate([leftover_rows, new_rows])
                            leftover_tokens = leftover_tokens[tokens_in_full_rows:].copy()

                        # ...and whole batches off the front of the rows.
                        while len(leftover_rows) >= self.batch_size:
                            batch = leftover_rows[:self.batch_size].copy()
                            leftover_rows = leftover_rows[self.batch_size:]
                            if not self._put_token_batch(batch):
                                return      # closed while waiting for room
                except Exception as error:
                    print(f"[tokenpool] skipping {file_name}: {error!r}")
                    self.untokenized_files.append(file_name)
        finally:
            with self._token_cond:
                self._running_tokenize_threads -= 1
                if self._running_tokenize_threads == 0:
                    # Callers may be waiting for a batch that will never come.
                    self._token_cond.notify_all()

    def _put_token_batch(self, batch):
        """Add a token batch, waiting while the pool is full.

        In : batch  np.ndarray   shape (batch_size, sequence_length)
        Out: bool   True when added, False when the pool closed meanwhile
        """
        with self._token_cond:
            self._token_cond.wait_for(
                lambda: (len(self._ready_token_batches) < self.max_batches_in_ram
                         or self._token_pool_closed))
            if self._token_pool_closed:
                return False
            self._ready_token_batches.append(batch)
            self._token_cond.notify_all()   # wake a waiting get_token_batch()
            return True

    def _load_tokenizer(self):
        """Load the tokenizer named in the settings.

        In : nothing.
        Out: a transformers tokenizer, e.g. GPT2TokenizerFast
        """
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_name)
