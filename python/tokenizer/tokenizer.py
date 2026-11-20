"""tokenizer.py — nanochat's tokenizer, the dictionary between text and token ids that every ClimbMix-trained model
here reads and writes (nanochat-d24, nanochat-d24-sft): its settings, training it (train_tokenizer), writing its
C-readable copy (write_tokenizer_bin) and loading it (load_tokenizer).

    from tokenizer.tokenizer import load_tokenizer
    encoding = load_tokenizer()
    encoding.encode_ordinary("Why is the sky blue?")      # [2771, 306, 262, 5974, 2856, 63]

The tokenizer: nanochat's (its scripts/tok_train.py), trained once by python/climbmix.py, then kept: rustbpe on the
first 2B characters of ClimbMix (each document cut at 10,000 characters), 32,768 ids = 32,759 merges + 9 special
tokens, GPT-4's split pattern with numbers of 1-2 digits. Its two files, in DATA/climbmix/:
    tokenizer.pkl            the tiktoken Encoding, pickled as nanochat saves it: the Python scripts load it
    climbmix_tokenizer.bin   each id's bytes in data_common's tokenizer-file format: for the C/CUDA side
Neither may ever change: every token file and every trained weight depends on these exact ids.

DATA (python/tokenizer/data/) is where every python/ script writes and reads what it makes: the tokenizer's files, the
token files (climbmix/, fineweb10B/), the chat rows (chat_sft/) and the fine-tune's starting weights (finetune_init/).
"""
import os
import pickle
import time

import numpy as np

from data_common import HEADER_INTS, TOKENIZER_FILE_MAGIC, TOKENIZER_FILE_VERSION

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOKENIZER_PICKLE = os.path.join(DATA, "climbmix", "tokenizer.pkl")
TOKENIZER_BIN = os.path.join(DATA, "climbmix", "climbmix_tokenizer.bin")
VOCAB_SIZE = 32768                         # nanochat's: 2^15
TOKENIZER_MAX_CHARS = 2_000_000_000        # nanochat's tok_train defaults
TOKENIZER_DOC_CAP = 10_000
# nanochat/tokenizer.py's, to the letter
SPECIAL_TOKENS = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                  "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def load_tokenizer(path=TOKENIZER_PICKLE):
    """Loads the pickled tiktoken Encoding; stops if it is not the trainers' 32,768 ids. A pickle runs code when
    loaded: only load files you trust.

    In : path  str  a tokenizer.pkl; default this folder's
    Out: the tiktoken.Encoding
    """
    with open(path, "rb") as pickle_file:
        encoding = pickle.load(pickle_file)
    if encoding.n_vocab != VOCAB_SIZE:
        raise SystemExit(f"{path} has {encoding.n_vocab:,} ids, the trainers' models {VOCAB_SIZE:,}")
    return encoding


def train_tokenizer(dataset, file_names, parquet_ram):
    """Trains nanochat's tokenizer on the first TOKENIZER_MAX_CHARS characters of the dataset's parquet files, read in
    name order as nanochat reads them, and pickles it to TOKENIZER_PICKLE.

    In : dataset      str    the Hub dataset, e.g. "karpathy/climbmix-400b-shuffle"
         file_names   list   its parquet files to read (2B characters take about 8 ClimbMix shards)
         parquet_ram  float  GB of parquet files the pool may hold
    Out: the tiktoken.Encoding
    """
    import pyarrow
    import pyarrow.parquet as pq
    import rustbpe
    import tiktoken
    from parquetpool import GB, ParquetPool, close_parquet_pool

    # 1. the files, into RAM by ParquetPool; they come in whatever order they finish, so kept and read by name
    parquet_pool = ParquetPool(dataset=dataset, allow_patterns=file_names, ram_limit=int(parquet_ram * GB))
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
        print(f"note: the {len(files)} files hold {characters:,} characters, fewer than {TOKENIZER_MAX_CHARS:,}: "
              f"more files for nanochat's vocabulary")

    # 3. rustbpe's merges, then the tiktoken encoding with the special tokens after them
    started = time.time()
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(texts(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    mergeable_ranks = {bytes(piece): rank for piece, rank in tokenizer.get_mergeable_ranks()}
    special_tokens = {name: len(mergeable_ranks) + offset for offset, name in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(name="rustbpe", pat_str=tokenizer.get_pattern(), mergeable_ranks=mergeable_ranks,
                                 special_tokens=special_tokens)
    os.makedirs(os.path.dirname(TOKENIZER_PICKLE), exist_ok=True)
    with open(TOKENIZER_PICKLE, "wb") as pickle_file:
        pickle.dump(encoding, pickle_file)
    print(f"trained the tokenizer in {time.time() - started:.0f}s: {encoding.n_vocab:,} ids, saved to {TOKENIZER_PICKLE}")
    return encoding


def write_tokenizer_bin(encoding):
    """Writes the tokenizer's C-readable copy, once: data_common's tokenizer file, each id's bytes (a special token's:
    its name), and <|bos|> where GPT-2's file has <|endoftext|>.

    In : encoding  the tiktoken.Encoding
    """
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
