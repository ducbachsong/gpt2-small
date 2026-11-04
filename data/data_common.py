"""data_common — the .bin files the C and CUDA trainers read, as in llm.c.

    python data/data_common.py data/fineweb10B/fineweb_val_000000.bin
                                    # prints a token file's header and its first tokens as text

Two kinds of file. Both start with a 1 KB header of 256 int32 numbers, so C
can read the header with one fread and check it before trusting the rest:

    token file      header[0] = 20240520     magic number: "this is a token file"
                    header[1] = 1            version
                    header[2] = N            how many tokens follow
                    then N uint16            one per token: GPT-2's 50,257 ids fit in 16 bits

    tokenizer file  header[0] = 20240328     magic number: "this is a tokenizer file"
                    header[1] = 2            version
                    header[2] = 50257        vocabulary size
                    header[3] = 50256        the id of <|endoftext|>
                    then, for each id:       its length in bytes (uint8), then those bytes

The tokenizer file is what lets C turn ids back into text: GPT-2's tokenizer
works on bytes, so every id is simply a short run of bytes, and decoding is
printing them one after the other.
"""
import os
import sys

import numpy as np

__all__ = ["write_token_file", "read_token_file", "write_tokenizer_file",
           "gpt2_tokenizer", "token_bytes", "decode", "EOT_TOKEN"]

TOKEN_FILE_MAGIC = 20240520
TOKEN_FILE_VERSION = 1
TOKENIZER_FILE_MAGIC = 20240328
TOKENIZER_FILE_VERSION = 2
HEADER_INTS = 256                   # 256 x 4 bytes = the 1 KB header
EOT_TOKEN = 50256                   # <|endoftext|>: between documents
HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "gpt2_tokenizer.bin")


def write_token_file(path, tokens):
    """Writes token ids to a token file (format at the top of this file).

    In : path    str     where to write, e.g. ".../fineweb_train_000001.bin"
         tokens  array   token ids, each 0..65535
    Out: None
    """
    tokens = np.asarray(tokens)
    if tokens.size and (tokens.min() < 0 or tokens.max() > 65535):
        raise ValueError("token ids must fit in 16 bits (0..65535)")
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKEN_FILE_MAGIC
    header[1] = TOKEN_FILE_VERSION
    header[2] = len(tokens)
    # Written whole under another name, then renamed: a trainer reading the folder while this runs never
    # sees half a file (llmc/dataloader.cuh).
    partial = path + ".partial"
    with open(partial, "wb") as token_file:
        token_file.write(header.tobytes())
        token_file.write(tokens.astype(np.uint16).tobytes())
    os.replace(partial, path)
    print(f"wrote {len(tokens):,} tokens to {path}", flush=True)


def read_token_file(path):
    """The token ids in a token file, as uint16, after checking its header."""
    with open(path, "rb") as token_file:
        header = np.frombuffer(token_file.read(HEADER_INTS * 4), dtype=np.int32)
        if header[0] != TOKEN_FILE_MAGIC or header[1] != TOKEN_FILE_VERSION:
            raise ValueError(f"{path} is not a version-{TOKEN_FILE_VERSION} token file")
        tokens = np.frombuffer(token_file.read(), dtype=np.uint16)
    if len(tokens) != header[2]:
        raise ValueError(f"{path}: the header says {header[2]} tokens, the file has {len(tokens)}")
    return tokens


def gpt2_tokenizer():
    """GPT-2's tokenizer from the Hugging Face Hub, the same one main's TokenPool uses."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10 ** 12   # we cut the text ourselves: no warning about 1024
    return tokenizer


def _bytes_to_unicode():
    """GPT-2's map from each of the 256 byte values to a printable character.

    GPT-2's vocabulary is stored as text, so every byte, even an unprintable one,
    is written as a printable character: a space is "Ġ", a newline "Ċ". The
    printable bytes ("!" .. "~" and most of Latin-1) stand for themselves; the
    other 68 are moved to characters 256 and up, in order.
    """
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    characters = printable[:]
    moved = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + moved)
            moved += 1
    return dict(zip(printable, (chr(c) for c in characters)))


def token_bytes(tokenizer):
    """The bytes of every token id, in id order: [b"!", b"\\"", ..., b"<|endoftext|>"]."""
    byte_of_character = {character: byte for byte, character in _bytes_to_unicode().items()}
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return [bytes(byte_of_character[character] for character in piece) for piece in pieces]


def write_tokenizer_file(path=TOKENIZER_PATH, tokenizer=None):
    """Writes the tokenizer file (format at the top of this file), once; later calls keep it."""
    if os.path.exists(path):
        return
    tokenizer = tokenizer or gpt2_tokenizer()
    pieces = token_bytes(tokenizer)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKENIZER_FILE_MAGIC
    header[1] = TOKENIZER_FILE_VERSION
    header[2] = len(pieces)
    header[3] = tokenizer.eos_token_id
    with open(path, "wb") as tokenizer_file:
        tokenizer_file.write(header.tobytes())
        for piece in pieces:
            if not 0 < len(piece) < 256:
                raise ValueError(f"a token of {len(piece)} bytes does not fit the file format")
            tokenizer_file.write(bytes([len(piece)]) + piece)
    print(f"wrote {len(pieces):,} tokens to {path}")


def decode(tokens, path=TOKENIZER_PATH):
    """Token ids back to text, using only the tokenizer file (what C does)."""
    with open(path, "rb") as tokenizer_file:
        header = np.frombuffer(tokenizer_file.read(HEADER_INTS * 4), dtype=np.int32)
        if header[0] != TOKENIZER_FILE_MAGIC or header[1] != TOKENIZER_FILE_VERSION:
            raise ValueError(f"{path} is not a version-{TOKENIZER_FILE_VERSION} tokenizer file")
        pieces = []
        for _ in range(header[2]):
            length = tokenizer_file.read(1)[0]
            pieces.append(tokenizer_file.read(length))
    return b"".join(pieces[token] for token in tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    token_ids = read_token_file(sys.argv[1])
    print(f"{sys.argv[1]}: {len(token_ids):,} tokens, "
          f"{int((token_ids == EOT_TOKEN).sum()):,} of them <|endoftext|>")
    print("first 200 tokens as text:\n" + decode(token_ids[:200]))
