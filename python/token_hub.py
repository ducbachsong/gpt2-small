"""token_hub.py — keeps a folder of token files in a dataset on the Hugging Face Hub, so they are tokenized
once and every rented GPU machine only downloads them: minutes, where climbmix.py takes over an hour for 6B tokens.

    python python/climbmix.py --train-tokens 6000M --threads 16     # once, on any machine: no GPU needed
    python python/token_hub.py upload   --repo <user>/climbmix-tokens-32768
    python python/token_hub.py download --repo <user>/climbmix-tokens-32768   # on each GPU machine
    ./train_8bit --data-ready 1 --steps 11444                     # starts only with the whole run's tokens there

upload needs a Hugging Face token that can write: `hf auth login` once, or HF_TOKEN in the environment. A new
dataset is made private (--public: public); an existing one keeps its setting. download needs a token only for a
private dataset. ducbachsong/climbmix-tokens-32768 is public (CC BY-NC 4.0, as ClimbMix).

What goes up and comes down: the folder's *.bin files (climbmix_val_000000.bin, climbmix_train_000001.bin, ...,
climbmix_tokenizer.bin) and tokenizer.pkl. About 2 GB per 1B tokens.

upload    stops if the folder's writer is still working (its `writing` file is there): only whole runs go up.
          Files already there with the same bytes are skipped, so a stopped upload can be run again.
download  puts a `writing` file with this process's id in the folder while it runs, as climbmix.py does, so a
          trainer started at the same time waits for the files still to come (cuda/helper/dataloader.cuh). Each
          file appears whole, the Hub's library downloads several at once, and files already there are skipped.
          At the end it checks every token file's header against its size.
"""
import argparse
import os
import sys
import time

import numpy as np

from data_common import HEADER_INTS, TOKEN_FILE_MAGIC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tokenizer", "data")   # what the scripts make and read: python/tokenizer/data/ (tokenizer.DATA)
PATTERNS = ["*.bin", "tokenizer.pkl"]          # what a token folder holds; never `writing`
BYTES_PER_TOKEN = 2                            # uint16


def token_files_of(folder):
    """The folder's token files (their header's magic number says so; the tokenizer file's is another), by name."""
    paths = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not name.endswith(".bin") or os.path.getsize(path) < HEADER_INTS * 4:
            continue
        if int(np.fromfile(path, dtype=np.int32, count=1)[0]) == TOKEN_FILE_MAGIC:
            paths.append(path)
    return paths


def check_token_files(folder):
    """Checks each token file is as long as its header says, and gives (files, tokens); stops on one cut short:
    a file that would stop the trainer later."""
    total = 0
    paths = token_files_of(folder)
    for path in paths:
        tokens = int(np.fromfile(path, dtype=np.int32, count=3)[2])
        expected_bytes = HEADER_INTS * 4 + tokens * BYTES_PER_TOKEN
        if os.path.getsize(path) != expected_bytes:
            raise SystemExit(f"{path}: its header says {tokens:,} tokens ({expected_bytes:,} bytes), "
                             f"but it has {os.path.getsize(path):,} bytes")
        total += tokens
    return len(paths), total


def upload(repo, folder, public):
    """Sends the folder's token files and tokenizer to the dataset `repo`, making it (private unless `public`) if needed."""
    from huggingface_hub import HfApi
    if os.path.exists(os.path.join(folder, "writing")):
        raise SystemExit(f"{folder}/writing is there: the writer is still working (or was stopped: then delete it)")
    files, tokens = check_token_files(folder)
    if files == 0:
        raise SystemExit(f"no token files in {folder}")
    print(f"uploading {files} token file(s), {tokens:,} tokens ({tokens * BYTES_PER_TOKEN / 1e9:.1f} GB), to {repo}")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
    started = time.time()
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, allow_patterns=PATTERNS,
                      commit_message=f"{files} token file(s), {tokens:,} tokens")
    visibility = "private" if api.dataset_info(repo).private else "public"
    print(f"done in {time.time() - started:.0f}s: https://huggingface.co/datasets/{repo} ({visibility})")


def download(repo, folder, workers):
    """Brings the dataset's token files and tokenizer into the folder, `writing` there while it runs."""
    from huggingface_hub import snapshot_download
    os.makedirs(folder, exist_ok=True)
    writing_marker = os.path.join(folder, "writing")
    with open(writing_marker, "w") as marker_file:
        marker_file.write(str(os.getpid()))
    started = time.time()
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=folder, allow_patterns=PATTERNS,
                          max_workers=workers)
    finally:
        os.remove(writing_marker)
    files, tokens = check_token_files(folder)
    seconds = time.time() - started
    print(f"{files} token file(s), {tokens:,} tokens in {folder} after {seconds:.0f}s "
          f"({tokens * BYTES_PER_TOKEN / 1e6 / max(seconds, 1e-9):.0f} MB/s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--repo", required=True, help="the dataset on the Hub: <user>/<name>")
    parser.add_argument("--folder", default=os.path.join(DATA, "climbmix"), help="the token files' folder")
    parser.add_argument("--public", action="store_true", help="upload: make a new dataset public")
    parser.add_argument("--workers", type=int, default=8, help="download: files at once")
    args = parser.parse_args()
    if args.action == "upload":
        upload(args.repo, args.folder, args.public)
    else:
        download(args.repo, args.folder, args.workers)


if __name__ == "__main__":
    sys.exit(main())
