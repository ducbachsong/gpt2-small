"""test — run hgdatadownloader against the real Hub, once, and print what happened.

    python test/hgdatadownloader-test.py

It downloads one small parquet file (~37 MB, babylm's test split), so it needs
a working network connection and it really does write to the cache. Nothing is
mocked: if this prints "all good", the module works against the live Hub.

Every step prints what it found, so a failure shows you where it stopped
instead of just saying False.
"""
import os
import sys

# The module under test lives one folder up, next to this test folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.old.hgdatadownloader import DataDownloader, find_local_parquet_files, format_bytes

DATASET = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = "data/CC-MAIN-2013-20/train-*.parquet" 
REVISION = "main"        # the smallest file in that repo


def main():
    # ── 1. look before downloading: metadata only, nothing written ──────────
    print("\n=== 1. what does the repo hold? ===")
    dd = DataDownloader(dataset=DATASET,allow_patterns=ALLOW_PATTERNS,revision=REVISION)
    files, total = dd.list_remote_files()
    for entry in files:
        print(f"    {format_bytes(entry['size']):>12}  "
              f"{'xet' if entry['xet'] else 'lfs'}  {entry['name']}")
    print(f"    {len(files)} file(s), {format_bytes(total)} in total")
    assert files, "the repo listed no files at all"

    # ── 2. download one file, twice: the second call must cost nothing ──────
    print(f"\n=== 2. download {ALLOW_PATTERNS} ===")
    dd = DataDownloader(dataset=DATASET, allow_patterns=ALLOW_PATTERNS)
    paths = dd.download_parquet_files()
    for path in paths:
        print(f"    {format_bytes(os.path.getsize(path)):>12}  {path}")
    assert paths, "nothing came back"
    for path in paths:
        assert os.path.exists(path), f"{path} is not on disk"
        assert path.endswith(".parquet"), f"{path} is not a parquet file"

    print("    asking again (should not download, should be the same list)")
    assert dd.download_parquet_files() is paths, "the second call re-downloaded"

    # ── 3. stream the same file, deleting as we go ──────────────────────────
    print(f"\n=== 3. stream {ALLOW_PATTERNS}, deleting after use ===")
    dd = DataDownloader(dataset=DATASET, allow_patterns=ALLOW_PATTERNS)
    streamed = []
    for path in dd.stream_parquet_files():
        size = os.path.getsize(path)            # readable while it is yielded
        print(f"    got {os.path.basename(path)}, {format_bytes(size)}")
        streamed.append(path)
    assert streamed, "streamed nothing"
    for path in streamed:
        assert not os.path.exists(path), f"{path} should have been deleted"
    print("    all streamed files are gone from disk, as asked")

    # ── 4. the local finder, on what step 2 left in the cache ───────────────
    print("\n=== 4. find the parquet files left in the snapshot ===")
    snapshot_dir = os.path.dirname(os.path.dirname(paths[0]))
    found = find_local_parquet_files(snapshot_dir)
    for path in found:
        print(f"    {os.path.relpath(path, snapshot_dir)}")
    print(f"    {len(found)} parquet file(s) under {snapshot_dir}")

    print("\nall good\n")


if __name__ == "__main__":
    main()
