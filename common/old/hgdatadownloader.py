"""hgdatadownloader — get dataset files off the Hugging Face Hub, once.

Downloads dataset files and hands back local paths. That is the whole job: it
does not read, parse or tokenize anything.

    import sys; sys.path.insert(0, "/path/to/huggingface-datadownloader")
    from hgdatadownloader import DataDownloader

    dd    = DataDownloader()                # the settings at the top of the file
    files = dd.download_parquet_files()     # local .parquet paths, downloaded once

The job stops at the files. Reading them is yours to do — point pyarrow,
pandas, polars or datasets at the paths that come back.

What to download is decided by the settings at the top of this file. Edit
them there, or pass any of them as a keyword for one downloader only — each
setting is a keyword argument of DataDownloader(), defaulting to the global:

    dd = DataDownloader(dataset="wikimedia/wikipedia",
                        allow_patterns=["20231101.vi/*.parquet"])
    dd = DataDownloader(max_workers=16)

Either way they end up as attributes: dd.dataset, dd.max_workers. ALLOW_PATTERNS
says which files to take — the same argument, under the same name, that
huggingface_hub's snapshot_download takes, and None takes the whole repo. Call
list_remote_files() first to see what a repo holds, then write the globs.

For a set too big to hold at once, stream_parquet_files() fetches one file at
a time and, unless told otherwise, keeps only one on disk.

One class, and three functions around it: free_bytes, format_bytes,
find_local_parquet_files. Everything real is huggingface_hub's snapshot_download —
what is here is the settings, the guards and the plumbing around one library
call.

Nothing here reads or writes os.environ except the xet_high_performance knob,
which huggingface_hub insists on taking from the environment. Where downloads
land is CACHE_DIR, not $HF_HOME, and the credential is TOKEN.
"""
import fnmatch
import glob
import os
import shutil
import sys
import time

__all__ = ["DataDownloader", "free_bytes", "format_bytes",
           "find_local_parquet_files"]

# ── the settings ────────────────────────────────────────────────────────────
# Edit these for good. Each one is also a keyword argument of DataDownloader(),
# which changes it for that one downloader and leaves the global alone.

# ── which dataset ───────────────────────────────────────────────────────────
DATASET = "nilq/babylm-100M"        # any HF dataset repo with parquet files
ALLOW_PATTERNS = None               # globs picking the files to download, e.g.
                                    # ["20231101.vi/*.parquet"], or a lone
                                    # "data/train-*.parquet". None means the
                                    # whole repo. Handed to snapshot_download
                                    # exactly as written — same name, same
                                    # meaning.
REPO_TYPE = "dataset"
REVISION = ""                       # "" means the default branch; use
                                    # "refs/convert/parquet" for repos whose
                                    # main branch is not parquet
TOKEN = ""                          # "" lets huggingface_hub find its own

# ── how to download ─────────────────────────────────────────────────────────
MAX_WORKERS = 16                     # how many files are fetched at once
HEADROOM = 1.2                      # require 20% more free disk than the payload
LOCAL_DIR = ""                      # set to keep a flat copy instead of the cache
CACHE_DIR = "~/.cache/huggingface/hub"      # where snapshot_download puts files
FORCE = False                       # download even if the disk guard objects
XET_HIGH_PERFORMANCE = True        # more parallel chunk fetches within one file
                                    # on Xet repos; only works if nothing has
                                    # imported huggingface_hub yet


def free_bytes(path):
    """Free space on the volume that holds `path`.

    In : path  str     e.g. "C:/Users/me/.cache/huggingface/hub"
                       Need not exist; the nearest existing parent is measured.
    Out: int           e.g. 215525027840
    """
    # A cache directory usually does not exist before the first download, and
    # disk_usage needs a real one. Walking up lands on the volume the files
    # will be written to, which is what the caller is really asking about.
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:          # at the root, and even that is not a dir
            break
        path = parent

    if not path:
        path = "."                  # nothing usable left; measure this volume
    return shutil.disk_usage(path).free


def format_bytes(n):
    """Render a byte count as a short human-readable string.

    In : n    int|float   e.g. 429632933        a rate works too
    Out: str              e.g. "409.7 MB"       also "888 B", "1.2 GB"
                          Binary units: 1 KB = 1024 B.
    """
    units = ("B", "KB", "MB", "GB", "TB")

    for unit in units:
        last_unit = unit == units[-1]
        # Stop at the first unit the number fits in; TB is the end of the road,
        # so anything that big is printed in TB however large it is.
        if abs(n) < 1024 or last_unit:
            if unit == "B":
                return f"{n:,.0f} B"        # whole bytes: "888 B", not "888.0 B"
            return f"{n:,.1f} {unit}"
        n /= 1024.0


def find_local_parquet_files(path, allow_patterns=None):
    """Find the parquet files inside a directory tree.

    In : path            str    e.g. "C:/Users/me/.cache/huggingface/hub/
                                datasets--nilq--babylm-100M/snapshots/a86a3e6c"
                                searched recursively
         allow_patterns  list[str] | str   e.g. ["data/train-*.parquet"]
                                matched against each file's path relative to
                                `path`, written with forward slashes
                                None -> keep every parquet found
    Out: list[str] sorted, absolute, e.g.
                ["C:/.../snapshots/a86a3e6c/data/train-00000-of-00002.parquet",
                 "C:/.../snapshots/a86a3e6c/data/train-00001-of-00002.parquet"]
                [] when the tree holds no parquet file
    """
    found = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not allow_patterns:
        return found

    if isinstance(allow_patterns, str):
        patterns = [allow_patterns]         # one glob, not a list of them
    else:
        patterns = list(allow_patterns)

    keep = []
    for found_path in found:
        # The patterns are repo-relative and written with "/", so compare
        # against the same shape rather than the absolute Windows path.
        relative = os.path.relpath(found_path, path).replace(os.sep, "/")
        for pattern in patterns:
            if fnmatch.fnmatch(relative, pattern):
                keep.append(found_path)
                break                       # one match is enough for this file
    return keep


class DataDownloader:
    """A chosen set of files from one dataset, downloaded once and cached.

    In : every setting at the top of the file, by keyword; anything not
         given keeps the global, e.g.
             dataset          str   e.g. "wikimedia/wikipedia"
             allow_patterns   list[str] | str | None
                                    e.g. ["20231101.vi/*.parquet"]
             repo_type        str   e.g. "dataset"
             revision         str   e.g. "refs/convert/parquet"; "" is default
             token            str   e.g. "hf_xxx"; "" finds your own
             max_workers      int   e.g. 16
             headroom         float e.g. 1.2
             local_dir        str   e.g. "D:/flat-copy"; "" uses the cache
             cache_dir        str   e.g. "D:/hf-cache"
             force            bool  e.g. True, past the disk guard
             xet_high_performance  bool  e.g. True
    Out: a DataDownloader, e.g.
         DataDownloader(dataset='nilq/babylm-100M',
                        allow_patterns=['data/train-*.parquet'])
         Nothing is downloaded until a method needs the files.
         Raises ValueError for a setting that cannot work, such as
         max_workers=0, and TypeError for a name that is not a setting.
    """

    def __init__(self, dataset=DATASET, allow_patterns=ALLOW_PATTERNS,
                 repo_type=REPO_TYPE, revision=REVISION, token=TOKEN,
                 max_workers=MAX_WORKERS, headroom=HEADROOM,
                 local_dir=LOCAL_DIR, cache_dir=CACHE_DIR, force=FORCE,
                 xet_high_performance=XET_HIGH_PERFORMANCE):
        # Catch what cannot work at all here, rather than halfway through a
        # download. A misspelled name needs no check: Python rejects it.
        if not dataset:
            raise ValueError("dataset must not be empty")
        if not cache_dir:
            raise ValueError("cache_dir must not be empty")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if headroom < 1.0:
            raise ValueError("headroom must be >= 1.0")

        self.dataset = dataset
        self.allow_patterns = allow_patterns
        self.repo_type = repo_type
        self.revision = revision
        self.token = token
        self.max_workers = max_workers
        self.headroom = headroom
        self.local_dir = local_dir
        self.cache_dir = cache_dir
        self.force = force
        self.xet_high_performance = xet_high_performance

        self._files = None              # filled in by the first download

    def __repr__(self):
        return (f"DataDownloader(dataset={self.dataset!r}, "
                f"allow_patterns={self.allow_patterns!r})")

    # ── look before downloading ─────────────────────────────────────────────
    def list_remote_files(self, allow_patterns=None):
        """List the matching files on the Hub, with sizes, without downloading.

        In : allow_patterns  list[str] | str   e.g. ["data/train-*.parquet"]
                             None -> self.allow_patterns, which is itself
                             None for every file in the repo
        Out: (files, total_bytes)
             files        list[dict] sorted by "name", e.g.
                            [{"name": "data/train-00000-of-00002.parquet",
                              "size": 152790334, "xet": False},
                             {"name": "data/train-00001-of-00002.parquet",
                              "size": 204588835, "xet": False}]
                            "name" str  repo-relative path
                            "size" int  bytes
                            "xet"  bool stored on Xet rather than plain LFS
             total_bytes  int   e.g. 357379169   what a download would move
        """
        from huggingface_hub import snapshot_download

        if allow_patterns is None:
            allow_patterns = self.allow_patterns

        # The settings spell "not set" as "", the Hub spells it None and reads
        # it as "you decide" — default branch, find your own token.
        infos = snapshot_download(
            self.dataset,
            repo_type=self.repo_type,
            revision=self.revision or None,
            allow_patterns=allow_patterns,
            token=self.token or None,
            dry_run=True,                   # metadata only: nothing is fetched
        )

        files = []
        total_bytes = 0
        for info in infos:
            size = info.file_size           # the Hub leaves this None sometimes
            if size is None:
                size = 0
            files.append({
                "name": info.filename,
                "size": size,
                # Only Xet-backed files carry xet_file_data, and older hub
                # versions do not set the attribute, hence getattr.
                "xet": getattr(info, "xet_file_data", None) is not None,
            })
            total_bytes += size

        files.sort(key=lambda entry: entry["name"])
        return files, total_bytes

    # ── the data ────────────────────────────────────────────────────────────
    def download_parquet_files(self):
        """Download every matching parquet file, and give back their paths.

        Downloads on the first call and remembers the result; only .parquet
        paths come back, though anything else the globs matched is downloaded.

        In : nothing.
        Out: list[str] sorted, local paths, e.g.
               ["C:/.../snapshots/a86a3e6c/data/train-00000-of-00002.parquet",
                "C:/.../snapshots/a86a3e6c/data/train-00001-of-00002.parquet"]
        """
        if self._files is None:             # not downloaded yet, so do it now
            patterns = self.allow_patterns
            snapshot_dir = self._download_snapshot(patterns)
            self._files = find_local_parquet_files(snapshot_dir, patterns)
        return self._files

    def stream_parquet_files(self, delete_after_use=True):
        """Download one parquet at a time, not the whole set at once.

        For a set that will not fit on disk: each file is deleted once you ask
        for the next, so read it before moving on — that makes this a one-pass
        reader. Pass False to keep every file instead.

        In : delete_after_use  bool   e.g. True, the default: the disk holds
                               one file at a time. False keeps them all.
        Out: generator of str, one local parquet path at a time, e.g.
               ".../data/CC-MAIN-2013-20/train-00000-of-00014.parquet"
             then ".../train-00001-of-00014.parquet", and so on.
             Calls sys.exit when no parquet file matches.
        """
        wanted, total = self.list_remote_files()
        names = []
        for entry in wanted:
            if entry["name"].endswith(".parquet"):
                names.append(entry["name"])
        if not names:
            sys.exit(f"[hgdatadownloader] no parquet files matched "
                     f"{self.allow_patterns!r} in {self.dataset}")

        if delete_after_use:
            deletion_note = " (deleted after use)"
        else:
            deletion_note = ""
        print(f"[hgdatadownloader] {len(names)} file(s), {format_bytes(total)} "
              f"total, one at a time{deletion_note}")

        previous = None
        for index, name in enumerate(names):
            # The previous file is deleted here rather than right after it was
            # yielded, because the consumer was still reading it until now.
            if previous is not None and delete_after_use:
                os.remove(previous)
                print(f"[hgdatadownloader] deleted {os.path.basename(previous)}")

            snapshot_dir = self._download_snapshot([name])  # <-- the download
            paths = find_local_parquet_files(snapshot_dir, [name])
            if not paths:
                sys.exit(f"[hgdatadownloader] {name} did not arrive")

            print(f"[hgdatadownloader] file {index + 1}/{len(names)}: "
                  f"{os.path.basename(paths[0])}")
            previous = paths[0]
            yield paths[0]

        # The last file has no next loop to clean it up.
        if previous is not None and delete_after_use:
            os.remove(previous)
            print(f"[hgdatadownloader] deleted {os.path.basename(previous)}")

    # ── the download itself ─────────────────────────────────────────────────
    def _enable_xet_high_performance(self):
        """Ask huggingface_hub for parallel chunk fetches, if the setting says so.

        In : nothing.
        Out: None. Prints a note when it is too late for it to work.
        """
        # huggingface_hub reads HF_XET_HIGH_PERFORMANCE once, when it is
        # imported, so setting it afterwards changes nothing. That is why the
        # download methods import it inside themselves, and why this checks
        # the constant rather than assuming the write worked.
        if self.xet_high_performance:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
            import huggingface_hub.constants as constants
            if not constants.HF_XET_HIGH_PERFORMANCE:
                print("[hgdatadownloader] xet_high_performance ignored: "
                      "huggingface_hub was already imported. Set "
                      "HF_XET_HIGH_PERFORMANCE=1 in the shell, or import "
                      "hgdatadownloader before transformers.")

    def _download_snapshot(self, allow_patterns):
        """Fetch the matching files and give back the directory they landed in.

        In : allow_patterns  list[str] | str | None   the globs for this one
                             call, e.g. ["data/train-00000-of-00014.parquet"]
        Out: str    e.g. "C:/Users/me/.cache/huggingface/hub/datasets--nilq--
                    babylm-100M/snapshots/a86a3e6c0c1977d895e8cdd5b02d47bb95d9"
             Calls sys.exit when nothing matches, or when the payload will not
             fit on disk and force is False.
        """
        self._enable_xet_high_performance()

        from huggingface_hub import snapshot_download

        files, total = self.list_remote_files(allow_patterns)
        if not files:
            sys.exit(f"[hgdatadownloader] no files matched {allow_patterns!r} "
                     f"in {self.dataset}")

        # local_dir asks for a flat copy of the files instead of the shared
        # cache layout. snapshot_download takes one or the other, never both,
        # and either way that is the volume the disk guard has to measure.
        # expanduser turns "~/..." into a real path; normpath keeps one
        # separator style.
        if self.local_dir:
            local_dir = os.path.normpath(os.path.expanduser(self.local_dir))
            cache_dir = None
            target = local_dir
        else:
            local_dir = None
            cache_dir = os.path.normpath(os.path.expanduser(self.cache_dir))
            target = cache_dir

        # The disk guard: a headroom of 1.2 demands 20% more room than the
        # payload.
        free = free_bytes(target)
        required = total * self.headroom
        if free <= required and not self.force:
            sys.exit(f"[hgdatadownloader] need ~{format_bytes(required)} but only "
                     f"{format_bytes(free)} free — narrow the patterns, "
                     f"or set force=True")

        print(f"[hgdatadownloader] fetching {len(files)} file(s), "
              f"{format_bytes(total)}, {self.max_workers} workers")

        # snapshot_download's thread pool is the fastest supported path: it
        # verifies each file and skips what is cached, so a re-run after a
        # crash costs almost nothing. Hand-rolling an async downloader here
        # would lose both.
        started = time.time()
        path = snapshot_download(
            self.dataset,
            repo_type=self.repo_type,
            revision=self.revision or None,     # "" means the default branch
            allow_patterns=allow_patterns,
            max_workers=self.max_workers,
            local_dir=local_dir,
            cache_dir=cache_dir,
            token=self.token or None,           # "" lets the Hub find its own
        )
        # An already-cached snapshot returns in no measurable time, and the
        # rate below divides by this, so keep it just above zero.
        elapsed = max(time.time() - started, 1e-9)
        rate = format_bytes(total / elapsed)

        print(f"[hgdatadownloader] done in {elapsed:.1f}s "
              f"({rate}/s effective) -> {path}")
        return path
