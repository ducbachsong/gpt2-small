"""checkpoint_hub.py — keeps a training run's checkpoints in a private model repo on the Hugging Face Hub, so a rented
machine that goes away takes no checkpoint with it, and a new machine can go on from the newest one.

    # on the training machine: the trainer, and this beside it
    torchrun --standalone --nproc_per_node=8 --no-python ./train_8bit --steps 10440 --data-ready 1 \\
        --checkpoint-dir ckpt_8bit > train.log 2>&1 &
    python data/checkpoint_hub.py watch --repo <user>/d24-8bit-checkpoints --dir ckpt_8bit --trainer-pid $!

    # on a new machine with as many GPUs (a checkpoint is one file per rank): the newest, then on from it
    python data/checkpoint_hub.py download --repo <user>/d24-8bit-checkpoints --dir ckpt_8bit
    torchrun ... ./train_8bit --steps 10440 --data-ready 1 --checkpoint-dir ckpt_8bit --resume ckpt_8bit

Both need a Hugging Face token (the repo is private): `hf auth login` once, or HF_TOKEN in the environment; watch's
must be able to write. The trainer never waits for the Hub: it saves on its own disk (llmc/helper/checkpoint.cuh,
checkpoint_8bit.cuh), and this process sends the save up while the GPUs go on.

What the Hub holds, as the trainer's folder does:
    step_00009000/   rank_0.bin ... rank_7.bin
    step_00010000/   rank_0.bin ... rank_7.bin
    latest           "step_00010000"

watch     every --seconds, reads the folder's `latest` (the trainer moves it only once every rank's file is whole).
          A step folder not on the Hub yet goes up in one commit with the Hub's new `latest`, and the oldest step
          folders go in the same commit, so the Hub keeps the newest --keep; then the repo's history is squashed, or
          the deleted files would still count against the account's storage. A save made while the last one was still
          going up is skipped if a newer one is there by then: only the newest matters. An upload that fails is tried
          again next time. It stops once the trainer's process (--trainer-pid) has ended and its last save is up, or
          once --final-step is up; with neither, at Ctrl-C.
download  brings one step folder (the Hub's `latest`, or --step) into --dir, checks every rank file's header (its
          step, and as many files as the run had GPUs), then writes --dir's `latest`, so `--resume <dir>` takes it.
"""
import argparse
import io
import os
import sys
import time

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # the Hub's bars would fill a long run's log; one line each instead

HEADER_INT64S = 32                      # a checkpoint file's header: 32 int64s (checkpoint.cuh, checkpoint_8bit.cuh)
CHECKPOINT_MAGICS = {20241103: "float or bfloat16 run", 20261118: "8-bit run"}   # header[0]
STEPS_DONE_INDEX = 2                     # header[2]: the step the file was saved at
WORLD_SIZE_INDEX = 4                     # header[4]: how many GPUs the run had
FOLDER_PREFIX = "step_"


def say(text):
    """Prints one line with the time in front, at once (the output usually goes to a log file)."""
    print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)


def folder_name_of(step):
    """Gives a step's checkpoint folder name, as checkpoint_folder_name in checkpoint.cuh: 1000 -> step_00001000."""
    return f"{FOLDER_PREFIX}{step:08d}"


def local_latest(folder):
    """Gives the step folder the trainer's `latest` names, or None while it has saved nothing."""
    try:
        with open(os.path.join(folder, "latest")) as latest_file:
            return latest_file.read().strip() or None
    except FileNotFoundError:
        return None


def rank_files_of(step_folder):
    """Gives a step folder's rank files by rank (rank_0.bin, rank_1.bin, ...); stops if a .partial is there:
    a rank still writing, so the folder is not whole."""
    names = os.listdir(step_folder)
    if any(name.endswith(".partial") for name in names):
        raise RuntimeError(f"{step_folder} still has a .partial file: not whole")
    ranks = [name for name in names if name.startswith("rank_") and name.endswith(".bin")]
    return sorted(ranks, key=lambda name: int(name[len("rank_"):-len(".bin")]))


def check_step_folder(step_folder, name):
    """Checks a step folder is one whole checkpoint: every rank file's header a checkpoint's, saved at the folder's
    step, by a run with as many GPUs as there are files. Gives (rank files, the run's kind); stops on anything else."""
    ranks = rank_files_of(step_folder)
    if not ranks:
        raise RuntimeError(f"{step_folder} has no rank files")
    step = int(name[len(FOLDER_PREFIX):])
    kinds = set()
    for rank_name in ranks:
        header = np.fromfile(os.path.join(step_folder, rank_name), dtype=np.int64, count=HEADER_INT64S)
        if len(header) < HEADER_INT64S or int(header[0]) not in CHECKPOINT_MAGICS:
            raise RuntimeError(f"{step_folder}/{rank_name} is not a checkpoint file")
        if int(header[STEPS_DONE_INDEX]) != step or int(header[WORLD_SIZE_INDEX]) != len(ranks):
            raise RuntimeError(f"{step_folder}/{rank_name}: saved at step {int(header[STEPS_DONE_INDEX])} by "
                               f"{int(header[WORLD_SIZE_INDEX])} GPUs, but the folder is step {step} with "
                               f"{len(ranks)} rank files")
        kinds.add(CHECKPOINT_MAGICS[int(header[0])])
    return ranks, ", ".join(sorted(kinds))


def hub_step_folders(api, repo):
    """Gives the step folders the Hub's repo holds, oldest first (the names sort as the steps do)."""
    files = api.list_repo_files(repo, repo_type="model")
    return sorted({path.split("/")[0] for path in files if path.startswith(FOLDER_PREFIX) and "/" in path})


def hub_latest(repo):
    """Gives the step folder the Hub's `latest` names, or None if the repo has no `latest` yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(repo, "latest", repo_type="model")
    except EntryNotFoundError:
        return None
    with open(path) as latest_file:
        return latest_file.read().strip() or None


def upload_step_folder(api, repo, folder, name, keep, squash):
    """Sends one step folder up with the Hub's new `latest` in one commit, deleting the Hub's oldest step folders in
    the same commit so it keeps the newest `keep`; then squashes the history so the deleted files free their storage."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    # 1. the folder, checked whole
    step_folder = os.path.join(folder, name)
    ranks, kind = check_step_folder(step_folder, name)
    total_bytes = sum(os.path.getsize(os.path.join(step_folder, rank_name)) for rank_name in ranks)

    # 2. what goes: the oldest on the Hub, past the newest `keep` with this one
    older = [old for old in hub_step_folders(api, repo) if old < name]
    going = older[:max(0, len(older) + 1 - keep)]

    # 3. one commit: the rank files, `latest`, the deletions
    operations = [CommitOperationAdd(f"{name}/{rank_name}", os.path.join(step_folder, rank_name)) for rank_name in ranks]
    operations.append(CommitOperationAdd("latest", io.BytesIO(f"{name}\n".encode())))
    operations += [CommitOperationDelete(f"{old}/", is_folder=True) for old in going]
    say(f"uploading {name}: {len(ranks)} rank file(s), {total_bytes / 1e9:.2f} GB ({kind})"
        + (f"; deleting {', '.join(going)}" if going else ""))
    started = time.time()
    api.create_commit(repo, repo_type="model", operations=operations,
                      commit_message=f"{name}: {len(ranks)} rank file(s)" + (f", {len(going)} older deleted" if going else ""))
    seconds = time.time() - started
    say(f"  up in {seconds:.0f}s ({total_bytes / 1e6 / max(seconds, 1e-9):.0f} MB/s)")

    # 4. the deleted files' storage back
    if squash and going:
        api.super_squash_history(repo_id=repo, repo_type="model", commit_message=f"{name} and the {keep - 1} before it")
        say("  history squashed: the deleted folders no longer use storage")


def process_alive(pid):
    """Tells if the process with this id is still running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                 # there, but another user's
        return True
    return True


def watch(repo, folder, keep, seconds, trainer_pid, final_step, squash):
    """Sends every new checkpoint the trainer saves in `folder` up to `repo`, until the trainer has ended and its
    last one is up (or --final-step is up)."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    uploaded = hub_latest(repo)
    say(f"watching {folder}/latest every {seconds}s for {repo} (newest {keep} kept; on the Hub now: {uploaded or 'nothing'})")
    final_name = folder_name_of(final_step) if final_step is not None else None
    while True:
        trainer_gone = trainer_pid is not None and not process_alive(trainer_pid)   # before reading `latest`: its last save seen
        name = local_latest(folder)
        if name is not None and name != uploaded:
            try:
                upload_step_folder(api, repo, folder, name, keep, squash)
                uploaded = name
            except Exception as error:      # the network, the Hub, a folder not whole: next time
                say(f"  {name} not up ({type(error).__name__}: {error}); trying again in {seconds}s")
        pending = name is not None and name != uploaded
        if not pending and final_name is not None and uploaded == final_name:
            say(f"{final_name} is up: the run's last checkpoint; done")
            return
        if not pending and trainer_gone:
            say(f"the trainer (process {trainer_pid}) has ended and its last checkpoint ({uploaded or 'none'}) is up; done")
            return
        time.sleep(seconds)


def download(repo, folder, step, workers):
    """Brings one checkpoint (the Hub's `latest`, or `step`) into `folder` and points `folder`'s `latest` at it."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    # 1. which one
    name = folder_name_of(step) if step is not None else hub_latest(repo)
    on_hub = hub_step_folders(api, repo)
    if name is None or name not in on_hub:
        raise SystemExit(f"{repo} has no checkpoint {name or '(no latest)'}; it has: {', '.join(on_hub) or 'none'}")

    # 2. its files, then the check, then `latest`
    say(f"downloading {name} from {repo} into {folder}")
    started = time.time()
    snapshot_download(repo_id=repo, repo_type="model", local_dir=folder, allow_patterns=[f"{name}/*"],
                      max_workers=workers)
    ranks, kind = check_step_folder(os.path.join(folder, name), name)
    with open(os.path.join(folder, "latest.partial"), "w") as latest_file:
        latest_file.write(f"{name}\n")
    os.replace(os.path.join(folder, "latest.partial"), os.path.join(folder, "latest"))
    total_bytes = sum(os.path.getsize(os.path.join(folder, name, rank_name)) for rank_name in ranks)
    seconds = time.time() - started
    say(f"{name}: {len(ranks)} rank file(s) ({kind}), {total_bytes / 1e9:.2f} GB in {seconds:.0f}s "
        f"({total_bytes / 1e6 / max(seconds, 1e-9):.0f} MB/s); go on with --resume {folder} on {len(ranks)} GPU(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["watch", "download"])
    parser.add_argument("--repo", required=True, help="the private model repo on the Hub: <user>/<name>")
    parser.add_argument("--dir", required=True, help="the trainer's --checkpoint-dir")
    parser.add_argument("--keep", type=int, default=2, help="watch: the newest step folders the Hub keeps")
    parser.add_argument("--seconds", type=int, default=30, help="watch: how often it reads `latest`")
    parser.add_argument("--trainer-pid", type=int, help="watch: stop once this process has ended and its last save is up")
    parser.add_argument("--final-step", type=int, help="watch: stop once this step's checkpoint is up (the run's --steps)")
    parser.add_argument("--no-squash", action="store_true", help="watch: keep the history (deleted folders keep their storage)")
    parser.add_argument("--step", type=int, help="download: this step's checkpoint instead of the Hub's latest")
    parser.add_argument("--workers", type=int, default=8, help="download: files at once")
    args = parser.parse_args()
    if args.keep < 1 or args.seconds < 1:
        raise SystemExit("--keep and --seconds must be at least 1")
    if args.action == "watch":
        watch(args.repo, args.dir, args.keep, args.seconds, args.trainer_pid, args.final_step, not args.no_squash)
    else:
        download(args.repo, args.dir, args.step, args.workers)


if __name__ == "__main__":
    sys.exit(main())
