#!/usr/bin/env python3
"""Find and optionally delete LoRA-merged full-model checkpoints.

The validation scripts create merged HuggingFace model directories for vLLM under
each eval output directory, e.g.:

  checkpoints/.../merged_lora_for_vllm_02af7ec95a

Those directories contain the base model merged with a LoRA adapter and are much
larger than adapter-only checkpoints. This script is dry-run by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


MERGED_PREFIX = "merged_lora_for_vllm_"
TMP_PREFIX = f".{MERGED_PREFIX}"
TMP_SUFFIX = ".tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or delete merged LoRA full-model checkpoints under checkpoints/."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="checkpoints",
        help="Checkpoint root to scan. Defaults to checkpoints.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched directories. Without this, only prints a dry-run.",
    )
    parser.add_argument(
        "--no-tmp",
        action="store_true",
        help="Do not include temporary .merged_lora_for_vllm_*.tmp directories.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary unless --delete fails.",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def is_target_dir(path: Path, include_tmp: bool) -> bool:
    name = path.name
    if name.startswith(MERGED_PREFIX):
        return True
    return include_tmp and name.startswith(TMP_PREFIX) and name.endswith(TMP_SUFFIX)


def directory_size(path: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not (Path(dirpath) / dirname).is_symlink()
        ]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                total += file_path.stat().st_size
            except OSError as exc:
                print(f"warning: could not stat {file_path}: {exc}", file=sys.stderr)
    return total


def find_targets(root: Path, include_tmp: bool) -> list[Path]:
    targets: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        matched = []
        kept = []
        for dirname in dirnames:
            candidate = current / dirname
            if candidate.is_symlink():
                kept.append(dirname)
            elif is_target_dir(candidate, include_tmp):
                matched.append(dirname)
                targets.append(candidate)
            else:
                kept.append(dirname)
        dirnames[:] = kept
    return sorted(targets)


def ensure_safe_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.exists():
        raise SystemExit(f"root does not exist: {root}")
    if not resolved.is_dir():
        raise SystemExit(f"root is not a directory: {root}")
    if resolved == Path("/"):
        raise SystemExit("refusing to scan filesystem root")
    return resolved


def main() -> int:
    args = parse_args()
    root = ensure_safe_root(Path(args.root))
    include_tmp = not args.no_tmp

    targets = find_targets(root, include_tmp=include_tmp)
    sized_targets = [(path, directory_size(path)) for path in targets]
    total_size = sum(size for _, size in sized_targets)

    mode = "DELETE" if args.delete else "DRY-RUN"
    if not args.quiet:
        print(f"[{mode}] root: {root}")
        print(f"[{mode}] include temporary dirs: {include_tmp}")
        print()
        for path, size in sized_targets:
            print(f"{human_size(size):>10}  {path}")
        if sized_targets:
            print()

    print(f"matched directories: {len(sized_targets)}")
    print(f"reclaimable size: {human_size(total_size)}")

    if not args.delete:
        print("dry-run only; rerun with --delete to remove these directories")
        return 0

    failed = False
    for path, _ in sized_targets:
        try:
            shutil.rmtree(path)
            if not args.quiet:
                print(f"deleted: {path}")
        except OSError as exc:
            failed = True
            print(f"failed to delete {path}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
