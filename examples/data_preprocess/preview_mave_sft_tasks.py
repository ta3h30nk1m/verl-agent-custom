#!/usr/bin/env python3
"""
Print one sample from each MAVE SFT task dataset.

The script expects the task-separated layout produced by mave_sft.py:
  <data-dir>/by_task/<task_type>/<split>.parquet

It falls back to JSONL when parquet cannot be read or does not exist.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data/mave_sft_amazon23"
DEFAULT_TASKS = (
    "single_attribute_qa",
    "evidence_grounded_extraction",
    "multi_attribute_card_completion",
    "product_customer_qa",
    "faceted_search_filtering",
)


def iter_jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def choose_jsonl_row(path: Path, *, index: int, random_sample: bool, seed: int, task: str) -> tuple[int, int | None, dict[str, Any]]:
    if random_sample:
        rng = random.Random(f"{seed}:{task}")
        chosen_idx = -1
        chosen_row = None
        count = 0
        for count, row in enumerate(iter_jsonl_rows(path), start=1):
            if rng.randrange(count) == 0:
                chosen_idx = count - 1
                chosen_row = row
        if chosen_row is None:
            raise ValueError(f"No rows available in {path}")
        return chosen_idx, count, chosen_row

    if index < 0:
        rows = list(iter_jsonl_rows(path))
        if not rows:
            raise ValueError(f"No rows available in {path}")
        idx = max(0, len(rows) + index)
        idx = min(idx, len(rows) - 1)
        return idx, len(rows), rows[idx]

    for idx, row in enumerate(iter_jsonl_rows(path)):
        if idx == index:
            return idx, None, row

    rows = list(iter_jsonl_rows(path))
    if not rows:
        raise ValueError(f"No rows available in {path}")
    return len(rows) - 1, len(rows), rows[-1]


def load_rows(task_dir: Path, split: str, *, prefer_parquet: bool) -> tuple[list[dict[str, Any]], Path] | None:
    parquet_path = task_dir / f"{split}.parquet"
    if prefer_parquet and parquet_path.exists():
        try:
            return read_parquet(parquet_path), parquet_path
        except ImportError:
            pass

    return None


def sample_row(task_dir: Path, split: str, args: argparse.Namespace, task: str) -> tuple[int, int | None, dict[str, Any], Path]:
    jsonl_path = task_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        idx, count, row = choose_jsonl_row(
            jsonl_path,
            index=args.index,
            random_sample=args.random,
            seed=args.seed,
            task=task,
        )
        return idx, count, row, jsonl_path

    loaded = load_rows(task_dir, split, prefer_parquet=True)
    if loaded is not None:
        rows, source_path = loaded
        idx, row = choose_row(rows, index=args.index, random_sample=args.random, seed=args.seed, task=task)
        return idx, len(rows), row, source_path

    raise FileNotFoundError(f"No {split}.parquet or {split}.jsonl under {task_dir}")


def discover_tasks(task_root: Path) -> list[str]:
    if not task_root.exists():
        return []
    tasks = [
        path.name
        for path in task_root.iterdir()
        if path.is_dir() and ((path / "train.parquet").exists() or (path / "train.jsonl").exists())
    ]
    ordered = [task for task in DEFAULT_TASKS if task in tasks]
    ordered.extend(sorted(task for task in tasks if task not in set(ordered)))
    return ordered


def choose_row(rows: list[dict[str, Any]], *, index: int, random_sample: bool, seed: int, task: str) -> tuple[int, dict[str, Any]]:
    if not rows:
        raise ValueError(f"No rows available for task {task}")
    if random_sample:
        rng = random.Random(f"{seed}:{task}")
        idx = rng.randrange(len(rows))
    else:
        idx = index
        if idx < 0:
            idx = len(rows) + idx
        idx = max(0, min(idx, len(rows) - 1))
    return idx, rows[idx]


def compact_text(text: Any, max_chars: int) -> str:
    text = str(text if text is not None else "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def pretty_json_or_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return compact_text(json.dumps(value, ensure_ascii=False, indent=2), max_chars)
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            return compact_text(json.dumps(parsed, ensure_ascii=False, indent=2), max_chars)
        except json.JSONDecodeError:
            pass
    return compact_text(value, max_chars)


def print_sample(task: str, idx: int, total: int | None, row: dict[str, Any], source_path: Path, args: argparse.Namespace) -> None:
    print("=" * 100)
    print(f"TASK: {task}")
    print(f"SOURCE: {source_path}")
    total_text = str(total) if total is not None else "unknown"
    print(f"ROW: {idx + 1}/{total_text}")
    print(f"PRODUCT_ID: {row.get('product_id', '')}")
    print(f"CATEGORY: {row.get('category', '')}")
    if row.get("attribute") is not None:
        print(f"ATTRIBUTE: {row.get('attribute')}")
    if row.get("target_attributes") is not None:
        print(f"TARGET_ATTRIBUTES: {row.get('target_attributes')}")
    print("-" * 100)
    print("PROMPT")
    print(pretty_json_or_text(row.get("prompt"), args.max_prompt_chars))
    print("-" * 100)
    print("RESPONSE")
    print(pretty_json_or_text(row.get("response"), args.max_response_chars))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one sample from each task-specific MAVE SFT dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--task-root-name", default="by_task")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--task", action="append", default=None, help="Specific task to preview. Can be repeated.")
    parser.add_argument("--index", type=int, default=0, help="0-based row index. Negative values count from the end.")
    parser.add_argument("--random", action="store_true", help="Sample one deterministic random row per task.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-parquet", action="store_true", help="Read parquet files instead of streaming JSONL.")
    parser.add_argument("--max-prompt-chars", type=int, default=5000)
    parser.add_argument("--max-response-chars", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_root = args.data_dir / args.task_root_name
    tasks = args.task or discover_tasks(task_root)
    if not tasks:
        raise RuntimeError(f"No task datasets found under {task_root}")

    missing = []
    for task in tasks:
        task_dir = task_root / task
        try:
            if args.prefer_parquet:
                loaded = load_rows(task_dir, args.split, prefer_parquet=True)
                if loaded is None:
                    idx, total, row, source_path = sample_row(task_dir, args.split, args, task)
                else:
                    rows, source_path = loaded
                    idx, row = choose_row(rows, index=args.index, random_sample=args.random, seed=args.seed, task=task)
                    total = len(rows)
            else:
                idx, total, row, source_path = sample_row(task_dir, args.split, args, task)
        except FileNotFoundError:
            missing.append(task)
            continue
        print_sample(task, idx, total, row, source_path, args)

    if missing:
        raise FileNotFoundError(f"Missing requested task dataset(s): {', '.join(missing)}")


if __name__ == "__main__":
    main()
