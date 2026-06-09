#!/usr/bin/env python3
"""
Build verl multiturn SFT data from the ToolBench release.

The script expects the ToolBench release zip under data/toolbench/data.zip by
default. It reuses ToolBench's official preprocessed ToolLLaMA JSON files and
converts their conversations to chat messages that only use system/user/assistant
roles, which keeps Hugging Face chat templates from failing on a raw "function"
role.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLBENCH_ROOT = REPO_ROOT / "data/toolbench"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/toolbench_sft"
TRAIN_MEMBER = "data/toolllama_G123_dfs_train.json"
EVAL_MEMBER = "data/toolllama_G123_dfs_eval.json"


def _json_array_items(fp: BinaryIO, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False

    while True:
        chunk = fp.read(chunk_size)
        if chunk:
            buffer += chunk.decode("utf-8")
        else:
            eof = True

        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    break
                if buffer[0] != "[":
                    raise ValueError("Expected a top-level JSON array.")
                started = True
                buffer = buffer[1:]
                continue

            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if not buffer:
                break

            try:
                item, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            buffer = buffer[idx:]
            if isinstance(item, dict):
                yield item

        if eof:
            break


def _open_member(toolbench_root: Path, member: str) -> BinaryIO:
    extracted = toolbench_root / member
    if extracted.exists():
        return extracted.open("rb")

    zip_path = toolbench_root / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find {extracted} or {zip_path}. Put ToolBench data.zip under {toolbench_root}."
        )
    zf = zipfile.ZipFile(zip_path)
    return zf.open(member, "r")


def _convert_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("from")
    content = str(message.get("value", ""))
    if role == "human":
        role = "user"
    if role == "gpt":
        role = "assistant"
    if role == "function":
        return {"role": "user", "content": "Observation:\n" + content}
    if role in {"system", "user", "assistant"}:
        return {"role": role, "content": content}
    return None


def convert_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
    raw_messages = record.get("conversations") or record.get("messages")
    if not isinstance(raw_messages, list):
        return None

    messages = []
    assistant_turns = 0
    function_turns = 0
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        converted = _convert_message(message)
        if converted is None:
            continue
        messages.append(converted)
        assistant_turns += int(converted["role"] == "assistant")
        function_turns += int(message.get("from") == "function")

    if not messages or assistant_turns == 0:
        return None
    return {
        "messages": messages,
        "id": record.get("id"),
        "split": split,
        "source_index": index,
        "assistant_turns": assistant_turns,
        "function_turns": function_turns,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]], *, allow_missing_parquet: bool) -> bool:
    try:
        import pandas as pd
    except ImportError:
        if allow_missing_parquet:
            return False
        raise RuntimeError("pandas/pyarrow are required to write parquet. Install repo requirements.txt.")
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        if allow_missing_parquet:
            return False
        raise RuntimeError("pyarrow or fastparquet is required to write parquet.")
    return True


def convert_split(
    toolbench_root: Path,
    member: str,
    split: str,
    *,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"records_seen": 0, "records_used": 0, "records_skipped": 0}
    with _open_member(toolbench_root, member) as f:
        for index, record in enumerate(_json_array_items(f)):
            stats["records_seen"] += 1
            row = convert_record(record, split=split, index=index)
            if row is None:
                stats["records_skipped"] += 1
                continue
            rows.append(row)
            stats["records_used"] += 1
            if limit is not None and len(rows) >= limit:
                break
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ToolBench ToolLLaMA data to verl multiturn SFT parquet.")
    parser.add_argument("--toolbench-root", type=Path, default=DEFAULT_TOOLBENCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-member", default=TRAIN_MEMBER)
    parser.add_argument("--eval-member", default=EVAL_MEMBER)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, train_stats = convert_split(args.toolbench_root, args.train_member, "train", limit=args.limit_train)
    test, test_stats = convert_split(args.toolbench_root, args.eval_member, "test", limit=args.limit_eval)

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_jsonl(args.output_dir / "all.jsonl", train + test)

    parquet_written = {"train": False, "test": False, "all": False}
    if args.write_parquet:
        parquet_written["train"] = write_parquet(
            args.output_dir / "train.parquet", train, allow_missing_parquet=args.allow_missing_parquet
        )
        parquet_written["test"] = write_parquet(
            args.output_dir / "test.parquet", test, allow_missing_parquet=args.allow_missing_parquet
        )
        parquet_written["all"] = write_parquet(
            args.output_dir / "all.parquet", train + test, allow_missing_parquet=args.allow_missing_parquet
        )

    metadata = {
        "toolbench_root": str(args.toolbench_root),
        "train_member": args.train_member,
        "eval_member": args.eval_member,
        "num_train": len(train),
        "num_test": len(test),
        "stats": {"train": train_stats, "test": test_stats},
        "parquet_written": parquet_written,
        "format": {
            "messages": "chat messages for data.multiturn.messages_key=messages",
            "function_role": "mapped to user messages prefixed with Observation:",
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
