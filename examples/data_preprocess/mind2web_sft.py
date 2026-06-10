#!/usr/bin/env python3
"""
Download Mind2Web and build step-level SFT data for next-action prediction.

The output is compatible with verl's MultiTurnSFTDataset:
  data.multiturn.enable=true
  data.multiturn.messages_key=messages

Official Mind2Web test files are distributed as a password-protected zip to
reduce benchmark contamination. This script downloads the zip from the dataset
repo and extracts it locally, but never uploads or redistributes the unzipped
test files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data/mind2web_raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/mind2web_sft"
DEFAULT_HF_REPO_ID = "osunlp/Mind2Web"
DEFAULT_SPLITS = ("train", "test_task", "test_website", "test_domain")

SYSTEM_PROMPT = (
    "You are a Mind2Web web agent. Given the task, previous actions, and the "
    "current webpage state, output exactly one JSON action and no explanation."
)

ACTION_SCHEMA_HINT = (
    'Return exactly one JSON object with keys "op", "value", "target_id", and '
    '"target". Valid op values are CLICK, TYPE, and SELECT. Use an empty string '
    'for value when the operation has no text or option value.'
)


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def iter_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if "actions" in data:
            return [data]
    raise ValueError(f"Unsupported Mind2Web json structure: {path}")


def find_split_files(raw_dir: Path, split: str) -> list[Path]:
    candidates = [
        raw_dir / split,
        raw_dir / "data" / split,
        raw_dir / "Mind2Web" / "data" / split,
    ]
    files: list[Path] = []
    for directory in candidates:
        if directory.exists():
            files.extend(sorted(directory.glob("*.json")))
    if files:
        return files
    return sorted(raw_dir.glob(f"**/data/{split}/*.json"))


def ensure_train_data(args: argparse.Namespace) -> None:
    if find_split_files(args.raw_dir, "train") or args.no_download:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Mind2Web.") from exc

    snapshot_download(
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        local_dir=args.raw_dir,
        allow_patterns=["data/train/*.json", "README.md"],
    )


def extract_zip(zip_path: Path, output_dir: Path, password: str) -> None:
    pwd = password.encode("utf-8")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir, pwd=pwd)
        return
    except RuntimeError:
        unzip = shutil.which("unzip")
        if not unzip:
            raise RuntimeError(
                f"Could not extract {zip_path}. Install unzip or provide an already extracted --raw-dir."
            )
        subprocess.run([unzip, "-P", password, "-o", str(zip_path), "-d", str(output_dir)], check=True)


def ensure_test_data(args: argparse.Namespace) -> None:
    if args.no_download or not args.download_test:
        return
    missing = [split for split in args.splits if split != "train" and not find_split_files(args.raw_dir, split)]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Mind2Web test.zip.") from exc

    zip_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.test_zip_name,
            repo_type="dataset",
            local_dir=args.raw_dir,
        )
    )
    extract_zip(zip_path, args.raw_dir, args.test_zip_password)


def parse_attributes(candidate: dict[str, Any]) -> dict[str, Any]:
    attrs = candidate.get("attributes")
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, str) and attrs.strip():
        try:
            parsed = json.loads(attrs)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def candidate_key(candidate: dict[str, Any]) -> str:
    attrs = parse_attributes(candidate)
    payload = {
        "backend_node_id": candidate.get("backend_node_id"),
        "tag": candidate.get("tag"),
        "attributes": attrs,
        "text": candidate.get("text") or candidate.get("text_context") or candidate.get("inner_text"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def candidate_text(candidate: dict[str, Any]) -> str:
    attrs = parse_attributes(candidate)
    tag = clean_space(candidate.get("tag")) or "element"
    text = clean_space(
        candidate.get("text")
        or candidate.get("text_context")
        or candidate.get("inner_text")
        or candidate.get("value")
    )
    attr_keys = ("role", "type", "name", "aria-label", "placeholder", "title", "value", "href", "id")
    rendered_attrs = []
    for key in attr_keys:
        value = clean_space(attrs.get(key))
        if value:
            rendered_attrs.append(f'{key}="{value[:160]}"')
    attr_text = " " + " ".join(rendered_attrs) if rendered_attrs else ""
    backend_id = clean_space(candidate.get("backend_node_id"))
    backend_text = f" backend_node_id={backend_id}" if backend_id else ""
    body = f" text=\"{text[:300]}\"" if text else ""
    return f"<{tag}{attr_text}{backend_text}>{body}".strip()


def dedup_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def choose_target_candidate(pos_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pos_candidates:
        return None
    for candidate in pos_candidates:
        if candidate.get("is_original_target") and candidate.get("is_top_level_target"):
            return candidate
    for candidate in pos_candidates:
        if candidate.get("is_top_level_target"):
            return candidate
    for candidate in pos_candidates:
        if candidate.get("is_original_target"):
            return candidate
    return pos_candidates[0]


def build_candidate_block(
    action: dict[str, Any],
    *,
    max_candidates: int,
    shuffle_candidates: bool,
) -> tuple[str, str | None, str]:
    pos_candidates = dedup_candidates(action.get("pos_candidates") or [])
    neg_candidates = dedup_candidates(action.get("neg_candidates") or [])
    target = choose_target_candidate(pos_candidates)
    target_key = candidate_key(target) if target else None

    target_candidates = [c for c in pos_candidates if candidate_key(c) == target_key]
    other_candidates = [c for c in pos_candidates + neg_candidates if candidate_key(c) != target_key]

    rng = random.Random(stable_int(str(action.get("action_uid", ""))))
    if shuffle_candidates:
        rng.shuffle(other_candidates)

    if max_candidates > 0:
        keep = target_candidates[:max_candidates]
        keep.extend(other_candidates[: max(0, max_candidates - len(keep))])
    else:
        keep = target_candidates + other_candidates

    if shuffle_candidates:
        rng.shuffle(keep)

    target_id = None
    target_rendered = candidate_text(target) if target else ""
    lines = []
    for idx, candidate in enumerate(keep):
        candidate_id = f"C{idx:03d}"
        if target_key and candidate_key(candidate) == target_key:
            target_id = candidate_id
        lines.append(f"{candidate_id}: {candidate_text(candidate)}")

    if not lines:
        return "none", None, target_rendered
    return "\n".join(lines), target_id, target_rendered


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return text[:half].rstrip() + f"\n... <truncated {omitted} chars> ...\n" + text[-half:].lstrip()


def operation_to_response(operation: dict[str, Any], target_id: str | None, target: str, action_repr: str) -> str:
    op = clean_space(operation.get("op") or operation.get("original_op")).upper()
    if op in {"HOVER", "ENTER"}:
        op = "CLICK"
    value = clean_space(operation.get("value"))
    payload = {
        "op": op or "CLICK",
        "value": value,
        "target_id": target_id,
        "target": target,
    }
    if not target and action_repr:
        payload["target"] = action_repr
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def previous_action_text(record: dict[str, Any], step_idx: int) -> str:
    action_reprs = record.get("action_reprs")
    if isinstance(action_reprs, list) and action_reprs:
        prev = [clean_space(x) for x in action_reprs[:step_idx] if clean_space(x)]
        return "\n".join(f"- {item}" for item in prev) if prev else "none"

    actions = record.get("actions") or []
    prev_lines = []
    for prev_action in actions[:step_idx]:
        operation = prev_action.get("operation") or {}
        op = clean_space(operation.get("op") or operation.get("original_op")).upper()
        value = clean_space(operation.get("value"))
        suffix = f": {value}" if value else ""
        prev_lines.append(f"- {op}{suffix}")
    return "\n".join(prev_lines) if prev_lines else "none"


def action_repr_at(record: dict[str, Any], step_idx: int) -> str:
    action_reprs = record.get("action_reprs")
    if isinstance(action_reprs, list) and step_idx < len(action_reprs):
        return clean_space(action_reprs[step_idx])
    return ""


def html_for_action(action: dict[str, Any], source: str, max_chars: int) -> str:
    if source == "raw_html":
        html = clean_space(action.get("raw_html"))
    else:
        html = clean_space(action.get("cleaned_html") or action.get("raw_html"))
    return truncate_text(html, max_chars)


def build_state_text(
    record: dict[str, Any],
    action: dict[str, Any],
    step_idx: int,
    args: argparse.Namespace,
) -> tuple[str, str | None, str]:
    candidate_block, target_id, target = build_candidate_block(
        action,
        max_candidates=args.max_candidates,
        shuffle_candidates=args.shuffle_candidates,
    )
    blocks = [
        "<state>",
        f"website: {clean_space(record.get('website'))}",
        f"domain: {clean_space(record.get('domain'))}",
        f"subdomain: {clean_space(record.get('subdomain'))}",
        f"step_index: {step_idx}",
        "previous_actions:",
        previous_action_text(record, step_idx),
    ]

    if args.state_source in {"html", "html_and_candidates"}:
        blocks.extend(
            [
                "current_html:",
                html_for_action(action, args.html_field, args.max_html_chars),
            ]
        )

    if args.state_source in {"candidates", "html_and_candidates"}:
        blocks.extend(["candidate_elements:", candidate_block])

    blocks.append("</state>")
    return "\n".join(blocks), target_id, target


def build_prompt(record: dict[str, Any], state_text: str) -> str:
    return "\n\n".join(
        [
            f"Task: {clean_space(record.get('confirmed_task'))}",
            f"Website: {clean_space(record.get('website'))}",
            "Current webpage state:\n" + state_text,
            ACTION_SCHEMA_HINT,
        ]
    )


def build_examples_for_record(
    record: dict[str, Any],
    *,
    split: str,
    source_file: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    examples = []
    actions = record.get("actions") or []
    if not isinstance(actions, list):
        return examples

    for step_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        operation = action.get("operation") or {}
        if not isinstance(operation, dict):
            continue
        state_text, target_id, target = build_state_text(record, action, step_idx, args)
        prompt = build_prompt(record, state_text)
        action_repr = action_repr_at(record, step_idx)
        response = operation_to_response(operation, target_id, target, action_repr)
        train_split = "train" if split == "train" else "test"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        examples.append(
            {
                "messages": messages,
                "prompt": prompt,
                "response": response,
                "split": train_split,
                "mind2web_split": split,
                "source_file": str(source_file.relative_to(args.raw_dir)) if source_file.is_relative_to(args.raw_dir) else str(source_file),
                "annotation_id": record.get("annotation_id"),
                "action_uid": action.get("action_uid"),
                "step_idx": step_idx,
                "website": record.get("website"),
                "domain": record.get("domain"),
                "subdomain": record.get("subdomain"),
                "confirmed_task": record.get("confirmed_task"),
                "action_repr": action_repr,
                "operation": operation,
                "target_id": target_id,
                "target": target,
                "state_source": args.state_source,
                "html_field": args.html_field,
                "num_pos_candidates": len(action.get("pos_candidates") or []),
                "num_neg_candidates": len(action.get("neg_candidates") or []),
            }
        )
    return examples


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "files_seen": 0,
        "records_seen": 0,
        "records_used": 0,
        "records_skipped_no_actions": 0,
        "examples_by_split": Counter(),
    }

    for split in args.splits:
        files = find_split_files(args.raw_dir, split)
        if not files:
            if split == "train":
                raise FileNotFoundError(f"No files found for required split {split} under {args.raw_dir}")
            continue
        for path in files:
            stats["files_seen"] += 1
            records = iter_records(path)
            for record in records:
                stats["records_seen"] += 1
                if not record.get("actions"):
                    stats["records_skipped_no_actions"] += 1
                    continue
                record_examples = build_examples_for_record(record, split=split, source_file=path, args=args)
                if args.limit_per_split is not None:
                    current_count = stats["examples_by_split"][split]
                    remaining = max(0, args.limit_per_split - current_count)
                    record_examples = record_examples[:remaining]
                if not record_examples:
                    continue
                examples.extend(record_examples)
                stats["records_used"] += 1
                stats["examples_by_split"][split] += len(record_examples)
                if args.limit_per_split is not None and stats["examples_by_split"][split] >= args.limit_per_split:
                    break
            if args.limit_per_split is not None and stats["examples_by_split"][split] >= args.limit_per_split:
                break

    stats["examples_by_split"] = dict(stats["examples_by_split"])
    return examples, stats


def sample_balanced_validation(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    split_names: list[str],
    ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"--balanced-val-ratio must be in (0, 1], got {ratio}")

    rows: list[dict[str, Any]] = []
    counts = {}
    for split in split_names:
        split_rows = list(grouped.get(split, []))
        if not split_rows:
            counts[split] = 0
            continue
        sample_size = max(1, int(len(split_rows) * ratio))
        rng = random.Random(seed + stable_int(split))
        indices = sorted(rng.sample(range(len(split_rows)), sample_size))
        sampled = [split_rows[idx] for idx in indices]
        rows.extend(sampled)
        counts[split] = len(sampled)
    return rows, counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def maybe_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        return False
    return True


def write_outputs(args: argparse.Namespace, examples: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in examples if row["split"] == "train"]
    test = [row for row in examples if row["split"] == "test"]

    grouped: dict[str, list[dict[str, Any]]] = {"train": train, "test": test, "all": examples}
    for split in args.splits:
        grouped[split] = [row for row in examples if row["mind2web_split"] == split]

    balanced_val_counts = None
    if args.balanced_val_ratio is not None:
        balanced_val, balanced_val_counts = sample_balanced_validation(
            grouped,
            split_names=args.balanced_val_splits,
            ratio=args.balanced_val_ratio,
            seed=args.balanced_val_seed,
        )
        grouped[args.balanced_val_output_name] = balanced_val

    for name, rows in grouped.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    parquet_written = {}
    for name, rows in grouped.items():
        if args.write_parquet:
            parquet_written[name] = maybe_write_parquet(args.output_dir / f"{name}.parquet", rows)
            if rows and not parquet_written[name] and not args.allow_missing_parquet:
                raise RuntimeError(
                    "Failed to write parquet. Install repo data dependencies, for example "
                    "`pip install pandas pyarrow`, or rerun with --no-write-parquet for JSONL-only output."
                )
        else:
            parquet_written[name] = False

    metadata = {
        "hf_repo_id": args.hf_repo_id,
        "raw_dir": str(args.raw_dir),
        "stats": stats,
        "num_examples": len(examples),
        "num_train": len(train),
        "num_test": len(test),
        "splits": list(args.splits),
        "state_source": args.state_source,
        "html_field": args.html_field,
        "max_html_chars": args.max_html_chars,
        "max_candidates": args.max_candidates,
        "shuffle_candidates": args.shuffle_candidates,
        "balanced_val_ratio": args.balanced_val_ratio,
        "balanced_val_splits": args.balanced_val_splits,
        "balanced_val_output_name": args.balanced_val_output_name,
        "balanced_val_counts": balanced_val_counts,
        "parquet_written": parquet_written,
        "format": {
            "messages": "chat messages for data.multiturn.messages_key=messages",
            "response": "JSON string with op, value, target_id, and target",
            "prompt": "task plus previous actions and current webpage state",
        },
        "train_command_hint": (
            "Use data.train_files=<output_dir>/train.parquet "
            "data.val_files=<output_dir>/test.parquet data.multiturn.enable=true "
            "data.multiturn.messages_key=messages"
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mind2Web next-action SFT data for verl.")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=list(DEFAULT_SPLITS))
    parser.add_argument("--no-download", action="store_true", help="Use files already present under --raw-dir.")
    parser.add_argument("--download-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test-zip-name", default="test.zip")
    parser.add_argument("--test-zip-password", default="mind2web")
    parser.add_argument(
        "--state-source",
        choices=["html", "candidates", "html_and_candidates"],
        default="html_and_candidates",
    )
    parser.add_argument("--html-field", choices=["cleaned_html", "raw_html"], default="cleaned_html")
    parser.add_argument("--max-html-chars", type=int, default=12000)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum rendered candidates per step. The positive target is kept when available.",
    )
    parser.add_argument("--shuffle-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--balanced-val-ratio",
        type=float,
        default=None,
        help=(
            "If set, sample this same fraction from each Mind2Web test split and "
            "write it as a separate validation file."
        ),
    )
    parser.add_argument(
        "--balanced-val-splits",
        nargs="+",
        default=["test_task", "test_website", "test_domain"],
        choices=["test_task", "test_website", "test_domain"],
    )
    parser.add_argument("--balanced-val-output-name", default="val_sample")
    parser.add_argument("--balanced-val-seed", type=int, default=0)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-parquet",
        action="store_true",
        help="Do not fail when pandas/pyarrow is unavailable and parquet files cannot be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_train_data(args)
    ensure_test_data(args)
    examples, stats = build_examples(args)
    write_outputs(args, examples, stats)


if __name__ == "__main__":
    main()
