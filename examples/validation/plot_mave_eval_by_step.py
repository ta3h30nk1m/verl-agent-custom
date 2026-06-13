#!/usr/bin/env python3
"""Collect MAVE eval summaries and plot metric curves by global step."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "primary_acc",
    "exact_match",
    "json_parse_error_rate",
    "field_value_accuracy",
    "value_match",
    "answer_match",
    "match_label_match",
)


TASKS = (
    "single_attribute_qa",
    "evidence_grounded_extraction",
    "multi_attribute_card_completion",
    "product_customer_qa",
    "faceted_search_filtering",
)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def parse_step(text: str) -> int | None:
    matches = re.findall(r"global_step_(\d+)", text)
    return int(matches[-1]) if matches else None


def infer_task(summary_path: Path, summary: dict[str, Any]) -> str:
    by_task = summary.get("by_task")
    if isinstance(by_task, dict) and len(by_task) == 1:
        return next(iter(by_task))
    text = str(summary_path)
    for task in TASKS:
        if task in text:
            return task
    return ""


def flatten_numeric(prefix: str, value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(next_prefix, item))
        return result
    if isinstance(value, bool):
        return {prefix: float(value)}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    return {}


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.rglob("summary.json"), key=lambda p: natural_key(str(p))):
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

        step = parse_step(str(summary_path)) or parse_step(str(summary.get("lora_adapter", "")))
        if step is None:
            continue

        task = infer_task(summary_path, summary)
        row: dict[str, Any] = {
            "task": task,
            "global_step": step,
            "run_name": summary_path.parent.name,
            "summary_path": str(summary_path),
            "output_dir": str(summary_path.parent),
            "eval_file": summary.get("eval_file", ""),
            "num_examples": summary.get("num_examples", ""),
            "model_path": summary.get("model_path", ""),
            "lora_adapter": summary.get("lora_adapter", ""),
        }

        metrics = flatten_numeric("", summary)
        for key, value in metrics.items():
            if key.startswith("by_task."):
                parts = key.split(".")
                if len(parts) >= 3 and parts[1] == task:
                    row[".".join(parts[2:])] = value
                row[key] = value
            else:
                row[key] = value
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise SystemExit("No MAVE summary rows found.")

    preferred = [
        "task",
        "global_step",
        "primary_acc",
        "exact_match",
        "field_value_accuracy",
        "value_match",
        "answer_match",
        "match_label_match",
        "json_parse_error_rate",
        "num_examples",
        "run_name",
        "summary_path",
        "lora_adapter",
    ]
    all_columns = sorted({key for row in rows for key in row})
    columns = [key for key in preferred if key in all_columns]
    columns.extend(key for key in all_columns if key not in columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


def select_metrics(rows: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    available = {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
    return [metric for metric in DEFAULT_METRICS if metric in available]


def plot_metric(rows: list[dict[str, Any]], metric: str, output_dir: Path, *, title_prefix: str) -> Path | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; wrote CSV only.")
        return None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if metric in row and row.get("task"):
            grouped[str(row["task"])].append(row)
    if not grouped:
        print(f"Skipping metric={metric}: no rows with that metric.")
        return None

    plt.figure(figsize=(10, 6))
    for task, task_rows in sorted(grouped.items()):
        task_rows = sorted(task_rows, key=lambda row: int(row["global_step"]))
        steps = [int(row["global_step"]) for row in task_rows]
        values = [float(row[metric]) for row in task_rows]
        plt.plot(steps, values, marker="o", linewidth=1.8, markersize=3.5, label=task)

    plt.xlabel("Global Step")
    plt.ylabel(metric)
    plt.title(f"{title_prefix}{metric} by Global Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metric.replace('.', '_')}_by_step.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote plot to {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MAVE evaluation metrics by checkpoint step.")
    parser.add_argument("--root", type=Path, default=Path("checkpoints/verl_agent_mave"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/verl_agent_mave/plots"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--task", action="append", choices=list(TASKS), default=None)
    parser.add_argument("--metric", action="append", default=None, help="Metric to plot. Can be repeated.")
    parser.add_argument("--keyword", action="append", default=None, help="Keep rows whose summary path contains this keyword. Can be repeated.")
    parser.add_argument("--title-prefix", default="MAVE ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.root)
    if args.task:
        rows = [row for row in rows if row.get("task") in set(args.task)]
    if args.keyword:
        rows = [
            row
            for row in rows
            if all(keyword in row["summary_path"] for keyword in args.keyword)
        ]

    if not rows:
        raise SystemExit(f"No rows matched under {args.root}")

    output_csv = args.csv or (args.output_dir / "mave_eval_by_step.csv")
    write_csv(rows, output_csv)

    metrics = select_metrics(rows, args.metric)
    for metric in metrics:
        plot_metric(rows, metric, args.output_dir, title_prefix=args.title_prefix)


if __name__ == "__main__":
    main()
