#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def flatten_json(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        rows: dict[str, Any] = {}
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten_json(value, next_prefix))
        return rows

    if isinstance(data, list):
        return {prefix: json.dumps(data, ensure_ascii=False)}

    return {prefix: data}


def parse_run_metadata(run_name: str) -> dict[str, str]:
    metadata = {
        "model_size": "",
        "global_step": "",
        "epoch": "",
        "is_lora": "false",
    }

    size_match = re.search(r"hyperclovax_(0_5B|1_5B|3B)", run_name)
    if size_match:
        metadata["model_size"] = size_match.group(1).replace("_", ".")

    step_match = re.search(r"global_step_(\d+)", run_name)
    if step_match:
        metadata["global_step"] = step_match.group(1)

    epoch_match = re.search(r"_(\d+)epoch_", run_name)
    if epoch_match:
        metadata["epoch"] = epoch_match.group(1)

    if "lora" in run_name.lower():
        metadata["is_lora"] = "true"

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect summary.json files under result folders into one CSV."
    )
    parser.add_argument(
        "--root",
        default="checkpoints/verl_agent_webshop",
        help="Root directory to search for summary.json files.",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/verl_agent_webshop/summary_results.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    summary_paths = sorted(root.rglob("summary.json"), key=lambda p: natural_key(str(p)))

    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        output_dir = summary_path.parent
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

        run_name = output_dir.name
        row: dict[str, Any] = {
            "run_name": run_name,
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
        }
        row.update(parse_run_metadata(run_name))
        row.update(flatten_json(summary))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")

    preferred_columns = [
        "run_name",
        "model_size",
        "is_lora",
        "epoch",
        "global_step",
        "num_episodes",
        "goal_start",
        "goal_end",
        "parallel_envs",
        "mean_score",
        "success_rate",
        "mean_episode_length",
        "output_dir",
        "summary_path",
    ]
    all_columns = sorted({key for row in rows for key in row.keys()})
    columns = [key for key in preferred_columns if key in all_columns]
    columns.extend(key for key in all_columns if key not in columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
