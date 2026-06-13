#!/usr/bin/env python3
"""Plot Mind2Web checkpoint evaluation metrics by training step.

The script scans checkpoint result directories such as:

  checkpoints/verl_agent_mind2web/*_global_step_1200/summary.json

It groups points by experiment name, where the experiment name is the directory
name before ``_global_step_<step>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"^(?P<experiment>.+?)_global_step_(?P<step>\d+)(?:_|$)")
DEFAULT_METRICS = (
    "exact_action_acc",
    "target_id_acc",
    "overall_exact_action_acc",
    "op_acc",
    "value_acc",
    "json_parse_error_rate",
)


@dataclass(frozen=True)
class ResultPoint:
    experiment: str
    step: int
    metrics: dict[str, float]
    num_examples: int | None
    num_target_id_examples: int | None
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Mind2Web summary.json metrics by checkpoint step."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/verl_agent_mind2web"),
        help="Root directory containing checkpoint evaluation result folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/mind2web_checkpoint_results"),
        help="Directory where plots and CSV summary will be written.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metric keys to plot from summary.json. Use --all-numeric-metrics "
            "to ignore this list and plot every numeric metric."
        ),
    )
    parser.add_argument(
        "--all-numeric-metrics",
        action="store_true",
        help="Plot every numeric key found in summary.json except counts.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Optional regex. Keep only experiment names matching this pattern.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Optional regex. Drop experiment names matching this pattern.",
    )
    parser.add_argument(
        "--csv-name",
        default="mind2web_checkpoint_results.csv",
        help="CSV filename to save under --out-dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving plots.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print and save the parsed table; do not import matplotlib.",
    )
    return parser.parse_args()


def numeric_metrics(data: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and key not in {"num_examples", "num_target_id_examples"}
    }


def load_result(result_dir: Path) -> ResultPoint | None:
    match = STEP_RE.match(result_dir.name)
    if match is None:
        return None

    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = numeric_metrics(summary)
    if not metrics:
        raise ValueError(f"no numeric metrics found in {summary_path}")

    num_examples = summary.get("num_examples")
    num_target_id_examples = summary.get("num_target_id_examples")

    return ResultPoint(
        experiment=match.group("experiment"),
        step=int(match.group("step")),
        metrics=metrics,
        num_examples=int(num_examples) if isinstance(num_examples, int) else None,
        num_target_id_examples=(
            int(num_target_id_examples) if isinstance(num_target_id_examples, int) else None
        ),
        result_dir=result_dir,
    )


def collect_results(root: Path, include: str | None, exclude: str | None) -> list[ResultPoint]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    points: list[ResultPoint] = []
    errors: list[str] = []

    for summary_path in sorted(root.rglob("summary.json")):
        result_dir = summary_path.parent
        try:
            point = load_result(result_dir)
        except Exception as exc:  # Keep scanning other checkpoints.
            errors.append(f"{result_dir}: {exc}")
            continue
        if point is None:
            continue
        if include_re and not include_re.search(point.experiment):
            continue
        if exclude_re and exclude_re.search(point.experiment):
            continue
        points.append(point)

    if errors:
        print("Skipped result folders with parse errors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    points.sort(key=lambda p: (p.experiment, p.step, str(p.result_dir)))
    return points


def selected_metrics(points: list[ResultPoint], requested: Iterable[str], all_numeric: bool) -> list[str]:
    available: set[str] = set()
    for point in points:
        available.update(point.metrics)

    if all_numeric:
        return sorted(available)

    missing = [metric for metric in requested if metric not in available]
    if missing:
        print(
            "Warning: requested metrics were not found and will be skipped: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    return [metric for metric in requested if metric in available]


def write_csv(points: list[ResultPoint], metrics: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "step",
        "num_examples",
        "num_target_id_examples",
        *metrics,
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row = {
                "experiment": point.experiment,
                "step": point.step,
                "num_examples": point.num_examples,
                "num_target_id_examples": point.num_target_id_examples,
                "result_dir": point.result_dir,
            }
            for metric in metrics:
                row[metric] = point.metrics.get(metric)
            writer.writerow(row)


def print_table(points: list[ResultPoint], metrics: list[str]) -> None:
    shown_metrics = metrics[:4]
    metric_headers = " ".join(f"{metric[:18]:>18s}" for metric in shown_metrics)
    print(f"{'experiment':72s} {'step':>7s} {'n':>7s} {metric_headers}")
    for point in points:
        n = "" if point.num_examples is None else str(point.num_examples)
        metric_values = []
        for metric in shown_metrics:
            value = point.metrics.get(metric)
            metric_values.append("" if value is None else f"{value:18.4f}")
        print(
            f"{point.experiment[:72]:72s} "
            f"{point.step:7d} "
            f"{n:>7s} "
            + " ".join(f"{value:>18s}" for value in metric_values)
        )
    if len(metrics) > len(shown_metrics):
        print(f"... {len(metrics) - len(shown_metrics)} more metric(s) written to CSV/plots")


def metric_title(metric: str) -> str:
    return metric.replace("_", " ").title()


def metric_filename(metric: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_")
    return f"{safe}_by_step.png"


def plot_metric(
    grouped: dict[str, list[ResultPoint]],
    metric: str,
    output_path: Path,
    show: bool,
) -> None:
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it or run with --list-only "
            "to generate only the CSV/table."
        ) from exc

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = 0
    for experiment, series in sorted(grouped.items()):
        series = sorted((point for point in series if metric in point.metrics), key=lambda p: p.step)
        if not series:
            continue
        xs = [point.step for point in series]
        ys = [point.metrics[metric] for point in series]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=experiment)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Step")
    ax.set_ylabel(metric_title(metric))
    ax.set_title(f"{metric_title(metric)} by Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    points = collect_results(args.root, args.include, args.exclude)
    if not points:
        print(f"No checkpoint results found under {args.root}", file=sys.stderr)
        return 1

    metrics = selected_metrics(points, args.metrics, args.all_numeric_metrics)
    if not metrics:
        print("No requested metrics found in collected summaries.", file=sys.stderr)
        return 1

    csv_path = args.out_dir / args.csv_name
    write_csv(points, metrics, csv_path)
    print_table(points, metrics)
    print(f"\nSaved CSV: {csv_path}")

    if args.list_only:
        return 0

    grouped: dict[str, list[ResultPoint]] = defaultdict(list)
    for point in points:
        grouped[point.experiment].append(point)

    for metric in metrics:
        output_path = args.out_dir / metric_filename(metric)
        plot_metric(grouped, metric, output_path, args.show)
        print(f"Saved plot: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
