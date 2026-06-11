#!/usr/bin/env python3
"""Plot WebShop checkpoint evaluation metrics by training step.

The script scans checkpoint result directories such as:

  checkpoints/verl_agent_webshop/*_global_step_1200_act_state/summary.json

It groups points by experiment name, where the experiment name is the directory
name before ``_global_step_<step>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"^(?P<experiment>.+?)_global_step_(?P<step>\d+)(?:_|$)")


@dataclass(frozen=True)
class ResultPoint:
    experiment: str
    step: int
    success_rate: float
    mean_score: float
    num_episodes: int | None
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WebShop success rate and mean score by checkpoint step."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/verl_agent_webshop"),
        help="Root directory containing checkpoint evaluation result folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/webshop_checkpoint_results"),
        help="Directory where plots and CSV summary will be written.",
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
        default="webshop_checkpoint_results.csv",
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


def metric_from_keys(data: dict, keys: Iterable[str]) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_from_episodes(path: Path) -> tuple[float, float, int]:
    scores: list[float] = []
    wins: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if isinstance(episode.get("score"), (int, float)):
                scores.append(float(episode["score"]))
            if "won" in episode:
                wins.append(1.0 if episode["won"] else 0.0)

    if not scores and not wins:
        raise ValueError(f"no score or won fields found in {path}")

    mean_score = sum(scores) / len(scores) if scores else sum(wins) / len(wins)
    success_rate = sum(wins) / len(wins) if wins else mean_score
    return success_rate, mean_score, max(len(scores), len(wins))


def load_result(result_dir: Path) -> ResultPoint | None:
    match = STEP_RE.match(result_dir.name)
    if match is None:
        return None

    experiment = match.group("experiment")
    step = int(match.group("step"))

    summary_path = result_dir / "summary.json"
    episodes_path = result_dir / "episodes.jsonl"

    success_rate = None
    mean_score = None
    num_episodes = None

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        success_rate = metric_from_keys(
            summary,
            ("success_rate", "sucess_rate", "win_rate", "won_rate", "accuracy"),
        )
        mean_score = metric_from_keys(
            summary,
            ("mean_score", "avg_score", "average_score", "score"),
        )
        if isinstance(summary.get("num_episodes"), int):
            num_episodes = int(summary["num_episodes"])

    if (success_rate is None or mean_score is None) and episodes_path.exists():
        ep_success_rate, ep_mean_score, ep_num_episodes = load_from_episodes(episodes_path)
        success_rate = ep_success_rate if success_rate is None else success_rate
        mean_score = ep_mean_score if mean_score is None else mean_score
        num_episodes = ep_num_episodes if num_episodes is None else num_episodes

    if success_rate is None or mean_score is None:
        raise ValueError(f"missing metrics in {result_dir}")

    return ResultPoint(
        experiment=experiment,
        step=step,
        success_rate=success_rate,
        mean_score=mean_score,
        num_episodes=num_episodes,
        result_dir=result_dir,
    )


def collect_results(root: Path, include: str | None, exclude: str | None) -> list[ResultPoint]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    points: list[ResultPoint] = []
    errors: list[str] = []

    for result_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        if not (result_dir / "summary.json").exists() and not (result_dir / "episodes.jsonl").exists():
            continue
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


def write_csv(points: list[ResultPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "step",
                "success_rate",
                "mean_score",
                "num_episodes",
                "result_dir",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "experiment": point.experiment,
                    "step": point.step,
                    "success_rate": point.success_rate,
                    "mean_score": point.mean_score,
                    "num_episodes": point.num_episodes,
                    "result_dir": point.result_dir,
                }
            )


def print_table(points: list[ResultPoint]) -> None:
    print(f"{'experiment':80s} {'step':>7s} {'succ':>8s} {'score':>8s} {'n':>6s}")
    for point in points:
        n = "" if point.num_episodes is None else str(point.num_episodes)
        print(
            f"{point.experiment[:80]:80s} "
            f"{point.step:7d} "
            f"{point.success_rate:8.4f} "
            f"{point.mean_score:8.4f} "
            f"{n:>6s}"
        )


def plot_metric(
    grouped: dict[str, list[ResultPoint]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it or run with --list-only "
            "to generate only the CSV/table."
        ) from exc

    fig, ax = plt.subplots(figsize=(12, 7))
    for experiment, series in sorted(grouped.items()):
        series = sorted(series, key=lambda p: p.step)
        xs = [p.step for p in series]
        ys = [getattr(p, metric_name) for p in series]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=experiment)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
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

    csv_path = args.out_dir / args.csv_name
    write_csv(points, csv_path)
    print_table(points)
    print(f"\nSaved CSV: {csv_path}")

    if args.list_only:
        return 0

    grouped: dict[str, list[ResultPoint]] = defaultdict(list)
    for point in points:
        grouped[point.experiment].append(point)

    success_path = args.out_dir / "success_rate_by_step.png"
    score_path = args.out_dir / "mean_score_by_step.png"
    plot_metric(grouped, "success_rate", "Success Rate", success_path, args.show)
    plot_metric(grouped, "mean_score", "Mean Score", score_path, args.show)
    print(f"Saved plot: {success_path}")
    print(f"Saved plot: {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
