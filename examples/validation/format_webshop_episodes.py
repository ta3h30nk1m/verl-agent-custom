#!/usr/bin/env python3
"""Render WebShop lite episode JSONL files into readable trajectories."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("checkpoints/verl_agent_webshop/webshop_lite_validation/episodes.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _last_match(pattern: str, text: str) -> str:
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))
    return matches[-1].group(1).strip() if matches else ""


def extract_task(prompt: str) -> str:
    patterns = [
        r"Current task:\s*(.*?)\n(?:\n?Subgoal state block:|Prior to this step|Current observation:)",
        r"Your task is to:\s*(.*?)\n(?:Prior to this step|Your current observation is:)",
        r"Task:\s*(.*?)\n\nObservation:",
    ]
    for pattern in patterns:
        value = _last_match(pattern, prompt)
        if value:
            return value.rstrip(".").strip()
    return ""


def extract_observation(prompt: str) -> str:
    patterns = [
        r"Current observation:\s*(.*?)\nCurrent admissible actions:",
        r"Your current observation is:\s*(.*?)\nYour admissible actions of the current situation are:",
        r"Observation:\s*(.*?)\nAdmissible actions:",
    ]
    for pattern in patterns:
        value = _last_match(pattern, prompt)
        if value:
            return value
    return ""


def extract_prompt_actions(prompt: str) -> list[str]:
    action_patterns = [
        r"Current admissible actions:\s*\[\n(.*?)\n\]\.?",
        r"Your admissible actions of the current situation are:\s*\[\n(.*?)\n\]\.?",
        r"Admissible actions:\s*\[\n(.*?)\n\]\.?",
    ]
    block = ""
    for pattern in action_patterns:
        block = _last_match(pattern, prompt)
        if block:
            break
    actions = []
    for line in block.splitlines():
        value = line.strip().rstrip(",")
        if not value:
            continue
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value.strip("'\"")
        actions.append(str(parsed))
    return actions


def extract_subgoal_state_block(prompt: str) -> str:
    return _last_match(r"Subgoal state block:\s*(.*?)\n\nOne-step context:", prompt)


def extract_one_step_context(prompt: str) -> str:
    return _last_match(r"One-step context:\s*(.*?)\n\nCurrent observation:", prompt)


def extract_history(prompt: str) -> str:
    return _last_match(
        r"Prior to this step, you have already taken .*? Below are the most recent .*? observations and the corresponding actions you took:\s*(.*?)\nYou are now at step",
        prompt,
    )


def get_clickables(step: dict[str, Any]) -> list[str]:
    available = step.get("info", {}).get("available_actions", {})
    clickables = available.get("clickables") or []
    has_search = available.get("has_search_bar")
    actions = [f"click[{item}]" for item in clickables]
    if has_search:
        actions.insert(0, "search[<query>]")
    return actions


def unwrap_observation(text: str) -> str:
    text = text.strip()
    if (text.startswith("'") and text.endswith("'.")) or (text.startswith('"') and text.endswith('".')):
        text = text[:-1].rstrip()
    elif text.endswith(".."):
        text = text[:-1]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.replace("' [SEP] '", "\n- ").replace(" [SEP] ", "\n- ")


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + "\n... [truncated]"


def fenced(text: str, language: str = "") -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def render_step(
    step: dict[str, Any],
    *,
    max_observation_chars: int,
    max_context_chars: int,
    max_actions: int,
    include_full_prompt: bool,
) -> str:
    prompt = step.get("prompt") or step.get("input") or ""
    observation = unwrap_observation(extract_observation(prompt))
    subgoal_state_block = extract_subgoal_state_block(prompt)
    one_step_context = extract_one_step_context(prompt)
    history = extract_history(prompt)
    actions = extract_prompt_actions(prompt) or get_clickables(step)
    shown_actions = actions if max_actions <= 0 else actions[:max_actions]
    action_suffix = "" if max_actions <= 0 or len(actions) <= max_actions else f"\n... {len(actions) - max_actions} more"

    lines = [
        f"### Turn {step.get('turn_index', '?')}",
    ]
    if subgoal_state_block:
        lines.extend(
            [
                "",
                "**Subgoal State**",
                "",
                fenced(truncate(subgoal_state_block, max_context_chars), "text"),
            ]
        )
    if one_step_context and one_step_context != "none":
        lines.extend(
            [
                "",
                "**One-Step Context**",
                "",
                fenced(truncate(unwrap_observation(one_step_context), max_context_chars), "text"),
            ]
        )
    if history:
        lines.extend(
            [
                "",
                "**History**",
                "",
                fenced(truncate(unwrap_observation(history), max_context_chars), "text"),
            ]
        )
    lines.extend(
        [
        "",
        "**Observation**",
        "",
        fenced(truncate(observation or "(missing)", max_observation_chars), "text"),
        "",
        "**Admissible Actions**",
        "",
        fenced("\n".join(shown_actions) + action_suffix if shown_actions else "(missing)", "text"),
        "",
        "**Assistant Action**",
        "",
        fenced(str(step.get("output", "")).strip() or "(empty)", "text"),
        "",
        f"- parsed_action: `{step.get('parsed_action', '')}`",
        f"- valid: `{bool(step.get('is_action_valid', False))}`",
        f"- reward: `{step.get('reward', 0.0)}`",
        f"- done: `{bool(step.get('done', False))}`",
        ]
    )
    info = step.get("info") or {}
    if "task_score" in info:
        lines.append(f"- task_score: `{info.get('task_score')}`")
    if "won" in info:
        lines.append(f"- won: `{bool(info.get('won'))}`")
    if include_full_prompt:
        lines.extend(["", "<details>", "<summary>Full model input</summary>", "", fenced(step.get("input", ""), "text"), "", "</details>"])
    return "\n".join(lines)


def render_episode(
    episode: dict[str, Any],
    *,
    max_observation_chars: int,
    max_context_chars: int,
    max_actions: int,
    include_full_prompt: bool,
) -> str:
    steps = episode.get("steps") or []
    task = extract_task(steps[0].get("prompt", "")) if steps else ""
    header_lines = [
        f"## Episode {episode.get('episode_index', '?')}",
        "",
        f"- score: `{episode.get('score', 0.0)}`",
        f"- won: `{bool(episode.get('won', False))}`",
        f"- episode_reward: `{episode.get('episode_reward', 0.0)}`",
        f"- episode_length: `{episode.get('episode_length', len(steps))}`",
    ]
    if task:
        header_lines.extend(["", "**Task**", "", fenced(task, "text")])
    body = [
        render_step(
            step,
            max_observation_chars=max_observation_chars,
            max_context_chars=max_context_chars,
            max_actions=max_actions,
            include_full_prompt=include_full_prompt,
        )
        for step in steps
    ]
    return "\n\n".join(["\n".join(header_lines)] + body)


def select_episodes(rows: list[dict[str, Any]], selected: list[int] | None) -> list[dict[str, Any]]:
    if not selected:
        return rows
    wanted = set(selected)
    return [row for row in rows if int(row.get("episode_index", -1)) in wanted]


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".jsonl"):
        return input_path.with_name(input_path.name[:-6] + ".pretty.md")
    return input_path.with_suffix(input_path.suffix + ".pretty.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format WebShop lite episodes.jsonl into readable Markdown.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--episode", type=int, action="append", help="Episode index to render. Can be repeated.")
    parser.add_argument("--split-episodes", action="store_true", help="Write one Markdown file per episode.")
    parser.add_argument("--max-observation-chars", type=int, default=2500)
    parser.add_argument("--max-context-chars", type=int, default=2500)
    parser.add_argument("--max-actions", type=int, default=40, help="0 means show all admissible actions.")
    parser.add_argument("--include-full-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_episodes(load_jsonl(args.input), args.episode)
    if not rows:
        raise ValueError("No episodes matched the requested filters.")

    output_path = args.output or default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split_episodes:
        output_dir = output_path if output_path.suffix == "" else output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        for episode in rows:
            episode_index = episode.get("episode_index", "unknown")
            path = output_dir / f"episode_{episode_index}.md"
            path.write_text(
                render_episode(
                    episode,
                    max_observation_chars=args.max_observation_chars,
                    max_context_chars=args.max_context_chars,
                    max_actions=args.max_actions,
                    include_full_prompt=args.include_full_prompt,
                )
                + "\n"
            )
        print(f"Wrote {len(rows)} episode files to {output_dir}")
        return

    rendered = ["\n".join(["# WebShop Lite Trajectories", "", f"- source: `{args.input}`", f"- episodes: `{len(rows)}`"])]
    rendered.extend(
        render_episode(
            episode,
            max_observation_chars=args.max_observation_chars,
            max_context_chars=args.max_context_chars,
            max_actions=args.max_actions,
            include_full_prompt=args.include_full_prompt,
        )
        for episode in rows
    )
    output_path.write_text("\n\n".join(rendered) + "\n")
    print(f"Wrote {len(rows)} episodes to {output_path}")


if __name__ == "__main__":
    main()
