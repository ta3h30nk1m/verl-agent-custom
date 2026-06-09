#!/usr/bin/env python3
"""
Build step-level WebShop SFT data with compact subgoal state blocks.

The input trajectories in user_session_logs/all_trajs store visited page states,
not explicit actions. This script reconstructs the expert action from each
state transition, renders a bounded-context prompt, and pairs it with the next
action for SFT.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBSHOP_ROOT = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop"
DEFAULT_TRAJ_DIR = WEBSHOP_ROOT / "user_session_logs/all_trajs"
DEFAULT_PRODUCT_FILE = WEBSHOP_ROOT / "data/items_shuffle.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/webshop_subgoal_sft"

SUBPAGE_NAMES = {"description", "features", "reviews", "attributes"}


def load_act_state_template() -> str:
    prompt_path = REPO_ROOT / "agent_system/environments/prompts/webshop.py"
    spec = importlib.util.spec_from_file_location("webshop_prompt_templates", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load WebShop prompt templates from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_webshop_prompt_template("act_state", use_history=False)


WEBSHOP_ACT_STATE_TEMPLATE = load_act_state_template()


class WebshopSubgoalStateTracker:
    CONTROL_CLICKS = {
        "search",
        "back to search",
        "next >",
        "< prev",
        "description",
        "features",
        "reviews",
        "attributes",
        "buy now",
    }
    TASK_TOKEN_STOPWORDS = {
        "and", "are", "for", "from", "item", "less", "like", "looking", "lower",
        "need", "than", "the", "this", "want", "with", "would",
    }

    def __init__(self, task_description: str):
        self.task_description = task_description
        self.task_text = task_description.lower()
        self.current_query = None
        self.queries_tried = []
        self.inspected_product = None
        self.selected_options = {}
        self.checked_detail_pages = set()
        self.visited_products = set()

    @staticmethod
    def _action_arg(action: str, name: str) -> str:
        action = str(action or "").strip()
        prefix = f"{name}["
        if action.lower().startswith(prefix) and action.endswith("]"):
            return action[len(prefix):-1].strip()
        return ""

    @staticmethod
    def _is_asin(text: str) -> bool:
        return bool(re.fullmatch(r"b[0-9a-z]{9}", str(text or "").strip().lower()))

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _join_or_none(values) -> str:
        values = list(values)
        return ", ".join(str(value) for value in values) if values else "none"

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @classmethod
    def _clean_observation_token(cls, text: str) -> str:
        token = cls._clean_text(text)
        if token.endswith("."):
            token = token[:-1].rstrip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            token = token[1:-1].strip()
        return token

    @classmethod
    def _observation_tokens(cls, current_observation: str) -> list[str]:
        return [
            token
            for token in (cls._clean_observation_token(part) for part in str(current_observation or "").split(" [SEP] "))
            if token
        ]

    @classmethod
    def _observation_field(cls, current_observation: str, label: str, next_labels: list[str]) -> str:
        text = str(current_observation or "")
        for line in text.splitlines():
            if line.lower().startswith(label.lower() + ":"):
                return cls._clean_text(line.split(":", 1)[1])
        next_label_pattern = "|".join(re.escape(next_label) for next_label in next_labels)
        pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s+(?:{next_label_pattern}):|$|')"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return cls._clean_text(match.group(1)) if match else ""

    def _structured_option_groups(self, current_observation: str, available_actions: list[str]) -> list[tuple[str, list[str]]]:
        candidate_norms = {self._norm(option) for option in self._candidate_options(available_actions)}
        groups = []
        in_options = False
        for line in str(current_observation or "").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("available options:"):
                in_options = True
                continue
            if not in_options:
                continue
            if not stripped.startswith("- ") or ":" not in stripped:
                break
            category, raw_values = stripped[2:].split(":", 1)
            values = [value.strip() for value in raw_values.split(", ") if value.strip()]
            if not candidate_norms or any(self._norm(value) in candidate_norms for value in values):
                groups.append((category.strip(), values))
        return groups

    def _raw_option_groups(self, current_observation: str, available_actions: list[str]) -> list[tuple[str, list[str]]]:
        tokens = self._observation_tokens(current_observation)
        candidate_by_norm = {self._norm(option): option for option in self._candidate_options(available_actions)}
        if not tokens or not candidate_by_norm:
            return []

        groups = []
        control = self.CONTROL_CLICKS | {"rating", "n.a.", "page"}
        for idx, token in enumerate(tokens[:-1]):
            token_norm = self._norm(token)
            next_norm = self._norm(tokens[idx + 1])
            if next_norm not in candidate_by_norm:
                continue
            if token_norm in candidate_by_norm or token_norm in control or self._is_asin(token_norm):
                continue
            if token_norm.startswith("price:") or token_norm.startswith("$") or token_norm.startswith("page "):
                continue

            values = []
            cursor = idx + 1
            while cursor < len(tokens):
                value_norm = self._norm(tokens[cursor])
                if value_norm not in candidate_by_norm:
                    break
                values.append(candidate_by_norm[value_norm])
                cursor += 1
            if values:
                groups.append((token, values))
        return groups

    def _option_groups(self, current_observation: str, available_actions: list[str]) -> list[tuple[str, list[str]]]:
        groups = self._structured_option_groups(current_observation, available_actions)
        return groups if groups else self._raw_option_groups(current_observation, available_actions)

    def _option_categories(self, current_observation: str, available_actions: list[str]) -> list[str]:
        return [category for category, _ in self._option_groups(current_observation, available_actions)]

    def _option_category_for_value(
        self,
        value: str,
        current_observation: str,
        available_actions: list[str],
    ) -> str:
        value_norm = self._norm(value)
        for category, values in self._option_groups(current_observation, available_actions):
            if any(self._norm(option_value) == value_norm for option_value in values):
                return category
        return "option"

    def _selected_options_text(self) -> str:
        if not self.selected_options:
            return "none"
        return ", ".join(f"{category}: {value}" for category, value in self.selected_options.items())

    def update(self, previous_action: str, current_observation: str, available_actions: list[str]) -> None:
        action = str(previous_action or "").strip()
        search_arg = self._action_arg(action, "search")
        click_arg = self._action_arg(action, "click")

        if search_arg:
            self.current_query = search_arg
            if search_arg not in self.queries_tried:
                self.queries_tried.append(search_arg)
            self.inspected_product = None
        elif click_arg:
            click_norm = self._norm(click_arg)
            if self._is_asin(click_norm):
                self.inspected_product = click_norm.upper()
                self.visited_products.add(self.inspected_product)
            elif click_norm in {"description", "features", "reviews", "attributes"}:
                self.checked_detail_pages.add(click_norm)
            elif click_norm == "back to search":
                self.inspected_product = None
            elif click_norm not in self.CONTROL_CLICKS:
                category = self._option_category_for_value(click_arg, current_observation, available_actions)
                self.selected_options[category] = click_arg

        obs_lower = str(current_observation or "").lower()
        for page_name in ("description", "features", "reviews", "attributes"):
            if f"'{page_name}'" in obs_lower:
                self.checked_detail_pages.add(page_name)

    def _candidate_options(self, available_actions: list[str]) -> list[str]:
        candidates = []
        for action in available_actions:
            click_arg = self._action_arg(action, "click")
            click_norm = self._norm(click_arg)
            if not click_arg or click_norm in self.CONTROL_CLICKS or self._is_asin(click_norm):
                continue
            candidates.append(click_arg)
        return candidates

    def _remaining_option_categories(self, current_observation: str, available_actions: list[str]) -> list[str]:
        selected_categories = {self._norm(category) for category in self.selected_options}
        return [
            category
            for category in self._option_categories(current_observation, available_actions)
            if self._norm(category) not in selected_categories
        ]

    def _phase(self, available_actions: list[str]) -> str:
        action_set = {self._norm(action) for action in available_actions}
        has_search = any(action.startswith("search[") for action in action_set)
        has_buy = "click[buy now]" in action_set
        has_product = any(self._is_asin(self._action_arg(action, "click")) for action in available_actions)
        has_options = bool(self._candidate_options(available_actions))

        if has_search:
            return "formulate_query"
        if has_product and not has_buy:
            return "browse_results"
        if has_buy and has_options:
            return "select_options"
        if has_buy:
            return "purchase"
        return "evaluate_item"

    def _current_item(self, current_observation: str) -> str:
        next_labels = ["Price", "Category", "Rating", "Selected options", "Available options", "Options"]
        for label in ("Title", "Product"):
            value = self._observation_field(current_observation, label, next_labels)
            if value and not self._is_asin(value):
                return value
        tokens = self._observation_tokens(current_observation)
        for idx, token in enumerate(tokens):
            token_norm = self._norm(token)
            if token_norm.startswith("price:") or token_norm.startswith("$") or token_norm.startswith("rating"):
                for candidate in reversed(tokens[:idx]):
                    candidate_norm = self._norm(candidate)
                    if (
                        candidate_norm not in self.CONTROL_CLICKS
                        and not self._is_asin(candidate_norm)
                        and not candidate_norm.startswith("page ")
                    ):
                        return candidate
        return self.inspected_product or "none"

    def _current_price(self, current_observation: str) -> str:
        next_labels = ["Category", "Rating", "Selected options", "Available options", "Options", "Features", "Description"]
        value = self._observation_field(current_observation, "Price", next_labels)
        if value:
            return value
        for token in self._observation_tokens(current_observation):
            token_norm = self._norm(token)
            if token_norm.startswith("price:"):
                return self._clean_text(token.split(":", 1)[1])
            if token_norm.startswith("$"):
                return token
        return "unknown"

    def render(self, current_observation: str, available_actions: list[str]) -> str:
        phase = self._phase(available_actions)
        remaining_options = self._remaining_option_categories(current_observation, available_actions)
        if phase == "select_options" and not remaining_options:
            phase = "purchase"
        purchase_ready = "click[buy now]" in {self._norm(action) for action in available_actions} and not remaining_options
        option_candidates = self._option_categories(current_observation, available_actions)
        lines = [
            "<state>",
            f"current_phase: {phase}",
        ]
        if phase == "formulate_query":
            lines.extend(
                [
                    f"queries_tried: {self._join_or_none(self.queries_tried)}",
                    f"items_inspected: {self._join_or_none(sorted(self.visited_products))}",
                ]
            )
        elif phase == "browse_results":
            lines.extend(
                [
                    f"current_query: {self.current_query or 'none'}",
                    f"items_already_inspected: {self._join_or_none(sorted(self.visited_products))}",
                ]
            )
        elif phase == "evaluate_item":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_available: {self._join_or_none(option_candidates)}",
                    f"features_checked: {str('features' in self.checked_detail_pages).lower()}",
                    f"description_checked: {str('description' in self.checked_detail_pages).lower()}",
                ]
            )
        elif phase == "select_options":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                    f"options_remaining: {self._join_or_none(remaining_options)}",
                ]
            )
        elif phase == "purchase":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                    f"all_options_filled: {str(purchase_ready).lower()}",
                ]
            )
        else:
            lines.extend(
                [
                    f"current_query: {self.current_query or 'none'}",
                    f"current_item: {self._current_item(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                ]
            )
        lines.append("</state>")
        return "\n".join(lines)


def _safe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(unquote(value))
    except (ValueError, SyntaxError):
        return value


def _as_list(value: Any) -> list[str]:
    value = _safe_literal(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [str(v).strip().lower() for v in value.values() if str(v).strip()]
    text = str(value).strip()
    return [text.lower()] if text else []


def _as_dict(value: Any) -> dict[str, str]:
    value = _safe_literal(value)
    if not isinstance(value, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in value.items()}


def _clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _price_to_float(price: Any) -> float | None:
    if price is None:
        return None
    if isinstance(price, list):
        return _price_to_float(price[0] if price else None)
    if isinstance(price, (int, float)):
        return float(price)
    match = re.search(r"\d+(?:\.\d+)?", str(price).replace(",", ""))
    return float(match.group(0)) if match else None


def _price_text(item: dict[str, Any]) -> str:
    if item.get("Price"):
        return _clean_space(item.get("Price"))
    pricing = item.get("pricing")
    if isinstance(pricing, list) and pricing:
        return "$" + " to $".join(str(x) for x in pricing[:2])
    return _clean_space(pricing)


def _normalize_product(item: dict[str, Any]) -> dict[str, Any]:
    asin = str(item.get("asin", "")).upper()
    options: dict[str, list[str]] = {}
    raw_options = item.get("options")
    if isinstance(raw_options, dict):
        for option_name, option_values in raw_options.items():
            values = [str(v).strip().lower() for v in option_values or [] if str(v).strip()]
            if values:
                options[str(option_name).lower()] = values
    elif item.get("customization_options"):
        for option_name, option_values in item.get("customization_options", {}).items():
            if not option_values:
                continue
            values = []
            for option_value in option_values:
                value = str(option_value.get("value", "")).strip().replace("/", " | ").lower()
                if value:
                    values.append(value)
            if values:
                options[str(option_name).lower()] = values
    bullets = item.get("BulletPoints", item.get("small_description")) or []
    return {
        "asin": asin,
        "title": _clean_space(item.get("Title", item.get("name"))),
        "price": _price_to_float(item.get("pricing")),
        "price_text": _price_text(item),
        "category": _clean_space(item.get("product_category")),
        "query": _clean_space(item.get("query")).lower(),
        "description": _clean_space(item.get("Description", item.get("full_description"))),
        "bullets": [_clean_space(x) for x in bullets if _clean_space(x)],
        "options": options,
    }


def _load_products(path: Path, needed_asins: set[str] | None = None) -> dict[str, dict[str, Any]]:
    products = {}
    if path.suffix == ".jsonl":
        with path.open() as f:
            lines = f
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                asin = str(record.get("id", "")).upper()
                if needed_asins is not None and asin not in needed_asins:
                    continue
                item = record.get("product") or {}
                item.setdefault("asin", asin)
                products[asin] = _normalize_product(item)
        return products

    for item in json.loads(path.read_text()):
        asin = str(item.get("asin", "")).upper()
        if not asin:
            continue
        if needed_asins is not None and asin not in needed_asins:
            continue
        products[asin] = _normalize_product(item)
    return products


def _collect_needed_asins(traj_dir: Path) -> set[str]:
    asins: set[str] = set()
    for path in traj_dir.glob("*.jsonl"):
        for row in _read_traj(path):
            asin = _asin_from_row(row)
            if asin:
                asins.add(asin)
            for result_asin in (row.get("content") or {}).get("search_result_asins") or []:
                asins.add(str(result_asin).upper())
    return asins


def _read_traj(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _session_id(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _keywords_from_row(row: dict[str, Any]) -> list[str]:
    content = row.get("content") or {}
    return _as_list(content.get("keywords"))


def _options_from_row(row: dict[str, Any]) -> dict[str, str]:
    content = row.get("content") or {}
    return _as_dict(content.get("options"))


def _asin_from_row(row: dict[str, Any]) -> str | None:
    content = row.get("content") or {}
    asin = content.get("asin")
    return str(asin).upper() if asin else None


def _subpage_from_url(row: dict[str, Any]) -> str | None:
    url = str(row.get("url", "")).lower()
    for name in SUBPAGE_NAMES:
        if f"/{name}/" in url or url.endswith(f"/{name}"):
            return name
    parts = [p for p in url.split("/") if p]
    for part in parts:
        if part in SUBPAGE_NAMES:
            return part
    return None


def infer_action(current: dict[str, Any], next_row: dict[str, Any]) -> str | None:
    next_page = next_row.get("page")
    current_page = current.get("page")

    if next_page == "search_results":
        keywords = _keywords_from_row(next_row)
        if keywords:
            return f"search[{' '.join(keywords)}]"
        return None

    if next_page == "item_page":
        cur_asin = _asin_from_row(current)
        next_asin = _asin_from_row(next_row)
        cur_options = _options_from_row(current)
        next_options = _options_from_row(next_row)

        if current_page == "item_sub_page":
            return "click[< prev]"
        if next_asin and (not cur_asin or next_asin != cur_asin):
            return f"click[{next_asin}]"
        for key, value in next_options.items():
            if cur_options.get(key) != value:
                return f"click[{value}]"
        if current_page == "search_results" and next_asin:
            return f"click[{next_asin}]"
        return None

    if next_page == "item_sub_page":
        subpage = _subpage_from_url(next_row)
        return f"click[{subpage}]" if subpage else None

    if next_page == "done":
        return "click[buy now]"

    if next_page == "index" and current_page != "index":
        return "click[back to search]"

    return None


def normalize_action(action: str) -> str:
    return _clean_space(action).lower()


def render_observation(row: dict[str, Any], products: dict[str, dict[str, Any]]) -> str:
    page = row.get("page")
    goal = row.get("goal") or {}
    content = row.get("content") or {}
    lines = [
        f"Page: {page}",
        f"Instruction: {_clean_space(goal.get('instruction_text'))}",
    ]

    if page == "index":
        lines.append("Available: search box")

    elif page == "search_results":
        keywords = _keywords_from_row(row)
        lines.append(f"Search query: {' '.join(keywords)}")
        lines.append(f"Result page: {content.get('page', 'unknown')}")
        lines.append("Visible products:")
        for asin in content.get("search_result_asins") or []:
            asin = str(asin).upper()
            product = products.get(asin, {})
            price = product.get("price_text") or "unknown price"
            title = product.get("title") or "unknown title"
            lines.append(f"- {asin}: {title}; price {price}")
        lines.append("Available: click[ASIN], click[next >], click[< prev]")

    elif page in {"item_page", "item_sub_page"}:
        asin = _asin_from_row(row)
        product = products.get(asin or "", {})
        lines.extend(
            [
                f"Product: {asin or 'unknown'}",
                f"Title: {product.get('title') or 'unknown'}",
                f"Price: {product.get('price_text') or 'unknown'}",
                f"Category: {product.get('category') or 'unknown'}",
            ]
        )
        if page == "item_sub_page":
            subpage = _subpage_from_url(row) or "detail"
            lines.append(f"Detail page: {subpage}")
            if subpage == "description":
                lines.append(f"Description: {product.get('description') or 'none'}")
            elif subpage == "features":
                bullets = product.get("bullets") or []
                lines.append("Features: " + ("; ".join(bullets[:8]) if bullets else "none"))
        else:
            options = product.get("options") or {}
            selected = _options_from_row(row)
            lines.append(f"Selected options: {json.dumps(selected, ensure_ascii=True, sort_keys=True)}")
            if options:
                lines.append("Available options:")
                for name, values in sorted(options.items()):
                    lines.append(f"- {name}: {', '.join(values[:20])}")
            else:
                lines.append("Available options: none")
            lines.append("Available: click[option], click[description], click[features], click[buy now], click[back to search]")

    else:
        lines.append(f"Content: {json.dumps(content, ensure_ascii=True, sort_keys=True)}")

    return "\n".join(lines)


def build_available_actions(row: dict[str, Any], products: dict[str, dict[str, Any]]) -> list[str]:
    page = row.get("page")
    content = row.get("content") or {}
    actions: list[str] = []

    if page == "index":
        return ["search[<your query>]", "click[search]"]

    if page == "search_results":
        actions.append("click[back to search]")
        try:
            page_num = int(content.get("page", 1))
        except (TypeError, ValueError):
            page_num = 1
        if page_num > 1:
            actions.append("click[< prev]")
        actions.append("click[next >]")
        for asin in content.get("search_result_asins") or []:
            actions.append(f"click[{str(asin).lower()}]")
        return actions

    if page == "item_sub_page":
        actions.extend(
            [
                "click[back to search]",
                "click[< prev]",
                "click[description]",
                "click[features]",
                "click[reviews]",
            ]
        )
        return actions

    if page == "item_page":
        actions.extend(
            [
                "click[back to search]",
                "click[< prev]",
                "click[description]",
                "click[features]",
                "click[reviews]",
                "click[buy now]",
            ]
        )
        asin = _asin_from_row(row)
        product = products.get(asin or "", {})
        for values in (product.get("options") or {}).values():
            for value in values:
                action = f"click[{value}]"
                if action not in actions:
                    actions.append(action)
        return actions

    return actions


def ensure_expert_action_available(actions: list[str], action: str) -> list[str]:
    if not action.startswith("click["):
        return actions
    if action not in actions:
        return actions + [action]
    return actions


def format_available_actions(actions: list[str]) -> str:
    return "\n".join(f"'{action}'," for action in actions)


def build_one_step_context(prev_observation: str | None, prev_action: str | None) -> str:
    if prev_observation is None or prev_action is None:
        return "none"
    return f"Previous observation: {prev_observation}\nPrevious action: {prev_action}"


def build_prompt(
    goal: dict[str, Any],
    state_block: str,
    observation: str,
    available_actions: list[str],
    prev_observation: str | None,
    prev_action: str | None,
) -> str:
    return WEBSHOP_ACT_STATE_TEMPLATE.format(
        task_description=_clean_space(goal.get("instruction_text")),
        subgoal_state_block=state_block,
        one_step_context=build_one_step_context(prev_observation, prev_action),
        current_observation=observation,
        available_actions=format_available_actions(available_actions),
    )


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    products = _load_products(args.product_file, needed_asins=_collect_needed_asins(args.traj_dir))
    examples: list[dict[str, Any]] = []
    stats = {
        "files_seen": 0,
        "files_used": 0,
        "files_skipped_reward": 0,
        "files_skipped_short": 0,
        "transitions_skipped": 0,
    }

    for path in sorted(args.traj_dir.glob("*.jsonl")):
        stats["files_seen"] += 1
        rows = _read_traj(path)
        if len(rows) < 2:
            stats["files_skipped_short"] += 1
            continue
        final_reward = rows[-1].get("reward")
        if args.only_success and final_reward != 1.0:
            stats["files_skipped_reward"] += 1
            continue
        if not args.only_success and args.min_final_reward is not None:
            if not isinstance(final_reward, (int, float)) or final_reward < args.min_final_reward:
                stats["files_skipped_reward"] += 1
                continue

        sid = _session_id(path)
        split = (
            "test"
            if args.test_session_cutoff > 0 and sid is not None and sid < args.test_session_cutoff
            else "train"
        )
        task_text = _clean_space((rows[0].get("goal") or {}).get("instruction_text"))
        tracker = WebshopSubgoalStateTracker(task_text)
        prev_observation = None
        prev_action = None
        file_examples = 0

        for step_idx, row in enumerate(rows[:-1]):
            action = infer_action(row, rows[step_idx + 1])
            if action is None:
                stats["transitions_skipped"] += 1
                continue
            action = normalize_action(action)
            observation = render_observation(row, products)
            available_actions = ensure_expert_action_available(
                build_available_actions(row, products),
                action,
            )
            state_block = tracker.render(observation, available_actions)
            prompt = build_prompt(
                row.get("goal") or {},
                state_block,
                observation,
                available_actions,
                prev_observation,
                prev_action,
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": action},
            ]
            examples.append(
                {
                    "messages": messages,
                    "prompt": prompt,
                    "response": action,
                    "source_file": path.name,
                    "session_id": sid,
                    "step_idx": step_idx,
                    "split": split,
                    "final_reward": final_reward,
                    "state_block": state_block,
                    "current_observation": observation,
                    "available_actions": available_actions,
                    "previous_observation": prev_observation,
                    "previous_action": prev_action,
                    "prompt_style": "act_state",
                }
            )
            next_observation = render_observation(rows[step_idx + 1], products)
            next_available_actions = build_available_actions(rows[step_idx + 1], products)
            tracker.update(action, next_observation, next_available_actions)
            prev_observation = observation
            prev_action = action
            file_examples += 1

        if file_examples:
            stats["files_used"] += 1

    return examples, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def maybe_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    pd.DataFrame(rows).to_parquet(path, index=False)
    return True


def apply_fixed_val_split(rows: list[dict[str, Any]], val_size: int, seed: int) -> None:
    if val_size <= 0:
        return
    if val_size >= len(rows):
        raise ValueError(f"--val-size ({val_size}) must be smaller than the number of examples ({len(rows)}).")

    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_indices = set(indices[:val_size])

    for idx, row in enumerate(rows):
        row["split"] = "test" if idx in val_indices else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--product-file", type=Path, default=DEFAULT_PRODUCT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-session-cutoff",
        type=int,
        default=0,
        help="Put trajectories with session id below this cutoff in test; 0 keeps all examples in train.",
    )
    parser.add_argument("--only-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--val-size",
        type=int,
        default=0,
        help="If >0, make an exact-size deterministic validation split from all examples.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed used with --val-size.",
    )
    parser.add_argument(
        "--min-final-reward",
        type=float,
        default=None,
        help="When --no-only-success is set, keep only trajectories with final reward >= this value.",
    )
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples, stats = build_examples(args)
    apply_fixed_val_split(examples, val_size=args.val_size, seed=args.split_seed)
    train = [row for row in examples if row["split"] == "train"]
    test = [row for row in examples if row["split"] == "test"]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_jsonl(args.output_dir / "all.jsonl", examples)

    parquet_written = False
    if args.write_parquet:
        parquet_written = (
            maybe_write_parquet(args.output_dir / "train.parquet", train)
            and maybe_write_parquet(args.output_dir / "test.parquet", test)
            and maybe_write_parquet(args.output_dir / "all.parquet", examples)
        )

    metadata = {
        "stats": stats,
        "num_examples": len(examples),
        "num_train": len(train),
        "num_test": len(test),
        "only_success": args.only_success,
        "min_final_reward": args.min_final_reward,
        "test_session_cutoff": args.test_session_cutoff,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "parquet_written": parquet_written,
        "format": {
            "messages": "chat messages for data.multiturn.messages_key=messages",
            "prompt_style": "act_state",
            "prompt": "user prompt rendered with agent_system/environments/prompts/webshop.py act_state template",
            "response": "expert next action reconstructed from the next page state",
            "available_actions": "admissible action strings reconstructed from each trajectory page state",
            "state_block": "WebShop subgoal tracker state rendered into the act_state user prompt",
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
