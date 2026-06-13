#!/usr/bin/env python3
"""Evaluate MAVE SFT task checkpoints with HF or vLLM generation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any



TASKS = (
    "single_attribute_qa",
    "evidence_grounded_extraction",
    "multi_attribute_card_completion",
    "product_customer_qa",
    "faceted_search_filtering",
)


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_json_dumps(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", _json_default)
    return json.dumps(value, **kwargs).encode("utf-8", "backslashreplace").decode("utf-8")


def _default_backend() -> str:
    if "INFERENCE_BACKEND" in os.environ:
        return os.environ["INFERENCE_BACKEND"]
    model_path = os.environ.get("MODEL_PATH", "")
    if "Qwen3" in model_path:
        return "hf"
    return "vllm"


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet eval data.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported eval data suffix: {path}")


def _messages_without_answer(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if messages is not None and not isinstance(messages, list):
        try:
            messages = list(messages)
        except TypeError:
            messages = None
    if isinstance(messages, list) and messages:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if isinstance(message, dict) and message.get("role") != "assistant"
        ]
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Each row must contain either messages or prompt.")
    return [{"role": "user", "content": str(prompt)}]


def _gold_response(row: dict[str, Any]) -> str:
    response = row.get("response")
    if response:
        return str(response)
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(messages, list):
        assistants = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
        if assistants:
            return str(assistants[-1].get("content", ""))
    return ""


def _parse_jsonish(text: Any) -> tuple[Any, str | None]:
    text = str(text or "").strip()
    candidates = [text]
    candidates.extend(item.strip() for item in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL))

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj:last_obj + 1])

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
        candidates.append(text[first_arr:last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue
    return {}, "json_parse_error"


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip().casefold()


def _norm_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return _norm_text(value)


def _norm_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _norm_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_norm_json(item) for item in value]
    return _norm_scalar(value)


def _dict_value_accuracy(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
    if not gold:
        return 1.0, pred == gold, {}
    per_key = {}
    correct = 0
    for key, gold_value in gold.items():
        pred_value = pred.get(key) if isinstance(pred, dict) else None
        match = _norm_scalar(pred_value) == _norm_scalar(gold_value)
        correct += int(match)
        per_key[key] = {
            "match": match,
            "pred": pred_value,
            "gold": gold_value,
        }
    return correct / len(gold), correct == len(gold), per_key


def _evidence_exact(pred: Any, gold: Any) -> bool:
    pred_items = pred if isinstance(pred, list) else []
    gold_items = gold if isinstance(gold, list) else []
    if len(pred_items) != len(gold_items):
        return False
    for pred_item, gold_item in zip(pred_items, gold_items):
        if not isinstance(pred_item, dict) or not isinstance(gold_item, dict):
            return False
        for key in ("source_id", "source", "text", "begin", "end"):
            if _norm_scalar(pred_item.get(key)) != _norm_scalar(gold_item.get(key)):
                return False
    return True


def _score(task_type: str, pred: Any, gold: Any) -> dict[str, Any]:
    if not isinstance(pred, dict):
        pred = {}
    if not isinstance(gold, dict):
        gold = {}

    if task_type == "single_attribute_qa":
        answerable_match = bool(pred.get("answerable")) == bool(gold.get("answerable"))
        value_match = _norm_scalar(pred.get("value")) == _norm_scalar(gold.get("value"))
        attribute_match = _norm_text(pred.get("attribute")) == _norm_text(gold.get("attribute"))
        exact_match = answerable_match and value_match and attribute_match
        return {
            "exact_match": exact_match,
            "primary_correct": exact_match,
            "value_match": value_match,
            "answerable_match": answerable_match,
            "attribute_match": attribute_match,
        }

    if task_type == "evidence_grounded_extraction":
        value_match = _norm_scalar(pred.get("value")) == _norm_scalar(gold.get("value"))
        attribute_match = _norm_text(pred.get("attribute")) == _norm_text(gold.get("attribute"))
        evidence_match = _evidence_exact(pred.get("evidence"), gold.get("evidence"))
        exact_match = value_match and attribute_match and evidence_match
        return {
            "exact_match": exact_match,
            "primary_correct": value_match,
            "value_match": value_match,
            "attribute_match": attribute_match,
            "evidence_exact_match": evidence_match,
        }

    if task_type == "multi_attribute_card_completion":
        value_acc, exact_match, per_key = _dict_value_accuracy(pred, gold)
        return {
            "exact_match": exact_match,
            "primary_correct": exact_match,
            "field_value_accuracy": value_acc,
            "per_key": per_key,
        }

    if task_type == "product_customer_qa":
        answer_match = _norm_scalar(pred.get("answer")) == _norm_scalar(gold.get("answer"))
        attribute_match = _norm_text(pred.get("attribute")) == _norm_text(gold.get("attribute"))
        exact_match = answer_match and attribute_match
        return {
            "exact_match": exact_match,
            "primary_correct": answer_match,
            "answer_match": answer_match,
            "attribute_match": attribute_match,
        }

    if task_type == "faceted_search_filtering":
        match_match = bool(pred.get("match")) == bool(gold.get("match"))
        matched_exact = _norm_json(pred.get("matched_attributes", {})) == _norm_json(gold.get("matched_attributes", {}))
        missing_exact = _norm_json(pred.get("missing_or_mismatched_attributes", {})) == _norm_json(gold.get("missing_or_mismatched_attributes", {}))
        exact_match = match_match and matched_exact and missing_exact
        return {
            "exact_match": exact_match,
            "primary_correct": match_match,
            "match_label_match": match_match,
            "matched_attributes_exact": matched_exact,
            "missing_or_mismatched_exact": missing_exact,
        }

    exact_match = _norm_json(pred) == _norm_json(gold)
    return {"exact_match": exact_match, "primary_correct": exact_match}


def _take_eval_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task:
        rows = [row for row in rows if row.get("task_type") in set(args.task)]
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.start_index:
        rows = rows[args.start_index:]
    if args.num_examples is not None:
        rows = rows[: args.num_examples]
    return rows


def _default_eval_file() -> str:
    task = os.environ.get("MAVE_TASK", "single_attribute_qa")
    split = os.environ.get("EVAL_SPLIT", "test")
    return f"data/mave_sft_amazon23/by_task/{task}/{split}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAVE SFT task data with a LoRA checkpoint.")
    parser.add_argument("--backend", default=_default_backend(), choices=["hf", "vllm"])
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"))
    parser.add_argument("--lora-adapter", default=os.environ.get("LORA_ADAPTER", None))
    parser.add_argument("--eval-file", default=os.environ.get("MAVE_EVAL_FILE", _default_eval_file()))
    parser.add_argument("--task", action="append", choices=list(TASKS), default=None)
    parser.add_argument("--split", default=os.environ.get("MAVE_SPLIT", ""))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "checkpoints/verl_agent_mave/validation"))
    parser.add_argument("--num-examples", type=int, default=int(os.environ["VAL_DATA_SIZE"]) if "VAL_DATA_SIZE" in os.environ else None)
    parser.add_argument("--start-index", type=int, default=int(os.environ.get("START_INDEX", 0)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", os.environ.get("PARALLEL_ENVS", 8))))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", 8192)))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.0)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    parser.add_argument("--do-sample", type=_str_to_bool, default=_str_to_bool(os.environ.get("DO_SAMPLE", "false")))
    parser.add_argument("--enable-thinking", type=_str_to_bool, default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--trust-remote-code", type=_str_to_bool, default=_str_to_bool(os.environ.get("TRUST_REMOTE_CODE", "true")))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.7)))
    parser.add_argument("--vllm-max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", int(os.environ.get("MAX_PROMPT_LENGTH", 8192)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=int(os.environ.get("VLLM_MAX_NUM_SEQS", os.environ.get("PARALLEL_ENVS", 8))))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", int(os.environ.get("MAX_PROMPT_LENGTH", 8192)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))))
    parser.add_argument("--vllm-enforce-eager", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_ENFORCE_EAGER", "false")))
    parser.add_argument("--vllm-merged-model-dir", default=os.environ.get("VLLM_MERGED_MODEL_DIR", ""))
    parser.add_argument("--vllm-force-merge-lora", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_FORCE_MERGE_LORA", "false")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer
    from examples.validation.mind2web_llm_eval import _make_generator

    torch.manual_seed(0)

    rows = _take_eval_rows(_load_rows(args.eval_file), args)
    if not rows:
        raise ValueError(f"No eval rows selected from {args.eval_file}.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = _make_generator(args, tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    results = []

    with predictions_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            generated = generator.generate([_messages_without_answer(row) for row in batch])

            for offset, (row, gen) in enumerate(zip(batch, generated)):
                gold, gold_error = _parse_jsonish(_gold_response(row))
                if gold_error:
                    raise ValueError(f"Could not parse gold response at row {start + offset}: {_gold_response(row)!r}")
                pred, parse_error = _parse_jsonish(gen["raw_output"])
                task_type = row.get("task_type") or (args.task[0] if args.task else "unknown")
                score = _score(task_type, pred, gold)
                record = {
                    "index": args.start_index + start + offset,
                    "task_type": task_type,
                    "product_id": row.get("product_id"),
                    "category": row.get("category"),
                    "attribute": row.get("attribute"),
                    "target_attributes": row.get("target_attributes"),
                    "gold": gold,
                    "prediction": pred,
                    "parse_error": parse_error,
                    "raw_output": gen["raw_output"],
                    "input": gen["input"],
                    **score,
                }
                results.append(record)
                f.write(_safe_json_dumps(record, ensure_ascii=False) + "\n")

            done = min(start + len(batch), len(rows))
            primary_acc = sum(item["primary_correct"] for item in results) / len(results)
            exact_acc = sum(item["exact_match"] for item in results) / len(results)
            print(f"[{done}/{len(rows)}] primary_acc={primary_acc:.4f} exact_match={exact_acc:.4f}", flush=True)

    task_groups = defaultdict(list)
    for item in results:
        task_groups[item["task_type"]].append(item)

    by_task = {}
    for task_type, items in sorted(task_groups.items()):
        by_task[task_type] = {
            "num_examples": len(items),
            "primary_acc": sum(item["primary_correct"] for item in items) / len(items),
            "exact_match": sum(item["exact_match"] for item in items) / len(items),
            "json_parse_error_rate": sum(1 for item in items if item["parse_error"]) / len(items),
        }
        metric_keys = sorted(
            key
            for key in set().union(*(item.keys() for item in items))
            if key.endswith("_match") or key.endswith("_exact") or key.endswith("_accuracy")
        )
        for key in metric_keys:
            values = [item[key] for item in items if isinstance(item.get(key), (bool, int, float))]
            if values:
                by_task[task_type][key] = sum(values) / len(values)

    summary = {
        "eval_file": args.eval_file,
        "num_examples": len(results),
        "primary_acc": sum(item["primary_correct"] for item in results) / len(results),
        "exact_match": sum(item["exact_match"] for item in results) / len(results),
        "json_parse_error_rate": sum(1 for item in results if item["parse_error"]) / len(results),
        "by_task": by_task,
        "parse_errors": dict(Counter(item["parse_error"] for item in results if item["parse_error"])),
        "backend": args.backend,
        "model_path": args.model_path,
        "lora_adapter": args.lora_adapter,
    }
    (output_dir / "summary.json").write_text(_safe_json_dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved predictions to {predictions_path}")
    print(_safe_json_dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
