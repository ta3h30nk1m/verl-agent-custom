#!/usr/bin/env python3
"""Lightweight Mind2Web validation for step-level next-action SFT data."""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from verl.utils.lora_ga import apply_loraga_base_delta


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _apply_chat_template(tokenizer, messages, enable_thinking):
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _stop_token_ids(tokenizer):
    token_ids = []
    for token in ("<|im_end|>", "<|endofturn|>", "<|stop|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            token_ids.append(token_id)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    return sorted(set(token_ids))


def _default_backend():
    if "INFERENCE_BACKEND" in os.environ:
        return os.environ["INFERENCE_BACKEND"]
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
    if "Qwen3" in model_path:
        return "hf"
    return "vllm"


def _raise_if_vllm_lacks_native_model(args):
    from vllm.model_executor.models.registry import ModelRegistry
    import vllm

    model_path = getattr(args, "_vllm_model_path", args.model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    architectures = list(getattr(config, "architectures", None) or [])
    if not architectures:
        return

    supported = set(ModelRegistry.get_supported_archs())
    missing = [arch for arch in architectures if arch not in supported]
    if not missing:
        return

    raise RuntimeError(
        "The current vLLM installation does not natively support "
        f"{', '.join(missing)} from {model_path}. "
        f"This env has vLLM {vllm.__version__}; use INFERENCE_BACKEND=hf for this model, "
        "switch to a vLLM-supported model such as Qwen2.5, or upgrade vLLM."
    )


def _has_saved_hf_model(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    return (
        any(path.glob("*.safetensors"))
        or any(path.glob("*.bin"))
        or any(path.glob("*.pt"))
        or (path / "model.safetensors.index.json").is_file()
        or (path / "pytorch_model.bin.index.json").is_file()
    )


def _default_merged_model_dir(args):
    key = f"{args.model_path}|{Path(args.lora_adapter).resolve()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return Path(args.output_dir) / f"merged_lora_for_vllm_{digest}"


_MERGED_LORA_METADATA = ".vllm_merged_lora_metadata.json"


def _write_merged_lora_metadata(args, output_dir):
    metadata = {
        "model_path": args.model_path,
        "lora_adapter": str(Path(args.lora_adapter).resolve()) if args.lora_adapter else "",
    }
    (Path(output_dir) / _MERGED_LORA_METADATA).write_text(json.dumps(metadata, indent=2) + "\n")


def _has_merged_lora_metadata(path):
    return (Path(path) / _MERGED_LORA_METADATA).is_file()


def _iter_hf_hub_cache_dirs():
    seen = set()
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value).expanduser()
            if path not in seen:
                seen.add(path)
                yield path

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        path = Path(hf_home).expanduser() / "hub"
        if path not in seen:
            seen.add(path)
            yield path

    for path in (Path.home() / ".cache" / "huggingface" / "hub", Path("/data1/huggingface_cache_dir/hub")):
        if path not in seen:
            seen.add(path)
            yield path


def _resolve_local_hf_snapshot(model_path):
    path = Path(model_path)
    if path.exists():
        return path
    if "/" not in model_path:
        return None

    repo_dir = f"models--{model_path.replace('/', '--')}"
    for hub_dir in _iter_hf_hub_cache_dirs():
        cache_dir = hub_dir / repo_dir
        if not cache_dir.is_dir():
            continue

        ref_path = cache_dir / "refs" / "main"
        if ref_path.is_file():
            snapshot = cache_dir / "snapshots" / ref_path.read_text().strip()
            if snapshot.is_dir():
                return snapshot

        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(snapshots_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
            for snapshot in snapshots:
                if snapshot.is_dir() and (snapshot / "config.json").exists():
                    return snapshot
    return None


def _copy_processor_sidecars_from_base(args, output_dir):
    output_dir = Path(output_dir)
    snapshot = _resolve_local_hf_snapshot(args.model_path)
    if snapshot is None:
        return False

    names = {"preprocessor_config.json", "processor_config.json"}
    for pattern in ("configuration_*.py", "processing_*.py", "image_processing_*.py", "modeling_*.py"):
        names.update(path.name for path in snapshot.glob(pattern))

    copied = []
    for name in sorted(names):
        src = snapshot / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)

    if copied:
        print(f"[vllm] copied processor sidecar files for merged model: {', '.join(copied)}", flush=True)
    return (output_dir / "preprocessor_config.json").exists()


def _save_processor_if_available(args, output_dir):
    copied_sidecars = _copy_processor_sidecars_from_base(args, output_dir)
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
    except Exception:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model_path,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as remote_exc:
            if copied_sidecars:
                print(f"[vllm] using copied processor sidecar files for merged model: {output_dir}", flush=True)
            else:
                print(f"[vllm] no AutoProcessor saved for merged model: {remote_exc}", flush=True)
            return

    processor.save_pretrained(output_dir)
    print(f"[vllm] saved processor files for merged model: {output_dir}", flush=True)


def _merge_lora_for_vllm(args, tokenizer):
    if not args.lora_adapter:
        return args.model_path

    merged_dir = Path(args.vllm_merged_model_dir) if args.vllm_merged_model_dir else _default_merged_model_dir(args)
    if _has_saved_hf_model(merged_dir) and not args.vllm_force_merge_lora:
        _save_processor_if_available(args, merged_dir)
        print(f"[vllm] using existing merged LoRA model: {merged_dir}", flush=True)
        return str(merged_dir)

    from peft import PeftModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    print(f"[vllm] merging LoRA adapter {args.lora_adapter} into base model {args.model_path}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    apply_loraga_base_delta(peft_model, args.lora_adapter, strict=False)
    merged_model = peft_model.merge_and_unload()

    tmp_dir = merged_dir.with_name(f".{merged_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    _save_processor_if_available(args, tmp_dir)
    _write_merged_lora_metadata(args, tmp_dir)

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    tmp_dir.rename(merged_dir)
    print(f"[vllm] saved merged model for vLLM: {merged_dir}", flush=True)
    return str(merged_dir)


def _cleanup_vllm_merged_model(args):
    if not getattr(args, "vllm_cleanup_merged_model", True):
        return
    if getattr(args, "backend", None) != "vllm" or not getattr(args, "lora_adapter", None):
        return

    model_path = getattr(args, "_vllm_model_path", "")
    if not model_path or model_path == args.model_path:
        return

    merged_dir = Path(model_path)
    expected_dir = Path(args.vllm_merged_model_dir) if args.vllm_merged_model_dir else _default_merged_model_dir(args)
    if merged_dir.resolve(strict=False) != expected_dir.resolve(strict=False):
        print(f"[vllm] skip cleanup for unexpected merged model path: {merged_dir}", flush=True)
        return
    if not merged_dir.exists():
        return

    is_default_dir = not args.vllm_merged_model_dir and merged_dir.name.startswith("merged_lora_for_vllm_")
    if not is_default_dir and not _has_merged_lora_metadata(merged_dir):
        print(f"[vllm] keeping custom merged model without cleanup metadata: {merged_dir}", flush=True)
        return

    shutil.rmtree(merged_dir)
    print(f"[vllm] removed merged model after eval: {merged_dir}", flush=True)


class HFGenerator:
    def __init__(self, args, tokenizer):
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        if args.lora_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, args.lora_adapter)
            apply_loraga_base_delta(self.model, args.lora_adapter, strict=False)
        self.model = self.model.to(args.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.args = args

    @torch.inference_mode()
    def generate(self, message_batches):
        formatted_prompts = [
            _apply_chat_template(self.tokenizer, messages, self.args.enable_thinking)
            for messages in message_batches
        ]
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
        ).to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": _stop_token_ids(self.tokenizer) or self.tokenizer.eos_token_id,
        }
        if self.args.do_sample:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        return [
            {
                "input": formatted,
                "raw_output": self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip(),
            }
            for formatted, output in zip(formatted_prompts, output_ids)
        ]


class VLLMGenerator:
    def __init__(self, args, tokenizer):
        from vllm import LLM, SamplingParams

        vllm_model_path = _merge_lora_for_vllm(args, tokenizer)
        args._vllm_model_path = vllm_model_path
        _raise_if_vllm_lacks_native_model(args)
        self.tokenizer = tokenizer
        self.args = args
        self.sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=args.top_p,
            truncate_prompt_tokens=args.max_prompt_length,
            stop_token_ids=_stop_token_ids(tokenizer),
        )
        self.llm = LLM(
            model=vllm_model_path,
            tokenizer=vllm_model_path,
            dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            enforce_eager=args.vllm_enforce_eager,
        )

    def close(self):
        if hasattr(self, "llm"):
            del self.llm

    def generate(self, message_batches):
        formatted_prompts = [
            _apply_chat_template(self.tokenizer, messages, self.args.enable_thinking)
            for messages in message_batches
        ]
        max_prompt_length = min(
            self.args.max_prompt_length,
            self.args.vllm_max_model_len - self.args.max_new_tokens,
        )
        if max_prompt_length < 1:
            raise ValueError(
                f"vLLM max_model_len={self.args.vllm_max_model_len} is too small for "
                f"max_new_tokens={self.args.max_new_tokens}."
            )
        tokenized_prompts = self.tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        outputs = self.llm.generate(
            [{"prompt_token_ids": input_ids} for input_ids in tokenized_prompts],
            self.sampling_params,
            use_tqdm=False,
        )
        return [
            {"input": formatted, "raw_output": output.outputs[0].text.strip()}
            for formatted, output in zip(formatted_prompts, outputs)
        ]


def _make_generator(args, tokenizer):
    if args.backend == "hf":
        return HFGenerator(args, tokenizer)
    if args.backend == "vllm":
        return VLLMGenerator(args, tokenizer)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _load_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet eval data.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported eval data suffix: {path}")


def _safe_json_dumps(value, **kwargs):
    return json.dumps(value, **kwargs).encode("utf-8", "backslashreplace").decode("utf-8")


def _messages_without_answer(row):
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(messages, tuple):
        messages = list(messages)
    if isinstance(messages, list) and messages:
        return [message for message in messages if message.get("role") != "assistant"]

    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Each row must contain either messages or prompt.")
    return [{"role": "user", "content": prompt}]


def _parse_jsonish_action(text):
    text = str(text or "").strip()
    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(item.strip() for item in fenced)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value, None
        except json.JSONDecodeError:
            continue
    return {}, "json_parse_error"


def _normalize_op(value):
    return str(value or "").strip().upper()


def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _target_id_from_text(text):
    match = re.search(r"\bC\d{3,}\b", str(text or ""))
    return match.group(0) if match else ""


def _gold_action(row):
    response = row.get("response")
    if not response:
        messages = row.get("messages")
        if isinstance(messages, str):
            messages = json.loads(messages)
        if isinstance(messages, list):
            assistants = [message for message in messages if message.get("role") == "assistant"]
            response = assistants[-1]["content"] if assistants else ""
    action, error = _parse_jsonish_action(response)
    if error:
        raise ValueError(f"Could not parse gold response as JSON: {response!r}")
    return action


def _score_prediction(pred, gold):
    pred_target_id = _normalize_text(pred.get("target_id")) or _target_id_from_text(pred.get("target"))
    gold_target_id = _normalize_text(gold.get("target_id"))
    pred_op = _normalize_op(pred.get("op"))
    gold_op = _normalize_op(gold.get("op"))
    pred_value = _normalize_text(pred.get("value"))
    gold_value = _normalize_text(gold.get("value"))

    target_id_answerable = bool(gold_target_id)
    target_id_match = target_id_answerable and pred_target_id == gold_target_id
    op_match = pred_op == gold_op
    value_match = pred_value == gold_value
    exact_match = target_id_match and op_match and value_match
    return {
        "pred_target_id": pred_target_id,
        "gold_target_id": gold_target_id,
        "pred_op": pred_op,
        "gold_op": gold_op,
        "pred_value": pred_value,
        "gold_value": gold_value,
        "target_id_answerable": target_id_answerable,
        "target_id_match": target_id_match,
        "op_match": op_match,
        "value_match": value_match,
        "exact_match": exact_match,
    }


def _take_eval_rows(rows, args):
    if args.split:
        rows = [row for row in rows if row.get("mind2web_split") == args.split or row.get("split") == args.split]
    if args.start_index:
        rows = rows[args.start_index :]
    if args.num_examples is not None:
        rows = rows[: args.num_examples]
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Mind2Web step-level multi-choice/action validation.")
    parser.add_argument("--backend", default=_default_backend(), choices=["hf", "vllm"])
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"))
    parser.add_argument("--lora-adapter", default=os.environ.get("LORA_ADAPTER", None))
    parser.add_argument("--eval-file", default=os.environ.get("MIND2WEB_EVAL_FILE", "data/mind2web_sft/test.jsonl"))
    parser.add_argument("--split", default=os.environ.get("MIND2WEB_SPLIT", ""))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "checkpoints/verl_agent_mind2web/validation"))
    parser.add_argument("--num-examples", type=int, default=int(os.environ["VAL_DATA_SIZE"]) if "VAL_DATA_SIZE" in os.environ else None)
    parser.add_argument("--start-index", type=int, default=int(os.environ.get("START_INDEX", 0)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", os.environ.get("PARALLEL_ENVS", 8))))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", 16384)))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.0)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    parser.add_argument("--do-sample", type=_str_to_bool, default=_str_to_bool(os.environ.get("DO_SAMPLE", "false")))
    parser.add_argument("--enable-thinking", type=_str_to_bool, default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--trust-remote-code", type=_str_to_bool, default=_str_to_bool(os.environ.get("TRUST_REMOTE_CODE", "false")))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.7)))
    parser.add_argument("--vllm-max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", int(os.environ.get("MAX_PROMPT_LENGTH", 16384)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=int(os.environ.get("VLLM_MAX_NUM_SEQS", os.environ.get("PARALLEL_ENVS", 8))))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", int(os.environ.get("MAX_PROMPT_LENGTH", 16384)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))))
    parser.add_argument("--vllm-enforce-eager", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_ENFORCE_EAGER", "false")))
    parser.add_argument("--vllm-merged-model-dir", default=os.environ.get("VLLM_MERGED_MODEL_DIR", ""))
    parser.add_argument("--vllm-force-merge-lora", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_FORCE_MERGE_LORA", "false")))
    parser.add_argument("--vllm-cleanup-merged-model", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_CLEANUP_MERGED_MODEL", "true")))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    generator = None

    try:
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

        with predictions_path.open("w") as f:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                message_batches = [_messages_without_answer(row) for row in batch]
                generated = generator.generate(message_batches)

                for offset, (row, gen) in enumerate(zip(batch, generated)):
                    gold = _gold_action(row)
                    pred, parse_error = _parse_jsonish_action(gen["raw_output"])
                    score = _score_prediction(pred, gold)
                    record = {
                        "index": args.start_index + start + offset,
                        "annotation_id": row.get("annotation_id"),
                        "action_uid": row.get("action_uid"),
                        "step_idx": row.get("step_idx"),
                        "mind2web_split": row.get("mind2web_split"),
                        "website": row.get("website"),
                        "domain": row.get("domain"),
                        "subdomain": row.get("subdomain"),
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
                answerable = [item for item in results if item["target_id_answerable"]]
                exact = sum(item["exact_match"] for item in answerable) / len(answerable) if answerable else 0.0
                target = sum(item["target_id_match"] for item in answerable) / len(answerable) if answerable else 0.0
                print(f"[{done}/{len(rows)}] exact_action_acc={exact:.4f} target_id_acc={target:.4f}", flush=True)

        denominator = len(results)
        target_id_results = [item for item in results if item["target_id_answerable"]]
        target_id_denominator = len(target_id_results)
        summary = {
            "eval_file": args.eval_file,
            "split": args.split,
            "num_examples": denominator,
            "num_target_id_examples": target_id_denominator,
            "exact_action_acc": (
                sum(item["exact_match"] for item in target_id_results) / target_id_denominator
                if target_id_denominator
                else 0.0
            ),
            "target_id_acc": (
                sum(item["target_id_match"] for item in target_id_results) / target_id_denominator
                if target_id_denominator
                else 0.0
            ),
            "overall_exact_action_acc": sum(item["exact_match"] for item in results) / denominator,
            "op_acc": sum(item["op_match"] for item in results) / denominator,
            "value_acc": sum(item["value_match"] for item in results) / denominator,
            "json_parse_error_rate": sum(1 for item in results if item["parse_error"]) / denominator,
            "backend": args.backend,
            "model_path": args.model_path,
            "lora_adapter": args.lora_adapter,
        }
        (output_dir / "summary.json").write_text(_safe_json_dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {predictions_path}")
        print(_safe_json_dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if generator is not None and hasattr(generator, "close"):
            generator.close()
        _cleanup_vllm_merged_model(args)


if __name__ == "__main__":
    main()
