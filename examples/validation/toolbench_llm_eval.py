#!/usr/bin/env python3
"""
Lightweight ToolBench interaction evaluator.

This follows the WebShop lite evaluator style: it runs an LLM, parses each
assistant action, executes the requested ToolBench API function, feeds the
observation back into the next prompt, and writes per-episode JSONL artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLBENCH_ROOT = REPO_ROOT / "data/toolbench"
DEFAULT_EXTRACT_DIR = DEFAULT_TOOLBENCH_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/verl_agent_toolbench/toolbench_lite_validation"
TEST_SETS = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)

SYSTEM_PROMPT = """You are AutoGPT, and you can use the provided APIs to answer the user's request.
At every turn, output exactly this format and no extra text:
Thought: <brief reasoning>
Action: <one available API name or Finish>
Action Input: <valid JSON object>

You must call Finish at the end. Finish arguments must be:
{"return_type": "give_answer", "final_answer": "..."}
or:
{"return_type": "give_up_and_restart"}
"""


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return repr(value)


def standardize_name(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def ensure_toolbench_data(root: Path, extract_dir: Path) -> None:
    required = [
        extract_dir / "test_instruction",
        extract_dir / "toolenv/tools",
    ]
    if all(path.exists() for path in required):
        return

    zip_path = root / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected extracted ToolBench data under {extract_dir} or zip at {zip_path}."
        )
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("__MACOSX/") or member.endswith("/.DS_Store"):
                continue
            if member.startswith("data/test_instruction/") or member.startswith("data/toolenv/"):
                zf.extract(member, root)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_queries(data_dir: Path, test_set: str, limit: int | None) -> list[dict[str, Any]]:
    path = data_dir / "test_instruction" / f"{test_set}.json"
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list.")
    if limit is not None:
        rows = rows[:limit]
    return rows


def api_param_schema(api: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    required = []
    for param in api.get("required_parameters") or []:
        name = param.get("name")
        if not name:
            continue
        required.append(name)
        properties[name] = {
            "type": str(param.get("type", "STRING")).lower(),
            "description": param.get("description", ""),
            "default": param.get("default", ""),
        }
    for param in api.get("optional_parameters") or []:
        name = param.get("name")
        if not name:
            continue
        properties[name] = {
            "type": str(param.get("type", "STRING")).lower(),
            "description": param.get("description", ""),
            "default": param.get("default", ""),
        }
    return {"type": "object", "properties": properties, "required": required}


def render_api_list(apis: list[dict[str, Any]], max_description_chars: int) -> str:
    lines = []
    for idx, api in enumerate(apis, start=1):
        action_name = action_name_for_api(api)
        desc = api.get("api_description") or api.get("description") or ""
        desc = re.sub(r"\s+", " ", str(desc)).strip()
        if max_description_chars > 0:
            desc = desc[:max_description_chars]
        lines.append(
            "\n".join(
                [
                    f"{idx}. {action_name}",
                    f"   tool: {api.get('tool_name')} / category: {api.get('category_name')}",
                    f"   description: {desc}",
                    "   parameters: " + json.dumps(api_param_schema(api), ensure_ascii=True),
                ]
            )
        )
    lines.append(
        'Finish: parameters {"return_type": "give_answer|give_up_and_restart", "final_answer": "answer text"}'
    )
    return "\n".join(lines)


def build_prompt(query: dict[str, Any], history: list[dict[str, str]], max_description_chars: int) -> str:
    api_list = query.get("api_list") or []
    history_text = "\n".join(f"{item['role']}:\n{item['content']}" for item in history) or "none"
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            "Task:\n" + str(query.get("query", "")),
            "Available APIs:\n" + render_api_list(api_list, max_description_chars),
            "Interaction history:\n" + history_text,
        ]
    )


def action_name_for_api(api: dict[str, Any]) -> str:
    return f"{standardize_name(api.get('api_name'))}_for_{standardize_name(api.get('tool_name'))}"


def parse_action(text: str) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    match = re.search(r"Action\s*:\s*([A-Za-z0-9_]+|Finish)", text)
    action = match.group(1).strip() if match else None
    input_match = re.search(r"Action Input\s*:\s*(.*)", text, flags=re.S)
    raw_input = input_match.group(1).strip() if input_match else "{}"
    raw_input = raw_input.strip()
    if raw_input.startswith("```"):
        raw_input = re.sub(r"^```(?:json)?", "", raw_input).strip()
        raw_input = re.sub(r"```$", "", raw_input).strip()
    try:
        parsed_input = json.loads(raw_input)
        if not isinstance(parsed_input, dict):
            parsed_input = {"value": parsed_input}
    except json.JSONDecodeError:
        parsed_input = {}
    return action, parsed_input, {"raw_action_input": raw_input}


@dataclass
class RegisteredApi:
    action_name: str
    doc: dict[str, Any]
    module_path: Path | None
    function_name: str | None


class ToolBenchExecutor:
    def __init__(self, tool_root: Path, toolbench_key: str | None, execution_mode: str):
        self.tool_root = tool_root
        self.toolbench_key = toolbench_key
        self.execution_mode = execution_mode
        self._module_cache: dict[Path, Any] = {}

    def registry_for_query(self, query: dict[str, Any]) -> dict[str, RegisteredApi]:
        registry = {}
        for api in query.get("api_list") or []:
            action_name = action_name_for_api(api)
            module_path, function_name = self._resolve_api(api)
            registry[action_name] = RegisteredApi(action_name, api, module_path, function_name)
        return registry

    def _resolve_api(self, api: dict[str, Any]) -> tuple[Path | None, str | None]:
        category = str(api.get("category_name") or "")
        tool_std = standardize_name(api.get("tool_name"))
        api_std = standardize_name(api.get("api_name"))
        candidates = [
            self.tool_root / category / tool_std / "api.py",
            self.tool_root / standardize_name(category) / tool_std / "api.py",
        ]
        module_path = next((path for path in candidates if path.exists()), None)
        if module_path is None:
            matches = list(self.tool_root.glob(f"*/{tool_std}/api.py"))
            module_path = matches[0] if matches else None
        if module_path is None:
            return None, None
        module = self._load_module(module_path)
        functions = {
            standardize_name(name): name
            for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
        }
        return module_path, functions.get(api_std)

    def _load_module(self, path: Path) -> Any:
        if path in self._module_cache:
            return self._module_cache[path]
        module_name = "toolbench_api_" + str(abs(hash(path)))
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load API module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._module_cache[path] = module
        return module

    def execute(self, registry: dict[str, RegisteredApi], action: str | None, arguments: dict[str, Any]) -> tuple[str, bool, bool]:
        if not action:
            return json.dumps({"error": "Could not parse Action line."}), False, False
        if action == "Finish":
            return json.dumps({"response": arguments}, ensure_ascii=True), True, True
        api = registry.get(action)
        if api is None:
            return json.dumps({"error": f"Unknown API action {action!r}."}, ensure_ascii=True), False, False
        if self.execution_mode == "mock":
            return json.dumps({"response": f"mock response for {action}", "arguments": arguments}, ensure_ascii=True), True, False
        if api.module_path is None or api.function_name is None:
            return json.dumps({"error": f"Could not resolve local API code for {action}."}, ensure_ascii=True), False, False

        module = self._load_module(api.module_path)
        fn = getattr(module, api.function_name)
        kwargs = dict(arguments)
        signature = inspect.signature(fn)
        if self.toolbench_key and "toolbench_rapidapi_key" in signature.parameters:
            kwargs["toolbench_rapidapi_key"] = self.toolbench_key
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        try:
            observation = fn(**kwargs)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True), False, False
        return json.dumps({"error": "", "response": _jsonable(observation)}, ensure_ascii=True), True, False


def _apply_chat_template(tokenizer, prompt: str, enable_thinking: bool | None) -> str:
    chat = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(chat, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(chat, **kwargs)


class HFGenerator:
    def __init__(self, args: argparse.Namespace, tokenizer):
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(args.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.args = args

    @torch.inference_mode()
    def generate_action(self, prompt: str) -> tuple[str, str]:
        formatted = _apply_chat_template(self.tokenizer, prompt, self.args.enable_thinking)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_prompt_length,
        ).to(self.model.device)
        generate_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.do_sample:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
        output_ids = self.model.generate(**inputs, **generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return formatted, self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class VLLMGenerator:
    def __init__(self, args: argparse.Namespace, tokenizer):
        from vllm import LLM, SamplingParams

        self.tokenizer = tokenizer
        self.args = args
        self.sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=args.top_p,
        )
        self.llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            enforce_eager=args.vllm_enforce_eager,
        )

    def generate_action(self, prompt: str) -> tuple[str, str]:
        formatted = _apply_chat_template(self.tokenizer, prompt, self.args.enable_thinking)
        outputs = self.llm.generate([formatted], self.sampling_params, use_tqdm=False)
        return formatted, outputs[0].outputs[0].text.strip()


def make_generator(args: argparse.Namespace, tokenizer):
    if args.backend == "hf":
        return HFGenerator(args, tokenizer)
    if args.backend == "vllm":
        return VLLMGenerator(args, tokenizer)
    raise ValueError(f"Unsupported backend: {args.backend}")


def run_episode(query: dict[str, Any], generator, executor: ToolBenchExecutor, args: argparse.Namespace) -> dict[str, Any]:
    registry = executor.registry_for_query(query)
    history: list[dict[str, str]] = []
    steps = []
    finished = False
    valid_tool_calls = 0
    execution_errors = 0
    final_answer = ""

    for turn_index in range(args.max_steps):
        prompt = build_prompt(query, history, args.max_api_description_chars)
        model_input, output = generator.generate_action(prompt)
        action, arguments, parse_info = parse_action(output)
        observation, ok, done = executor.execute(registry, action, arguments)
        valid_tool_calls += int(ok and not done)
        execution_errors += int(not ok)

        history.append({"role": "assistant", "content": output})
        history.append({"role": "function", "content": observation})

        if action == "Finish":
            finished = True
            final_answer = str(arguments.get("final_answer", ""))

        steps.append(
            {
                "turn_index": turn_index,
                "prompt": prompt,
                "input": model_input,
                "output": output,
                "action": action,
                "action_input": arguments,
                "parse_info": parse_info,
                "observation": observation,
                "execution_ok": ok,
                "done": done,
            }
        )
        if done:
            break

    return {
        "query_id": query.get("query_id"),
        "query": query.get("query"),
        "finished": finished,
        "success": bool(finished and final_answer),
        "final_answer": final_answer,
        "valid_tool_calls": valid_tool_calls,
        "execution_errors": execution_errors,
        "episode_length": len(steps),
        "available_actions": list(registry.keys()) + ["Finish"],
        "steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ToolBench LLM interaction eval with real or mock API execution.")
    parser.add_argument("--backend", default=os.environ.get("INFERENCE_BACKEND", "vllm"), choices=["hf", "vllm"])
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"))
    parser.add_argument("--toolbench-root", type=Path, default=DEFAULT_TOOLBENCH_ROOT)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-set", default=os.environ.get("TOOLBENCH_TEST_SET", "G1_instruction"), choices=TEST_SETS)
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("VAL_DATA_SIZE", 4)))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", 8)))
    parser.add_argument("--execution-mode", default=os.environ.get("TOOLBENCH_EXECUTION_MODE", "real"), choices=["real", "mock"])
    parser.add_argument("--toolbench-key", default=os.environ.get("TOOLBENCH_KEY") or os.environ.get("RAPIDAPI_KEY"))
    parser.add_argument("--max-api-description-chars", type=int, default=int(os.environ.get("MAX_API_DESCRIPTION_CHARS", 1200)))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", 8192)))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.0)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    parser.add_argument("--do-sample", type=_str_to_bool, default=_str_to_bool(os.environ.get("DO_SAMPLE", "false")))
    parser.add_argument("--enable-thinking", type=_str_to_bool, default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--trust-remote-code", type=_str_to_bool, default=_str_to_bool(os.environ.get("TRUST_REMOTE_CODE", "false")))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.7)))
    parser.add_argument("--vllm-max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", int(os.environ.get("MAX_PROMPT_LENGTH", 8192)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=int(os.environ.get("VLLM_MAX_NUM_SEQS", 8)))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", int(os.environ.get("MAX_PROMPT_LENGTH", 8192)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 512)))))
    parser.add_argument("--vllm-enforce-eager", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_ENFORCE_EAGER", "false")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    ensure_toolbench_data(args.toolbench_root, args.extract_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generator = make_generator(args, tokenizer)
    executor = ToolBenchExecutor(
        tool_root=args.extract_dir / "toolenv/tools",
        toolbench_key=args.toolbench_key,
        execution_mode=args.execution_mode,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "episodes.jsonl"

    queries = load_queries(args.extract_dir, args.test_set, args.num_episodes)
    results = []
    with output_path.open("w") as f:
        for query in queries:
            result = run_episode(query, generator, executor, args)
            result["test_set"] = args.test_set
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=True) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "query_id": result["query_id"],
                        "finished": result["finished"],
                        "success": result["success"],
                        "valid_tool_calls": result["valid_tool_calls"],
                        "execution_errors": result["execution_errors"],
                    },
                    ensure_ascii=True,
                )
            )

    summary = {
        "test_set": args.test_set,
        "num_episodes": len(results),
        "finished_rate": sum(x["finished"] for x in results) / max(1, len(results)),
        "success_rate": sum(x["success"] for x in results) / max(1, len(results)),
        "avg_valid_tool_calls": sum(x["valid_tool_calls"] for x in results) / max(1, len(results)),
        "avg_execution_errors": sum(x["execution_errors"] for x in results) / max(1, len(results)),
        "output_path": str(output_path),
        "execution_mode": args.execution_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
