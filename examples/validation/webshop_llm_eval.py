import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
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


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _apply_chat_template(tokenizer, prompt, enable_thinking):
    chat = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(chat, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(chat, **kwargs)


def _first_action_match(text, prompt_style):
    prompt_style = str(prompt_style).lower()
    base_prompt_style = prompt_style[:-6] if prompt_style.endswith("_state") else prompt_style
    actions = ["search", "click"]
    if base_prompt_style == "react":
        actions.append("think")
    pattern = re.compile(rf"({'|'.join(actions)})\[[^\]]+\]", re.IGNORECASE | re.DOTALL)
    return pattern.search(text)


def _extract_first_action(text, prompt_style):
    match = _first_action_match(text, prompt_style)
    return match.group(0).strip() if match else text.strip()


class FirstActionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length, prompt_style):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.prompt_style = prompt_style

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_length :]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _first_action_match(text, self.prompt_style) is not None


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
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
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
    patterns = ("*.safetensors", "*.bin", "*.pt")
    return any(path.glob(pattern) for pattern in patterns) or (path / "model.safetensors.index.json").is_file() or (path / "pytorch_model.bin.index.json").is_file()


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
    except Exception as exc:
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
    print(
        f"[vllm] merging LoRA adapter {args.lora_adapter} into base model {args.model_path}",
        flush=True,
    )
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
    def generate_actions(self, prompts):
        formatted_prompts = [
            _apply_chat_template(self.tokenizer, prompt, self.args.enable_thinking)
            for prompt in prompts
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
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.do_sample:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p

        output_ids = self.model.generate(
            **inputs,
            # stopping_criteria=StoppingCriteriaList(
            #     [
            #         FirstActionStoppingCriteria(
            #             tokenizer=self.tokenizer,
            #             prompt_length=inputs["input_ids"].shape[-1],
            #             prompt_style=self.args.prompt_style,
            #         )
            #     ]
            # ),
            **generate_kwargs,
        )
        results = []
        prompt_length = inputs["input_ids"].shape[-1]
        for formatted, output in zip(formatted_prompts, output_ids):
            new_tokens = output[prompt_length:]
            raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append((formatted, _extract_first_action(raw_output, self.args.prompt_style), raw_output))
        return results

    def generate_action(self, prompt):
        return self.generate_actions([prompt])[0]


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

    def generate_actions(self, prompts):
        formatted_prompts = [
            _apply_chat_template(self.tokenizer, prompt, self.args.enable_thinking)
            for prompt in prompts
        ]
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=False)
        results = []
        for formatted, output in zip(formatted_prompts, outputs):
            raw_output = output.outputs[0].text.strip()
            results.append((formatted, _extract_first_action(raw_output, self.args.prompt_style), raw_output))
        return results

    def generate_action(self, prompt):
        return self.generate_actions([prompt])[0]


def _make_generator(args, tokenizer):
    if args.backend == "hf":
        return HFGenerator(args, tokenizer)
    if args.backend == "vllm":
        return VLLMGenerator(args, tokenizer)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _make_config(args):
    return OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "history_length": args.history_length,
                "max_steps": args.max_steps,
                "webshop": {
                    "use_small": args.use_small,
                    "human_goals": args.human_goals,
                    "prompt_style": args.prompt_style,
                },
            }
        }
    )


def _make_env(args, config):
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "agent_system/environments/env_package/webshop/webshop/data"
    if args.use_small:
        file_path = data_dir / "items_shuffle_1000.json"
        attr_path = data_dir / "items_ins_v2_1000.json"
    else:
        file_path = data_dir / "items_shuffle.json"
        attr_path = data_dir / "items_ins_v2.json"

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": args.human_goals,
        "file_path": str(file_path),
        "attr_path": str(attr_path),
        "val_goal_start": args.goal_start,
        "val_goal_end": args.goal_end,
    }
    raw_env = build_webshop_envs(
        seed=args.seed + 1000,
        env_num=args.parallel_envs,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker={"num_cpus": args.num_cpus_per_env_worker, "num_gpus": 0},
    )
    return WebshopEnvironmentManager(raw_env, webshop_projection, config)


def run_episode(env, generator, episode_index, args):
    obs, _ = env.reset(kwargs=None)
    steps = []
    episode_reward = 0.0
    final_info = {}

    for turn_index in range(args.max_steps):
        prompt = obs["text"][0]
        model_input, output, raw_output = generator.generate_action(prompt)
        parsed_actions, valids = webshop_projection([output], prompt_style=args.prompt_style)

        next_obs, rewards, dones, infos = env.step([output])
        reward = float(_jsonable(rewards[0]))
        done = bool(_jsonable(dones[0]))
        info = _jsonable(infos[0])
        episode_reward += reward
        final_info = info

        steps.append(
            {
                "turn_index": turn_index,
                "input": model_input,
                "prompt": prompt,
                "output": output,
                "raw_output": raw_output,
                "parsed_action": parsed_actions[0],
                "is_action_valid": bool(valids[0]),
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        obs = next_obs
        if done:
            break

    return {
        "episode_index": episode_index,
        "goal_index": args.goal_start + episode_index if args.goal_start is not None else episode_index,
        "score": float(final_info.get("task_score", 0.0)),
        "won": bool(final_info.get("won", False)),
        "episode_reward": episode_reward,
        "episode_length": len(steps),
        "input": steps[0]["input"] if steps else "",
        "output": "\n".join(step["output"] for step in steps),
        "steps": steps,
    }


def run_episode_batch(env, generator, episode_start_index, batch_size, args):
    env.set_active_env_num(batch_size)
    obs, _ = env.reset(kwargs=None)
    active = [True] * batch_size
    states = [
        {
            "steps": [],
            "episode_reward": 0.0,
            "final_info": {},
        }
        for _ in range(batch_size)
    ]

    for turn_index in range(args.max_steps):
        active_indices = [idx for idx, is_active in enumerate(active) if is_active]
        if not active_indices:
            break

        prompts = [obs["text"][idx] for idx in active_indices]
        generated = generator.generate_actions(prompts)
        outputs = ["click[__done__]"] * batch_size
        generated_by_env = {}
        for env_idx, result in zip(active_indices, generated):
            model_input, output, raw_output = result
            outputs[env_idx] = output
            generated_by_env[env_idx] = {
                "input": model_input,
                "prompt": obs["text"][env_idx],
                "output": output,
                "raw_output": raw_output,
            }

        parsed_actions, valids = webshop_projection(outputs.copy(), prompt_style=args.prompt_style)
        next_obs, rewards, dones, infos = env.step(outputs)

        for env_idx in active_indices:
            reward = float(_jsonable(rewards[env_idx]))
            done = bool(_jsonable(dones[env_idx]))
            info = _jsonable(infos[env_idx])
            states[env_idx]["episode_reward"] += reward
            states[env_idx]["final_info"] = info

            step = generated_by_env[env_idx]
            step.update(
                {
                    "turn_index": turn_index,
                    "parsed_action": parsed_actions[env_idx],
                    "is_action_valid": bool(valids[env_idx]),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
            states[env_idx]["steps"].append(step)
            if done:
                active[env_idx] = False

        obs = next_obs

    results = []
    for env_idx, state in enumerate(states):
        episode_index = episode_start_index + env_idx
        steps = state["steps"]
        final_info = state["final_info"]
        results.append(
            {
                "episode_index": episode_index,
                "goal_index": args.goal_start + episode_index if args.goal_start is not None else episode_index,
                "score": float(final_info.get("task_score", 0.0)),
                "won": bool(final_info.get("won", False)),
                "episode_reward": state["episode_reward"],
                "episode_length": len(steps),
                "input": steps[0]["input"] if steps else "",
                "output": "\n".join(step["output"] for step in steps),
                "steps": steps,
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight WebShop + LLM validation without RayPPOTrainer.")
    parser.add_argument("--backend", default=_default_backend(), choices=["hf", "vllm"])
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"))
    parser.add_argument("--lora-adapter", default=os.environ.get("LORA_ADAPTER", None))
    parser.add_argument("--output-dir", default="checkpoints/verl_agent_webshop/webshop_lite_validation")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("VAL_DATA_SIZE", 4)))
    parser.add_argument("--goal-start", type=int, default=int(os.environ.get("WEBSHOP_GOAL_START", 0)))
    parser.add_argument("--goal-end", type=int, default=int(os.environ.get("WEBSHOP_GOAL_END", 500)))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", 15)))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("HISTORY_LENGTH", 2)))
    parser.add_argument("--prompt-style", default=os.environ.get("WEBSHOP_PROMPT_STYLE", "direct"), choices=["tagged", "direct", "react", "act","act_state"])
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", 4096)))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.0)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    parser.add_argument("--do-sample", type=_str_to_bool, default=_str_to_bool(os.environ.get("DO_SAMPLE", "false")))
    parser.add_argument("--enable-thinking", type=_str_to_bool, default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--trust-remote-code", type=_str_to_bool, default=_str_to_bool(os.environ.get("TRUST_REMOTE_CODE", "false")))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.7)))
    parser.add_argument("--vllm-max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", int(os.environ.get("MAX_PROMPT_LENGTH", 4096)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=int(os.environ.get("VLLM_MAX_NUM_SEQS", 8)))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", int(os.environ.get("MAX_PROMPT_LENGTH", 4096)) + int(os.environ.get("MAX_RESPONSE_LENGTH", 256)))))
    parser.add_argument("--vllm-enforce-eager", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_ENFORCE_EAGER", "false")))
    parser.add_argument("--vllm-merged-model-dir", default=os.environ.get("VLLM_MERGED_MODEL_DIR", ""))
    parser.add_argument("--vllm-force-merge-lora", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_FORCE_MERGE_LORA", "false")))
    parser.add_argument("--vllm-cleanup-merged-model", type=_str_to_bool, default=_str_to_bool(os.environ.get("VLLM_CLEANUP_MERGED_MODEL", "true")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    parser.add_argument("--ray-num-cpus", type=int, default=int(os.environ.get("RAY_NUM_CPUS", 2)))
    parser.add_argument("--num-cpus-per-env-worker", type=float, default=float(os.environ.get("NUM_CPUS_PER_ENV_WORKER", 0.1)))
    parser.add_argument("--parallel-envs", type=int, default=int(os.environ.get("PARALLEL_ENVS", 1)))
    parser.add_argument("--use-small", type=_str_to_bool, default=_str_to_bool(os.environ.get("WEBSHOP_USE_SMALL", "true")))
    parser.add_argument("--human-goals", type=_str_to_bool, default=_str_to_bool(os.environ.get("WEBSHOP_HUMAN_GOALS", "false")))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.parallel_envs < 1:
        raise ValueError(f"--parallel-envs must be >= 1, got {args.parallel_envs}.")
    torch.manual_seed(args.seed)
    generator = None
    env = None

    try:
        if not ray.is_initialized():
            ray.init(num_cpus=args.ray_num_cpus, num_gpus=0, include_dashboard=False)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        generator = _make_generator(args, tokenizer)

        config = _make_config(args)
        env = _make_env(args, config)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "episodes.jsonl"
        results = []

        with output_path.open("w") as f:
            episode_index = 0
            while episode_index < args.num_episodes:
                batch_size = min(args.parallel_envs, args.num_episodes - episode_index)
                batch_results = run_episode_batch(env, generator, episode_index, batch_size, args)
                for result in batch_results:
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(
                        f"[{result['episode_index'] + 1}/{args.num_episodes}] "
                        f"score={result['score']:.4f} won={result['won']} "
                        f"steps={result['episode_length']}"
                    )
                f.flush()
                episode_index += batch_size

        summary = {
            "num_episodes": len(results),
            "goal_start": args.goal_start,
            "goal_end": args.goal_end,
            "parallel_envs": args.parallel_envs,
            "mean_score": sum(result["score"] for result in results) / len(results) if results else 0.0,
            "success_rate": sum(1.0 for result in results if result["won"]) / len(results) if results else 0.0,
            "mean_episode_length": sum(result["episode_length"] for result in results) / len(results) if results else 0.0,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved episodes to {output_path}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if env is not None:
            env.envs.close()
        if generator is not None and hasattr(generator, "close"):
            generator.close()
        _cleanup_vllm_merged_model(args)


if __name__ == "__main__":
    main()
