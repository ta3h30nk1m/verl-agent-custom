#!/usr/bin/env python3
"""Initialize a target HyperCLOVAX LoRA adapter from a trained source adapter.

This implements the single-source gradient-projection transfer used in the
EvolvingCL Gradtransfer path: estimate base-layer gradients on the calibration
SFT set for both source and target models, align source LoRA A/B through the
source/target gradient singular subspaces, and save a target PEFT adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from verl.trainer.fsdp_sft_trainer import collate_sft_batch
from verl.utils import hf_tokenizer
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.lora_utils import find_language_layer_prefixes, resolve_lora_target_modules
from verl.utils.py_functional import convert_to_regular_types


@dataclass
class LoraModuleInfo:
    name: str
    layer_idx: int
    relative_name: str
    module: nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-path", required=True, help="Base model path for the trained source LoRA.")
    parser.add_argument("--source-lora-path", required=True, help="PEFT adapter directory trained on the source model.")
    parser.add_argument("--target-model-path", required=True, help="Base model path for the target LoRA initialization.")
    parser.add_argument("--output-dir", required=True, help="Directory where the initialized target PEFT adapter is saved.")
    parser.add_argument("--train-files", required=True, help="Calibration parquet file(s), comma-separated if multiple.")
    parser.add_argument("--multiturn", action="store_true", help="Use MultiTurnSFTDataset with messages.")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument("--response-key", default="answer")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--truncation", choices=["error", "left", "right"], default="left")
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--lora-target-scope", default="llm")
    parser.add_argument("--target-lora-rank", type=int, default=None)
    parser.add_argument("--target-lora-alpha", type=int, default=None)
    parser.add_argument("--projection-rank", type=int, default=None, help="SVD rank used for projection. Defaults to LoRA rank.")
    parser.add_argument(
        "--projection-method",
        choices=["simple", "orthogonal", "scale_b", "delta_svd"],
        default="simple",
        help="simple matches the active Gradtransfer simple_mapping branch.",
    )
    parser.add_argument("--gradient-scale", type=float, default=1.0)
    parser.add_argument("--normalize-gradients", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_paths(value: str) -> list[str]:
    return [path for path in (item.strip() for item in value.split(",")) if path]


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_base_model(model_path: str, args: argparse.Namespace) -> nn.Module:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype_from_name(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    return model


def get_layer_prefix_and_count(model: nn.Module) -> tuple[str, int]:
    modules = dict(model.named_modules())
    candidates = []
    for prefix in find_language_layer_prefixes(model):
        module = modules[prefix]
        child_names = [name for name, _ in module.named_children()]
        numeric_children = [int(name) for name in child_names if name.isdigit()]
        if numeric_children:
            candidates.append((prefix, len(numeric_children)))
    if not candidates:
        raise ValueError("Could not find a language-model layer stack.")
    return max(candidates, key=lambda item: item[1])


def peft_layer_prefix(base_layer_prefix: str) -> str:
    return f"base_model.model.{base_layer_prefix}" if base_layer_prefix else "base_model.model"


def is_lora_module(module: nn.Module) -> bool:
    return (
        hasattr(module, "base_layer")
        and hasattr(module.base_layer, "weight")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and "default" in module.lora_A
        and "default" in module.lora_B
    )


def collect_lora_modules(model: nn.Module, layer_prefix: str) -> list[LoraModuleInfo]:
    prefix = f"{layer_prefix}."
    modules: list[LoraModuleInfo] = []
    for name, module in model.named_modules():
        if not is_lora_module(module) or not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        layer_str, sep, relative_name = rest.partition(".")
        if not sep or not layer_str.isdigit() or not relative_name:
            continue
        modules.append(
            LoraModuleInfo(
                name=name,
                layer_idx=int(layer_str),
                relative_name=relative_name,
                module=module,
            )
        )
    return modules


def make_dataset(args: argparse.Namespace, tokenizer) -> Subset:
    config = {
        "prompt_key": args.prompt_key,
        "response_key": args.response_key,
        "max_length": args.max_length,
        "pad_to_max_length": args.pad_to_max_length,
        "truncation": args.truncation,
        "multiturn": {"messages_key": args.messages_key},
    }
    dataset_cls = MultiTurnSFTDataset if args.multiturn else SFTDataset
    dataset = dataset_cls(split_paths(args.train_files), tokenizer, config)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.num_calibration_samples > 0:
        indices = indices[: args.num_calibration_samples]
    if not indices:
        raise ValueError("Calibration dataset is empty.")
    return Subset(dataset, indices)


def compute_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch.get("position_ids", None)
    loss_mask = batch["loss_mask"][:, :-1].reshape(-1)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if position_ids is not None:
        model_inputs["position_ids"] = position_ids
    outputs = model(**model_inputs)
    shift_logits = outputs.logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous().reshape(-1).to(shift_logits.device)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels,
        reduction="none",
    )
    loss_mask = loss_mask.to(token_loss.device)
    return (token_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)


def get_record_gradient_hook(
    target_params: list[tuple[str, torch.nn.Parameter]],
    record_dict: dict[str, torch.Tensor],
):
    """Record already materialized grads and immediately free their GPU storage.

    This mirrors the EvolvingCL Gradtransfer hook pattern: every parameter hook
    scans target parameters whose `.grad` has been populated by autograd,
    accumulates those tensors on CPU, then sets `.grad = None` before backward
    proceeds to the next parameters.
    """

    def record_gradient_hook(grad):
        for name, param in target_params:
            if param.requires_grad and param.grad is not None:
                grad_cpu = param.grad.detach().float().cpu()
                if name not in record_dict:
                    record_dict[name] = grad_cpu
                else:
                    record_dict[name].add_(grad_cpu)
                param.grad = None
        return grad

    return record_gradient_hook


def estimate_base_weight_gradients(
    model: nn.Module,
    lora_modules: list[LoraModuleInfo],
    dataset: Subset,
    tokenizer,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, torch.Tensor]:
    for param in model.parameters():
        param.requires_grad_(False)

    target_params = []
    for info in lora_modules:
        module = info.module
        module.base_layer.weight.requires_grad_(True)
        target_params.append((info.name, module.base_layer.weight))

    device = torch.device(args.device)
    model.to(device)
    model.train()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda data_list: collate_sft_batch(data_list, pad_token_id),
    )

    gradients = {}
    hooks = []
    record_gradient_hook = get_record_gradient_hook(target_params, gradients)
    for _, param in target_params:
        hooks.append(param.register_hook(record_gradient_hook))

    num_batches = 0
    try:
        for batch in tqdm(dataloader, desc=desc):
            batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            loss = compute_loss(model, batch)
            loss.backward()
            record_gradient_hook(None)
            num_batches += 1
            for _, param in target_params:
                param.grad = None
            model.zero_grad(set_to_none=True)
            del loss
            del batch
    finally:
        for hook in hooks:
            hook.remove()

    if num_batches == 0:
        raise ValueError("No calibration batches were produced.")
    for name, grad in gradients.items():
        grad.mul_(-1.0 / num_batches)
        if args.normalize_gradients:
            grad.div_(grad.norm() + 1e-12)

    model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return gradients


def map_target_layer_to_source(target_idx: int, target_count: int, source_count: int) -> int:
    if target_count <= 1 or source_count <= 1:
        return 0
    return round(target_idx * (source_count - 1) / (target_count - 1))


def project_lora_weights(
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    source_grad: torch.Tensor,
    target_grad: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    source_a = source_a.float().to(projection_device)
    source_b = source_b.float().to(projection_device)
    source_grad = source_grad.float().to(projection_device)
    target_grad = target_grad.float().to(projection_device)

    u_t, s_t, vh_t = torch.linalg.svd(target_grad, full_matrices=False)
    u_s, s_s, vh_s = torch.linalg.svd(source_grad, full_matrices=False)
    rank = source_a.shape[0]
    projection_rank = args.projection_rank or rank
    rank = min(rank, projection_rank, u_t.shape[1], u_s.shape[1], vh_t.shape[0], vh_s.shape[0])
    if rank <= 0:
        raise ValueError("Projection rank became zero.")

    eps = 1e-8
    u_t_r = u_t[:, :rank]
    u_s_r = u_s[:, :rank]
    v_t_r = vh_t[:rank, :].T
    v_s_r = vh_s[:rank, :].T

    if args.projection_method == "simple":
        scale = torch.sqrt(s_t[:rank] / (s_s[:rank] + eps))
        p_b = (u_t_r * scale.unsqueeze(0)) @ u_s_r.T
        p_a = (v_t_r * scale.unsqueeze(0)) @ v_s_r.T
    elif args.projection_method == "scale_b":
        p_b = (u_t_r * (s_t[:rank] / (s_s[:rank] + eps)).unsqueeze(0)) @ u_s_r.T
        p_a = v_t_r @ v_s_r.T
    elif args.projection_method == "delta_svd":
        delta_u_s, _, delta_vh_s = torch.linalg.svd(source_b @ source_a, full_matrices=False)
        rank = min(rank, delta_u_s.shape[1], delta_vh_s.shape[0])
        p_b = (u_t[:, :rank] * torch.sqrt(s_t[:rank]).unsqueeze(0)) @ delta_u_s[:, :rank].T
        p_a = (vh_t[:rank, :].T * torch.sqrt(s_t[:rank]).unsqueeze(0)) @ delta_vh_s[:rank, :].T
    else:
        p_b = u_t_r @ u_s_r.T
        p_a = v_t_r @ v_s_r.T

    scale = math.sqrt(args.gradient_scale)
    target_a = (source_a @ p_a.T) * scale
    target_b = (p_b @ source_b) * scale
    target_a = target_a.cpu()
    target_b = target_b.cpu()
    if projection_device.type == "cuda":
        torch.cuda.empty_cache()
    return target_a, target_b


def copy_projected_weights(
    target_modules: list[LoraModuleInfo],
    source_modules: list[LoraModuleInfo],
    target_gradients: dict[str, torch.Tensor],
    source_gradients: dict[str, torch.Tensor],
    target_layer_count: int,
    source_layer_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_by_key = {(info.layer_idx, info.relative_name): info for info in source_modules}
    projected = []
    skipped = []
    layer_mapping = {}

    with torch.no_grad():
        for target_info in target_modules:
            source_idx = map_target_layer_to_source(target_info.layer_idx, target_layer_count, source_layer_count)
            layer_mapping[str(target_info.layer_idx)] = source_idx
            source_info = source_by_key.get((source_idx, target_info.relative_name))
            if source_info is None:
                skipped.append({"target": target_info.name, "reason": "missing matching source LoRA module"})
                continue

            source_a = source_info.module.lora_A["default"].weight.detach().cpu()
            source_b = source_info.module.lora_B["default"].weight.detach().cpu()
            target_a_param = target_info.module.lora_A["default"].weight
            target_b_param = target_info.module.lora_B["default"].weight
            if tuple(target_a_param.shape) != (source_a.shape[0], target_a_param.shape[1]):
                skipped.append({"target": target_info.name, "source": source_info.name, "reason": "LoRA rank mismatch"})
                continue
            if tuple(target_b_param.shape) != (target_b_param.shape[0], source_b.shape[1]):
                skipped.append({"target": target_info.name, "source": source_info.name, "reason": "LoRA rank mismatch"})
                continue

            target_a, target_b = project_lora_weights(
                source_a,
                source_b,
                source_gradients[source_info.name],
                target_gradients[target_info.name],
                args,
            )
            if tuple(target_a.shape) != tuple(target_a_param.shape) or tuple(target_b.shape) != tuple(target_b_param.shape):
                skipped.append(
                    {
                        "target": target_info.name,
                        "source": source_info.name,
                        "reason": f"projected shape mismatch A={tuple(target_a.shape)} B={tuple(target_b.shape)}",
                    }
                )
                continue
            target_a_param.copy_(target_a.to(dtype=target_a_param.dtype))
            target_b_param.copy_(target_b.to(dtype=target_b_param.dtype))
            projected.append({"target": target_info.name, "source": source_info.name})

    if not projected:
        raise ValueError(f"No LoRA modules were projected. First skipped items: {skipped[:5]}")
    return {"projected": projected, "skipped": skipped, "layer_mapping": layer_mapping}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_adapter_config = PeftConfig.from_pretrained(args.source_lora_path)
    target_rank = args.target_lora_rank or int(source_adapter_config.r)
    target_alpha = args.target_lora_alpha or int(source_adapter_config.lora_alpha)

    tokenizer = hf_tokenizer(args.target_model_path, trust_remote_code=args.trust_remote_code)
    calibration_dataset = make_dataset(args, tokenizer)

    print(f"Loading target model: {args.target_model_path}", flush=True)
    target_base = load_base_model(args.target_model_path, args)
    target_layer_prefix, target_layer_count = get_layer_prefix_and_count(target_base)
    target_modules = resolve_lora_target_modules(
        target_base,
        convert_to_regular_types(args.target_modules),
        args.lora_target_scope,
    )
    target_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=target_rank,
        lora_alpha=target_alpha,
        target_modules=target_modules,
        bias="none",
    )
    target_model = get_peft_model(target_base, target_lora_config)
    target_lora_modules = collect_lora_modules(target_model, peft_layer_prefix(target_layer_prefix))
    if not target_lora_modules:
        raise ValueError("No target LoRA modules were found.")
    target_gradients = estimate_base_weight_gradients(
        target_model,
        target_lora_modules,
        calibration_dataset,
        tokenizer,
        args,
        desc="Target gradients",
    )

    print(f"Loading source model: {args.source_model_path}", flush=True)
    source_base = load_base_model(args.source_model_path, args)
    source_layer_prefix, source_layer_count = get_layer_prefix_and_count(source_base)
    source_model = PeftModel.from_pretrained(source_base, args.source_lora_path, is_trainable=True)
    source_lora_modules = collect_lora_modules(source_model, peft_layer_prefix(source_layer_prefix))
    if not source_lora_modules:
        raise ValueError("No source LoRA modules were found.")
    source_rank = source_lora_modules[0].module.lora_A["default"].weight.shape[0]
    if source_rank != target_rank:
        raise ValueError(
            f"Source adapter rank ({source_rank}) differs from target rank ({target_rank}). "
            "Use the same rank or regenerate the target adapter config."
        )
    source_gradients = estimate_base_weight_gradients(
        source_model,
        source_lora_modules,
        calibration_dataset,
        tokenizer,
        args,
        desc="Source gradients",
    )
    source_model.to("cpu")
    del source_model
    del source_base
    torch.cuda.empty_cache()
    gc.collect()

    summary = copy_projected_weights(
        target_lora_modules,
        source_lora_modules,
        target_gradients,
        source_gradients,
        target_layer_count,
        source_layer_count,
        args,
    )
    summary.update(
        {
            "source_model_path": args.source_model_path,
            "source_lora_path": args.source_lora_path,
            "target_model_path": args.target_model_path,
            "train_files": split_paths(args.train_files),
            "num_calibration_samples": len(calibration_dataset),
            "projection_method": args.projection_method,
            "projection_rank": args.projection_rank or target_rank,
            "gradient_scale": args.gradient_scale,
            "target_layer_prefix": target_layer_prefix,
            "source_layer_prefix": source_layer_prefix,
            "target_layer_count": target_layer_count,
            "source_layer_count": source_layer_count,
        }
    )

    target_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "lora_transfer_init_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Saved initialized target LoRA adapter to {output_dir} "
        f"({len(summary['projected'])} modules projected, {len(summary['skipped'])} skipped).",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
