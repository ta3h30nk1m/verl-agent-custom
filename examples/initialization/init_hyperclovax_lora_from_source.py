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
from verl.utils.lora_ga import save_loraga_base_delta
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
    parser.add_argument(
        "--init-method",
        choices=["gradtransfer", "gradtransfer_loraga"],
        default="gradtransfer",
        help="gradtransfer_loraga applies a LoRA-GA base offset after Gradtransfer initialization.",
    )
    parser.add_argument(
        "--num-projection-steps",
        type=int,
        default=1,
        help="Repeat Gradtransfer projection N times, updating the target base between attempts like double/tripletransfer.",
    )
    parser.add_argument(
        "--projection-accumulation-scale",
        type=float,
        default=1.0,
        help="Scale used when applying interim projected deltas to the target base and combining repeated projections.",
    )
    parser.add_argument(
        "--source-gradient-mode",
        choices=["lora_first_then_zero", "always_lora", "always_zero"],
        default="lora_first_then_zero",
        help="Source model state used when estimating source gradients across repeated projection steps.",
    )
    parser.add_argument(
        "--loraga-gamma",
        type=float,
        default=32.0,
        help="LoRA-GA gamma used to scale the SVD factors from the post-transfer target gradient.",
    )
    parser.add_argument(
        "--loraga-disable-norm-clip",
        action="store_true",
        help="Disable LoRA-GA offset clipping against the current base-plus-transfer weight magnitude.",
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


def get_lora_scaling(module: nn.Module) -> float:
    if hasattr(module, "scaling") and "default" in module.scaling:
        return float(module.scaling["default"])
    rank = int(module.r["default"])
    alpha = float(module.lora_alpha["default"])
    return alpha / rank


def zero_lora_weights(lora_modules: list[LoraModuleInfo]) -> None:
    with torch.no_grad():
        for info in lora_modules:
            info.module.lora_A["default"].weight.zero_()
            info.module.lora_B["default"].weight.zero_()


def snapshot_lora_weights(lora_modules: list[LoraModuleInfo]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        info.name: (
            info.module.lora_A["default"].weight.detach().cpu().clone(),
            info.module.lora_B["default"].weight.detach().cpu().clone(),
        )
        for info in lora_modules
    }


def restore_lora_weights(lora_modules: list[LoraModuleInfo], weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for info in lora_modules:
            source_a, source_b = weights[info.name]
            a_param = info.module.lora_A["default"].weight
            b_param = info.module.lora_B["default"].weight
            a_param.copy_(source_a.to(device=a_param.device, dtype=a_param.dtype))
            b_param.copy_(source_b.to(device=b_param.device, dtype=b_param.dtype))


def project_lora_step(
    target_modules: list[LoraModuleInfo],
    source_modules: list[LoraModuleInfo],
    target_gradients: dict[str, torch.Tensor],
    source_gradients: dict[str, torch.Tensor],
    target_layer_count: int,
    source_layer_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_by_key = {(info.layer_idx, info.relative_name): info for info in source_modules}
    weights = {}
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
            target_a_shape = tuple(target_info.module.lora_A["default"].weight.shape)
            target_b_shape = tuple(target_info.module.lora_B["default"].weight.shape)
            if target_a_shape != (source_a.shape[0], target_a_shape[1]):
                skipped.append({"target": target_info.name, "source": source_info.name, "reason": "LoRA rank mismatch"})
                continue
            if target_b_shape != (target_b_shape[0], source_b.shape[1]):
                skipped.append({"target": target_info.name, "source": source_info.name, "reason": "LoRA rank mismatch"})
                continue

            target_a, target_b = project_lora_weights(
                source_a,
                source_b,
                source_gradients[source_info.name],
                target_gradients[target_info.name],
                args,
            )
            if tuple(target_a.shape) != target_a_shape or tuple(target_b.shape) != target_b_shape:
                skipped.append(
                    {
                        "target": target_info.name,
                        "source": source_info.name,
                        "reason": f"projected shape mismatch A={tuple(target_a.shape)} B={tuple(target_b.shape)}",
                    }
                )
                continue
            weights[target_info.name] = {"a": target_a, "b": target_b, "source": source_info.name}
            projected.append({"target": target_info.name, "source": source_info.name})

    if not projected:
        raise ValueError(f"No LoRA modules were projected. First skipped items: {skipped[:5]}")
    return {"weights": weights, "projected": projected, "skipped": skipped, "layer_mapping": layer_mapping}


def copy_lora_weights_to_target(
    target_modules: list[LoraModuleInfo],
    weights: dict[str, dict[str, Any]],
) -> None:
    target_by_name = {info.name: info for info in target_modules}
    with torch.no_grad():
        for name, item in weights.items():
            module = target_by_name[name].module
            a_param = module.lora_A["default"].weight
            b_param = module.lora_B["default"].weight
            a_param.copy_(item["a"].to(device=a_param.device, dtype=a_param.dtype))
            b_param.copy_(item["b"].to(device=b_param.device, dtype=b_param.dtype))


def apply_projected_delta_to_base(
    target_modules: list[LoraModuleInfo],
    weights: dict[str, dict[str, Any]],
    scale: float,
) -> None:
    target_by_name = {info.name: info for info in target_modules}
    with torch.no_grad():
        for name, item in weights.items():
            module = target_by_name[name].module
            delta = item["b"].float() @ item["a"].float()
            delta.mul_(get_lora_scaling(module) * scale)
            delta = delta.to(device=module.base_layer.weight.device, dtype=module.base_layer.weight.dtype)
            module.base_layer.weight.add_(delta)


def restore_projected_base_updates(
    target_modules: list[LoraModuleInfo],
    step_weights: list[dict[str, dict[str, Any]]],
    scale: float,
) -> None:
    for weights in reversed(step_weights):
        apply_projected_delta_to_base(target_modules, weights, -scale)


def combine_projection_steps(step_weights: list[dict[str, dict[str, Any]]], accumulation_scale: float) -> dict[str, dict[str, Any]]:
    if not step_weights:
        raise ValueError("No projection steps were computed.")
    names = set(step_weights[0])
    for weights in step_weights[1:]:
        names.intersection_update(weights)
    if not names:
        raise ValueError("No LoRA modules were projected in every projection step.")

    combine_scale = math.sqrt(accumulation_scale) if len(step_weights) > 1 else 1.0
    combined = {}
    for name in sorted(names):
        a = sum(weights[name]["a"].float() for weights in step_weights) * combine_scale
        b = sum(weights[name]["b"].float() for weights in step_weights) * combine_scale
        combined[name] = {"a": a.cpu(), "b": b.cpu(), "source": step_weights[0][name]["source"]}
    return combined


def initialize_with_gradtransfer(
    target_model: nn.Module,
    target_modules: list[LoraModuleInfo],
    source_model: nn.Module,
    source_modules: list[LoraModuleInfo],
    calibration_dataset: Subset,
    tokenizer,
    target_layer_count: int,
    source_layer_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.num_projection_steps < 1:
        raise ValueError("--num-projection-steps must be >= 1.")
    if args.projection_accumulation_scale <= 0:
        raise ValueError("--projection-accumulation-scale must be > 0.")

    source_lora_snapshot = snapshot_lora_weights(source_modules)
    all_step_weights = []
    base_update_steps: list[dict[str, dict[str, Any]]] = []
    projected_steps = []
    skipped_steps = []
    layer_mapping = {}

    try:
        for step_idx in range(args.num_projection_steps):
            print(f"Projection step {step_idx + 1}/{args.num_projection_steps}", flush=True)
            if args.source_gradient_mode == "always_zero" or (
                args.source_gradient_mode == "lora_first_then_zero" and step_idx > 0
            ):
                zero_lora_weights(source_modules)
            else:
                restore_lora_weights(source_modules, source_lora_snapshot)

            target_gradients = estimate_base_weight_gradients(
                target_model,
                target_modules,
                calibration_dataset,
                tokenizer,
                args,
                desc=f"Target gradients step {step_idx + 1}",
            )
            source_gradients = estimate_base_weight_gradients(
                source_model,
                source_modules,
                calibration_dataset,
                tokenizer,
                args,
                desc=f"Source gradients step {step_idx + 1}",
            )
            restore_lora_weights(source_modules, source_lora_snapshot)
            step_summary = project_lora_step(
                target_modules,
                source_modules,
                target_gradients,
                source_gradients,
                target_layer_count,
                source_layer_count,
                args,
            )
            all_step_weights.append(step_summary["weights"])
            projected_steps.append(step_summary["projected"])
            skipped_steps.append(step_summary["skipped"])
            layer_mapping.update(step_summary["layer_mapping"])

            del target_gradients
            del source_gradients
            torch.cuda.empty_cache()
            gc.collect()

            if step_idx + 1 < args.num_projection_steps:
                apply_projected_delta_to_base(
                    target_modules,
                    step_summary["weights"],
                    args.projection_accumulation_scale,
                )
                base_update_steps.append(step_summary["weights"])
    finally:
        restore_projected_base_updates(target_modules, base_update_steps, args.projection_accumulation_scale)
        restore_lora_weights(source_modules, source_lora_snapshot)

    combined_weights = combine_projection_steps(all_step_weights, args.projection_accumulation_scale)
    copy_lora_weights_to_target(target_modules, combined_weights)
    projected = [{"target": name, "source": item["source"]} for name, item in combined_weights.items()]
    return {
        "projected": projected,
        "skipped": [item for skipped in skipped_steps for item in skipped],
        "projected_steps": projected_steps,
        "layer_mapping": layer_mapping,
    }


def apply_loraga_initialization(
    target_model: nn.Module,
    target_modules: list[LoraModuleInfo],
    calibration_dataset: Subset,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    if args.loraga_gamma <= 0:
        raise ValueError("--loraga-gamma must be > 0.")

    gradients = estimate_base_weight_gradients(
        target_model,
        target_modules,
        calibration_dataset,
        tokenizer,
        args,
        desc="LoRA-GA target gradients",
    )
    projection_device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    delta_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    applied = []

    with torch.no_grad():
        for info in target_modules:
            module = info.module
            grad = gradients[info.name].float().to(projection_device)
            u, _, vh = torch.linalg.svd(-grad, full_matrices=False)
            rank = module.lora_A["default"].weight.shape[0]
            if u.shape[1] >= 2 * rank:
                b = u[:, rank : 2 * rank]
            else:
                b = u[:, :rank]
            if vh.shape[0] < rank:
                raise ValueError(f"LoRA-GA rank {rank} exceeds gradient rank for {info.name}.")
            a = vh[:rank, :]

            out_dim = grad.shape[0]
            factor = (out_dim**0.25) / math.sqrt(args.loraga_gamma)
            a = a * factor
            b = b * factor

            prev_a = module.lora_A["default"].weight.detach().float().to(projection_device)
            prev_b = module.lora_B["default"].weight.detach().float().to(projection_device)
            scaling = get_lora_scaling(module)
            offset = (b @ a) * scaling
            prev_delta = (prev_b @ prev_a) * scaling

            clip_ratio = None
            if not args.loraga_disable_norm_clip:
                denominator = torch.max(torch.abs(offset))
                if denominator > 0:
                    numerator = torch.max(torch.abs(module.base_layer.weight.detach().to(projection_device).float() + prev_delta))
                    ratio = numerator / denominator
                    if ratio < 1:
                        clip_ratio = float(ratio.item())
                        scale = torch.sqrt(ratio)
                        a = a * scale
                        b = b * scale

            delta_a = torch.cat([a, prev_a], dim=0)
            delta_b = torch.cat([b, -prev_b], dim=1)
            base_delta = (delta_b @ delta_a) * scaling
            module.base_layer.weight.sub_(base_delta.to(device=module.base_layer.weight.device, dtype=module.base_layer.weight.dtype))
            module.lora_A["default"].weight.copy_(a.to(device=module.lora_A["default"].weight.device, dtype=module.lora_A["default"].weight.dtype))
            module.lora_B["default"].weight.copy_(b.to(device=module.lora_B["default"].weight.device, dtype=module.lora_B["default"].weight.dtype))
            delta_factors[info.name] = (delta_a.cpu(), delta_b.cpu())
            applied.append(
                {
                    "target": info.name,
                    "delta_rank": int(delta_a.shape[0]),
                    "clip_ratio": clip_ratio,
                }
            )

    del gradients
    if projection_device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return delta_factors, {"loraga_applied": applied, "loraga_gamma": args.loraga_gamma}


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

    summary = initialize_with_gradtransfer(
        target_model,
        target_lora_modules,
        source_model,
        source_lora_modules,
        calibration_dataset,
        tokenizer,
        target_layer_count,
        source_layer_count,
        args,
    )
    source_model.to("cpu")
    del source_model
    del source_base
    torch.cuda.empty_cache()
    gc.collect()

    loraga_delta_factors = None
    if args.init_method == "gradtransfer_loraga":
        loraga_delta_factors, loraga_summary = apply_loraga_initialization(
            target_model,
            target_lora_modules,
            calibration_dataset,
            tokenizer,
            args,
        )
        summary.update(loraga_summary)

    summary.update(
        {
            "source_model_path": args.source_model_path,
            "source_lora_path": args.source_lora_path,
            "target_model_path": args.target_model_path,
            "train_files": split_paths(args.train_files),
            "num_calibration_samples": len(calibration_dataset),
            "init_method": args.init_method,
            "num_projection_steps": args.num_projection_steps,
            "projection_method": args.projection_method,
            "projection_rank": args.projection_rank or target_rank,
            "projection_accumulation_scale": args.projection_accumulation_scale,
            "source_gradient_mode": args.source_gradient_mode,
            "gradient_scale": args.gradient_scale,
            "target_layer_prefix": target_layer_prefix,
            "source_layer_prefix": source_layer_prefix,
            "target_layer_count": target_layer_count,
            "source_layer_count": source_layer_count,
        }
    )

    target_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    if loraga_delta_factors is not None:
        save_loraga_base_delta(
            output_dir,
            loraga_delta_factors,
            metadata={
                "init_method": args.init_method,
                "loraga_gamma": args.loraga_gamma,
                "num_projection_steps": args.num_projection_steps,
                "projection_accumulation_scale": args.projection_accumulation_scale,
                "source_model_path": args.source_model_path,
                "source_lora_path": args.source_lora_path,
                "target_model_path": args.target_model_path,
            },
        )
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
