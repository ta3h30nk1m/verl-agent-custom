"""Helpers for LoRA-GA adapters that carry a base-weight offset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


LORAGA_BASE_DELTA_FILENAME = "loraga_base_delta.safetensors"
LORAGA_METADATA_FILENAME = "loraga_base_delta_metadata.json"


def _is_lora_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "base_layer")
        and hasattr(module.base_layer, "weight")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and "default" in module.lora_A
        and "default" in module.lora_B
    )


def iter_lora_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if _is_lora_module(module):
            yield name, module


def loraga_base_delta_path(adapter_path: str | Path) -> Path:
    return Path(adapter_path) / LORAGA_BASE_DELTA_FILENAME


def has_loraga_base_delta(adapter_path: str | Path) -> bool:
    return loraga_base_delta_path(adapter_path).is_file()


def save_loraga_base_delta(
    adapter_path: str | Path,
    delta_factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save base-offset factors as LoRA-style A/B tensors.

    Each entry stores an offset equivalent to
    ``module.scaling["default"] * (B @ A)``. Loading the adapter subtracts that
    offset from the base layer before the trainable LoRA adapter is used.
    """

    adapter_path = Path(adapter_path)
    tensors: dict[str, torch.Tensor] = {}
    for module_name, (delta_a, delta_b) in delta_factors.items():
        tensors[f"{module_name}.lora_A.default.weight"] = delta_a.detach().cpu().contiguous()
        tensors[f"{module_name}.lora_B.default.weight"] = delta_b.detach().cpu().contiguous()
    if not tensors:
        return

    save_file(tensors, adapter_path / LORAGA_BASE_DELTA_FILENAME)
    if metadata is not None:
        (adapter_path / LORAGA_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def apply_loraga_base_delta(
    model: torch.nn.Module,
    adapter_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Apply a saved LoRA-GA base offset to a loaded PEFT model."""

    delta_path = loraga_base_delta_path(adapter_path)
    if not delta_path.is_file():
        return {"applied": [], "missing": [], "unexpected": []}

    tensors = load_file(str(delta_path), device="cpu")
    used_keys: set[str] = set()
    applied: list[str] = []
    missing: list[str] = []

    with torch.no_grad():
        for module_name, module in iter_lora_modules(model):
            a_key = f"{module_name}.lora_A.default.weight"
            b_key = f"{module_name}.lora_B.default.weight"
            if a_key not in tensors or b_key not in tensors:
                missing.append(module_name)
                continue

            delta_a = tensors[a_key].to(device=module.base_layer.weight.device, dtype=torch.float32)
            delta_b = tensors[b_key].to(device=module.base_layer.weight.device, dtype=torch.float32)
            offset = delta_b @ delta_a
            scaling = float(module.scaling["default"]) if hasattr(module, "scaling") else 1.0
            offset.mul_(scaling)
            module.base_layer.weight.data.sub_(offset.to(dtype=module.base_layer.weight.dtype))
            used_keys.update({a_key, b_key})
            applied.append(module_name)

    unexpected = sorted(set(tensors) - used_keys)
    if strict and (missing or unexpected):
        raise ValueError(
            "LoRA-GA base delta did not match the loaded adapter: "
            f"{len(missing)} missing modules, {len(unexpected)} unexpected tensors."
        )
    return {"applied": applied, "missing": missing, "unexpected": unexpected}


def copy_loraga_base_delta_files(src_adapter_path: str | Path | None, dst_adapter_path: str | Path) -> bool:
    if src_adapter_path is None:
        return False
    src_adapter_path = Path(src_adapter_path)
    dst_adapter_path = Path(dst_adapter_path)
    delta_path = src_adapter_path / LORAGA_BASE_DELTA_FILENAME
    if not delta_path.is_file():
        return False

    dst_adapter_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(delta_path, dst_adapter_path / LORAGA_BASE_DELTA_FILENAME)
    metadata_path = src_adapter_path / LORAGA_METADATA_FILENAME
    if metadata_path.is_file():
        shutil.copy2(metadata_path, dst_adapter_path / LORAGA_METADATA_FILENAME)
    return True
