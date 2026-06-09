# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


_LANGUAGE_MODULELIST_SUFFIXES = (
    "layers",
    "decoder.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.layers",
    "language_model.layers",
    "language_model.model.layers",
    "text_model.layers",
    "text_model.encoder.layers",
    "llm.layers",
    "llm.model.layers",
)

_NON_LANGUAGE_NAME_PARTS = (
    "audio",
    "acoustic",
    "av_",
    "clip",
    "image",
    "mm_projector",
    "multi_modal",
    "multimodal",
    "perceiver",
    "projector",
    "resampler",
    "speech",
    "vision",
    "visual",
)

_ATTENTION_NAME_PARTS = (
    "attention",
    "attn",
    "self_attn",
    "self_attention",
)

_EMBED_OR_HEAD_NAME_PARTS = (
    "embed",
    "embedding",
    "embeddings",
    "lm_head",
    "score",
)


def _as_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _has_numeric_children(module: nn.Module) -> bool:
    child_names = [name for name, _ in module.named_children()]
    return bool(child_names) and all(name.isdigit() for name in child_names)


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    value = value.lower()
    return any(needle in value for needle in needles)


def _is_language_layer_container(name: str, module: nn.Module) -> bool:
    if not name or not _has_numeric_children(module):
        return False
    lowered = name.lower()
    if _contains_any(lowered, _NON_LANGUAGE_NAME_PARTS):
        return False
    return isinstance(module, nn.ModuleList) or lowered.endswith(_LANGUAGE_MODULELIST_SUFFIXES)


def _find_language_layer_prefixes(model: nn.Module) -> list[str]:
    prefixes = [name for name, module in model.named_modules() if _is_language_layer_container(name, module)]

    # Keep the deepest containers only. For LoRA target discovery, layer-level
    # containers are safer than broad roots such as `model` on multimodal models.
    deepest = []
    for prefix in prefixes:
        if not any(other != prefix and other.startswith(f"{prefix}.") for other in prefixes):
            deepest.append(prefix)
    return deepest


def _is_under_any_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(f"{prefix}.") for prefix in prefixes)


def find_language_layer_classes(model: nn.Module) -> set[type[nn.Module]]:
    modules_by_name = dict(model.named_modules())
    classes = set()
    for prefix in _find_language_layer_prefixes(model):
        container = modules_by_name[prefix]
        for child in container.children():
            classes.add(child.__class__)
    return classes


def _is_linear_like(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    class_name = module.__class__.__name__.lower()
    return "linear" in class_name and hasattr(module, "weight")


def _matches_target_name(name: str, target_names: list[str]) -> bool:
    return any(name == target_name or name.endswith(f".{target_name}") for target_name in target_names)


def resolve_lora_target_modules(model: nn.Module, target_modules, target_scope: str | None = None):
    """Resolve LoRA targets, optionally restricting them to the LLM backbone.

    `target_scope=all` preserves PEFT's normal behavior. `llm` restricts targets
    to linear modules inside language-model layer stacks, and `llm_attention`
    further restricts them to attention submodules. This prevents PEFT's
    `all-linear` shortcut from attaching adapters to vision/audio encoders or
    multimodal projection modules.
    """

    scope = str(target_scope or "all").replace("-", "_").lower()
    if scope in ("all", "global", "none"):
        return target_modules
    if scope in ("language", "language_model", "text", "llm_all_linear"):
        scope = "llm"
    if scope in ("attention", "attn", "language_attention", "llm_attn"):
        scope = "llm_attention"
    if scope not in ("llm", "llm_attention"):
        raise ValueError(f"Unsupported model.lora_target_scope={target_scope!r}. Use one of: all, llm, llm_attention.")

    language_prefixes = _find_language_layer_prefixes(model)
    if not language_prefixes:
        raise ValueError(
            "Could not find language-model layer containers for model.lora_target_scope="
            f"{target_scope!r}. Set model.lora_target_scope=all or pass explicit model.target_modules."
        )

    requested = target_modules
    if isinstance(requested, str) and requested == "all-linear":
        target_names = None
    else:
        target_names = [str(name) for name in _as_list(requested)]

    resolved = []
    for name, module in model.named_modules():
        if not name or not _is_under_any_prefix(name, language_prefixes):
            continue
        if _contains_any(name, _EMBED_OR_HEAD_NAME_PARTS):
            continue
        if scope == "llm_attention" and not _contains_any(name, _ATTENTION_NAME_PARTS):
            continue
        if not _is_linear_like(module):
            continue
        if target_names is not None and not _matches_target_name(name, target_names):
            continue
        resolved.append(name)

    if not resolved:
        raise ValueError(
            "No LoRA target modules matched "
            f"target_modules={target_modules!r}, lora_target_scope={target_scope!r}, "
            f"language_prefixes={language_prefixes!r}."
        )
    return resolved
