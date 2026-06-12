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
"""FSDP SFT trainer with token-level teacher distillation."""

from __future__ import annotations

import importlib
import logging
import re

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from peft import PeftModel
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.utils.debug import log_gpu_memory_usage


logger = logging.getLogger(__file__)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(name).lower()]


def _none_if_null(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _parse_kd_divergence(name, default_beta=0.5) -> tuple[str, float]:
    name = str(name or "rkl").strip().lower().replace(" ", "")
    aliases = {
        "kl": "fkl",
        "fkl": "fkl",
        "forwardkl": "fkl",
        "forward_kl": "fkl",
        "rkl": "rkl",
        "reversekl": "rkl",
        "reverse_kl": "rkl",
        "js": "jsd",
        "jsd": "jsd",
        "jensen_shannon": "jsd",
        "skew_kl": "skew_fkl",
        "skew_fkl": "skew_fkl",
        "skew_forwardkl": "skew_fkl",
        "skew_forward_kl": "skew_fkl",
        "skew_rkl": "skew_rkl",
        "skew_reversekl": "skew_rkl",
        "skew_reverse_kl": "skew_rkl",
    }
    if name in aliases:
        return aliases[name], float(default_beta)

    match = re.match(r"jsd\(([^)]+)\)", name)
    if match is not None:
        return "jsd", float(match.group(1))

    if name.startswith("jsd_"):
        return "jsd", float(name.split("_", 1)[1])

    raise ValueError(f"Unknown KD divergence: {name}")


def kd_divergence_token_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_type: str = "rkl",
    temperature: float = 1.0,
    jsd_beta: float = 0.5,
    skew_alpha: float = 0.1,
) -> torch.Tensor:
    """Return per-token KD loss for logits shaped [..., vocab]."""

    student_logits = student_logits.float() / temperature
    teacher_logits = teacher_logits.float() / temperature

    log_q = F.log_softmax(student_logits, dim=-1)
    log_p = F.log_softmax(teacher_logits, dim=-1)
    q = log_q.exp()
    p = log_p.exp()

    if loss_type == "fkl":
        token_loss = (p * (log_p - log_q)).sum(dim=-1)
    elif loss_type == "rkl":
        token_loss = (q * (log_q - log_p)).sum(dim=-1)
    elif loss_type == "jsd":
        beta = min(max(float(jsd_beta), 1e-6), 1.0 - 1e-6)
        m = beta * p + (1.0 - beta) * q
        log_m = m.clamp_min(1e-8).log()
        token_loss = beta * (p * (log_p - log_m)).sum(dim=-1)
        token_loss = token_loss + (1.0 - beta) * (q * (log_q - log_m)).sum(dim=-1)
    elif loss_type == "skew_fkl":
        alpha = min(max(float(skew_alpha), 1e-6), 1.0 - 1e-6)
        m = alpha * p + (1.0 - alpha) * q
        log_m = m.clamp_min(1e-8).log()
        token_loss = (p * (log_p - log_m)).sum(dim=-1)
    elif loss_type == "skew_rkl":
        alpha = min(max(float(skew_alpha), 1e-6), 1.0 - 1e-6)
        m = alpha * q + (1.0 - alpha) * p
        log_m = m.clamp_min(1e-8).log()
        token_loss = (q * (log_q - log_m)).sum(dim=-1)
    else:
        raise ValueError(f"Unknown KD divergence: {loss_type}")

    return token_loss * (temperature**2)


class FSDPSFTKDTrainer(FSDPSFTTrainer):
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        distill_config = config.get("distillation", {})
        self.kd_loss_type, self.kd_jsd_beta = _parse_kd_divergence(
            distill_config.get("loss_type", "rkl"),
            distill_config.get("jsd_beta", 0.5),
        )
        self.kd_temperature = float(distill_config.get("temperature", 1.0))
        self.kd_skew_alpha = float(distill_config.get("skew_alpha", 0.1))
        self.kd_coeff = float(distill_config.get("kd_coeff", 0.0))
        self.kd_balancing_coeff = float(distill_config.get("kd_balancing_coeff", 0.0))
        self.teacher_model: PreTrainedModel | PeftModel | None = None
        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        self._build_teacher_model()

    def _build_teacher_model(self):
        teacher_config = self.config.get("teacher", {})
        if not teacher_config.get("enable", False):
            raise ValueError("FSDPSFTKDTrainer requires teacher.enable=true.")

        teacher_model_path = _none_if_null(teacher_config.get("model_path", None))
        if teacher_model_path is None:
            raise ValueError("teacher.model_path must be set for KD training.")

        external_lib = _none_if_null(teacher_config.get("external_lib", None))
        if external_lib is not None:
            importlib.import_module(external_lib)

        local_teacher_path = copy_to_local(src=teacher_model_path, verbose=True)
        trust_remote_code = bool(teacher_config.get("trust_remote_code", self.config.model.trust_remote_code))
        teacher_hf_config = AutoConfig.from_pretrained(local_teacher_path, trust_remote_code=trust_remote_code)
        torch_dtype = _dtype_from_name(teacher_config.get("torch_dtype", "bfloat16"))

        kwargs = {
            "config": teacher_hf_config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        attn_implementation = _none_if_null(teacher_config.get("attn_implementation", "flash_attention_2"))
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        if self.device_mesh.get_rank() == 0:
            print(f"Loading KD teacher model from {teacher_model_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(local_teacher_path, **kwargs)

        teacher_lora_path = _none_if_null(teacher_config.get("lora_adapter_path", None))
        if teacher_lora_path is not None:
            if self.device_mesh.get_rank() == 0:
                print(f"Loading KD teacher LoRA adapter from {teacher_lora_path}")
            teacher_model = PeftModel.from_pretrained(teacher_model, teacher_lora_path, is_trainable=False)

        teacher_model.to(self.device_name)
        teacher_model.eval()
        teacher_model.config.use_cache = False
        for param in teacher_model.parameters():
            param.requires_grad_(False)

        self.teacher_model = teacher_model
        log_gpu_memory_usage("After KD teacher allocation", logger=logger)

    def _compute_loss_terms(self, batch):
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if use_sp:
            raise NotImplementedError("KD SFT trainer currently supports ulysses_sequence_parallel_size=1 only.")

        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            student_outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            student_logits = student_outputs.logits[..., :-1, :].contiguous()

            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()

        if student_logits.size(-1) != teacher_logits.size(-1):
            raise ValueError(
                "Student and teacher vocab sizes differ: "
                f"student={student_logits.size(-1)}, teacher={teacher_logits.size(-1)}. "
                "Token-level KD requires compatible tokenizers/vocabularies."
            )

        shift_labels = input_ids[:, 1:].contiguous().reshape(-1).to(student_logits.device)
        flat_student_logits = student_logits.view(-1, student_logits.size(-1))
        flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

        token_sft_loss = loss_fct(flat_student_logits.float(), shift_labels)
        loss_mask = loss_mask.to(token_sft_loss.device)
        valid_token_this_rank = torch.sum(loss_mask)
        sft_loss = torch.sum(token_sft_loss * loss_mask) / (valid_token_this_rank + 1e-8)

        token_kd_loss = kd_divergence_token_loss(
            flat_student_logits,
            flat_teacher_logits.detach(),
            loss_type=self.kd_loss_type,
            temperature=self.kd_temperature,
            jsd_beta=self.kd_jsd_beta,
            skew_alpha=self.kd_skew_alpha,
        )
        kd_loss = torch.sum(token_kd_loss * loss_mask) / (valid_token_this_rank + 1e-8)

        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)
            dp_size = torch.distributed.get_world_size()
            sft_loss = sft_loss * dp_size
            kd_loss = kd_loss * dp_size

        if self.kd_balancing_coeff != 0.0:
            coeff = self.kd_balancing_coeff
            loss = (1.0 - coeff) * sft_loss + coeff * kd_loss
        elif self.kd_coeff != 0.0:
            loss = sft_loss + self.kd_coeff * kd_loss
        else:
            loss = kd_loss

        return loss, sft_loss.detach(), kd_loss.detach()

    def _compute_loss_and_backward(self, batch, do_backward=True):
        loss, sft_loss, kd_loss = self._compute_loss_terms(batch)
        if do_backward:
            loss.backward()
        return loss, sft_loss, kd_loss

    def training_step(self, batch):
        self.fsdp_model.train()
        self.teacher_model.eval()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        step_sft_loss = 0
        step_kd_loss = 0
        for micro_batch in micro_batches:
            loss, sft_loss, kd_loss = self._compute_loss_terms(batch=micro_batch)
            (loss / n_micro_batches).backward()
            step_loss += loss.detach().item() / n_micro_batches
            step_sft_loss += sft_loss.item() / n_micro_batches
            step_kd_loss += kd_loss.item() / n_micro_batches

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.trainable_parameters, max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        log_gpu_memory_usage("After offload weights", logger=logger)

        metrics = {
            "train/loss": torch.tensor(step_loss, device=self.device_name),
            "train/sft_loss": torch.tensor(step_sft_loss, device=self.device_name),
            "train/kd_loss": torch.tensor(step_kd_loss, device=self.device_name),
        }
        for value in metrics.values():
            if is_cuda_available:
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(value)
                value /= self.ulysses_device_mesh.size(0)

        return {
            "train/loss": metrics["train/loss"].detach().item(),
            "train/sft_loss": metrics["train/sft_loss"].detach().item(),
            "train/kd_loss": metrics["train/kd_loss"].detach().item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    def validation_step(self, batch):
        self.fsdp_model.eval()
        self.teacher_model.eval()
        with torch.no_grad():
            loss, sft_loss, kd_loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(sft_loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(kd_loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                torch.distributed.all_reduce(sft_loss)
                torch.distributed.all_reduce(kd_loss)
                denom = self.ulysses_device_mesh.size(0)
                loss /= denom
                sft_loss /= denom
                kd_loss /= denom
        return loss, sft_loss, kd_loss

    def _validate_and_log(self, tracking, step):
        rank = self.device_mesh.get_rank()
        val_losses = []
        val_sft_losses = []
        val_kd_losses = []
        for data in self.val_dataloader:
            val_data = self._to_tensordict(data)
            val_loss, val_sft_loss, val_kd_loss = self.validation_step(val_data)
            val_losses.append(val_loss)
            val_sft_losses.append(val_sft_loss)
            val_kd_losses.append(val_kd_loss)

        if rank == 0:
            if val_losses:
                metric = {
                    "val/loss": torch.mean(torch.stack(val_losses)).detach().item(),
                    "val/sft_loss": torch.mean(torch.stack(val_sft_losses)).detach().item(),
                    "val/kd_loss": torch.mean(torch.stack(val_kd_losses)).detach().item(),
                }
                tracking.log(data=metric, step=step)
            else:
                print("Skip validation because the validation dataloader is empty.")
        torch.distributed.barrier()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTKDTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
