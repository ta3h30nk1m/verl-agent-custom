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
"""FSDP SFT trainer with Beta-weighted teacher distillation."""

from __future__ import annotations

import logging

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.fsdp_sft_kd_trainer import FSDPSFTKDTrainer
from verl.trainer.fsdp_sft_trainer import create_sft_dataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup


logger = logging.getLogger(__file__)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


class EqualWeighting(nn.Module):
    def forward(self, losses, features=None):
        means = [loss.mean() for loss in losses]
        total = torch.stack(means).sum() if means else None
        weights = [1.0 for _ in means]
        return total, weights


class TaskWeighting(nn.Module):
    """Learn one log-variance per weighted loss."""

    def __init__(self, num_losses):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))

    def forward(self, losses, features=None):
        total = 0.0
        weights = []
        for idx, loss in enumerate(losses):
            log_var = self.log_vars[idx].clamp(-5.0, 5.0)
            precision = torch.exp(-log_var)
            total = total + precision * loss.float().mean() + log_var
            weights.append(float(precision.detach().cpu()))
        return total, weights


class InstanceWeighting(nn.Module):
    """Predict one log-variance per sample and per loss from summary features."""

    def __init__(self, num_losses, feature_dim=4, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_losses),
        )

    def forward(self, losses, features=None):
        if features is None:
            return EqualWeighting().forward(losses)
        log_vars = self.mlp(features.detach().to(self.mlp[0].weight.dtype)).clamp(-5.0, 5.0)
        loss_mat = torch.stack([loss.float() for loss in losses], dim=-1)
        precision = torch.exp(-log_vars.float())
        total = (precision * loss_mat + log_vars).sum(dim=-1).mean()
        weights = precision.mean(dim=0).detach().cpu().tolist()
        return total, weights


class FSDPSFTBetaKDTrainer(FSDPSFTKDTrainer):
    LOSS_ALIASES = {
        "kl": "fkl",
        "fkl": "fkl",
        "forward_kl": "fkl",
        "forwardkl": "fkl",
        "rkl": "rkl",
        "reverse_kl": "rkl",
        "reversekl": "rkl",
        "js": "js",
        "jsd": "js",
        "tvd": "tvd",
        "mse": "mse_probs",
        "mse_probs": "mse_probs",
        "mse-probs": "mse_probs",
        "mse_logits": "mse_logits",
        "mse-logits": "mse_logits",
        "cos": "cosine_probs",
        "cosine": "cosine_probs",
        "cosine_probs": "cosine_probs",
        "cosine-probs": "cosine_probs",
        "cosine_logits": "cosine_logits",
        "cosine-logits": "cosine_logits",
    }

    def __init__(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset):
        beta_config = config.get("beta_kd", {})
        self.beta_kd_losses = self._parse_loss_names(beta_config.get("losses", "fkl"))
        self.beta_kd_temperature = float(beta_config.get("temperature", 1.0))
        self.beta_kd_weight_ce = _as_bool(beta_config.get("weight_ce", False))
        self.beta_kd_mode = str(beta_config.get("weighting", "instance")).lower()
        self.beta_kd_hidden_dim = int(beta_config.get("hidden_dim", 16))
        self.beta_kd_weighting = None
        self._beta_kd_last_weights = {}
        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        self._build_beta_weighting()

    def _build_beta_weighting(self):
        num_weighted = len(self.beta_kd_losses) + int(self.beta_kd_weight_ce)
        if num_weighted <= 0:
            return

        if self.beta_kd_mode in {"equal", "type1"}:
            weighting = EqualWeighting()
        elif self.beta_kd_mode in {"instance", "type3"}:
            weighting = InstanceWeighting(num_weighted, feature_dim=4, hidden_dim=self.beta_kd_hidden_dim)
        else:
            weighting = TaskWeighting(num_weighted)

        weighting = weighting.to(self.device_name)
        self.beta_kd_weighting = weighting

        weighting_params = [param for param in weighting.parameters() if param.requires_grad]
        if weighting_params:
            self.optimizer.add_param_group({"params": weighting_params, "weight_decay": 0.0})
            self.trainable_parameters.extend(weighting_params)
            self._rebuild_lr_scheduler()

        if self.device_mesh.get_rank() == 0:
            print(
                "Using BetaKD: "
                f"losses={self.beta_kd_losses}, weighting={self.beta_kd_mode}, "
                f"weight_ce={self.beta_kd_weight_ce}, trainable_weight_params={sum(p.numel() for p in weighting_params)}"
            )

    def _rebuild_lr_scheduler(self):
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        lr_scheduler = self.config.optim.get("lr_scheduler", "cosine")
        if lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif lr_scheduler == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {lr_scheduler}")

    def _parse_loss_names(self, spec):
        if spec is None:
            return ["fkl"]
        if isinstance(spec, str):
            parts = [part.strip() for part in spec.replace("+", ",").split(",") if part.strip()]
        else:
            parts = list(spec)

        names = []
        for name in parts:
            key = str(name).strip().lower().replace(" ", "_")
            key = self.LOSS_ALIASES.get(key, key)
            if key not in self.LOSS_ALIASES.values():
                raise ValueError(f"Unsupported beta_kd loss: {name}")
            names.append(key)
        return names

    @staticmethod
    def _masked_mean(token_loss, mask):
        mask = mask.float()
        denom = mask.sum(dim=-1)
        loss = (token_loss * mask).sum(dim=-1) / denom.clamp_min(1.0)
        return torch.where(denom > 0, loss, loss.new_zeros(loss.shape))

    @staticmethod
    def _standardize(x, eps=1e-6):
        x = x.float()
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(eps)

    def _kd_per_sample(self, name, student_logits, teacher_logits, mask):
        temp = self.beta_kd_temperature
        s = student_logits / temp
        t = teacher_logits / temp
        log_q = F.log_softmax(s, dim=-1, dtype=torch.float32)
        log_p = F.log_softmax(t, dim=-1, dtype=torch.float32)
        q = log_q.exp()
        p = log_p.exp()

        if name == "fkl":
            token_loss = -(p * log_q).sum(dim=-1) * (temp**2)
        elif name == "rkl":
            token_loss = (q * (log_q - log_p)).sum(dim=-1) * (temp**2)
        elif name == "js":
            m = 0.5 * (p + q)
            log_m = m.clamp_min(1e-8).log()
            token_loss = 0.5 * (p * (log_p - log_m)).sum(dim=-1)
            token_loss = token_loss + 0.5 * (q * (log_q - log_m)).sum(dim=-1)
            token_loss = token_loss * (temp**2)
        elif name == "tvd":
            token_loss = 0.5 * (p - q).abs().sum(dim=-1)
        elif name == "mse_probs":
            token_loss = (p - q).pow(2).mean(dim=-1) * (temp**2)
        elif name == "mse_logits":
            token_loss = (self._standardize(s) - self._standardize(t)).pow(2).mean(dim=-1) * (temp**2)
        elif name == "cosine_probs":
            token_loss = (1.0 - F.cosine_similarity(q, p, dim=-1)) * (temp**2)
        elif name == "cosine_logits":
            token_loss = (1.0 - F.cosine_similarity(self._standardize(s), self._standardize(t), dim=-1)) * (temp**2)
        else:
            raise ValueError(f"Unknown beta_kd loss: {name}")

        return self._masked_mean(token_loss, mask)

    def _instance_features(self, student_logits, teacher_logits, mask):
        temp = self.beta_kd_temperature
        q = F.softmax(student_logits / temp, dim=-1, dtype=torch.float32)
        p = F.softmax(teacher_logits / temp, dim=-1, dtype=torch.float32)

        s_ent = -(q * q.clamp_min(1e-8).log()).sum(dim=-1)
        t_ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
        s_conf = q.max(dim=-1).values
        t_conf = p.max(dim=-1).values

        return torch.stack(
            [
                self._masked_mean(s_ent, mask),
                self._masked_mean(t_ent, mask),
                self._masked_mean(s_conf, mask),
                self._masked_mean(t_conf, mask),
            ],
            dim=-1,
        )

    def _compute_loss_terms(self, batch):
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if use_sp:
            raise NotImplementedError("BetaKD SFT trainer currently supports ulysses_sequence_parallel_size=1 only.")

        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, :-1].to(self.device_name)
        mask = loss_mask > 0

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
                "BetaKD requires compatible tokenizers/vocabularies."
            )

        labels = input_ids[:, 1:].contiguous()
        safe_labels = labels.masked_fill(~mask, 0)
        log_q = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        nll = -log_q.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        ce_per_sample = self._masked_mean(nll, loss_mask)
        ce_mean = ce_per_sample.mean()

        kd_losses = [self._kd_per_sample(name, student_logits, teacher_logits.detach(), loss_mask) for name in self.beta_kd_losses]
        kd_metric = torch.stack([loss.mean() for loss in kd_losses]).sum() if kd_losses else ce_mean.new_zeros(())

        total = ce_mean.new_zeros(())
        weighted_losses = []
        weighted_names = []
        if self.beta_kd_weight_ce:
            weighted_losses.append(ce_per_sample)
            weighted_names.append("ce")
        else:
            total = total + ce_mean

        weighted_losses.extend(kd_losses)
        weighted_names.extend(self.beta_kd_losses)

        if not weighted_losses or self.beta_kd_weighting is None:
            loss = ce_mean + kd_metric
            self._beta_kd_last_weights = {}
        else:
            features = self._instance_features(student_logits, teacher_logits.detach(), loss_mask) if self.beta_kd_mode in {"instance", "type3"} else None
            weighted_total, weights = self.beta_kd_weighting(weighted_losses, features)
            loss = total + weighted_total
            self._beta_kd_last_weights = {name: float(weight) for name, weight in zip(weighted_names, weights)}

        return loss, ce_mean.detach(), kd_metric.detach()

    def _sync_beta_weighting_grads(self):
        if self.beta_kd_weighting is None or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        for param in self.beta_kd_weighting.parameters():
            if param.grad is None:
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)

    def training_step(self, batch):
        self.fsdp_model.train()
        self.teacher_model.eval()
        if self.beta_kd_weighting is not None:
            self.beta_kd_weighting.train()

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

        self._sync_beta_weighting_grads()

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

        metrics = {
            "train/loss": metrics["train/loss"].detach().item(),
            "train/sft_loss": metrics["train/sft_loss"].detach().item(),
            "train/kd_loss": metrics["train/kd_loss"].detach().item(),
            "train/lr(1e-3)": lr * 1e3,
        }
        for name, weight in self._beta_kd_last_weights.items():
            metrics[f"train/betakd_weight/{name}"] = weight
        return metrics


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

    trainer = FSDPSFTBetaKDTrainer(
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
