# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
"""
Run WebShop validation from a saved verl actor checkpoint.

This intentionally reuses RayPPOTrainer._validate() so the metric computation
matches training-time validation, but skips the PPO training loop.
"""

import os
import time
from functools import partial
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.fs import copy_to_local


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_webshop_validation(config)


def run_webshop_validation(config) -> None:
    if "webshop" not in config.env.env_name.lower():
        raise ValueError("main_webshop_validation only supports env.env_name=Webshop")

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = OmegaConf.to_container(config.get("ray_init", {}), resolve=True)
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "include_dashboard": False, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    metrics = WebshopValidationRunner().run(config)
    pprint(metrics)


class WebshopValidationRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[webshop-val] resolving model path", flush=True)
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments.env_manager import WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        if config.actor_rollout_ref.actor.strategy not in ["fsdp", "fsdp2"]:
            raise NotImplementedError("WebShop validation runner currently supports fsdp/fsdp2 checkpoints.")
        if config.algorithm.adv_estimator == "gae":
            raise ValueError("WebShop validation does not need a critic. Set algorithm.adv_estimator=grpo or gigpo.")

        trust_remote_code = config.data.get("trust_remote_code", False)
        print(f"[webshop-val] loading tokenizer/processor from {local_path}", flush=True)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        print("[webshop-val] tokenizer/processor loaded", flush=True)

        print("[webshop-val] creating WebShop validation envs", flush=True)
        val_envs = self._make_webshop_val_envs(config)
        print("[webshop-val] WebShop validation envs created", flush=True)
        time.sleep(config.data.val_batch_size * 0.1)

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id}
        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}

        reward_manager_cls = EpisodeRewardManager
        print("[webshop-val] creating reward manager and datasets", flush=True)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        print("[webshop-val] building RayPPOTrainer", flush=True)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=None,
            val_envs=val_envs,
        )
        print("[webshop-val] initializing actor/rollout workers", flush=True)
        trainer.init_workers()
        print("[webshop-val] actor/rollout workers initialized", flush=True)
        trainer.global_steps = 0
        trainer._load_checkpoint()
        print("[webshop-val] starting validation rollout", flush=True)
        metrics = trainer._validate()
        assert metrics, f"{metrics=}"
        return metrics

    @staticmethod
    def _make_webshop_val_envs(config):
        from agent_system.environments.env_manager import WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json")
            attr_path = os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json")
        else:
            file_path = os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle.json")
            attr_path = os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/webshop/webshop/data/items_ins_v2.json")

        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": config.env.webshop.human_goals,
            "file_path": os.path.abspath(file_path),
            "attr_path": os.path.abspath(attr_path),
            "val_goal_start": config.env.webshop.get("val_goal_start", 0),
            "val_goal_end": config.env.webshop.get("val_goal_end", 500),
        }
        resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
        raw_val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        return WebshopEnvironmentManager(raw_val_envs, partial(webshop_projection), config)


if __name__ == "__main__":
    main()
