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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopSubgoalStateTracker:
    CONTROL_CLICKS = {
        "search",
        "back to search",
        "next >",
        "< prev",
        "description",
        "features",
        "reviews",
        "attributes",
        "buy now",
    }
    TASK_TOKEN_STOPWORDS = {
        "and", "are", "for", "from", "item", "less", "like", "looking", "lower",
        "need", "than", "the", "this", "want", "with", "would",
    }

    def __init__(self, task_description: str):
        self.task_description = task_description
        self.task_text = task_description.lower()
        self.current_query = None
        self.queries_tried = []
        self.inspected_product = None
        self.selected_options = {}
        self.checked_detail_pages = set()
        self.visited_products = set()

    @staticmethod
    def _action_arg(action: str, name: str) -> str:
        action = str(action or "").strip()
        prefix = f"{name}["
        if action.lower().startswith(prefix) and action.endswith("]"):
            return action[len(prefix):-1].strip()
        return ""

    @staticmethod
    def _is_asin(text: str) -> bool:
        return bool(re.fullmatch(r"b[0-9a-z]{9}", str(text or "").strip().lower()))

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _join_or_none(values) -> str:
        values = list(values)
        return ", ".join(str(value) for value in values) if values else "none"

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @classmethod
    def _clean_observation_token(cls, text: str) -> str:
        token = cls._clean_text(text)
        if token.endswith("."):
            token = token[:-1].rstrip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            token = token[1:-1].strip()
        return token

    @classmethod
    def _observation_tokens(cls, current_observation: str) -> List[str]:
        return [
            token
            for token in (cls._clean_observation_token(part) for part in str(current_observation or "").split(" [SEP] "))
            if token
        ]

    @classmethod
    def _observation_field(cls, current_observation: str, label: str, next_labels: List[str]) -> str:
        text = str(current_observation or "")
        for line in text.splitlines():
            if line.lower().startswith(label.lower() + ":"):
                return cls._clean_text(line.split(":", 1)[1])
        next_label_pattern = "|".join(re.escape(next_label) for next_label in next_labels)
        pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s+(?:{next_label_pattern}):|$|')"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return cls._clean_text(match.group(1)) if match else ""

    def _structured_option_groups(self, current_observation: str, available_actions: List[str]) -> List[Tuple[str, List[str]]]:
        candidate_norms = {self._norm(option) for option in self._candidate_options(available_actions)}
        groups = []
        in_options = False
        for line in str(current_observation or "").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("available options:"):
                in_options = True
                continue
            if not in_options:
                continue
            if not stripped.startswith("- ") or ":" not in stripped:
                break
            category, raw_values = stripped[2:].split(":", 1)
            values = [value.strip() for value in raw_values.split(", ") if value.strip()]
            if not candidate_norms or any(self._norm(value) in candidate_norms for value in values):
                groups.append((category.strip(), values))
        return groups

    def _raw_option_groups(self, current_observation: str, available_actions: List[str]) -> List[Tuple[str, List[str]]]:
        tokens = self._observation_tokens(current_observation)
        candidate_by_norm = {self._norm(option): option for option in self._candidate_options(available_actions)}
        if not tokens or not candidate_by_norm:
            return []

        groups = []
        control = self.CONTROL_CLICKS | {"rating", "n.a.", "page"}
        for idx, token in enumerate(tokens[:-1]):
            token_norm = self._norm(token)
            next_norm = self._norm(tokens[idx + 1])
            if next_norm not in candidate_by_norm:
                continue
            if token_norm in candidate_by_norm or token_norm in control or self._is_asin(token_norm):
                continue
            if token_norm.startswith("price:") or token_norm.startswith("$") or token_norm.startswith("page "):
                continue

            values = []
            cursor = idx + 1
            while cursor < len(tokens):
                value_norm = self._norm(tokens[cursor])
                if value_norm not in candidate_by_norm:
                    break
                values.append(candidate_by_norm[value_norm])
                cursor += 1
            if values:
                groups.append((token, values))
        return groups

    def _option_groups(self, current_observation: str, available_actions: List[str]) -> List[Tuple[str, List[str]]]:
        groups = self._structured_option_groups(current_observation, available_actions)
        return groups if groups else self._raw_option_groups(current_observation, available_actions)

    def _option_categories(self, current_observation: str, available_actions: List[str]) -> List[str]:
        return [category for category, _ in self._option_groups(current_observation, available_actions)]

    def _option_category_for_value(
        self,
        value: str,
        current_observation: str,
        available_actions: List[str],
    ) -> str:
        value_norm = self._norm(value)
        for category, values in self._option_groups(current_observation, available_actions):
            if any(self._norm(option_value) == value_norm for option_value in values):
                return category
        return "option"

    def _selected_options_text(self) -> str:
        if not self.selected_options:
            return "none"
        return ", ".join(f"{category}: {value}" for category, value in self.selected_options.items())

    def update(self, previous_action: str, current_observation: str, available_actions: List[str]):
        action = str(previous_action or "").strip()
        search_arg = self._action_arg(action, "search")
        click_arg = self._action_arg(action, "click")

        if search_arg:
            self.current_query = search_arg
            if search_arg not in self.queries_tried:
                self.queries_tried.append(search_arg)
            self.inspected_product = None
        elif click_arg:
            click_norm = self._norm(click_arg)
            if self._is_asin(click_norm):
                self.inspected_product = click_norm.upper()
                self.visited_products.add(self.inspected_product)
            elif click_norm in {"description", "features", "reviews", "attributes"}:
                self.checked_detail_pages.add(click_norm)
            elif click_norm == "back to search":
                self.inspected_product = None
            elif click_norm not in self.CONTROL_CLICKS:
                category = self._option_category_for_value(click_arg, current_observation, available_actions)
                self.selected_options[category] = click_arg

        obs_lower = str(current_observation or "").lower()
        for page_name in ("description", "features", "reviews", "attributes"):
            if f"'{page_name}'" in obs_lower:
                self.checked_detail_pages.add(page_name)

    def _candidate_options(self, available_actions: List[str]) -> List[str]:
        candidates = []
        for action in available_actions:
            click_arg = self._action_arg(action, "click")
            click_norm = self._norm(click_arg)
            if not click_arg or click_norm in self.CONTROL_CLICKS or self._is_asin(click_norm):
                continue
            candidates.append(click_arg)
        return candidates

    def _remaining_option_categories(self, current_observation: str, available_actions: List[str]) -> List[str]:
        selected_categories = {self._norm(category) for category in self.selected_options}
        return [
            category
            for category in self._option_categories(current_observation, available_actions)
            if self._norm(category) not in selected_categories
        ]

    def _phase(self, available_actions: List[str]) -> str:
        action_set = {self._norm(action) for action in available_actions}
        has_search = any(action.startswith("search[") for action in action_set)
        has_buy = "click[buy now]" in action_set
        has_product = any(self._is_asin(self._action_arg(action, "click")) for action in available_actions)
        has_options = bool(self._candidate_options(available_actions))

        if has_search:
            return "formulate_query"
        if has_product and not has_buy:
            return "browse_results"
        if has_buy and has_options:
            return "select_options"
        if has_buy:
            return "purchase"
        return "evaluate_item"

    def _current_item(self, current_observation: str) -> str:
        next_labels = ["Price", "Category", "Rating", "Selected options", "Available options", "Options"]
        for label in ("Title", "Product"):
            value = self._observation_field(current_observation, label, next_labels)
            if value and not self._is_asin(value):
                return value
        tokens = self._observation_tokens(current_observation)
        for idx, token in enumerate(tokens):
            token_norm = self._norm(token)
            if token_norm.startswith("price:") or token_norm.startswith("$") or token_norm.startswith("rating"):
                for candidate in reversed(tokens[:idx]):
                    candidate_norm = self._norm(candidate)
                    if (
                        candidate_norm not in self.CONTROL_CLICKS
                        and not self._is_asin(candidate_norm)
                        and not candidate_norm.startswith("page ")
                    ):
                        return candidate
        return self.inspected_product or "none"

    def _current_price(self, current_observation: str) -> str:
        next_labels = ["Category", "Rating", "Selected options", "Available options", "Options", "Features", "Description"]
        value = self._observation_field(current_observation, "Price", next_labels)
        if value:
            return value
        for token in self._observation_tokens(current_observation):
            token_norm = self._norm(token)
            if token_norm.startswith("price:"):
                return self._clean_text(token.split(":", 1)[1])
            if token_norm.startswith("$"):
                return token
        return "unknown"

    def render(self, current_observation: str, available_actions: List[str]) -> str:
        phase = self._phase(available_actions)
        remaining_options = self._remaining_option_categories(current_observation, available_actions)
        if phase == "select_options" and not remaining_options:
            phase = "purchase"
        purchase_ready = "click[buy now]" in {self._norm(action) for action in available_actions} and not remaining_options
        option_candidates = self._option_categories(current_observation, available_actions)
        lines = [
            "<state>",
            f"current_phase: {phase}",
        ]
        if phase == "formulate_query":
            lines.extend(
                [
                    f"queries_tried: {self._join_or_none(self.queries_tried)}",
                    f"items_inspected: {self._join_or_none(sorted(self.visited_products))}",
                ]
            )
        elif phase == "browse_results":
            lines.extend(
                [
                    f"current_query: {self.current_query or 'none'}",
                    f"items_already_inspected: {self._join_or_none(sorted(self.visited_products))}",
                ]
            )
        elif phase == "evaluate_item":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_available: {self._join_or_none(option_candidates)}",
                    f"features_checked: {str('features' in self.checked_detail_pages).lower()}",
                    f"description_checked: {str('description' in self.checked_detail_pages).lower()}",
                ]
            )
        elif phase == "select_options":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                    f"options_remaining: {self._join_or_none(remaining_options)}",
                ]
            )
        elif phase == "purchase":
            lines.extend(
                [
                    f"current_item: {self._current_item(current_observation)}",
                    f"price: {self._current_price(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                    f"all_options_filled: {str(purchase_ready).lower()}",
                ]
            )
        else:
            lines.extend(
                [
                    f"current_query: {self.current_query or 'none'}",
                    f"current_item: {self._current_item(current_observation)}",
                    f"options_selected: {self._selected_options_text()}",
                ]
            )
        lines.append("</state>")
        return "\n".join(lines)


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.prompt_style = str(config.env.webshop.get("prompt_style", "tagged")).lower()
        self.use_subgoal_state_prompt = self.prompt_style.endswith("_state")
        super().__init__(envs, projection_f, config)

    def set_active_env_num(self, active_env_num: int):
        if hasattr(self.envs, "set_active_env_num"):
            self.envs.set_active_env_num(active_env_num)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        self.memory.reset(batch_size=len(infos))
        self.subgoal_trackers = [WebshopSubgoalStateTracker(task) for task in self.tasks]
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, prompt_style=self.prompt_style)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)
        for i, tracker in enumerate(self.subgoal_trackers):
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            tracker.update(actions[i], next_obs[i], available_actions)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if self.prompt_style == "react":
            actions.append("think[<your reasoning>]")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            use_history = not self.use_subgoal_state_prompt and not init and self.config.env.history_length > 0
            template = get_webshop_prompt_template(self.prompt_style, use_history=use_history)
            prompt_kwargs = {
                "task_description": self.tasks[i],
                "current_observation": text_obs[i],
                "available_actions": reformatted_available_actions,
                "example": get_webshop_prompt_example(self.prompt_style),
            }
            if self.use_subgoal_state_prompt:
                tracker = self.subgoal_trackers[i]
                one_step_context = "none"
                if len(self.memory[i]) > 0:
                    last_record = self.memory[i][-1]
                    one_step_context = (
                        f"Previous observation: {last_record['text_obs']}\n"
                        f"Previous action: {last_record['action']}"
                    )
                prompt_kwargs.update(
                    {
                        "subgoal_state_block": tracker.render(text_obs[i], available_actions),
                        "one_step_context": one_step_context,
                    }
                )

            if not use_history:
                obs = template.format(
                    **prompt_kwargs,
                )
            else:
                obs = template.format(
                    **prompt_kwargs,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                )

                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    fallback_template = get_webshop_prompt_template(self.prompt_style, use_history=False)
                    obs = fallback_template.format(
                        **prompt_kwargs,
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path,
                    'val_goal_start': config.env.webshop.get('val_goal_start', 0),
                    'val_goal_end': config.env.webshop.get('val_goal_end', 500),
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
