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

from typing import List
import re

def _extract_raw_action(text: str, allow_think: bool):
    candidates = ["search", "click"]
    if allow_think:
        candidates.append("think")

    # Prefer the first explicit action-looking span. This tolerates models that
    # prepend labels such as "Action:" despite being asked to output only action.
    pattern = re.compile(rf"({'|'.join(candidates)})\[[^\]]+\]", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(0).strip().lower()
    return text.strip().lower()


def webshop_projection(actions: List[str], prompt_style: str = "tagged"):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format depends on prompt_style:
        tagged: <think>...</think><action>search[...] or click[...]</action>
        direct/act and *_state variants: search[...] or click[...]
        react: think[...], search[...], or click[...]
    """

    valids = [0] * len(actions)
    prompt_style = str(prompt_style).lower()
    base_prompt_style = prompt_style[:-6] if prompt_style.endswith("_state") else prompt_style
    raw_action_styles = {"direct", "react", "act"}

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        if base_prompt_style in raw_action_styles:
            extracted_action = _extract_raw_action(original_str, allow_think=(base_prompt_style == "react"))
            actions[i] = extracted_action
            allowed_prefixes = ("search[", "click[")
            if base_prompt_style == "react":
                allowed_prefixes = ("think[",) + allowed_prefixes
            valids[i] = int(extracted_action.startswith(allowed_prefixes) and extracted_action.endswith("]"))

            if re.search(r'[\u4e00-\u9fff]', original_str):
                valids[i] = 0
            continue

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
