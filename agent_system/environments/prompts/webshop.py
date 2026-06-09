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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_DIRECT_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Choose the single best next action. Respond with only one admissible action string, such as search[...] or click[...]. Do not include reasoning, tags, explanations, or extra text.
"""

WEBSHOP_DIRECT_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Choose the single best next action. Respond with only one admissible action string, such as search[...] or click[...]. Do not include reasoning, tags, explanations, or extra text.
"""

WEBSHOP_REACT_EXAMPLE = """
Example episode:
Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.

Observation: 'Search'.
Admissible actions:
[
'think[<your reasoning>]',
'search[<your query>]',
'click[search]',
]
Answer: search[3 ounce bright citrus deodorant sensitive skin]

Observation: 'Back to Search' [SEP] 'Page 1 (Total results: 50)' [SEP] 'Next >' [SEP] 'B078GWRC1J' [SEP] 'Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] '$10.99' [SEP] 'B078GTKVXY' [SEP] 'Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] '$10.99' [SEP] 'B08KBVJ4XN' [SEP] 'Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)' [SEP] '$15.95'.
Admissible actions:
[
'think[<your reasoning>]',
'click[back to search]',
'click[next >]',
'click[b078gwrc1j]',
'click[b078gtkvxy]',
'click[b08kbvj4xn]',
]
Answer: think[B078GWRC1J is a bright citrus deodorant under 50 dollars. I should inspect it first.]

Observation: OK.
Admissible actions:
[
'think[<your reasoning>]',
'click[back to search]',
'click[next >]',
'click[b078gwrc1j]',
'click[b078gtkvxy]',
'click[b08kbvj4xn]',
]
Answer: click[b078gwrc1j]

Observation: 'Back to Search' [SEP] '< Prev' [SEP] 'scent' [SEP] 'assorted scents' [SEP] 'bright citrus' [SEP] 'calming lavender' [SEP] 'ginger fresh' [SEP] 'simply non-scents' [SEP] 'size' [SEP] 'travel set (4-pack)' [SEP] '3 ounce (pack of 1)' [SEP] '3-ounce (2-pack)' [SEP] 'Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] 'Price: $10.99' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Reviews' [SEP] 'Buy Now'.
Admissible actions:
[
'think[<your reasoning>]',
'click[back to search]',
'click[< prev]',
'click[description]',
'click[features]',
'click[reviews]',
'click[buy now]',
'click[bright citrus]',
'click[3 ounce (pack of 1)]',
]
Answer: think[This item matches sensitive skin, bright citrus, and the price limit. I should select bright citrus and the 3 ounce pack before buying.]

Observation: OK.
Answer: click[bright citrus]
Observation: You have clicked bright citrus.
Answer: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Answer: click[buy now]
"""

WEBSHOP_REACT_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
At each step, answer with exactly one action string.
You may answer with think[...] to reason. When you are ready to act, answer with one admissible shopping action: search[...] or click[...].

{example}

Current task: {task_description}.
Current observation: {current_observation}.
Current admissible actions:
[
{available_actions}
].

Answer with exactly one action string.
"""

WEBSHOP_REACT_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
At each step, answer with exactly one action string.
You may answer with think[...] to reason. When you are ready to act, answer with one admissible shopping action: search[...] or click[...].

{example}

Current task: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step}.
Current observation: {current_observation}.
Current admissible actions:
[
{available_actions}
].

Answer with exactly one action string.
"""

WEBSHOP_ACT_EXAMPLE = """
Example episode:
Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.

Observation: 'Search'.
Admissible actions:
[
'search[<your query>]',
'click[search]',
]
Answer: search[3 ounce bright citrus deodorant sensitive skin]

Observation: 'Back to Search' [SEP] 'Page 1 (Total results: 50)' [SEP] 'Next >' [SEP] 'B078GWRC1J' [SEP] 'Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] '$10.99' [SEP] 'B078GTKVXY' [SEP] 'Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] '$10.99' [SEP] 'B08KBVJ4XN' [SEP] 'Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)' [SEP] '$15.95'.
Admissible actions:
[
'click[back to search]',
'click[next >]',
'click[b078gwrc1j]',
'click[b078gtkvxy]',
'click[b08kbvj4xn]',
]
Answer: click[b078gwrc1j]

Observation: 'Back to Search' [SEP] '< Prev' [SEP] 'scent' [SEP] 'assorted scents' [SEP] 'bright citrus' [SEP] 'calming lavender' [SEP] 'ginger fresh' [SEP] 'simply non-scents' [SEP] 'size' [SEP] 'travel set (4-pack)' [SEP] '3 ounce (pack of 1)' [SEP] '3-ounce (2-pack)' [SEP] 'Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce' [SEP] 'Price: $10.99' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Reviews' [SEP] 'Buy Now'.
Admissible actions:
[
'click[back to search]',
'click[< prev]',
'click[description]',
'click[features]',
'click[reviews]',
'click[buy now]',
'click[bright citrus]',
'click[3 ounce (pack of 1)]',
]
Answer: click[bright citrus]
Observation: You have clicked bright citrus.
Answer: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Answer: click[buy now]
"""

WEBSHOP_ACT_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
At each step, answer with exactly one admissible action string: search[...] or click[...].

{example}

Current task: {task_description}.
Current observation: {current_observation}.
Current admissible actions:
[
{available_actions}
].

Answer with exactly one action string.
"""

WEBSHOP_ACT_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
At each step, answer with exactly one admissible action string: search[...] or click[...].

{example}

Current task: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step}.
Current observation: {current_observation}.
Current admissible actions:
[
{available_actions}
].

Answer with exactly one action string.
"""

WEBSHOP_ACT_STATE_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
At each step, answer with exactly one admissible action string: search[...] or click[...].

Current task: {task_description}.

Subgoal state block:
{subgoal_state_block}

One-step context:
{one_step_context}

Current observation: {current_observation}.
Current admissible actions:
[
{available_actions}
].

Answer with exactly one action string.
"""

WEBSHOP_PROMPT_TEMPLATES = {
    "tagged": {
        "no_history": WEBSHOP_TEMPLATE_NO_HIS,
        "history": WEBSHOP_TEMPLATE,
        "example": "",
    },
    "direct": {
        "no_history": WEBSHOP_DIRECT_TEMPLATE_NO_HIS,
        "history": WEBSHOP_DIRECT_TEMPLATE,
        "example": "",
    },
    "react": {
        "no_history": WEBSHOP_REACT_TEMPLATE_NO_HIS,
        "history": WEBSHOP_REACT_TEMPLATE,
        "example": WEBSHOP_REACT_EXAMPLE,
    },
    "act": {
        "no_history": WEBSHOP_ACT_TEMPLATE_NO_HIS,
        "history": WEBSHOP_ACT_TEMPLATE,
        "example": WEBSHOP_ACT_EXAMPLE,
    },
    "act_state": {
        "no_history": WEBSHOP_ACT_STATE_TEMPLATE,
        "history": WEBSHOP_ACT_STATE_TEMPLATE,
        "example": "",
    },
}


def get_webshop_prompt_template(prompt_style: str, use_history: bool):
    if prompt_style not in WEBSHOP_PROMPT_TEMPLATES:
        valid_styles = ", ".join(sorted(WEBSHOP_PROMPT_TEMPLATES))
        raise ValueError(f"Unknown WebShop prompt_style={prompt_style!r}. Valid styles: {valid_styles}")
    key = "history" if use_history else "no_history"
    return WEBSHOP_PROMPT_TEMPLATES[prompt_style][key]


def get_webshop_prompt_example(prompt_style: str):
    if prompt_style not in WEBSHOP_PROMPT_TEMPLATES:
        valid_styles = ", ".join(sorted(WEBSHOP_PROMPT_TEMPLATES))
        raise ValueError(f"Unknown WebShop prompt_style={prompt_style!r}. Valid styles: {valid_styles}")
    return WEBSHOP_PROMPT_TEMPLATES[prompt_style]["example"].strip()
