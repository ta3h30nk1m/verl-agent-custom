#!/usr/bin/env bash
set -xeuo pipefail

# Example:
#   CUDA_VISIBLE_DEVICES=0 \
#   MAVE_TASK=multi_attribute_card_completion \
#   LORA_ADAPTER=outputs/mave_hyperclovax_multi_attribute_card_completion_lora_steps5000_lr2e-4_r64/global_step_5000 \
#   bash eval_lora_mave.sh

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/load_experiment_env.sh"

export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${PWD}/.cache/huggingface/modules"}
mkdir -p "${HF_MODULES_CACHE}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${1:-0}}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION:-2}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

MAVE_TASK=${MAVE_TASK:-multi_attribute_card_completion}
MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}
DATA_DIR=${DATA_DIR:-"${PWD}/data/mave_sft_amazon23"}
EVAL_SPLIT=${EVAL_SPLIT:-test}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/verl_agent_mave/${MAVE_TASK}_$(basename "${LORA_ADAPTER:-base}")"}

bash examples/validation/run_mave_lora_eval.sh
