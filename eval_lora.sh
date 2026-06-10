#!/usr/bin/env bash

export HF_HOME=/data1/huggingface_cache_dir/
export HF_MODULES_CACHE=${PWD}/.cache/huggingface/modules
mkdir -p "${HF_MODULES_CACHE}"

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
PARALLEL_ENVS=16 \
VLLM_FLASH_ATTN_VERSION=2 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
LORA_ADAPTER="outputs/webshop_hyperclovax_3B_lora_3e-4/global_step_1263" \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B \
OUTPUT_DIR="checkpoints/verl_agent_webshop/webshop_lite_hyperclovax_3B_3epoch_act_state" \
CUDA_VISIBLE_DEVICES=$1 \
bash examples/validation/run_webshop_lite.sh

# LORA_ADAPTER="outputs/webshop_hyperclovax_1_5B_lora_3e-4/global_step_421" \
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
