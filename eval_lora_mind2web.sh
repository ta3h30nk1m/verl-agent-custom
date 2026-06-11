#!/usr/bin/env bash


export HF_MODULES_CACHE=${PWD}/.cache/huggingface/modules

mkdir -p "${HF_MODULES_CACHE}"

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
PARALLEL_ENVS=16 \
VLLM_FLASH_ATTN_VERSION=2 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
LORA_ADAPTER="outputs/mind2web_hyperclovax_0_5B_lora_ep6_lr1e-4_r64/global_step_2910" \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
OUTPUT_DIR="checkpoints/verl_agent_mind2web/mind2web_hyperclovax_0_5B_lora_1e-4_step_2910" \
CUDA_VISIBLE_DEVICES=$1 \
bash examples/validation/run_mind2web.sh

# LORA_ADAPTER="outputs/webshop_hyperclovax_1_5B_lora_3e-4/global_step_421" \
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
