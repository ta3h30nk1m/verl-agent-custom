#!/usr/bin/env bash
set -xeuo pipefail

export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${PWD}/.cache/huggingface/modules"}
mkdir -p "${HF_MODULES_CACHE}"

MAVE_TASK=${MAVE_TASK:-single_attribute_qa}
DATA_DIR=${DATA_DIR:-"${PWD}/data/mave_sft_amazon23"}
EVAL_SPLIT=${EVAL_SPLIT:-test}
MAVE_EVAL_FILE=${MAVE_EVAL_FILE:-"${DATA_DIR}/by_task/${MAVE_TASK}/${EVAL_SPLIT}.parquet"}
MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/verl_agent_mave/${MAVE_TASK}_eval"}
BACKEND=${BACKEND:-vllm}

python3 -m examples.validation.mave_lora_eval \
    --backend "${BACKEND}" \
    --model-path "${MODEL_PATH}" \
    --eval-file "${MAVE_EVAL_FILE}" \
    --task "${MAVE_TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --lora-adapter "${LORA_ADAPTER:-}" \
    --batch-size "${BATCH_SIZE:-${PARALLEL_ENVS:-8}}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH:-8192}" \
    --max-new-tokens "${MAX_RESPONSE_LENGTH:-512}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --top-p "${TOP_P:-1.0}" \
    --do-sample "${DO_SAMPLE:-false}" \
    --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
    --trust-remote-code "${TRUST_REMOTE_CODE:-true}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.7}" \
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-8704}" \
    --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-${PARALLEL_ENVS:-8}}" \
    --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8704}" \
    --vllm-enforce-eager "${VLLM_ENFORCE_EAGER:-false}" \
    "$@"
