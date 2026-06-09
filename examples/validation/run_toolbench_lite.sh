#!/usr/bin/env bash
set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
INFERENCE_BACKEND=${INFERENCE_BACKEND:-vllm}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-8}
TOOLBENCH_TEST_SET=${TOOLBENCH_TEST_SET:-G1_instruction}
TOOLBENCH_EXECUTION_MODE=${TOOLBENCH_EXECUTION_MODE:-real}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_API_DESCRIPTION_CHARS=${MAX_API_DESCRIPTION_CHARS:-1200}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/verl_agent_toolbench/toolbench_lite_validation}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.7}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}

export CUDA_VISIBLE_DEVICES

python3 -m examples.validation.toolbench_llm_eval \
    --backend "$INFERENCE_BACKEND" \
    --model-path "$MODEL_PATH" \
    --num-episodes "$VAL_DATA_SIZE" \
    --max-steps "$MAX_STEPS" \
    --test-set "$TOOLBENCH_TEST_SET" \
    --execution-mode "$TOOLBENCH_EXECUTION_MODE" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-new-tokens "$MAX_RESPONSE_LENGTH" \
    --max-api-description-chars "$MAX_API_DESCRIPTION_CHARS" \
    --output-dir "$OUTPUT_DIR" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm-max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --vllm-enforce-eager "$VLLM_ENFORCE_EAGER" \
    "$@"
