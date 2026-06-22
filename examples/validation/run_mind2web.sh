#!/usr/bin/env bash
set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B}
LORA_ADAPTER=${LORA_ADAPTER:-}
if [[ -z "${INFERENCE_BACKEND+x}" ]]; then
    case "$MODEL_PATH" in
        Qwen/Qwen3*|*/Qwen3*)
            INFERENCE_BACKEND=hf
            ;;
        *)
            INFERENCE_BACKEND=vllm
            ;;
    esac
fi

MIND2WEB_EVAL_FILE=${MIND2WEB_EVAL_FILE:-data/mind2web_sft/test.jsonl}
MIND2WEB_SPLIT=${MIND2WEB_SPLIT:-}
VAL_DATA_SIZE=${VAL_DATA_SIZE:--1}
START_INDEX=${START_INDEX:-0}
BATCH_SIZE=${BATCH_SIZE:-${PARALLEL_ENVS:-8}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/verl_agent_mind2web/mind2web_validation}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-$BATCH_SIZE}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}
VLLM_MERGED_MODEL_DIR=${VLLM_MERGED_MODEL_DIR:-}
VLLM_FORCE_MERGE_LORA=${VLLM_FORCE_MERGE_LORA:-false}

export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -XX:ActiveProcessorCount=1 -XX:ParallelGCThreads=1 -XX:ConcGCThreads=1"

adapter_args=()
if [[ -n "$LORA_ADAPTER" ]]; then
    adapter_args=(--lora-adapter "$LORA_ADAPTER")
fi

python3 -m examples.validation.mind2web_llm_eval \
    --backend "$INFERENCE_BACKEND" \
    --model-path "$MODEL_PATH" \
    "${adapter_args[@]}" \
    --eval-file "$MIND2WEB_EVAL_FILE" \
    --split "$MIND2WEB_SPLIT" \
    --num-examples "$VAL_DATA_SIZE" \
    --start-index "$START_INDEX" \
    --batch-size "$BATCH_SIZE" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-new-tokens "$MAX_RESPONSE_LENGTH" \
    --output-dir "$OUTPUT_DIR" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm-max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --vllm-enforce-eager "$VLLM_ENFORCE_EAGER" \
    --vllm-merged-model-dir "$VLLM_MERGED_MODEL_DIR" \
    --vllm-force-merge-lora "$VLLM_FORCE_MERGE_LORA" \
    --trust-remote-code true \
    "$@"
