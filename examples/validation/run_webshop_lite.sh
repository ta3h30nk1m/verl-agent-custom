#!/usr/bin/env bash
set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
LORA_ADAPTER=${LORA_ADAPTER:-}
# if [[ -z "${INFERENCE_BACKEND+x}" ]]; then
#     case "$MODEL_PATH" in
#         Qwen/Qwen3*|*/Qwen3*)
#             INFERENCE_BACKEND=hf
#             ;;
#         *)
#             INFERENCE_BACKEND=vllm
#             ;;
#     esac
# fi

INFERENCE_BACKEND=hf

VAL_DATA_SIZE=${VAL_DATA_SIZE:-4}
WEBSHOP_GOAL_START=${WEBSHOP_GOAL_START:-0}
WEBSHOP_GOAL_END=${WEBSHOP_GOAL_END:-500}
MAX_STEPS=${MAX_STEPS:-15}
HISTORY_LENGTH=${HISTORY_LENGTH:-2}
WEBSHOP_PROMPT_STYLE=${WEBSHOP_PROMPT_STYLE:-act_state}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/verl_agent_webshop/webshop_lite_qwen3_8B_act_state}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-2}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}

export CUDA_VISIBLE_DEVICES
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -XX:ActiveProcessorCount=1 -XX:ParallelGCThreads=1 -XX:ConcGCThreads=1"

adapter_args=()
if [[ -n "$LORA_ADAPTER" ]]; then
    adapter_args=(--lora-adapter "$LORA_ADAPTER")
fi

python3 -m examples.validation.webshop_llm_eval \
    --backend "$INFERENCE_BACKEND" \
    --model-path "$MODEL_PATH" \
    "${adapter_args[@]}" \
    --num-episodes "$VAL_DATA_SIZE" \
    --goal-start "$WEBSHOP_GOAL_START" \
    --goal-end "$WEBSHOP_GOAL_END" \
    --max-steps "$MAX_STEPS" \
    --history-length "$HISTORY_LENGTH" \
    --prompt-style "$WEBSHOP_PROMPT_STYLE" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-new-tokens "$MAX_RESPONSE_LENGTH" \
    --output-dir "$OUTPUT_DIR" \
    --ray-num-cpus "$RAY_NUM_CPUS" \
    --num-cpus-per-env-worker "$NUM_CPUS_PER_ENV_WORKER" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm-max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --vllm-enforce-eager "$VLLM_ENFORCE_EAGER" \
    --trust-remote-code true \
    "$@"
