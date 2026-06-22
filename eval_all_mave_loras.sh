#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    cat >&2 <<'EOF'
Usage: bash eval_all_mave_loras.sh <gpu_id> <keyword> [keyword...]

Evaluates MAVE LoRA checkpoints whose adapter path:
  - starts under outputs/mave*
  - contains every keyword you pass
  - contains one MAVE task name, unless MAVE_TASK_OVERRIDE is set

Examples:
  bash eval_all_mave_loras.sh 0 product_customer_qa steps5000
  bash eval_all_mave_loras.sh 0 multi_attribute 1_5B global_step_1000

Optional env vars:
  LORA_ROOT=outputs
  OUTPUT_ROOT=checkpoints/verl_agent_mave
  VALIDATION_SCRIPT=eval_lora_mave.sh
  EVAL_SPLIT=test
  VAL_DATA_SIZE=1000
  THREE_B_CONDA_ENV=verl-agent-webshop-vllm3b
  DRY_RUN=1
  SKIP_EXISTING=1
  MAVE_TASK_OVERRIDE=product_customer_qa
EOF
    exit 2
fi

GPU_ID=$1
shift
KEYWORDS=("$@")

LORA_ROOT=${LORA_ROOT:-outputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/verl_agent_mave}
VALIDATION_SCRIPT=${VALIDATION_SCRIPT:-eval_lora_mave.sh}
THREE_B_CONDA_ENV=${THREE_B_CONDA_ENV:-verl-agent-webshop-vllm3b}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
EVAL_SPLIT=${EVAL_SPLIT:-test}

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/load_experiment_env.sh"

TASKS=(
    single_attribute_qa
    evidence_grounded_extraction
    multi_attribute_card_completion
    product_customer_qa
    faceted_search_filtering
)

export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${PWD}/.cache/huggingface/modules}
mkdir -p "${HF_MODULES_CACHE}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export PARALLEL_ENVS=${PARALLEL_ENVS:-8}
export VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION:-2}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

read_adapter_field() {
    local adapter_config=$1
    local field=$2
    python3 - "$adapter_config" "$field" <<'PY'
import json
import sys

path, field = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
value = data.get(field, "")
print(value if value is not None else "")
PY
}

sanitize_path_part() {
    local value=$1
    value=${value#${LORA_ROOT}/}
    value=${value//\//_}
    value=${value//./_}
    value=${value//-/_}
    printf '%s' "$value"
}

matches_keywords() {
    local value=$1
    local keyword

    for keyword in "${KEYWORDS[@]}"; do
        if [[ "${value}" != *"${keyword}"* ]]; then
            return 1
        fi
    done
}

infer_task() {
    local value=$1
    local task
    if [[ -n "${MAVE_TASK_OVERRIDE:-}" ]]; then
        printf '%s' "${MAVE_TASK_OVERRIDE}"
        return 0
    fi
    for task in "${TASKS[@]}"; do
        if [[ "${value}" == *"${task}"* ]]; then
            printf '%s' "${task}"
            return 0
        fi
    done
    return 1
}

uses_half_b_model() {
    local model_path=$1
    [[ "${model_path}" == *"0.5B"* || "${model_path}" == *"0_5B"* ]]
}

append_model_constraints() {
    local model_path=$1
    local -n env_args_ref=$2

    if uses_half_b_model "${model_path}"; then
        local max_response_length=${MAX_RESPONSE_LENGTH:-512}
        local max_model_len=8192
        local max_prompt_length=$((max_model_len - max_response_length))

        if (( max_prompt_length < 1 )); then
            echo "MAX_RESPONSE_LENGTH=${max_response_length} leaves no prompt room for 0.5B max_model_len=${max_model_len}" >&2
            exit 1
        fi

        env_args_ref+=(
            "MAX_RESPONSE_LENGTH=${max_response_length}"
            "MAX_PROMPT_LENGTH=${max_prompt_length}"
            "VLLM_MAX_MODEL_LEN=${max_model_len}"
            "VLLM_MAX_NUM_BATCHED_TOKENS=${max_model_len}"
        )
    fi
}

run_eval() {
    local lora_adapter=$1
    local model_path=$2
    local task=$3
    local output_dir=$4

    local -a env_args=(
        "LORA_ADAPTER=${lora_adapter}"
        "MODEL_PATH=${model_path}"
        "MAVE_TASK=${task}"
        "EVAL_SPLIT=${EVAL_SPLIT}"
        "OUTPUT_DIR=${output_dir}"
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    )

    append_model_constraints "${model_path}" env_args

    if [[ "${model_path}" == "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" || "${lora_adapter}" == *"3B"* ]]; then
        env "${env_args[@]}" conda run --no-capture-output -n "${THREE_B_CONDA_ENV}" bash "${VALIDATION_SCRIPT}"
    else
        env "${env_args[@]}" bash "${VALIDATION_SCRIPT}"
    fi
}

mapfile -t adapter_configs < <(
    find "${LORA_ROOT}" \
        -path '*/merged_lora_for_vllm_*' -prune -o \
        -path "${LORA_ROOT}/mave*/global_step_*/adapter_config.json" -type f -print \
        | sort -V
)

if (( ${#adapter_configs[@]} == 0 )); then
    echo "No MAVE LoRA adapter_config.json files found under ${LORA_ROOT}" >&2
    exit 1
fi

matched_adapter_configs=()
for adapter_config in "${adapter_configs[@]}"; do
    lora_adapter=${adapter_config%/adapter_config.json}
    if matches_keywords "${lora_adapter}"; then
        matched_adapter_configs+=("${adapter_config}")
    fi
done

if (( ${#matched_adapter_configs[@]} == 0 )); then
    echo "No MAVE LoRA checkpoints matched keywords: ${KEYWORDS[*]}" >&2
    exit 1
fi

echo "Found ${#matched_adapter_configs[@]} matching MAVE LoRA checkpoints under ${LORA_ROOT}"
echo "GPU: ${GPU_ID}"
echo "Keywords: ${KEYWORDS[*]}"
echo "Validation script: ${VALIDATION_SCRIPT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Eval split: ${EVAL_SPLIT}"
echo "3B conda env: ${THREE_B_CONDA_ENV}"
echo "Skip existing summary.json: ${SKIP_EXISTING}"

for adapter_config in "${matched_adapter_configs[@]}"; do
    lora_adapter=${adapter_config%/adapter_config.json}
    model_path=$(read_adapter_field "${adapter_config}" base_model_name_or_path)

    if [[ -z "${model_path}" ]]; then
        echo "Skipping ${lora_adapter}: base_model_name_or_path is empty" >&2
        continue
    fi

    if ! task=$(infer_task "${lora_adapter}"); then
        echo "Skipping ${lora_adapter}: could not infer MAVE task from path. Set MAVE_TASK_OVERRIDE." >&2
        continue
    fi

    output_name=$(sanitize_path_part "${lora_adapter}")
    output_dir="${OUTPUT_ROOT}/${task}/${output_name}_${EVAL_SPLIT}"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/summary.json" ]]; then
        echo
        echo "=== Skipping ${lora_adapter} ==="
        echo "summary.json already exists: ${output_dir}/summary.json"
        continue
    fi

    echo
    echo "=== Evaluating ${lora_adapter} ==="
    echo "TASK=${task}"
    echo "MODEL_PATH=${model_path}"
    echo "OUTPUT_DIR=${output_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        env_args=(
            "LORA_ADAPTER=${lora_adapter}"
            "MODEL_PATH=${model_path}"
            "MAVE_TASK=${task}"
            "EVAL_SPLIT=${EVAL_SPLIT}"
            "OUTPUT_DIR=${output_dir}"
            "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        )
        append_model_constraints "${model_path}" env_args
        printf 'DRY_RUN env:'
        printf ' %q' "${env_args[@]}"
        printf '\n'
        if [[ "${model_path}" == "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" || "${lora_adapter}" == *"3B"* ]]; then
            echo "DRY_RUN: conda run --no-capture-output -n ${THREE_B_CONDA_ENV} bash ${VALIDATION_SCRIPT}"
        else
            echo "DRY_RUN: bash ${VALIDATION_SCRIPT}"
        fi
        continue
    fi

    run_eval "${lora_adapter}" "${model_path}" "${task}" "${output_dir}"
done
