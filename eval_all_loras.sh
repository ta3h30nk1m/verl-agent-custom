#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-${CUDA_VISIBLE_DEVICES:-0}}
LORA_ROOT=${LORA_ROOT:-outputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/verl_agent_webshop}
VALIDATION_SCRIPT=${VALIDATION_SCRIPT:-examples/validation/run_webshop_lite.sh}
THREE_B_CONDA_ENV=${THREE_B_CONDA_ENV:-verl-agent-webshop-vllm3b}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export HF_HOME=${HF_HOME:-/data1/huggingface_cache_dir/}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${PWD}/.cache/huggingface/modules}
mkdir -p "${HF_MODULES_CACHE}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export PARALLEL_ENVS=${PARALLEL_ENVS:-16}
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
    value=${value#outputs/}
    value=${value//\//_}
    value=${value//./_}
    value=${value//-/_}
    printf '%s' "$value"
}

run_eval() {
    local lora_adapter=$1
    local model_path=$2
    local output_dir=$3

    local -a env_args=(
        "LORA_ADAPTER=${lora_adapter}"
        "MODEL_PATH=${model_path}"
        "OUTPUT_DIR=${output_dir}"
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    )

    if [[ "${model_path}" == *"3B"* ]]; then
        env "${env_args[@]}" conda run --no-capture-output -n "${THREE_B_CONDA_ENV}" bash "${VALIDATION_SCRIPT}"
    else
        env "${env_args[@]}" bash "${VALIDATION_SCRIPT}"
    fi
}

mapfile -t adapter_configs < <(
    find "${LORA_ROOT}" \
        -path '*/webshop_online_validation/*' -prune -o \
        -path '*/global_step_*/adapter_config.json' -type f -print \
        | sort -V
)

if (( ${#adapter_configs[@]} == 0 )); then
    echo "No LoRA adapter_config.json files found under ${LORA_ROOT}" >&2
    exit 1
fi

echo "Found ${#adapter_configs[@]} LoRA checkpoints under ${LORA_ROOT}"
echo "GPU: ${GPU_ID}"
echo "3B conda env: ${THREE_B_CONDA_ENV}"
echo "Skip existing summary.json: ${SKIP_EXISTING}"

for adapter_config in "${adapter_configs[@]}"; do
    lora_adapter=${adapter_config%/adapter_config.json}
    model_path=$(read_adapter_field "${adapter_config}" base_model_name_or_path)

    if [[ -z "${model_path}" ]]; then
        echo "Skipping ${lora_adapter}: base_model_name_or_path is empty" >&2
        continue
    fi

    output_name=$(sanitize_path_part "${lora_adapter}")
    output_dir="${OUTPUT_ROOT}/webshop_lite_${output_name}_act_state"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/summary.json" ]]; then
        echo
        echo "=== Skipping ${lora_adapter} ==="
        echo "summary.json already exists: ${output_dir}/summary.json"
        continue
    fi

    echo
    echo "=== Evaluating ${lora_adapter} ==="
    echo "MODEL_PATH=${model_path}"
    echo "OUTPUT_DIR=${output_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        if [[ "${model_path}" == *"3B"* ]]; then
            echo "DRY_RUN: conda run --no-capture-output -n ${THREE_B_CONDA_ENV} bash ${VALIDATION_SCRIPT}"
        else
            echo "DRY_RUN: bash ${VALIDATION_SCRIPT}"
        fi
        continue
    fi

    run_eval "${lora_adapter}" "${model_path}" "${output_dir}"
done
