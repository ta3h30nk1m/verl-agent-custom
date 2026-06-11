#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    cat >&2 <<'EOF'
Usage: bash eval_all_mind2web_loras.sh <gpu_id> <keyword> [keyword...]

Evaluates LoRA checkpoints whose experiment folder:
  - starts with mind2web
  - contains every keyword you pass

Examples:
  bash eval_all_mind2web_loras.sh 0 lr1e-4
  bash eval_all_mind2web_loras.sh 0 0_5B lr3e-4

Optional env vars:
  LORA_ROOT=outputs
  OUTPUT_ROOT=checkpoints/verl_agent_mind2web
  VALIDATION_SCRIPT=examples/validation/run_mind2web.sh
  THREE_B_CONDA_ENV=verl-agent-webshop-vllm3b
  DRY_RUN=1
  SKIP_EXISTING=0
EOF
    exit 2
fi

GPU_ID=$1
shift
KEYWORDS=("$@")

LORA_ROOT=${LORA_ROOT:-outputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/verl_agent_mind2web}
VALIDATION_SCRIPT=${VALIDATION_SCRIPT:-examples/validation/run_mind2web.sh}
THREE_B_CONDA_ENV=${THREE_B_CONDA_ENV:-verl-agent-webshop-vllm3b}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}


export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${PWD}/.cache/huggingface/modules}
mkdir -p "${HF_MODULES_CACHE}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export PARALLEL_ENVS=${PARALLEL_ENVS:-32}
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

run_eval() {
    local lora_adapter=$1
    local model_path=$2
    local output_dir=$3
    local experiment_dir=${lora_adapter%/global_step_*}
    local experiment_name=${experiment_dir##*/}

    local -a env_args=(
        "LORA_ADAPTER=${lora_adapter}"
        "MODEL_PATH=${model_path}"
        "OUTPUT_DIR=${output_dir}"
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    )

    if [[ "${experiment_name}" == *"3B"* || "${model_path}" == "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" ]]; then
        env "${env_args[@]}" conda run --no-capture-output -n "${THREE_B_CONDA_ENV}" bash "${VALIDATION_SCRIPT}"
    else
        env "${env_args[@]}" bash "${VALIDATION_SCRIPT}"
    fi
}

mapfile -t adapter_configs < <(
    find "${LORA_ROOT}" \
        -path '*/mind2web_online_validation/*' -prune -o \
        -path "${LORA_ROOT}/mind2web*/global_step_*/adapter_config.json" -type f -print \
        | sort -V
)

if (( ${#adapter_configs[@]} == 0 )); then
    echo "No Mind2Web LoRA adapter_config.json files found under ${LORA_ROOT}" >&2
    exit 1
fi

matched_adapter_configs=()
for adapter_config in "${adapter_configs[@]}"; do
    lora_adapter=${adapter_config%/adapter_config.json}
    experiment_dir=${lora_adapter%/global_step_*}
    experiment_name=${experiment_dir##*/}

    if matches_keywords "${experiment_name}"; then
        matched_adapter_configs+=("${adapter_config}")
    fi
done

if (( ${#matched_adapter_configs[@]} == 0 )); then
    echo "No Mind2Web LoRA checkpoints matched keywords: ${KEYWORDS[*]}" >&2
    exit 1
fi

echo "Found ${#matched_adapter_configs[@]} matching Mind2Web LoRA checkpoints under ${LORA_ROOT}"
echo "GPU: ${GPU_ID}"
echo "Keywords: ${KEYWORDS[*]}"
echo "Validation script: ${VALIDATION_SCRIPT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "3B conda env: ${THREE_B_CONDA_ENV}"
echo "Skip existing summary.json: ${SKIP_EXISTING}"

for adapter_config in "${matched_adapter_configs[@]}"; do
    lora_adapter=${adapter_config%/adapter_config.json}
    model_path=$(read_adapter_field "${adapter_config}" base_model_name_or_path)

    if [[ -z "${model_path}" ]]; then
        echo "Skipping ${lora_adapter}: base_model_name_or_path is empty" >&2
        continue
    fi

    output_name=$(sanitize_path_part "${lora_adapter}")
    output_dir="${OUTPUT_ROOT}/${output_name}"

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
        experiment_dir=${lora_adapter%/global_step_*}
        experiment_name=${experiment_dir##*/}
        if [[ "${experiment_name}" == *"3B"* || "${model_path}" == "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" ]]; then
            echo "DRY_RUN: conda run --no-capture-output -n ${THREE_B_CONDA_ENV} bash ${VALIDATION_SCRIPT}"
        else
            echo "DRY_RUN: bash ${VALIDATION_SCRIPT}"
        fi
        continue
    fi

    run_eval "${lora_adapter}" "${model_path}" "${output_dir}"
done
