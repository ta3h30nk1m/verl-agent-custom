#!/usr/bin/env bash
set -xeuo pipefail

# Run Mind2Web LoRA SFT for the three HyperCLOVAX models on one GPU,
# sequentially. Token env vars are loaded from .experiment_env when present.

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/load_experiment_env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_DIR=${DATA_DIR:-"${PWD}/data/mind2web_sft"}
PREPARE_DATA=${PREPARE_DATA:-true}
BALANCED_VAL_RATIO=${BALANCED_VAL_RATIO:-0.1}
BALANCED_VAL_SEED=${BALANCED_VAL_SEED:-0}
BALANCED_VAL_OUTPUT_NAME=${BALANCED_VAL_OUTPUT_NAME:-val_sample}

if [ "${PREPARE_DATA}" = true ]; then
    python3 -m examples.data_preprocess.mind2web_sft \
        --no-download \
        --output-dir "${DATA_DIR}" \
        --balanced-val-ratio "${BALANCED_VAL_RATIO}" \
        --balanced-val-seed "${BALANCED_VAL_SEED}" \
        --balanced-val-output-name "${BALANCED_VAL_OUTPUT_NAME}"
fi

COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    PREPARE_DATA=false
    DATA_DIR="${DATA_DIR}"
    BALANCED_VAL_OUTPUT_NAME="${BALANCED_VAL_OUTPUT_NAME}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    MAX_LENGTH="${MAX_LENGTH:-8192}"
    LR="${LR:-2e-4}"
    LORA_RANK="${LORA_RANK:-64}"
    LORA_ALPHA="${LORA_ALPHA:-64}"
    LORA_TARGET_SCOPE="${LORA_TARGET_SCOPE:-llm}"
    LOSS_VAL_ENABLE="${LOSS_VAL_ENABLE:-true}"
    LOSS_VAL_FREQ="${LOSS_VAL_FREQ:-200}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    LOGGER="${LOGGER:-['console','wandb']}"
)

# env "${COMMON_ENV[@]}" \
#     MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
#     SAVE_PATH=outputs/mind2web_hyperclovax_1_5B_lora \
#     EXPERIMENT_NAME=mind2web-hyperclovax-1.5b-lora \
#     bash examples/sft/mind2web/run_hyperclovax_lora_single_gpu.sh

# env "${COMMON_ENV[@]}" \
#     MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
#     SAVE_PATH=outputs/mind2web_hyperclovax_0_5B_lora \
#     EXPERIMENT_NAME=mind2web-hyperclovax-0.5b-lora \
#     bash examples/sft/mind2web/run_hyperclovax_lora_single_gpu.sh


    # LORA_ADAPTER_PATH=outputs/init_lora_0_5b_to_3b_norm_scale1 \

env "${COMMON_ENV[@]}" \
    LORA_ADAPTER_PATH=outputs/init_lora_0_5b_to_3b_double_normalize_simple_scale1 \
    MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B \
    SAVE_PATH=outputs/mind2web_hyperclovax_3B_lora_graddoubletransfer_normalize_scale1 \
    EXPERIMENT_NAME=mind2web-hyperclovax-3b-vision-lr2e-4 \
    bash examples/sft/mind2web/run_hyperclovax_lora_single_gpu.sh
