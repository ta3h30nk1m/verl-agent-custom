#!/usr/bin/env bash
set -xeuo pipefail

# Run one MAVE LoRA SFT job per task, sequentially.
# Token env vars are loaded from .experiment_env when present.

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/load_experiment_env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_DIR=${DATA_DIR:-"${PWD}/data/mave_sft_amazon23"}
PREPARE_DATA=${PREPARE_DATA:-false}
AMAZON23_META_DIR=${AMAZON23_META_DIR:-"${PWD}/data/amazon_reviews_2023_meta"}
RAW_DIR=${RAW_DIR:-"${PWD}/data/mave_raw"}

TASKS=${TASKS:-"single_attribute_qa evidence_grounded_extraction multi_attribute_card_completion product_customer_qa faceted_search_filtering"}

COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    PREPARE_DATA="${PREPARE_DATA}"
    DOWNLOAD_MAVE_LABELS="${DOWNLOAD_MAVE_LABELS:-true}"
    DATA_DIR="${DATA_DIR}"
    AMAZON23_META_DIR="${AMAZON23_META_DIR}"
    RAW_DIR="${RAW_DIR}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
    MAX_STEPS="${MAX_STEPS:-}"
    MAX_STEPS_STRICT="${MAX_STEPS_STRICT:-true}"
    MAX_STEPS_TOTAL_EPOCHS_CAP="${MAX_STEPS_TOTAL_EPOCHS_CAP:-1000000}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    MAX_LENGTH="${MAX_LENGTH:-4096}"
    LR="${LR:-2e-4}"
    LORA_RANK="${LORA_RANK:-64}"
    LORA_ALPHA="${LORA_ALPHA:-128}"
    LORA_TARGET_SCOPE="${LORA_TARGET_SCOPE:-llm}"
    LOSS_VAL_ENABLE="${LOSS_VAL_ENABLE:-true}"
    LOSS_VAL_FREQ="${LOSS_VAL_FREQ:--1}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    LOGGER="${LOGGER:-['console','wandb']}"
    MODEL_PATH="${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}"
)

for task in ${TASKS}; do
    env "${COMMON_ENV[@]}" \
        MAVE_TASK="${task}" \
        SAVE_PATH="${SAVE_ROOT:-outputs}/mave_hyperclovax_${task}_lora" \
        EXPERIMENT_NAME="mave-${task}-hyperclovax-lora" \
        bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh

    # Avoid rebuilding the same MAVE dataset for every task in this sequential launcher.
    PREPARE_DATA=false
    COMMON_ENV[2]="PREPARE_DATA=false"
done
