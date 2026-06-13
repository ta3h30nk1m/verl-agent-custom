#!/usr/bin/env bash
set -xeuo pipefail

usage() {
    echo "Usage:"
    echo "  MAVE_TASK=<task_type> run_hyperclovax_lora_single_gpu.sh [save_path] [other_configs...]"
    echo "  run_hyperclovax_lora_single_gpu.sh <task_type> [save_path] [other_configs...]"
    echo "  run_hyperclovax_lora_single_gpu.sh <nproc_per_node> <task_type> <save_path> [other_configs...]"
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

VALID_TASKS=(
    single_attribute_qa
    evidence_grounded_extraction
    multi_attribute_card_completion
    product_customer_qa
    faceted_search_filtering
)

is_valid_task() {
    local candidate=$1
    local task
    for task in "${VALID_TASKS[@]}"; do
        if [ "${candidate}" = "${task}" ]; then
            return 0
        fi
    done
    return 1
}

if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    if [ "$#" -lt 3 ]; then
        usage
        exit 1
    fi
    nproc_per_node=$1
    MAVE_TASK=$2
    save_path=$3
    shift 3
elif [ "$#" -gt 0 ] && is_valid_task "$1"; then
    nproc_per_node=${NPROC_PER_NODE:-1}
    MAVE_TASK=$1
    shift 1
    if [ "$#" -gt 0 ]; then
        save_path=$1
        shift 1
    else
        save_path=${SAVE_PATH:-outputs/mave_hyperclovax_${MAVE_TASK}_lora}
    fi
else
    nproc_per_node=${NPROC_PER_NODE:-1}
    MAVE_TASK=${MAVE_TASK:-single_attribute_qa}
    save_path=${SAVE_PATH:-outputs/mave_hyperclovax_${MAVE_TASK}_lora}
fi

if ! is_valid_task "${MAVE_TASK}"; then
    echo "Invalid MAVE_TASK=${MAVE_TASK}"
    usage
    exit 1
fi

DATA_DIR=${DATA_DIR:-"${PWD}/data/mave_sft_amazon23"}
TASK_OUTPUT_DIR_NAME=${TASK_OUTPUT_DIR_NAME:-by_task}
TASK_DATA_DIR=${TASK_DATA_DIR:-"${DATA_DIR}/${TASK_OUTPUT_DIR_NAME}/${MAVE_TASK}"}
PREPARE_DATA=${PREPARE_DATA:-false}
DOWNLOAD_MAVE_LABELS=${DOWNLOAD_MAVE_LABELS:-true}
AMAZON23_META_DIR=${AMAZON23_META_DIR:-"${PWD}/data/amazon_reviews_2023_meta"}
RAW_DIR=${RAW_DIR:-"${PWD}/data/mave_raw"}
VAL_RATIO=${VAL_RATIO:-0.02}
TEST_RATIO=${TEST_RATIO:-0.02}

if [ "${PREPARE_DATA}" = true ]; then
    preprocess_args=(
        --raw-dir "${RAW_DIR}"
        --output-dir "${DATA_DIR}"
        --amazon23-meta-dir "${AMAZON23_META_DIR}"
        --val-ratio "${VAL_RATIO}"
        --test-ratio "${TEST_RATIO}"
    )
    if [ "${DOWNLOAD_MAVE_LABELS}" = true ]; then
        preprocess_args+=(--download-mave-labels)
    fi
    if [ -n "${POSITIVE_JSONL:-}" ]; then
        preprocess_args+=(--positive-jsonl "${POSITIVE_JSONL}")
    fi
    if [ -n "${POSITIVE_LABEL_JSONL:-}" ]; then
        preprocess_args+=(--positive-label-jsonl "${POSITIVE_LABEL_JSONL}")
    fi
    if [ -n "${NEGATIVE_JSONL:-}" ]; then
        preprocess_args+=(--negative-jsonl "${NEGATIVE_JSONL}")
    fi
    if [ -n "${AMAZON23_FULL_OUTPUT:-}" ]; then
        preprocess_args+=(--amazon23-full-output "${AMAZON23_FULL_OUTPUT}")
    fi
    if [ "${OVERWRITE_AMAZON23_FULL:-false}" = true ]; then
        preprocess_args+=(--overwrite-amazon23-full)
    fi
    if [ -n "${MAX_PRODUCTS:-}" ]; then
        preprocess_args+=(--max-products "${MAX_PRODUCTS}")
    fi
    if [ -n "${MAX_EXAMPLES_PER_TASK:-}" ]; then
        preprocess_args+=(--max-examples-per-task "${MAX_EXAMPLES_PER_TASK}")
    fi
    if [ -n "${MAVE_PREPROCESS_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        extra_args=(${MAVE_PREPROCESS_EXTRA_ARGS})
        preprocess_args+=("${extra_args[@]}")
    fi
    python3 -m examples.data_preprocess.mave_sft "${preprocess_args[@]}"
fi

if [ ! -d "${TASK_DATA_DIR}" ]; then
    echo "Task data directory does not exist: ${TASK_DATA_DIR}"
    echo "Run with PREPARE_DATA=true or generate data with examples.data_preprocess.mave_sft first."
    exit 1
fi

TRAIN_FILES=${TRAIN_FILES:-"${TASK_DATA_DIR}/train.parquet"}
if [ -z "${VAL_FILES+x}" ]; then
    if [ -s "${TASK_DATA_DIR}/val.jsonl" ]; then
        VAL_FILES="${TASK_DATA_DIR}/val.parquet"
    elif [ -s "${TASK_DATA_DIR}/test.jsonl" ]; then
        VAL_FILES="${TASK_DATA_DIR}/test.parquet"
    else
        VAL_FILES="${TRAIN_FILES}"
    fi
fi

MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}
MAX_LENGTH=${MAX_LENGTH:-4096}
PAD_TO_MAX_LENGTH=${PAD_TO_MAX_LENGTH:-false}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
REQUESTED_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
MAX_STEPS=${MAX_STEPS:-}
MAX_STEPS_STRICT=${MAX_STEPS_STRICT:-true}
MAX_STEPS_TOTAL_EPOCHS_CAP=${MAX_STEPS_TOTAL_EPOCHS_CAP:-1000000}
if [ -n "${MAX_STEPS}" ] && [ "${MAX_STEPS_STRICT}" = true ]; then
    TOTAL_EPOCHS="${MAX_STEPS_TOTAL_EPOCHS_CAP}"
fi
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
SP_SIZE=${SP_SIZE:-1}
LR=${LR:-2e-4}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}

if [ -z "${SAVE_PATH_SUFFIX+x}" ]; then
    if [ -n "${MAX_STEPS}" ]; then
        SAVE_PATH_SUFFIX="steps${MAX_STEPS}_lr${LR}_r${LORA_RANK}"
    else
        SAVE_PATH_SUFFIX="ep${REQUESTED_TOTAL_EPOCHS}_lr${LR}_r${LORA_RANK}"
    fi
fi
if [ -n "${SAVE_PATH_SUFFIX}" ]; then
    save_path="${save_path}_${SAVE_PATH_SUFFIX}"
fi

LOSS_VAL_ENABLE=${LOSS_VAL_ENABLE:-true}
LOSS_VAL_FREQ=${LOSS_VAL_FREQ:--1}
LOSS_VAL_BEFORE_TRAIN=${LOSS_VAL_BEFORE_TRAIN:-false}
SAVE_FREQ=${SAVE_FREQ:--1}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-mave-${MAVE_TASK}-hyperclovax-lora-r${LORA_RANK}-sp${SP_SIZE}}
LOGGER=${LOGGER:-"['console','wandb']"}

trainer_step_args=()
if [ -n "${MAX_STEPS}" ]; then
    trainer_step_args+=(trainer.total_training_steps="${MAX_STEPS}")
fi

echo "MAVE_TASK=${MAVE_TASK}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "VAL_FILES=${VAL_FILES}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "REQUESTED_TOTAL_EPOCHS=${REQUESTED_TOTAL_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS:-<unset>}"
echo "MAX_STEPS_STRICT=${MAX_STEPS_STRICT}"

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.pad_to_max_length="${PAD_TO_MAX_LENGTH}" \
    data.truncation=left \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=true \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha="${LORA_ALPHA}" \
    model.target_modules="${LORA_TARGET_MODULES}" \
    model.lora_target_scope="${LORA_TARGET_SCOPE}" \
    model.trust_remote_code=true \
    optim.lr="${LR}" \
    optim.warmup_steps_ratio=0.03 \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name=mave-sft \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${trainer_step_args[@]}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.loss_validation_enable="${LOSS_VAL_ENABLE}" \
    trainer.test_freq="${LOSS_VAL_FREQ}" \
    trainer.val_before_train="${LOSS_VAL_BEFORE_TRAIN}" \
    trainer.logger="${LOGGER}" \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding=true \
    "$@"
