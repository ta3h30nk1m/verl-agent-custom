#!/usr/bin/env bash
set -xeuo pipefail

usage() {
    echo "Usage:"
    echo "  run_hyperclovax_lora_single_gpu.sh [save_path] [other_configs...]"
    echo "  run_hyperclovax_lora_single_gpu.sh <nproc_per_node> <save_path> [other_configs...]"
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    if [ "$#" -lt 2 ]; then
        usage
        exit 1
    fi
    nproc_per_node=$1
    save_path=$2
    shift 2
elif [ "$#" -gt 0 ]; then
    nproc_per_node=${NPROC_PER_NODE:-1}
    save_path=$1
    shift 1
else
    nproc_per_node=${NPROC_PER_NODE:-1}
    save_path=${SAVE_PATH:-outputs/mind2web_hyperclovax_lora}
fi

DATA_DIR=${DATA_DIR:-"${PWD}/data/mind2web_sft"}
PREPARE_DATA=${PREPARE_DATA:-false}
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

TRAIN_FILES=${TRAIN_FILES:-"${DATA_DIR}/train.parquet"}
if [ -z "${VAL_FILES+x}" ]; then
    if [ -s "${DATA_DIR}/${BALANCED_VAL_OUTPUT_NAME}.parquet" ]; then
        VAL_FILES="${DATA_DIR}/${BALANCED_VAL_OUTPUT_NAME}.parquet"
    else
        VAL_FILES="${DATA_DIR}/test.parquet"
    fi
fi

MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}
MAX_LENGTH=${MAX_LENGTH:-8192}
PAD_TO_MAX_LENGTH=${PAD_TO_MAX_LENGTH:-false}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
SP_SIZE=${SP_SIZE:-1}
LR=${LR:-2e-4}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-null}

if [ -z "${SAVE_PATH_SUFFIX+x}" ]; then
    SAVE_PATH_SUFFIX="ep${TOTAL_EPOCHS}_lr${LR}_r${LORA_RANK}"
fi
if [ -n "${SAVE_PATH_SUFFIX}" ]; then
    save_path="${save_path}_${SAVE_PATH_SUFFIX}"
fi

LOSS_VAL_ENABLE=${LOSS_VAL_ENABLE:-true}
LOSS_VAL_FREQ=${LOSS_VAL_FREQ:--1}
LOSS_VAL_BEFORE_TRAIN=${LOSS_VAL_BEFORE_TRAIN:-false}
SAVE_FREQ=${SAVE_FREQ:--1}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-mind2web-hyperclovax-lora-r${LORA_RANK}-sp${SP_SIZE}}
LOGGER=${LOGGER:-"['console','wandb']"}

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
    model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
    model.trust_remote_code=true \
    optim.lr="${LR}" \
    optim.warmup_steps_ratio=0.03 \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name=mind2web-sft \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.loss_validation_enable="${LOSS_VAL_ENABLE}" \
    trainer.test_freq="${LOSS_VAL_FREQ}" \
    trainer.val_before_train="${LOSS_VAL_BEFORE_TRAIN}" \
    trainer.logger="${LOGGER}" \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding=true \
    "$@"
