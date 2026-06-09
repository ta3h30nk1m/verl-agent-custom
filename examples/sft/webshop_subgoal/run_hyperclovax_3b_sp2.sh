#!/usr/bin/env bash
set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_hyperclovax_3b_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

TRAIN_FILES=${TRAIN_FILES:-"${PWD}/data/webshop_subgoal_sft/train.parquet"}
VAL_FILES=${VAL_FILES:-"${PWD}/data/webshop_subgoal_sft/test.parquet"}
MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B}
MAX_LENGTH=${MAX_LENGTH:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-8}
SP_SIZE=${SP_SIZE:-2}
LR=${LR:-2e-4}

export WANDB_API_KEY=v1_VNxn5dtAZcV3WnBUyWw9pu2ux25_DvuJ7PJ4BviVwtIkxYQ8WHf3HIU3wRaF52nskJqE9Oj233fhJ

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation=left \
    model.partial_pretrain="${MODEL_PATH}" \
    optim.lr="${LR}" \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name=webshop-sft \
    trainer.experiment_name=webshop-hyperclovax-1.5b-instruct-sp2 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding=true \
    "$@"
