#!/usr/bin/env bash
set -xeuo pipefail

usage() {
    echo "Usage:"
    echo "  run_hyperclovax_lora_sp2.sh [save_path] [other_configs...]"
    echo "  run_hyperclovax_lora_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

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
    save_path=${SAVE_PATH:-outputs/webshop_hyperclovax_lora}
fi

DATA_DIR=${DATA_DIR:-"${PWD}/data/webshop_subgoal_sft_new"}
PREPARE_DATA=${PREPARE_DATA:-true}
VAL_SIZE=${VAL_SIZE:-16}
SPLIT_SEED=${SPLIT_SEED:-0}
TEST_SESSION_CUTOFF=${TEST_SESSION_CUTOFF:-0}

if [ "${PREPARE_DATA}" = true ]; then
    preprocess_args=(
        --output-dir "${DATA_DIR}"
        --val-size "${VAL_SIZE}"
        --split-seed "${SPLIT_SEED}"
        --test-session-cutoff "${TEST_SESSION_CUTOFF}"
    )
    if [ -n "${TRAJ_DIR:-}" ]; then
        preprocess_args+=(--traj-dir "${TRAJ_DIR}")
    fi
    if [ -n "${PRODUCT_FILE:-}" ]; then
        preprocess_args+=(--product-file "${PRODUCT_FILE}")
    fi
    python3 -m examples.data_preprocess.webshop_subgoal_sft "${preprocess_args[@]}"
fi

TRAIN_FILES=${TRAIN_FILES:-"${DATA_DIR}/train.parquet"}

if [ -z "${VAL_FILES+x}" ]; then
    if [ -s "${DATA_DIR}/test.jsonl" ]; then
        VAL_FILES="${DATA_DIR}/test.parquet"
    else
        VAL_FILES="${TRAIN_FILES}"
    fi
fi

MODEL_PATH=${MODEL_PATH:-naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B}
MAX_LENGTH=${MAX_LENGTH:-4096}
PAD_TO_MAX_LENGTH=${PAD_TO_MAX_LENGTH:-false}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
SP_SIZE=${SP_SIZE:-1}
LR=${LR:-2e-4}
LOSS_VAL_ENABLE=${LOSS_VAL_ENABLE:-false}
LOSS_VAL_FREQ=${LOSS_VAL_FREQ:--1}
LOSS_VAL_BEFORE_TRAIN=${LOSS_VAL_BEFORE_TRAIN:-false}
ONLINE_VAL_ENABLE=${ONLINE_VAL_ENABLE:-true}
ONLINE_VAL_FREQ=${ONLINE_VAL_FREQ:-200}
ONLINE_VAL_GOAL_START=${ONLINE_VAL_GOAL_START:-0}
ONLINE_VAL_GOAL_END=${ONLINE_VAL_GOAL_END:-16}
ONLINE_VAL_NUM_EPISODES=${ONLINE_VAL_NUM_EPISODES:-$((ONLINE_VAL_GOAL_END - ONLINE_VAL_GOAL_START))}
ONLINE_VAL_PROMPT_STYLE=${ONLINE_VAL_PROMPT_STYLE:-act_state}
ONLINE_VAL_HISTORY_LENGTH=${ONLINE_VAL_HISTORY_LENGTH:-1}
ONLINE_VAL_MAX_STEPS=${ONLINE_VAL_MAX_STEPS:-15}
ONLINE_VAL_MAX_PROMPT_LENGTH=${ONLINE_VAL_MAX_PROMPT_LENGTH:-4096}
ONLINE_VAL_MAX_NEW_TOKENS=${ONLINE_VAL_MAX_NEW_TOKENS:-256}
ONLINE_VAL_TEMPERATURE=${ONLINE_VAL_TEMPERATURE:-0.6}
ONLINE_VAL_DO_SAMPLE=${ONLINE_VAL_DO_SAMPLE:-true}
ONLINE_VAL_DEVICE=${ONLINE_VAL_DEVICE:-cuda}
ONLINE_VAL_CUDA_VISIBLE_DEVICES=${ONLINE_VAL_CUDA_VISIBLE_DEVICES:-null}
ONLINE_VAL_RAY_NUM_CPUS=${ONLINE_VAL_RAY_NUM_CPUS:-2}
ONLINE_VAL_NUM_CPUS_PER_ENV_WORKER=${ONLINE_VAL_NUM_CPUS_PER_ENV_WORKER:-0.1}
ONLINE_VAL_OUTPUT_DIR=${ONLINE_VAL_OUTPUT_DIR:-"${save_path}/webshop_online_validation"}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-null}



EXPERIMENT_NAME=${EXPERIMENT_NAME:-webshop-hyperclovax-1.5b-instruct-lora-r${LORA_RANK}-sp${SP_SIZE}}
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
    trainer.project_name=webshop-sft \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.loss_validation_enable="${LOSS_VAL_ENABLE}" \
    trainer.test_freq="${LOSS_VAL_FREQ}" \
    trainer.val_before_train="${LOSS_VAL_BEFORE_TRAIN}" \
    trainer.online_webshop_validation.enable="${ONLINE_VAL_ENABLE}" \
    trainer.online_webshop_validation.freq="${ONLINE_VAL_FREQ}" \
    trainer.online_webshop_validation.num_episodes="${ONLINE_VAL_NUM_EPISODES}" \
    trainer.online_webshop_validation.goal_start="${ONLINE_VAL_GOAL_START}" \
    trainer.online_webshop_validation.goal_end="${ONLINE_VAL_GOAL_END}" \
    trainer.online_webshop_validation.prompt_style="${ONLINE_VAL_PROMPT_STYLE}" \
    trainer.online_webshop_validation.history_length="${ONLINE_VAL_HISTORY_LENGTH}" \
    trainer.online_webshop_validation.max_steps="${ONLINE_VAL_MAX_STEPS}" \
    trainer.online_webshop_validation.max_prompt_length="${ONLINE_VAL_MAX_PROMPT_LENGTH}" \
    trainer.online_webshop_validation.max_new_tokens="${ONLINE_VAL_MAX_NEW_TOKENS}" \
    trainer.online_webshop_validation.temperature="${ONLINE_VAL_TEMPERATURE}" \
    trainer.online_webshop_validation.do_sample="${ONLINE_VAL_DO_SAMPLE}" \
    trainer.online_webshop_validation.device="${ONLINE_VAL_DEVICE}" \
    trainer.online_webshop_validation.cuda_visible_devices="${ONLINE_VAL_CUDA_VISIBLE_DEVICES}" \
    trainer.online_webshop_validation.ray_num_cpus="${ONLINE_VAL_RAY_NUM_CPUS}" \
    trainer.online_webshop_validation.num_cpus_per_env_worker="${ONLINE_VAL_NUM_CPUS_PER_ENV_WORKER}" \
    trainer.online_webshop_validation.output_dir="${ONLINE_VAL_OUTPUT_DIR}" \
    trainer.logger="${LOGGER}" \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding=true \
    "$@"
