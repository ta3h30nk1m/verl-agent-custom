#!/usr/bin/env bash
set -xeuo pipefail

_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
. "${_SCRIPT_DIR}/../../../scripts/load_experiment_env.sh"
unset _SCRIPT_DIR


MAX_STEPS=5000 \
CUDA_VISIBLE_DEVICES=2 \
TOTAL_EPOCHS=1 \
LR=2e-4 \
MAVE_TASK=multi_attribute_card_completion \
SAVE_FREQ=100 \
LORA_ADAPTER_PATH=outputs/mave/macc_init_lora_0_5b_to_1_5b_simple_scale2 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_PATH=outputs/mave_hyperclovax_multi_attribute_card_completion_lora_1_5B_lr2e-4_gradtransfer_norm_scale2 \
EXPERIMENT_NAME=mave-macc-hyperclovax-1.5b-lora-lr2e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
