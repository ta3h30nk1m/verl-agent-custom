#!/usr/bin/env bash
set -xeuo pipefail

_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
. "${_SCRIPT_DIR}/../../../scripts/load_experiment_env.sh"
unset _SCRIPT_DIR


MAX_STEPS=5000 \
CUDA_VISIBLE_DEVICES=2 \
TOTAL_EPOCHS=1 \
LR=1e-4 \
MAVE_TASK=product_customer_qa \
SAVE_FREQ=100 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_PATH=outputs/mave_hyperclovax_product_customer_qa_lora_1_5B_lr1e-4 \
EXPERIMENT_NAME=mave-pcqa-hyperclovax-1.5b-lora-lr1e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"


# LORA_ADAPTER_PATH=outputs/mave/pcqa_init_lora_0_5b_to_1_5b_simple_scale1 \
