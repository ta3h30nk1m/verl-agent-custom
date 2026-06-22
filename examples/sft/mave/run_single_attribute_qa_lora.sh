#!/usr/bin/env bash
set -xeuo pipefail

_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
. "${_SCRIPT_DIR}/../../../scripts/load_experiment_env.sh"
unset _SCRIPT_DIR


CUDA_VISIBLE_DEVICES=1 \
LR=2e-4 \
MAVE_TASK=single_attribute_qa \
MAX_STEPS=5000 \
SAVE_FREQ=100 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_PATH=outputs/mave_hyperclovax_single_attribute_qa_lora_1_5B_lr2e-4 \
EXPERIMENT_NAME=mave-saqa-hyperclovax-1.5b-lora-lr2e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
