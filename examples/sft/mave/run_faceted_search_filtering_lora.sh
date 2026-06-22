#!/usr/bin/env bash
set -xeuo pipefail

_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
. "${_SCRIPT_DIR}/../../../scripts/load_experiment_env.sh"
unset _SCRIPT_DIR

CUDA_VISIBLE_DEVICES=1 \
LR=5e-5 \
MAVE_TASK=faceted_search_filtering \
MAX_STEPS=500 \
SAVE_FREQ=10 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B \
SAVE_PATH=outputs/mave_hyperclovax_faceted_search_filtering_lora_3B_lr5e-5 \
EXPERIMENT_NAME=mave-fsf-hyperclovax-3b-lora-lr5e-5 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"

# LORA_ADAPTER_PATH=outputs/mave/fsf_init_lora_0_5b_to_1_5b_simple_scale1 \

# LORA_ADAPTER_PATH=outputs/mave/fsf_init_lora_0_5b_to_1_5b_simple_scale1 \
