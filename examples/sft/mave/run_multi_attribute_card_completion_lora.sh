#!/usr/bin/env bash
set -xeuo pipefail


MAX_STEPS=5000 \
CUDA_VISIBLE_DEVICES=0 \
TOTAL_EPOCHS=1 \
LR=2e-4 \
MAVE_TASK=multi_attribute_card_completion \
SAVE_FREQ=100 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_PATH=outputs/mave_hyperclovax_multi_attribute_card_completion_lora_1_5B_lr2e-4 \
EXPERIMENT_NAME=mave-macc-hyperclovax-1.5b-lora-lr2e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
