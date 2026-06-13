#!/usr/bin/env bash
set -xeuo pipefail


MAX_STEPS=5000 \
CUDA_VISIBLE_DEVICES=1 \
TOTAL_EPOCHS=1 \
LR=2e-4 \
MAVE_TASK=product_customer_qa \
SAVE_FREQ=100 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_PATH=outputs/mave_hyperclovax_product_customer_qa_lora_1_5B_lr2e-4 \
EXPERIMENT_NAME=mave-pcqa-hyperclovax-1.5b-lora-lr2e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
