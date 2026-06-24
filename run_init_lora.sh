#!/usr/bin/env bash

. "$(dirname -- "${BASH_SOURCE[0]}")/scripts/load_experiment_env.sh"

export CUDA_VISIBLE_DEVICES=$1

python examples/initialization/init_hyperclovax_lora_from_source.py \
  --source-model-path naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --source-lora-path outputs/mave_hyperclovax_product_customer_qa_lora_0_5B_lr2e-4_steps5000_lr2e-4_r64/global_step_3000 \
  --target-model-path naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
  --output-dir outputs/mave/pcqa_init_lora_0_5b_to_1_5b_double_normalize_scale_b0_5 \
  --gradient-scale 0.5 \
  --train-files data/mave_sft_amazon23/by_task/product_customer_qa/train.parquet \
  --multiturn \
  --num-calibration-samples 128 \
  --calibration-batch-size 2 \
  --projection-method scale_b \
  --num-projection-steps 2 \
  --normalize-gradients \
  --loraga-gamma 1024 \
  --overwrite

# evidence_grounded_extraction
# faceted_search_filtering
# multi_attribute_card_completion
# product_customer_qa
# single_attribute_qa
# data/mave_sft_amazon23/by_task/product_customer_qa/train.parquet
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B

# data/mind2web_sft/train.parquet
# outputs/mind2web_hyperclovax_0_5B_lora_ep6_lr1e-4_r64/global_step_2910

# outputs/mave_hyperclovax_evidence_grounded_extraction_lora_0_5B_lr2e-4_steps5000_lr2e-4_r64/global_step_3000

# --normalize-gradients \

  # --init-method gradtransfer_loraga \
