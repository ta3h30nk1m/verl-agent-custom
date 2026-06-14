

export CUDA_VISIBLE_DEVICES=$1

python examples/initialization/init_hyperclovax_lora_from_source.py \
  --source-model-path naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --source-lora-path outputs/mind2web_hyperclovax_0_5B_lora_ep6_lr1e-4_r64/global_step_2910 \
  --target-model-path naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B \
  --output-dir outputs/init_lora_0_5b_to_3b_scale_b_scale2 \
  --normalize-gradients \
  --gradient-scale 2.0 \
  --train-files data/mind2web_sft/train.parquet \
  --multiturn \
  --num-calibration-samples 128 \
  --calibration-batch-size 1 \
  --projection-method scale_b \
  --overwrite

# evidence_grounded_extraction
# faceted_search_filtering
# multi_attribute_card_completion
# product_customer_qa
# single_attribute_qa
# data/mave_sft_amazon23/by_task/product_customer_qa/train.parquet
# naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B

# data/mind2web_sft/train.parquet