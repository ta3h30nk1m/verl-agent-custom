# Important commands

```bash
pip install --no-cache-dir ray[default,cgraph]==2.43.0 timm
bash hyperclovax3b_vllm_setup.sh
```

## target model
Made from Naver HyperclovaX 

- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
- naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
- naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B
- naver-hyperclovax/HyperCLOVAX-SEED-Think-14B

## supporting sft training 

- WebShop
- Mind2Web
- ToolBench

## common env

필요하면 repository root의 `.experiment_env`에 토큰을 한 번만 적어둔다.

```bash
HF_TOKEN=<your_hf_token>
WANDB_API_KEY=<your_wandb_api_key>
```

각 실행 스크립트는 시작할 때 `scripts/load_experiment_env.sh`를 통해 이 파일을 읽어서 환경변수로 export한다.

<!-- export HF_MODULES_CACHE="${PWD}/.cache/huggingface/modules"
 mkdir -p "${HF_MODULES_CACHE}" -->

<!-- 평가에서 vLLM을 쓸 때 자주 쓰는 기본값:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
``` -->


## Mind2Web SFT

### Mind2Web data prepare

```bash
python3 -m examples.data_preprocess.mind2web_sft \
  --output-dir data/mind2web_sft
```

학습 스크립트에서 데이터 준비까지 같이 하려면 training 커맨드에 `PREPARE_DATA=true`를 추가한다. 이 경우 데이터 생성 후 학습도 바로 이어서 시작된다.

### Mind2Web training

0.5B LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
SAVE_PATH=outputs/mind2web_hyperclovax_0_5B_lora \
EXPERIMENT_NAME=mind2web-hyperclovax-0.5b-lora \
TOTAL_EPOCHS=6 \
SAVE_FREQ=200 \
LR=1e-4 \
bash examples/sft/mind2web/run_hyperclovax_lora_single_gpu.sh
```

3B Vision LoRA 또는 transferred LoRA 초기화:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B \
LORA_ADAPTER_PATH=outputs/init_lora_0_5b_to_3b_double_normalize_simple_scale1 \
SAVE_PATH=outputs/mind2web_hyperclovax_3B_lora_transfer \
EXPERIMENT_NAME=mind2web-hyperclovax-3b-vision-transfer-lora \
TOTAL_EPOCHS=3 \
SAVE_FREQ=200 \
LR=2e-4 \
bash examples/sft/mind2web/run_hyperclovax_lora_single_gpu.sh
```

### Mind2Web eval

단일 checkpoint 평가:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
LORA_ADAPTER=outputs/mind2web_hyperclovax_1_5B_lora_ep3_lr2e-4_r64/global_step_1200 \
MIND2WEB_EVAL_FILE=data/mind2web_sft/test.jsonl \
OUTPUT_DIR=checkpoints/verl_agent_mind2web/mind2web_hyperclovax_1_5B_lora_global_step_1200 \
PARALLEL_ENVS=8 \
BATCH_SIZE=8 \
MAX_PROMPT_LENGTH=16384 \
MAX_RESPONSE_LENGTH=256 \
bash examples/validation/run_mind2web.sh
```

0.5B 모델처럼 max length 제약이 필요할 때:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
LORA_ADAPTER=outputs/mind2web_hyperclovax_0_5B_lora_ep6_lr1e-4_r64/global_step_2910 \
OUTPUT_DIR=checkpoints/verl_agent_mind2web/mind2web_hyperclovax_0_5B_lora_global_step_2910 \
PARALLEL_ENVS=16 \
MAX_RESPONSE_LENGTH=256 \
MAX_PROMPT_LENGTH=7936 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
bash examples/validation/run_mind2web.sh
```

샘플 일부만 빠르게 평가:

```bash
CUDA_VISIBLE_DEVICES=0 \
VAL_DATA_SIZE=200 \
START_INDEX=0 \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
LORA_ADAPTER=outputs/mind2web_hyperclovax_1_5B_lora_ep3_lr2e-4_r64/global_step_1200 \
OUTPUT_DIR=checkpoints/verl_agent_mind2web/debug_step1200 \
bash examples/validation/run_mind2web.sh
```

keyword에 매칭되는 Mind2Web LoRA checkpoint 전체 평가:

```bash
DRY_RUN=1 \
LORA_ROOT=outputs \
OUTPUT_ROOT=checkpoints/verl_agent_mind2web \
bash eval_all_mind2web_loras.sh 0 mind2web 1_5B
```

```bash
SKIP_EXISTING=1 \
LORA_ROOT=outputs \
OUTPUT_ROOT=checkpoints/verl_agent_mind2web \
bash eval_all_mind2web_loras.sh 0 mind2web 1_5B
```

### Mind2Web plotting

```bash
python3 scripts/plot_mind2web_checkpoint_results.py \
  --root checkpoints/verl_agent_mind2web \
  --out-dir plots/mind2web_checkpoint_results \
  --metrics exact_action_acc target_id_acc overall_exact_action_acc op_acc value_acc json_parse_error_rate
```

특정 실험명만 필터링:

```bash
python3 scripts/plot_mind2web_checkpoint_results.py \
  --root checkpoints/verl_agent_mind2web \
  --out-dir plots/mind2web_checkpoint_results_1_5B \
  --include 'mind2web.*1_5B' \
  --metrics exact_action_acc target_id_acc op_acc value_acc
```

CSV/table만 확인:

```bash
python3 scripts/plot_mind2web_checkpoint_results.py \
  --root checkpoints/verl_agent_mind2web \
  --out-dir plots/mind2web_checkpoint_results \
  --list-only
```


## MAVE SFT

지원 task:

- `single_attribute_qa`
- `evidence_grounded_extraction`
- `multi_attribute_card_completion`
- `product_customer_qa`
- `faceted_search_filtering`

### MAVE data prepare

```bash
python3 -m examples.data_preprocess.mave_sft \
  --raw-dir data/mave_raw \
  --output-dir data/mave_sft_amazon23 \
  --amazon23-meta-dir data/amazon_reviews_2023_meta \
  --download-mave-labels \
  --val-ratio 0.02 \
  --test-ratio 0.02
```

학습 스크립트에서 데이터 준비까지 같이 하려면 training 커맨드에 `PREPARE_DATA=true`를 추가한다. 이 경우 데이터 생성 후 학습도 바로 이어서 시작된다.

### MAVE single-task training

```bash
CUDA_VISIBLE_DEVICES=0 \
MAVE_TASK=faceted_search_filtering \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
DATA_DIR="${PWD}/data/mave_sft_amazon23" \
SAVE_PATH=outputs/mave_hyperclovax_faceted_search_filtering_lora_1_5B \
EXPERIMENT_NAME=mave-fsf-hyperclovax-1.5b-lora \
MAX_STEPS=500 \
SAVE_FREQ=10 \
LR=1e-4 \
LR_SCHEDULER=constant \
LORA_RANK=64 \
LORA_ALPHA=128 \
TRAIN_BATCH_SIZE=16 \
MICRO_BATCH_SIZE_PER_GPU=1 \
LOGGER="['console','wandb']" \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh
```

0.5B 모델 예시:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAVE_TASK=product_customer_qa \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
SAVE_PATH=outputs/mave_hyperclovax_product_customer_qa_lora_0_5B \
MAX_STEPS=1000 \
SAVE_FREQ=100 \
LR=2e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh
```

기존 LoRA adapter에서 이어서 학습:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAVE_TASK=faceted_search_filtering \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
LORA_ADAPTER_PATH=outputs/mave/fsf_init_lora_0_5b_to_1_5b_double_normalize_scale1 \
SAVE_PATH=outputs/mave_hyperclovax_faceted_search_filtering_lora_1_5B_transfer \
MAX_STEPS=500 \
SAVE_FREQ=10 \
LR=1e-4 \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh
```

모든 MAVE task를 순차 실행:

```bash
CUDA_VISIBLE_DEVICES=0 \
PREPARE_DATA=true \
TASKS="single_attribute_qa evidence_grounded_extraction multi_attribute_card_completion product_customer_qa faceted_search_filtering" \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
SAVE_ROOT=outputs \
MAX_STEPS=500 \
SAVE_FREQ=100 \
LR=2e-4 \
bash run_mave_hyperclova_tasks.sh
```

### MAVE eval

단일 checkpoint 평가:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAVE_TASK=faceted_search_filtering \
EVAL_SPLIT=test \
MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
LORA_ADAPTER=outputs/mave_hyperclovax_faceted_search_filtering_lora_1_5B_steps500_lr1e-4_r64/global_step_500 \
OUTPUT_DIR=checkpoints/verl_agent_mave/faceted_search_filtering/faceted_search_filtering_step500_test \
PARALLEL_ENVS=8 \
BATCH_SIZE=8 \
bash examples/validation/run_mave_lora_eval.sh
```

샘플 일부만 빠르게 평가:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAVE_TASK=faceted_search_filtering \
VAL_DATA_SIZE=200 \
START_INDEX=0 \
LORA_ADAPTER=outputs/mave_hyperclovax_faceted_search_filtering_lora_1_5B_steps500_lr1e-4_r64/global_step_500 \
OUTPUT_DIR=checkpoints/verl_agent_mave/debug_fsf_step500 \
bash examples/validation/run_mave_lora_eval.sh
```

keyword에 매칭되는 MAVE LoRA checkpoint 전체 평가:

```bash
DRY_RUN=1 \
LORA_ROOT=outputs \
OUTPUT_ROOT=checkpoints/verl_agent_mave \
bash eval_all_mave_loras.sh 0 faceted_search_filtering 1_5B
```

```bash
SKIP_EXISTING=1 \
LORA_ROOT=outputs \
OUTPUT_ROOT=checkpoints/verl_agent_mave \
EVAL_SPLIT=test \
bash eval_all_mave_loras.sh 0 faceted_search_filtering 1_5B
```

### MAVE plotting

`summary.json`들을 모아 CSV와 plot을 만든다.

```bash
python3 -m examples.validation.plot_mave_eval_by_step \
  --root checkpoints/verl_agent_mave \
  --output-dir plots/mave_eval_by_step \
  --task faceted_search_filtering \
  --metric primary_acc \
  --metric exact_match \
  --metric match_label_match \
  --metric json_parse_error_rate \
  --group-by run_id
```

모든 task를 task별 line으로 그리기:

```bash
python3 -m examples.validation.plot_mave_eval_by_step \
  --root checkpoints/verl_agent_mave \
  --output-dir plots/mave_eval_by_step_all \
  --group-by task
```
