set -euo pipefail
set -x

ENGINE=${ENGINE:-vllm}
CKPT_PATH=${CKPT_PATH:-}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
    if [[ "$1" == *global_step_* || -d "$1/actor" ]]; then
        CKPT_PATH=$1
        shift
        if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
            MODEL_PATH=$1
            shift
        fi
    else
        MODEL_PATH=$1
        shift
    fi
fi

resume_args=(trainer.resume_mode=disable)
if [[ -n "$CKPT_PATH" ]]; then
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$CKPT_PATH"
    )
fi

if [[ -n "$CKPT_PATH" ]]; then
    echo "Validating checkpoint: $CKPT_PATH"
else
    echo "Validating base model without checkpoint: $MODEL_PATH"
fi

ulimit -u 65536
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -XX:ActiveProcessorCount=1 -XX:ParallelGCThreads=1 -XX:ConcGCThreads=1"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python3 - <<'PYTORCH_CUDA_CHECK'
import sys
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    if required_arch not in supported_arches:
        print(
            f"Current PyTorch ({torch.__version__}, CUDA {torch.version.cuda}) does not support GPU arch {required_arch}.",
            file=sys.stderr,
        )
        print(f"Supported arches: {supported_arches}", file=sys.stderr)
        print("Install a PyTorch build with CUDA 12.8+ for Blackwell/sm_120 GPUs.", file=sys.stderr)
        sys.exit(1)
PYTORCH_CUDA_CHECK

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
val_data_size=${VAL_DATA_SIZE:-4}
val_batch_size=${VAL_BATCH_SIZE:-1}
n_gpus_per_node=${N_GPUS_PER_NODE:-1}
ray_num_cpus=${RAY_NUM_CPUS:-$((val_batch_size + 4))}
train_data_size=${TRAIN_DATA_SIZE:-$n_gpus_per_node}
validation_data_dir=${VALIDATION_DATA_DIR:-checkpoints/verl_agent_webshop/webshop_validation/validation_generations}
webshop_prompt_style=${WEBSHOP_PROMPT_STYLE:-direct}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-8}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-True}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}

python3 - "$num_cpus_per_env_worker" "$val_batch_size" "$ray_num_cpus" <<'PY_RAY_CPU_BUDGET_CHECK'
import sys

num_cpus_per_env_worker = float(sys.argv[1])
val_batch_size = int(sys.argv[2])
ray_num_cpus = float(sys.argv[3])
required = val_batch_size * num_cpus_per_env_worker + 1.0

if ray_num_cpus < required:
    print(
        f"RAY_NUM_CPUS={ray_num_cpus:g} is too small for "
        f"VAL_BATCH_SIZE={val_batch_size} and "
        f"NUM_CPUS_PER_ENV_WORKER={num_cpus_per_env_worker:g}.",
        file=sys.stderr,
    )
    print(
        f"Use RAY_NUM_CPUS >= {required:g}, or reduce NUM_CPUS_PER_ENV_WORKER.",
        file=sys.stderr,
    )
    sys.exit(1)
PY_RAY_CPU_BUDGET_CHECK

# We only use data preparation to indicate text modality and validation size.
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"

python3 -m verl.trainer.main_webshop_validation \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_batch_size" \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$train_data_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload="$actor_param_offload" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$actor_optimizer_offload" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$n_gpus_per_node" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_num_seqs="$rollout_max_num_seqs" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$rollout_max_num_batched_tokens" \
    actor_rollout_ref.rollout.max_model_len="$rollout_max_model_len" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager="$rollout_enforce_eager" \
    actor_rollout_ref.rollout.free_cache_engine="$rollout_free_cache_engine" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    env.webshop.prompt_style="$webshop_prompt_style" \
    'trainer.logger=[console]' \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name=webshop_validation \
    trainer.n_gpus_per_node="$n_gpus_per_node" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.validation_data_dir="$validation_data_dir" \
    ray_init.num_cpus="$ray_num_cpus" \
    +ray_init.num_gpus="$n_gpus_per_node" \
    +ray_init.include_dashboard=False \
    "${resume_args[@]}" \
    "$@"
