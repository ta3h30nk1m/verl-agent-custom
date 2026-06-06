set -euo pipefail
set -x

ENGINE=${ENGINE:-vllm}
CKPT_PATH=${CKPT_PATH:-}
MODEL_PATH=${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}

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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

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

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-1}
val_data_size=${VAL_DATA_SIZE:-500}
val_batch_size=${VAL_BATCH_SIZE:-10}
n_gpus_per_node=${N_GPUS_PER_NODE:-1}
ray_num_cpus=${RAY_NUM_CPUS:-$((val_batch_size + 4))}
train_data_size=${TRAIN_DATA_SIZE:-$n_gpus_per_node}

if [[ "$num_cpus_per_env_worker" == "1" && "$ray_num_cpus" -le "$val_batch_size" ]]; then
    echo "RAY_NUM_CPUS=$ray_num_cpus leaves no CPU slot for actor/rollout workers when VAL_BATCH_SIZE=$val_batch_size." >&2
    echo "Use RAY_NUM_CPUS >= VAL_BATCH_SIZE + 2, e.g. RAY_NUM_CPUS=$((val_batch_size + 4))." >&2
    exit 1
fi

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
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$train_data_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$n_gpus_per_node" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    'trainer.logger=[console]' \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name=webshop_validation \
    trainer.n_gpus_per_node="$n_gpus_per_node" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    ray_init.num_cpus="$ray_num_cpus" \
    +ray_init.num_gpus="$n_gpus_per_node" \
    +ray_init.include_dashboard=False \
    "${resume_args[@]}" \
    "$@"
