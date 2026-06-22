#!/usr/bin/env bash

_load_experiment_env_xtrace=0
case $- in
    *x*)
        _load_experiment_env_xtrace=1
        set +x
        ;;
esac

_load_experiment_env_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
_load_experiment_env_root=$(cd -- "${_load_experiment_env_dir}/.." >/dev/null 2>&1 && pwd)
EXPERIMENT_ENV_FILE=${EXPERIMENT_ENV_FILE:-"${_load_experiment_env_root}/.experiment_env"}

if [ -f "${EXPERIMENT_ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    . "${EXPERIMENT_ENV_FILE}"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN
fi

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
fi

if [ "${_load_experiment_env_xtrace}" = 1 ]; then
    set -x
fi

unset _load_experiment_env_xtrace
unset _load_experiment_env_dir
unset _load_experiment_env_root
