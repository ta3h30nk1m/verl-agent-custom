#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

BASE_ENV="${BASE_ENV:-verl-agent-webshop}"
ENV_NAME="${ENV_NAME:-verl-agent-webshop-vllm3b}"
VLLM_SRC="${VLLM_SRC:-${ROOT_DIR}/third_party/vllm-hyperclovax-vision-seed}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/NAVER-Cloud-HyperCLOVA-X/vllm.git}"
VLLM_REPO_BRANCH="${VLLM_REPO_BRANCH:-v0.9.2rc2_hyperclovax_vision_seed}"
RECREATE="${RECREATE:-0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
VLLM_SKIP_FA3="${VLLM_SKIP_FA3:-1}"
VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
FORCE_REINSTALL_VLLM="${FORCE_REINSTALL_VLLM:-0}"

if ! command -v conda >/dev/null 2>&1; then
    CONDA_SH="${HOME}/miniforge3/etc/profile.d/conda.sh"
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "conda was not found. Install miniforge/conda or set PATH first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
fi

if [[ ! -f "${VLLM_SRC}/setup.py" ]]; then
    if [[ -e "${VLLM_SRC}" ]]; then
        echo "vLLM source path exists but setup.py was not found: ${VLLM_SRC}" >&2
        echo "Remove that path or set VLLM_SRC=/path/to/a/valid/vllm checkout." >&2
        exit 1
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo "git was not found, but ${VLLM_SRC} needs to be cloned." >&2
        exit 1
    fi
    echo "[setup] cloning patched HyperCLOVAX vLLM source"
    mkdir -p "$(dirname "${VLLM_SRC}")"
    git clone --branch "${VLLM_REPO_BRANCH}" --depth 1 "${VLLM_REPO_URL}" "${VLLM_SRC}"
fi

if [[ ! -f "${VLLM_SRC}/setup.py" ]]; then
    echo "vLLM source not found at ${VLLM_SRC}" >&2
    echo "Set VLLM_SRC=/path/to/vllm-hyperclovax-vision-seed if you keep it elsewhere." >&2
    exit 1
fi

ensure_vllm_skip_fa3_patch() {
    if grep -q "VLLM_SKIP_FA3" "${VLLM_SRC}/setup.py"; then
        return
    fi

    echo "[setup] patching vLLM setup.py to support VLLM_SKIP_FA3=1"
    python - "${VLLM_SRC}/setup.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
old = '''    if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
        ext_modules.append(
            CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
'''
new = '''    skip_fa3 = os.environ.get("VLLM_SKIP_FA3", "0").lower() in {
        "1", "true", "yes", "y", "on"
    }
    if not skip_fa3 and (envs.VLLM_USE_PRECOMPILED
                         or get_nvcc_cuda_version() >= Version("12.3")):
        ext_modules.append(
            CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
'''
if old not in source:
    raise SystemExit("Could not patch setup.py automatically; FA3 block pattern was not found.")
path.write_text(source.replace(old, new))
PY
}

ensure_vllm_skip_fa3_patch

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

if [[ "${RECREATE}" == "1" ]] && env_exists "${ENV_NAME}"; then
    echo "[setup] removing existing conda env: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

if env_exists "${ENV_NAME}"; then
    echo "[setup] conda env already exists: ${ENV_NAME}"
else
    if ! env_exists "${BASE_ENV}"; then
        echo "base conda env does not exist: ${BASE_ENV}" >&2
        echo "Create ${BASE_ENV} first, or run with BASE_ENV=<existing-env>." >&2
        exit 1
    fi
    echo "[setup] cloning ${BASE_ENV} -> ${ENV_NAME}"
    conda create -n "${ENV_NAME}" --clone "${BASE_ENV}" -y
fi

PIP_INSTALL_ARGS=(install --no-build-isolation -e "${VLLM_SRC}")
if [[ "${FORCE_REINSTALL_VLLM}" == "1" ]]; then
    PIP_INSTALL_ARGS+=(--force-reinstall)
fi

echo "[setup] installing patched HyperCLOVAX vLLM from ${VLLM_SRC}"
env \
    MAX_JOBS="${MAX_JOBS}" \
    VLLM_SKIP_FA3="${VLLM_SKIP_FA3}" \
    VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
    conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip "${PIP_INSTALL_ARGS[@]}"

echo "[setup] pinning packages known to work with the patched vLLM build"
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install "mistral_common==1.6.2"

# External flash-attn wheels can be ABI-incompatible with this Torch/vLLM pair.
# vLLM's in-tree flash-attention extension is used instead.
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip uninstall -y flash-attn flash_attn || true

mkdir -p "${ROOT_DIR}/.cache/huggingface/modules"

echo "[setup] verifying imports"
conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import torch
import vllm

from mistral_common.protocol.instruct.messages import ImageChunk, TextChunk

import vllm._C
import vllm._moe_C
import vllm.cumem_allocator
import vllm.model_executor.models.hyperclovax_vision
import vllm.model_executor.models.pixtral
import vllm.vllm_flash_attn._vllm_fa2_C

print("python import check: ok")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("mistral_common ImageChunk/TextChunk:", ImageChunk.__name__, TextChunk.__name__)
PY

cat <<EOF

[setup] done

Use it with:
  conda activate ${ENV_NAME}

For HyperCLOVAX Vision eval, keep this in your shell or eval script:
  export HF_MODULES_CACHE=${ROOT_DIR}/.cache/huggingface/modules

Useful rebuild options:
  RECREATE=1 bash ${SCRIPT_NAME}
  FORCE_REINSTALL_VLLM=1 bash ${SCRIPT_NAME}
  VLLM_SRC=/path/to/vllm-hyperclovax-vision-seed bash ${SCRIPT_NAME}
  VLLM_REPO_BRANCH=<branch> bash ${SCRIPT_NAME}
EOF
