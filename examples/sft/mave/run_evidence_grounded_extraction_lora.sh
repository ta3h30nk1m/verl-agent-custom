#!/usr/bin/env bash
set -xeuo pipefail

MAVE_TASK=evidence_grounded_extraction \
SAVE_PATH=${SAVE_PATH:-outputs/mave_hyperclovax_evidence_grounded_extraction_lora} \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
