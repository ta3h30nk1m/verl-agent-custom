#!/usr/bin/env bash
set -xeuo pipefail

MAVE_TASK=faceted_search_filtering \
SAVE_PATH=${SAVE_PATH:-outputs/mave_hyperclovax_faceted_search_filtering_lora} \
bash examples/sft/mave/run_hyperclovax_lora_single_gpu.sh "$@"
