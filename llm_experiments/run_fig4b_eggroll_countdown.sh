#!/usr/bin/env bash
set -euo pipefail

# Figure 4b EGGROLL countdown run (RWKV 7g1.5B, single GPU).
# Requires: pip install -e . in this repo and a local .venv with JAX/CUDA.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"

cd "$REPO_ROOT"

exec "$VENV_PYTHON" -m llm_experiments.general_do_evolution \
  --task countdownn \
  --noiser eggroll \
  --model-choice 7g1.5B \
  --parallel-generations-per-gpu 1536 \
  --generations-per-prompt 256 \
  --sigma 7e-4 \
  --lr-scale 0.125 \
  --seed 0 \
  --temperature 0.0 \
  --parallel-validations 128 \
  --thinking-length 1000 \
  --answer-length 0
