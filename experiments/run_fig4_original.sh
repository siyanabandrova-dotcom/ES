#!/usr/bin/env bash
set -euo pipefail

# Run the original Fig.4 settings with unmodified EGGROLL code.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="/home/siana/eggroll-vllm/.venv"
export PYTHONPATH="/home/siana/eggroll_compat:${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing venv at $VENV_PATH. Install dependencies first."
  exit 1
fi

source "$VENV_PATH/bin/activate"

RAY_PORT="${RAY_PORT:-6379}"
RAY_IP="127.0.0.1"
RAY_ADDRESS="${RAY_IP}:${RAY_PORT}"

cleanup() {
  ray stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting Ray head at ${RAY_ADDRESS}..."
ray start --head --node-ip-address="$RAY_IP" --port="$RAY_PORT" \
  --num-cpus="$(nproc)" --num-gpus="$(nvidia-smi -L | wc -l)" --block &

sleep 5
export RAY_ADDRESS

python "$REPO_ROOT/es_lora_multinode.py" \
  --sigma 0.001 \
  --learning-rate 0.0002 \
  --max-tokens 1024 \
  --model-name "Qwen/Qwen3-4B" \
  --population-size 256 \
  --steps-per-adapter 4 \
  --lora-r 1 \
  --task "math:deepscaler40k" \
  --normalize-with-std \
  --prompt-batch-size 4 \
  --samples-per-prompt 1 \
  --temperature 0.0 \
  --steps-per-eval 5 \
  --name-prefix "fig4b_original_fast"
