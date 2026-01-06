#!/bin/bash

# Interactive multi-GPU launcher for testing
# Usage:
#   1. Request interactive session: srun --gpus=4 --time=03:00:00 --pty /bin/bash --login
#   2. Run this script: bash run_multigpu_interactive.sh

echo "==================================="
echo "Multi-GPU ES LoRA Interactive Run"
echo "==================================="

# --- Setup environment ---
echo "Activating environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

echo "Setting directories..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm
export WANDB_DIR=$SCRATCH/for_esvllm/wandb

# --- Check GPU availability ---
echo ""
echo "Checking GPUs..."
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch sees $NUM_GPUS GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --- Configuration ---
# IMPORTANT: population_size must be divisible by NUM_GPUS
# For 4 GPUs: use 16, 32, 64, 128, etc.
# For 2 GPUs: use 16, 32, 64, etc.

echo ""
echo "==================================="
echo "Running ES training..."
echo "==================================="

python es_lora_nccl_async4.py \
    --sigma 0.001 \
    --learning-rate 0.001 \
    --max-tokens 1024 \
    --steps-per-adapter 4 \
    --task math2:deepscaler40k \
    --prompt-batch-size 16 \
    --no-use-wandb \
    --name-prefix "multigpu-debug" \
    --model-name "Qwen/Qwen3-1.7B" \
    --temperature 0.0 \
    --samples-per-prompt 1 \
    --population-size 64 \
    --normalize-with-std

echo ""
echo "==================================="
echo "Run completed with exit code: $?"
echo "==================================="
