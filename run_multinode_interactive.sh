#!/bin/bash

# Interactive multi-node launcher for testing
# Usage:
#   1. Request interactive session: srun --nodes=2 --gpus-per-node=4 --ntasks-per-node=1 --time=03:00:00 --pty /bin/bash --login
#   2. Run this script: bash run_multinode_interactive.sh

echo "==================================="
echo "Multi-Node ES LoRA Interactive Run"
echo "==================================="

# --- Setup environment ---
echo "Activating environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

echo "Setting directories..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm
export WANDB_DIR=$SCRATCH/for_esvllm/wandb

# --- Setup Ray Cluster ---
echo ""
echo "Setting up Ray cluster..."
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
HEAD_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(hostname --ip-address)
RAY_PORT=6379

echo "Starting Ray head on $HEAD_NODE..."
ray start --head --port=$RAY_PORT --num-cpus=0 --block &
sleep 10

if [ "$NUM_NODES" -gt 1 ]; then
    WORKER_NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -n +2)
    for NODE in $WORKER_NODES; do
        echo "Starting Ray worker on $NODE..."
        srun --nodes=1 --ntasks=1 -w $NODE \
            ray start --address=$HEAD_NODE_IP:$RAY_PORT --num-cpus=0 --block &
        sleep 5
    done
    sleep 10
fi

export RAY_HEAD_NODE_IP=$HEAD_NODE_IP

# --- Check GPU availability ---
echo ""
echo "Checking GPUs..."
GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE:-4} | sed 's/(.*//')
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
echo "Total: $TOTAL_GPUS GPUs across $NUM_NODES node(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --- Configuration ---
# IMPORTANT: population_size must be divisible by TOTAL_GPUS
# For 8 GPUs (2x4): use 64, 128, 256, etc.
# For 16 GPUs (2x8): use 128, 256, 512, etc.

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
    --name-prefix "multinode-debug" \
    --model-name "Qwen/Qwen3-1.7B" \
    --temperature 0.0 \
    --samples-per-prompt 1 \
    --population-size 64 \
    --normalize-with-std \
    --ray-address auto \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE

echo ""
echo "==================================="
echo "Run completed with exit code: $?"
echo "==================================="

# Cleanup
ray stop
