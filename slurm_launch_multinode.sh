#!/bin/bash

#SBATCH --job-name=es_lora_multinode
#SBATCH --nodes=1                 # Request 2 nodes (adjust as needed)
#SBATCH --gpus-per-task=4         # GPUs per node (adjust as needed)
#SBATCH --time=24:00:00            # Time limit hrs:min:sec
#SBATCH --output=/home/s5e/asims.s5e/Documents/esvllm-outer/hyperscale-es-vllm/logs/async4-1node_%j.log
#SBATCH --cpus-per-task=288       # Ensure enough CPUs for Ray actors
#SBATCH --ntasks-per-node=1       # One task per node for Ray


# --- Create logs directory if it doesn't exist ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting multi-node job $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Log file: $LOG_DIR/es_lora_multinode_$SLURM_JOB_ID.log"
echo "---------------------------------"

# --- Parse Command-Line Arguments ---
sigma=${1}
learning_rate=${2}
max_tokens=${3}
model_name=${4}
population_size=${5}
steps_per_adapter=${6}
lora_r=${7}
task=${8}
normalize_with_std=${9}
prompt_batch_size=${10}
sub_dataset_size=${11}
name_prefix=${12}

# Run with:
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode.sh <sigma> <learning_rate> <max_tokens> <model_name> <population_size> <steps_per_adapter> <lora_r> <task> <normalize_with_std> <prompt_batch_size> <sub_dataset_size> <name_prefix>

# Example for 2 nodes with 4 GPUs each (population_size=128, 16 per GPU):
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode.sh 0.001 0.001 4096 "Qwen/Qwen3-4B" 1024 4 1 "math2:deepscaler40k" "normalize-with-std" 16 "null" "async4-1node"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode.sh 0.001 0.001 1024 "Qwen/Qwen3-0.6B" 128 4 1 "math2:deepscaler40k" "normalize-with-std" 16 "null" "async4-1node-debug"

# --- Echo parameters for logging ---
echo "Parameters:"
echo "  sigma: $sigma"
echo "  learning_rate: $learning_rate"
echo "  max_tokens: $max_tokens"
echo "  model_name: $model_name"
echo "  population_size: $population_size"
echo "  steps_per_adapter: $steps_per_adapter"
echo "  lora_r: $lora_r"
echo "  task: $task"
echo "  normalize_with_std: $normalize_with_std"
echo "  prompt_batch_size: $prompt_batch_size"
echo "  sub_dataset_size: $sub_dataset_size"
echo "  name_prefix: $name_prefix"
echo "---------------------------------"

if [[ "$sub_dataset_size" == "None" ]] || [[ "$sub_dataset_size" == "null" ]] || [[ -z "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD=""
else
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

# --- Activate Environment ---
echo "Activating virtual environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

# --- Set WandB directory ---
export WANDB_DIR=$SCRATCH/for_esvllm/wandb

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm

# --- Setup Ray Cluster ---
echo "Setting up Ray cluster across nodes..."

# Get the head node hostname
HEAD_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Head node: $HEAD_NODE"

# Get IP address of head node
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname --ip-address)
echo "Head node IP: $HEAD_NODE_IP"

# Ray port
RAY_PORT=6379

# Start Ray head node
echo "Starting Ray head node on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    ray start --head --port=$RAY_PORT --num-cpus=0 \
    --block &

# Give head node time to start
sleep 10

# Start Ray worker nodes
WORKER_NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -n +2)
for NODE in $WORKER_NODES; do
    echo "Starting Ray worker on $NODE..."
    srun --nodes=1 --ntasks=1 -w $NODE \
        ray start --address=$HEAD_NODE_IP:$RAY_PORT --num-cpus=0 \
        --block &
    sleep 5
done

# Wait for all nodes to join
sleep 15

echo "Ray cluster is ready!"
echo "Checking Ray cluster status..."
ray status

# Export head node IP for use in Python script
export RAY_HEAD_NODE_IP=$HEAD_NODE_IP

# --- Verify GPU visibility on head node ---
echo "Verifying GPU visibility on head node..."
GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch sees $GPUS_PER_NODE GPUs on head node"

# --- Calculate number of nodes and GPUs ---
NUM_NODES=$SLURM_JOB_NUM_NODES

echo "Running with $NUM_NODES nodes and $GPUS_PER_NODE GPUs per node"

# --- Run the Python Script on Head Node ---
echo "Starting Python script on head node..."
python es_lora_nccl_async4.py \
    --sigma $sigma \
    --learning-rate $learning_rate \
    --max-tokens $max_tokens \
    --model-name $model_name \
    --population-size $population_size \
    --steps-per-adapter $steps_per_adapter \
    --lora-r $lora_r \
    --task $task \
    --${normalize_with_std} \
    --prompt-batch-size $prompt_batch_size \
    $DATASET_SIZE_CMD \
    --name-prefix $name_prefix \
    --use-wandb \
    --ray-address auto \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE

# Capture exit code
EXIT_CODE=$?

# --- Cleanup Ray Cluster ---
echo "Shutting down Ray cluster..."
ray stop

echo "---------------------------------"
echo "Job finished with exit code $EXIT_CODE"
echo "---------------------------------"

exit $EXIT_CODE
