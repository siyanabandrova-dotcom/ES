#!/bin/bash

#SBATCH --job-name=eggroll_vllm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00 
#SBATCH --output=/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/logs/multinode_n1-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1 

# --- Create logs directory if it doesn't exist ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Nodes involved: $SLURM_JOB_NODELIST"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs per node: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Log file: $LOG_DIR/es_lora_$SLURM_JOB_ID.log"
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
samples_per_prompt=${11}
temperature=${12}
pass_at_k=${13}
steps_per_eval=${14}
sub_dataset_size=${15}
name_prefix=${16}

# Run with:
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode_n1.sh <sigma> <learning_rate> <max_tokens> <model_name> <population_size> <steps_per_adapter> <lora_r> <task> <normalize_with_std> <prompt_batch_size> <samples_per_prompt> <temperature> <pass_at_k> <steps_per_eval> <sub_dataset_size> <name_prefix>

# Example for 2 nodes with 4 GPUs each (population_size=128, 16 per GPU):
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode_n1.sh 0.001 0.001 4096 "Qwen/Qwen3-4B" 1024 4 1 "math2:deepscaler40k" "normalize-with-std" 16 1 0.0 "no-pass-at-k" 10 "null" "multinode-test6_1"

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
echo "  samples_per_prompt: $samples_per_prompt"
echo "  temperature: $temperature"
echo "  pass_at_k: $pass_at_k"
echo "  steps_per_eval: $steps_per_eval"
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

# ==========================================
# === RAY CLUSTER SETUP (MULTI-NODE) ===
echo "Setting up Ray Cluster..."

# 1. Get the list of nodes and the head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 2. Port configuration
port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

echo "Head node: $head_node ($head_node_ip)"
echo "Ray Head IP: $ip_head"

# 3. Start Ray Head on the primary node
echo "Starting Ray Head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &

# 4. Wait briefly for head to initialize
sleep 10

# 5. Start Ray Workers on the remaining nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    for ((i=1; i<=worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray Worker on $node_i..."
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &
    done
fi

# 6. Wait for all nodes to register
echo "Waiting for Ray workers to connect..."
sleep 20
python -c "import ray; ray.init(address='auto'); print('Ray Cluster Resources:', ray.cluster_resources())"
# ==========================================


# --- Run the Python Script (Head Node Only) ---
echo "Starting Python script..."
# Note: The python script connects to the Ray cluster we just built
python es_lora_multinode.py \
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
    --samples-per-prompt $samples_per_prompt \
    --temperature $temperature \
    --${pass_at_k} \
    --steps-per-eval $steps_per_eval \
    $DATASET_SIZE_CMD \
    --name-prefix $name_prefix \
    --use-wandb

PYTHON_EXIT_CODE=$?
echo "---------------------------------"
if [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "Job timed out after 3 hours"
elif [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Job finished with error code $PYTHON_EXIT_CODE"
else
    echo "Job finished successfully"
fi
echo "---------------------------------"

# Clean up Ray cluster
echo "Stopping Ray cluster..."
ray stop
echo "Ray cluster stopped"