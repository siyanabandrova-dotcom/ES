#!/bin/bash

#SBATCH --job-name=es_lora_nccl   # A name for your job
#SBATCH --gpus=1                  # Request 1 GPU (adjust as needed)
#SBATCH --time=24:00:00           # Time limit hrs:min:sec (from your srun)
#SBATCH --output=/home/s5e/asims.s5e/Documents/esvllm-outer/hyperscale-es-vllm/logs/es_lora_%j.log    # Log file path (%u is user, %j is job ID)

# --- Create logs directory if it doesn't exist ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Log file: $LOG_DIR/es_lora_$SLURM_JOB_ID.log"
echo "---------------------------------"

# --- Parse Command-Line Arguments ---
# Default values from your original command/Args class are used as examples
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
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k" no-normalize-with-std 16 16 "A"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k-boxed" no-normalize-with-std 16 16 "A"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "countdown" no-normalize-with-std 16 16 "A"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "zeros" no-normalize-with-std 3 3 "A"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.0001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k-boxed" no-normalize-with-std 16 16 "Alr"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 64 "Qwen/Qwen3-0.6B" 128 4 4 "zeros" no-normalize-with-std 3 3 "B"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.01 0.01 64 "Qwen/Qwen3-0.6B" 128 4 4 "zeros" no-normalize-with-std 3 3 "B"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.01 0.001 64 "Qwen/Qwen3-0.6B" 128 4 4 "zeros" no-normalize-with-std 3 3 "B"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.01 64 "Qwen/Qwen3-0.6B" 128 4 4 "zeros" no-normalize-with-std 3 3 "B"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 64 "Qwen/Qwen2.5-0.5B" 128 4 4 "zeros" no-normalize-with-std 3 3 "B"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k-boxed" no-normalize-with-std 16 1000 "C"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 64 4 4 "gsm8k-boxed" no-normalize-with-std 16 1000 "D64"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 256 4 4 "gsm8k-boxed" no-normalize-with-std 16 1000 "D256"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k-boxed" normalize-with-std 16 1000 "Estd"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k" normalize-with-std 16 "null" "F"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "gsm8k-boxed" normalize-with-std 16 "null" "F"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B" 128 4 4 "countdown" normalize-with-std 16 "null" "F"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B-Base" 128 4 4 "gsm8k" normalize-with-std 16 "null" "E"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B-Base" 128 4 4 "gsm8k-boxed" normalize-with-std 16 "null" "E"
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch.sh 0.001 0.001 1024 "Qwen/Qwen3-4B-Base" 128 4 4 "countdown" normalize-with-std 16 "null" "E"

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
# (Using the environment from your srun command)
echo "Activating virtual environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

# --- Change to Working Directory ---
# (Using the directory from your srun command)
echo "Changing to working directory..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm

# --- Run the Python Script ---
echo "Starting Python script..."
python es_lora_nccl_async.py \
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
    --use-wandb

echo "---------------------------------"
echo "Job finished with exit code $?"
echo "---------------------------------"

# See readme for example of how to submit.
