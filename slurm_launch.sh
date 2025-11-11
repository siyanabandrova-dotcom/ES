#!/bin/bash

#SBATCH --job-name=es_lora_nccl   # A name for your job
#SBATCH --gpus=1                  # Request 1 GPU (adjust as needed)
#SBATCH --time=03:00:00           # Time limit hrs:min:sec (from your srun)
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
prompt_batch_size=${9}
sub_dataset_size=${10}
name_prefix=${11}

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
echo "  prompt_batch_size: $prompt_batch_size"
echo "  name_prefix: $name_prefix"
echo "---------------------------------"

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
    --model-name "$model_name" \
    --population-size $population_size \
    --steps-per-adapter $steps_per_adapter \
    --lora-r $lora_r \
    --task $task \
    --prompt-batch-size $prompt_batch_size \
    --sub-dataset-size $sub_dataset_size \
    --name-prefix "$name_prefix" \
    --use-wandb

echo "---------------------------------"
echo "Job finished with exit code $?"
echo "---------------------------------"

# See readme for example of how to submit.

# run with:
# sbatch slurm_launch.sh 0.01 0.01 1024 "Qwen/Qwen2-0.5B" 1000 10 4 "gsm8k" 32 1000 "time2"