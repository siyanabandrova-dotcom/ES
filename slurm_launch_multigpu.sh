#!/bin/bash

#SBATCH --job-name=eggroll_vllm
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/s5e/asims.s5e/Documents/esvllm-outer/hyperscale-es-vllm/logs/multigpu-%j.log
#SBATCH --cpus-per-task=16

# --- Create logs directory if it doesn't exist ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
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
sub_dataset_size=${11}
name_prefix=${12}

# Run with:
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multigpu.sh <sigma> <learning_rate> <max_tokens> <model_name> <population_size> <steps_per_adapter> <lora_r> <task> <normalize_with_std> <prompt_batch_size> <sub_dataset_size> <name_prefix>

# Example for 4 GPUs with population_size=64 (16 per GPU):
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multigpu.sh 0.001 0.001 4096 "Qwen/Qwen3-4B" 1024 1 4 "math2:deepscaler40k" "normalize-with-std" 16 "null" "multigpu-test1"

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

# --- Verify GPU visibility ---
echo "Verifying GPU visibility..."
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"

# --- Run the Python Script ---
echo "Starting Python script..."
python es_lora_multigpu.py \
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
