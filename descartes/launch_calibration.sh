#!/bin/bash

#SBATCH --job-name=grpo_calibration
#SBATCH --gpus=4                  # Request 4 GPUs
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=/home/s5e/asims.s5e/Documents/esvllm-outer/hyperscale-es-vllm/descartes/logs/calib1_%j.log
#SBATCH --cpus-per-task=16        # Ensure enough CPUs for vLLM/Ray

# --- Create logs directory ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/descartes/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"
echo "Log file: $LOG_DIR/calib_$SLURM_JOB_ID.log"
echo "---------------------------------"

# --- Parse Command-Line Arguments ---
model_name=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
prompt_batch_size=${2:-4}
samples_per_prompt=${3:-4}  # GRPO Group Size
learning_rate=${4:-1e-6}
task=${5:-"gsm8k"}
max_steps=${6:-100}
name_prefix=${7:-"calib-run"}

# Usage:
# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/descartes/launch_calibration.sh "Qwen/Qwen2.5-0.5B-Instruct" 4 4 0.000001 "gsm8k" 500 "experiment-1"

# --- Echo parameters for logging ---
echo "Parameters:"
echo "  model_name: $model_name"
echo "  prompt_batch_size: $prompt_batch_size"
echo "  samples_per_prompt: $samples_per_prompt"
echo "  learning_rate: $learning_rate"
echo "  task: $task"
echo "  max_steps: $max_steps"
echo "  name_prefix: $name_prefix"
echo "  num_gpus: $NUM_GPUS"
echo "---------------------------------"

# --- Activate Environment ---
echo "Activating virtual environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

# --- Set WandB directory ---
export WANDB_DIR=$SCRATCH/for_esvllm/wandb

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm/descartes

# --- Verify GPU visibility ---
echo "Verifying GPU visibility..."
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"

# --- Run the Python Script (Argparse Syntax) ---
echo "Starting Python script..."

python train_calibration.py \
    --model-name "$model_name" \
    --prompt-batch-size $prompt_batch_size \
    --samples-per-prompt $samples_per_prompt \
    --learning-rate $learning_rate \
    --task "$task" \
    --max-steps $max_steps \
    --num-gpus $NUM_GPUS \
    --name-prefix "$name_prefix" \
    --use-wandb

echo "---------------------------------"
echo "Job finished with exit code $?"
echo "---------------------------------"