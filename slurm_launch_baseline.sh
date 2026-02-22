#!/bin/bash

#SBATCH --job-name=baseline_qwen8b_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/s5j/alv31415.s5j/logs/hyperscale-es-vllm/baseline_single-%j.log
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonio.leonvillares@stx.ox.ac.uk

# --- Create logs directory if it doesn't exist ---
LOG_DIR="/scratch/s5j/alv31415.s5j/logs/hyperscale-es-vllm/"
mkdir -p "$LOG_DIR"

echo "---------------------------------"
echo "Starting BASELINE EVALUATION (SINGLE NODE) job $SLURM_JOB_ID on $(hostname)"
echo "Node: $SLURM_JOB_NODELIST"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Log file: $LOG_DIR/baseline_single-$SLURM_JOB_ID.log"
echo "---------------------------------"

# -----------------------------------------
# User-settable parameters (edit these)
# -----------------------------------------
max_tokens="4096"
model_name="Qwen/Qwen3-8B"
task="math2:deepscaler40k"
prompt_batch_size="16"
samples_per_prompt="1"
temperature="0.0"
pass_at_k="no-pass-at-k"
steps_per_eval="10"
# Set to "null" or "None" or empty string to use full dataset
sub_dataset_size="null"
name_prefix="baseline-qwen3-8b-deepscaler-single-eval"
num_iterations="300"
tensor_parallel_size="1"  # 4 GPUs for 110B model

# -----------------------------------------

# --- Echo parameters for logging ---
echo "Parameters:"
echo "  model_name: $model_name"
echo "  task: $task"
echo "  max_tokens: $max_tokens"
echo "  prompt_batch_size: $prompt_batch_size"
echo "  samples_per_prompt: $samples_per_prompt"
echo "  temperature: $temperature"
echo "  pass_at_k: $pass_at_k"
echo "  steps_per_eval: $steps_per_eval"
echo "  sub_dataset_size: $sub_dataset_size"
echo "  name_prefix: $name_prefix"
echo "  num_iterations: $num_iterations"
echo "  tensor_parallel_size: $tensor_parallel_size"
echo "---------------------------------"

if [[ "$sub_dataset_size" == "None" ]] || [[ "$sub_dataset_size" == "null" ]] || [[ -z "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD=""
else
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

# --- Activate Environment ---
echo "Activating virtual environment..."
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"

# --- Set WandB directory ---
export WANDB_DIR="$SCRATCH/for_esvllm/wandb"

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd "$HOME/hyperscale/hyperscale-es-vllm" || exit 1

# --- Run the Python Script ---
echo "Starting Baseline Evaluation Python script (Single Node)..."

# Build flag strings for optional flags (only add if non-empty)
PASSATK_FLAG=""
if [[ -n "$pass_at_k" ]]; then
    PASSATK_FLAG="--${pass_at_k}"
fi

python baseline_eval.py \
    --max-tokens "$max_tokens" \
    --model-name "$model_name" \
    --task "$task" \
    --prompt-batch-size "$prompt_batch_size" \
    --samples-per-prompt "$samples_per_prompt" \
    --temperature "$temperature" \
    $PASSATK_FLAG \
    --steps-per-eval "$steps_per_eval" \
    $DATASET_SIZE_CMD \
    --name-prefix "$name_prefix" \
    --num-iterations "$num_iterations" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --use-wandb

PYTHON_EXIT_CODE=$?
echo "---------------------------------"
if [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "Job timed out"
elif [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Job finished with error code $PYTHON_EXIT_CODE"
else
    echo "Job finished successfully"
fi
echo "---------------------------------"