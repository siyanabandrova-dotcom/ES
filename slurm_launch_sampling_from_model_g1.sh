#!/bin/bash

#SBATCH --job-name=eggroll_vllm_sample
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/logs/sampling-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# --- Create logs directory if it doesn't exist ---
LOG_DIR="$HOME/Documents/esvllm-outer/hyperscale-es-vllm/logs"
mkdir -p $LOG_DIR

echo "---------------------------------"
echo "Starting SAMPLING job $SLURM_JOB_ID on $(hostname)"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "---------------------------------"

# --- Parse Command-Line Arguments (Preserved for compatibility) ---
sigma=${1}            # Unused in sampling
learning_rate=${2}    # Unused in sampling
max_tokens=${3}
model_name=${4}
population_size=${5}  # Unused in sampling
steps_per_adapter=${6} # Unused in sampling
lora_r=${7}           # Unused in sampling
task=${8}
normalize_with_std=${9} # Unused in sampling
prompt_batch_size=${10}
samples_per_prompt=${11}
temperature=${12}
pass_at_k=${13}       # Expects "pass-at-k" or "no-pass-at-k"
steps_per_eval=${14}  # Unused in sampling
sub_dataset_size=${15}
name_prefix=${16}     # Unused in sampling

# --- Handle Sub-dataset Size ---
if [[ "$sub_dataset_size" == "None" ]] || [[ "$sub_dataset_size" == "null" ]] || [[ -z "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD=""
else
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

# --- Handle Pass@k Flag ---
# Tyro expects --pass-at-k or --no-pass-at-k.
# Assuming input is "pass-at-k" or "no-pass-at-k", we just prepend --
PASS_AT_K_CMD="--${pass_at_k}"

# --- Determine TP Size ---
# Automatically set Tensor Parallelism to the number of GPUs on this node
TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# --- Activate Environment ---
echo "Activating virtual environment..."
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm

# --- Run the Python Script ---
echo "Starting Simplified Sampling Script..."
echo "Model: $model_name"
echo "Task: $task"
echo "TP Size: $TP_SIZE"

python sampling_from_model.py \
    --model-name "$model_name" \
    --task "$task" \
    --batch-size $prompt_batch_size \
    --max-tokens $max_tokens \
    --temperature $temperature \
    --samples-per-prompt $samples_per_prompt \
    --tensor-parallel-size $TP_SIZE \
    $PASS_AT_K_CMD \
    $DATASET_SIZE_CMD

PYTHON_EXIT_CODE=$?

echo "---------------------------------"
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Job finished with error code $PYTHON_EXIT_CODE"
else
    echo "Job finished successfully"
fi
echo "---------------------------------"


# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode_just_sampling_g1.sh 0.001 0.001 32 "/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/tmp/merged_models/D3-drawegg-jsd-nopenalty-drawegg-boxed-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768928661" 32 4 1 "drawegg-boxed-jsd" "normalize-with-std" 1 20480 1.0 "no-pass-at-k" 10 "null" "D3-drawegg-jsd-nopenalty"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode_just_sampling_g1.sh 0.001 0.001 32 "/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/tmp/merged_models/D3-drawegg-jsd-nopenalty-drawegg-boxed-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768934132" 32 4 1 "drawegg-boxed-jsd" "normalize-with-std" 1 20480 1.0 "no-pass-at-k" 10 "null" "D3-drawegg-jsd-nopenalty"

# sbatch $HOME/Documents/esvllm-outer/hyperscale-es-vllm/slurm_launch_multinode_just_sampling_g1.sh 0.001 0.001 32 "/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/tmp/merged_models/D3-drawchick-jsd-nopenalty-drawchick-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768927361" 32 4 1 "drawchick-jsd" "normalize-with-std" 1 20480 1.0 "no-pass-at-k" 10 "null" "D3-drawchick-jsd-nopenalty"