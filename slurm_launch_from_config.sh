#!/bin/bash

# =============================================================================
# Config-Based Experiment Launcher
# =============================================================================
# Reads experiment configurations from a CSV file and launches jobs

set -e

# --- Configuration ---
BASE_SCRIPT="slurm_launch_base.sh"
TEMPLATE_FILE="experiments_template.csv" # Set to "" to skip expansion and use CONFIG_FILE directly
CONFIG_FILE="experiments.csv"
EXPAND_SCRIPT="expand_config.py"
DELAY_SECONDS=5
EXPERIMENT_DIR="/scratch/s5e/alv31415.s5e/experiments"
USE_DEPENDENCIES=false  # Set to false to use real-time waiting instead

# =============================================================================
# Template Expansion
# =============================================================================

if [[ -n "$TEMPLATE_FILE" ]]; then
    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo "ERROR: Template file '$TEMPLATE_FILE' not found!"
        exit 1
    fi
    if [ ! -f "$EXPAND_SCRIPT" ]; then
        echo "ERROR: Expand script '$EXPAND_SCRIPT' not found!"
        exit 1
    fi
    echo "Expanding template '$TEMPLATE_FILE' → '$CONFIG_FILE'..."
    python3 "$EXPAND_SCRIPT" "$TEMPLATE_FILE" "$CONFIG_FILE"
    echo "Expansion complete."
    echo ""
fi

# =============================================================================
# Functions
# =============================================================================

create_experiment_script() {
    local sigma=$1
    local lr=$2
    local max_tokens=$3
    local model=$4
    local pop_size=$5
    local prompt_bs=$6
    local name=$7
    local normalize_with_std=$8
    local scale_lr_in_grad=$9
    local num_nodes=${10}
    local gpus_per_node=${11}
    local task=${12}
    local output_script=${13}
    
    cp "$BASE_SCRIPT" "$output_script"
    
    # Override SBATCH directives for node/GPU count
    local cpus_per_task=$((gpus_per_node * 72))
    sed -i "s|^#SBATCH --nodes=.*|#SBATCH --nodes=$num_nodes|" "$output_script"
    sed -i "s|^#SBATCH --gpus-per-node=.*|#SBATCH --gpus-per-node=$gpus_per_node|" "$output_script"
    sed -i "s|^#SBATCH --cpus-per-task=.*|#SBATCH --cpus-per-task=$cpus_per_task|" "$output_script"

    # Override user-settable variables
    sed -i "s|^num_nodes=.*|num_nodes=\"$num_nodes\"|" "$output_script"
    sed -i "s|^gpus_per_node=.*|gpus_per_node=\"$gpus_per_node\"|" "$output_script"
    sed -i "s|^sigma=.*|sigma=\"$sigma\"|" "$output_script"
    sed -i "s|^learning_rate=.*|learning_rate=\"$lr\"|" "$output_script"
    sed -i "s|^max_tokens=.*|max_tokens=\"$max_tokens\"|" "$output_script"
    sed -i "s|^model_name=.*|model_name=\"$model\"|" "$output_script"
    sed -i "s|^population_size=.*|population_size=\"$pop_size\"|" "$output_script"
    sed -i "s|^prompt_batch_size=.*|prompt_batch_size=\"$prompt_bs\"|" "$output_script"
    sed -i "s|^name_prefix=.*|name_prefix=\"$name\"|" "$output_script"
    sed -i "s|^normalize_with_std=.*|normalize_with_std=\"$normalize_with_std\"|" "$output_script"
    sed -i "s|^scale_lr_in_grad=.*|scale_lr_in_grad=\"$scale_lr_in_grad\"|" "$output_script"
    sed -i "s|^task=.*|task=\"$task\"|" "$output_script"
    
    chmod +x "$output_script"
}

submit_job_with_dependency() {
    local script=$1
    local dependency=$2
    local begin_offset_seconds=$3  # seconds from now to start

    local begin_flag="--begin=now+${begin_offset_seconds}seconds"

    if [ -z "$dependency" ]; then
        job_output=$(sbatch "$begin_flag" "$script")
    else
        job_output=$(sbatch "$begin_flag" --dependency=after:$dependency "$script")
    fi

    echo "$job_output" | grep -oP '\d+'
}

submit_job() {
    local script=$1
    sbatch "$script" | grep -oP '\d+'
}

# =============================================================================
# Main Execution
# =============================================================================

mkdir -p "$EXPERIMENT_DIR"

echo "=========================================="
echo "Config-Based Experiment Launcher"
echo "=========================================="
echo "Base script: $BASE_SCRIPT"
echo "Config file: $CONFIG_FILE"
echo "Delay between jobs: $DELAY_SECONDS seconds"
echo "Using dependencies: $USE_DEPENDENCIES"
echo "=========================================="
echo ""

# Check files exist
if [ ! -f "$BASE_SCRIPT" ]; then
    echo "ERROR: Base script '$BASE_SCRIPT' not found!"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Read experiments from config file
experiments=()

while IFS=',' read -r sigma lr max_tokens model pop_size prompt_bs name normalize_with_std scale_lr_in_grad num_nodes gpus_per_node task || [ -n "$sigma" ]; do
    # Skip comments and empty lines
    [[ "$sigma" =~ ^#.*$ ]] && continue
    [[ -z "$sigma" ]] && continue
    
    # Trim whitespace
    sigma=$(echo "$sigma" | xargs)
    lr=$(echo "$lr" | xargs)
    max_tokens=$(echo "$max_tokens" | xargs)
    model=$(echo "$model" | xargs)
    pop_size=$(echo "$pop_size" | xargs)
    prompt_bs=$(echo "$prompt_bs" | xargs)
    name=$(echo "$name" | xargs)
    normalize_with_std=$(echo "$normalize_with_std" | xargs)
    scale_lr_in_grad=$(echo "$scale_lr_in_grad" | xargs)
    num_nodes=$(echo "$num_nodes" | xargs)
    gpus_per_node=$(echo "$gpus_per_node" | xargs)
    task=$(echo "$task" | xargs)

    # Default to 32 nodes / 4 GPUs if not specified
    num_nodes=${num_nodes:-32}
    gpus_per_node=${gpus_per_node:-4}
    
    experiments+=("$sigma|$lr|$max_tokens|$model|$pop_size|$prompt_bs|$name|$normalize_with_std|$scale_lr_in_grad|$num_nodes|$gpus_per_node|$task")
done < "$CONFIG_FILE"

echo "Loaded ${#experiments[@]} experiments from config file"
echo ""

if [ ${#experiments[@]} -eq 0 ]; then
    echo "ERROR: No valid experiments found in config file!"
    exit 1
fi

submitted_jobs=()
previous_job_id=""

for i in "${!experiments[@]}"; do
    experiment_num=$((i + 1))
    
    IFS='|' read -r sigma lr max_tokens model pop_size prompt_bs name normalize_with_std scale_lr_in_grad num_nodes gpus_per_node task <<< "${experiments[$i]}"
    
    echo "=========================================="
    echo "Experiment $experiment_num/${#experiments[@]}: $name"
    echo "=========================================="
    echo "Parameters:"
    echo "  sigma: $sigma"
    echo "  learning_rate: $lr"
    echo "  max_tokens: $max_tokens"
    echo "  model_name: $model"
    echo "  population_size: $pop_size"
    echo "  prompt_batch_size: $prompt_bs"
    echo "  normalize_with_std: $normalize_with_std"
    echo "  scale_lr_in_grad: $scale_lr_in_grad"
    echo "  num_nodes: $num_nodes"
    echo "  gpus_per_node: $gpus_per_node"
    echo "  cpus_per_task (derived): $((gpus_per_node * 72))"
    echo "  task: $task"
    echo ""
    
    experiment_script="$EXPERIMENT_DIR/exp_${name}_$(date +%Y%m%d_%H%M%S)_$i.sh"
    
    create_experiment_script "$sigma" "$lr" "$max_tokens" "$model" "$pop_size" "$prompt_bs" "$name" "$normalize_with_std" "$scale_lr_in_grad" "$num_nodes" "$gpus_per_node" "$task" "$experiment_script"
    
    begin_offset=$((i * DELAY_SECONDS))

    if [ "$USE_DEPENDENCIES" = true ]; then
        echo "Submitting with dependency (starts in ${begin_offset}s)..."
        job_id=$(submit_job_with_dependency "$experiment_script" "$previous_job_id" "$begin_offset")

        if [ -z "$previous_job_id" ]; then
            echo "✓ Job ID: $job_id (starts in ${begin_offset}s)"
        else
            echo "✓ Job ID: $job_id (starts in ${begin_offset}s, after $previous_job_id)"
        fi

        previous_job_id=$job_id
    else
        echo "Submitting job (starts in ${begin_offset}s)..."
        job_id=$(sbatch "--begin=now+${begin_offset}seconds" "$experiment_script" | grep -oP '\d+')
        echo "✓ Job ID: $job_id (starts in ${begin_offset}s)"
    fi

    submitted_jobs+=("$job_id:$name")
    echo ""
done

echo "=========================================="
echo "All Experiments Submitted!"
echo "=========================================="
echo "Total jobs: ${#submitted_jobs[@]}"
echo ""
echo "Job Summary:"
for job_info in "${submitted_jobs[@]}"; do
    IFS=':' read -r job_id job_name <<< "$job_info"
    echo "  Job ID: $job_id | Name: $job_name"
done
echo ""
echo "Useful Commands:"
echo "  Monitor all jobs: squeue -u \$USER"
echo "  Check job details: scontrol show job <JOB_ID>"
echo "  View job logs: tail -f /scratch/s5e/alv31415.s5e/logs/hyperscale-es-vllm/multinode_n16-<JOB_ID>.log"
echo ""

# Create cancel script
cancel_script="$EXPERIMENT_DIR/cancel_all_jobs_$(date +%Y%m%d_%H%M%S).sh"
echo "#!/bin/bash" > "$cancel_script"
echo "# Cancel all jobs from this experiment batch" >> "$cancel_script"
for job_info in "${submitted_jobs[@]}"; do
    job_id=$(echo "$job_info" | cut -d: -f1)
    echo "scancel $job_id" >> "$cancel_script"
done
chmod +x "$cancel_script"

echo "To cancel all jobs: $cancel_script"
echo "=========================================="