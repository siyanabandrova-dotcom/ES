#!/bin/bash

# =============================================================================
# Config-Based Experiment Launcher
# =============================================================================
# Reads experiment configurations from a CSV file and launches jobs

set -e

# --- Configuration ---
BASE_SCRIPT="slurm_launch_multinode_n32.sh"
CONFIG_FILE="experiments_config.csv"
DELAY_MINUTES=10
EXPERIMENT_DIR="/scratch/s5j/alv31415.s5j/experiments"
USE_DEPENDENCIES=true  # Set to false to use real-time waiting instead

# =============================================================================
# Functions
# =============================================================================

create_experiment_script() {
    local sigma=$1
    local lr=$2
    local model=$3
    local pop_size=$4
    local prompt_bs=$5
    local name=$6
    local output_script=$7
    
    cp "$BASE_SCRIPT" "$output_script"
    
    sed -i "s|^sigma=.*|sigma=\"$sigma\"|" "$output_script"
    sed -i "s|^learning_rate=.*|learning_rate=\"$lr\"|" "$output_script"
    sed -i "s|^model_name=.*|model_name=\"$model\"|" "$output_script"
    sed -i "s|^population_size=.*|population_size=\"$pop_size\"|" "$output_script"
    sed -i "s|^prompt_batch_size=.*|prompt_batch_size=\"$prompt_bs\"|" "$output_script"
    sed -i "s|^name_prefix=.*|name_prefix=\"$name\"|" "$output_script"
    
    chmod +x "$output_script"
}

submit_job_with_dependency() {
    local script=$1
    local dependency=$2
    
    if [ -z "$dependency" ]; then
        job_output=$(sbatch "$script")
    else
        job_output=$(sbatch --dependency=after:$dependency --begin=now+${DELAY_MINUTES}minutes "$script")
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
echo "Delay between jobs: $DELAY_MINUTES minutes"
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

while IFS=',' read -r sigma lr model pop_size prompt_bs name || [ -n "$sigma" ]; do
    # Skip comments and empty lines
    [[ "$sigma" =~ ^#.*$ ]] && continue
    [[ -z "$sigma" ]] && continue
    
    # Trim whitespace
    sigma=$(echo "$sigma" | xargs)
    lr=$(echo "$lr" | xargs)
    model=$(echo "$model" | xargs)
    pop_size=$(echo "$pop_size" | xargs)
    prompt_bs=$(echo "$prompt_bs" | xargs)
    name=$(echo "$name" | xargs)
    
    experiments+=("$sigma|$lr|$model|$pop_size|$prompt_bs|$name")
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
    
    IFS='|' read -r sigma lr model pop_size prompt_bs name <<< "${experiments[$i]}"
    
    echo "=========================================="
    echo "Experiment $experiment_num/${#experiments[@]}: $name"
    echo "=========================================="
    echo "Parameters:"
    echo "  sigma: $sigma"
    echo "  learning_rate: $lr"
    echo "  model_name: $model"
    echo "  population_size: $pop_size"
    echo "  prompt_batch_size: $prompt_bs"
    echo ""
    
    experiment_script="$EXPERIMENT_DIR/exp_${name}_$(date +%Y%m%d_%H%M%S).sh"
    
    # UPDATED: Pass prompt_bs to function
    create_experiment_script "$sigma" "$lr" "$model" "$pop_size" "$prompt_bs" "$name" "$experiment_script"
    
    if [ "$USE_DEPENDENCIES" = true ]; then
        echo "Submitting with dependency..."
        job_id=$(submit_job_with_dependency "$experiment_script" "$previous_job_id")
        
        if [ -z "$previous_job_id" ]; then
            echo "✓ Job ID: $job_id (starts immediately)"
        else
            echo "✓ Job ID: $job_id (starts ${DELAY_MINUTES}min after $previous_job_id)"
        fi
        
        previous_job_id=$job_id
    else
        echo "Submitting job..."
        job_id=$(submit_job "$experiment_script")
        echo "✓ Job ID: $job_id"
        
        if [ $experiment_num -lt ${#experiments[@]} ]; then
            echo "Waiting $DELAY_MINUTES minutes..."
            for ((min=DELAY_MINUTES; min>0; min--)); do
                echo "  Time remaining: $min minute(s)..."
                sleep 60
            done
        fi
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
echo "  View job logs: tail -f /scratch/s5j/alv31415.s5j/logs/hyperscale-es-vllm/multinode_n32-<JOB_ID>.log"
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