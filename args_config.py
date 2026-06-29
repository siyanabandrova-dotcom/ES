from dataclasses import dataclass


@dataclass
class Args:
    """ES Fine-tuning for Countdown Task with multi-engine NCCL sync and LoRA population."""

    model_name: str = "Qwen/Qwen2-0.5B"
    # --- ES Hyperparameters ---
    sigma: float = 0.001
    population_size: int = 128
    num_iterations: int = 300
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    task: str = "zeros"  # Options: "zeros", "countdown", "math:deepscaler40k", ...
    prompt_batch_size: int = 2
    pass_at_k: bool = False
    normalize_with_std: bool = False
    scale_lr_in_grad: bool = False

    # --- LoRA Config ---
    lora_r: int = 4
    lora_alpha: int = None
    steps_per_adapter: int = 4
    learning_rate: float = 0.001

    # --- Runtime Config ---
    num_gpus: int = None
    num_engines: int = None
    tensor_parallel_size: int = 1  # Number of GPUs per engine for tensor parallelism
    verbose: bool = True
    base_seed: int = 0
    sub_dataset_size: int = None
    steps_per_eval: int = 10  # -1 to disable
    eval_batch_size: int = 128
    es_update_chunk_size: int = None  # Auto-select based on lora_r if None

    # --- WandB ---
    use_wandb: bool = False
    wandb_project: str = "hyperscalees-vllm"
    name_prefix: str = "debug"

    # --- Checkpointing ---
    save_freq: int = 50  # None: no saving, -1: saves at last step
    checkpoint_dir: str = None  # If None, will use EXPERIMENT_DIR/run_name/checkpoints
    resume_from: str = None  # Path to checkpoint to resume from

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r

        if self.tensor_parallel_size == 1:
            tp_config = {
                "Qwen/Qwen1.5-110B": 4,
                "Qwen/Qwen1.5-110B-Chat": 4,
                "Qwen/Qwen2.5-1.5B": 1,
                "Qwen/Qwen2.5-14B": 2,
                "Qwen/Qwen2.5-32B": 4,
                "Qwen/Qwen2.5-32B-Instruct": 4,
                "Qwen/Qwen2.5-72B": 4,
                "Qwen/Qwen2.5-72B-Instruct": 4,
                "Qwen/Qwen3-1.7B": 1,
                "Qwen/Qwen3-4B": 1,
                "Qwen/Qwen3-4B-Base": 1,
                "Qwen/Qwen3-8B": 1,
                "Qwen/Qwen3-30B": 2,
                "Qwen/Qwen3-30B-Base": 2,
                "Qwen/Qwen3-32B": 4,
            }

            for model_pattern, tp_size in tp_config.items():
                if model_pattern in self.model_name:
                    self.tensor_parallel_size = tp_size
                    print(
                        f"Auto-configured tensor_parallel_size={tp_size} for model {self.model_name}",
                        flush=True,
                    )
                    break
