#!/usr/bin/env python3
"""Baseline Evaluation - Single Node - Run Qwen 110B on batches without ES-LoRA training"""

import argparse
from datetime import datetime
import json
import os
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import tyro
import wandb

from tasks import MathTask, CountdownTask, ZerosTask, MathTask2, RandomTask, DrawEggTask, DrawChickTask

print("IMPORTS: All imports completed successfully", flush=True)
print("=" * 80, flush=True)

# Default Hyperparameters
EXPERIMENT_DIR = os.path.expandvars("$SCRATCH/for_es_lora/experiments")
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

@dataclass
class Args:
    """Baseline Evaluation Arguments"""
    model_name: str = "Qwen/Qwen1.5-110B" 
    num_iterations: int = 300
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    task: str = "math2:deepscaler40k"
    prompt_batch_size: int = 32
    pass_at_k: bool = False

    # --- Runtime Config ---
    tensor_parallel_size: int = 4  # Default for 110B model on 4 GPUs
    verbose: bool = True
    base_seed: int = 0
    sub_dataset_size: int = None
    steps_per_eval: int = 10
    eval_batch_size: int = 32

    # --- WandB ---
    use_wandb: bool = True
    wandb_project: str = "hyperscalees-vllm"
    name_prefix: str = "baseline-eval"

    def __post_init__(self):
        # Auto-configure tensor_parallel_size based on model if needed
        if "110" in self.model_name or "72" in self.model_name:
            if self.tensor_parallel_size == 1:
                self.tensor_parallel_size = 4
                print(f"Auto-configured tensor_parallel_size=4 for model {self.model_name}", flush=True)


def make_task(args):
    """Initialize task based on args - matching es_lora_multinode API exactly"""
    if args.task.startswith("math2:"):
        dataset_name = args.task.split("math2:")[1]
        task = MathTask2(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            dataset_name=dataset_name,
            datset_size=args.sub_dataset_size,  # Note: typo 'datset_size' matches es_lora
            apply_chat_template=False,
        )
    elif args.task == "countdown":
        task = CountdownTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            datset_size=args.sub_dataset_size,
            end_token=None
        )
    elif args.task == "zeros":
        task = ZerosTask(
            batch_size=args.prompt_batch_size,
            max_tokens=args.max_tokens
        )
    elif args.task == "gsm8k":
        task = MathTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            dataset_name="openai/gsm8k",
            split="train",
            datset_size=args.sub_dataset_size,
            answer_format="none"
        )
    elif args.task == "gsm8k-boxed":
        task = MathTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            dataset_name="openai/gsm8k",
            split="train",
            datset_size=args.sub_dataset_size,
            answer_format="boxed"
        )
    elif args.task == "random":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_random_number=4,
            seed=args.base_seed,
            answer_format="none",
        )
    elif args.task == "random-boxed":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_random_number=4,
            seed=args.base_seed,
            answer_format="boxed",
        )
    elif args.task.startswith("drawegg"):
        distance_metric = args.task.split("-")[-1]
        assert distance_metric in ["jsd", "tvd", "chi2", "kl"], f"Unknown distance metric: {distance_metric}"
        task = DrawEggTask(
            batch_size=args.prompt_batch_size,
            answer_format="boxed" if "boxed" in args.task else "none",
            distance_metric=distance_metric,
            pass_at_k=args.pass_at_k,
        )
    elif args.task.startswith("drawchick"):
        distance_metric = args.task.split("-")[-1]
        assert distance_metric in ["jsd", "tvd", "chi2", "kl"], f"Unknown distance metric: {distance_metric}"
        task = DrawChickTask(
            batch_size=args.prompt_batch_size,
            distance_metric=distance_metric,
            pass_at_k=args.pass_at_k,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    return task


def generate_and_score(llm, prompts, sampling_params, task_obj, answers, args):
    """
    Generates responses and calculates fitness without LoRA.
    This is the baseline model performance.
    """
    print(f"Starting generation for {len(prompts)} prompts", flush=True)
    
    request_outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )
    
    # Calculate fitness
    fitness_list = []
    distinct_counts = []
    total_responses = 0
    num_truncated = 0
    mean_char_lengths = []
    mean_token_lengths = []
    responses_for_logging = []
    all_sample_stds = []
    all_pass_at_k_fitnesses = []
    all_mean_fitnesses = []
    all_task_info = {}

    num_prompts = len(answers)

    for i, output in enumerate(request_outputs):
        prompt_idx = i % num_prompts
        gt_answer = answers[prompt_idx]

        # Collect all responses for this prompt
        responses = [o.text for o in output.outputs]

        # Get fitness
        fit, model_answers, sample_fitnesses, task_info = task_obj.get_fitness(
            responses, gt_answer, pass_at_k=args.pass_at_k
        )

        # Collect task-specific info
        for k, v in task_info.items():
            if k not in all_task_info:
                all_task_info[k] = []
            all_task_info[k].append(v)

        # Collect stats
        sample_char_lens = []
        sample_token_lens = []
        model_answers_set = set()

        if isinstance(model_answers, (list, tuple)):
            for ma in model_answers:
                if ma is not None:
                    if isinstance(ma, list):
                        model_answers_set.add(tuple(ma))
                    else:
                        model_answers_set.add(ma)
        elif model_answers is not None:
            model_answers_set.add(model_answers)

        distinct_counts.append(len(model_answers_set))

        for resp_text in responses:
            sample_char_lens.append(len(resp_text))
            # Approximate token count (for logging)
            sample_token_lens.append(len(resp_text.split()))
            total_responses += 1
            # Check if truncated (using finish_reason from first output)
            if len(output.outputs) > 0 and hasattr(output.outputs[0], 'finish_reason'):
                if output.outputs[0].finish_reason == "length":
                    num_truncated += 1

        mean_char_lengths.append(np.mean(sample_char_lens) if sample_char_lens else 0.0)
        mean_token_lengths.append(np.mean(sample_token_lens) if sample_token_lens else 0.0)

        # Store fitness
        fitness_list.append(fit)

        # Track sample statistics
        if len(sample_fitnesses) > 1:
            all_sample_stds.append(np.std(sample_fitnesses))
            all_pass_at_k_fitnesses.append(np.max(sample_fitnesses))
            all_mean_fitnesses.append(np.mean(sample_fitnesses))
        else:
            all_sample_stds.append(0.0)
            all_pass_at_k_fitnesses.append(fit)
            all_mean_fitnesses.append(fit)

        # Log first few responses
        if i < 3:
            responses_for_logging.append(responses[0] if responses else "")

    # Aggregate task info
    info_dict = {}
    for k, v_list in all_task_info.items():
        info_dict[k] = float(np.mean(v_list))

    # Add additional stats
    info_dict["mean_distinct_counts"] = float(np.mean(distinct_counts))
    info_dict["prop_truncated"] = float(num_truncated / max(1, total_responses))
    info_dict["mean_char_length"] = float(np.mean(mean_char_lengths))
    info_dict["mean_token_length"] = float(np.mean(mean_token_lengths))
    info_dict["std_in_samples"] = float(np.mean(all_sample_stds))
    info_dict["pass_at_k_fitness"] = float(np.mean(all_pass_at_k_fitnesses))
    info_dict["mean_sample_fitness"] = float(np.mean(all_mean_fitnesses))

    print(f"Generation complete. Mean fitness: {np.mean(fitness_list):.4f}", flush=True)
    
    return (fitness_list, info_dict, responses_for_logging)


def main(args: Args):
    print("\n--- Initializing Baseline Evaluation (Single Node) ---")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"Batch size: {args.prompt_batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Number of iterations: {args.num_iterations}")

    # Initialize task
    task = make_task(args)
    print(f"Task initialized: {args.task}")

    # Initialize tokenizer for sampling params
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup evaluation task if needed (matching es_lora logic)
    do_eval = False
    if "math2:" in args.task and args.steps_per_eval > 0:
        do_eval = True
        print("--- Configuring Evaluation Tasks ---")
        
        eval_sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.base_seed + 12345,
            max_tokens=args.max_tokens,
            n=1,
            stop=[tokenizer.eos_token],
        )
        eval_task = MathTask2(
            batch_size=args.eval_batch_size,
            seed=args.base_seed + 12345,
            dataset_name="math-eval",
            datset_size=None,
            apply_chat_template=task.apply_chat_template,
        )
        print(f"Training on {args.task}, evaluating on {eval_task.split_names}.")
    else:
        eval_sampling_params = None
        eval_task = None
        if args.steps_per_eval > 0:
            print("Note: Evaluation only supported for math2: tasks")

    # Initialize WandB
    if args.use_wandb:
        run_name = f"{args.name_prefix}_{args.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        print(f"WandB initialized: {run_name}")

    # Setup sampling parameters (matching es_lora)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
        stop=[tokenizer.eos_token],
    )

    # Initialize vLLM model (single instance with tensor parallelism)
    print(f"\n--- Initializing vLLM with TP={args.tensor_parallel_size} ---")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_tokens,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )
    print(f"vLLM model initialized successfully")

    # Signal handlers
    def sig_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("\n--- Starting Baseline Evaluation Loop ---")
    
    fitnesses_so_far = []
    total_time_start = time.time()

    for step in range(args.num_iterations):
        print(f"\n\n======= Evaluation Step {step} / {args.num_iterations} =======")
        step_start = time.time()

        # --- EVALUATION ON EVAL SET ---
        eval_info_dict_all = {}
        if do_eval and args.steps_per_eval > 0 and step % args.steps_per_eval == 0:
            print(f"\n--- Running Evaluation Set at Step {step} ---")
            eval_start = time.time()
            prompts, answers = eval_task.get_eval_batch()
            
            # Run evaluation
            (eval_fitness_list, eval_info_dict, eval_sample_output) = generate_and_score(
                llm, prompts, eval_sampling_params, eval_task, answers, args
            )
            
            eval_info_dict_all = {f"eval/{k}": v for k, v in eval_info_dict.items()}
            
            # Compute eval metrics
            eval_fitnesses = np.array(eval_fitness_list)
            
            # Check if eval_task has split_names (for multi-task evaluation)
            if hasattr(eval_task, 'split_names'):
                eval_task_names = eval_task.split_names
                # Reshape to (num_tasks, batch_size_per_task)
                all_fitnesses_shaped = eval_fitnesses.reshape(len(eval_task_names), eval_task.batch_size)
                print(f"\n--------------------------------")
                for eval_task_name, fitness_array in zip(eval_task_names, all_fitnesses_shaped):
                    mean_fitness = float(np.mean(fitness_array))
                    eval_info_dict_all[f"eval/{eval_task_name}_mean_fitness"] = mean_fitness
                    print(f"EVAL {eval_task_name}: Mean fitness: {mean_fitness:.4f}")
                print(f"--------------------------------\n")
            else:
                # Single task evaluation
                eval_mean = float(np.mean(eval_fitnesses))
                eval_std = float(np.std(eval_fitnesses))
                print(f"\n--------------------------------")
                print(f"EVAL: Mean fitness: {eval_mean:.4f}, Std: {eval_std:.4f}")
                print(f"--------------------------------\n")
            
            eval_time = time.time() - eval_start
            if args.verbose: print(f"Evaluation complete in {eval_time:.4f}s")

        if args.use_wandb:
            wandb.log(eval_info_dict_all, step = step)


        """
        # --- TRAINING SET EVALUATION ---
        # KEY: This evaluates the base model on the same batch that ES-LoRA would use at this step
        print(f"\n--- Evaluating on Training Batch {step} ---")
        gen_start = time.time()
        prompts, answers = task.get_batch()
        
        # Generate and score
        (fitness_list, info_dict_all, sample_output) = generate_and_score(
            llm, prompts, sampling_params, task, answers, args
        )
        
        gen_time = time.time() - gen_start
        print(f"Generation complete in {gen_time:.4f}s")
        
        # Print first few responses for logging
        if args.verbose:
            print("\n----Sample Responses:")
            for text in sample_output[:2]:
                print(text)
            print("----\n")
        
        # Convert to numpy array - this is per-prompt fitness (shape: num_prompts)
        fitnesses = np.array(fitness_list)
        
        # Compute statistics (matching es_lora's computation)
        # In es_lora: mean_fitness = np.mean(fitnesses_shaped) where shape is (population_size, num_prompts)
        # For baseline: we just have (num_prompts,) so mean is across prompts
        mean_fitness = float(np.mean(fitnesses))
        min_fitness = float(np.min(fitnesses))
        max_fitness = float(np.max(fitnesses))
        std_fitness = float(np.std(fitnesses))
        
        print(f"\n=== Results for Step {step} ===")
        print(f"Mean fitness: {mean_fitness:.4f}")
        print(f"Min fitness: {min_fitness:.4f}")
        print(f"Max fitness: {max_fitness:.4f}")
        print(f"Std fitness: {std_fitness:.4f}")
        
        if args.verbose:
            for k, v in info_dict_all.items():
                print(f"  {k}: {v:.4f}")

        fitnesses_so_far.append(mean_fitness)
        print(f"\nFitnesses so far: {fitnesses_so_far}\n")

        # Log to WandB
        step_time = time.time() - step_start
        if args.use_wandb:
            log_dict = {
                "mean_fitness": mean_fitness,
                "min_fitness": min_fitness,
                "max_fitness": max_fitness,
                "std_fitness": std_fitness,
                "step": step,
                "time/generation": gen_time,
                "time/step": step_time,
                "total_time": time.time() - total_time_start,
                **info_dict_all,
                **eval_info_dict_all,
            }
            wandb.log(log_dict)

        print(f"======= Step {step} finished in {step_time:.4f}s =======\n")

    total_time = time.time() - total_time_start
    print(f"\n--- Baseline Evaluation Complete ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final fitnesses: {fitnesses_so_far}")
    """
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("BASELINE EVAL SCRIPT STARTED - Parsing arguments...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    args = tyro.cli(Args)

    print("=" * 80, flush=True)
    print("ARGUMENTS PARSED - Starting main function...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    main(args)