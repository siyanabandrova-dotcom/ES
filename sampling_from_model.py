#!/usr/bin/env python3
"""Simplified vLLM Inference and Scoring (No Training/LoRA)"""

import os
import sys
import time
import json  # Added import
import numpy as np
from dataclasses import dataclass
import tyro
from vllm import LLM, SamplingParams

# Assuming tasks.py exists in the same directory as per original script
from tasks import MathTask, CountdownTask, ZerosTask, MathTask2, RandomTask, DrawEggTask, DrawChickTask

@dataclass
class Args:
    """Simplified Inference Configuration"""
    model_name: str = "Qwen/Qwen2-0.5B" 
    
    # --- Task Config ---
    task: str = "zeros"  # Options: "zeros", "gsm8k", "countdown", "drawegg-jsd", etc.
    batch_size: int = 2
    sub_dataset_size: int = None
    pass_at_k: bool = False # Note: Requires n > 1 if True

    # --- Generation Config ---
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    base_seed: int = 0

    # --- Runtime Config ---
    tensor_parallel_size: int = 1  # Number of GPUs to use
    gpu_memory_utilization: float = 0.90

    # --- Output Config ---
    output_dir: str = "sampling_outputs" # Directory to save JSONs

def get_task(args: Args):
    """Factory function to initialize the correct task object."""
    if args.task == "zeros":
        return ZerosTask(batch_size=args.batch_size, max_tokens=args.max_tokens)
    elif args.task == "gsm8k":
        return MathTask(batch_size=args.batch_size, seed=args.base_seed, dataset_name="openai/gsm8k", split="train", datset_size=args.sub_dataset_size, answer_format="none")
    elif args.task == "gsm8k-boxed":
        return MathTask(batch_size=args.batch_size, seed=args.base_seed, dataset_name="openai/gsm8k", split="train", datset_size=args.sub_dataset_size, answer_format="boxed")
    elif args.task == "countdown":
        return CountdownTask(batch_size=args.batch_size, seed=args.base_seed, datset_size=args.sub_dataset_size, end_token=None)
    elif args.task.startswith("math2:"):
        dataset_name = args.task.split("math2:")[1]
        return MathTask2(batch_size=args.batch_size, seed=args.base_seed, dataset_name=dataset_name, datset_size=args.sub_dataset_size, apply_chat_template=False)
    elif args.task == "random":
        return RandomTask(batch_size=args.batch_size, max_random_number=4, seed=args.base_seed, answer_format="none")
    elif args.task == "random-boxed":
        return RandomTask(batch_size=args.batch_size, max_random_number=4, seed=args.base_seed, answer_format="boxed")
    elif args.task.startswith("drawegg"):
        # Simplified parsing for drawegg variants
        boxed = "boxed" in args.task
        metric = "tvd" if "tvd" in args.task else "jsd"
        answer_fmt = "boxed" if boxed else "none"
        # Force batch_size to 1 for Draw tasks as per class assertion
        return DrawEggTask(batch_size=1, answer_format=answer_fmt, distance_metric=metric, pass_at_k=args.pass_at_k)
    elif args.task.startswith("drawchick"):
        metric = "tvd" if "tvd" in args.task else "jsd"
        # Force batch_size to 1 for Draw tasks as per class assertion
        return DrawChickTask(batch_size=1, distance_metric=metric, pass_at_k=args.pass_at_k)
    else:
        raise ValueError(f"Unknown task: {args.task}")

def main(args: Args):
    print("=" * 80)
    print(f"Loading Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"GPUs (TP): {args.tensor_parallel_size}")
    print("=" * 80)

    # 1. Initialize Task
    task = get_task(args)
    
    # 2. Get Data Batch
    print("Fetching prompts from task...")
    prompts, answers = task.get_batch()
    print(f"Loaded {len(prompts)} prompts.")

    # 3. Initialize vLLM
    # Note: We rely on vLLM's internal handling of TP, no Ray actors needed here.
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(1024, 2 * args.max_tokens),
        seed=args.base_seed
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
    )

    # 4. Generate
    print("Generating responses...")
    start_time = time.time()
    request_outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start_time
    print(f"Generation complete in {duration:.2f}s")

    # 5. Score and Print
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    total_fitness = 0.0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for i, output in enumerate(request_outputs):
        prompt_text = prompts[i]
        gt_answer = answers[i]
        
        # Extract generated texts for this prompt
        generated_texts = [o.text for o in output.outputs]
        
        # Calculate fitness
        fit, model_answers, sample_fitnesses, task_info = task.get_fitness(
            generated_texts, 
            gt_answer, 
            pass_at_k=args.pass_at_k
        )
        
        total_fitness += fit

        # --- SPECIAL HANDLING FOR DRAW TASKS ---
        # Check if the task is an instance of DrawEggTask or DrawChickTask
        if isinstance(task, (DrawEggTask, DrawChickTask)):
            print(f"Detected Draw Task. Calculating and saving counts...")
            
            # Recalculate counts using the method exposed in tasks.py
            # (get_fitness does this internally but we want the raw data to save)
            counts, _ = task.get_counts(generated_texts)
            
            # Construct filename
            safe_model_name = args.model_name.replace("/", "_")
            json_filename = f"{args.task}_seed{args.base_seed}_{safe_model_name}.json"
            json_path = os.path.join(args.output_dir, json_filename)
            
            # Prepare data
            save_data = {
                "task": args.task,
                "model": args.model_name,
                "seed": args.base_seed,
                "temperature": args.temperature,
                "samples_per_prompt": args.samples_per_prompt,
                "metrics": {k: float(v) for k, v in task_info.items()}, # Ensure float for JSON serialization
                "target_counts": task.target_counts.tolist(),
                "generated_counts": counts.tolist()
            }
            
            try:
                with open(json_path, "w") as f:
                    json.dump(save_data, f, indent=2)
                print(f"Saved counts to: {json_path}")
            except Exception as e:
                print(f"Error saving JSON: {e}")
        # ---------------------------------------

        # Print details for the first few items
        print(f"\n--- Prompt {i} ---")
        print(f"Prompt: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"Prompt: {prompt_text}")
        
        for j, text in enumerate(generated_texts):
            s_fit = sample_fitnesses[j] if j < len(sample_fitnesses) else "N/A"
            # Truncate output for display
            display_text = text[:100] + "..." if len(text) > 100 else text
            display_text = display_text.replace("\n", " ") 
            print(f"  > Sample {j} (Fit: {s_fit:.4f}): {display_text}")
        
        print(f"Aggregate Fitness for Prompt {i}: {fit:.4f}")

    avg_fitness = total_fitness / len(prompts)
    print("\n" + "=" * 80)
    print(f"OVERALL AVERAGE FITNESS: {avg_fitness:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)