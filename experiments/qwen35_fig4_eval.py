#!/usr/bin/env python3
import os
import sys
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPAT_ROOT = "/home/siana/eggroll_compat"
sys.path.insert(0, COMPAT_ROOT)
sys.path.insert(0, REPO_ROOT)

from tasks import CountdownTask  # noqa: E402


def _ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def _strip_prompt(prompt: str, decoded: str) -> str:
    if decoded.startswith(prompt):
        return decoded[len(prompt):].lstrip()
    return decoded


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    outputs = []
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    for prompt, out_ids, input_len in zip(prompts, output_ids, input_lengths):
        decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
        completion = _strip_prompt(prompt, decoded)
        outputs.append(completion)
    return outputs


def main() -> None:
    model_name = "Qwen/Qwen3.5-2B"
    max_tokens = 1024
    eval_batch_size = 16
    base_seed = 0

    os.chdir(REPO_ROOT)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_tokenizer_padding(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    eval_task = CountdownTask(
        batch_size=eval_batch_size,
        seed=base_seed + 12345,
        datset_size=None,
        end_token=None,
    )

    prompts, answers = eval_task.get_batch()
    outputs = []
    for i in range(0, len(prompts), eval_batch_size):
        batch_prompts = prompts[i:i + eval_batch_size]
        outputs.extend(generate_batch(model, tokenizer, batch_prompts, max_tokens))

    fitnesses = []
    for completion, gt_answer in zip(outputs, answers):
        fitness, _model_answer = eval_task.get_fitness_single_sample(
            completion,
            gt_answer,
        )
        fitnesses.append(fitness)

    mean_fitness = float(np.mean(fitnesses))
    print("\n--------------------------------")
    print(f"EVAL countdown: Mean fitness: {mean_fitness:.4f}")
    print("--------------------------------\n")


if __name__ == "__main__":
    main()
