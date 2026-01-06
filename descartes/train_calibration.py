import os
import re
import torch
import argparse
import shutil
import numpy as np
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- Mocks for 'gem.utils.math_grader' ---
# def extract_answer(text: str) -> Optional[str]:
#     if "\\boxed{" in text:
#         try:
#             parts = text.split("\\boxed{")
#             last_part = parts[-1]
#             count = 1
#             for i, char in enumerate(last_part):
#                 if char == "{": count += 1
#                 if char == "}": count -= 1
#                 if count == 0:
#                     return last_part[:i]
#         except:
#             return None
#     return None

# def grade(model_answer, gt_answer, fast=False):
#     if model_answer is None: return False
#     clean_model = str(model_answer).strip().replace(" ", "")
#     clean_gt = str(gt_answer).strip().replace(" ", "")
#     return clean_model == clean_gt

# --- Helper for Confidence Extraction ---
def extract_confidence(text: str) -> float:
    matches = re.findall(r"\\boxed\s*\{\s*([0-9]*\.?[0-9]+)\s*\}", text)
    if matches:
        try:
            val = float(matches[-1])
            return max(0.0, min(1.0, val))
        except ValueError:
            return 0.0
    return 0.0

# --- User's Task Code ---
def boxed_reward_fn(model_answer, gt_answer, fast=False,):
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    return is_correct

class MathTask2:
    def __init__(self, batch_size, tokenizer=None, dataset_name="gsm8k", datset_size=None, apply_chat_template=False):
        self.dataset_name = dataset_name
        dataset_names_dict = {
            "gsm8k": ("axon-rl/GSM-8k", "train", True),
            "asdiv2k": ("axon-rl/ASDIV-2k", "train", True),
            "math12k": ("axon-rl/MATH-12k", "train", True),
            "orz57k": ("axon-rl/ORZ-57k", "train", True),
            "deepscaler40k": ("axon-rl/DeepScaleR-40K", "train", True),
            "math-eval": ("axon-rl/math-eval", ["math", "amc", "olympiad_bench", "minerva", "aime24"], False),
        }
        
        # Fallback logic
        if dataset_name.lower() not in dataset_names_dict:
            print(f"Warning: {dataset_name} not found. Defaulting to GSM8K public.")
            dataset_name = "gsm8k"
            dataset_names_dict["gsm8k"] = ("openai/gsm8k", "main", True)

        dataset_name, splits, is_train = dataset_names_dict[dataset_name.lower()]
        self.is_train = is_train
        
        if is_train:
            try:
                self.dataset = load_dataset(dataset_name, split=splits)
            except:
                self.dataset = load_dataset(dataset_name, "main", split=splits)
                
            if datset_size is not None:
                self.dataset = self.dataset.select(range(datset_size))
        else:
            self.split_names = splits
            self.dataset = load_dataset(dataset_name)
            
        self.apply_chat_template = apply_chat_template
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if is_train:
            self.idx = 0

    @staticmethod
    def check_correct(generation: str, gt_answer: str) -> tuple[bool, str]:
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = gt_answer
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        model_answer = extract_answer(generation)
        is_correct = False
        if model_answer is not None:
            for correct_answer in correct_answers:
                is_correct = boxed_reward_fn(model_answer, correct_answer, fast=True)
                if is_correct:
                    break
        return is_correct, model_answer
    
    def _format_conversation(self, example):
        prob = example.get('problem', example.get('question', ''))
        instruction_str = "Please reason step by step, and put your final answer within \\boxed{}."
        problem = f"{prob}\n{instruction_str}"
        if self.apply_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {problem}\nAssistant: <think"
        
    def _format_examples(self, examples):
        batch_prompts = [self._format_conversation(example) for example in examples]
        batch_answers = [example["answer"] for example in examples]    
        return batch_prompts, batch_answers

    def get_batch(self):
        assert self.is_train, f"get_batch can only be called on a train dataset."
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples_list = [self.dataset[int(i)] for i in indices]
        return self._format_examples(examples_list)

    def get_batch_with_raw(self):
        """Returns (MathPrompts, RawQuestions, GTAnswers)."""
        assert self.is_train
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples_list = [self.dataset[int(i)] for i in indices]
        
        math_prompts, gt_answers = self._format_examples(examples_list)
        raw_questions = [ex.get('problem', ex.get('question', '')) for ex in examples_list]
        
        return math_prompts, raw_questions, gt_answers

    def batch_check_correct(self, generation, gt_answer):
        is_correct, model_answer = self.check_correct(generation, gt_answer)
        return 1.0 if is_correct else 0.0, model_answer

# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="RL Calibration Training")
    
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4, help="GRPO group size")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4)
    
    # Flags
    parser.add_argument("--grpo-normalization", action="store_true", default=True)
    parser.add_argument("--no-grpo-normalization", action="store_false", dest="grpo_normalization")
    parser.add_argument("--std-normalization", action="store_true", default=False)
    
    # WandB
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="rl-calibration")
    parser.add_argument("--name-prefix", type=str, default="debug")
    
    return parser.parse_args()

# --- Main Loop ---

def main():
    args = parse_args()
    
    print(f"--- Starting RL (Math + Calibration) ---")
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"{args.name_prefix}-{args.task}", config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading Learner (PyTorch)...")
    learner = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device).train()
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.learning_rate)

    print(">>> Loading Actor (vLLM)...")
    actor = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.vllm_gpu_util,
        dtype="bfloat16",
        enforce_eager=True
    )
    sampling_params = SamplingParams(n=args.samples_per_prompt, temperature=0.8, max_tokens=256)

    # 2. Setup Task
    task_handler = MathTask2(
        batch_size=args.prompt_batch_size, 
        tokenizer=tokenizer, 
        dataset_name=args.task
    )

    # 3. Training Loop
    global_step = 0
    while global_step < args.max_steps:
        print(f"\n{'='*15} Step {global_step} {'='*15}")
        
        # --- A. Data Preparation ---
        math_prompts, raw_questions, gt_answers = task_handler.get_batch_with_raw()
        
        calib_prompts = []
        for q in raw_questions:
            txt = (f"The question is {q}. "
                   "What is your current probability of getting the correct answer? "
                   "Please choose the closest from [0.0, 0.25, 0.5, 0.75, 1.0], "
                   "and output your answer in \\boxed{}.")
            calib_prompts.append(f"User: {txt}\nAssistant: \n")

        all_prompts = math_prompts + calib_prompts
        
        # --- B. Rollout ---
        print(f"[Rollout] Generating {len(all_prompts)} prompts x {args.samples_per_prompt} samples...")
        outputs = actor.generate(all_prompts, sampling_params)
        
        n_math = len(math_prompts)
        math_outputs = outputs[:n_math]
        calib_outputs = outputs[n_math:]
        
        train_prompts = []
        train_completions = []
        train_rewards = []
        train_types = [] # 0=Math, 1=Calib
        
        # --- C. Score Math ---
        prompt_accuracies = [] 

        for i, req in enumerate(math_outputs):
            gt = gt_answers[i]
            prompt_correct_count = 0.0
            sample_rewards = []
            sample_texts = []
            
            for sample in req.outputs:
                reward, _ = task_handler.batch_check_correct(sample.text, gt)
                prompt_correct_count += reward
                sample_rewards.append(reward)
                sample_texts.append(sample.text)

                if i == 0 and len(sample_texts) == 1:
                     print(f"  [Math] GT: {gt} | Rew: {reward}")

            acc = prompt_correct_count / args.samples_per_prompt
            prompt_accuracies.append(acc)
            
            train_prompts.extend([req.prompt] * args.samples_per_prompt)
            train_completions.extend(sample_texts)
            train_rewards.extend(sample_rewards)
            train_types.extend([0] * args.samples_per_prompt)

        # --- D. Score Calibration ---
        for i, req in enumerate(calib_outputs):
            target_acc = prompt_accuracies[i]
            sample_rewards = []
            sample_texts = []
            
            for sample in req.outputs:
                pred_prob = extract_confidence(sample.text)
                err = abs(pred_prob - target_acc)
                rew = 1.0 - err
                
                sample_rewards.append(rew)
                sample_texts.append(sample.text)

                if i == 0 and len(sample_texts) == 1:
                     print(f"  [Calib] Target: {target_acc} | Pred: {pred_prob} | Rew: {rew:.2f}")

            train_prompts.extend([req.prompt] * args.samples_per_prompt)
            train_completions.extend(sample_texts)
            train_rewards.extend(sample_rewards)
            train_types.extend([1] * args.samples_per_prompt)

        # --- E. Compute Advantages (GRPO) ---
        flat_rewards = torch.tensor(train_rewards, device=device, dtype=torch.float32)
        total_groups = len(train_prompts) // args.samples_per_prompt
        rewards_matrix = flat_rewards.view(total_groups, args.samples_per_prompt)
        
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        
        if args.grpo_normalization:
             advantages = (rewards_matrix - mean_r) / std_r
        else:
             advantages = (rewards_matrix - mean_r)

        flat_advantages = advantages.view(-1)
        
        m_avg = flat_rewards[torch.tensor(train_types)==0].mean().item()
        c_avg = flat_rewards[torch.tensor(train_types)==1].mean().item()
        print(f"[Stats] Math Avg: {m_avg:.3f} | Calib Avg: {c_avg:.3f}")

        # --- F. Update Step ---
        full_texts = [p + c for p, c in zip(train_prompts, train_completions)]
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        labels = inputs.input_ids.clone()
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in train_prompts]
        for idx, pl in enumerate(prompt_lens):
            labels[idx, :pl] = -100 

        optimizer.zero_grad()
        outputs = learner(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        mask = (shift_labels != -100).float()
        seq_losses = (token_losses * mask).sum(dim=1)
        
        loss = (seq_losses * flat_advantages).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
        optimizer.step()
        
        print(f"[Train] Loss: {loss.item():.4f}")
        if args.use_wandb:
            wandb.log({"loss": loss.item(), "math_rew": m_avg, "calib_rew": c_avg, "step": global_step})

        # --- G. Sync Weights ---
        if (global_step + 1) % 10 == 0:
            print("[Sync] Refreshing Actor Weights...")
            save_path = "temp_weights_sync"
            learner.save_pretrained(save_path)
            del actor
            torch.cuda.empty_cache()
            actor = LLM(
                model=save_path, 
                tensor_parallel_size=args.num_gpus, 
                gpu_memory_utilization=args.vllm_gpu_util, 
                dtype="bfloat16", 
                enforce_eager=True
            )

        global_step += 1

if __name__ == "__main__":
    main()