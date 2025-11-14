import re
import numpy as np
from datasets import load_dataset
from typing import List, Optional, Dict, Any


class ZerosTask:
    def __init__(self, batch_size, max_tokens):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompts = [
            "Output 3 numbers and then stop: ",
            "Hello, my name is",
            "Write some random numbers: ",
        ]
        assert batch_size <= len(self.prompts), f"{batch_size=} must be <= {len(self.prompts)=}"

    def get_batch(self):
        indices = np.arange(self.batch_size) % len(self.prompts)
        batch_prompts = [self.prompts[i] for i in indices]
        return batch_prompts, [None for _ in batch_prompts]
       
    def get_fitnesses(self, generations, answers):
        return [sum(c == "0" for c in g)/self.max_tokens for g in generations], [None for _ in generations]
    
    def get_fitness(self, generation, answer):
        return sum(c == "0" for c in generation)/self.max_tokens, None


class MathTask:
    def __init__(self, batch_size, dataset_name="openai/gsm8k", split="train", datset_size=None, answer_format="none"):
        self.dataset = load_dataset(dataset_name, "main", split=split)
        if datset_size is not None:
            self.dataset = self.dataset.select(range(datset_size))
        assert batch_size <= len(self.dataset), f"{batch_size=} must be <= {len(self.dataset)=}"
        self.batch_size = batch_size
        self.ans_format = answer_format
        self.idx = 0

    def _format_conversation(self, example):
        return {"prompt": f"User: {example['question']}\n\nAssistant: <think"}
    
    def _extract_gt_answer(self, text):
        return text.split('####')[-1].strip()
    
    def _extract_model_answer(self, text):
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore =[
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        if self.ans_format == "none":
            match = re.findall(regex_pattern, text)
            if match:
                match = match[-1] # take the last regex match
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                text = match.strip()

                for regex in regexes_to_ignore:
                    text = re.sub(regex, "", text)
                return text, "answer extracted"
            else:
                # print("NO REGEX MATCH FOUND")
                return None, "No regex match found"

        elif self.ans_format == "boxed":
            splits = text.split("boxed{")
            if len(splits) < 2:
                return None, "No `boxed{` found"
            else:
                text = splits[-1].strip() # take the last `boxed{`
                
                match = re.findall(regex_pattern, text)
                if match:
                    match = match[0] # take the first regex match
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    text = match.strip()

                    for regex in regexes_to_ignore:
                        text = re.sub(regex, "", text)
                    return text, "answer extracted"
                else:
                    return None, "No regex match found"
        
        else:
            raise ValueError(f"Unknown ans_format {self.ans_format}")

    def get_batch(self):
        """Returns a list of prompt and answer strings of length batch_size."""
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        examples = [self.dataset[i] for i in indices]
        self.idx += self.batch_size
        batch_prompts = [self._format_conversation(example)["prompt"] for example in examples]
        batch_answers = [self._extract_gt_answer(example["answer"]) for example in examples]    
        return batch_prompts, batch_answers

    def get_fitnesses(self, generations, gt_answers):
        assert len(generations) == len(gt_answers), f"{len(generations)=} must be equal to {len(gt_answers)=}"
        model_answers = [self._extract_model_answer(gen)[0] for gen in generations]
        is_corrects = [1.0 if (ma == ga) else 0.0 for ma, ga in zip(model_answers, gt_answers)]
        return is_corrects, model_answers
    
    def get_fitness(self, generation, gt_answer):
        model_answer = self._extract_model_answer(generation)[0]
        is_correct = 1.0 if (model_answer == gt_answer) else 0.0
        return is_correct, model_answer
    

class CountdownTask:
    def __init__(self, batch_size, datset_size=None, end_token: Optional[str] = None):
        data_path = "countdown.json"
        self.dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"{self.dataset=}")
        if datset_size is not None:
            self.dataset = self.dataset.select(range(datset_size))
        assert batch_size <= len(self.dataset), f"{batch_size=} must be <= {len(self.dataset)=}"
        self.batch_size = batch_size
        self.end_token = end_token
        self.idx = 0

    def get_batch(self):
        """Returns a list of prompt and answer strings of length batch_size."""
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        examples = [self.dataset[i] for i in indices]
        self.idx += self.batch_size
        batch_prompts = [example["context"] for example in examples]
        batch_answers = [(example["numbers"], example["target"]) for example in examples]
        return batch_prompts, batch_answers

    @staticmethod
    def _format_reward_function(response: str, end_token: Optional[str] = None) -> float:
        """
        Checks if the response follows the format <think>...</think><answer>...</answer>
        """
        # Strip end token if present
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_regex = r"<think>.*?<\/think>"
        answer_regex = r"<answer>.*?<\/answer>"
        full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

        think_match = re.search(think_regex, response, re.DOTALL)
        answer_match = re.search(answer_regex, response, re.DOTALL)
        full_format_match = re.match(full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0
        reward = 0.0
        if think_match:
            reward += 0.1
        if answer_match:
            reward += 0.5
        return reward

    @staticmethod
    def _answer_reward_function(response: str, numbers: List[int] = None, target: int = None) -> float:
        """
        Checks if the last <answer>...</answer> uses all numbers exactly once and evaluates to the target.
        Returns 1.0 if the last one is correct, else 0.0.
        """
        answer_regex = r"<answer>(.*?)<\/answer>"
        all_matches = re.findall(answer_regex, response, re.DOTALL)

        if not all_matches:
            return 0.0, None

        # Only check the last answer
        answer_content = all_matches[-1]
        
        allowed_chars = r"^[0-9+\-*/() ]+$"

        if not answer_content:
            return 0.0, answer_content
        if not re.match(allowed_chars, answer_content):
            return 0.0, answer_content

        # Check numbers used
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0, answer_content

        # Try evaluating
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0, answer_content
        except:
            return 0.0, answer_content

        return 0.0, answer_content
    
    def get_fitnesses(self, generations, answers):
        fitnesses = []
        model_answers = []
        for generation, answer in zip(generations, answers):
            reward, model_answer = self.get_fitness(generation, answer)
            fitnesses.append(reward)
            model_answers.append(model_answer)
        return fitnesses, model_answers
    
    def get_fitness(self, generation, answer):
        numbers, target = answer
        format_reward = self._format_reward_function("<think>" + generation, self.end_token)
        answer_reward, model_answer = self._answer_reward_function(generation, numbers, target)
        reward = format_reward * 0.1 + answer_reward
        return reward, model_answer
    