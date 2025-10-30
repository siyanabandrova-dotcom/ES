# Hyperscale ES with vLLM

### To install on isambard:

Install uv: https://docs.isambard.ac.uk/user-documentation/guides/python/#uv-installation-and-usage

Then install vllm:
```shell
$ mkdir vllm_env
$ cd vllm_env
$ uv venv --seed -p=3.12
$ source .venv/bin/activate(vllm_env) $ srun --gpus 1 uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.10.2/vllm
```


### To run:
```
# Setup
srun --gpus=1 --time=03:00:00 --pty /bin/bash --login
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate
cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm

# Example experiments
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --use_wandb --max_tokens 50
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --use_wandb --max_tokens 500 --task gsm8k --model Qwen/Qwen2-1.5B-Instruct
```

### Current status:

- Toy task of outputting zeros trains well, e.g: `python vllm_random_lora_generation.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --use_wandb --max_tokens 50`

- Also works with noise reuse and is much faster, e.g: `python vllm_random_lora_generation.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --use_wandb --max_tokens 50 --steps_per_adapter 4`

- GSM8K implemented with two templates (with and without `\boxed`)

- **TODO:** Model "Qwen/Qwen2-0.5B" fits but "Qwen/Qwen2-1.5B" fails with memory error, e.g. after 3 es steps... <--Fix this!

- **TODO:** The speed seems identical with different numbers of GPUs <-- Fix this!



