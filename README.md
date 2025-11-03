# Hyperscale ES with vLLM

### To install on isambard:

Follow instructions to install uv: https://docs.isambard.ac.uk/user-documentation/guides/python/#uv-installation-and-usage

Then below to install vllm:
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
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 50 --steps_per_adapter 4 --use_wandb
python vllm_random_lora_generation3.py --sigma 0.0001 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 50 --steps_per_adapter 4
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 500 --task gsm8k --model Qwen/Qwen2-1.5B-Instruct --use_wandb
```

### Scripts:

- `vllm_random_lora_generation1.py` - First working version. Just evolving the lora parameters with no merging, and incorrect noise.
- `vllm_random_lora_generation2.py` - Corrected version with merging lora adapters and eggroll noise. Not working.
- `vllm_random_lora_generation3.py` - Implemented to be line-by-line compared to version2. Eggroll noise but just evolving the lora parameters.
