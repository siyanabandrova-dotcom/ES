# Hyperscale ES with vLLM

### To install on isambard:

Install uv: https://docs.isambard.ac.uk/user-documentation/guides/python/#uv-installation-and-usage

Then install vllm:
```shell
$ cd $SCRATCH/uv_envs
$ mkdir vllm_env
$ cd vllm_env
$ uv venv --seed -p=3.12
$ source .venv/bin/activate(vllm_env) $ srun --gpus 1 uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.10.2/vllm
```


### To run:
```
# Setup
srun --gpus=1 --time=03:00:00 --pty /bin/bash --login
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate && cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm && export WANDB_DIR=$SCRATCH/for_esvllm/wandb
python es_lora_nccl_async.py --sigma 0.01 --max-tokens 64 --learning-rate 0.001
python es_lora_nccl_async.py --no-use-wandb --sigma 0.001 --max-tokens 64 --population-size 4 --samples-per-prompt 3 --temperature 0.7


# Example experiments
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 50 --steps_per_adapter 4 --use_wandb
python vllm_random_lora_generation3.py --sigma 0.0001 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 50 --steps_per_adapter 4
python vllm_random_lora_generation2.py --sigma 0.01 --lora_r 4 --population_size 100 --learning_rate 0.001 --name_prefix B --max_tokens 500 --task gsm8k --model Qwen/Qwen2-1.5B-Instruct --use_wandb
python vllm_efficient_base1.py --no-use-wandb
python vllm_efficient_base1_sync.py --no-use-wandb
python es_lora_nccl.py --no-use-wandb
python es_lora_nccl_sync.py --no-use-wandb
python es_lora_nccl_async.py --no-use-wandb
python es_lora_nccl_async.py --sigma 0.01 --max-tokens 64 --learning-rate 0.01 --no-use-wandb
python es_lora_nccl_async.py --sigma 0.01 --learning-rate 0.01 --max-tokens 1024 --steps-per-adapter 10 --name-prefix time2 --task gsm8k --prompt-batch-size 32
python es_lora_nccl_async.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --name-prefix time2 --task gsm8k --prompt-batch-size 4 --no-use-wandb
python es_lora_nccl_async.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --name-prefix time2 --task zeros --prompt-batch-size 2 --population-size 10 --no-use-wandb
python es_lora_nccl_async.py --sigma 0.0 --learning-rate 0.0 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 2 --population-size 10 --no-use-wandb --name-prefix A
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 16 --sub-dataset-size 16 --population-size 100 --no-use-wandb --name-prefix A
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 64 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 4 --sub-dataset-size 4 --population-size 10 --no-use-wandb --name-prefix A
python es_lora_nccl_async.py --sigma 0.0 --learning-rate 0.0 --max-tokens 10000 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 32 --sub-dataset-size 1000 --no-use-wandb --name-prefix A --model-name "Qwen/Qwen3-1.7B" --temperature 0.7 --samples-per-prompt 4 --population_size 2
python es_lora_nccl_async.py --sigma 0.0 --learning-rate 0.0 --max-tokens 10000 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 32 --sub-dataset-size 1000 --no-use-wandb --name-prefix A --model-name "Qwen/Qwen3-4B" --temperature 0.7 --samples-per-prompt 4 --population_size 2
python es_lora_nccl_async.py --sigma 0.0 --learning-rate 0.0 --max-tokens 10000 --steps-per-adapter 4 --task zeros --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix A --model-name "Qwen/Qwen3-4B" --temperature 0.7 --samples-per-prompt 4 --population_size 2
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task countdown --prompt-batch-size 16 --sub-dataset-size 16 --no-use-wandb --name-prefix A --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 4 --population_size 2
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 64 --steps-per-adapter 4 --task zeros --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k-boxed --prompt-batch-size 16 --sub-dataset-size 1000 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-1.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 10

python es_lora_nccl_async.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --task zeros --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async3.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --task zeros --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async2.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --task gem:game:GuessTheNumber-v0-easy --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async3.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --task gem:game:GuessTheNumber-v0-easy --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async3.py --sigma 0.01 --learning-rate 0.01 --max-tokens 64 --steps-per-adapter 4 --task gem:math:DeepScaleR40K --prompt-batch-size 3 --sub-dataset-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen2.5-0.5B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k --prompt-batch-size 128 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async3.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gem:math:DeepScaleR40K --prompt-batch-size 128 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 64 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 3 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-0.6B" --temperature 0.0 --samples-per-prompt 1 --population_size 100
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 128 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 100 --steps_per_eval -1
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 4096 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 128 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 1000

python es_lora_nccl_async2.py --sigma 0.001 --learning-rate 0.001 --max-tokens 128 --steps-per-adapter 4 --task random-boxed --prompt-batch-size 32 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-0.6B" --temperature 0.7 --population_size 100 --steps_per_eval -1 --pass_at_k --samples-per-prompt 8

python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k-boxed --prompt-batch-size 16 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 128 --normalize-with-std
python es_lora_nccl_async2.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task gsm8k-boxed --prompt-batch-size 16 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 128 --normalize-with-std
python es_lora_nccl_async2.py --sigma 0.001 --learning-rate 0.001 --max-tokens 4096 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 16 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B-Base" --temperature 0.0 --samples-per-prompt 1 --population_size 128 --normalize-with-std
python es_lora_nccl_async2.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 16 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 128 --normalize-with-std
python es_lora_nccl_async.py --sigma 0.001 --learning-rate 0.001 --max-tokens 1024 --steps-per-adapter 4 --task math2:deepscaler40k --prompt-batch-size 16 --no-use-wandb --name-prefix debug --model-name "Qwen/Qwen3-4B" --temperature 0.0 --samples-per-prompt 1 --population_size 128 --normalize-with-std
```







