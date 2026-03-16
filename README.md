# Hyperscale ES with vLLM

## To install on isambard:

Install miniforge: https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge (and ensure in conda `(base)` env).

Ensure paths set in bashrc:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Install uv: https://docs.isambard.ac.uk/user-documentation/guides/python/#uv-installation-and-usage

Then install vllm:
```shell
cd $SCRATCH && mkdir uv_envs && cd uv_envs && mkdir vllm_env
cd $SCRATCH/uv_envs/vllm_env
uv venv --seed -p=3.12
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate
# To support MoE + TP + Multi-LoRA, use vLLM 0.16.0 or later
srun --gpus 1 uv pip install vllm>=0.16.0 --torch-backend=auto
uv pip install tyro
uv pip install wandb
uv pip install weave
uv pip install peft
uv pip install datasets
uv pip install gem-llm
uv pip install pylatexenc
```

Download model to cache, eg:

```python
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate
srun --pty bash
python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-8B"
model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
```


## Code:

The main script is `es_lora_multinode.py`. This is an almost single file implementation, apart from the tasks which are in `tasks.py`. (`es_lora_multigpu.py` is an old single-node version of the code.)

`es_lora_multinode_moe.py` is the updated version that supports **Mixture-of-Experts (MoE)** models with Tensor Parallelism and Multi-LoRA (requires vLLM >= 0.16.0). It is backward compatible with `es_lora_multinode.py`.

`es_lora_multinode_2.py` is the old version of the code without checkpoint saving [09jan25]


## Tasks:

- `zeros`: Task is to output all zeros.
- `gsm8k`, `gsm8k-boxed`: GSM8K questions without and with requiring `\\boxed{}` formatting.
- `countdown`: Questions in countdown.json from https://github.com/VsonicV/es-fine-tuning-paper.
- `random`, `random-boxed`: Aim is to guess the random number without and with requiring `\\boxed{}` formatting. (For exploring behaviour with pass@k objective.)
- `math2:gsm8k`, `math2:asdiv2k`, `math2:math12k`, `math2:orz57k`, `math2:deepscaler40k`: Math questions. These tasks trigger automatic evaluation on `["math", "amc", "olympiad_bench", "minerva", "aime24"]`.


## Launching jobs:

`slurm_launch_multinode_n1.sh` and `slurm_launch_multinode_n4.sh` launch jobs on 1 and 4 nodes respectively. (`slurm_launch_multigpu.sh` launches the old single-node version.)

General for 4 nodes:

`sbatch slurm_launch_multinode_n4.sh <sigma> <learning_rate> <max_tokens> <model_name> <population_size> <steps_per_adapter> <lora_r> <task> <normalize_with_std> <prompt_batch_size> <sub_dataset_size> <name_prefix>`

Example with population size 1024 etc:

`sbatch slurm_launch_multinode_n4.sh 0.001 0.001 4096 "Qwen/Qwen3-1.7B" 1024 4 1 "math2:deepscaler40k" "normalize-with-std" 16 "null" "A1"`

Small debug example:

`sbatch slurm_launch_multinode_n4.sh 0.001 0.001 32 "Qwen/Qwen3-0.6B" 32 4 1 "zeros" "normalize-with-std" 16 "null" "debug"`


## Timings:

Some example timings for the following config:

```bash
### CONFIG ###
# 4 node job (16 gpus)
model_name: Qwen/Qwen3-1.7B
population_size: 1024
max_tokens: 4096
samples_per_prompt: 1
task: math2:deepscaler40k
prompt_batch_size: 16
lora_r: 1
lora_alpha: 1
steps_per_adapter: 4
```

VLLM speeds: input ~= 150-250 toks/s, output ~= 5k-7k toks/s

Iteration part times: total: 545.1316s,  LoRA gen: 1.3563s, vLLM+Score: 501.7295s, Aggregation: 0.0020s, ES update: 17.6253s, broadcast: 0.9900s

Just under 12 hrs for 100 iterations.

# Caution:

- Turn checkpoint saving back on,
- Turn wandb back on,
- Remove TP for 4B and 1.7B
- Increase job time