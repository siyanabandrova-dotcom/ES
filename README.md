# Hyperscale ES with vLLM

## Code:

The main script is `es_lora_multinode.py`. This is an almost single file implementation, apart from the tasks which are in `tasks.py`. (`es_lora_multigpu.py` is an old single-node version of the code.)

`es_lora_multinode_moe.py` is the updated version that supports **Mixture-of-Experts (MoE)** models with Tensor Parallelism and Multi-LoRA (requires vLLM >= 0.16.0). It is backward compatible with `es_lora_multinode.py`.


## Tasks:

- `zeros`: Task is to output all zeros.
- `gsm8k`, `gsm8k-boxed`: GSM8K questions without and with requiring `\\boxed{}` formatting.
- `countdown`: Questions in countdown.json from https://github.com/VsonicV/es-fine-tuning-paper.
- `random`, `random-boxed`: Aim is to guess the random number without and with requiring `\\boxed{}` formatting. (For exploring behaviour with pass@k objective.)
- `math2:gsm8k`, `math2:asdiv2k`, `math2:math12k`, `math2:orz57k`, `math2:deepscaler40k`: Math questions. These tasks trigger automatic evaluation on `["math", "amc", "olympiad_bench", "minerva", "aime24"]`.


## Installation:

Ensure paths set in bashrc:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Install uv: https://docs.isambard.ac.uk/user-documentation/guides/python/#uv-installation-and-usage

Then install vllm:
```bash
cd $SCRATCH && mkdir uv_envs && cd uv_envs && mkdir vllm_env
cd $SCRATCH/uv_envs/vllm_env
uv venv --seed -p=3.12
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate
# To support MoE + TP + Multi-LoRA, use vLLM 0.16.0 or later
srun --gpus 1 uv pip install vllm==0.17.0 --torch-backend=auto
uv pip install tyro
uv pip install wandb
uv pip install weave
uv pip install peft
uv pip install datasets
uv pip install gem-llm
uv pip install pylatexenc
```

Download model to cache, eg:

```bash
source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate
srun --pty bash
python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-8B"
model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
```


## Launching jobs:

Use `sbatch slurm_launch_multinode_n1.sh` and `sbatch slurm_launch_multinode_n16.sh` launch jobs on 1 and 16 nodes of 4 gpus respectively. Hyperparameters can be edited within the launch scripts.