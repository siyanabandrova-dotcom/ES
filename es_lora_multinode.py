#!/usr/bin/env python3
"""ES-LoRA Training with NCCL and async evaluation"""

import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass
import copy
import math

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
import tyro
import wandb
import weave
from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file

from tasks import MathTask, CountdownTask, ZerosTask, MathTask2, RandomTask

print("IMPORTS: All imports completed successfully", flush=True)
print("=" * 80, flush=True)

# Default Hyperparameters
EXPERIMENT_DIR = "/dev/shm/outputs_es_lora"
LORA_POPULATION_PATH = "/dev/shm/es_lora_population_async"

@dataclass
class Args:
    """ES Fine-tuning for Countdown Task with multi-engine NCCL sync and LoRA population"""
    model_name: str = "Qwen/Qwen2-0.5B" 
    # --- ES Hyperparameters ---
    sigma: float = 0.001
    population_size: int = 128
    num_iterations: int = 1000
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    task: str = "zeros"  # Options: "zeros", "gsm8k", "gsm8k-boxed", ...
    prompt_batch_size: int = 2
    pass_at_k: bool = False
    normalize_with_std: bool = False

    # --- LoRA Config ---
    lora_r: int = 4
    lora_alpha: int = None
    steps_per_adapter: int = 4
    learning_rate: float = 0.001

    # --- Runtime Config ---
    experiment_dir: str = EXPERIMENT_DIR
    num_gpus: int = None
    num_engines: int = None
    verbose: bool = True
    base_seed: int = 0
    sub_dataset_size: int = None
    steps_per_eval: int = 10 # -1 to disable
    eval_batch_size: int = 16 

    # --- WandB ---
    use_wandb: bool = False
    wandb_project: str = "hyperscalees-vllm-multinode"
    name_prefix: str = f"debug"

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def map_peft_updates_to_vllm(peft_updates_dict, vllm_shapes_dict, device: torch.device):
    vllm_updates_dict = {
        name: torch.zeros(shape, device=device) for name, shape in vllm_shapes_dict.items()
        if name.endswith(".base_layer.weight")
    }
    for peft_name, weight_update in peft_updates_dict.items():
        vllm_name = peft_name.replace("base_model.model.", "")
        if "self_attn.q_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.q_proj", "self_attn.qkv_proj")
            start = 0
            end = weight_update.shape[0]
            vllm_updates_dict[vllm_name][start:end] += weight_update
        elif "self_attn.k_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.k_proj", "self_attn.qkv_proj")
            peft_q_name = peft_name.replace("k_proj", "q_proj")
            start = peft_updates_dict[peft_q_name].shape[0]
            end = start + weight_update.shape[0]
            vllm_updates_dict[vllm_name][start:end] += weight_update
        elif "self_attn.v_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.v_proj", "self_attn.qkv_proj")
            peft_q_name = peft_name.replace("v_proj", "q_proj")
            peft_k_name = peft_name.replace("v_proj", "k_proj")
            start = peft_updates_dict[peft_q_name].shape[0] + peft_updates_dict[peft_k_name].shape[0]
            end = start + weight_update.shape[0]
            vllm_updates_dict[vllm_name][start:end] += weight_update
        elif "self_attn.o_proj" in vllm_name:
            vllm_updates_dict[vllm_name] += weight_update
        elif "mlp.gate_proj" in vllm_name:
            vllm_name = vllm_name.replace("mlp.gate_proj", "mlp.gate_up_proj")
            start = 0
            end = weight_update.shape[0]
            vllm_updates_dict[vllm_name][start:end] += weight_update
        elif "mlp.up_proj" in vllm_name:
            vllm_name = vllm_name.replace("mlp.up_proj", "mlp.gate_up_proj")
            peft_gate_name = peft_name.replace("up_proj", "gate_proj")
            start = peft_updates_dict[peft_gate_name].shape[0]
            end = start + weight_update.shape[0]
            vllm_updates_dict[vllm_name][start:end] += weight_update
        elif "mlp.down_proj" in vllm_name:
            vllm_updates_dict[vllm_name] += weight_update
        else:
            raise ValueError(f"Unexpected PEFT layer name: {peft_name}")
    return vllm_updates_dict


def _stateless_init_process_group(master_address, master_port, gpu_rank, world_size, device):
    """Initializes PyNcclCommunicator using StatelessProcessGroup."""
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
    except ImportError:
        print("Warning: vLLM distributed modules not found. NCCL features will not work.")
        return None
        
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=gpu_rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

def get_rng_noise(
        base_seed: int,
        num_pop_pairs: int,
        pop_pair_idx: int,
        num_layers: int,
        layer_idx: int,
        step: int,
        shapes: list,
        ) -> dict[torch.device, torch.Generator]:
    """
    Create a dictionary of RNGs, one for each device.
    All RNGs are seeded with the same ID to ensure deterministic noise
    across different devices.
    """
    id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    torch_rng = torch.Generator().manual_seed(id)

    noise_a, noise_b = (torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=shape,
                    generator=torch_rng,
                ) for shape in shapes)
    return noise_a, noise_b

class WorkerExtension:
    """
    Custom extension for vLLM workers to handle ES update and NCCL broadcast.
    This class is passed to the vLLM engine via 'worker_extension_cls'.
    """

    def get_transport_info(self):
        """Returns the IP and a free port from the worker's perspective."""
        return get_ip(), get_open_port()

    @torch.no_grad()
    def apply_lora_es_update(self, normalized_fitnesses: list[tuple[int, float]], peft_shapes_dict, es_step: int, args: Args):
        """
        Computes and applies the ES update delta to the base model weights.
        This must only run on the master engine (gpu_rank 0).
        """
        if self.gpu_rank != 0: 
            return False
        
        peft_updates_dict = {name: torch.zeros(x, device=self.device) for name, x in peft_shapes_dict.items()}
        vllm_shapes_dict = {name: x.shape for name, x in self.model_runner.model.named_parameters()}
        
        pop_step = es_step // args.steps_per_adapter
        for pop_pair_idx in range(args.population_size // 2):
            pop_idx_1 = pop_pair_idx * 2
            pop_idx_2 = pop_pair_idx * 2 + 1

            fitness1 = normalized_fitnesses[pop_idx_1]
            fitness2 = normalized_fitnesses[pop_idx_2]

            for layer_idx, (peft_name, weight_shape) in enumerate(peft_shapes_dict.items()):
                lora_b_shape, lora_a_shape = (weight_shape[0], args.lora_r), (args.lora_r, weight_shape[1])
                noise_a, noise_b = get_rng_noise(
                    base_seed=args.base_seed,
                    num_pop_pairs=args.population_size//2,
                    pop_pair_idx=pop_idx_1//2,
                    num_layers=len(peft_shapes_dict.keys()),
                    layer_idx=layer_idx,
                    step=pop_step,
                    shapes=[lora_a_shape, lora_b_shape],
                )
                noise_a = noise_a.to(self.device)
                noise_b = noise_b.to(self.device)
                noise_b *= math.sqrt(args.sigma)
                noise_a *= math.sqrt(args.sigma)
                noise = torch.matmul(noise_b, noise_a)
                assert noise.shape == weight_shape, f"{peft_name}: {noise.shape=} vs {weight_shape=}"
                peft_updates_dict[peft_name] += (noise * (fitness1 - fitness2))

        vllm_updates_dict = map_peft_updates_to_vllm(peft_updates_dict, vllm_shapes_dict, self.device)

        # Debug: Check if updates are non-zero
        max_peft_update = max([v.abs().max().item() for v in peft_updates_dict.values()])
        max_vllm_update = max([v.abs().max().item() for v in vllm_updates_dict.values()])
        print(f"ES UPDATE DEBUG: max_peft_update={max_peft_update:.6e}, max_vllm_update={max_vllm_update:.6e}", flush=True)

        # Check fitness differences
        fitness_diffs = [abs(normalized_fitnesses[i] - normalized_fitnesses[i+1]) for i in range(0, len(normalized_fitnesses)-1, 2)]
        max_fitness_diff = max(fitness_diffs) if fitness_diffs else 0
        print(f"ES UPDATE DEBUG: max_fitness_diff={max_fitness_diff:.6e}, population_size={args.population_size}, sigma={args.sigma}", flush=True)

        # Store a sample weight before update for debugging
        sample_param_name = None
        sample_param_before = None

        for i, (name, param) in enumerate(self.model_runner.model.named_parameters()):
            if name in vllm_updates_dict:
                if sample_param_name is None:
                    sample_param_name = name
                    sample_param_before = param.data.clone().cpu()

                update = vllm_updates_dict[name]
                gradient = (1.0 / (args.population_size * args.sigma + 1e-8)) * update * args.learning_rate
                if sample_param_name == name:
                    print(f"ES UPDATE DEBUG: gradient.abs().max()={gradient.abs().max().item():.6e}, lr={args.learning_rate}", flush=True)
                param.data.add_(gradient)  # Use .data.add_() to ensure in-place update

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Check if weights actually changed
        if sample_param_name is not None:
            sample_param_after = None
            for name, param in self.model_runner.model.named_parameters():
                if name == sample_param_name:
                    sample_param_after = param.data.clone().cpu()
                    break

            if sample_param_after is not None:
                weight_diff = (sample_param_after - sample_param_before).abs().max().item()
                print(f"ES UPDATE: Max weight change in {sample_param_name}: {weight_diff:.6e}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()
        return True
    
    def init_inter_engine_group(self, master_address: str, master_port: int, gpu_rank: int, world_size: int):
        self.device = self.model_runner.device
        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, gpu_rank, world_size, self.device
        )
        return True

    @torch.no_grad()
    def broadcast_all_weights(self, src_rank: int):
        # NOTE: ALL ranks must participate in NCCL broadcast,
        # including the source rank. The source sends, others receive.

        print(f"WORKER {self.gpu_rank}: broadcast_all_weights called, src_rank={src_rank}", flush=True)

        if not self.inter_pg:
            # NCCL not available - this will require weights to be sent via Ray
            # Return False to signal caller to use Ray-based broadcast instead
            print(f"WORKER {self.gpu_rank}: No NCCL inter_pg available, returning False", flush=True)
            return False

        try:
            is_source = (self.gpu_rank == int(src_rank))
            role = "sender" if is_source else "receiver"
            print(f"WORKER {self.gpu_rank}: Starting NCCL broadcast as {role} (src={src_rank})...", flush=True)

            param_count = 0
            for name, param in self.model_runner.model.named_parameters():
                # ALL ranks must call broadcast - source sends, others receive
                self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
                param_count += 1

            print(f"WORKER {self.gpu_rank}: Broadcast {param_count} parameters, synchronizing...", flush=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"WORKER {self.gpu_rank}: Broadcast complete ({role})", flush=True)
            return True
        except Exception as e:
            print(f"WORKER {self.gpu_rank}: NCCL broadcast failed: {e}", flush=True)
            return False

    @torch.no_grad()
    def get_model_state_dict(self):
        """Get the current model state dict (for Ray-based broadcast)"""
        return {name: param.cpu().clone() for name, param in self.model_runner.model.named_parameters()}

    @torch.no_grad()
    def set_model_state_dict(self, state_dict):
        """Set the model state dict (for Ray-based broadcast)"""
        model_params = dict(self.model_runner.model.named_parameters())
        for name, param in state_dict.items():
            if name in model_params:
                model_params[name].data.copy_(param.to(model_params[name].device))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

class ESNcclLLM(LLM):
    """vLLM subclass using the custom WorkerExtension."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)
        
        # Placeholders for LoRA generation data
        self.lora_init_state_dict = None
        self.lora_init_shapes = None
        self.lora_config_data = None

    def setup_local_lora_generation(self, peft_state_dict, peft_shapes_dict, lora_config_dict, rank: int):
        """Receives the initial LoRA state to be able to reconstruct adapters locally."""
        self.lora_init_state_dict = peft_state_dict
        self.lora_init_shapes = peft_shapes_dict
        self.lora_config_data = lora_config_dict
        self.rank = rank
        
        self.lora_storage_path = f"{LORA_POPULATION_PATH}_{self.rank}"
        
        if os.path.exists(self.lora_storage_path):
            shutil.rmtree(self.lora_storage_path)
        os.makedirs(self.lora_storage_path, exist_ok=True)
        return True
    
    

    def generate_local_adapters(self, population_indices: list[int], es_step: int, args: Args):
        """
        Generates LoRA adapter files in the LOCAL /dev/shm of this worker node.
        Returns the absolute paths to these files.
        """
        adapter_paths = []
        pop_step = es_step // args.steps_per_adapter
        
        # Ensure config is JSON serializable
        config_to_save = copy.deepcopy(self.lora_config_data)
        if "target_modules" in config_to_save and isinstance(config_to_save["target_modules"], (set, tuple)):
            config_to_save["target_modules"] = list(config_to_save["target_modules"])

        for pop_idx in population_indices:
            adapter_path = os.path.join(self.lora_storage_path, f"pop_{pop_idx}")
            os.makedirs(adapter_path, exist_ok=True)
            adapter_paths.append(adapter_path)
            
            # Save config
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(config_to_save, f)

            # Generate weights (sanitized)
            local_state_dict = {}
            for layer_idx, (peft_name, weight_shape) in enumerate(self.lora_init_shapes.items()):
                # Generate LoRA A and B names from the base_layer.weight name
                lora_a_name_raw = peft_name.replace("base_layer.weight", "lora_A.default.weight")
                lora_b_name_raw = peft_name.replace("base_layer.weight", "lora_B.default.weight")

                # 2. PEFT uses ".lora_A.default.weight" but vLLM expects ".lora_A.weight"
                lora_a_name = lora_a_name_raw.replace(".lora_A.default.weight", ".lora_A.weight")
                lora_b_name = lora_b_name_raw.replace(".lora_B.default.weight", ".lora_B.weight")

                # Get base (initial) weights and clone to CPU
                lora_a = self.lora_init_state_dict[lora_a_name_raw].clone().cpu()
                lora_b = self.lora_init_state_dict[lora_b_name_raw].clone().cpu()

                lora_b_shape, lora_a_shape = (weight_shape[0], args.lora_r), (args.lora_r, weight_shape[1])

                noise_a, noise_b = get_rng_noise(
                    base_seed=args.base_seed,
                    num_pop_pairs=args.population_size//2,
                    pop_pair_idx=pop_idx//2,
                    num_layers=len(self.lora_init_shapes.keys()),
                    layer_idx=layer_idx,
                    step=pop_step,
                    shapes=[lora_a_shape, lora_b_shape],
                )

                noise_b *= math.sqrt(args.sigma)
                noise_a *= math.sqrt(args.sigma)

                # Zero out the weights (before then setting them to noise)
                lora_a.zero_()
                lora_b.zero_()

                # Antithetic sampling
                lora_a.add_(noise_a)
                if pop_idx % 2 == 1:
                    lora_b.add_(-noise_b)
                else:
                    lora_b.add_(noise_b)

                # Debug: Check if LoRA weights are non-zero (only for first layer of first few adapters)
                if layer_idx == 0 and pop_idx < 4:
                    max_a = lora_a.abs().max().item()
                    max_b = lora_b.abs().max().item()
                    print(f"LORA GEN DEBUG: pop_idx={pop_idx}, layer={layer_idx}, max_a={max_a:.6e}, max_b={max_b:.6e}, sigma={math.sqrt(args.sigma):.6e}", flush=True)

                local_state_dict[lora_a_name] = lora_a
                local_state_dict[lora_b_name] = lora_b

            # Save tensors
            save_file(local_state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))

        # Debug: Verify first adapter exists and has non-zero weights
        if len(adapter_paths) > 0:
            from safetensors import safe_open
            first_adapter = adapter_paths[0]
            with safe_open(os.path.join(first_adapter, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if len(keys) > 0:
                    first_tensor = f.get_tensor(keys[0])
                    print(f"LORA GEN DEBUG: First adapter first tensor max: {first_tensor.abs().max().item():.6e}", flush=True)
        
        return adapter_paths
    
    def generate_and_score(self, prompts, sampling_params, lora_requests, task_obj, answers):
        """
        Generates responses AND calculates fitness/stats on the GPU worker.
        """
        # Debug: Check if LoRA requests are being passed
        if lora_requests is not None:
            if isinstance(lora_requests, list) and len(lora_requests) > 0:
                print(f"GENERATE DEBUG: Received {len(lora_requests)} LoRA requests", flush=True)
                print(f"GENERATE DEBUG: First LoRA: name={lora_requests[0].lora_name}, id={lora_requests[0].lora_int_id}, path={lora_requests[0].lora_path}", flush=True)
                if len(lora_requests) > 1:
                    print(f"GENERATE DEBUG: Second LoRA: name={lora_requests[1].lora_name}, id={lora_requests[1].lora_int_id}, path={lora_requests[1].lora_path}", flush=True)
        else:
            print(f"GENERATE DEBUG: LoRA requests is None", flush=True)

        request_outputs = self.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests,
            use_tqdm=True,
        )

        # 2. Calculate fitness immediately (Local CPU)
        fitness_list = []
        distinct_counts = []
        total_responses = 0
        num_truncated = 0
        mean_char_lengths = []
        mean_token_lengths = []
        responses_for_logging = []
        
        num_prompts = len(answers)
        
        # Process linearly.
        pop_responses_buffer = ""

        for i, output in enumerate(request_outputs):
            prompt_idx = i % num_prompts
            pop_idx = i // num_prompts
            gt_answer = answers[prompt_idx]
            
            sample_fitnesses = []
            sample_char_lens = []
            sample_token_lens = []
            model_answers_set = set()

            # Format current sample for potential logging
            if pop_idx < 2 and prompt_idx < 3:
                current_prompt_log = f"\n[PROMPT {prompt_idx}]: {prompts[i]}\n"
            for j, sample in enumerate(output.outputs):
                text = sample.text
                fit, model_ans = task_obj.get_fitness(text, gt_answer)
                sample_fitnesses.append(fit)
                if model_ans:
                    model_answers_set.add(model_ans)
                
                if sample.finish_reason == "length":
                    num_truncated += 1
                
                sample_char_lens.append(len(text))
                sample_token_lens.append(len(sample.token_ids))
                total_responses += 1
                if pop_idx < 2 and prompt_idx < 3:
                    current_prompt_log += f"\n------SAMPLE {j+1}: {text} || FIT={fit}\n"

            if pop_idx < 2 and prompt_idx < 3:
                pop_responses_buffer += current_prompt_log

            if (i + 1) % num_prompts == 0 and pop_responses_buffer != "":
                if pop_responses_buffer:
                    header = f"-----POP {pop_idx} BATCH LOG-----\n"
                    responses_for_logging.append(header + pop_responses_buffer)
                    pop_responses_buffer = ""

            fitness_list.append(sample_fitnesses)
            distinct_counts.append(len(model_answers_set))
            mean_char_lengths.append(np.mean(sample_char_lens))
            mean_token_lengths.append(np.mean(sample_token_lens))

        info = {
            "total_responses": total_responses,
            "prop_truncated": num_truncated / total_responses if total_responses > 0 else 0.0,
            "mean_char_length": np.mean(mean_char_lengths),
            "mean_token_length": np.mean(mean_token_lengths),
            "mean_distinct_counts": np.mean(distinct_counts),
        }

        return fitness_list, info, responses_for_logging

def launch_engines(num_engines, model_name, population_size, lora_r):
    """Launches multiple vLLM engines, each dedicated to one GPU via Ray Placement Groups."""
    print(f"Creating {num_engines} placement groups (1 GPU each).")
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    print(f"Launching {num_engines} ESNcclLLM Ray actors.")
    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="es_lora_multinode.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
            enable_lora=True,
            max_loras=(population_size + num_engines - 1) // num_engines,
            max_lora_rank=max(lora_r, 8),
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
        )
        for strategy in strategies
    ]
    return engines, pgs

def main(args: Args):
    print("MAIN: Entered main function", flush=True)
    sys.stdout.flush()

    # --- 1. Initialize Ray FIRST (Connect to the cluster created by Slurm) ---
    # Do this before counting GPUs, otherwise only see local GPUs.
    print("MAIN: Connecting to Ray Cluster...", flush=True)
    # address="auto" picks up the RAY_ADDRESS env var set by your bash script
    ray.init(address="auto", include_dashboard=False, ignore_reinit_error=True)
    
    # --- 2. Query Ray for TOTAL Cluster Resources ---
    print("MAIN: Querying Ray for total cluster resources...", flush=True)
    resources = ray.cluster_resources()
    total_gpus = int(resources.get("GPU", 0))
    
    if total_gpus == 0:
        raise ValueError("Ray cluster reports 0 GPUs! Check your Slurm/Ray configuration.")

    args.num_gpus = total_gpus
    args.num_engines = args.num_gpus
    
    print(f"MAIN: Ray detected {args.num_gpus} GPUs across the cluster.", flush=True)
    print(f"MAIN: Launching {args.num_engines} engines (1 per GPU).", flush=True)
    sys.stdout.flush()

    assert args.population_size % 2 ==0, f"{args.population_size=} must be even for antithetic sampling."
    assert args.population_size % args.num_engines == 0, f"{args.population_size=} must be divisible by {args.num_engines=}."
    loras_per_engine = args.population_size // args.num_engines

    if args.samples_per_prompt > 1:
        assert args.temperature > 0.0, f"{args.samples_per_prompt=} requires {args.temperature=} > 0.0."
    if args.pass_at_k:
        assert args.samples_per_prompt > 1, f"{args.samples_per_prompt=} but {args.pass_at_k}"

    print("\n--- Arguments ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print(f"Detected {args.num_gpus} GPUs. Launching {args.num_engines} vLLM engines.")
    print("-----------------\n")
    sys.stdout.flush()

    fitnesses_so_far = []

    # set global random seed
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    
    # --- Setup output directories ---
    # NOTE: LORA_POPULATION_PATH is handled locally on each node by the workers.

    # --- WandB Setup ---
    run_name = f"{args.name_prefix}-" if args.name_prefix != "" else ""
    run_name += f"{args.task.replace(':', '_')}-"
    run_name += f"{args.model_name.split('/')[-1]}-"
    run_name += f"P{args.population_size}-"
    run_name += f"B{args.prompt_batch_size}-"
    run_name += f"S{args.samples_per_prompt}-"
    run_name += f"D{args.sub_dataset_size}-" if args.sub_dataset_size is not None else ""
    run_name += f"std-" if args.normalize_with_std else "no_std-"
    run_name += f"l{args.max_tokens}-"
    run_name += f"n{args.steps_per_adapter}-"
    run_name += f"lr{args.learning_rate}-"
    run_name += f"sigma{args.sigma}-"
    run_name += f"r{args.lora_r}-"
    run_name += f"alpha{args.lora_alpha}-"
    run_name += f"seed{args.base_seed}-"
    run_name += f"gpus{args.num_gpus}-"
    run_name += f"-{int(time.time())}"
    if args.use_wandb:
        print("MAIN: Initializing WandB...", flush=True)
        sys.stdout.flush()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print("MAIN: WandB initialized", flush=True)
        sys.stdout.flush()
        weave.init(args.wandb_project)
        print("MAIN: Weave initialized", flush=True)
        sys.stdout.flush()

    # Initialize Ray
    print("MAIN: Initializing Ray...", flush=True)
    sys.stdout.flush()
    ray.init(address="auto", include_dashboard=False, ignore_reinit_error=True)
    print("MAIN: Ray initialized successfully", flush=True)
    sys.stdout.flush()

    print("--- Preparing Initial Master LoRA Checkpoint ---", flush=True)
    sys.stdout.flush()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Load on CPU solely to extract PEFT shapes and config for the Head Node's logic.
    # Do NOT save this to disk for the workers; workers load 'args.model_name' directly.
    print("MAIN: Loading base model to CPU for structure extraction...", flush=True)
    sys.stdout.flush()
    base_model_pure_hf_host = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    print("MAIN: Base model loaded", flush=True)
    sys.stdout.flush()

    # Create PEFT model locally to capture shapes
    print("MAIN: Creating PEFT model wrapper...", flush=True)
    sys.stdout.flush()
    peft_model = get_peft_model(base_model_pure_hf_host, lora_config)
    peft_model.print_trainable_parameters()

    # Capture initial states to broadcast to workers
    print("MAIN: Capturing PEFT state dict...", flush=True)
    sys.stdout.flush()
    peft_state_dict = copy.deepcopy(peft_model.state_dict())
    peft_shapes_dict = {name: x.shape for name, x in peft_model.named_parameters() if name.endswith(".base_layer.weight")}
    lora_config_dict = lora_config.to_dict()
    if "target_modules" in lora_config_dict and isinstance(lora_config_dict["target_modules"], (set, tuple)):
        lora_config_dict["target_modules"] = list(lora_config_dict["target_modules"])

    print("MAIN: Cleaning up CPU model...", flush=True)
    sys.stdout.flush()
    del base_model_pure_hf_host
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Base Checkpoint structure ready.", flush=True)
    sys.stdout.flush()

    # Task Factory
    if args.task == "zeros":
        task = ZerosTask(
            batch_size=args.prompt_batch_size,
            max_tokens=args.max_tokens
        )
    elif args.task == "gsm8k":
        task = MathTask(
            batch_size=args.prompt_batch_size,
            dataset_name="openai/gsm8k",
            split="train",
            datset_size=args.sub_dataset_size,
            answer_format="none"
        )
    elif args.task == "gsm8k-boxed":
        task = MathTask(
            batch_size=args.prompt_batch_size,
            dataset_name="openai/gsm8k",
            split="train",
            datset_size=args.sub_dataset_size,
            answer_format="boxed"
        )
    elif args.task == "countdown":
        task = CountdownTask(
            batch_size=args.prompt_batch_size,
            datset_size=args.sub_dataset_size,
            end_token=None
        )
    elif args.task.startswith("math2:"):
        dataset_name = args.task.split("math2:")[1]
        task = MathTask2(
            batch_size=args.prompt_batch_size,
            # tokenizer=tokenizer,
            dataset_name=dataset_name,
            datset_size=args.sub_dataset_size,
            apply_chat_template=False,
        )
    elif args.task == "random":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_tokens=args.max_tokens,
            answer_format="none",
            max_random_number=args.samples_per_prompt,
        )
    elif args.task == "random-boxed":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_tokens=args.max_tokens,
            answer_format="boxed",
            max_random_number=args.samples_per_prompt,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
        stop=[tokenizer.eos_token],
    )
    do_eval = False
    if "math2:" in args.task and args.steps_per_eval > 0:
        do_eval = True
        print("--- Configuring Evaluation Tasks ---")

        # Ensure eval_batch_size is divisible by num_engines for multi-GPU
        if args.eval_batch_size % args.num_engines != 0:
            original_size = args.eval_batch_size
            args.eval_batch_size = ((args.eval_batch_size + args.num_engines - 1) // args.num_engines) * args.num_engines
            print(f"Adjusted eval_batch_size from {original_size} to {args.eval_batch_size} to be divisible by {args.num_engines} GPUs")

        eval_sampling_params = SamplingParams(
            temperature=0.0,
            seed=args.base_seed + 12345,
            max_tokens=args.max_tokens,
            n=1,
            stop=[tokenizer.eos_token],
        )
        eval_task = MathTask2(
            batch_size=args.eval_batch_size,
            # tokenizer=tokenizer,
            dataset_name="math-eval",
            datset_size=None,
            apply_chat_template=task.apply_chat_template,
        )
        print(f"Training on {args.task}, evaluating on {eval_task.split_names}.")

    # Launch engines
    print(f"MAIN: Launching {args.num_engines} vLLM engines...", flush=True)
    sys.stdout.flush()
    engines, pgs = launch_engines(args.num_engines, args.model_name, args.population_size, args.lora_r)
    print("Engines launched successfully.", flush=True)
    sys.stdout.flush()

    # Init inter-engine communicator once
    print("Initializing inter-engine NCCL group...")
    
    # 1. Ask Engine 0 (Rank 0) for its IP and a free port.
    #    collective_rpc returns a list of results (one per TP worker). 
    #    Since TP=1, take the first element [0].
    master_info = ray.get(engines[0].collective_rpc.remote("get_transport_info", args=()))[0]
    master_address, master_port = master_info
    print(f"Rank 0 determined Master Address: {master_address}, Port: {master_port}")

    # 2. Broadcast this address/port to ALL engines so they can connect/bind.
    init_results = ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])
    # Verify all engines initialized successfully
    for i, result in enumerate(init_results):
        if not result[0]:  # collective_rpc returns a list, take first element
            raise RuntimeError(f"NCCL group initialization failed on engine {i}!")
    print("NCCL group initialized successfully on all engines.")

    # --- Setup Local LoRA Generation on Workers ---
    print("Broadcasting initial LoRA state to workers for local generation...")
    # Pass the initial state dict, shapes, and config so workers can regenerate adapters locally
    peft_state_dict_ref = ray.put(peft_state_dict)
    peft_shapes_dict_ref = ray.put(peft_shapes_dict)
    lora_config_dict_ref = ray.put(lora_config_dict)
    
    ray.get([
        engines[i].setup_local_lora_generation.remote(
            peft_state_dict_ref, peft_shapes_dict_ref, lora_config_dict_ref, i
        )
        for i in range(args.num_engines)
    ])
    print("Workers configured for local LoRA generation.")

    def sig_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("\n--- Starting ASYNCHRONOUS ES Training Loop ---")

    # Map indices to engines. Engine i handles indices [i*batch ... (i+1)*batch]
    engine_pop_indices = []
    for i in range(args.num_engines):
        indices = list(range(i * loras_per_engine, (i + 1) * loras_per_engine))
        engine_pop_indices.append(indices)

    lora_int_id = 1
    total_time = time.time()

    for es_step in range(args.num_iterations):
        print(f"\n\n======= ES Step {es_step} / {args.num_iterations} =======")
        total_iter_start = time.time()

        # --- EVALUATION LOOP (Before training step or periodically) ---
        eval_info_dict_all = {}
        if args.steps_per_eval > 0 and es_step % args.steps_per_eval == 0 and do_eval:
            print(f"\n--- Running Evaluation at Step {es_step} ---")
            # 2. Evaluate Population
            eval_start = time.time()
            prompts, answers = eval_task.get_eval_batch()
            assert len(prompts) % args.num_engines == 0, f"{len(prompts)=} must be divisible by {args.num_engines=}"
            eval_requests_per_engine = len(prompts) // args.num_engines
            task_ref = ray.put(eval_task)
            answers_ref = ray.put(answers)
            all_refs = []

            for engine_idx in range(args.num_engines):
                llm = engines[engine_idx]
                engine_prompts = prompts[
                    engine_idx * eval_requests_per_engine : (engine_idx + 1) * eval_requests_per_engine
                ]
                engine_answers = answers[
                    engine_idx * eval_requests_per_engine : (engine_idx + 1) * eval_requests_per_engine
                ]

                # Launch the remote task (non-blocking)
                ref = llm.generate_and_score.remote(
                    engine_prompts,
                    eval_sampling_params,
                    lora_requests=None,
                    task_obj=task_ref,
                    answers=engine_answers
                )
                all_refs.append(ref)
            # GATHER: Wait for ALL evaluations to complete (single blocking call)
            if args.verbose: print(f"EVAL: Waiting for {len(all_refs)} asynchronous evaluations to complete...")
            results = ray.get(all_refs)
            list_of_fitness_arrays = []
            for i, res in enumerate(results):
                (eng_fitness, info_dict, eng_sample_output) = res
                # Reshape flat lists to (Loras_per_engine, Prompts, Samples)
                eng_fitness_np = np.array(eng_fitness)
                list_of_fitness_arrays.append(eng_fitness_np)
                if i == 0:
                    eval_info_dict_all = {k: [] for k in info_dict.keys()}
                for k, v in info_dict.items():
                    eval_info_dict_all[k].append(v)
            eval_info_dict_all = {f"eval/{k}": np.mean(v) for k, v in eval_info_dict_all.items()}
            eval_task_names = eval_task.split_names
            all_fitnesses_shaped = np.concatenate(list_of_fitness_arrays, axis=0).reshape(len(eval_task_names), eval_task.batch_size)
            print(f"\n--------------------------------")
            for eval_task_name, fitness_array in zip(eval_task_names, all_fitnesses_shaped):
                mean_fitness = np.mean(fitness_array)
                eval_info_dict_all[f"eval/{eval_task_name}_mean_fitness"] = mean_fitness
                print(f"EVAL {eval_task_name}: Mean fitness: {mean_fitness:.4f}")
            print(f"--------------------------------\n")
            eval_time = time.time() - eval_start
            if args.verbose: print(f"EVAL complete in {eval_time:.4f}s")

        # 1. Generate local LoRA adapters directly on the workers
        if es_step % args.steps_per_adapter == 0:
            lora_gen_start = time.time()
            if args.verbose: print(f"Triggering distributed LoRA generation on {args.num_engines} engines...")
            
            # Parallel call to all engines to generate their specific adapters
            # Each engine returns the list of PATHS it generated locally
            # These paths are valid on the worker node, but maybe not on head node.
            # These paths are used to construct LoRARequests that are sent BACK to the same worker.
            engine_paths = ray.get([
                engines[i].generate_local_adapters.remote(
                    engine_pop_indices[i], es_step, args
                )
                for i in range(args.num_engines)
            ])
            
            lora_gen_time = time.time() - lora_gen_start
            if args.verbose: print(f"Distributed LoRA adapter generation complete in {lora_gen_time:.4f}s")
        else:
            lora_gen_time = 0.0
            # Important: Rank-specific paths (each engine has its own /dev/shm directory)
            engine_paths = []
            for i in range(args.num_engines):
                paths = [os.path.join(f"{LORA_POPULATION_PATH}_{i}", f"pop_{idx}") for idx in engine_pop_indices[i]]
                engine_paths.append(paths)

        # 2. Evaluate Population
        vllm_start = time.time()
        prompts, answers = task.get_batch()
        
        task_ref = ray.put(task)
        answers_ref = ray.put(answers)
        all_refs = []

        for engine_idx in range(args.num_engines):
            llm = engines[engine_idx]
            
            # Paths allocated to this engine
            local_paths = engine_paths[engine_idx]
            pop_indices = engine_pop_indices[engine_idx]
            
            # Expand for batch size (prompts) as have N adapters and M prompts
            # and want to run every adapter on every prompt.
            
            # Create list of (prompt, lora_req) tuples to keep order aligned
            engine_batch_prompts = []
            engine_batch_lora_reqs = []
            
            for path_idx, lora_path in enumerate(local_paths):
                # Unique ID for cache: pop_id + step (to invalidate old cache if needed, though folder overwrite handles it mostly)
                pop_id = pop_indices[path_idx]
                req = LoRARequest(
                    lora_name=f"adapter_{pop_id}",
                    lora_int_id=pop_id + 1 + (es_step * 10000), # Ensure ID changes if weight changes
                    lora_path=lora_path
                )
                
                # Repeat for all prompts
                engine_batch_lora_reqs.extend([req] * len(prompts))
                engine_batch_prompts.extend(prompts)
            
            # Launch the remote task (non-blocking)
            ref = llm.generate_and_score.remote(
                engine_batch_prompts, 
                sampling_params, 
                lora_requests=engine_batch_lora_reqs,
                task_obj=task_ref,
                answers=answers_ref
            )
            all_refs.append(ref)
            
        # GATHER: Wait for ALL evaluations to complete (single blocking call)
        if args.verbose: print(f"Waiting for {len(all_refs)} asynchronous evaluations to complete...")
        results = ray.get(all_refs)
        vllm_time = time.time() - vllm_start
        if args.verbose: print(f"vLLM evals + fitness calc complete in {vllm_time:.4f}s")
        
        aggregation_start = time.time()
        list_of_fitness_arrays = []
        for i, res in enumerate(results):
            (eng_fitness, info_dict, eng_sample_output) = res
            # Reshape flat lists to (Loras_per_engine, Prompts, Samples)
            eng_fitness_np = np.array(eng_fitness).reshape(loras_per_engine, len(prompts), args.samples_per_prompt)
            list_of_fitness_arrays.append(eng_fitness_np)
            if i == 0:
                info_dict_all = {k: [] for k in info_dict.keys()}
            for k, v in info_dict.items():
                info_dict_all[k].append(v)
        info_dict_all = {k: np.mean(v) for k, v in info_dict_all.items()}
        all_fitnesses_shaped = np.concatenate(list_of_fitness_arrays, axis=0)
        assert all_fitnesses_shaped.shape == (args.population_size, len(prompts), args.samples_per_prompt), \
            f"Fitness array shape mismatch! Got {all_fitnesses_shaped.shape}, expected {(args.population_size, len(prompts), args.samples_per_prompt)}"
        aggregation_time = time.time() - aggregation_start
        if args.verbose: print(f"Results aggregation complete in {aggregation_time:.4f}s")

        # all_fitnesses_shaped: Shape (population_size, num_prompts, samples_per_prompt)
        if args.pass_at_k:
            fitnesses_shaped = np.max(all_fitnesses_shaped, axis=2)  # Shape: (population_size, num_prompts)
        else:
            fitnesses_shaped = np.mean(all_fitnesses_shaped, axis=2)  # Shape: (population_size, num_prompts)
        fitness_per_prompt = np.mean(fitnesses_shaped, axis=0, keepdims=True)  # Shape: (1, num_prompts)
        fitness_per_pop = np.mean(fitnesses_shaped, axis=1)  # Shape: (population_size,) (for logging)
        normalized_fitnesses = np.mean(fitnesses_shaped - fitness_per_prompt, axis=1) # Shape: (population_size,)
        normalized_fitnesses_std = np.std(normalized_fitnesses)
        if args.normalize_with_std:
            normalized_fitnesses = normalized_fitnesses / (normalized_fitnesses_std + 1e-8)

        # Logging
        if args.verbose:
            for pop_idx in range(2):
                print(f"\n----POP {pop_idx}:")
                generations_for_logging = results[0][2]
                for text in generations_for_logging:
                    print(text)
                print(f"----FITNESS: {fitness_per_pop[pop_idx]:.4f}, NORMALIZED FITNESS: {normalized_fitnesses[pop_idx]:.4f}\n")
            print(f"\nFitness per prompt (averaged over population): {fitness_per_prompt}")
        mean_fitness = float(np.mean(fitnesses_shaped))
        min_fitness = float(np.min(fitnesses_shaped))
        max_fitness = float(np.max(fitnesses_shaped))
        std_normalized_fitness = float(normalized_fitnesses_std)
        pass_at_k_fitness = float(np.mean(np.max(all_fitnesses_shaped, axis=2)))
        std_in_samples = float(np.std(all_fitnesses_shaped, axis=2).mean()) if args.samples_per_prompt > 1 else 0.0
        print(f"Mean fitness: {mean_fitness:.4f}, min: {min_fitness:.4f}, max: {max_fitness:.4f}, std_normalized_fitness: {std_normalized_fitness:.4f}, pass@k fitness: {pass_at_k_fitness:.4f}, std_in_samples: {std_in_samples:.4f}, distinct_answers: {info_dict_all.get('mean_distinct_counts', -1.0):.4f}, prop_truncated: {info_dict_all.get('prop_truncated', -1.0):.4f}")
        for k, v in info_dict_all.items():
            print(f"  {k}: {v:.4f}")

        # Compute ES update ONLY on engine 0
        update_start = time.time()
        ray.get(engines[0].collective_rpc.remote(
            "apply_lora_es_update", 
            args=(normalized_fitnesses, peft_shapes_dict, es_step, args)
        ))
        update_time = time.time() - update_start
        if args.verbose: print(f"Applied ES update on Engine 0 in {update_time:.4f}s")

        # 4. Broadcast updated weights
        print("BROADCAST: Starting weight broadcast to all engines...", flush=True)
        sys.stdout.flush()
        broadcast_start = time.time()
        print(f"BROADCAST: Calling broadcast_all_weights on {len(engines)} engines...", flush=True)
        sys.stdout.flush()

        # Create remote calls
        broadcast_refs = []
        for i, e in enumerate(engines):
            print(f"BROADCAST: Dispatching call to engine {i}...", flush=True)
            sys.stdout.flush()
            ref = e.collective_rpc.remote("broadcast_all_weights", args=(0,))
            broadcast_refs.append(ref)
            print(f"BROADCAST: Engine {i} call dispatched (ref: {ref})", flush=True)
            sys.stdout.flush()

        print(f"BROADCAST: All {len(broadcast_refs)} calls dispatched, waiting for results...", flush=True)
        sys.stdout.flush()
        broadcast_results = ray.get(broadcast_refs)
        print(f"BROADCAST: Received results from all engines", flush=True)
        sys.stdout.flush()

        # Check if any engine failed NCCL broadcast
        failed_engines = [i for i, result in enumerate(broadcast_results) if not result[0]]

        if failed_engines:
            if args.verbose:
                print(f"NCCL broadcast failed on engines {failed_engines}. Falling back to Ray-based broadcast...")

            # Fallback: use Ray to broadcast weights
            # Get state dict from engine 0
            state_dict_refs = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))
            state_dict = state_dict_refs[0]  # collective_rpc returns list

            # Broadcast via Ray's object store
            state_dict_ref = ray.put(state_dict)

            # Set state dict on all other engines
            ray.get([
                engines[i].collective_rpc.remote("set_model_state_dict", args=(state_dict_ref,))
                for i in range(1, args.num_engines)  # Skip engine 0 (source)
            ])

        broadcast_time = time.time() - broadcast_start
        method = "Ray" if failed_engines else "NCCL"
        if args.verbose:
            print(f"Broadcasted updated weights to all engines in {broadcast_time:.4f}s ({method})", flush=True)
            sys.stdout.flush()

        # 5. Logging and WandB
        total_iter_end = time.time()
        iter_time = total_iter_end - total_iter_start
        if args.use_wandb:
            wandb.log({
                "mean_fitness": mean_fitness,
                "min_fitness": min_fitness,
                "max_fitness": max_fitness,
                "std_normalized_fitness": std_normalized_fitness,
                "pass_at_k_fitness": pass_at_k_fitness,
                "std_in_samples": std_in_samples,
                "es_step": es_step,
                "pop_step": es_step // args.steps_per_adapter,
                "time/vllm": vllm_time,
                "time/aggregation": aggregation_time,
                "time/lora_gen": lora_gen_time,
                "time/update": update_time,
                "time/broadcast": broadcast_time,
                "time/iteration": iter_time,
                "total_time": time.time() - total_time,
                **info_dict_all,
                **eval_info_dict_all,
            })
        if args.verbose:
            total_time2 = vllm_time + aggregation_time + lora_gen_time + update_time + broadcast_time
            print(f"TIMES: total: {iter_time:.4f}s (or {total_time2}s),  LoRA gen: {lora_gen_time:.4f}s, vLLM+Score: {vllm_time:.4f}s, Aggregation: {aggregation_time:.4f}s, ES update: {update_time:.4f}s, broadcast: {broadcast_time:.4f}s", flush=True)
            sys.stdout.flush()

        fitnesses_so_far.append(mean_fitness)
        print(f"\n---\nFitnesses so far: {fitnesses_so_far}\n---\n", flush=True)
        print(f"======= ES Step {es_step} finished =======\n", flush=True)
        sys.stdout.flush()

    print("\n--- ES Training Complete ---")

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("SCRIPT STARTED - Parsing arguments...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    args = tyro.cli(Args)

    print("=" * 80, flush=True)
    print("ARGUMENTS PARSED - Starting main function...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    main(args)