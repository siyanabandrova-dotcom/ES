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
from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file

# Default Hyperparameters
EXPERIMENT_DIR = "/dev/shm/outputs_es_lora"
LORA_POPULATION_PATH = "/dev/shm/es_lora_population_async"

@dataclass
class Args:
    """ES Fine-tuning for Countdown Task with multi-engine NCCL sync and LoRA population"""
    model_name: str = "Qwen/Qwen2-0.5B" # Example small model for fast testing "Qwen/Qwen2.5-3B-Instruct"
    # --- ES Hyperparameters ---
    sigma: float = 0.001
    population_size: int = 100
    num_iterations: int = 1000
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    task: str = "zeros"  # Options: "zeros", "gsm8k", "gsm8k-boxed"
    prompt_batch_size: int = 2
    pass_at_k: bool = False # Whether to optimize for pass@k (for tasks like GSM8K)
    
    # --- LoRA Config ---
    lora_r: int = 4
    lora_alpha: int = 1
    steps_per_adapter: int = 4
    learning_rate: float = 0.001

    # --- Runtime Config ---
    experiment_dir: str = EXPERIMENT_DIR
    num_gpus: int = None
    num_engines: int = None
    verbose: bool = True
    base_seed: int = 0
    sub_dataset_size: int = 1000

    # --- WandB ---
    use_wandb: bool = True
    wandb_project: str = "hyperscalees-vllm-nccl"
    name_prefix: str = f"C-async"


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def map_peft_updates_to_vllm(peft_updates_dict, vllm_params_dict):
    vllm_updates_dict = {
        name: torch.zeros_like(x) for name, x in vllm_params_dict.items()
        if name.endswith(".base_layer.weight")
    }
    for peft_name, weight_update in peft_updates_dict.items():
        # Example peft: base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
        # Example vllm: model.layers.0.self_attn.qkv_proj.base_layer.weight
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

@torch.no_grad()
def create_lora_adapter_files(
    peft_model, peft_params_dict, peft_state_dict, peft_shapes_dict, adapter_paths, es_step, args: Args
):
    """
    Creates and saves the LoRA A and B weights for a single population member, 
    incorporating the ES noise.
    """
    pop_step = es_step // args.steps_per_adapter
    for pop_idx in range(args.population_size):
        peft_model.load_state_dict(peft_state_dict)
        # Create unique LoRA name and path for this adapter at this step
        for layer_idx, (peft_name, weight_shape) in enumerate(peft_shapes_dict.items()):
            lora_a_name = peft_name.replace("base_layer.weight", "lora_A.default.weight")
            lora_b_name = peft_name.replace("base_layer.weight", "lora_B.default.weight")
            lora_a = peft_params_dict[lora_a_name]
            lora_b = peft_params_dict[lora_b_name]
            lora_b_shape, lora_a_shape = (weight_shape[0], args.lora_r), (args.lora_r, weight_shape[1])
            assert lora_a.shape == lora_a_shape, f"{lora_a.shape=} vs {lora_a_shape=}"
            assert lora_b.shape == lora_b_shape, f"{lora_b.shape=} vs {lora_b_shape=}"
            noise_a, noise_b = get_rng_noise(
                base_seed=args.base_seed,
                num_pop_pairs=args.population_size//2,
                pop_pair_idx=pop_idx//2,
                num_layers=len(peft_shapes_dict.keys()),
                layer_idx=layer_idx,
                step=pop_step,
                shapes=[lora_a_shape, lora_b_shape],
            )
            noise_b *= math.sqrt(args.sigma)
            noise_a *= math.sqrt(args.sigma)
            lora_a.zero_()
            lora_b.zero_()
            lora_a.add_(noise_a)
            if pop_idx % 2 == 1:
                lora_b.add_(-noise_b)
            else:
                lora_b.add_(noise_b)

        adapter_path = adapter_paths[pop_idx]
        peft_model.save_pretrained(adapter_path)

    gc.collect()
    torch.cuda.empty_cache()

@ray.remote
def process_engine_outputs_and_calc_fitness(
    engine_outputs, 
    task_obj, 
    answers, 
    engine_lora_count, 
    prompt_count, 
    samples_per_prompt
) -> np.ndarray:
    """
    This task runs remotely. It gets the outputs from one engine,
    calculates fitness for all of them, and returns just the fitness array.
    """
    
    # 1. Parse outputs and calculate fitness (combines old loops)
    fitness_array = np.zeros((engine_lora_count, prompt_count, samples_per_prompt))
    
    for i, output in enumerate(engine_outputs):
        # Calculate which pop_idx and prompt_idx this output corresponds to
        pop_idx_local = i // prompt_count  # LoRA index relative to this engine
        prompt_idx = i % prompt_count
        
        answer_to_q = answers[prompt_idx]
        
        for sample_idx, sample_output in enumerate(output.outputs):
            response_text = sample_output.text
            fitness = task_obj.get_fitness(response_text, answer_to_q)
            fitness_array[pop_idx_local, prompt_idx, sample_idx] = fitness
            
    return fitness_array

class WorkerExtension:
    """
    Custom extension for vLLM workers to handle ES update and NCCL broadcast.
    This class is passed to the vLLM engine via 'worker_extension_cls'.
    """
    @torch.no_grad()
    def apply_lora_es_update(self, normalized_fitnesses: list[tuple[int, float]], peft_shapes_dict, es_step: int, args: Args):
        """
        Computes and applies the ES update delta to the base model weights.
        This must only run on the master engine (gpu_rank 0).
        """
        if self.gpu_rank != 0: 
            return False
        
        peft_updates_dict = {name: torch.zeros(x, device=self.device) for name, x in peft_shapes_dict.items()}
        vllm_params_dict = {name: x for name, x in self.model_runner.model.named_parameters()}
        
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

        vllm_updates_dict = map_peft_updates_to_vllm(peft_updates_dict, vllm_params_dict)

        for i, (name, update) in enumerate(vllm_updates_dict.items()):
            if name in vllm_params_dict:
                gradient = (1.0 / (args.population_size * args.sigma + 1e-8)) * update * args.learning_rate
                vllm_params_dict[name] += gradient

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return True
    
    def init_inter_engine_group(self, master_address: str, master_port: int, gpu_rank: int, world_size: int):
        """Initializes the NCCL communication group across all engines."""
        self.device = self.model_runner.device
        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, gpu_rank, world_size, self.device
        )
        return True

    @torch.no_grad()
    def broadcast_all_weights(self, src_rank: int):
        """Broadcasts all base model weights from src_rank (Engine 0) to all others."""
        if not self.inter_pg:
            return False

        for name, param in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    
class ESNcclLLM(LLM):
    """vLLM subclass using the custom WorkerExtension."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name, population_size, lora_r):
    """Launches multiple vLLM engines, each dedicated to one GPU via Ray Placement Groups."""
    # Strict 1-GPU isolation via PGs
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
        # Note: worker_extension_cls must point to the class name defined in this file
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="es_lora_nccl_async.WorkerExtension",
            dtype="float16", 
            enable_prefix_caching=False,
            enforce_eager=False,
            enable_lora=True,
            max_loras=(population_size + num_engines - 1) // num_engines,
            max_lora_rank=max(lora_r, 8),
            gpu_memory_utilization=0.75,
        )
        for strategy in strategies
    ]
    return engines, pgs


def main(args: Args):
    # --- Setup/Init ---
    args.num_gpus = torch.cuda.device_count()
    args.num_engines = args.num_gpus
    assert args.population_size % 2 ==0, f"{args.population_size=} must be even for antithetic sampling."
    assert args.population_size % args.num_engines == 0, f"{args.population_size=} must be divisible by {args.num_engines=}."
    loras_per_engine = args.population_size // args.num_engines
    if args.samples_per_prompt > 1:
        assert args.temperature > 0.0, f"{args.samples_per_prompt=} requires {args.temperature=} > 0.0."
    if args.pass_at_k:
        assert args.samples_per_prompt > 1, f"{args.samples_per_prompt=} but {args.pass_at_k}"
    assert args.task in ["zeros", "gsm8k", "gsm8k-boxed"], f"Unknown task: {args.task}"
    
    print("--- Arguments ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print(f"Detected {args.num_gpus} GPUs. Launching {args.num_engines} vLLM engines.")
    print("-----------------\n")

    fitnesses_so_far = []

    # set global random seed
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))
    
    # --- Setup output directories and adapter paths ---
    if os.path.exists(LORA_POPULATION_PATH):
        shutil.rmtree(LORA_POPULATION_PATH)
    os.makedirs(LORA_POPULATION_PATH, exist_ok=True)
    
    adapter_paths = []
    for pop_idx in range(args.population_size):
        adapter_path = os.path.join(LORA_POPULATION_PATH, f"pop_{pop_idx}")
        adapter_paths.append(adapter_path)

    # --- WandB Setup ---
    run_name = f"{args.name_prefix}-" if args.name_prefix != "" else ""
    run_name += f"{args.task}-"
    run_name += f"{args.model_name.split('/')[-1]}-"
    run_name += f"P{args.population_size}-"
    run_name += f"B{args.prompt_batch_size}-"
    run_name += f"S{args.samples_per_prompt}-"
    run_name += f"n{args.steps_per_adapter}-"
    run_name += f"lr{args.learning_rate}-"
    run_name += f"sigma{args.sigma}-"
    run_name += f"r{args.lora_r}-"
    run_name += f"alpha{args.lora_alpha}-"
    run_name += f"seed{args.base_seed}-"
    run_name += f"gpus{args.num_gpus}-"
    run_name += f"-{int(time.time())}"
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Initialize Ray
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Prepare an HF checkpoint for vLLM to load
    logging_dir = f"{args.experiment_dir}/es_lora_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_model_path = f"{logging_dir}/base_model"
    if os.path.exists(base_model_path): shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)

    print("--- Preparing Initial Master LoRA Checkpoint ---")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model_pure_hf_host = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(base_model_path)
    base_model_pure_hf_host.save_pretrained(base_model_path) 

    peft_model = get_peft_model(base_model_pure_hf_host, lora_config)
    peft_model.print_trainable_parameters()
    peft_state_dict = copy.deepcopy(peft_model.state_dict())
    peft_params_dict = {name: param for name, param in peft_model.named_parameters()}
    # print(f"{[(name, x.shape) for name, x in peft_model.named_parameters()]=}")
    peft_shapes_dict = {name: x.shape for name, x in peft_model.named_parameters() if name.endswith(".base_layer.weight")}
    del base_model_pure_hf_host
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Base Checkpoint ready.")

    # Task
    if args.task == "zeros":
        from tasks import ZerosTask
        task = ZerosTask(batch_size=args.prompt_batch_size, max_tokens=args.max_tokens)
    elif args.task == "gsm8k":
        from tasks import MathTask
        task = MathTask(batch_size=args.prompt_batch_size,
                        dataset_name="openai/gsm8k",
                        split="train",
                        datset_size=args.sub_dataset_size,
                        answer_format="none"
            )
    elif args.task == "gsm8k-boxed":
        from tasks import MathTask
        task = MathTask(batch_size=args.prompt_batch_size,
                        dataset_name="openai/gsm8k",
                        split="train",
                        datset_size=args.sub_dataset_size,
                        answer_format="boxed"
            )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Launch engines
    engines, pgs = launch_engines(args.num_engines, base_model_path, args.population_size, args.lora_r)
    print("Engines launched successfully.")

    # Init inter-engine communicator once
    print("Initializing inter-engine NCCL group...")
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])
    print("NCCL group initialized.")


    def cleanup():
        print("\nCleaning up Ray resources, adapter files, and WandB...")
        for llm in engines:
            try: ray.kill(llm) 
            except Exception: pass
        print("Ray actors killed.")
        for pg in pgs:
            try: remove_placement_group(pg)
            except Exception: pass
        print("Placement groups removed.")
        ray.shutdown()
        print("Ray shutdown complete.")
        if args.use_wandb: wandb.finish()
        print("WandB finished.")
        if os.path.exists(LORA_POPULATION_PATH): shutil.rmtree(LORA_POPULATION_PATH)
        print("Adapter files removed.")
        print("Cleanup complete.\n")

    def sig_handler(sig, frame):
        # cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("\n--- Starting ASYNCHRONOUS ES Training Loop ---")
    
    # Store LoRARequest objects
    lora_requests = []
    for pop_idx in range(args.population_size):
        lora_name = f"adapter_{pop_idx}"
        lora_int_id = pop_idx + 1 # Start from 1
        lora_path = os.path.join(LORA_POPULATION_PATH, f"pop_{pop_idx}")
        lora_requests.append(
            LoRARequest(lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path)
        )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
    )

    total_time = time.time()
    
    for es_step in range(args.num_iterations):
        print(f"\n\n======= ES Step {es_step} / {args.num_iterations} =======")
        total_iter_start = time.time()

        # 1. Create and save population of noisy LoRA adapters
        if es_step % args.steps_per_adapter == 0:
            lora_gen_start = time.time()
            if args.verbose: print(f"Creating and saving {args.population_size} noisy LoRA adapters...")
            create_lora_adapter_files(
                peft_model, peft_params_dict, peft_state_dict, peft_shapes_dict, adapter_paths, es_step, args
            )
            lora_gen_time = time.time() - lora_gen_start
            if args.verbose: print(f"LoRA adapter generation complete in {lora_gen_time:.4f}s")
        else:
            lora_gen_time = 0.0

        # 2. Evaluate Population (ASYNCHRONOUS SCATTER/GATHER)
        # SCATTER: Launch all tasks asynchronously and collect references
        vllm_start = time.time()
        all_refs = []
        for engine_idx in range(args.num_engines):
            llm = engines[engine_idx]
            engine_lora_requests = lora_requests[
                engine_idx * loras_per_engine : (engine_idx + 1) * loras_per_engine]

            # Launch the remote task (non-blocking)
            prompts, answers = task.get_batch()
            repeated_engine_lora_requests = []
            repeated_prompts = []
            for lora_request in engine_lora_requests:
                repeated_engine_lora_requests.extend([lora_request] * len(prompts))
                repeated_prompts.extend(prompts)
            assert len(repeated_prompts) == len(repeated_engine_lora_requests), f"{len(repeated_prompts)=} != {len(repeated_engine_lora_requests)=}"

            ref = llm.generate.remote(
                repeated_prompts, 
                sampling_params, 
                lora_request=repeated_engine_lora_requests,
                use_tqdm=False,
            )
            # Store the reference and its original index
            all_refs.append((engine_idx, ref))

        # GATHER: Wait for ALL evaluations to complete (single blocking call)
        if args.verbose: print(f"Waiting for {len(all_refs)} asynchronous evaluations to complete...")
        vllm_time = time.time() - vllm_start
        if args.verbose: print(f"vLLM evals complete in {vllm_time:.4f}s")
        
        if args.verbose: print("Parsing vLLM outputs and calculating fitness in parallel...")
        fitness_start = time.time()
        
        # Put the task and answers in the object store once for all workers to read
        task_ref = ray.put(task)
        answers_ref = ray.put(answers)

        fitness_tasks = []
        engine_indices = [] # To keep track of the order

        # Launch parallel processing tasks, passing object references (futures)
        # We are NOT pulling the large string data to this driver script.
        for engine_idx, ref in all_refs:
            engine_lora_count = loras_per_engine # This assumes equal split, which your code does
            
            fitness_tasks.append(
                process_engine_outputs_and_calc_fitness.remote(
                    ref, 
                    task_ref, 
                    answers_ref, 
                    engine_lora_count, 
                    len(prompts), 
                    args.samples_per_prompt
                )
            )
            engine_indices.append(engine_idx)

        # Now, gather the results. This is only lists of small floats.
        # This is memory-efficient.
        list_of_fitness_arrays = ray.get(fitness_tasks)
        
        # We need to re-sort the arrays in case ray.get returns them out of order
        # (though it usually preserves it)
        sorted_fitness_arrays = [
            arr for _, arr in sorted(zip(engine_indices, list_of_fitness_arrays))
        ]

        # Concatenate the results from all engines into one big array
        all_fitnesses_shaped = np.concatenate(sorted_fitness_arrays, axis=0)
        
        # Verify the final shape
        expected_shape = (args.population_size, len(prompts), args.samples_per_prompt)
        assert all_fitnesses_shaped.shape == expected_shape, \
            f"Fitness array shape mismatch! Got {all_fitnesses_shaped.shape}, expected {expected_shape}"
        
        fitness_time = time.time() - fitness_start
        if args.verbose: print(f"Fitness calculation complete in {fitness_time:.4f}s")

        engine_0_prompts = []
        engine_0_responses = []
        if args.verbose:
            print("Gathering sample outputs from Engine 0 for logging...")
            try:
                # Get data from *only* the first engine (index 0)
                # all_refs is a list of (engine_idx, ref) tuples
                first_engine_ref = all_refs[0][1] 
                engine_0_outputs = ray.get(first_engine_ref) # Safe, just one engine's data
                
                engine_lora_count = loras_per_engine
                num_prompts_for_parsing = len(prompts) # from the main script scope
                
                # Pre-allocate lists for this engine's data
                engine_0_prompts = [["" for _ in range(num_prompts_for_parsing)] for _ in range(engine_lora_count)]
                engine_0_responses = [[[] for _ in range(num_prompts_for_parsing)] for _ in range(engine_lora_count)]

                for i, output in enumerate(engine_0_outputs):
                    pop_idx_local = i // num_prompts_for_parsing  # LoRA index relative to this engine
                    prompt_idx = i % num_prompts_for_parsing
                    
                    if pop_idx_local < engine_lora_count and prompt_idx < num_prompts_for_parsing:
                         engine_0_prompts[pop_idx_local][prompt_idx] = output.prompt
                         engine_0_responses[pop_idx_local][prompt_idx] = [o.text for o in output.outputs]
                    
            except Exception as e:
                print(f"Warning: Failed to gather log data from engine 0. {e}")
                # Clear them so we don't try to print later
                engine_0_prompts = []
                engine_0_responses = []

        # all_fitnesses_shaped: Shape (population_size, num_prompts, samples_per_prompt)
        if args.pass_at_k:
            fitnesses_shaped = np.max(all_fitnesses_shaped, axis=2)  # Shape: (population_size, num_prompts)
        else:
            fitnesses_shaped = np.mean(all_fitnesses_shaped, axis=2)  # Shape: (population_size, num_prompts)
        fitness_per_prompt = np.mean(fitnesses_shaped, axis=0, keepdims=True)  # Shape: (1, num_prompts)
        fitness_per_pop = np.mean(fitnesses_shaped, axis=1)  # Shape: (population_size,) (for logging)
        normalized_fitnesses = np.mean(fitnesses_shaped - fitness_per_prompt, axis=1) # Shape: (population_size,)
        normalized_fitnesses_std = np.std(normalized_fitnesses)
        normalized_fitnesses = normalized_fitnesses / (normalized_fitnesses_std + 1e-8)

        # Logging
        if args.verbose:
            num_pops_to_print = min(4, args.population_size)
            num_qs_to_print = min(2, len(prompts))
            num_responses_to_print = min(2, args.samples_per_prompt)
            
            # Check if we successfully gathered log data
            has_log_data = len(engine_0_prompts) > 0 and len(engine_0_responses) > 0

            for pop_idx in range(num_pops_to_print):
                print(f"\n----POP {pop_idx}:")
                
                # Check if this pop_idx is one we have log data for (i.e., from engine 0)
                if has_log_data and pop_idx < len(engine_0_prompts):
                    for q_idx in range(num_qs_to_print):
                        # Ensure we don't go out of bounds if prompt count changed (shouldn't happen)
                        if q_idx < len(engine_0_prompts[pop_idx]):
                            print(f"PROMPT {q_idx}: {engine_0_prompts[pop_idx][q_idx]}")
                            print(f"RESPONSES: {engine_0_responses[pop_idx][q_idx][:num_responses_to_print]}")
                            print(f"FITNESSES: {[x.item() for x in all_fitnesses_shaped[pop_idx][q_idx]]}")
                        
                # Always print the summary fitness, which we have for everyone
                print(f"FITNESS: {fitness_per_pop[pop_idx]:.4f}, NORMALIZED FITNESS: {normalized_fitnesses[pop_idx]:.4f}")

            print(f"Fitness per prompt (averaged over population): {fitness_per_prompt}")
        
        mean_fitness = float(np.mean(fitnesses_shaped))
        min_fitness = float(np.min(fitnesses_shaped))
        max_fitness = float(np.max(fitnesses_shaped))
        std_normalized_fitness = float(normalized_fitnesses_std)
        pass_at_k_fitness = float(np.mean(np.max(all_fitnesses_shaped, axis=2)))
        std_in_samples = float(np.std(all_fitnesses_shaped, axis=2).mean()) if args.samples_per_prompt > 1 else 0.0
        current_time = time.time()
        print(f"Mean fitness: {mean_fitness:.4f}, min: {min_fitness:.4f}, max: {max_fitness:.4f}, std_normalized_fitness: {std_normalized_fitness:.4f}")
        fitnesses_so_far.append(mean_fitness)
        print(f"\nFitnesses so far: {fitnesses_so_far}\n")

        # Compute ES update ONLY on engine 0
        update_start = time.time()
        ray.get(engines[0].collective_rpc.remote(
            "apply_lora_es_update", 
            args=(normalized_fitnesses, peft_shapes_dict, es_step, args)
        ))
        update_time = time.time() - update_start

        if args.verbose: print(f"Applied ES update on Engine 0 in {update_time:.4f}s")
        if args.use_wandb: wandb.log({"time/update_application": update_time, "es_step": es_step})

        # 4. Broadcast updated weights from engine 0 to all engines (NCCL)
        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        broadcast_time = time.time() - broadcast_start
        if args.verbose: print(f"Broadcasted updated weights to all engines in {broadcast_time:.4f}s (NCCL sync)")
        if args.use_wandb: wandb.log({"time/broadcast": broadcast_time, "es_step": es_step})

        total_iter_end = time.time()
        iter_time = total_iter_end - total_iter_start
        
        if args.use_wandb: wandb.log({"time_per_es_step": iter_time, "es_step": es_step})

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
                "time/fitness": fitness_time,
                "time/lora_gen": lora_gen_time,
                "time/update": update_time,
                "time/broadcast": broadcast_time,
                "time/iteration": iter_time,
                "total_time": current_time - total_time,
            })
        
        if args.verbose:
            total_time2 = vllm_time + fitness_time + lora_gen_time + update_time + broadcast_time
            print(f"TIMES: total: {iter_time:.4f}s (or {total_time2}s),  LoRA gen: {lora_gen_time:.4f}s, vLLM: {vllm_time:.4f}s, fitness: {fitness_time:.4f}s, ES update: {update_time:.4f}s, broadcast: {broadcast_time:.4f}s")
        print(f"======= ES Step {es_step} finished =======\n")

    print("\n--- ES Training Complete ---")
    # cleanup()


if __name__ == "__main__":
    # Ensure WorkerExtension is defined before main execution
    args = tyro.cli(Args)
    main(args)
