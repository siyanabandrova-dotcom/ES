import torch
import torch.optim as optim
from dataclasses import dataclass
import os
import shutil
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import ray
import tyro
import numpy as np
from safetensors.torch import save_file
import copy
import wandb
import math

@dataclass
class Args:
    """Train an LLM with ES and LoRA on GSM8K."""
    model: str = "Qwen/Qwen2-0.5B"
    output_dir: str = "./es_lora_results"
    num_gpus: int = None

    # --- ES Hyperparameters ---
    num_steps: int = 100
    population_size: int = 100      # must be even for antithetic sampling
    learning_rate: float = 0.1      # learning rate for Adam optimizer
    sigma: float = 0.1              # standard deviation for the ES noise
    lora_r: int = 1                 # lora rank
    lora_alpha: int = 16            # lora alpha
    steps_per_adapter: int = 1      # number of ES steps to reuse the same adapter population

    # --- Generation Hyperparameters ---
    max_tokens: int = 50            # max tokens to generate
    task: str = "zeros"             # task to use: "zeros" or "gsm8k", "gsm8k-boxed"
    sub_dataset_size: int = 16      # size of the gsm8k subset to use (if task is gsm8k)
    prompt_batch_size: int = 2      # prompts per population member

    # --- Other settings ---
    save_interval: int = -1          # save checkpoint every n steps
    seed: int = 0

    # --- WandB ---
    use_wandb: bool = False
    wandb_project: str = "hyperscalees-vllm-1"
    name_prefix: str = f"A"


def get_rng_noise(
        base_seed: int,
        num_pop_pairs: int,
        pop_pair_idx: int,
        num_layers: int,
        layer_idx: int,
        step: int,
        shapes: list,
        devices: list,
        ) -> dict[torch.device, torch.Generator]:
    """
    Create a dictionary of RNGs, one for each device.
    All RNGs are seeded with the same ID to ensure deterministic noise
    across different devices.
    """
    assert all(device == devices[0] for device in devices), "All devices must be the same for this function."
    id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    torch_rng = torch.Generator(device=devices[0]).manual_seed(id)

    noise_a, noise_b = (torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=shape,
                    device=device,
                    generator=torch_rng,
                ) for shape, device in zip(shapes, devices))

    return noise_a, noise_b

def main(args: Args):
    # print the args
    args.num_gpus = torch.cuda.device_count()
    print("Args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    fitnesses_log = []
    ADAPTER_POPULATION_PATH = f"/dev/shm/es_lora_population_{args.name_prefix}"

    # --- WandB Setup ---
    run_name = f"{args.name_prefix}-" if args.name_prefix != "" else ""
    run_name += f"{args.task}-"
    run_name += f"pop{args.population_size}-"
    run_name += f"s{args.steps_per_adapter}-"
    run_name += f"lr{args.learning_rate}-"
    run_name += f"sigma{args.sigma}-"
    run_name += f"r{args.lora_r}-"
    run_name += f"alpha{args.lora_alpha}-"
    run_name += f"{args.model.split('/')[-1]}-"
    run_name += f"seed{args.seed}-"
    run_name += f"gpus{args.num_gpus}-"
    run_name += f"-{int(time.time())}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # Ensure population size is even
    if args.population_size % 2 != 0:
        raise ValueError(f"population_size must be even for antithetic sampling, but got {args.population_size}")

    # Initialize Ray for distributed vLLM
    ray.init(num_gpus=args.num_gpus)

    # --- Setup output directories ---
    os.makedirs(ADAPTER_POPULATION_PATH, exist_ok=True)

    print(f"\n--- Step 1: Creating Master LoRA Adapter ---")

    # 1. Define LoRA Config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 2. Load the base model and create the "master" PEFT model
    print("Loading base model to create master PEFT model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        # device_map="cuda:0",
        device_map="auto",
        trust_remote_code=True
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    # 3. Get shapes and devices of trainable parameters
    params_dict = {}
    grads_dict = {}
    base_names = []
    for name, param in peft_model.named_parameters():
        params_dict[name] = param
        if name.endswith(".base_layer.weight"):
            base_names.append(name.split(".base_layer.weight")[0])
        if ".lora_" in name:
            grads_dict[name] = torch.zeros_like(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params_info = []
    param_devices = set()
    for name, param in peft_model.named_parameters():
        if "layers.0" in name:
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}, Shape: {param.shape}, Device: {param.device}")
        if param.requires_grad:
            trainable_params_info.append((name, param.shape, param.device))
            param_devices.add(param.device)
            
    print(f"\nFound {len(trainable_params_info)} trainable parameter tensors.")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"PEFT parameter devices: {param_devices}\n")

    # 4. Initialize Adam optimizer
    optimizer = optim.Adam(
        [param for name, param in peft_model.named_parameters() if param.requires_grad],
        lr=args.learning_rate
    )

    # 5. Define prompt and sampling params
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

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    
    sampling_params_with_lora = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=stop_tokens
    )

    # 6. Initialize VLLM with LoRA enabled
    print(f"Loading base model {args.model} into VLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2 if args.num_gpus % 2 == 0 else 1,
        pipeline_parallel_size=args.num_gpus // 2 if args.num_gpus % 2 == 0 else args.num_gpus,
        enable_lora=True,
        trust_remote_code=True,
        max_loras=args.population_size,
        max_lora_rank=max(args.lora_r, 8),
    )
    
    # 7. Define the BASE LoRA requests
    adapter_paths = []
    for pop_idx in range(args.population_size):
        adapter_path = os.path.join(ADAPTER_POPULATION_PATH, f"adapter_{pop_idx}")
        adapter_paths.append(adapter_path)
    
    # --- Start ES Training Loop ---
    print("\n--- Starting ES Training Loop ---")
    for es_step in range(args.num_steps):
        pop_step = es_step // args.steps_per_adapter
        print(f"\n======= ES Step {es_step} / {args.num_steps}, Population {pop_step} =======")

        # --- 1. Create Population of Noisy Adapters ---
        with torch.no_grad():
            if es_step % args.steps_per_adapter == 0:
                print(f"Creating {args.population_size} noisy adapters from master model...")
                start_time = time.time()
                master_state_dict = copy.deepcopy(peft_model.state_dict())
            
                for pop_idx in range(args.population_size):
                    peft_model.load_state_dict(master_state_dict)
                    # Create unique LoRA name and path for this adapter at this step
                    for layer_idx, base_name in enumerate(base_names):
                        lora_a_name = f"{base_name}.lora_A.default.weight"
                        lora_b_name = f"{base_name}.lora_B.default.weight"
                        lora_a = params_dict[lora_a_name]
                        lora_b = params_dict[lora_b_name]
                        noise_a, noise_b = get_rng_noise(
                            base_seed=args.seed,
                            num_pop_pairs=args.population_size//2,
                            pop_pair_idx=pop_idx//2,
                            num_layers=len(base_names),
                            layer_idx=layer_idx,
                            step=pop_step,
                            shapes=[lora_a.shape, lora_b.shape],
                            devices=[lora_a.device, lora_b.device],
                        )
                        noise_b *= math.sqrt(args.sigma)
                        noise_a *= math.sqrt(args.sigma)
                        lora_a.add_(noise_a)
                        if pop_idx % 2 == 1:
                            lora_b.add_(-noise_b)
                        else:
                            lora_b.add_(noise_b)

                    adapter_path = adapter_paths[pop_idx]
                    peft_model.save_pretrained(adapter_path)
                
                peft_model.load_state_dict(master_state_dict)
                print(f"Population adapter creation time: {time.time() - start_time:.2f} seconds")
                print(f"--- Population generated and saved ---")

        # --- 2. Evaluate Population ---
        print(f"Evaluating {args.population_size} adapters with vLLM...")
        start_time = time.time()

        batch_prompts, batch_answers = task.get_batch()
        all_prompts = []
        all_lora_requests = []
        for pop_idx, adapter_path in enumerate(adapter_paths):
            lora_name = f"popstep_{pop_step}_adapter_{pop_idx}"
            lora_int_id = (pop_step * args.population_size) + pop_idx + 1
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=adapter_path,
            )
            for prompt in batch_prompts:
                all_prompts.append(prompt)
                all_lora_requests.append(lora_request)
        
        outputs_batch = llm.generate(
            prompts=all_prompts,
            sampling_params=sampling_params_with_lora,
            lora_request=all_lora_requests,
        )

        outputs_with_lora = []
        for i in range(0, len(outputs_batch), args.prompt_batch_size):
            outputs_with_lora.append(outputs_batch[i : i + args.prompt_batch_size])

        # --- 3. Get Fitness Scores ---
        fitness_scores = []
        for pop_idx, output_group in enumerate(outputs_with_lora):
            responses = [output_group[i].outputs[0].text for i in range(len(output_group))]
            texts = [batch_prompts[i] + responses[i] for i in range(len(output_group))]

            fitnesses = task.get_fitnesses(responses, batch_answers)
            fitness = np.mean(fitnesses)
            fitness_scores.append(fitness)

            if pop_idx < 4 or pop_idx >= args.population_size - 4:
                print(f"  Pop {pop_idx}: {texts}, Answers: {batch_answers}, Fitnesses: {[f'{f:.4f}' for f in fitnesses]}, Mean Fitness: {fitness:.4f}")
        print(f"Inference time: {time.time() - start_time:.2f} seconds")

        # --- 4. Update Master Model (The ES Gradient Step) ---
        print("--- Updating master model parameters ---")
        start_time = time.time()

        # 4.1. Standardize (z-score) fitnesses
        fitnesses = np.array(fitness_scores)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses) + 1e-8
        normalized_fitnesses = (fitnesses - mean_fitness) / std_fitness
        fitnesses_log.append(mean_fitness)

        max_fitness = np.max(fitnesses)
        print(f"Step {es_step} Fitness: Mean={mean_fitness:.4f}, Std={std_fitness:.4f}, Max={max_fitness:.4f}")
        print(f"All fitnesses: {[f'{f:.4f}' for f in fitnesses]}")

        # --- Log metrics to WandB ---
        if args.use_wandb:
            wandb.log({
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "max_fitness": max_fitness,
                "es_step": es_step,
                "pop_step": pop_step,
            })

        # 4.2. Zero grads
        grads_dict = {name: torch.zeros_like(x) for name, x in grads_dict.items()}

        # 4.3. Reconstruct noise and calculate fitness-weighted sum
        for pop_idx in range(args.population_size):
            if pop_idx % 2 == 0:
                for layer_idx, base_name in enumerate(base_names):
                    full_base_name = f"{base_name}.base_layer.weight"
                    lora_a_name = f"{base_name}.lora_A.default.weight"
                    lora_b_name = f"{base_name}.lora_B.default.weight"
                    lora_a = params_dict[lora_a_name]
                    lora_b = params_dict[lora_b_name]
                    noise_a, noise_b = get_rng_noise(
                        base_seed=args.seed,
                        num_pop_pairs=args.population_size//2,
                        pop_pair_idx=pop_idx//2,
                        num_layers=len(base_names),
                        layer_idx=layer_idx,
                        step=pop_step,
                        shapes=[lora_a.shape, lora_b.shape],
                        devices=[lora_a.device, lora_b.device],
                    )
                    noise_b *= math.sqrt(args.sigma)
                    noise_a *= math.sqrt(args.sigma)
                    fitness1 = normalized_fitnesses[pop_idx]
                    fitness2 = normalized_fitnesses[pop_idx+1]
                    grads_dict[lora_a_name].add_(noise_a.to(grads_dict[lora_a_name].device) * (fitness1 + fitness2))
                    grads_dict[lora_b_name].add_(-noise_b.to(grads_dict[lora_b_name].device) * (fitness1 - fitness2))

        # 4.4. Apply the gradient to the master model via the optimizer
        # The ES gradient estimate is: (1 / (N * sigma)) * sum(F_i * E_i)
        # We are *maximizing* fitness, so optimizer should *ascend* the gradient.
        # Adam *minimizes*, so we feed it the *negative* gradient.
        optimizer.zero_grad()
        
        for i, (name, grad) in enumerate(grads_dict.items()):
            param = params_dict[name]
            if i == 0:
                print(f"----{name}: {param.data.flatten()[:5]}, {grad.flatten()[:5]}")
            gradient = (1.0 / (args.population_size * args.sigma)) * grad
            param.grad = -gradient
            
        optimizer.step()
        
        print(f"Model update time: {time.time() - start_time:.2f} seconds")
        print("--- Master model parameters updated ---")
        print(f"\nLogged mean fitnesses so far: {[f'{f:.4f}' for f in fitnesses_log]}")

    print("\n--- ES Training Complete ---")
    # Clean up temporary population adapters
    print(f"Cleaning up {ADAPTER_POPULATION_PATH}...")
    try:
        shutil.rmtree(ADAPTER_POPULATION_PATH)
    except Exception as e:
        print(f"Warning: Could not clean up adapter path: {e}")
    
    ray.shutdown()
    wandb.finish()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: This script requires a GPU and CUDA to run effectively.")
        print("VLLM may not work correctly on CPU-only machines.")
    args = tyro.cli(Args)
    main(args)