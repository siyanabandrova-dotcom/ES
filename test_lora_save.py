"""
Test script to replicate the LoRA adapter generation logic and verify key names.
"""
import os
import json
import copy
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file, load_file

# Model to test
MODEL_NAME = "Qwen/Qwen2-0.5B"
OUTPUT_DIR = "/tmp/test_lora_adapter"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

print("=" * 80)
print("Step 1: Create PEFT model and extract state dict")
print("=" * 80)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

# Create LoRA config
lora_config = LoraConfig(
    r=4,
    lora_alpha=1,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create PEFT model
peft_model = get_peft_model(base_model, lora_config)
peft_state_dict = copy.deepcopy(peft_model.state_dict())
peft_shapes_dict = {
    name: param.shape
    for name, param in peft_model.named_parameters()
    if name.endswith(".base_layer.weight")
}

print(f"Number of base_layer.weight params: {len(peft_shapes_dict)}")
print("\nFirst 3 base_layer.weight params:")
for i, (name, shape) in enumerate(list(peft_shapes_dict.items())[:3]):
    print(f"  {name}")

print("\n" + "=" * 80)
print("Step 2: Simulate the generate_local_adapters logic")
print("=" * 80)

# Prepare config for saving
lora_config_dict = lora_config.to_dict()
if "target_modules" in lora_config_dict and isinstance(lora_config_dict["target_modules"], (set, tuple)):
    lora_config_dict["target_modules"] = list(lora_config_dict["target_modules"])

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
adapter_path = os.path.join(OUTPUT_DIR, "adapter_test")
os.makedirs(adapter_path, exist_ok=True)

# Save config
with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
    json.dump(lora_config_dict, f)

# Generate weights (simulating the worker logic)
local_state_dict = {}
for layer_idx, (peft_name, weight_shape) in enumerate(peft_shapes_dict.items()):
    # Generate LoRA A and B names from the base_layer.weight name
    lora_a_name_raw = peft_name.replace("base_layer.weight", "lora_A.default.weight")
    lora_b_name_raw = peft_name.replace("base_layer.weight", "lora_B.default.weight")

    # Sanitize keys for vLLM:
    # 1. PEFT uses "base_model.model.model.*" but vLLM expects "base_model.model.*"
    # 2. PEFT uses ".lora_A.default.weight" but vLLM expects ".lora_A.weight"
    lora_a_name = lora_a_name_raw.replace("base_model.model.model.", "base_model.model.")
    lora_a_name = lora_a_name.replace(".lora_A.default.weight", ".lora_A.weight")
    lora_b_name = lora_b_name_raw.replace("base_model.model.model.", "base_model.model.")
    lora_b_name = lora_b_name.replace(".lora_B.default.weight", ".lora_B.weight")

    # Get base (initial) weights
    print(f"\nLayer {layer_idx}: {peft_name}")
    print(f"  Raw LoRA A: {lora_a_name_raw}")
    print(f"  Raw LoRA B: {lora_b_name_raw}")
    print(f"  Sanitized LoRA A: {lora_a_name}")
    print(f"  Sanitized LoRA B: {lora_b_name}")
    print(f"  A exists in peft_state_dict: {lora_a_name_raw in peft_state_dict}")
    print(f"  B exists in peft_state_dict: {lora_b_name_raw in peft_state_dict}")

    if lora_a_name_raw not in peft_state_dict:
        print(f"  ERROR: {lora_a_name_raw} not found in state dict!")
        break

    lora_a = peft_state_dict[lora_a_name_raw].clone().cpu()
    lora_b = peft_state_dict[lora_b_name_raw].clone().cpu()

    # Save with sanitized names
    local_state_dict[lora_a_name] = lora_a
    local_state_dict[lora_b_name] = lora_b

    if layer_idx >= 2:  # Just show first 3
        break

print("\n" + "=" * 80)
print("Step 3: Save and verify the adapter")
print("=" * 80)

# Save tensors
save_file(local_state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))

print(f"Saved adapter to: {adapter_path}")
print(f"Number of tensors saved: {len(local_state_dict)}")
print("\nSaved tensor keys (first 10):")
for i, key in enumerate(list(local_state_dict.keys())[:10]):
    print(f"  {i}: {key}")

# Load back and verify
loaded_state_dict = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
print(f"\nLoaded {len(loaded_state_dict)} tensors from safetensors")
print("Loaded tensor keys match saved keys:", set(loaded_state_dict.keys()) == set(local_state_dict.keys()))

print("\n" + "=" * 80)
print("Step 4: Check if vLLM would accept these keys")
print("=" * 80)

# Simulate vLLM's parse_fine_tuned_lora_name check
def check_vllm_compatible(name):
    """
    Simplified version of vLLM's parse_fine_tuned_lora_name logic.
    The key should match the pattern expected by vLLM.
    """
    if not (name.startswith("base_model.model.") or name.startswith("model.")):
        return False, "Does not start with 'base_model.model.' or 'model.'"

    # Check for the problematic pattern: should NOT have "base_model.model.model."
    if "base_model.model.model." in name:
        return False, "Contains 'base_model.model.model.' (should be 'base_model.model.')"

    # vLLM expects the pattern to end with .lora_A.weight or .lora_B.weight
    # NOT .lora_A.default.weight or .lora_B.default.weight
    parts = name.split(".")
    if parts[-1] == "weight" and (parts[-2] == "lora_A" or parts[-2] == "lora_B"):
        return True, "OK"

    if ".default.weight" in name:
        return False, "Contains '.default.weight' (should be just '.weight')"

    return False, "Does not match expected pattern (should end with .lora_A.weight or .lora_B.weight)"

print("Checking all saved keys against vLLM compatibility:")
all_compatible = True
for key in local_state_dict.keys():
    compatible, reason = check_vllm_compatible(key)
    if not compatible:
        print(f"  ❌ {key}")
        print(f"     Reason: {reason}")
        all_compatible = False

if all_compatible:
    print("  ✓ All keys are vLLM compatible!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if all_compatible:
    print("✓ The adapter generation logic is correct!")
    print("  The keys have been properly sanitized for vLLM.")
else:
    print("✗ The adapter generation logic has issues.")
    print("  Some keys are not properly sanitized for vLLM.")
