"""
Debug script to examine LoRA parameter names from PEFT and vLLM.
This script loads a model with PEFT LoRA and prints the parameter names
to understand the correct naming convention.
"""
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Model to test
MODEL_NAME = "Qwen/Qwen2-0.5B"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

print("=" * 80)
print("Loading model and creating PEFT LoRA adapter...")
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

print("\n" + "=" * 80)
print("PEFT Model Parameter Names (first 20):")
print("=" * 80)
for i, (name, param) in enumerate(peft_model.named_parameters()):
    if i < 20:
        print(f"{i:3d}. {name:80s} {param.shape}")

print("\n" + "=" * 80)
print("PEFT State Dict Keys (first 30):")
print("=" * 80)
state_dict = peft_model.state_dict()
for i, key in enumerate(list(state_dict.keys())[:30]):
    print(f"{i:3d}. {key}")

print("\n" + "=" * 80)
print("Looking for base_layer.weight parameters:")
print("=" * 80)
base_layer_params = [name for name in peft_model.named_parameters() if name[0].endswith(".base_layer.weight")]
for name, param in base_layer_params[:5]:
    print(f"  {name}")
    # Derive the LoRA A and B names
    lora_a_name = name.replace("base_layer.weight", "lora_A.default.weight")
    lora_b_name = name.replace("base_layer.weight", "lora_B.default.weight")
    print(f"    -> LoRA A: {lora_a_name}")
    print(f"    -> LoRA B: {lora_b_name}")
    print(f"    -> A exists in state_dict: {lora_a_name in state_dict}")
    print(f"    -> B exists in state_dict: {lora_b_name in state_dict}")
    print()

print("\n" + "=" * 80)
print("Analyzing key structure patterns:")
print("=" * 80)

# Find one example of each LoRA component
lora_a_example = [k for k in state_dict.keys() if "lora_A" in k][0]
lora_b_example = [k for k in state_dict.keys() if "lora_B" in k][0]
base_layer_example = [k for k in state_dict.keys() if "base_layer.weight" in k][0]

print(f"Example LoRA A key: {lora_a_example}")
print(f"Example LoRA B key: {lora_b_example}")
print(f"Example base_layer key: {base_layer_example}")

print("\n" + "=" * 80)
print("Checking what vLLM expects:")
print("=" * 80)
print("According to vLLM's parse_fine_tuned_lora_name() function,")
print("the expected pattern should match one of these:")
print("  - base_model.model.<layer>.<module>  (for merged models)")
print("  - model.<layer>.<module>  (for direct models)")
print("\nThe issue is that PEFT uses 'base_model.model.model.*' but vLLM expects 'base_model.model.*'")

print("\n" + "=" * 80)
print("Testing key transformations:")
print("=" * 80)

test_key = lora_a_example
print(f"Original PEFT key: {test_key}")

# Test different transformations
transform1 = test_key.replace("base_model.model.model.", "base_model.model.")
print(f"Transform 1 (remove one 'model.'): {transform1}")

transform2 = test_key.replace("base_model.model.", "")
print(f"Transform 2 (remove 'base_model.model.'): {transform2}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("The keys in PEFT have the pattern:")
print("  base_model.model.model.layers.X.module.lora_A.default.weight")
print("\nFor vLLM to accept them, they should be:")
print("  base_model.model.layers.X.module.lora_A.default.weight")
print("\nSo we need to replace 'base_model.model.model.' with 'base_model.model.'")
print("(i.e., remove ONE occurrence of '.model')")
