"""
Debug script to verify noise generation matches between async2 and async6
"""
import torch
import math

def get_rng_noise(
        base_seed: int,
        num_pop_pairs: int,
        pop_pair_idx: int,
        num_layers: int,
        layer_idx: int,
        step: int,
        shapes: list,
        ):
    """From both async2 and async6 - should be identical"""
    id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    torch_rng = torch.Generator().manual_seed(id)

    noise_a, noise_b = (torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=shape,
                    generator=torch_rng,
                ) for shape in shapes)
    return noise_a, noise_b

# Test parameters
base_seed = 0
population_size = 128
lora_r = 4
sigma = 0.001
num_layers = 168  # Number of LoRA layers in Qwen2-0.5B

# Simulate first layer, first population pair, step 0
lora_a_shape = (lora_r, 896)
lora_b_shape = (896, lora_r)

print("=" * 80)
print("Testing noise generation for population pair 0, layer 0, step 0")
print("=" * 80)

# Generate noise for pop_idx 0 and 1 (first antithetic pair)
for pop_idx in [0, 1, 2, 3]:
    pop_pair_idx = pop_idx // 2
    noise_a, noise_b = get_rng_noise(
        base_seed=base_seed,
        num_pop_pairs=population_size//2,
        pop_pair_idx=pop_pair_idx,
        num_layers=num_layers,
        layer_idx=0,
        step=0,
        shapes=[lora_a_shape, lora_b_shape],
    )

    noise_b *= math.sqrt(sigma)
    noise_a *= math.sqrt(sigma)

    # Simulate async2 behavior: start from zeros
    lora_a = torch.zeros(lora_a_shape)
    lora_b = torch.zeros(lora_b_shape)

    lora_a.add_(noise_a)
    if pop_idx % 2 == 1:
        lora_b.add_(-noise_b)
    else:
        lora_b.add_(noise_b)

    # Compute effective weight: lora_B @ lora_A
    effective_weight = torch.matmul(lora_b, lora_a)

    print(f"\nPopulation {pop_idx} (pair {pop_pair_idx}, {'odd' if pop_idx % 2 == 1 else 'even'}):")
    print(f"  noise_a: mean={noise_a.mean():.6f}, std={noise_a.std():.6f}")
    print(f"  noise_b: mean={noise_b.mean():.6f}, std={noise_b.std():.6f}")
    print(f"  lora_a: mean={lora_a.mean():.6f}, std={lora_a.std():.6f}")
    print(f"  lora_b: mean={lora_b.mean():.6f}, std={lora_b.std():.6f}")
    print(f"  effective_weight: mean={effective_weight.mean():.6f}, std={effective_weight.std():.6f}")

print("\n" + "=" * 80)
print("Check antithetic property:")
print("=" * 80)

# Generate pair 0 (pop 0 and 1)
noise_a_0, noise_b_0 = get_rng_noise(0, 64, 0, 168, 0, 0, [lora_a_shape, lora_b_shape])
noise_a_1, noise_b_1 = get_rng_noise(0, 64, 0, 168, 0, 0, [lora_a_shape, lora_b_shape])

print(f"noise_a for pop 0 and pop 1 are identical: {torch.allclose(noise_a_0, noise_a_1)}")
print(f"noise_b for pop 0 and pop 1 are identical: {torch.allclose(noise_b_0, noise_b_1)}")

# The only difference should be the sign of noise_b
lora_a_0 = torch.zeros(lora_a_shape)
lora_b_0 = torch.zeros(lora_b_shape)
lora_a_0.add_(noise_a_0 * math.sqrt(sigma))
lora_b_0.add_(noise_b_0 * math.sqrt(sigma))

lora_a_1 = torch.zeros(lora_a_shape)
lora_b_1 = torch.zeros(lora_b_shape)
lora_a_1.add_(noise_a_1 * math.sqrt(sigma))
lora_b_1.add_(-noise_b_1 * math.sqrt(sigma))

eff_0 = torch.matmul(lora_b_0, lora_a_0)
eff_1 = torch.matmul(lora_b_1, lora_a_1)

print(f"Effective weights for pop 0 and pop 1 are opposite: {torch.allclose(eff_0, -eff_1)}")
print(f"  eff_0 mean: {eff_0.mean():.9f}")
print(f"  eff_1 mean: {eff_1.mean():.9f}")
print(f"  eff_0 + eff_1 mean: {(eff_0 + eff_1).mean():.9f}")
