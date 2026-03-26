#!/usr/bin/env python3
"""
Expands a template experiments config CSV into a flat config ready for slurm_launch_from_config.sh.
"""

import re
import sys
import math
from itertools import product as iproduct

# Added max_tokens at index 2
COLS = ["sigma", "learning_rate", "max_tokens", "model_name", "population_size",
        "prompt_batch_size", "name_prefix", "normalize_with_std",
        "scale_lr_in_grad", "num_nodes", "gpus_per_node", "task"]

def parse_field(field):
    field = field.strip()
    m = re.match(r'^\[(.+)\]$', field)
    if m:
        return [v.strip() for v in m.group(1).split('|')]
    return [field]

def auto_nodes_gpus(model, pop):
    pop = int(pop)
    model_lower = model.lower()
    if "110b" in model_lower:
        tp = 4
        loras_per_engine = 64
    elif "72b" in model_lower:
        tp = 4
        loras_per_engine = 128
    elif "32b" in model_lower:
        tp = 4   
        loras_per_engine = 256
    elif "14b" in model_lower:
        tp = 2
        loras_per_engine = 256
    else:
        tp = 1   
        loras_per_engine = 256

    num_engines = max(1, pop // loras_per_engine)
    total_gpus = num_engines * tp

    if total_gpus <= 1:
        return "1", "1"
    elif total_gpus <= 2:
        return "1", "2"
    elif total_gpus <= 4:
        return "1", "4"
    else:
        nodes = math.ceil(total_gpus / 4)
        return str(nodes), "4"

def expand_line(line):
    fields = line.split(',')
    if len(fields) != len(COLS):
        raise ValueError(f"Expected {len(COLS)} columns, got {len(fields)}: {line!r}")
    options = [parse_field(f) for f in fields]
    rows = []
    for combo in iproduct(*options):
        combo = list(combo)
        # Resolved indices: model=3, pop=4, num_nodes=9, gpus_per_node=10
        if combo[9] == 'auto' or combo[10] == 'auto':
            nodes, gpus = auto_nodes_gpus(combo[3], combo[4])
            if combo[9] == 'auto':
                combo[9] = nodes
            if combo[10] == 'auto':
                combo[10] = gpus
        rows.append(','.join(combo))
    return rows

def expand_file(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    out = []
    total_expanded = 0
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.startswith('#') or stripped.strip() == '':
            out.append(stripped)
            continue
        expanded = expand_line(stripped)
        out.extend(expanded)
        if len(expanded) > 1:
            total_expanded += len(expanded)

    result = '\n'.join(out) + '\n'

    with open(output_path, 'w') as f:
        f.write(result)

    print(f"Expanded {input_path} → {output_path} ({total_expanded} rows generated)", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('_template', '_config')
    expand_file(input_path, output_path)