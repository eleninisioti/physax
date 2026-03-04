#!/usr/bin/env python3
"""
Generate an animated GIF of a single Physax run on a 16x16 grid.

Usage:
    source .venv/bin/activate
    python3 make_gif.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from tqdm import trange

from physax import make_config, init_population, cycle_step, save_grid_gif

# ── Parameters ────────────────────────────────────────────────────────
GRID_SIDE = 16
POP_SIZE = GRID_SIDE * GRID_SIDE  # 256
N_CYCLES = 1000
SNAPSHOT_INTERVAL = 10  # capture every 10 cycles -> 100 frames
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "experiment_results", "physax_run.gif")

def main():
    print(f"=== Physax GIF Generator ===")
    print(f"Grid: {GRID_SIDE}x{GRID_SIDE} = {POP_SIZE} cells")
    print(f"Cycles: {N_CYCLES}, Snapshot every {SNAPSHOT_INTERVAL} cycles")
    print(f"Seed: {SEED}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    cfg = make_config(
        pop_size=POP_SIZE,
        initial_pop=1,
        use_species_color=True,
    )

    key = random.PRNGKey(SEED)
    k1, k2 = random.split(key)

    print("Initializing population...")
    pop = init_population(k1, cfg)

    cycle_step_fn = partial(cycle_step, cfg)

    # JIT-compile cycle_step
    print("JIT compiling (first cycle)...")
    k_first, k2 = random.split(k2)
    pop, _ = jax.jit(cycle_step_fn)(pop, k_first)
    jax.block_until_ready(pop)
    print("JIT compiled.")
    print()

    # Run simulation, capturing snapshots
    snapshots = []
    cycle_keys = random.split(k2, N_CYCLES)

    # Capture initial state (cycle 0)
    snapshots.append({
        'cycle': 0,
        'alive': np.array(pop['alive']),
        'genome_len': np.array(pop['genome_len']),
        'color': np.array(pop['color']),
    })

    jit_step = jax.jit(cycle_step_fn)

    for i in trange(N_CYCLES, desc="Simulating"):
        pop, stats = jit_step(pop, cycle_keys[i])

        cycle_num = i + 1
        if cycle_num % SNAPSHOT_INTERVAL == 0:
            # Block until ready before transferring to host
            pop_ready = jax.block_until_ready(pop)
            snapshots.append({
                'cycle': cycle_num,
                'alive': np.array(pop_ready['alive']),
                'genome_len': np.array(pop_ready['genome_len']),
                'color': np.array(pop_ready['color']),
            })

    print(f"\nCaptured {len(snapshots)} snapshots (cycle 0 + {len(snapshots)-1} during simulation)")
    pop_final = jax.block_until_ready(pop)
    alive_count = int(np.sum(np.array(pop_final['alive'])))
    print(f"Final population: {alive_count}/{POP_SIZE}")
    print()

    # Generate GIF
    save_grid_gif(snapshots, OUTPUT_PATH, cfg)
    print("Done!")


if __name__ == "__main__":
    main()
