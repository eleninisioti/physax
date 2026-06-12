import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from functools import partial
import numpy as np
from tqdm import trange
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from physax.population import init_population, cycle_step

def run_simulation(key, cfg, total_cycles, log_interval=10000, use_wandb=False):
    """Run the simulation for total_cycles."""
    print(f"=== JAX PHYSIS SIMULATION ===")
    print(f"Population capacity: {cfg.pop_size}, Initial: {cfg.initial_pop}")
    print(f"Steps per update: {cfg.steps_per_update}")
    print(f"Total cycles: {total_cycles}, Log interval: {log_interval}")
    print()

    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed. Run: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project="physis-jax",
                config={
                    "total_cycles": total_cycles,
                    "pop_size": cfg.pop_size,
                    "initial_pop": cfg.initial_pop,
                    "max_genome_len": cfg.max_genome_len,
                    "steps_per_update": cfg.steps_per_update,
                    "copy_mutation_rate": cfg.copy_mutation_rate,
                    "divide_insert_rate": cfg.divide_insert_rate,
                    "divide_delete_rate": cfg.divide_delete_rate,
                }
            )

    k1, k2 = random.split(key)
    pop = init_population(k1, cfg)

    cycle_step_fn = partial(cycle_step, cfg)

    def scan_cycles(pop, keys):
        def step(pop, key):
            pop, stats = cycle_step_fn(pop, key)
            return pop, stats
        return lax.scan(step, pop, keys)

    jit_scan = jax.jit(scan_cycles)

    n_chunks = total_cycles // log_interval
    all_stats = []
    cycle_keys = random.split(k2, total_cycles)

    try:
        for chunk in trange(n_chunks, desc="Running"):
            start = chunk * log_interval
            end = (chunk + 1) * log_interval
            chunk_keys = cycle_keys[start:end]

            pop, stats = jit_scan(pop, chunk_keys)
            pop = jax.block_until_ready(pop)

            cycle_num = end
            pop_size = int(stats['pop_size'][-1])
            births = int(jnp.sum(stats['births']))
            #avg_len = float(stats['avg_genome_len'][-1])
            q_len = stats['q_genome_len'][-1]

            # SS: print percentiles, not avg
            #print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, AvgLen={avg_len:.1f}")
            print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, percentiles={q_len}")

            if use_wandb:
                wandb.log({
                    "cycle": cycle_num,
                    "population/size": pop_size,
                    "population/births_interval": births,
                    # SS: use median from percentiles: INDEX MAY CHANGE IF DIFFERENT PERCENTILES USED
                    #"genome/avg_len": avg_len,
                    "genome/avg_len": q_len[3],
                })

            snapshot = {
                'cycle': cycle_num,
                'alive': np.array(pop['alive']),
                'genome_len': np.array(pop['genome_len']),
                'color': np.array(pop['color'])
            }

            chunk_rec = {
                'cycle': cycle_num,
                'pop_size': pop_size,
                'births': births,
                # SS: record percentiles, not avg
                #'avg_len': avg_len,
                'q_len': q_len,
                'snapshot': snapshot
            }

            all_stats.append(chunk_rec)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    if use_wandb:
        wandb.finish()

    return pop, all_stats


