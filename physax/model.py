import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from functools import partial
from tqdm import trange
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from physax.config import *
from physax.agent import create_organism_state, parse_genome, apply_divide_mutations, init_organism, create_ancestor_genome
from physax.virtual_machine import organism_update


def cycle_step(cfg, pop, key):
    """Execute one cycle: step all organisms, handle births."""

    k_exec, k_birth, k_place = random.split(key, 3)

    # 1. Execute all alive organisms (steps_per_update steps each)
    exec_keys = random.split(k_exec, cfg.pop_size)
    pop = jax.vmap(lambda state, k: organism_update(state, k, cfg))(pop, exec_keys)

    # 2. Age all alive organisms
    pop['age'] = jnp.where(pop['alive'], pop['age'] + 1, pop['age'])

    # 3. Handle births
    has_child = pop['has_child']
    n_births = jnp.sum(has_child)

    # Prepare child tapes for organisms that divided
    # Copy child + child_len into child_tape + child_tape_len
    pop['child_tape'] = jnp.where(
        has_child[:, None],
        pop['child'],
        pop['child_tape']
    )
    pop['child_tape_len'] = jnp.where(
        has_child,
        pop['child_len'],
        pop['child_tape_len']
    )

    # Apply divide mutations
    mut_keys = random.split(k_birth, cfg.pop_size)

    def mutate_color(key, color):
        """Apply small HSV drift to color for lineage tracking."""
        k1, k2 = random.split(key)
        noise_h = random.uniform(k1, (1,), minval=-0.05, maxval=0.05)
        noise_sv = random.uniform(k2, (2,), minval=-0.02, maxval=0.02)
        noise = jnp.concatenate([noise_h, noise_sv])
        h = (color[0] + noise[0]) % 1.0
        s = jnp.clip(color[1] + noise[1], 0.0, 1.0)
        v = jnp.clip(color[2] + noise[2], 0.0, 1.0)
        return jnp.array([h, s, v])

    def mutate_one(mut_key, tape, tape_len, has, parent_color):
        new_tape, new_len = apply_divide_mutations(mut_key, tape, tape_len, cfg)
        new_color = mutate_color(mut_key, parent_color)
        tape_out = jnp.where(has, new_tape, tape)
        len_out = jnp.where(has, new_len, tape_len)
        color_out = jnp.where(has, new_color, parent_color)
        return tape_out, len_out, color_out

    mutated_tapes, mutated_lens, child_colors = jax.vmap(mutate_one)(
        mut_keys, pop['child_tape'], pop['child_tape_len'], has_child, pop['color']
    )

    # Spatial reproduction on 2D toroidal grid (OldestNurse)
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    W = grid_side
    H = (cfg.pop_size + W - 1) // W

    parent_indices = jnp.arange(cfg.pop_size)
    y = parent_indices // W
    x = parent_indices % W

    dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1])

    ny = (y[:, None] + dy[None, :]) % H
    nx = (x[:, None] + dx[None, :]) % W

    neighbor_indices = ny * W + nx
    neighbor_valid = neighbor_indices < cfg.pop_size
    neighbor_indices_safe = jnp.where(neighbor_valid, neighbor_indices, parent_indices[:, None])

    neighbor_alive = pop['alive'][neighbor_indices_safe]
    neighbor_age = pop['age'][neighbor_indices_safe]

    # Score: empty cells get very high score, alive cells get their age
    # SS: need to reduce default base_score from 1e9 so as not to lose precision
    #base_score = jnp.where(neighbor_alive, neighbor_age.astype(jnp.float32), 1e9)
    base_score = jnp.where(neighbor_alive, neighbor_age.astype(jnp.float32), 1e5)
    valid_score = jnp.where(neighbor_valid, base_score, -1.0)

    # Break ties randomly
    noise = random.uniform(k_place, (cfg.pop_size, 8)) * 0.5
    final_score = valid_score + noise

    best_neighbor_local_idx = jnp.argmax(final_score, axis=1)
    target_indices = jnp.take_along_axis(
        neighbor_indices_safe, best_neighbor_local_idx[:, None], axis=1
    ).squeeze(1)

    # Parse child genomes and build child states
    child_parsed = jax.vmap(lambda g, l: parse_genome(g, l, cfg))(mutated_tapes, mutated_lens)

    def build_child_state(genome, genome_len, color, parsed):
        state = create_organism_state(cfg)
        state['genome'] = genome
        state['genome_len'] = genome_len
        state['color'] = color
        state['alive'] = jnp.bool_(True)
        state['n_ses'] = parsed['n_ses']
        state['separator_pos'] = parsed['separator_pos']
        state['n_instructions'] = parsed['n_instructions']
        state['instruction_table'] = parsed['instruction_table']
        state['instruction_lengths'] = parsed['instruction_lengths']
        # IP starts right after separator
        ip_start = (parsed['separator_pos'] + 1) % jnp.maximum(genome_len, 1)
        state['se_values'] = state['se_values'].at[0].set(ip_start)
        return state

    child_states = jax.vmap(build_child_state)(
        mutated_tapes, mutated_lens, child_colors, child_parsed
    )

    # Place children using gather approach (avoids scatter conflicts with duplicates)
    # For each cell j, check if any birthing parent targets it
    all_cells = jnp.arange(cfg.pop_size)
    # targets_match[j, k] = True if parent k targets cell j AND has_child[k]
    targets_match = (target_indices[None, :] == all_cells[:, None]) & has_child[None, :]
    # any_child_placed[j] = True if any parent places a child at cell j
    any_child_placed = jnp.any(targets_match, axis=1)
    # source_parent[j] = index of (first) parent placing a child at cell j
    source_parent = jnp.argmax(targets_match.astype(jnp.int32), axis=1)

    # Gather child states from source parents
    gathered_children = jax.tree.map(lambda x: x[source_parent], child_states)

    # Conditionally replace: cell j gets child state if a child was placed, else keeps current
    pop = jax.tree.map(
        lambda c, p: jnp.where(
            any_child_placed.reshape((-1,) + (1,) * (c.ndim - 1)),
            c, p
        ),
        gathered_children, pop
    )

    # Post-reproduction cleanup: reset parent's child state
    blank_child = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
    pop['child'] = jnp.where(has_child[:, None], blank_child, pop['child'])
    pop['child_len'] = jnp.where(has_child, jnp.int32(0), pop['child_len'])
    pop['child_copied'] = jnp.where(
        has_child[:, None],
        jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_),
        pop['child_copied']
    )
    pop['already_allocated'] = jnp.where(has_child, jnp.bool_(False), pop['already_allocated'])
    pop['has_child'] = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    pop['child_tape'] = jnp.where(
        has_child[:, None],
        jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
        pop['child_tape']
    )
    pop['child_tape_len'] = jnp.where(has_child, jnp.int32(0), pop['child_tape_len'])

    # Stats
    alive_count = jnp.sum(pop['alive'])
    ## SS: calculate percentiles rather than just the avg/min
    #avg_genome_len = jnp.sum(jnp.where(pop['alive'], pop['genome_len'], 0)) / jnp.maximum(alive_count, 1)
    #min_genome_len = jnp.min(jnp.where(pop['alive'], pop['genome_len'], cfg.max_genome_len))
    q_genome_len = jnp.nanpercentile(jnp.where(pop['alive'], pop['genome_len'], jnp.nan), PERCENTILES )

    stats = {
        'pop_size': alive_count,
        'births': n_births,
        # SS: store percentiles (was quartiles) not avgs
        #'avg_genome_len': avg_genome_len,
        'q_genome_len': q_genome_len,
    }

    return pop, stats


def init_population(key, cfg):
    """Initialize population with ancestor genomes at random positions."""
    ancestor_genome, ancestor_len = create_ancestor_genome(cfg)

    k1, k2 = random.split(key)
    perm = random.permutation(k1, cfg.pop_size)
    alive_indices = perm[:cfg.initial_pop]

    is_alive = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    is_alive = is_alive.at[alive_indices].set(True)

    # Random HSV colors
    k_h, k_s, k_v = random.split(k2, 3)
    h = random.uniform(k_h, (cfg.pop_size, 1))
    s = random.uniform(k_s, (cfg.pop_size, 1), minval=0.7, maxval=1.0)
    v = random.uniform(k_v, (cfg.pop_size, 1), minval=0.8, maxval=1.0)
    colors = jnp.concatenate([h, s, v], axis=-1)

    def init_one(i, alive_mask_val, col):
        state = init_organism(ancestor_genome, ancestor_len, col, cfg)
        state['alive'] = alive_mask_val
        return state

    pop = jax.vmap(init_one)(jnp.arange(cfg.pop_size), is_alive, colors)
    return pop


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
