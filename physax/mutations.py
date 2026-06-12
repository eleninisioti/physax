import jax.numpy as jnp
from jax import random
from physax.constants import *

def apply_divide_mutations(key, child_tape, child_tape_len, cfg):
    """Apply divide mutations to child tape after successful divide.
    Order: point mutation, insertion, deletion (matching CellGeneticCodeTape.divide()).
    """
    k1, k2, k3, k4, k5, k6, k7, k8 = random.split(key, 8)

    # Point mutation (rate = divide_mutation_rate, typically 0.0)
    do_point = random.uniform(k1) < cfg.divide_mutation_rate
    point_pos = random.randint(k2, (), 0, jnp.maximum(child_tape_len, 1))
    point_val = random.randint(k3, (), 0, UP_IS_SIZE).astype(jnp.int32)
    child_tape = jnp.where(
        do_point & (point_pos < child_tape_len),
        child_tape.at[point_pos].set(point_val),
        child_tape
    )

    # Insertion (rate = divide_insert_rate)
    do_insert = random.uniform(k4) < cfg.divide_insert_rate
    insert_pos = random.randint(k5, (), 0, jnp.maximum(child_tape_len + 1, 1))
    insert_val = random.randint(k3, (), 0, UP_IS_SIZE).astype(jnp.int32)
    can_insert = do_insert & (child_tape_len < cfg.max_genome_len - 1)
    # Shift right from insert_pos
    indices = jnp.arange(cfg.max_genome_len)
    shifted_right = jnp.where(
        indices > insert_pos,
        jnp.where(indices - 1 < cfg.max_genome_len, child_tape[jnp.clip(indices - 1, 0, cfg.max_genome_len - 1)], BLANK),
        child_tape[indices]
    )
    shifted_right = shifted_right.at[insert_pos].set(insert_val)
    child_tape = jnp.where(can_insert, shifted_right, child_tape)
    child_tape_len = jnp.where(can_insert, child_tape_len + 1, child_tape_len)

    # Deletion (rate = divide_delete_rate)
    do_delete = random.uniform(k6) < cfg.divide_delete_rate
    can_delete = do_delete & (child_tape_len > 1)
    delete_pos = random.randint(k7, (), 0, jnp.maximum(child_tape_len - 1, 1))
    # Shift left from delete_pos
    shifted_left = jnp.where(
        indices >= delete_pos,
        jnp.where(indices + 1 < cfg.max_genome_len, child_tape[jnp.clip(indices + 1, 0, cfg.max_genome_len - 1)], BLANK),
        child_tape[indices]
    )
    child_tape = jnp.where(can_delete, shifted_left, child_tape)
    child_tape_len = jnp.where(can_delete, child_tape_len - 1, child_tape_len)

    return child_tape, child_tape_len


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


