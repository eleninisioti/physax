
import jax
import jax.numpy as jnp
from jax import random
from physax import (make_config, create_ancestor_genome, init_organism,
                    vm_execute_one, parse_genome)

def debug():
    cfg = make_config()
    genome, length = create_ancestor_genome(cfg)
    color = jnp.array([0.5, 0.8, 0.9])
    state = init_organism(genome, length, color, cfg)

    print(f"Initial State:")
    print(f"Genome Len: {state['genome_len']}")
    print(f"N SEs: {state['n_ses']}, N Instr: {state['n_instructions']}")
    print(f"Separator Pos: {state['separator_pos']}")
    print(f"IP (SE[0]): {state['se_values'][0]}")
    print(f"Genome: {state['genome'][:int(state['genome_len'])]}")

    # Print instruction table
    n_instr = int(state['n_instructions'])
    for i in range(n_instr):
        ilen = int(state['instruction_lengths'][i])
        instr = state['instruction_table'][i, :ilen]
        print(f"  Instr {i}: {instr}")

    key = random.PRNGKey(42)

    # Run for 400 steps
    for i in range(400):
        se = state['se_values']
        ip = int(se[0])
        print(f"\nStep {i}: IP={ip}, SE[0..4]={se[:5]}, "
              f"allocated={state['already_allocated']}, "
              f"child_len={state['child_len']}")

        k, key = random.split(key)
        state = vm_execute_one(state, k, cfg)

        if state['has_child']:
            print(f"!!! DIVIDE at step {i} !!!")
            print(f"Child Len: {state['child_len']}")
            n_copied = int(jnp.sum(state['child_copied'].astype(jnp.int32)))
            print(f"Child Copied Count: {n_copied}")
            print(f"Child: {state['child'][:int(state['child_len'])]}")
            break

if __name__ == "__main__":
    debug()
