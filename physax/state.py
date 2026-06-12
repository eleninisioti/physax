import jax
import jax.numpy as jnp
from physax.constants import BLANK
from physax.config import Config

def create_organism_state(cfg: Config):
    """Create empty organism state arrays."""
    return {
        'genome': jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
        'genome_len': jnp.int32(0),
        # Structural elements: SE[0] = IP, SE[1..] = registers/stacks/queues
        'se_values': jnp.zeros(cfg.max_se_count, dtype=jnp.int32),
        'n_ses': jnp.int32(1),  # At least IP
        'separator_pos': jnp.int32(0),
        'n_instructions': jnp.int32(0),
        # Instruction table: raw normalized micro-ops per instruction
        'instruction_table': jnp.full((cfg.max_instructions, cfg.max_micro_ops), BLANK, dtype=jnp.int32),
        'instruction_lengths': jnp.zeros(cfg.max_instructions, dtype=jnp.int32),
        # Child allocation
        'child': jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
        'child_len': jnp.int32(0),
        'child_copied': jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_),
        'already_allocated': jnp.bool_(False),
        # Execution state
        'age': jnp.int32(0),
        'alive': jnp.bool_(True),
        'has_child': jnp.bool_(False),
        'counter': jnp.int32(0),
        'gestation_time': jnp.int32(2147483647),
        # Child tape for placement after divide
        'child_tape': jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32),
        'child_tape_len': jnp.int32(0),
        # Executed tracking (for fitness/merit computation)
        'executed': jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_),
        # Visualization
        'color': jnp.zeros(3, dtype=jnp.float32),
        'child_color': jnp.zeros(3, dtype=jnp.float32),
    }


