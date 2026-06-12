import jax
import jax.numpy as jnp
import jax.lax as lax
from physax.constants import *

def build_structure(genome, genome_len, cfg):
    """Scan genome before SEP. Count R/S/Q markers as structural elements.
    SE[0] is always IP (implicit). R/S/Q each add one SE (simplification:
    all treated as registers; stacks/queues not deeply supported).
    Returns n_ses, separator_pos.
    """
    # Scan positions 0..genome_len-1 looking for R, S, Q, SEP
    # Stop at first SEP
    positions = jnp.arange(cfg.max_genome_len)
    valid = positions < genome_len

    def scan_fn(carry, pos):
        n_ses, sep_pos, found_sep = carry
        gene = jnp.where(valid[pos], genome[pos], BLANK)
        is_sep = (gene == SEP) & ~found_sep
        is_r = (gene == R) & ~found_sep
        is_s = (gene == S) & ~found_sep
        is_q = (gene == Q) & ~found_sep
        is_se = is_r | is_s | is_q
        n_ses = jnp.where(is_se & (n_ses < cfg.max_se_count), n_ses + 1, n_ses)
        sep_pos = jnp.where(is_sep, pos, sep_pos)
        found_sep = found_sep | is_sep
        return (n_ses, sep_pos, found_sep), None

    # Start with 1 SE (the implicit IP = SE[0])
    (n_ses, separator_pos, _), _ = lax.scan(
        scan_fn,
        (jnp.int32(1), jnp.int32(0), jnp.bool_(False)),
        jnp.arange(cfg.max_genome_len)
    )

    return n_ses, separator_pos


def build_instruction_set(genome, genome_len, separator_pos, cfg):
    """Build instruction table from genome, matching UP.java buildInstructionSet + createInstruction.

    Scans genome[0..separator_pos-1] for I markers.
    Each pair of consecutive I markers (or I→SEP) defines one instruction.
    Content between markers is read, opcodes normalized via abs(val) % UP_IS_SIZE,
    then operands skipped based on N_OPERANDS lookup.
    """
    positions = jnp.arange(cfg.max_genome_len)
    valid = positions < separator_pos

    # Find I marker positions
    is_i_marker = (genome == I) & valid
    # Collect I positions (sorted, padded with max_genome_len)
    i_positions = jnp.where(is_i_marker, positions, cfg.max_genome_len)
    i_positions = jnp.sort(i_positions)

    # Count actual I markers
    n_i_markers = jnp.sum(is_i_marker.astype(jnp.int32))

    # For each instruction index, determine start (after I marker) and stop (next I or SEP)
    def create_one_instruction(instr_idx):
        """Create instruction number instr_idx."""
        i_pos = i_positions[instr_idx]
        valid_instr = instr_idx < n_i_markers

        start = i_pos + 1  # Content starts after the I marker

        # Find the end: next I marker or separator_pos
        next_i_pos = jnp.where(
            (instr_idx + 1) < n_i_markers,
            i_positions[instr_idx + 1],
            separator_pos
        )
        stop = next_i_pos
        size = stop - start

        # Read raw content from genome[start:stop]
        raw = jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32)
        def read_raw(j):
            pos = start + j
            return jnp.where((j < size) & (pos < genome_len), genome[pos], BLANK)
        raw = jax.vmap(read_raw)(jnp.arange(cfg.max_micro_ops))

        # Normalize: walk through raw, normalize opcodes, skip operands
        def normalize_step(carry, j):
            out_arr, next_opcode_pos = carry
            is_opcode_pos = (j == next_opcode_pos) & (j < size)
            val = raw[j]
            # Normalize opcode: abs(val) % UP_IS_SIZE
            normalized = jnp.abs(val) % UP_IS_SIZE
            # Look up operand count for this opcode
            n_ops = N_OPERANDS[normalized]
            # If this is an opcode position, normalize it; operand positions stay raw
            out_val = jnp.where(is_opcode_pos, normalized, val)
            out_arr = jnp.where(j < size, out_arr.at[j].set(out_val), out_arr)
            # Next opcode position is current + 1 + n_ops
            next_opcode_pos = jnp.where(is_opcode_pos, j + 1 + n_ops, next_opcode_pos)
            return (out_arr, next_opcode_pos), None

        init_arr = jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32)
        (result_arr, _), _ = lax.scan(
            normalize_step,
            (init_arr, jnp.int32(0)),  # First opcode at position 0
            jnp.arange(cfg.max_micro_ops)
        )

        # Mask invalid instruction
        result_arr = jnp.where(valid_instr, result_arr, jnp.full(cfg.max_micro_ops, BLANK, dtype=jnp.int32))
        length = jnp.where(valid_instr, jnp.minimum(size, cfg.max_micro_ops), jnp.int32(0))

        return result_arr, length

    instruction_table, instruction_lengths = jax.vmap(create_one_instruction)(
        jnp.arange(cfg.max_instructions)
    )

    n_instructions = jnp.minimum(n_i_markers, cfg.max_instructions)

    return instruction_table, instruction_lengths, n_instructions


def parse_genome(genome, genome_len, cfg):
    """Parse genome into phenotype: structure + instruction set."""
    n_ses, separator_pos = build_structure(genome, genome_len, cfg)
    instruction_table, instruction_lengths, n_instructions = build_instruction_set(
        genome, genome_len, separator_pos, cfg
    )
    return {
        'n_ses': n_ses,
        'separator_pos': separator_pos,
        'n_instructions': n_instructions,
        'instruction_table': instruction_table,
        'instruction_lengths': instruction_lengths,
    }


