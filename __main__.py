"""
Physis Minimal - JAX Version

Faithful port of the original Physis/ARCHE universal processor.
Parallelized digital evolution simulation using JAX.
- Population runs in parallel via vmap
- Cycles via lax.scan with interleaved birth handling
- Fixed-size padded genomes for JAX compatibility
"""
import jax
# SS: uncomment to force use of CPU
#jax.config.update('jax_platforms', 'cpu')

import jax.numpy as jnp
import jax.lax as lax
from jax import random
from functools import partial
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
try:
    # SS: modified to stop a warning message
    # import imageio
    import imageio.v2 as imageio
except ImportError:
    imageio = None
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==========================================
# 1. CONSTANTS (Gene values — matching UP.java)
# ==========================================

NOP = 0
IN = 1
OUT = 2
LOAD = 3
STORE = 4
MOVE = 5
ALLOCATE = 6
COMPARE = 7
IFZERO = 8
JUMP = 9
DEC = 10
INC = 11
DIVIDE = 12
SDIR = 13
GDIR = 14
SEND = 15
RECEIVE = 16
ADD = 17
SUB = 18
MUL = 19
DIV_OP = 20
MOD = 21
AND = 22
OR = 23
XOR = 24
NEG = 25
NOT = 26
SHIFT_L = 27
SHIFT_R = 28
FORK_TH = 29
KILL_TH = 30
R = 31
S = 32
Q = 33
I = 34
B = 35
SEP = 36
CLEAR = 37
CINC = 38
CDEC = 39
IS_SEP = 40
REL_LOAD = 41
REL_STORE = 42
IFNOTZERO = 43

UP_IS_SIZE = 44
BLANK = -1

# Number of operands for each opcode (index = opcode, 44 entries)
# From the instruction set definition file
N_OPERANDS = jnp.array([
    0,  # 0: NOP
    1,  # 1: IN
    1,  # 2: OUT
    2,  # 3: LOAD
    2,  # 4: STORE
    2,  # 5: MOVE
    1,  # 6: ALLOCATE
    3,  # 7: COMPARE
    1,  # 8: IFZERO
    1,  # 9: JUMP
    1,  # 10: DEC
    1,  # 11: INC
    0,  # 12: DIVIDE
    1,  # 13: SDIR
    1,  # 14: GDIR
    1,  # 15: SEND
    1,  # 16: RECEIVE
    3,  # 17: ADD
    3,  # 18: SUB
    3,  # 19: MUL
    3,  # 20: DIV
    3,  # 21: MOD
    3,  # 22: AND
    3,  # 23: OR
    3,  # 24: XOR
    2,  # 25: NEG
    2,  # 26: NOT
    2,  # 27: SHIFT_L
    2,  # 28: SHIFT_R
    1,  # 29: FORK_TH
    0,  # 30: KILL_TH
    0,  # 31: R
    0,  # 32: S
    0,  # 33: Q
    0,  # 34: I
    0,  # 35: B
    0,  # 36: SEPARATOR
    1,  # 37: CLEAR
    1,  # 38: CINC
    1,  # 39: CDEC
    2,  # 40: IS_SEP
    3,  # 41: REL_LOAD
    3,  # 42: REL_STORE
    1,  # 43: IFNOTZERO
], dtype=jnp.int32)

# SS: plotting values for percentile plots
LS = ['dotted','dashdot','dashed','solid','dashed','dashdot','dotted']
PERCENTILES = jnp.array([5,10,25,50,75,90,95])

# ==========================================
# 2. CONFIGURATION
# ==========================================

class Config:
    """Simulation configuration with fixed sizes for JAX."""
    max_genome_len: int = 256
    max_se_count: int = 16
    max_instructions: int = 64
    max_micro_ops: int = 32
    pop_size: int = 1024
    initial_pop: int = 1

    steps_per_update: int = 34
    copy_mutation_rate: float = 0.009
    divide_mutation_rate: float = 0.0
    divide_insert_rate: float = 0.0013
    divide_delete_rate: float = 0.0013
    min_allocation_ratio: float = 0.5
    max_allocation_ratio: float = 2.0
    min_proliferation_ratio: float = 0.80

    use_species_color: bool = True


def make_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ==========================================
# 3. ORGANISM STATE (as JAX arrays)
# ==========================================

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


# ==========================================
# 4. GENOME PARSING
# ==========================================

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


# ==========================================
# 5. VM EXECUTION
# ==========================================

def tape_read(genome, genome_len, child, child_len, already_allocated, position):
    """Unified address space read. Matches CellGeneticCodeTape.read().
    position = abs(position % total_size)
    [0, parent_len) -> parent, [parent_len, total) -> child
    """
    total_size = genome_len + jnp.where(already_allocated, child_len, 0)
    total_size = jnp.maximum(total_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < genome_len
    parent_val = genome[jnp.clip(pos, 0, genome_len - 1)]
    child_idx = jnp.clip(pos - genome_len, 0, child_len - 1)
    child_val = child[child_idx]
    return jnp.where(in_parent, parent_val, child_val)


def tape_write(genome, child, child_copied, genome_len, child_len, already_allocated,
               position, value, key, copy_mutation_rate):
    """Unified address space write. Matches CellGeneticCodeTape.write().
    Writes to parent or child. Copy mutation only on child writes.
    Returns (genome, child, child_copied).
    """
    total_size = genome_len + jnp.where(already_allocated, child_len, 0)
    total_size = jnp.maximum(total_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < genome_len

    # Parent write
    parent_idx = jnp.clip(pos, 0, genome_len - 1)
    genome = jnp.where(
        in_parent,
        genome.at[parent_idx].set(value),
        genome
    )

    # Child write
    child_idx = jnp.clip(pos - genome_len, 0, child_len - 1)
    in_child = ~in_parent & already_allocated

    # Copy mutation: replace with random gene
    k1, k2 = random.split(key)
    do_mutate = random.uniform(k1) < copy_mutation_rate
    mutated_value = random.randint(k2, (), 0, UP_IS_SIZE).astype(jnp.int32)
    final_value = jnp.where(do_mutate & in_child, mutated_value, value)

    child = jnp.where(
        in_child,
        child.at[child_idx].set(final_value),
        child
    )
    child_copied = jnp.where(
        in_child,
        child_copied.at[child_idx].set(True),
        child_copied
    )

    return genome, child, child_copied


def tape_fetch_inst(genome, genome_len, ip_val):
    """Fetch instruction from parent memory only. Out-of-bounds returns BLANK.
    Matches CellGeneticCodeTape.fetchInst().
    """
    in_bounds = (ip_val >= 0) & (ip_val < genome_len)
    idx = jnp.clip(ip_val, 0, genome_len - 1)
    return jnp.where(in_bounds, genome[idx], BLANK)


def vm_execute_one(state, key, cfg):
    """Execute one compound instruction. Matches UP.java execute().

    1. fetchInst(IP) from parent memory
    2. Map to instruction index: abs(fetched) % n_instructions
    3. Execute micro-ops via lax.scan
    4. IP += 1 unless divide returned
    """
    genome = state['genome']
    genome_len = state['genome_len']
    se_values = state['se_values']
    n_ses = state['n_ses']
    n_instructions = state['n_instructions']
    instruction_table = state['instruction_table']
    instruction_lengths = state['instruction_lengths']
    child = state['child']
    child_len = state['child_len']
    child_copied = state['child_copied']
    already_allocated = state['already_allocated']
    separator_pos = state['separator_pos']
    has_child = state['has_child']
    counter = state['counter']
    gestation_time = state['gestation_time']

    ip_val = se_values[0]  # SE[0] = IP
    executed = state['executed']

    # 1. Fetch instruction from parent tape
    fetched = tape_fetch_inst(genome, genome_len, ip_val)

    # Mark IP position as executed (matches CellGeneticCodeTape.fetchInst() setting EXECUTED)
    in_bounds = (ip_val >= 0) & (ip_val < genome_len)
    clip_ip = jnp.clip(ip_val, 0, cfg.max_genome_len - 1)
    executed = jnp.where(in_bounds, executed.at[clip_ip].set(True), executed)

    # 2. Map to instruction index
    safe_n_instr = jnp.maximum(n_instructions, 1)
    instr_idx = jnp.abs(fetched) % safe_n_instr
    instr_idx = jnp.where(n_instructions > 0, instr_idx, 0)
    instr = instruction_table[instr_idx]
    instr_len = instruction_lengths[instr_idx]

    # 3. Execute micro-ops
    # The fillOperands logic: read operands from instruction array;
    # if not enough, read from tape and advance IP.

    # We need to track: position in instruction array, IP (for overflow reads),
    # SE values, child state, divide_returned flag, and a "next_opcode_pos"
    # to know which positions are opcodes vs operands.

    # Pre-split keys for all micro-op steps
    step_keys = random.split(key, cfg.max_micro_ops + 1)

    def micro_op_step(carry, step_idx):
        (se_vals, child_arr, child_cop, genome_arr, already_alloc, child_l,
         ip_for_overflow, pos_in_instr, next_opcode_pos, divide_returned,
         cntr, gest_time, has_ch) = carry

        step_key = step_keys[step_idx]

        # Are we at a valid position?
        at_valid = (pos_in_instr < instr_len) & ~divide_returned

        # Read current value from instruction
        safe_pos = jnp.clip(pos_in_instr, 0, cfg.max_micro_ops - 1)
        cur_val = instr[safe_pos]

        # Is this position an opcode?
        is_opcode = (pos_in_instr == next_opcode_pos) & at_valid

        # Get opcode (already normalized during parsing)
        opcode = jnp.where(is_opcode, cur_val, NOP)

        # Get number of operands for this opcode
        safe_opcode = jnp.clip(opcode, 0, UP_IS_SIZE - 1)
        n_ops = jnp.where(is_opcode, N_OPERANDS[safe_opcode], jnp.int32(0))

        # fillOperands: read n_ops operands starting from pos_in_instr+1
        # If not enough in instruction, fetch from tape (parent) and advance ip
        ops_start = pos_in_instr + 1
        remaining_in_instr = jnp.maximum(instr_len - ops_start, 0)

        # Read up to 3 operands
        def read_operand(op_idx, ip_ov):
            """Read one operand. From instruction if available, else from tape."""
            instr_pos = ops_start + op_idx
            from_instr = op_idx < remaining_in_instr
            safe_ipos = jnp.clip(instr_pos, 0, cfg.max_micro_ops - 1)
            instr_val = instr[safe_ipos]
            tape_val = tape_read(genome_arr, genome_len, child_arr, child_l, already_alloc, ip_ov)
            val = jnp.where(from_instr, instr_val, tape_val)
            # Advance IP only when reading from tape
            new_ip = jnp.where(from_instr, ip_ov, ip_ov + 1)
            return val, new_ip

        op0, ip_ov1 = read_operand(jnp.int32(0), ip_for_overflow)
        op1, ip_ov2 = read_operand(jnp.int32(1), ip_ov1)
        op2, ip_ov3 = read_operand(jnp.int32(2), ip_ov2)

        # Select appropriate IP based on actual n_ops
        ip_after_ops = jnp.where(n_ops >= 3, ip_ov3, jnp.where(n_ops >= 2, ip_ov2, jnp.where(n_ops >= 1, ip_ov1, ip_for_overflow)))

        # Normalize operands: abs(val) % Short.MAX_VALUE, then % n_ses for SE indexing
        def norm_op(val):
            v = jnp.where(val < 0, -val, val) % 32767
            return v % jnp.maximum(n_ses, 1)

        o0 = norm_op(op0)
        o1 = norm_op(op1)
        o2 = norm_op(op2)

        # Helper to read/write SEs
        def se_read(idx):
            return se_vals[jnp.clip(idx, 0, cfg.max_se_count - 1)]

        def se_write(se_arr, idx, val):
            return se_arr.at[jnp.clip(idx, 0, cfg.max_se_count - 1)].set(val)

        # Compute tape_size for CINC/CDEC
        tape_size = genome_len + jnp.where(already_alloc, child_l, 0)
        tape_size = jnp.maximum(tape_size, 1)

        # ---- Execute opcode ----
        # Default: no change
        new_se = se_vals
        new_child = child_arr
        new_child_cop = child_cop
        new_genome = genome_arr
        new_already_alloc = already_alloc
        new_child_l = child_l
        new_ip_ov = ip_after_ops
        new_divide_ret = divide_returned
        new_cntr = cntr
        new_gest = gest_time
        new_has_ch = has_ch
        did_jump = jnp.bool_(False)

        # NOP (0): do nothing — also handles descriptors R/S/Q/I/B/SEP which fall to default
        # IN (1): NOP stub (consume 1 operand, do nothing)
        # OUT (2): NOP stub
        # SDIR (13), GDIR (14), SEND (15), RECEIVE (16): NOP stubs
        # FORK_TH (29), KILL_TH (30): NOP stubs

        # LOAD (3): SE[o1] = tape.read(SE[o0])
        is_load = is_opcode & (opcode == LOAD)
        load_addr = se_read(o0)
        load_val = tape_read(new_genome, genome_len, new_child, new_child_l, new_already_alloc, load_addr)
        new_se = jnp.where(is_load, se_write(new_se, o1, load_val), new_se)

        # STORE (4): tape.write(SE[o0], SE[o1]) — destination is o0, data is o1
        is_store = is_opcode & (opcode == STORE)
        store_dest = se_read(o0)
        store_data = se_read(o1)
        k_store = step_key
        g_s, c_s, cc_s = tape_write(
            new_genome, new_child, new_child_cop, genome_len, new_child_l, new_already_alloc,
            store_dest, store_data, k_store, cfg.copy_mutation_rate
        )
        new_genome = jnp.where(is_store, g_s, new_genome)
        new_child = jnp.where(is_store, c_s, new_child)
        new_child_cop = jnp.where(is_store, cc_s, new_child_cop)

        # MOVE (5): SE[o1] = SE[o0]
        is_move = is_opcode & (opcode == MOVE)
        new_se = jnp.where(is_move, se_write(new_se, o1, se_read(o0)), new_se)

        # ALLOCATE (6): allocate child if conditions met
        is_allocate = is_opcode & (opcode == ALLOCATE)
        alloc_size = se_read(o0)
        alloc_possible = (
            ~new_already_alloc &
            (alloc_size > (cfg.min_allocation_ratio * genome_len).astype(jnp.int32)) &
            (alloc_size < (cfg.max_allocation_ratio * genome_len).astype(jnp.int32))
        )
        do_alloc = is_allocate & alloc_possible
        new_already_alloc = jnp.where(do_alloc, True, new_already_alloc)
        new_child_l = jnp.where(do_alloc, alloc_size, new_child_l)
        # Fill child with BLANK
        blank_child = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
        new_child = jnp.where(do_alloc, blank_child, new_child)
        new_child_cop = jnp.where(do_alloc, jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_), new_child_cop)

        # COMPARE (7): SE[o2] = sign(SE[o0] - SE[o1])
        is_compare = is_opcode & (opcode == COMPARE)
        cmp_a = se_read(o0)
        cmp_b = se_read(o1)
        cmp_result = jnp.where(cmp_a < cmp_b, -1, jnp.where(cmp_a == cmp_b, 0, 1))
        new_se = jnp.where(is_compare, se_write(new_se, o2, cmp_result), new_se)

        # IFZERO (8): if SE[o0] != 0, skip next instruction (IP += 1)
        is_ifzero = is_opcode & (opcode == IFZERO)
        ifz_val = se_read(o0)
        ifz_skip = ifz_val != 0
        # IP increment happens via incrementing se_values[0]
        new_se = jnp.where(
            is_ifzero & ifz_skip,
            se_write(new_se, jnp.int32(0), new_se[0] + 1),
            new_se
        )

        # JUMP (9): IP = SE[o0]
        is_jump = is_opcode & (opcode == JUMP)
        jump_target = se_read(o0)
        new_se = jnp.where(is_jump, se_write(new_se, jnp.int32(0), jump_target), new_se)
        did_jump = did_jump | is_jump

        # DEC (10): SE[o0] -= 1
        is_dec = is_opcode & (opcode == DEC)
        new_se = jnp.where(is_dec, se_write(new_se, o0, se_read(o0) - 1), new_se)

        # INC (11): SE[o0] += 1
        is_inc = is_opcode & (opcode == INC)
        new_se = jnp.where(is_inc, se_write(new_se, o0, se_read(o0) + 1), new_se)

        # DIVIDE (12): check allocation + proliferation, produce child
        is_divide = is_opcode & (opcode == DIVIDE)
        n_copied = jnp.sum(new_child_cop.astype(jnp.int32))
        prolif_possible = (
            new_already_alloc &
            (n_copied > (cfg.min_proliferation_ratio * genome_len).astype(jnp.int32))
        )
        do_divide = is_divide & prolif_possible

        # On successful divide: record gestation, set has_child
        new_gest = jnp.where(do_divide, new_cntr, new_gest)
        new_has_ch = jnp.where(do_divide, True, new_has_ch)
        new_divide_ret = jnp.where(is_divide, True, new_divide_ret)

        # On failed divide: IP += 1
        failed_divide = is_divide & ~prolif_possible
        new_se = jnp.where(
            failed_divide,
            se_write(new_se, jnp.int32(0), new_se[0] + 1),
            new_se
        )

        # ADD (17): SE[o2] = SE[o0] + SE[o1]
        is_add = is_opcode & (opcode == ADD)
        new_se = jnp.where(is_add, se_write(new_se, o2, se_read(o0) + se_read(o1)), new_se)

        # SUB (18): SE[o2] = SE[o0] - SE[o1]
        is_sub = is_opcode & (opcode == SUB)
        new_se = jnp.where(is_sub, se_write(new_se, o2, se_read(o0) - se_read(o1)), new_se)

        # MUL (19): SE[o2] = SE[o0] * SE[o1]
        is_mul = is_opcode & (opcode == MUL)
        new_se = jnp.where(is_mul, se_write(new_se, o2, se_read(o0) * se_read(o1)), new_se)

        # DIV (20): SE[o2] = SE[o0] / SE[o1] (integer, skip if divisor=0)
        is_div = is_opcode & (opcode == DIV_OP)
        divisor = se_read(o1)
        safe_divisor = jnp.where(divisor == 0, 1, divisor)
        div_result = se_read(o0) // safe_divisor
        new_se = jnp.where(is_div & (divisor != 0), se_write(new_se, o2, div_result), new_se)

        # MOD (21): SE[o2] = SE[o0] % SE[o1] (skip if divisor=0)
        is_mod = is_opcode & (opcode == MOD)
        mod_divisor = se_read(o1)
        safe_mod_div = jnp.where(mod_divisor == 0, 1, mod_divisor)
        mod_result = se_read(o0) % safe_mod_div
        new_se = jnp.where(is_mod & (mod_divisor != 0), se_write(new_se, o2, mod_result), new_se)

        # AND (22): SE[o2] = SE[o0] & SE[o1]
        is_and = is_opcode & (opcode == AND)
        new_se = jnp.where(is_and, se_write(new_se, o2, se_read(o0) & se_read(o1)), new_se)

        # OR (23): SE[o2] = SE[o0] | SE[o1]
        is_or = is_opcode & (opcode == OR)
        new_se = jnp.where(is_or, se_write(new_se, o2, se_read(o0) | se_read(o1)), new_se)

        # XOR (24): SE[o2] = SE[o0] ^ SE[o1]
        is_xor = is_opcode & (opcode == XOR)
        new_se = jnp.where(is_xor, se_write(new_se, o2, se_read(o0) ^ se_read(o1)), new_se)

        # NEG (25): SE[o1] = -SE[o0]
        is_neg = is_opcode & (opcode == NEG)
        new_se = jnp.where(is_neg, se_write(new_se, o1, -se_read(o0)), new_se)

        # NOT (26): SE[o1] = ~SE[o0]
        is_not = is_opcode & (opcode == NOT)
        new_se = jnp.where(is_not, se_write(new_se, o1, ~se_read(o0)), new_se)

        # SHIFT_L (27): SE[o1] = SE[o0] << 1
        is_shl = is_opcode & (opcode == SHIFT_L)
        new_se = jnp.where(is_shl, se_write(new_se, o1, se_read(o0) << 1), new_se)

        # SHIFT_R (28): SE[o1] = SE[o0] >> 1
        is_shr = is_opcode & (opcode == SHIFT_R)
        new_se = jnp.where(is_shr, se_write(new_se, o1, se_read(o0) >> 1), new_se)

        # CLEAR (37): SE[o0] = 0
        is_clear = is_opcode & (opcode == CLEAR)
        new_se = jnp.where(is_clear, se_write(new_se, o0, 0), new_se)

        # CINC (38): SE[o0] = (SE[o0] + 1) % tape_size
        is_cinc = is_opcode & (opcode == CINC)
        cinc_val = (se_read(o0) + 1) % tape_size
        new_se = jnp.where(is_cinc, se_write(new_se, o0, cinc_val), new_se)

        # CDEC (39): SE[o0] -= 1, wrap to tape_size-1 if negative
        is_cdec = is_opcode & (opcode == CDEC)
        cdec_raw = se_read(o0) - 1
        cdec_val = jnp.where(cdec_raw < 0, tape_size - 1, cdec_raw)
        new_se = jnp.where(is_cdec, se_write(new_se, o0, cdec_val), new_se)

        # IS_SEP (40): SE[o1] = 1 if tape.read(SE[o0]) == SEP else 0
        # Wait — re-reading UP.java: is_sep(src, dst) reads SE[src], not tape.read(SE[src])
        # Actually: ses[src % ssize].read() == SEPARATOR → it reads the SE value, not from tape
        is_issep = is_opcode & (opcode == IS_SEP)
        issep_val = jnp.where(se_read(o0) == SEP, jnp.int32(1), jnp.int32(0))
        new_se = jnp.where(is_issep, se_write(new_se, o1, issep_val), new_se)

        # REL_LOAD (41): SE[o2] = tape.read(SE[o0] + SE[o1])
        is_rload = is_opcode & (opcode == REL_LOAD)
        rload_addr = se_read(o0) + se_read(o1)
        rload_val = tape_read(new_genome, genome_len, new_child, new_child_l, new_already_alloc, rload_addr)
        new_se = jnp.where(is_rload, se_write(new_se, o2, rload_val), new_se)

        # REL_STORE (42): tape.write(SE[o0] + SE[o1], SE[o2])
        is_rstore = is_opcode & (opcode == REL_STORE)
        rstore_addr = se_read(o0) + se_read(o1)
        rstore_data = se_read(o2)
        k_rstore = step_key
        g_rs, c_rs, cc_rs = tape_write(
            new_genome, new_child, new_child_cop, genome_len, new_child_l, new_already_alloc,
            rstore_addr, rstore_data, k_rstore, cfg.copy_mutation_rate
        )
        new_genome = jnp.where(is_rstore, g_rs, new_genome)
        new_child = jnp.where(is_rstore, c_rs, new_child)
        new_child_cop = jnp.where(is_rstore, cc_rs, new_child_cop)

        # IFNOTZERO (43): if SE[o0] == 0, skip next instruction (IP += 1)
        is_ifnz = is_opcode & (opcode == IFNOTZERO)
        ifnz_val = se_read(o0)
        ifnz_skip = ifnz_val == 0
        new_se = jnp.where(
            is_ifnz & ifnz_skip,
            se_write(new_se, jnp.int32(0), new_se[0] + 1),
            new_se
        )

        # Update counter (counts micro-ops, matching UP.java counter++ per micro-op in while loop)
        new_cntr = jnp.where(is_opcode, new_cntr + 1, new_cntr)

        # Advance position: if this was an opcode, next position = pos + 1 + n_ops
        # (operands are already consumed). If not opcode, shouldn't happen (we skip).
        ops_consumed_from_instr = jnp.minimum(n_ops, remaining_in_instr)
        new_pos = jnp.where(
            is_opcode,
            pos_in_instr + 1 + ops_consumed_from_instr,
            pos_in_instr  # Stay put if not at a valid opcode position
        )

        # Update next_opcode_pos
        new_next_opcode = jnp.where(is_opcode, new_pos, next_opcode_pos)

        # Update IP for overflow
        new_ip_ov = jnp.where(is_opcode, ip_after_ops, ip_for_overflow)
        # For JUMP: the outer loop will add 1, so we set IP = target and did_jump
        # (But IP overflow tracking is separate from SE[0]; SE[0] is the real IP)

        new_carry = (new_se, new_child, new_child_cop, new_genome, new_already_alloc, new_child_l,
                     new_ip_ov, new_pos, new_next_opcode, new_divide_ret,
                     new_cntr, new_gest, new_has_ch)
        return new_carry, did_jump

    init_carry = (se_values, child, child_copied, genome, already_allocated, child_len,
                  ip_val, jnp.int32(0), jnp.int32(0), jnp.bool_(False),
                  counter, gestation_time, has_child)

    final_carry, did_jumps = lax.scan(micro_op_step, init_carry, jnp.arange(cfg.max_micro_ops))

    (se_values, child, child_copied, genome, already_allocated, child_len,
     ip_for_overflow, _, _, divide_returned,
     counter, gestation_time, has_child) = final_carry

    any_jump = jnp.any(did_jumps)

    # Post micro-ops: IP += 1 unless divide returned
    # In UP.java: after the while loop, incrementIP(1) is called.
    # But JUMP sets IP directly (and the +1 still applies after).
    # And DIVIDE returns early (no +1).
    # IFZERO/IFNOTZERO already added their +1 to IP inside the loop.
    # So: IP += 1 unconditionally, unless divide_returned.
    # But if JUMP happened, IP was already set to target in the loop.
    # The +1 still applies! In UP.java, jump() calls setIP(), then execute() does incrementIP(1).
    # Wait — actually DIVIDE calls return before incrementIP. Let me re-read.
    # In UP.java execute(): the DIVIDE case does return (exits execute method),
    # so incrementIP(1) at the bottom is NOT reached for DIVIDE.
    # For JUMP: setIP(target), then incrementIP(1) is called. So effective IP = target + 1? No!
    # Actually wait — looking at UP.java more carefully:
    # jump() calls setIP(ses[address % ssize].read()) which directly sets IP.value
    # Then after the while loop, incrementIP(1) adds 1 to IP.
    # So yes, IP ends up at target + 1 after a jump. But wait... that seems wrong for the ancestor.
    # Let me check: instruction 6 is "jump 3" which means jump to SE[3].
    # But the ancestor's usage: SE[3] is the "return_reg" which stores the IP value.
    # The ancestor's instruction 2 does "move 0 3" which copies IP (SE[0]) to SE[3].
    # Then later instruction 6 does "jump 3" meaning IP = SE[3]. Then +1 is applied.
    # So if SE[3] stored the IP of instruction 2's execution, the jump goes to SE[3]+1.
    # Actually, in UP.java, IP is set via setIP then incrementIP(1) after the while loop.
    # BUT: the jump instruction inside the while loop sets IP. The while loop continues
    # executing remaining micro-ops in the compound instruction. Then incrementIP(1).
    # So effective: IP_final = (whatever IP was after all micro-ops in the instruction) + 1
    # For jump specifically: IP = SE[addr], then after loop: IP += 1
    #
    # OK but there's a subtlety: does the while loop continue after jump?
    # Yes! There's no break/return for JUMP — it just sets IP and continues to next micro-op.
    # Only DIVIDE does return.
    #
    # So the +1 applies to whatever IP is at the end of the compound instruction.

    new_ip = se_values[0] + 1
    new_ip = jnp.where(divide_returned, se_values[0], new_ip)

    # On successful divide: restart (IP = separator_pos + 1, counter = 0)
    do_divide_success = has_child & ~state['has_child']  # Newly set
    restart_ip = (separator_pos + 1) % jnp.maximum(genome_len, 1)
    new_ip = jnp.where(do_divide_success, restart_ip, new_ip)
    counter = jnp.where(do_divide_success, jnp.int32(0), counter)

    se_values = se_values.at[0].set(new_ip)

    new_state = state.copy()
    new_state['se_values'] = se_values
    new_state['child'] = child
    new_state['child_len'] = child_len
    new_state['child_copied'] = child_copied
    new_state['already_allocated'] = already_allocated
    new_state['genome'] = genome
    new_state['has_child'] = has_child
    new_state['counter'] = counter
    new_state['gestation_time'] = gestation_time
    new_state['executed'] = executed
    return new_state


def organism_update(state, key, cfg):
    """Execute steps_per_update compound instructions on one organism."""
    def step_fn(state, step_key):
        should_exec = state['alive'] & ~state['has_child']
        new_state = vm_execute_one(state, step_key, cfg)
        # Only apply if should execute
        result = jax.tree.map(
            lambda n, o: jnp.where(should_exec, n, o),
            new_state, state
        )
        return result, None

    keys = random.split(key, cfg.steps_per_update)
    state, _ = lax.scan(step_fn, state, keys)
    return state


# ==========================================
# 6. MUTATION
# ==========================================

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


# ==========================================
# 7. POPULATION INITIALIZATION
# ==========================================

def create_ancestor_genome(cfg):
    """Create the 78-gene arche.replicator ancestor genome."""
    # Hardcoded from physis/data/genebank/arche/arche.replicator
    g = jnp.array([
        R, B, R, B, R, B, R,                           # Structure: 4 registers (+ IP = 5 SEs)
        I, MOVE, NOP, 2,                                # I0: move 0 2 (store_ptr)
        I, CLEAR, 1,                                    # I1: clear 1 (clear_counter)
        I, MOVE, NOP, 3,                                # I2: move 0 3 (s_lab: store IP to return_reg)
        I, INC, 1,                                      # I3: inc 1 (inc_counter)
        I, CINC, 2,                                     # I4: cinc 2 (cinc_pointer)
        I, LOAD, 2, 4, IS_SEP, 4, 4, IFZERO, 4,        # I5: load 2 4; is_sep 4 4; ifzero 4
        I, JUMP, 3,                                     # I6: jump 3 (jump_by_reg)
        I, ALLOCATE, 1,                                 # I7: allocate 1
        I, MOVE, 1, 2,                                  # I8: move 1 2 (fill_pointer)
        I, LOAD, 1, 4,                                  # I9: load 1 4 (load_data)
        I, REL_STORE, 1, 2, 4,                          # I10: rel_store 1 2 4
        I, DEC, 1,                                      # I11: dec 1
        I, IFNOTZERO, 1,                                # I12: ifnotzero 1
        I, DIVIDE,                                      # I13: divide
        SEP,                                            # Separator at position 60
        # Code section (positions 61-77): 17 entries mapping to 14 instructions
        0, 1, 2, 3, 4, 5, 6, 3, 7, 8, 2, 11, 9, 10, 12, 6, 13
    ], dtype=jnp.int32)

    genome = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
    genome = genome.at[:len(g)].set(g)
    genome_len = jnp.int32(len(g))

    return genome, genome_len


def init_organism(genome, genome_len, color, cfg):
    """Initialize an organism from a genome."""
    state = create_organism_state(cfg)
    state['genome'] = genome
    state['genome_len'] = genome_len
    state['color'] = color

    parsed = parse_genome(genome, genome_len, cfg)
    state['n_ses'] = parsed['n_ses']
    state['separator_pos'] = parsed['separator_pos']
    state['n_instructions'] = parsed['n_instructions']
    state['instruction_table'] = parsed['instruction_table']
    state['instruction_lengths'] = parsed['instruction_lengths']

    # IP starts right after separator
    ip_start = (parsed['separator_pos'] + 1) % jnp.maximum(genome_len, 1)
    state['se_values'] = state['se_values'].at[0].set(ip_start)

    return state


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


# ==========================================
# 8. CYCLE STEP (single cycle for all organisms)
# ==========================================

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


# ==========================================
# 9. VISUALIZATION
# ==========================================

def compute_snapshot_properties(snap, max_genome_len):
    """Compute fitness, merit, and effective length from a snapshot dict.
    Pure NumPy, called at snapshot time (not JIT'd).
    Returns (effective_length, merit, fitness, fertile) arrays of shape (pop_size,).
    """
    mask = np.arange(max_genome_len)[None, :] < snap['genome_len'][:, None]
    effective_length = np.sum(snap['executed'] & mask, axis=1)
    merit = effective_length.astype(np.float64)  # bonus=1.0, no tasks
    gt = snap['gestation_time']
    INVALID = 2147483647
    fertile = gt < INVALID
    fitness = np.where(fertile, merit / np.maximum(gt, 1).astype(np.float64), 0.0)
    return effective_length, merit, fitness, fertile


# SS: use percentiles not avg -- include births
#def plot_metrics(timestamps, pop_sizes, avg_lens, filename="metrics.png"):
def plot_metrics(timestamps, pop_sizes, births, q_lens, filename="metrics.png"):
    """Plot population size and average genome length over time."""

    # SS: wider figure for longer runs
    #fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(20, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    # SS: plot births as well as pop size
    #ax1.set_ylabel('Population Size', color=color)
    ax1.set_ylabel('Population Size / ...Births', color=color)
    ax1.plot(timestamps, pop_sizes, color=color, label='Pop Size')
    # SS: plot births
    ax1.plot(timestamps, births, color=color, ls='dotted', lw=0.7, label='Births')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'

    # SS: plot percentiles, not avg
    # ax2.set_ylabel('Avg Genome Length', color=color)
    ax2.set_ylabel('Percentile Genome Lengths', color=color)
    #ax2.plot(timestamps, avg_lens, color=color, linestyle='--', label='Avg Len')
    for i, ls in enumerate(LS): # quartiles/percentiles
        qv = [t[i] for t in q_lens]
        ax2.plot(timestamps, qv, color=color, linestyle=ls, lw=0.8)

    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Simulation Metrics')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved metrics plot to {filename}")


def save_grid_gif(snapshots, filename, cfg):
    """Generate a GIF of the 2D grid representation."""
    print("Generating GIF...")
    frames = []
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    max_len = cfg.max_genome_len

    for i, snap in enumerate(snapshots):
        alive_mask = snap['alive']
        genome_lens = snap['genome_len']
        pad_size = grid_side * grid_side - cfg.pop_size

        alive_grid = np.pad(alive_mask, (0, pad_size), constant_values=False).reshape(grid_side, grid_side)

        if cfg.use_species_color and 'color' in snap:
            colors = snap['color']
            colors_padded = np.pad(colors, ((0, pad_size), (0, 0)), constant_values=0.0)
            hsv_grid = colors_padded.reshape(grid_side, grid_side, 3)
            rgb = mcolors.hsv_to_rgb(hsv_grid)
        else:
            len_grid = np.pad(genome_lens, (0, pad_size), constant_values=0).reshape(grid_side, grid_side)
            norm_len = np.clip(len_grid / max_len, 0, 1)
            cmap = plt.get_cmap('viridis')
            rgba = cmap(norm_len)
            rgb = rgba[..., :3]

        mask = alive_grid[..., None]
        final_img = np.where(mask, rgb, 0.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(final_img, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"Cycle {snap['cycle']}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)

        frames.append(imageio.imread(buf) if imageio else np.array(Image.open(buf)))

    if imageio:
        imageio.mimsave(filename, frames, fps=10)
        print(f"Saved GIF to {filename}")
    else:
        print("imageio not installed, cannot save GIF.")


# Physis color spectrum (matching ColorRange.java)
# Anchor points: dark blue → blue → cyan → green → yellow → red → pink
PHYSIS_SPECTRUM = [
    (0, 0, 96/255),      # dark blue
    (0, 0, 1),            # blue
    (0, 1, 1),            # cyan
    (0, 1, 0),            # green
    (1, 1, 0),            # yellow
    (1, 0, 0),            # red
    (1, 128/255, 128/255) # pink
]
physis_cmap = mcolors.LinearSegmentedColormap.from_list('physis', PHYSIS_SPECTRUM, N=128)


def save_physis_view_gif(snapshots, filename, cfg, view_mode='all'):
    """Generate a GIF using Physis-style property-based coloring.

    view_mode: 'fitness', 'merit', 'age', 'all' (3-panel), or 'species' (HSV lineage).
    Snapshots must include 'alive', 'genome_len', 'executed', 'gestation_time', 'age'.
    """
    if not imageio:
        print("imageio not installed, cannot save GIF.")
        return

    print(f"Generating physis-view GIF (mode={view_mode})...")
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    pad_size = grid_side * grid_side - cfg.pop_size

    # Backward-compatible: species mode uses existing HSV colors
    if view_mode == 'species':
        save_grid_gif(snapshots, filename, cfg)
        return

    # Compute properties for all snapshots
    props_list = []
    for snap in snapshots:
        eff_len, merit, fitness, fertile = compute_snapshot_properties(snap, cfg.max_genome_len)
        props_list.append({
            'effective_length': eff_len,
            'merit': merit,
            'fitness': fitness,
            'fertile': fertile,
            'age': snap['age'].astype(np.float64),
            'alive': snap['alive'],
            'cycle': snap['cycle'],
        })

    # Determine which views to render
    if view_mode == 'all':
        views = ['fitness', 'merit', 'age']
    else:
        views = [view_mode]

    # Compute running max for BY_MAX_EVER_REACHED normalization
    max_ever = {v: 0.0 for v in views}
    for p in props_list:
        for v in views:
            vals = p[v][p['alive']] if np.any(p['alive']) else np.array([0.0])
            if len(vals) > 0:
                max_ever[v] = max(max_ever[v], float(np.max(vals)))
    # Ensure non-zero
    for v in views:
        max_ever[v] = max(max_ever[v], 1.0)

    frames = []
    n_views = len(views)
    figw = 5 * n_views + 0.5
    figh = 5.0

    for pi, p in enumerate(props_list):
        fig, axes = plt.subplots(1, n_views, figsize=(figw, figh), squeeze=False)
        axes = axes[0]

        alive = p['alive']
        fertile = p['fertile']
        pop_count = int(np.sum(alive))

        for vi, view_name in enumerate(views):
            ax = axes[vi]
            vals = p[view_name]

            # Normalize to [0, 1] by max-ever
            normed = vals / max_ever[view_name]
            normed = np.clip(normed, 0, 1)

            # Build RGB grid
            # Map through physis colormap
            rgba = physis_cmap(normed)
            rgb = rgba[:, :3]  # (pop_size, 3)

            # Dead: black; alive but not fertile: dark gray
            dead_mask = ~alive
            newborn_mask = alive & ~fertile
            rgb[dead_mask] = [0.0, 0.0, 0.0]
            rgb[newborn_mask] = [64/255, 64/255, 64/255]

            # Reshape to grid
            rgb_padded = np.pad(rgb, ((0, pad_size), (0, 0)), constant_values=0.0)
            grid_img = rgb_padded.reshape(grid_side, grid_side, 3)

            ax.imshow(grid_img, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            # Gridlines
            for gx in range(grid_side + 1):
                ax.axvline(gx - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            for gy in range(grid_side + 1):
                ax.axhline(gy - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax.set_title(view_name.capitalize(), fontsize=11, fontweight='bold')

        fig.suptitle(f"Cycle {p['cycle']}  |  Pop: {pop_count}/{cfg.pop_size}",
                     fontsize=10, y=0.02)
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf).convert('RGB')))

        if (pi + 1) % 20 == 0:
            print(f"  Frame {pi + 1}/{len(props_list)}")

    imageio.mimsave(filename, frames, fps=10)
    print(f"Saved physis-view GIF to {filename}")


# ==========================================
# 10. MAIN SIMULATION
# ==========================================

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


# ==========================================
# 11. ENTRY POINT
# ==========================================
# SS: print whether running on cpu or gpu
print('device:', jax.devices()[0].platform)

if __name__ == "__main__":
    # SS: varying pop_size, initial_pop, total_cycles, log_interval
    # other parameters may need to be changed also, in the Config class

    cfg = make_config(
        #pop_size=256,
        pop_size=4096,
        #initial_pop=1,
        initial_pop=10,
    )

    key = random.PRNGKey(42)
    pop, stats = run_simulation(
        key,
        cfg,
        #total_cycles=2000,
        total_cycles=10_000,
        #log_interval=50,
        log_interval=50,
        use_wandb=False,
    )

    print("\n=== FINAL STATE ===")
    alive = pop['alive']
    alive_count = jnp.sum(alive)
    # SS: use percentiles, not avg
    #avg_len = jnp.sum(jnp.where(alive, pop['genome_len'], 0)) / jnp.maximum(alive_count, 1)
    q_lens = jnp.nanpercentile(jnp.where(alive, pop['genome_len'], jnp.nan), PERCENTILES )

    print(f"Alive: {int(alive_count)}")
    # SS: use percentiles, not avg
    #print(f"Avg genome length: {float(avg_len):.1f}")
    print(f"min={q_lens[0]:.0f}, lq={q_lens[1]:.1f}, med={q_lens[2]:.0f}, uq={q_lens[3]:.0f}, max={q_lens[4]:.0f}")

    timestamps = [s['cycle'] for s in stats]
    pop_sizes = [s['pop_size'] for s in stats]
    # SS: use percentiles, not avg -- also include births
    #avg_lens = [s['avg_len'] for s in stats]
    births = [s['births'] for s in stats]
    q_lens = [s['q_len'] for s in stats]

    # SS: use percentiles, not avg -- also include births
    #plot_metrics(timestamps, pop_sizes, avg_lens, "simulation_metrics.png")
    plot_metrics(timestamps, pop_sizes, births, q_lens, "simulation_metrics.png")

    snapshots = [s['snapshot'] for s in stats]
    save_grid_gif(snapshots, "evolution.gif", cfg)
