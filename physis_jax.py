"""
Physis Minimal - JAX Version

Parallelized digital evolution simulation using JAX.
- Population runs in parallel via vmap
- Cycles via lax.scan with interleaved birth handling
- Fixed-size padded genomes for JAX compatibility
"""

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


# ==========================================
# 1. CONSTANTS (Gene values)
# ==========================================

R = 0
S = 1
B = 2
I = 3
SEP = 4

MOVE = 10
LOAD = 11
STORE = 12
JUMP = 20
IFZERO = 21
INC = 30
DEC = 31
ADD = 32
SUB = 33
ALLOCATE = 40
DIVIDE = 41
READ_SIZE = 50

NOP = -1
EMPTY = -1


# ==========================================
# 2. CONFIGURATION
# ==========================================

class Config:
    """Simulation configuration with fixed sizes for JAX."""
    max_genome_len: int = 100
    max_registers: int = 8
    max_instructions: int = 20
    max_ops_per_instr: int = 10
    pop_size: int = 1000
    initial_pop: int = 10
    max_age: int = 40000  # In cycles
    
    point_mutation_rate: float = 0.01
    indel_rate: float = 0.005
    

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
        'genome': jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
        'genome_len': jnp.int32(0),
        'registers': jnp.zeros(cfg.max_registers, dtype=jnp.int32),
        'ip': jnp.int32(0),
        'code_start': jnp.int32(0),
        'n_regs': jnp.int32(1),
        'n_instructions': jnp.int32(0),
        'age': jnp.int32(0),
        'alive': jnp.bool_(True),
        'has_child': jnp.bool_(False),
        'child_genome': jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
        'child_len': jnp.int32(0),
        'instr_ops': jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32),
        'instr_args': jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32),
        'instr_n_ops': jnp.zeros(cfg.max_instructions, dtype=jnp.int32),
    }


# ==========================================
# 4. GENOME PARSING
# ==========================================

def parse_genome(genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Parse genome into phenotype using vectorized operations."""
    
    # Create position indices
    positions = jnp.arange(cfg.max_genome_len)
    valid_pos = positions < genome_len
    
    # A. Count registers: Rs at start before B, I, or SEP
    is_R = (genome == R) & valid_pos
    is_marker = ((genome == B) | (genome == I) | (genome == SEP)) & valid_pos
    
    # Find first marker position
    marker_positions = jnp.where(is_marker, positions, cfg.max_genome_len)
    first_marker = jnp.min(marker_positions)
    
    # Count Rs before first marker
    n_regs = jnp.sum(is_R & (positions < first_marker))
    n_regs = jnp.maximum(n_regs, 1)
    
    # Language section starts after first marker (skip B if present)
    language_start = jnp.where(genome[first_marker] == B, first_marker + 1, first_marker)
    language_start = jnp.minimum(language_start, genome_len)
    
    # B. Find SEP position (end of instruction definitions)
    is_SEP = (genome == SEP) & valid_pos & (positions >= language_start)
    sep_positions = jnp.where(is_SEP, positions, cfg.max_genome_len)
    sep_pos = jnp.min(sep_positions)
    
    # C. Find instruction markers (I) between language_start and sep_pos
    is_I = (genome == I) & valid_pos & (positions >= language_start) & (positions < sep_pos)
    
    # Get positions of I markers
    I_positions = jnp.where(is_I, positions, cfg.max_genome_len)
    I_positions = jnp.sort(I_positions)  # Valid positions first
    
    # D. Parse each instruction definition
    # For simplicity, use a scan but only over max_instructions (small, e.g. 20)
    instr_ops = jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32)
    instr_args = jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32)
    instr_n_ops = jnp.zeros(cfg.max_instructions, dtype=jnp.int32)
    
    def parse_one_instruction(instr_idx):
        """Parse instruction at I_positions[instr_idx]."""
        start = I_positions[instr_idx]
        valid_instr = start < cfg.max_genome_len
        
        # Find end of this instruction (next I or SEP)
        next_markers = jnp.where(
            ((genome == I) | (genome == SEP)) & (positions > start) & valid_pos,
            positions,
            cfg.max_genome_len
        )
        end = jnp.min(next_markers)
        
        # Parse ops within this instruction
        ops_row = jnp.full(cfg.max_ops_per_instr, NOP, dtype=jnp.int32)
        args_row = jnp.zeros((cfg.max_ops_per_instr, 2), dtype=jnp.int32)
        
        def parse_op(carry, op_idx):
            ptr, ops_row, args_row, n_ops = carry
            
            at_end = (ptr >= end) | (ptr >= genome_len)
            gene = jnp.where(at_end, NOP, genome[ptr])
            
            is_2arg = (gene == MOVE) | (gene == LOAD) | (gene == STORE) | (gene == ADD) | (gene == SUB)
            is_1arg = (gene == READ_SIZE) | (gene == ALLOCATE) | (gene == INC) | (gene == DEC) | (gene == JUMP) | (gene == IFZERO)
            is_0arg = gene == DIVIDE
            is_op = is_2arg | is_1arg | is_0arg
            
            valid_op = is_op & ~at_end & (op_idx < cfg.max_ops_per_instr)
            
            ops_row = jnp.where(valid_op, ops_row.at[op_idx].set(gene), ops_row)
            
            arg0 = jnp.where((ptr + 1 < genome_len) & (is_1arg | is_2arg), genome[ptr + 1], 0)
            arg1 = jnp.where((ptr + 2 < genome_len) & is_2arg, genome[ptr + 2], 0)
            args_row = jnp.where(valid_op, args_row.at[op_idx, 0].set(arg0), args_row)
            args_row = jnp.where(valid_op, args_row.at[op_idx, 1].set(arg1), args_row)
            
            advance = jnp.where(is_2arg, 3, jnp.where(is_1arg, 2, jnp.where(is_0arg, 1, 1)))
            ptr = jnp.where(valid_op, ptr + advance, jnp.where(~at_end, ptr + 1, ptr))
            n_ops = jnp.where(valid_op, n_ops + 1, n_ops)
            
            return (ptr, ops_row, args_row, n_ops), None
        
        # Start parsing after the I marker
        init_ptr = jnp.where(valid_instr, start + 1, cfg.max_genome_len)
        (_, ops_row, args_row, n_ops), _ = lax.scan(
            parse_op,
            (init_ptr, ops_row, args_row, jnp.int32(0)),
            jnp.arange(cfg.max_ops_per_instr)
        )
        
        return ops_row, args_row, n_ops, valid_instr
    
    # Parse all instructions (vmap over instruction indices)
    all_ops, all_args, all_n_ops, all_valid = jax.vmap(parse_one_instruction)(
        jnp.arange(cfg.max_instructions)
    )
    
    # Only keep valid instructions
    instr_ops = jnp.where(all_valid[:, None], all_ops, instr_ops)
    instr_args = jnp.where(all_valid[:, None, None], all_args, instr_args)
    instr_n_ops = jnp.where(all_valid, all_n_ops, instr_n_ops)
    
    n_instructions = jnp.sum(all_valid.astype(jnp.int32))
    n_instructions = jnp.maximum(n_instructions, 1)
    
    # Code section starts after SEP
    code_start = jnp.where(sep_pos < genome_len, sep_pos + 1, genome_len)
    
    return {
        'n_regs': n_regs,
        'n_instructions': n_instructions,
        'code_start': code_start,
        'instr_ops': instr_ops,
        'instr_args': instr_args,
        'instr_n_ops': instr_n_ops,
    }


# ==========================================
# 5. VM EXECUTION (single step)
# ==========================================

def vm_step(state: dict, cfg: Config):
    """Execute one VM step for an organism."""
    genome = state['genome']
    genome_len = state['genome_len']
    registers = state['registers']
    ip = state['ip']
    code_start = state['code_start']
    n_regs = state['n_regs']
    n_instructions = state['n_instructions']
    instr_ops = state['instr_ops']
    instr_args = state['instr_args']
    child_genome = state['child_genome']
    child_len = state['child_len']
    has_child = state['has_child']
    
    code_len = genome_len - code_start
    valid_code = code_len > 0
    
    ip = jnp.where(
        valid_code & ((ip < code_start) | (ip >= genome_len)),
        code_start,
        ip
    )
    
    op_code = jnp.where(valid_code & (ip < genome_len), genome[ip], 0)
    ip = ip + 1
    ip = jnp.where(ip >= genome_len, code_start, ip)
    
    instr_idx = jnp.where(n_instructions > 0, op_code % n_instructions, 0)
    
    def exec_op(carry, op_idx):
        registers, child_genome, child_len, has_child, ip = carry
        
        op = instr_ops[instr_idx, op_idx]
        arg0 = instr_args[instr_idx, op_idx, 0]
        arg1 = instr_args[instr_idx, op_idx, 1]
        
        valid_op = op != NOP
        
        def get_reg(idx):
            return registers[idx % n_regs]
        
        def set_reg(idx, val):
            return registers.at[idx % n_regs].set(val)
        
        # READ_SIZE
        registers = jnp.where(
            (op == READ_SIZE) & valid_op, 
            set_reg(arg0, genome_len), 
            registers
        )
        
        # ALLOCATE
        is_allocate = (op == ALLOCATE)
        alloc_size = get_reg(arg0)
        valid_alloc = (alloc_size > 0) & (alloc_size < cfg.max_genome_len)
        child_len = jnp.where(is_allocate & valid_op & valid_alloc, alloc_size, child_len)
        child_genome = jnp.where(
            is_allocate & valid_op & valid_alloc,
            jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
            child_genome
        )
        
        # LOAD: registers[arg1] = genome[registers[arg0]]
        is_load = op == LOAD
        load_addr = get_reg(arg0)
        load_val = jnp.where((load_addr >= 0) & (load_addr < genome_len), genome[load_addr], 0)
        registers = jnp.where(is_load & valid_op, set_reg(arg1, load_val), registers)
        
        # STORE: child[registers[arg1]] = registers[arg0]
        is_store = op == STORE
        store_addr = get_reg(arg1)
        store_val = get_reg(arg0)
        valid_store = (child_len > 0) & (store_addr >= 0) & (store_addr < child_len)
        child_genome = jnp.where(
            is_store & valid_op & valid_store,
            child_genome.at[store_addr].set(store_val),
            child_genome
        )
        
        # MOVE: registers[arg1] = registers[arg0]
        registers = jnp.where(
            (op == MOVE) & valid_op, 
            set_reg(arg1, get_reg(arg0)), 
            registers
        )
        
        # INC: registers[arg0] += 1
        registers = jnp.where(
            (op == INC) & valid_op, 
            set_reg(arg0, get_reg(arg0) + 1), 
            registers
        )
        
        # DEC: registers[arg0] -= 1
        registers = jnp.where(
            (op == DEC) & valid_op, 
            set_reg(arg0, get_reg(arg0) - 1), 
            registers
        )
        
        # ADD: registers[arg0] += registers[arg1]
        registers = jnp.where(
            (op == ADD) & valid_op, 
            set_reg(arg0, get_reg(arg0) + get_reg(arg1)), 
            registers
        )
        
        # SUB: registers[arg0] -= registers[arg1]
        registers = jnp.where(
            (op == SUB) & valid_op, 
            set_reg(arg0, get_reg(arg0) - get_reg(arg1)), 
            registers
        )
        
        # IFZERO: skip next if register is zero
        is_ifzero = op == IFZERO
        skip = get_reg(arg0) == 0
        ip = jnp.where(is_ifzero & valid_op & skip, ip + 1, ip)
        ip = jnp.where(ip >= genome_len, code_start, ip)
        
        # JUMP
        is_jump = op == JUMP
        jump_target = jnp.where(code_len > 0, code_start + (arg0 % code_len), code_start)
        ip = jnp.where(is_jump & valid_op, jump_target, ip)
        
        # DIVIDE: give birth
        is_divide = op == DIVIDE
        valid_divide = child_len > 0
        has_child = jnp.where(is_divide & valid_op & valid_divide, True, has_child)
        
        return (registers, child_genome, child_len, has_child, ip), None
    
    (registers, child_genome, child_len, has_child, ip), _ = lax.scan(
        exec_op,
        (registers, child_genome, child_len, has_child, ip),
        jnp.arange(cfg.max_ops_per_instr)
    )
    
    new_state = state.copy()
    new_state['registers'] = registers
    new_state['ip'] = ip
    new_state['child_genome'] = child_genome
    new_state['child_len'] = child_len
    new_state['has_child'] = has_child
    
    return new_state


# ==========================================
# 6. MUTATION
# ==========================================

def mutate_genome(key: jax.Array, genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Apply mutations to a genome."""
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # Point mutation
    do_point = random.uniform(k1) < cfg.point_mutation_rate
    point_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    point_val = random.randint(k3, (), 0, 61)
    genome = jnp.where(
        do_point & (point_idx < genome_len),
        genome.at[point_idx].set(point_val),
        genome
    )
    
    # Indel mutation
    do_indel = random.uniform(k4) < cfg.indel_rate
    do_insert = random.uniform(k5) < 0.5
    indel_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    insert_val = random.randint(k3, (), 0, 61)
    
    def do_insertion(args):
        genome, genome_len = args
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(indices > indel_idx, genome[indices - 1], genome[indices])
        shifted = shifted.at[indel_idx].set(insert_val)
        new_len = jnp.minimum(genome_len + 1, cfg.max_genome_len)
        return shifted, new_len
    
    def do_deletion(args):
        genome, genome_len = args
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(
            indices >= indel_idx, 
            jnp.where(indices < cfg.max_genome_len - 1, genome[indices + 1], EMPTY),
            genome[indices]
        )
        new_len = jnp.maximum(genome_len - 1, 5)
        return shifted, new_len
    
    genome, genome_len = lax.cond(
        do_indel,
        lambda args: lax.cond(
            do_insert & (args[1] < cfg.max_genome_len - 1), 
            do_insertion, 
            lambda a: lax.cond(args[1] > 5, do_deletion, lambda x: x, a),
            args
        ),
        lambda args: args,
        (genome, genome_len)
    )
    
    return genome, genome_len


# ==========================================
# 7. POPULATION INITIALIZATION
# ==========================================

def create_ancestor_genome(cfg: Config):
    """Create the ancestor genome."""
    g = []
    # Hardware: 4 registers
    g += [R, R, R, R, B]
    
    # Instructions
    g += [I, READ_SIZE, 1]                           # I0: R1 = size
    g += [I, ALLOCATE, 1]                            # I1: allocate R1 bytes
    g += [I, LOAD, 2, 0, STORE, 0, 2, INC, 2]        # I2: copy loop body
    g += [I, MOVE, 1, 3, SUB, 3, 2]                  # I3: R3 = R1 - R2
    g += [I, IFZERO, 3]                              # I4: skip if done
    g += [I, JUMP, 2]                                # I5: loop back
    g += [I, DIVIDE]                                 # I6: birth
    g += [SEP]
    
    # Code
    g += [0, 1, 2, 3, 4, 5, 6]
    
    genome = jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32)
    genome = genome.at[:len(g)].set(jnp.array(g, dtype=jnp.int32))
    genome_len = jnp.int32(len(g))
    
    return genome, genome_len


def init_organism(genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Initialize an organism from a genome."""
    state = create_organism_state(cfg)
    state['genome'] = genome
    state['genome_len'] = genome_len
    
    parsed = parse_genome(genome, genome_len, cfg)
    state['n_regs'] = parsed['n_regs']
    state['n_instructions'] = parsed['n_instructions']
    state['code_start'] = parsed['code_start']
    state['instr_ops'] = parsed['instr_ops']
    state['instr_args'] = parsed['instr_args']
    state['instr_n_ops'] = parsed['instr_n_ops']
    state['ip'] = parsed['code_start']
    
    return state


def init_population(key: jax.Array, cfg: Config):
    """Initialize population with ancestor genomes."""
    ancestor_genome, ancestor_len = create_ancestor_genome(cfg)
    
    def init_one(i):
        state = init_organism(ancestor_genome, ancestor_len, cfg)
        state['alive'] = i < cfg.initial_pop
        return state
    
    pop = jax.vmap(init_one)(jnp.arange(cfg.pop_size))
    return pop


# ==========================================
# 8. CYCLE STEP (single cycle for all organisms)
# ==========================================

def cycle_step(cfg: Config, pop: dict, key: jax.Array):
    """Execute one cycle: step all organisms, handle births."""
    
    # Step all alive organisms that haven't reproduced yet
    def step_one(state):
        should_step = state['alive'] & ~state['has_child']
        return lax.cond(
            should_step,
            lambda s: vm_step(s, cfg),
            lambda s: s,
            state
        )
    
    pop = jax.vmap(step_one)(pop)
    
    # Age all alive organisms
    pop['age'] = jnp.where(pop['alive'], pop['age'] + 1, pop['age'])
    
    # Death by age
    too_old = pop['age'] >= cfg.max_age
    pop['alive'] = pop['alive'] & ~too_old
    
    # Collect births
    has_child = pop['has_child']
    n_births = jnp.sum(has_child)
    
    # Mutate children
    mut_keys = random.split(key, cfg.pop_size)
    
    def mutate_one(args):
        key, genome, length, has = args
        return lax.cond(
            has,
            lambda g: mutate_genome(key, g[0], g[1], cfg),
            lambda g: (g[0], g[1]),
            (genome, length)
        )
    
    mutated_genomes, mutated_lens = jax.vmap(mutate_one)(
        (mut_keys, pop['child_genome'], pop['child_len'], has_child)
    )
    
    # Fully vectorized child placement - O(1) parallel on GPU (O(log n) for cumsum)
    alive_mask = pop['alive']
    
    # Compute ranks using cumsum
    empty_rank = jnp.cumsum(~alive_mask) - 1  # "I am the Nth empty slot"
    child_rank = jnp.cumsum(has_child) - 1     # "I am the Nth child"
    
    n_empty = jnp.sum(~alive_mask)
    n_children = jnp.sum(has_child)
    n_to_assign = jnp.minimum(n_empty, n_children)
    
    # Build inverse lookup: child_at_rank[r] = parent_index with child_rank == r
    parent_indices = jnp.arange(cfg.pop_size)
    child_at_rank = jnp.zeros(cfg.pop_size, dtype=jnp.int32)
    child_at_rank = child_at_rank.at[child_rank].set(
        jnp.where(has_child, parent_indices, 0)
    )
    
    # For each slot: check if receives child and from which parent
    receives_child = (~alive_mask) & (empty_rank < n_to_assign)
    parent_for_slot = child_at_rank[empty_rank]
    
    # Gather genomes from parents - O(1) parallel
    slot_genomes = mutated_genomes[parent_for_slot]
    slot_lens = mutated_lens[parent_for_slot]
    
    # Parse ALL genomes in parallel - O(1) on GPU with vmap
    all_parsed = jax.vmap(lambda g, l: parse_genome(g, l, cfg))(slot_genomes, slot_lens)
    
    # Build new organism states in parallel
    def build_state(genome, genome_len, parsed):
        state = create_organism_state(cfg)
        state['genome'] = genome
        state['genome_len'] = genome_len
        state['n_regs'] = parsed['n_regs']
        state['n_instructions'] = parsed['n_instructions']
        state['code_start'] = parsed['code_start']
        state['instr_ops'] = parsed['instr_ops']
        state['instr_args'] = parsed['instr_args']
        state['instr_n_ops'] = parsed['instr_n_ops']
        state['ip'] = parsed['code_start']
        return state
    
    new_states = jax.vmap(build_state)(slot_genomes, slot_lens, all_parsed)
    
    # Merge: where receives_child use new_state, else keep old
    def merge_field(old, new):
        rc = receives_child
        for _ in range(old.ndim - 1):
            rc = rc[..., None]
        return jnp.where(rc, new, old)
    
    pop = jax.tree.map(merge_field, pop, new_states)
    
    # Reset child buffers for parents who gave birth
    pop['has_child'] = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    pop['child_len'] = jnp.where(has_child, jnp.int32(0), pop['child_len'])
    
    # Stats
    alive_count = jnp.sum(pop['alive'])
    avg_genome_len = jnp.sum(jnp.where(pop['alive'], pop['genome_len'], 0)) / jnp.maximum(alive_count, 1)
    
    stats = {
        'pop_size': alive_count,
        'births': n_births,
        'avg_genome_len': avg_genome_len,
    }
    
    return pop, stats


# ==========================================
# 9. MAIN SIMULATION
# ==========================================

def run_simulation(key: jax.Array, cfg: Config, total_cycles: int, 
                   log_interval: int = 10000, use_wandb: bool = False):
    """Run the simulation for total_cycles."""
    print(f"=== JAX PHYSIS SIMULATION ===")
    print(f"Population capacity: {cfg.pop_size}, Initial: {cfg.initial_pop}")
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
                    "max_age": cfg.max_age,
                    "point_mutation_rate": cfg.point_mutation_rate,
                    "indel_rate": cfg.indel_rate,
                }
            )
    
    # Initialize
    k1, k2 = random.split(key)
    pop = init_population(k1, cfg)
    
    # JIT compile the cycle step
    cycle_step_fn = partial(cycle_step, cfg)
    
    def scan_cycles(pop, keys):
        def step(pop, key):
            pop, stats = cycle_step_fn(pop, key)
            return pop, stats
        return lax.scan(step, pop, keys)
    
    jit_scan = jax.jit(scan_cycles)
    
    # Run in chunks for logging
    n_chunks = total_cycles // log_interval
    all_stats = []
    
    cycle_keys = random.split(k2, total_cycles)
    
    for chunk in trange(n_chunks, desc="Running"):
        start = chunk * log_interval
        end = (chunk + 1) * log_interval
        chunk_keys = cycle_keys[start:end]
        
        pop, stats = jit_scan(pop, chunk_keys)
        
        # Log last stats of chunk
        cycle_num = end
        pop_size = int(stats['pop_size'][-1])
        births = int(jnp.sum(stats['births']))
        avg_len = float(stats['avg_genome_len'][-1])
        
        print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, AvgLen={avg_len:.1f}")
        
        if use_wandb:
            wandb.log({
                "cycle": cycle_num,
                "population/size": pop_size,
                "population/births_interval": births,
                "genome/avg_len": avg_len,
            })
        
        all_stats.append(jax.tree.map(lambda x: np.array(x), stats))
    
    if use_wandb:
        wandb.finish()
    
    return pop, all_stats


# ==========================================
# 10. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    cfg = make_config(
        pop_size=1000,
        initial_pop=10,
        max_age=40000,
        point_mutation_rate=0.02,
        indel_rate=0.01,
    )
    
    key = random.PRNGKey(42)
    pop, stats = run_simulation(
        key, 
        cfg, 
        total_cycles=500_000,
        log_interval=10000,
        use_wandb=False,
    )
    
    print("\n=== FINAL STATE ===")
    alive = pop['alive']
    print(f"Final population: {int(jnp.sum(alive))}")
    print(f"Avg genome length: {float(jnp.mean(jnp.where(alive, pop['genome_len'], 0))):.1f}")
