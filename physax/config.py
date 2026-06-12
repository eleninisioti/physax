import jax.numpy as jnp
import jax
from jax import random
from typing import NamedTuple, Any

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

OP_NAMES = {
    0: 'NOP', 1: 'IN', 2: 'OUT', 3: 'LOAD', 4: 'STORE', 5: 'MOVE', 6: 'ALLOCATE', 
    7: 'COMPARE', 8: 'IFZERO', 9: 'JUMP', 10: 'DEC', 11: 'INC', 12: 'DIVIDE', 
    13: 'SDIR', 14: 'GDIR', 15: 'SEND', 16: 'RECEIVE', 17: 'ADD', 18: 'SUB', 
    19: 'MUL', 20: 'DIV_OP', 21: 'MOD', 22: 'AND', 23: 'OR', 24: 'XOR', 25: 'NEG', 
    26: 'NOT', 27: 'SHIFT_L', 28: 'SHIFT_R', 29: 'FORK_TH', 30: 'KILL_TH', 
    31: 'R', 32: 'S', 33: 'Q', 34: 'I', 35: 'B', 36: 'SEP', 37: 'CLEAR', 
    38: 'CINC', 39: 'CDEC', 40: 'IS_SEP', 41: 'REL_LOAD', 42: 'REL_STORE', 
    43: 'IFNOTZERO'
}

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


class VMContext(NamedTuple):
    """Context holding all execution state during a compound instruction."""
    se_vals: jnp.ndarray
    child_arr: jnp.ndarray
    child_cop: jnp.ndarray
    genome_arr: jnp.ndarray
    already_alloc: jnp.ndarray
    child_l: jnp.ndarray
    ip_for_overflow: jnp.ndarray
    pos_in_instr: jnp.ndarray
    next_opcode_pos: jnp.ndarray
    divide_returned: jnp.ndarray
    cntr: jnp.ndarray
    gest_time: jnp.ndarray
    has_ch: jnp.ndarray
    did_jump: jnp.ndarray

    step_key: jnp.ndarray
    genome_len: jnp.ndarray
    n_ses: jnp.ndarray
    tape_size: jnp.ndarray
    o0: jnp.ndarray
    o1: jnp.ndarray
    o2: jnp.ndarray


def se_read(ctx: VMContext, cfg: Config, idx):
    return ctx.se_vals[jnp.clip(idx, 0, cfg.max_se_count - 1)]

def se_write(ctx: VMContext, cfg: Config, idx, val):
    return ctx._replace(
        se_vals=ctx.se_vals.at[jnp.clip(idx, 0, cfg.max_se_count - 1)].set(val)
    )

def tape_read(ctx: VMContext, position):
    total_size = jnp.maximum(ctx.tape_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < ctx.genome_len
    parent_val = ctx.genome_arr[jnp.clip(pos, 0, ctx.genome_len - 1)]
    child_idx = jnp.clip(pos - ctx.genome_len, 0, ctx.child_l - 1)
    child_val = ctx.child_arr[child_idx]
    return jnp.where(in_parent, parent_val, child_val)

def tape_write(ctx: VMContext, cfg: Config, position, value):
    total_size = jnp.maximum(ctx.tape_size, 1)
    pos = jnp.abs(position) % total_size
    in_parent = pos < ctx.genome_len

    # Parent write
    parent_idx = jnp.clip(pos, 0, ctx.genome_len - 1)
    new_genome = jnp.where(in_parent, ctx.genome_arr.at[parent_idx].set(value), ctx.genome_arr)

    # Child write
    child_idx = jnp.clip(pos - ctx.genome_len, 0, ctx.child_l - 1)
    in_child = ~in_parent & ctx.already_alloc

    # Copy mutation: replace with random gene
    k1, k2 = random.split(ctx.step_key)
    do_mutate = random.uniform(k1) < cfg.copy_mutation_rate
    mutated_value = random.randint(k2, (), 0, UP_IS_SIZE).astype(jnp.int32)
    final_value = jnp.where(do_mutate & in_child, mutated_value, value)

    new_child = jnp.where(in_child, ctx.child_arr.at[child_idx].set(final_value), ctx.child_arr)
    new_child_cop = jnp.where(in_child, ctx.child_cop.at[child_idx].set(True), ctx.child_cop)

    return ctx._replace(genome_arr=new_genome, child_arr=new_child, child_cop=new_child_cop)


def get_opcode_functions(cfg: Config):
    """Returns a list of 44 pure functions mapping ctx -> ctx for jax.lax.switch."""
    
    def op_nop(ctx: VMContext) -> VMContext:
        """No operation."""
        return ctx

    def op_load(ctx: VMContext) -> VMContext:
        """Load from tape at address SE[o0] into SE[o1]."""
        val = tape_read(ctx, se_read(ctx, cfg, ctx.o0))
        return se_write(ctx, cfg, ctx.o1, val)

    def op_store(ctx: VMContext) -> VMContext:
        """Store value from SE[o1] into tape at address SE[o0]."""
        return tape_write(ctx, cfg, se_read(ctx, cfg, ctx.o0), se_read(ctx, cfg, ctx.o1))

    def op_move(ctx: VMContext) -> VMContext:
        """Copy value from SE[o0] into SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, se_read(ctx, cfg, ctx.o0))

    def op_allocate(ctx: VMContext) -> VMContext:
        """Allocate child tape of size SE[o0]."""
        alloc_size = se_read(ctx, cfg, ctx.o0)
        alloc_possible = (
            ~ctx.already_alloc &
            (alloc_size > (cfg.min_allocation_ratio * ctx.genome_len).astype(jnp.int32)) &
            (alloc_size < (cfg.max_allocation_ratio * ctx.genome_len).astype(jnp.int32))
        )
        blank_child = jnp.full(cfg.max_genome_len, BLANK, dtype=jnp.int32)
        blank_cop = jnp.zeros(cfg.max_genome_len, dtype=jnp.bool_)
        
        return ctx._replace(
            already_alloc=jnp.where(alloc_possible, True, ctx.already_alloc),
            child_l=jnp.where(alloc_possible, alloc_size, ctx.child_l),
            child_arr=jnp.where(alloc_possible, blank_child, ctx.child_arr),
            child_cop=jnp.where(alloc_possible, blank_cop, ctx.child_cop)
        )

    def op_compare(ctx: VMContext) -> VMContext:
        """Compare SE[o0] and SE[o1], write result (-1, 0, 1) into SE[o2]."""
        a, b = se_read(ctx, cfg, ctx.o0), se_read(ctx, cfg, ctx.o1)
        res = jnp.where(a < b, -1, jnp.where(a == b, 0, 1))
        return se_write(ctx, cfg, ctx.o2, res)

    def op_ifzero(ctx: VMContext) -> VMContext:
        """If SE[o0] is NOT zero, skip the next instruction (IP += 1)."""
        skip = se_read(ctx, cfg, ctx.o0) != 0
        return se_write(ctx, cfg, jnp.int32(0), jnp.where(skip, se_read(ctx, cfg, jnp.int32(0)) + 1, se_read(ctx, cfg, jnp.int32(0))))

    def op_jump(ctx: VMContext) -> VMContext:
        """Jump IP to the value stored in SE[o0]."""
        return se_write(ctx, cfg, jnp.int32(0), se_read(ctx, cfg, ctx.o0))._replace(did_jump=jnp.bool_(True))

    def op_dec(ctx: VMContext) -> VMContext:
        """Decrement SE[o0] by 1."""
        return se_write(ctx, cfg, ctx.o0, se_read(ctx, cfg, ctx.o0) - 1)

    def op_inc(ctx: VMContext) -> VMContext:
        """Increment SE[o0] by 1."""
        return se_write(ctx, cfg, ctx.o0, se_read(ctx, cfg, ctx.o0) + 1)

    def op_divide(ctx: VMContext) -> VMContext:
        """Trigger cell division if copying thresholds are met."""
        n_copied = jnp.sum(ctx.child_cop.astype(jnp.int32))
        prolif_possible = ctx.already_alloc & (n_copied > (cfg.min_proliferation_ratio * ctx.genome_len).astype(jnp.int32))
        
        new_se_vals = jnp.where(
            ~prolif_possible, 
            ctx.se_vals.at[jnp.clip(jnp.int32(0), 0, cfg.max_se_count - 1)].set(se_read(ctx, cfg, jnp.int32(0)) + 1), 
            ctx.se_vals
        )
        return ctx._replace(
            gest_time=jnp.where(prolif_possible, ctx.cntr, ctx.gest_time),
            has_ch=jnp.where(prolif_possible, True, ctx.has_ch),
            divide_returned=jnp.bool_(True),
            se_vals=new_se_vals
        )

    def op_add(ctx: VMContext) -> VMContext:
        """Add SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) + se_read(ctx, cfg, ctx.o1))

    def op_sub(ctx: VMContext) -> VMContext:
        """Subtract SE[o1] from SE[o0], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) - se_read(ctx, cfg, ctx.o1))

    def op_mul(ctx: VMContext) -> VMContext:
        """Multiply SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) * se_read(ctx, cfg, ctx.o1))

    def op_div(ctx: VMContext) -> VMContext:
        """Divide SE[o0] by SE[o1], store in SE[o2]."""
        div = se_read(ctx, cfg, ctx.o1)
        safe_div = jnp.where(div == 0, 1, div)
        res = se_read(ctx, cfg, ctx.o0) // safe_div
        new_ctx = se_write(ctx, cfg, ctx.o2, res)
        return jax.tree.map(lambda n, o: jnp.where(div != 0, n, o), new_ctx, ctx)

    def op_mod(ctx: VMContext) -> VMContext:
        """Modulo SE[o0] by SE[o1], store in SE[o2]."""
        div = se_read(ctx, cfg, ctx.o1)
        safe_div = jnp.where(div == 0, 1, div)
        res = se_read(ctx, cfg, ctx.o0) % safe_div
        new_ctx = se_write(ctx, cfg, ctx.o2, res)
        return jax.tree.map(lambda n, o: jnp.where(div != 0, n, o), new_ctx, ctx)

    def op_and(ctx: VMContext) -> VMContext:
        """Bitwise AND of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) & se_read(ctx, cfg, ctx.o1))

    def op_or(ctx: VMContext) -> VMContext:
        """Bitwise OR of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) | se_read(ctx, cfg, ctx.o1))

    def op_xor(ctx: VMContext) -> VMContext:
        """Bitwise XOR of SE[o0] and SE[o1], store in SE[o2]."""
        return se_write(ctx, cfg, ctx.o2, se_read(ctx, cfg, ctx.o0) ^ se_read(ctx, cfg, ctx.o1))

    def op_neg(ctx: VMContext) -> VMContext:
        """Negate SE[o0], store in SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, -se_read(ctx, cfg, ctx.o0))

    def op_not(ctx: VMContext) -> VMContext:
        """Bitwise NOT of SE[o0], store in SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, ~se_read(ctx, cfg, ctx.o0))

    def op_shift_l(ctx: VMContext) -> VMContext:
        """Bitwise left shift SE[o0] by 1, store in SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, se_read(ctx, cfg, ctx.o0) << 1)

    def op_shift_r(ctx: VMContext) -> VMContext:
        """Bitwise right shift SE[o0] by 1, store in SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, se_read(ctx, cfg, ctx.o0) >> 1)

    def op_clear(ctx: VMContext) -> VMContext:
        """Set SE[o0] to 0."""
        return se_write(ctx, cfg, ctx.o0, 0)

    def op_cinc(ctx: VMContext) -> VMContext:
        """Circular increment SE[o0] (wraps around at tape_size)."""
        return se_write(ctx, cfg, ctx.o0, (se_read(ctx, cfg, ctx.o0) + 1) % jnp.maximum(ctx.tape_size, 1))

    def op_cdec(ctx: VMContext) -> VMContext:
        """Circular decrement SE[o0] (wraps around at tape_size)."""
        val = se_read(ctx, cfg, ctx.o0) - 1
        sz = jnp.maximum(ctx.tape_size, 1)
        return se_write(ctx, cfg, ctx.o0, jnp.where(val < 0, sz - 1, val))

    def op_is_sep(ctx: VMContext) -> VMContext:
        """Check if tape[SE[o0]] == SEP token. Store 1 (True) or 0 (False) in SE[o1]."""
        return se_write(ctx, cfg, ctx.o1, jnp.where(se_read(ctx, cfg, ctx.o0) == SEP, jnp.int32(1), jnp.int32(0)))

    def op_rel_load(ctx: VMContext) -> VMContext:
        """Load from tape at (SE[o0] + SE[o1]) into SE[o2]."""
        addr = se_read(ctx, cfg, ctx.o0) + se_read(ctx, cfg, ctx.o1)
        return se_write(ctx, cfg, ctx.o2, tape_read(ctx, addr))

    def op_rel_store(ctx: VMContext) -> VMContext:
        """Store SE[o2] to tape at (SE[o0] + SE[o1])."""
        addr = se_read(ctx, cfg, ctx.o0) + se_read(ctx, cfg, ctx.o1)
        return tape_write(ctx, cfg, addr, se_read(ctx, cfg, ctx.o2))

    def op_ifnotzero(ctx: VMContext) -> VMContext:
        """If SE[o0] IS zero, skip the next instruction (IP += 1)."""
        skip = se_read(ctx, cfg, ctx.o0) == 0
        return se_write(ctx, cfg, jnp.int32(0), jnp.where(skip, se_read(ctx, cfg, jnp.int32(0)) + 1, se_read(ctx, cfg, jnp.int32(0))))

    return [
        op_nop, op_nop, op_nop, op_load, op_store, op_move, op_allocate, op_compare,
        op_ifzero, op_jump, op_dec, op_inc, op_divide, op_nop, op_nop, op_nop, op_nop,
        op_add, op_sub, op_mul, op_div, op_mod, op_and, op_or, op_xor, op_neg, op_not,
        op_shift_l, op_shift_r, op_nop, op_nop, op_nop, op_nop, op_nop, op_nop, op_nop,
        op_nop, op_clear, op_cinc, op_cdec, op_is_sep, op_rel_load, op_rel_store, op_ifnotzero
    ]
