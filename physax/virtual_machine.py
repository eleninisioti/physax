import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from physax.config import *


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
