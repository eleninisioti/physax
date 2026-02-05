import enum
from collections import deque

import numpy as np


# ==========================================
# 1. THE PHYSICS (Atomic Laws)
# ==========================================

class Gene(enum.IntEnum):
    # Structural (Only used in Dynamic Mode)
    R = 0
    S = 1
    B = 2
    I = 3
    SEP = 4

    # Atomic Operations (The "Assembler")
    # These are the fundamental laws of the universe
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


# ==========================================
# 2. THE PHENOTYPE (The Machine)
# ==========================================


class Instruction:
    """A macro-instruction composed of atomic ops."""

    def __init__(self, ops=None, args=None):
        self.ops = ops if ops is not None else []
        self.args = args if args is not None else []


class Phenotype:
    """The hardware spec."""

    def __init__(self, static_mode=False):
        self.n_regs = 0
        self.n_stacks = 0
        self.instructions = []
        self.code_start = 0

        if static_mode:
            self._build_static_standard_library()

    def _build_static_standard_library(self):
        """
        For Static Mode: Creates a 1-to-1 mapping where every Atomic Op
        is directly available as an instruction.
        """
        self.n_regs = 4  # Fixed hardware
        self.n_stacks = 1

        for i in range(60):
            new_instr = Instruction()
            if i in [m.value for m in Gene]:
                arity = 1
                if i in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB]: arity = 2
                if i in [Gene.DIVIDE, Gene.READ_SIZE, Gene.JUMP, Gene.IFZERO]: arity = 1
                new_instr.ops.append(i)
            self.instructions.append(new_instr)


# ==========================================
# 3. THE ORGANISM (Virtual Machine)
# ==========================================

class Organism:
    def __init__(self, genome: np.ndarray, mode='dynamic'):
        self.genome = genome
        self.mode = mode
        self.ip = 0
        self.child_buffer = None
        self.age = 0
        self.registers = []
        self.stacks = []

        # Build Phenotype
        if self.mode == 'static':
            self.phenotype = Phenotype(static_mode=True)
            self.phenotype.code_start = 0
        else:
            self.phenotype = self._build_dynamic_phenotype()

        self._init_memory()
        self.ip = self.phenotype.code_start

    def _build_dynamic_phenotype(self) -> Phenotype:
        p = Phenotype()
        ptr = 0
        limit = len(self.genome)

        # A. Define Structure
        while ptr < limit:
            g = self.genome[ptr]
            if g == Gene.R:
                p.n_regs += 1
            elif g == Gene.S:
                p.n_stacks += 1
            elif g == Gene.B:
                ptr += 1; break
            elif g in [Gene.I, Gene.SEP]:
                break
            ptr += 1

        if p.n_regs == 0: p.n_regs = 1

        # B. Define Instructions
        while ptr < limit:
            if self.genome[ptr] == Gene.SEP:
                ptr += 1;
                break

            if self.genome[ptr] == Gene.I:
                new_instr = Instruction()
                ptr += 1
                while ptr < limit:
                    atom = self.genome[ptr]
                    if atom in [Gene.I, Gene.SEP]: break

                    # Determine arity for each atomic operation
                    if atom in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB]:
                        arity = 2
                    elif atom in [Gene.READ_SIZE, Gene.ALLOCATE, Gene.INC, Gene.DEC, Gene.JUMP, Gene.IFZERO]:
                        arity = 1
                    elif atom == Gene.DIVIDE:
                        arity = 0
                    else:
                        arity = 1  # default

                    args = []
                    ptr += 1
                    for _ in range(arity):
                        args.append(self.genome[ptr] if ptr < limit else 0)
                        ptr += 1

                    new_instr.ops.append(atom)
                    new_instr.args.append(args)

                p.instructions.append(new_instr)
            else:
                ptr += 1

        p.code_start = ptr
        return p

    def _init_memory(self):
        self.registers = [0] * self.phenotype.n_regs
        self.stacks = [deque(maxlen=16) for _ in range(self.phenotype.n_stacks)]

    def _get(self, idx):
        total = self.phenotype.n_regs + self.phenotype.n_stacks
        if total == 0: return 0
        target = idx % total
        if target < self.phenotype.n_regs: return self.registers[target]
        s_idx = target - self.phenotype.n_regs
        return self.stacks[s_idx][-1] if self.stacks[s_idx] else 0

    def _set(self, idx, val):
        total = self.phenotype.n_regs + self.phenotype.n_stacks
        if total == 0: return
        target = idx % total
        if target < self.phenotype.n_regs:
            self.registers[target] = val
        else:
            s_idx = target - self.phenotype.n_regs
            self.stacks[s_idx].append(val)

    def step(self):
        if not self.phenotype.instructions: return None

        # Wrap IP to code section if out of bounds
        code_len = len(self.genome) - self.phenotype.code_start
        if code_len <= 0: return None

        if self.ip < self.phenotype.code_start or self.ip >= len(self.genome):
            self.ip = self.phenotype.code_start

        op_code = self.genome[self.ip]
        self.ip += 1

        # Wrap after increment
        if self.ip >= len(self.genome):
            self.ip = self.phenotype.code_start

        if self.mode == 'static':
            instr_idx = op_code % len(self.phenotype.instructions)
            instr = self.phenotype.instructions[instr_idx]
        else:
            if len(self.phenotype.instructions) == 0: return None
            instr_idx = op_code % len(self.phenotype.instructions)
            instr = self.phenotype.instructions[instr_idx]

        ops_to_run = instr.ops
        baked_args = instr.args

        for i, atom in enumerate(ops_to_run):
            current_args = []

            if self.mode == 'dynamic':
                current_args = baked_args[i]
            else:
                arity = 2 if atom in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB] else 1
                if atom in [Gene.DIVIDE, Gene.READ_SIZE, Gene.JUMP, Gene.IFZERO]:
                    arity = 1 if atom != Gene.DIVIDE and atom != Gene.READ_SIZE else 0

                for _ in range(arity):
                    if self.ip < len(self.genome):
                        current_args.append(self.genome[self.ip])
                        self.ip += 1
                    else:
                        current_args.append(0)

            # --- ATOMIC LOGIC ---
            if atom == Gene.READ_SIZE:
                dest = current_args[0] if current_args else 0
                self._set(dest, len(self.genome))

            elif atom == Gene.ALLOCATE:
                sz = self._get(current_args[0])
                if 0 < sz < 300: self.child_buffer = np.zeros(sz, dtype=int)

            elif atom == Gene.LOAD:
                addr = self._get(current_args[0])
                val = self.genome[addr] if 0 <= addr < len(self.genome) else 0
                self._set(current_args[1], val)

            elif atom == Gene.STORE:
                if self.child_buffer is not None:
                    addr = self._get(current_args[1])
                    if 0 <= addr < len(self.child_buffer):
                        self.child_buffer[addr] = self._get(current_args[0])

            elif atom == Gene.MOVE:
                self._set(current_args[1], self._get(current_args[0]))

            elif atom == Gene.INC:
                self._set(current_args[0], self._get(current_args[0]) + 1)

            elif atom == Gene.SUB:
                self._set(current_args[0], self._get(current_args[0]) - self._get(current_args[1]))

            elif atom == Gene.IFZERO:
                if self._get(current_args[0]) == 0:
                    self.ip += 1

            elif atom == Gene.JUMP:
                target = current_args[0]
                if self.mode == 'static':
                    self.ip = target % len(self.genome)
                else:
                    code_len = len(self.genome) - self.phenotype.code_start
                    if code_len > 0:
                        self.ip = self.phenotype.code_start + (target % code_len)

            elif atom == Gene.DIVIDE:
                if self.child_buffer is not None:
                    baby = Organism(self.child_buffer, mode=self.mode)
                    self.child_buffer = None
                    return baby

        return None
