import random

import numpy as np

from organism import Gene, Organism


# ==========================================
# 5. WORLD (Environment)
# ==========================================

class World:
    def __init__(self, mode='dynamic', pop_size=100, cpu_cycles=200, initial_pop=10):
        self.mode = mode
        self.pop_size = pop_size
        self.cpu_cycles = cpu_cycles
        ancestor = create_static_ancestor() if mode == 'static' else create_dynamic_ancestor()
        self.population = [Organism(ancestor.copy(), mode=mode) for _ in range(initial_pop)]

        # Stats tracking
        self.births_this_epoch = 0
        self.deaths_this_epoch = 0
        self.total_births = 0
        self.total_deaths = 0

    def mutate(self, genome):
        g = genome.copy()
        rate = 0.01
        if random.random() < rate:
            g[random.randint(0, len(g)-1)] = random.randint(0, 60)

        if random.random() < 0.005:
            idx = random.randint(0, len(g)-1)
            if random.random() < 0.5 and len(g) > 5:
                g = np.delete(g, idx)
            else:
                g = np.insert(g, idx, random.randint(0, 60))
        return g

    def step(self):
        new_babies = []
        for org in self.population:
            for _ in range(self.cpu_cycles):
                baby = org.step()
                if baby:
                    mutated_g = self.mutate(baby.genome)
                    viable_baby = Organism(mutated_g, mode=self.mode)
                    new_babies.append(viable_baby)
                    break
            org.age += 1

        # Track deaths (aged out)
        prev_count = len(self.population)
        survivors = [o for o in self.population if o.age < 80]
        aged_deaths = prev_count - len(survivors)

        # Track births
        self.births_this_epoch = len(new_babies)
        self.total_births += self.births_this_epoch

        self.population = survivors + new_babies

        # Track culled organisms (population cap)
        culled = 0
        if len(self.population) > self.pop_size:
            culled = len(self.population) - self.pop_size
            random.shuffle(self.population)
            self.population = self.population[:self.pop_size]

        self.deaths_this_epoch = aged_deaths + culled
        self.total_deaths += self.deaths_this_epoch

    def get_avg_len(self):
        if not self.population: return 0
        return np.mean([len(o.genome) for o in self.population])


# ==========================================
# 4. ANCESTOR GENOMES
# ==========================================

def create_static_ancestor():
    """Standard 'Tierra-style' ancestor."""
    g = [
        Gene.READ_SIZE, 1,
        Gene.ALLOCATE, 1,
        Gene.LOAD, 2, 0,
        Gene.STORE, 0, 2,
        Gene.INC, 2,
        Gene.SUB, 3, 1, 2,
        Gene.IFZERO, 3,
        Gene.JUMP, 4,
        Gene.DIVIDE
    ]
    return np.array(g, dtype=int)


def create_dynamic_ancestor():
    """'Physis-style' ancestor with definitions + code.

    Uses 4 registers:
    - R0: temp for copy
    - R1: genome size
    - R2: copy index (0, 1, 2, ...)
    - R3: remaining = size - copied
    """
    g = []
    # Hardware: 4 registers
    g += [Gene.R, Gene.R, Gene.R, Gene.R, Gene.B]

    # Instructions (each starts with I marker)
    g += [Gene.I, Gene.READ_SIZE, 1]                              # I0: R1 = size
    g += [Gene.I, Gene.ALLOCATE, 1]                               # I1: allocate R1 bytes
    g += [Gene.I, Gene.LOAD, 2, 0, Gene.STORE, 0, 2, Gene.INC, 2] # I2: R0=genome[R2]; child[R2]=R0; R2++
    g += [Gene.I, Gene.MOVE, 1, 3, Gene.SUB, 3, 2]                # I3: R3=R1; R3=R3-R2 (remaining)
    g += [Gene.I, Gene.IFZERO, 3]                                 # I4: skip next if R3==0
    g += [Gene.I, Gene.JUMP, 2]                                   # I5: jump to I2 (loop)
    g += [Gene.I, Gene.DIVIDE]                                    # I6: divide (give birth)
    g += [Gene.SEP]

    # Code: call instructions 0,1,2,3,4,5,2,3,4,5,... until divide
    # Simple linear sequence: 0,1,2,3,4,5,6
    g += [0, 1, 2, 3, 4, 5, 6]

    return np.array(g, dtype=int)
