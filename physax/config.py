
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


