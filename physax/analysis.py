import numpy as np

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
#
