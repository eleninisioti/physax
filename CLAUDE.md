# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physax (Physis JAX) is a GPU-accelerated digital evolution simulation built with JAX. It simulates self-replicating digital organisms whose genomes encode both the instruction set (language) and the program (code) — a "dynamic phenotype" where evolution can modify both the program and the hardware it runs on. JAX port of the original [Physis](https://codeberg.org/egri-nagy/physis).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main simulation
python physax.py

# Debug single organism replication (400 steps, prints state each step)
python debug_repro.py

# Verify population simulation (600 cycles on a 4x4 grid)
python verify_sim.py
```

No test framework, linter, or formal build system is configured.

## Architecture

The entire simulation lives in a single file: **`physax.py`** (~973 lines), organized into 11 numbered sections:

1. **Constants** — Gene value instruction set alphabet (R, S, B, I, SEP, MOVE, LOAD, STORE, JUMP, IFZERO, INC, DEC, ADD, SUB, ALLOCATE, DIVIDE, READ_SIZE). Sentinel: `NOP = EMPTY = -1`.
2. **Configuration** — `Config` class + `make_config()` factory. Note: the `__main__` entry point overrides several defaults (e.g., `pop_size=4096`, `max_age=inf`, higher mutation rates).
3. **Organism State** — `create_organism_state()` returns a dict of JAX arrays (genome, registers, ip, alive, child_genome, color, etc.). Population is a struct-of-arrays with shape `(pop_size, ...)`.
4. **Genome Parsing** — `parse_genome()` extracts phenotype (register count, instruction table, code section) from a genome array. Uses `jax.vmap` + `lax.scan`.
5. **VM Execution** — `vm_step()` executes one virtual machine step. `DIVIDE` triggers reproduction.
6. **Mutation** — `mutate_genome()` applies point mutations, indels, and HSV color drift.
7. **Population Init** — `create_ancestor_genome()` builds the seed genome. `init_population()` places initial organisms on the grid.
8. **Cycle Step** — `cycle_step()` is the core per-timestep function: step all organisms, age/death, mutate children, spatial reproduction on a 2D toroidal grid (8-connected neighborhood).
9. **Visualization** — Metrics plots and animated GIF generation.
10. **Main Simulation** — `run_simulation()` uses `lax.scan` in JIT-compiled chunks with optional W&B logging.
11. **Entry Point** — `__main__` block with production config.

Supporting files: `debug_repro.py` and `verify_sim.py` import from `physax` as a module for manual verification. `explain_code.md` and `genomes.md` contain detailed documentation.

## Critical JAX Constraints

All functions participating in `jax.jit` or `jax.vmap` must:
- **No Python control flow** in traced code — use `lax.cond`, `lax.scan`, `jnp.where` instead of `if`/`for`
- **Fixed-size arrays only** — no dynamic shapes; use padded arrays with length tracking
- **No side effects** — no print, no mutation of external state inside traced functions
- Pattern: write single-organism logic, then `jax.vmap` over the population

## Key Design Patterns

- **Organism state as plain dicts** of JAX arrays (not registered pytrees or dataclasses)
- **`lax.scan` for loops** — both within-instruction operation sequences and multi-cycle simulation chunks
- **Spatial grid reproduction** — population array indexed as 2D toroidal grid; offspring placed in empty or oldest neighbor cells
- **Color-based species tracking** — HSV colors drift with mutations for visual lineage tracking
- **Optional deps** — `wandb` and `imageio` imported with try/except, degrade gracefully

## Environment Notes

- `physax.py` sets `CUDA_VISIBLE_DEVICES=0` at import time; falls back to CPU on non-CUDA systems
- JAX version pinned to 0.4.28
