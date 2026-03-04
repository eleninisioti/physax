#!/usr/bin/env python3
"""
Baseline comparison experiment: Java Physis vs Physax.

Runs both systems N_RUNS times on a 16x16 grid with 1 ancestor,
collects per-cycle statistics, and plots comparison with error bars.

Usage:
    source .venv/bin/activate
    export PATH="/usr/local/opt/openjdk/bin:$PATH"
    python3 run_experiment.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess
import tempfile
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from physax import make_config, init_population, cycle_step

# ── Experiment parameters ─────────────────────────────────────────────
N_RUNS = 20
N_CYCLES = 1000
GRID_W, GRID_H = 16, 16
POP_SIZE = GRID_W * GRID_H  # 256
INITIAL_POP = 1
STEPS_PER_UPDATE = 34
COPY_MUT = 0.009
DIV_INS = 0.0013
DIV_DEL = 0.0013

JAVA_PATH = "/usr/local/opt/openjdk/bin/java"
PHYSIS_DIR = Path(__file__).parent / "physis"
OUTPUT_DIR = Path(__file__).parent / "experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Java Physis runner ────────────────────────────────────────────────

def run_java_physis(run_id, seed):
    """Run one Java Physis simulation, return per-cycle stats dict."""
    physis_home = str(PHYSIS_DIR.resolve()) + "/"

    # Stats conf — place temp files inside physis dir structure
    tmp_stats_conf_name = f"_tmp_stats_{run_id}.conf"
    tmp_stats_conf_path = PHYSIS_DIR / "data" / "statistics" / tmp_stats_conf_name
    tmp_stats_data_name = f"_tmp_stats_{run_id}"
    tmp_stats_data_path = PHYSIS_DIR / "results" / tmp_stats_data_name

    with open(tmp_stats_conf_path, "w") as f:
        f.write("sample_rate=1\n")
        f.write("number_of_living_organisms=yes\nnumber_of_fertile_organisms=yes\n")
        f.write("maximum_genome_length=yes\naverage_genome_length=yes\n")
        for m in ["minimum_fitness", "maximum_fitness", "maximum_fitness_ever_reached",
                   "average_fitness", "minimum_merit", "maximum_merit",
                   "maximum_merit_ever_reached", "average_merit",
                   "maximum_age", "maximum_age_ever_reached", "average_age",
                   "maximum_genome_length_ever_reached",
                   "maximum_effective_length", "maximum_effective_length_ever_reached",
                   "average_effective_length"]:
            f.write(f"{m}=no\n")

    # Props file
    tmpdir = tempfile.mkdtemp(prefix=f"physis_run{run_id}_")
    props_file = os.path.join(tmpdir, "run.props")
    with open(props_file, "w") as f:
        f.write(f"physis_home={physis_home}\n")
        f.write(f"max_number_of_updates={N_CYCLES + 1}\n")
        f.write(f"random_seed={seed}\n")
        f.write("random_number_generator=JavaUtilRandom\n")
        f.write("seed_organism1=data/genebank/arche/arche.replicator\n")
        f.write("virtual_machine_class_name=arche.UP\n")
        f.write("instructionset=data/instructionsets/arche\n")
        f.write("nurse_class_name=physis.core.nursing.OldestNurse\n")
        f.write("lifespace_class_name=physis.core.lifespace.Lattice2DLifeSpace\n")
        f.write(f"2Dlattice_xsize={GRID_W}\n2Dlattice_ysize={GRID_H}\n")
        f.write("scheduler_class_name=physis.core.scheduler.ConstantScheduler\n")
        f.write(f"average_time_slice={STEPS_PER_UPDATE}\n")
        f.write(f"copy_mutation_rate={COPY_MUT}\n")
        f.write("divide_mutation_rate=0.0\n")
        f.write(f"divide_insert_rate={DIV_INS}\ndivide_delete_rate={DIV_DEL}\n")
        f.write("min_allocation_ratio=0.5\nmax_allocation_ratio=2\n")
        f.write("min_proliferation_ratio=0.80\n")
        f.write("gene_bank_enabled=false\n")
        f.write("expiration_time=100\nfossil_time=1600\n")
        f.write("spawn_threshold=0\ngene_bank_max_size=100000\n")
        f.write(f"gene_bank_output_file=results/_tmp_gb_{run_id}\n")
        f.write("task_filename=data/tasks/standard.tasks\n")
        f.write("max_number_of_tasks=20\ninput_data_higher_bound=13\n")
        f.write(f"statistics_configuration=data/statistics/{tmp_stats_conf_name}\n")
        f.write(f"statistics_data_file=results/{tmp_stats_data_name}\n")
        f.write("trigger_file=data/triggers/extract\n")

    env = os.environ.copy()
    env["PATH"] = "/usr/local/opt/openjdk/bin:" + env.get("PATH", "")
    result = subprocess.run(
        ["java", "-cp", str(PHYSIS_DIR / "classes"), "physis.core.PHYSIS", props_file],
        capture_output=True, text=True, env=env, timeout=300
    )

    if result.returncode != 0:
        print(f"  Java run {run_id} error: {result.stderr[:200]}", flush=True)

    stats = parse_java_stats(str(tmp_stats_data_path))

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmp_stats_conf_path.unlink(missing_ok=True)
    tmp_stats_data_path.unlink(missing_ok=True)
    # Cleanup genebank file if created
    gb_path = PHYSIS_DIR / "results" / f"_tmp_gb_{run_id}"
    gb_path.unlink(missing_ok=True)

    return stats


def parse_java_stats(filepath):
    """Parse tab-delimited Java Physis statistics file."""
    cycles, pop_sizes, avg_genome_lens, max_genome_lens, fertile_counts = [], [], [], [], []

    with open(filepath, "r") as f:
        header = f.readline().strip().split("\t")
        col_map = {name.strip(): i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < len(col_map):
                continue
            cycles.append(int(parts[col_map["update"]]))
            pop_sizes.append(int(parts[col_map["number_of_living_organisms"]]))
            fertile_counts.append(int(parts[col_map["number_of_fertile_organisms"]]))
            max_genome_lens.append(int(parts[col_map["maximum_genome_length"]]))
            avg_genome_lens.append(float(parts[col_map["average_genome_length"]]))

    pop_arr = np.array(pop_sizes)
    # Derive births from population changes (exact before grid fills, 0 after)
    births = np.diff(pop_arr, prepend=pop_arr[0])
    births = np.maximum(births, 0)

    return {
        "cycles": np.array(cycles),
        "pop_size": pop_arr,
        "fertile": np.array(fertile_counts),
        "avg_genome_len": np.array(avg_genome_lens),
        "max_genome_len": np.array(max_genome_lens),
        "births": births,
    }


# ── Physax runner (in-process, lax.scan batched, JIT compiles once) ───

def run_all_physax(n_runs, seeds):
    """Run all physax simulations in a single process.
    Uses lax.scan to batch all N_CYCLES on-device, then transfers once."""
    cfg = make_config(pop_size=POP_SIZE, initial_pop=INITIAL_POP)

    cycle_step_fn = partial(cycle_step, cfg)

    def scan_all_cycles(pop, keys):
        """Run N_CYCLES on device, collecting per-cycle stats as scan output."""
        def scan_body(pop, key):
            pop, stats = cycle_step_fn(pop, key)
            cycle_data = {
                'alive': pop['alive'],             # (pop_size,) bool
                'genome_len': pop['genome_len'],   # (pop_size,) int32
                'births': stats['births'],         # scalar
                'pop_size': stats['pop_size'],     # scalar
                'avg_genome_len': stats['avg_genome_len'],  # scalar
            }
            return pop, cycle_data
        _, all_data = jax.lax.scan(scan_body, pop, keys)
        return all_data

    jit_scan = jax.jit(scan_all_cycles)

    # Warmup JIT (compile with a full-size run)
    print("  Compiling JIT (warmup)...", flush=True)
    warmup_key = random.PRNGKey(0)
    wk1, wk2 = random.split(warmup_key)
    warmup_pop = init_population(wk1, cfg)
    warmup_keys = random.split(wk2, N_CYCLES)
    warmup_out = jit_scan(warmup_pop, warmup_keys)
    jax.block_until_ready(warmup_out)
    del warmup_out, warmup_pop
    print("  JIT compiled.", flush=True)

    all_runs = []
    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        print(f"  Physax run {run_idx+1}/{n_runs} (seed={seed})", flush=True)

        key = random.PRNGKey(seed)
        k1, k2 = random.split(key)
        pop = init_population(k1, cfg)
        cycle_keys = random.split(k2, N_CYCLES)

        # Run all cycles on device
        data = jit_scan(pop, cycle_keys)
        data = jax.block_until_ready(data)

        # Single host transfer
        alive_all = np.array(data['alive'])           # (N_CYCLES, POP_SIZE)
        genome_len_all = np.array(data['genome_len']) # (N_CYCLES, POP_SIZE)
        births_all = np.array(data['births'])         # (N_CYCLES,)
        pop_sizes = np.array(data['pop_size'])        # (N_CYCLES,)
        avg_gls = np.array(data['avg_genome_len'])    # (N_CYCLES,)
        del data

        # Compute genome length stats from full arrays (numpy, on host)
        max_gls = np.zeros(N_CYCLES)
        p10s = np.zeros(N_CYCLES)
        p25s = np.zeros(N_CYCLES)
        p50s = np.zeros(N_CYCLES)
        p75s = np.zeros(N_CYCLES)
        p90s = np.zeros(N_CYCLES)

        for i in range(N_CYCLES):
            alive_lens = genome_len_all[i][alive_all[i]]
            if len(alive_lens) > 0:
                max_gls[i] = np.max(alive_lens)
                p10s[i] = np.percentile(alive_lens, 10)
                p25s[i] = np.percentile(alive_lens, 25)
                p50s[i] = np.percentile(alive_lens, 50)
                p75s[i] = np.percentile(alive_lens, 75)
                p90s[i] = np.percentile(alive_lens, 90)

        all_runs.append({
            "cycles": np.arange(1, N_CYCLES + 1),
            "pop_size": pop_sizes,
            "avg_genome_len": avg_gls,
            "max_genome_len": max_gls,
            "births": births_all,
            "p10_genome_len": p10s,
            "p25_genome_len": p25s,
            "p50_genome_len": p50s,
            "p75_genome_len": p75s,
            "p90_genome_len": p90s,
        })

    return all_runs


# ── Plotting ──────────────────────────────────────────────────────────

def compute_stats_across_runs(all_runs, key):
    """Compute median and percentile bands across runs for a given metric."""
    mat = np.stack([r[key] for r in all_runs], axis=0)
    return {
        "median": np.median(mat, axis=0),
        "p25": np.percentile(mat, 25, axis=0),
        "p75": np.percentile(mat, 75, axis=0),
        "p10": np.percentile(mat, 10, axis=0),
        "p90": np.percentile(mat, 90, axis=0),
    }


def plot_comparison(java_runs, physax_runs):
    """Create comparison figure with 2 panels."""
    # Align on common cycle range
    java_len = min(len(r["cycles"]) for r in java_runs)
    physax_len = min(len(r["cycles"]) for r in physax_runs)
    common_len = min(java_len, physax_len)

    for runs in [java_runs, physax_runs]:
        for r in runs:
            for k in r:
                r[k] = r[k][:common_len]

    cycles = java_runs[0]["cycles"]

    java_pop = compute_stats_across_runs(java_runs, "pop_size")
    physax_pop = compute_stats_across_runs(physax_runs, "pop_size")
    java_avg_gl = compute_stats_across_runs(java_runs, "avg_genome_len")
    physax_avg_gl = compute_stats_across_runs(physax_runs, "avg_genome_len")
    java_max_gl = compute_stats_across_runs(java_runs, "max_genome_len")
    physax_max_gl = compute_stats_across_runs(physax_runs, "max_genome_len")
    physax_p10 = compute_stats_across_runs(physax_runs, "p10_genome_len")
    physax_p50 = compute_stats_across_runs(physax_runs, "p50_genome_len")
    physax_p90 = compute_stats_across_runs(physax_runs, "p90_genome_len")
    physax_births = compute_stats_across_runs(physax_runs, "births")
    java_births = compute_stats_across_runs(java_runs, "births")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── Panel 1: Population size + births ─────────────────────────────
    ax1 = axes[0]
    ax1.set_ylabel("Population Size", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1.plot(cycles, java_pop["median"], color="tab:blue", linewidth=2, label="Java pop")
    ax1.fill_between(cycles, java_pop["p25"], java_pop["p75"], alpha=0.2, color="tab:blue")
    ax1.fill_between(cycles, java_pop["p10"], java_pop["p90"], alpha=0.08, color="tab:blue")

    ax1.plot(cycles, physax_pop["median"], color="tab:cyan", linewidth=2,
             linestyle="--", label="Physax pop")
    ax1.fill_between(cycles, physax_pop["p25"], physax_pop["p75"], alpha=0.2, color="tab:cyan")
    ax1.fill_between(cycles, physax_pop["p10"], physax_pop["p90"], alpha=0.08, color="tab:cyan")

    # Births on secondary y axis
    ax1b = ax1.twinx()
    ax1b.set_ylabel("Births / Cycle (smoothed)", color="tab:orange")
    ax1b.tick_params(axis="y", labelcolor="tab:orange")

    window = 20
    kernel = np.ones(window) / window

    # Java births (derived from population growth)
    java_births_smooth = np.convolve(java_births["median"], kernel, mode="same")
    ax1b.plot(cycles, java_births_smooth, color="tab:orange", alpha=0.8,
              linewidth=1.5, label="Java births")

    # Physax births (actual count)
    physax_births_smooth = np.convolve(physax_births["median"], kernel, mode="same")
    ax1b.plot(cycles, physax_births_smooth, color="tab:red", alpha=0.7,
              linewidth=1.5, linestyle="--", label="Physax births")
    ax1b.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(
        f"Baseline Comparison: Java Physis vs Physax\n"
        f"({N_RUNS} runs, {GRID_W}x{GRID_H} grid, 1 ancestor, "
        f"copy_mut={COPY_MUT}, div_ins={DIV_INS})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ── Panel 2: Genome length percentiles ────────────────────────────
    ax2 = axes[1]
    ax2.set_ylabel("Genome Length")
    ax2.set_xlabel("Cycle")

    # Java: avg genome length (cross-run median + bands)
    ax2.plot(cycles, java_avg_gl["median"], color="tab:blue", linewidth=2,
             label="Java avg")
    ax2.fill_between(cycles, java_avg_gl["p25"], java_avg_gl["p75"],
                      alpha=0.15, color="tab:blue")

    # Java: max genome length
    ax2.plot(cycles, java_max_gl["median"], color="tab:blue", linewidth=1,
             linestyle=":", alpha=0.6, label="Java max")

    # Physax: avg genome length
    ax2.plot(cycles, physax_avg_gl["median"], color="tab:red", linewidth=2,
             linestyle="--", label="Physax avg")
    ax2.fill_between(cycles, physax_avg_gl["p25"], physax_avg_gl["p75"],
                      alpha=0.15, color="tab:red")

    # Physax: organism percentiles (cross-run medians)
    ax2.plot(cycles, physax_p50["median"], color="tab:cyan", linewidth=1.5,
             linestyle="--", label="Physax p50")
    ax2.fill_between(cycles, physax_p10["median"], physax_p90["median"],
                      alpha=0.1, color="tab:cyan", label="Physax p10-p90")

    # Physax: max genome length
    ax2.plot(cycles, physax_max_gl["median"], color="tab:red", linewidth=1,
             linestyle=":", alpha=0.6, label="Physax max")

    # Reference line
    ax2.axhline(y=78, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.text(cycles[-1] * 0.02, 78 + 1, "ancestor (78)", fontsize=8, color="gray")

    ax2.legend(loc="lower right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "baseline_comparison.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved plot to {outpath}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"=== Baseline Experiment: {N_RUNS} runs x {N_CYCLES} cycles ===", flush=True)
    print(f"Grid: {GRID_W}x{GRID_H} = {POP_SIZE} cells, {INITIAL_POP} initial organism(s)", flush=True)
    print(flush=True)

    seeds = [1000 + i * 137 for i in range(N_RUNS)]

    # ── Run Java Physis ───────────────────────────────────────────────
    print("Running Java Physis...", flush=True)
    java_runs = []
    for i in range(N_RUNS):
        print(f"  Java run {i+1}/{N_RUNS} (seed={seeds[i]})", flush=True)
        try:
            stats = run_java_physis(i, seeds[i])
            java_runs.append(stats)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"  Completed {len(java_runs)}/{N_RUNS} Java runs", flush=True)
    print(flush=True)

    # ── Run Physax (all in-process, JIT compiles once) ────────────────
    print("Running Physax...", flush=True)
    physax_runs = run_all_physax(N_RUNS, seeds)
    print(f"  Completed {len(physax_runs)}/{N_RUNS} Physax runs", flush=True)
    print(flush=True)

    # ── Save raw data ─────────────────────────────────────────────────
    np.savez(
        OUTPUT_DIR / "raw_data.npz",
        java_pop=np.stack([r["pop_size"] for r in java_runs]),
        java_births=np.stack([r["births"] for r in java_runs]),
        java_avg_gl=np.stack([r["avg_genome_len"] for r in java_runs]),
        java_max_gl=np.stack([r["max_genome_len"] for r in java_runs]),
        physax_pop=np.stack([r["pop_size"] for r in physax_runs]),
        physax_avg_gl=np.stack([r["avg_genome_len"] for r in physax_runs]),
        physax_max_gl=np.stack([r["max_genome_len"] for r in physax_runs]),
        physax_births=np.stack([r["births"] for r in physax_runs]),
        physax_p10=np.stack([r["p10_genome_len"] for r in physax_runs]),
        physax_p25=np.stack([r["p25_genome_len"] for r in physax_runs]),
        physax_p50=np.stack([r["p50_genome_len"] for r in physax_runs]),
        physax_p75=np.stack([r["p75_genome_len"] for r in physax_runs]),
        physax_p90=np.stack([r["p90_genome_len"] for r in physax_runs]),
        cycles=java_runs[0]["cycles"][:min(len(r["cycles"]) for r in java_runs)],
    )
    print(f"Saved raw data to {OUTPUT_DIR / 'raw_data.npz'}", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────
    plot_comparison(java_runs, physax_runs)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n=== Summary ===", flush=True)
    java_final_pop = [r["pop_size"][-1] for r in java_runs]
    physax_final_pop = [r["pop_size"][-1] for r in physax_runs]
    java_final_gl = [r["avg_genome_len"][-1] for r in java_runs]
    physax_final_gl = [r["avg_genome_len"][-1] for r in physax_runs]

    print(f"Final pop (cycle {N_CYCLES}):")
    print(f"  Java:   {np.mean(java_final_pop):.1f} +/- {np.std(java_final_pop):.1f}")
    print(f"  Physax: {np.mean(physax_final_pop):.1f} +/- {np.std(physax_final_pop):.1f}")
    print(f"Final avg genome len:")
    print(f"  Java:   {np.mean(java_final_gl):.1f} +/- {np.std(java_final_gl):.1f}")
    print(f"  Physax: {np.mean(physax_final_gl):.1f} +/- {np.std(physax_final_gl):.1f}")
    print(flush=True)


if __name__ == "__main__":
    main()
