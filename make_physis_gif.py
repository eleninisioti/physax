#!/usr/bin/env python3
"""
Generate an animated GIF of a single Java Physis run (16x16 grid, 1000 cycles).

Creates a two-panel animation:
  - Left: 16x16 grid where cells fill left-to-right, top-to-bottom as population grows
  - Right: Population size and avg genome length curves building up over time

Usage:
    source .venv/bin/activate
    export PATH="/usr/local/opt/openjdk/bin:$PATH"
    python3 make_physis_gif.py
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import imageio.v2 as imageio

# ── Configuration ────────────────────────────────────────────────────────────
N_CYCLES = 1000
GRID_W, GRID_H = 16, 16
POP_SIZE = GRID_W * GRID_H  # 256
SEED = 42
STEPS_PER_UPDATE = 34
COPY_MUT = 0.009
DIV_INS = 0.0013
DIV_DEL = 0.0013
FRAME_EVERY = 10  # capture a frame every N cycles
FPS = 10

JAVA_PATH = "/usr/local/opt/openjdk/bin/java"
PHYSIS_DIR = Path(__file__).parent / "physis"
OUTPUT_DIR = Path(__file__).parent / "experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "physis_run.gif"


# ── Run Java Physis ──────────────────────────────────────────────────────────

def run_java_physis():
    """Run Java Physis and return per-cycle stats."""
    physis_home = str(PHYSIS_DIR.resolve()) + "/"
    run_id = "gif"

    # Stats configuration
    stats_conf_path = PHYSIS_DIR / "data" / "statistics" / f"_tmp_stats_{run_id}.conf"
    stats_data_path = PHYSIS_DIR / "results" / f"_tmp_stats_{run_id}"

    with open(stats_conf_path, "w") as f:
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
    tmpdir = tempfile.mkdtemp(prefix="physis_gif_")
    props_file = os.path.join(tmpdir, "run.props")
    with open(props_file, "w") as f:
        f.write(f"physis_home={physis_home}\n")
        f.write(f"max_number_of_updates={N_CYCLES + 1}\n")
        f.write(f"random_seed={SEED}\n")
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
        f.write(f"statistics_configuration=data/statistics/_tmp_stats_{run_id}.conf\n")
        f.write(f"statistics_data_file=results/_tmp_stats_{run_id}\n")
        f.write("trigger_file=data/triggers/extract\n")

    print("Running Java Physis...", flush=True)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/opt/openjdk/bin:" + env.get("PATH", "")
    result = subprocess.run(
        ["java", "-cp", str(PHYSIS_DIR / "classes"), "physis.core.PHYSIS", props_file],
        capture_output=True, text=True, env=env, timeout=300
    )
    if result.returncode != 0:
        print(f"Java stderr: {result.stderr[:500]}", flush=True)

    # Parse stats
    stats = parse_stats(str(stats_data_path))
    print(f"Parsed {len(stats['cycles'])} cycles of data.", flush=True)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    stats_conf_path.unlink(missing_ok=True)
    stats_data_path.unlink(missing_ok=True)
    (PHYSIS_DIR / "results" / f"_tmp_gb_{run_id}").unlink(missing_ok=True)

    return stats


def parse_stats(filepath):
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
    return {
        "cycles": np.array(cycles),
        "pop_size": np.array(pop_sizes),
        "fertile": np.array(fertile_counts),
        "avg_genome_len": np.array(avg_genome_lens),
        "max_genome_len": np.array(max_genome_lens),
    }


# ── GIF Generation ──────────────────────────────────────────────────────────

def make_gif(stats):
    """Create animated GIF from per-cycle statistics."""
    cycles = stats["cycles"]
    pop_size = stats["pop_size"]
    avg_gl = stats["avg_genome_len"]
    max_gl = stats["max_genome_len"]

    # Determine which cycles to render as frames
    frame_indices = list(range(0, len(cycles), FRAME_EVERY))
    if frame_indices[-1] != len(cycles) - 1:
        frame_indices.append(len(cycles) - 1)

    # Precompute a stable RNG-based cell fill order so the grid fills deterministically
    rng = np.random.RandomState(123)
    fill_order = np.arange(POP_SIZE)
    rng.shuffle(fill_order)

    # Color map for the grid: base color is a teal/blue, genome length shifts hue
    base_color = np.array([0.15, 0.65, 0.85])  # HSV-ish teal

    frames = []
    print(f"Generating {len(frame_indices)} frames...", flush=True)

    for fi, idx in enumerate(frame_indices):
        cyc = cycles[idx]
        n_alive = pop_size[idx]
        agl = avg_gl[idx]
        mgl = max_gl[idx]

        fig, (ax_grid, ax_chart) = plt.subplots(
            1, 2, figsize=(10, 4.5),
            gridspec_kw={"width_ratios": [1, 1.5]}
        )

        # ── Left panel: 16x16 grid ──────────────────────────────────────
        grid = np.zeros((GRID_H, GRID_W, 3))  # RGB, black = empty

        # Fill cells according to fill_order
        alive_cells = fill_order[:n_alive]
        for cell_idx in alive_cells:
            row = cell_idx // GRID_W
            col = cell_idx % GRID_W
            # Slight color variation based on position for visual interest
            h = (0.55 + 0.002 * cell_idx) % 1.0
            s = 0.7
            v = 0.85
            r, g, b = mcolors.hsv_to_rgb([h, s, v])
            grid[row, col] = [r, g, b]

        ax_grid.imshow(grid, interpolation="nearest", aspect="equal")
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        # Draw gridlines
        for x in range(GRID_W + 1):
            ax_grid.axvline(x - 0.5, color="gray", linewidth=0.3, alpha=0.5)
        for y in range(GRID_H + 1):
            ax_grid.axhline(y - 0.5, color="gray", linewidth=0.3, alpha=0.5)

        ax_grid.set_title(f"16x16 Grid  |  Cycle {cyc}", fontsize=11, fontweight="bold")
        ax_grid.text(
            0.5, -0.08,
            f"Pop: {n_alive}/{POP_SIZE}   Avg GL: {agl:.1f}   Max GL: {mgl:.0f}",
            transform=ax_grid.transAxes, ha="center", fontsize=9, color="#333333"
        )

        # ── Right panel: time series charts ──────────────────────────────
        t = cycles[:idx + 1]

        # Population size (left y-axis)
        color_pop = "#2176AE"
        ax_chart.set_xlabel("Cycle", fontsize=10)
        ax_chart.set_ylabel("Population Size", color=color_pop, fontsize=10)
        ax_chart.plot(t, pop_size[:idx + 1], color=color_pop, linewidth=2, label="Population")
        ax_chart.tick_params(axis="y", labelcolor=color_pop)
        ax_chart.set_xlim(0, N_CYCLES)
        ax_chart.set_ylim(0, POP_SIZE + 10)
        ax_chart.axhline(y=POP_SIZE, color=color_pop, linestyle=":", alpha=0.3, linewidth=0.8)

        # Avg genome length (right y-axis)
        ax2 = ax_chart.twinx()
        color_gl = "#D7263D"
        ax2.set_ylabel("Genome Length", color=color_gl, fontsize=10)
        ax2.plot(t, avg_gl[:idx + 1], color=color_gl, linewidth=2,
                 linestyle="--", label="Avg Genome Len")
        ax2.plot(t, max_gl[:idx + 1], color="#F46036", linewidth=1,
                 linestyle=":", alpha=0.6, label="Max Genome Len")
        ax2.tick_params(axis="y", labelcolor=color_gl)
        # Set genome length y-axis range based on full data
        gl_max = max(np.max(max_gl), 100)
        ax2.set_ylim(0, gl_max * 1.1)
        ax2.axhline(y=78, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax2.text(N_CYCLES * 0.02, 78 + gl_max * 0.015, "ancestor (78)",
                 fontsize=7, color="gray")

        # Combined legend
        lines1, labels1 = ax_chart.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_chart.legend(lines1 + lines2, labels1 + labels2,
                        loc="center right", fontsize=8)

        ax_chart.set_title("Java Physis - 16x16 Grid", fontsize=11, fontweight="bold")
        ax_chart.grid(True, alpha=0.2)

        fig.tight_layout()

        # Render frame to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())  # drop alpha channel
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  Frame {fi + 1}/{len(frame_indices)}", flush=True)

    print(f"Saving GIF to {OUTPUT_FILE} ({len(frames)} frames, {FPS} fps)...", flush=True)
    imageio.mimsave(str(OUTPUT_FILE), frames, fps=FPS, loop=0)
    print("Done.", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats = run_java_physis()
    make_gif(stats)
