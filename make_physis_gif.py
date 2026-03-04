#!/usr/bin/env python3
"""
Generate an animated GIF of a single Java Physis run (16x16 grid, 1000 cycles).

Uses GridDumper to export per-cell data, then renders with the same
Physis-spectrum 3-panel visualization as make_gif.py.

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
FRAME_EVERY = 10  # dump interval in cycles
FPS = 10

# View mode: "all" (3-panel Fitness|Merit|Age), "fitness", "merit", "age"
VIEW_MODE = "all"

JAVA_PATH = "/usr/local/opt/openjdk/bin/java"
PHYSIS_DIR = Path(__file__).parent / "physis"
OUTPUT_DIR = Path(__file__).parent / "experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "physis_run.gif"

# Physis color spectrum (matching ColorRange.java)
PHYSIS_SPECTRUM = [
    (0, 0, 96/255),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (1, 128/255, 128/255)
]
physis_cmap = mcolors.LinearSegmentedColormap.from_list('physis', PHYSIS_SPECTRUM, N=128)


# ── Run Java GridDumper ──────────────────────────────────────────────────────

def run_grid_dumper():
    """Run Java GridDumper and return parsed per-cell snapshots."""
    physis_home = str(PHYSIS_DIR.resolve()) + "/"

    tmpdir = tempfile.mkdtemp(prefix="physis_gif_")
    props_file = os.path.join(tmpdir, "run.props")
    dump_tsv = os.path.join(tmpdir, "grid_dump.tsv")

    # Stats config (needed by StatisticsFactory even though we don't use aggregate stats)
    run_id = "gif"
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

    print("Running Java GridDumper...", flush=True)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/opt/openjdk/bin:" + env.get("PATH", "")
    result = subprocess.run(
        [JAVA_PATH, "-cp", str(PHYSIS_DIR / "classes"),
         "physis.core.GridDumper", props_file, dump_tsv, str(FRAME_EVERY)],
        capture_output=True, text=True, env=env, timeout=300
    )
    if result.returncode != 0:
        print(f"Java stderr: {result.stderr[:1000]}", flush=True)
        raise RuntimeError("GridDumper failed")

    # Parse TSV
    snapshots = parse_grid_dump(dump_tsv)
    print(f"Parsed {len(snapshots)} grid snapshots.", flush=True)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    stats_conf_path.unlink(missing_ok=True)
    stats_data_path.unlink(missing_ok=True)
    (PHYSIS_DIR / "results" / f"_tmp_gb_{run_id}").unlink(missing_ok=True)

    return snapshots


def parse_grid_dump(filepath):
    """Parse GridDumper TSV output into snapshot dicts."""
    data = {}  # cycle -> list of rows
    with open(filepath, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            cycle = int(parts[0])
            if cycle not in data:
                data[cycle] = []
            data[cycle].append(parts)

    snapshots = []
    for cycle in sorted(data.keys()):
        rows = data[cycle]
        alive = np.zeros(POP_SIZE, dtype=bool)
        fertile = np.zeros(POP_SIZE, dtype=bool)
        age = np.zeros(POP_SIZE, dtype=np.float64)
        fitness = np.zeros(POP_SIZE, dtype=np.float64)
        merit = np.zeros(POP_SIZE, dtype=np.float64)
        genome_len = np.zeros(POP_SIZE, dtype=np.int32)
        effective_len = np.zeros(POP_SIZE, dtype=np.int32)

        for parts in rows:
            x, y = int(parts[1]), int(parts[2])
            idx = x * GRID_H + y  # match Java grid indexing: x is outer, y is inner
            alive[idx] = int(parts[3]) == 1
            fertile[idx] = int(parts[4]) == 1
            age[idx] = int(parts[5])
            fitness[idx] = float(parts[6])
            merit[idx] = int(parts[7])
            genome_len[idx] = int(parts[8])
            effective_len[idx] = int(parts[9])

        snapshots.append({
            'cycle': cycle,
            'alive': alive,
            'fertile': fertile,
            'age': age,
            'fitness': fitness,
            'merit': merit,
            'genome_len': genome_len,
            'effective_len': effective_len,
        })

    return snapshots


# ── GIF Generation ──────────────────────────────────────────────────────────

def make_gif(snapshots):
    """Create animated GIF from per-cell grid snapshots using Physis spectrum."""
    if VIEW_MODE == 'all':
        views = ['fitness', 'merit', 'age']
    else:
        views = [VIEW_MODE]

    # Compute running max for BY_MAX_EVER_REACHED normalization
    max_ever = {v: 0.0 for v in views}
    for snap in snapshots:
        for v in views:
            vals = snap[v][snap['alive']] if np.any(snap['alive']) else np.array([0.0])
            if len(vals) > 0:
                max_ever[v] = max(max_ever[v], float(np.max(vals)))
    for v in views:
        max_ever[v] = max(max_ever[v], 1.0)

    frames = []
    n_views = len(views)
    figw = 5 * n_views + 0.5
    figh = 5.0

    print(f"Generating {len(snapshots)} frames...", flush=True)

    for fi, snap in enumerate(snapshots):
        fig, axes = plt.subplots(1, n_views, figsize=(figw, figh), squeeze=False)
        axes = axes[0]

        alive = snap['alive']
        fertile = snap['fertile']
        pop_count = int(np.sum(alive))

        for vi, view_name in enumerate(views):
            ax = axes[vi]
            vals = snap[view_name]

            normed = vals / max_ever[view_name]
            normed = np.clip(normed, 0, 1)

            rgba = physis_cmap(normed)
            rgb = rgba[:, :3]

            # Dead: black; alive but not fertile: dark gray
            rgb[~alive] = [0.0, 0.0, 0.0]
            rgb[alive & ~fertile] = [64/255, 64/255, 64/255]

            grid_img = rgb.reshape(GRID_W, GRID_H, 3)

            ax.imshow(grid_img, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for gx in range(GRID_W + 1):
                ax.axvline(gx - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            for gy in range(GRID_H + 1):
                ax.axhline(gy - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax.set_title(view_name.capitalize(), fontsize=11, fontweight='bold')

        fig.suptitle(f"Java Physis  |  Cycle {snap['cycle']}  |  Pop: {pop_count}/{POP_SIZE}",
                     fontsize=10, y=0.02)
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  Frame {fi + 1}/{len(snapshots)}", flush=True)

    print(f"Saving GIF to {OUTPUT_FILE} ({len(frames)} frames, {FPS} fps)...", flush=True)
    imageio.mimsave(str(OUTPUT_FILE), frames, fps=FPS, loop=0)
    print("Done.", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    snapshots = run_grid_dumper()
    make_gif(snapshots)
