import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
from PIL import Image
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from physax.config import LS, PERCENTILES
from physax.analysis import compute_snapshot_properties


def plot_metrics(timestamps, pop_sizes, births, q_lens, filename="metrics.png"):
    """Plot population size and average genome length over time."""

    # SS: wider figure for longer runs
    #fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(20, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    # SS: plot births as well as pop size
    #ax1.set_ylabel('Population Size', color=color)
    ax1.set_ylabel('Population Size / ...Births', color=color)
    ax1.plot(timestamps, pop_sizes, color=color, label='Pop Size')
    # SS: plot births
    ax1.plot(timestamps, births, color=color, ls='dotted', lw=0.7, label='Births')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'

    # SS: plot percentiles, not avg
    # ax2.set_ylabel('Avg Genome Length', color=color)
    ax2.set_ylabel('Percentile Genome Lengths', color=color)
    #ax2.plot(timestamps, avg_lens, color=color, linestyle='--', label='Avg Len')
    for i, ls in enumerate(LS): # quartiles/percentiles
        qv = [t[i] for t in q_lens]
        ax2.plot(timestamps, qv, color=color, linestyle=ls, lw=0.8)

    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Simulation Metrics')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved metrics plot to {filename}")


def save_grid_gif(snapshots, filename, cfg):
    """Generate a GIF of the 2D grid representation."""
    print("Generating GIF...")
    frames = []
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    max_len = cfg.max_genome_len

    for i, snap in enumerate(snapshots):
        alive_mask = snap['alive']
        genome_lens = snap['genome_len']
        pad_size = grid_side * grid_side - cfg.pop_size

        alive_grid = np.pad(alive_mask, (0, pad_size), constant_values=False).reshape(grid_side, grid_side)

        if cfg.use_species_color and 'color' in snap:
            colors = snap['color']
            colors_padded = np.pad(colors, ((0, pad_size), (0, 0)), constant_values=0.0)
            hsv_grid = colors_padded.reshape(grid_side, grid_side, 3)
            rgb = mcolors.hsv_to_rgb(hsv_grid)
        else:
            len_grid = np.pad(genome_lens, (0, pad_size), constant_values=0).reshape(grid_side, grid_side)
            norm_len = np.clip(len_grid / max_len, 0, 1)
            cmap = plt.get_cmap('viridis')
            rgba = cmap(norm_len)
            rgb = rgba[..., :3]

        mask = alive_grid[..., None]
        final_img = np.where(mask, rgb, 0.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(final_img, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"Cycle {snap['cycle']}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)

        frames.append(imageio.imread(buf) if imageio else np.array(Image.open(buf)))

    if imageio:
        imageio.mimsave(filename, frames, fps=10)
        print(f"Saved GIF to {filename}")
    else:
        print("imageio not installed, cannot save GIF.")


# Physis color spectrum (matching ColorRange.java)
# Anchor points: dark blue → blue → cyan → green → yellow → red → pink
PHYSIS_SPECTRUM = [
    (0, 0, 96/255),      # dark blue
    (0, 0, 1),            # blue
    (0, 1, 1),            # cyan
    (0, 1, 0),            # green
    (1, 1, 0),            # yellow
    (1, 0, 0),            # red
    (1, 128/255, 128/255) # pink
]
physis_cmap = mcolors.LinearSegmentedColormap.from_list('physis', PHYSIS_SPECTRUM, N=128)


def save_physis_view_gif(snapshots, filename, cfg, view_mode='all'):
    """Generate a GIF using Physis-style property-based coloring.

    view_mode: 'fitness', 'merit', 'age', 'all' (3-panel), or 'species' (HSV lineage).
    Snapshots must include 'alive', 'genome_len', 'executed', 'gestation_time', 'age'.
    """
    if not imageio:
        print("imageio not installed, cannot save GIF.")
        return

    print(f"Generating physis-view GIF (mode={view_mode})...")
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    pad_size = grid_side * grid_side - cfg.pop_size

    # Backward-compatible: species mode uses existing HSV colors
    if view_mode == 'species':
        save_grid_gif(snapshots, filename, cfg)
        return

    # Compute properties for all snapshots
    props_list = []
    for snap in snapshots:
        eff_len, merit, fitness, fertile = compute_snapshot_properties(snap, cfg.max_genome_len)
        props_list.append({
            'effective_length': eff_len,
            'merit': merit,
            'fitness': fitness,
            'fertile': fertile,
            'age': snap['age'].astype(np.float64),
            'alive': snap['alive'],
            'cycle': snap['cycle'],
        })

    # Determine which views to render
    if view_mode == 'all':
        views = ['fitness', 'merit', 'age']
    else:
        views = [view_mode]

    # Compute running max for BY_MAX_EVER_REACHED normalization
    max_ever = {v: 0.0 for v in views}
    for p in props_list:
        for v in views:
            vals = p[v][p['alive']] if np.any(p['alive']) else np.array([0.0])
            if len(vals) > 0:
                max_ever[v] = max(max_ever[v], float(np.max(vals)))
    # Ensure non-zero
    for v in views:
        max_ever[v] = max(max_ever[v], 1.0)

    frames = []
    n_views = len(views)
    figw = 5 * n_views + 0.5
    figh = 5.0

    for pi, p in enumerate(props_list):
        fig, axes = plt.subplots(1, n_views, figsize=(figw, figh), squeeze=False)
        axes = axes[0]

        alive = p['alive']
        fertile = p['fertile']
        pop_count = int(np.sum(alive))

        for vi, view_name in enumerate(views):
            ax = axes[vi]
            vals = p[view_name]

            # Normalize to [0, 1] by max-ever
            normed = vals / max_ever[view_name]
            normed = np.clip(normed, 0, 1)

            # Build RGB grid
            # Map through physis colormap
            rgba = physis_cmap(normed)
            rgb = rgba[:, :3]  # (pop_size, 3)

            # Dead: black; alive but not fertile: dark gray
            dead_mask = ~alive
            newborn_mask = alive & ~fertile
            rgb[dead_mask] = [0.0, 0.0, 0.0]
            rgb[newborn_mask] = [64/255, 64/255, 64/255]

            # Reshape to grid
            rgb_padded = np.pad(rgb, ((0, pad_size), (0, 0)), constant_values=0.0)
            grid_img = rgb_padded.reshape(grid_side, grid_side, 3)

            ax.imshow(grid_img, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            # Gridlines
            for gx in range(grid_side + 1):
                ax.axvline(gx - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            for gy in range(grid_side + 1):
                ax.axhline(gy - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax.set_title(view_name.capitalize(), fontsize=11, fontweight='bold')

        fig.suptitle(f"Cycle {p['cycle']}  |  Pop: {pop_count}/{cfg.pop_size}",
                     fontsize=10, y=0.02)
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf).convert('RGB')))

        if (pi + 1) % 20 == 0:
            print(f"  Frame {pi + 1}/{len(props_list)}")

    imageio.mimsave(filename, frames, fps=10)
    print(f"Saved physis-view GIF to {filename}")



