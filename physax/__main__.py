import jax
import jax.numpy as jnp
from jax import random
from physax.config import make_config
from physax.simulation import run_simulation
from physax.viz import plot_metrics, save_grid_gif, save_physis_view_gif
from physax.constants import PERCENTILES
# SS: print whether running on cpu or gpu
print('device:', jax.devices()[0].platform)

if __name__ == "__main__":
    # SS: varying pop_size, initial_pop, total_cycles, log_interval
    # other parameters may need to be changed also, in the Config class

    cfg = make_config(
        #pop_size=256,
        pop_size=4096,
        #initial_pop=1,
        initial_pop=10,
    )

    key = random.PRNGKey(42)
    pop, stats = run_simulation(
        key,
        cfg,
        #total_cycles=2000,
        total_cycles=10_000,
        #log_interval=50,
        log_interval=50,
        use_wandb=False,
    )

    print("\n=== FINAL STATE ===")
    alive = pop['alive']
    alive_count = jnp.sum(alive)
    # SS: use percentiles, not avg
    #avg_len = jnp.sum(jnp.where(alive, pop['genome_len'], 0)) / jnp.maximum(alive_count, 1)
    q_lens = jnp.nanpercentile(jnp.where(alive, pop['genome_len'], jnp.nan), PERCENTILES )

    print(f"Alive: {int(alive_count)}")
    # SS: use percentiles, not avg
    #print(f"Avg genome length: {float(avg_len):.1f}")
    print(f"min={q_lens[0]:.0f}, lq={q_lens[1]:.1f}, med={q_lens[2]:.0f}, uq={q_lens[3]:.0f}, max={q_lens[4]:.0f}")

    timestamps = [s['cycle'] for s in stats]
    pop_sizes = [s['pop_size'] for s in stats]
    # SS: use percentiles, not avg -- also include births
    #avg_lens = [s['avg_len'] for s in stats]
    births = [s['births'] for s in stats]
    q_lens = [s['q_len'] for s in stats]

    # SS: use percentiles, not avg -- also include births
    #plot_metrics(timestamps, pop_sizes, avg_lens, "simulation_metrics.png")
    plot_metrics(timestamps, pop_sizes, births, q_lens, "simulation_metrics.png")

    snapshots = [s['snapshot'] for s in stats]
    save_grid_gif(snapshots, "evolution.gif", cfg)
