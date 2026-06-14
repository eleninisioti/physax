import jax
import jax.numpy as jnp
from jax import random
from physax.config import make_config, PERCENTILES, UNCLASSIFIED
from physax.model import Model
from physax.agent import Agent
from physax.visualization import plot_metrics, save_grid_gif, save_custom_3panel_gif
from physax.genome_analysis import analyze_and_plot_top_genomes
import argparse
import os

print('device:', jax.devices()[0].platform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=4096 * 40)
    parser.add_argument('--initial_pop', type=int, default=1000)
    parser.add_argument('--total_cycles', type=int, default=50_000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--toy', action='store_true', help='Run a very small toy scenario for debugging')
    args = parser.parse_args()

    if args.toy:
        print("--- RUNNING IN TOY/DEBUG MODE ---")
        cfg = make_config(
            pop_size=32,
            initial_pop=1,
            max_genome_len=128
        )
        args.total_cycles = 300
        log_interval = 1
    else:
        cfg = make_config(
            pop_size=args.pop_size,
            initial_pop=args.initial_pop,
        )
        log_interval = args.log_interval

    from pathlib import Path
    from datetime import datetime
    
    # Read base path from .env if it exists
    base_path = Path("output")
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("BASE_PATH="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    base_path = Path(val)
                    break
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = base_path / f"run_{timestamp}_{args.total_cycles}cycles"
    path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {path}")

    model = Model(cfg)
    key = random.PRNGKey(42)
    try:
        pop, stats = model.run_simulation(
            key,
            total_cycles=args.total_cycles,
            log_interval=log_interval,
            use_wandb=False,
            output_dir=str(path)
        )
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        pass

    print("\n=== FINAL STATE ===")
    alive = pop.alive
    alive_count = jnp.sum(alive)
    q_lens = jnp.nanpercentile(jnp.where(alive, pop.genome_len, jnp.nan), PERCENTILES )

    print(f"Alive: {int(alive_count)}")
    print(f"min={q_lens[0]:.0f}, lq={q_lens[1]:.1f}, med={q_lens[2]:.0f}, uq={q_lens[3]:.0f}, max={q_lens[4]:.0f}")

    if len(stats) > 0:
        timestamps = [s['cycle'] for s in stats]
        pop_sizes = [s['pop_size'] for s in stats]
        births = [s['births'] for s in stats]
        q_lens = [s['q_len'] for s in stats]

        plot_metrics(timestamps, pop_sizes, births, q_lens, str(path / "simulation_metrics.png"))

        snapshots = [s['snapshot'] for s in stats]
        # save_grid_gif(snapshots, str(path / "evolution.gif"), cfg)
        save_custom_3panel_gif(snapshots, str(path / "evolution_3panel.gif"), cfg)
        
        # Analyze and plot top genomes
        top_hashes = analyze_and_plot_top_genomes(stats, str(path / "top_genomes.png"))
        
        from physax.model import global_self_replicating_genomes
        import numpy as np
        
        top_genomes_dict = {}
        for h in top_hashes:
            if h in global_self_replicating_genomes:
                top_genomes_dict[str(h)] = global_self_replicating_genomes[h]
                
        if top_genomes_dict:
            np.savez(str(path / "genomes_details.npz"), **top_genomes_dict)
            print("Saved top genomes to genomes_details.npz")
    else:
        print("No stats collected, skipping plots and saves.")
    
    if args.toy:
        print("\n--- TOY DEBUG LOG ---")
        for chunk in stats:
            cyc = chunk['cycle']
            pop_cnt = chunk['pop_size']
            b = chunk['births']
            status_arr = chunk['snapshot']['status']
            alive_arr = chunk['snapshot']['alive']
            
            # Count statuses among alive
            status_counts = {}
            for st in [0, 1, 2, 3, 4]:
                status_counts[st] = int(jnp.sum((status_arr == st) & alive_arr))
                
            print(f"Cycle {cyc}: Pop={pop_cnt}, Births={b}, Status Breakdown (UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD)={status_counts}")
