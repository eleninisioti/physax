"""
Physis Minimal - Single-file digital evolution simulation.

Tracks genome composition metrics over evolution:
- Hardware length (registers/stacks section)
- Language length (instruction definitions)
- Code length (actual program)
- Number of unique instructions

Saves results to pickle file and optionally logs to Weights & Biases.
"""

import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

from organism import Gene
from world import World

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==========================================
# 6. GENOME ANALYSIS
# ==========================================

def analyze_genome(genome: np.ndarray) -> dict:
    """Parse a dynamic-mode genome and return section lengths."""
    ptr = 0
    limit = len(genome)
    
    # A. Hardware section
    hardware_start = 0
    while ptr < limit:
        g = genome[ptr]
        if g == Gene.B:
            ptr += 1
            break
        elif g in [Gene.I, Gene.SEP]:
            break
        ptr += 1
    
    hardware_len = ptr - hardware_start
    if ptr > 0 and genome[ptr - 1] == Gene.B:
        hardware_len -= 1

    language_start = ptr
    
    # B. Instruction definitions
    num_instructions = 0
    while ptr < limit:
        if genome[ptr] == Gene.SEP:
            ptr += 1
            break
        if genome[ptr] == Gene.I:
            num_instructions += 1
        ptr += 1
    
    language_len = ptr - language_start
    if ptr > language_start and genome[ptr - 1] == Gene.SEP:
        language_len -= 1

    # C. Code section
    code_len = limit - ptr
    
    return {
        'hardware_len': hardware_len,
        'language_len': language_len,
        'code_len': code_len,
        'num_instructions': num_instructions,
        'total_len': len(genome)
    }


def get_population_stats(population: list, world: 'World' = None) -> dict:
    """Get average genome composition stats for a population."""
    if not population:
        return {
            'hardware_len': 0, 'language_len': 0, 'code_len': 0,
            'num_instructions': 0, 'total_len': 0, 'pop_size': 0,
            'births': 0, 'deaths': 0, 'total_births': 0, 'total_deaths': 0
        }
    
    stats = [analyze_genome(org.genome) for org in population]
    
    result = {
        'hardware_len': np.mean([s['hardware_len'] for s in stats]),
        'language_len': np.mean([s['language_len'] for s in stats]),
        'code_len': np.mean([s['code_len'] for s in stats]),
        'num_instructions': np.mean([s['num_instructions'] for s in stats]),
        'total_len': np.mean([s['total_len'] for s in stats]),
        'pop_size': len(population),
        'births': world.births_this_epoch if world else 0,
        'deaths': world.deaths_this_epoch if world else 0,
        'total_births': world.total_births if world else 0,
        'total_deaths': world.total_deaths if world else 0
    }
    return result


# ==========================================
# 7. EXPERIMENT RUNNER
# ==========================================

def run_tracking_experiment(epochs=500, pop_size=150, cpu_cycles=200, log_interval=10, 
                           use_wandb=False, wandb_project="physis-minimal", initial_pop=10,
                           trial_num=None):
    """Run experiment and track genome composition over time."""
    
    trial_str = f" (Trial {trial_num})" if trial_num is not None else ""
    print(f"=== GENOME COMPOSITION TRACKING{trial_str} ===")
    print(f"Running {epochs} epochs, initial pop {initial_pop}, max pop {pop_size}, CPU cycles {cpu_cycles}")
    print()
    
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed. Run: pip install wandb")
            use_wandb = False
        else:
            run_name = f"trial-{trial_num}" if trial_num is not None else None
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "epochs": epochs,
                    "pop_size": pop_size,
                    "initial_pop": initial_pop,
                    "cpu_cycles": cpu_cycles,
                    "log_interval": log_interval,
                    "mode": "dynamic",
                    "trial_num": trial_num,
                }
            )
            print(f"Logging to wandb project: {wandb_project}")
    
    world = World(mode='dynamic', pop_size=pop_size, cpu_cycles=cpu_cycles, initial_pop=initial_pop)
    
    history = {
        'epoch': [],
        'pop_size': [],
        'hardware_len': [],
        'language_len': [],
        'code_len': [],
        'num_instructions': [],
        'total_len': [],
        'births': [],
        'deaths': [],
        'total_births': [],
        'total_deaths': []
    }
    
    metadata = {
        'epochs': epochs,
        'pop_size': pop_size,
        'initial_pop': initial_pop,
        'cpu_cycles': cpu_cycles,
        'log_interval': log_interval,
        'mode': 'dynamic',
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"{'Epoch':<8} {'Pop':<6} {'Births':<8} {'Deaths':<8} {'Hardware':<10} {'Language':<10} {'Code':<8} {'Total':<8}")
    print("-" * 90)
    
    for t in range(epochs):
        world.step()
        
        if t % log_interval == 0:
            stats = get_population_stats(world.population, world)
            
            history['epoch'].append(t)
            history['pop_size'].append(stats['pop_size'])
            history['hardware_len'].append(stats['hardware_len'])
            history['language_len'].append(stats['language_len'])
            history['code_len'].append(stats['code_len'])
            history['num_instructions'].append(stats['num_instructions'])
            history['total_len'].append(stats['total_len'])
            history['births'].append(stats['births'])
            history['deaths'].append(stats['deaths'])
            history['total_births'].append(stats['total_births'])
            history['total_deaths'].append(stats['total_deaths'])
            
            if use_wandb:
                wandb.log({
                    "epoch": t,
                    "population/size": stats['pop_size'],
                    "population/births": stats['births'],
                    "population/deaths": stats['deaths'],
                    "population/total_births": stats['total_births'],
                    "population/total_deaths": stats['total_deaths'],
                    "genome/hardware_len": stats['hardware_len'],
                    "genome/language_len": stats['language_len'],
                    "genome/code_len": stats['code_len'],
                    "genome/num_instructions": stats['num_instructions'],
                    "genome/total_len": stats['total_len'],
                    "ratios/hardware_pct": 100 * stats['hardware_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                    "ratios/language_pct": 100 * stats['language_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                    "ratios/code_pct": 100 * stats['code_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                })
            
            print(f"{t:<8} {stats['pop_size']:<6} {stats['births']:<8} {stats['deaths']:<8} "
                  f"{stats['hardware_len']:<10.1f} {stats['language_len']:<10.1f} "
                  f"{stats['code_len']:<8.1f} {stats['total_len']:<8.1f}")
            
            if stats['pop_size'] == 0:
                print("\nPopulation extinct!")
                break
    
    return {'history': history, 'metadata': metadata}


def save_results(data: dict, output_dir: str = 'output', use_wandb: bool = False):
    """Save results to pickle file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_path / f'genome_composition_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nResults saved to: {filename}")
    
    if use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact('genome_composition', type='dataset')
        artifact.add_file(str(filename))
        wandb.log_artifact(artifact)
        wandb.finish()
        print("wandb run finished.")
    
    return filename


# ==========================================
# 8. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Physis Minimal - Digital Evolution Simulation')
    parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--pop-size', type=int, default=1000, help='Max population size')
    parser.add_argument('--initial-pop', type=int, default=10, help='Initial population size')
    parser.add_argument('--cpu-cycles', type=int, default=2000, help='CPU cycles per organism per epoch')
    parser.add_argument('--log-interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='physis-minimal', help='W&B project name')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
    
    args = parser.parse_args()
    
    for trial in range(args.trials):
        trial_num = trial + 1 if args.trials > 1 else None
        
        data = run_tracking_experiment(
            epochs=args.epochs,
            pop_size=args.pop_size,
            cpu_cycles=args.cpu_cycles,
            log_interval=args.log_interval,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            initial_pop=args.initial_pop,
            trial_num=trial_num
        )
        
        save_results(data, args.output_dir, use_wandb=args.wandb)
