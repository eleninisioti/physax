# Physax: A JAX implementation of Physis



#### Installation

To install the project and set up the environment using [`uv`](https://docs.astral.sh/uv/), run:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will create a virtual environment, activate it, and install all dependencies (including PyTorch with the `cu121` setup).


### Execute the simulation:

```bash

CUDA_VISIBLE_DEVICES=2 python -m physax --pop_size 65536 --initial_pop 1000 --total_cycles 6000 --log_interval 50
CUDA_VISIBLE_DEVICES=2 python -m physax --pop_size 16384 --initial_pop 50 --total_cycles 6000 --log_interval 50

CUDA_VISIBLE_DEVICES=0 python -m physax --toy

# re-run visualization of the run (pass folder name, base path should be in the .env file):
python -m physax.visualization --folder run_2026-06-14_12-22-49_300cycles
```