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

CUDA_VISIBLE_DEVICES=2 python -m physax --pop_size 16384 --initial_pop 1000 --total_cycles 3000 --log_interval 50

```