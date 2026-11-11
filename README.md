# 3D UAV Mesh Benchmark & Self-Healing System

A modular Python simulation for disaster-response UAV mesh placement. The pipeline models survivor hotspots and Al-Hourani air-to-ground coverage, evaluates three placement baselines, optimizes a fleet with constrained NSGA-II, simulates a UAV failure, and runs a self-healing controller.

## Project layout

```text
src/
  config.py             Simulation settings
  physics.py            Environment and channel/coverage model
  graph_engine.py       Mesh connectivity, articulation, and coverage metrics
  baselines.py          K-Means, K-Means + MST, connected greedy
  nsga2_optimizer.py    Constrained NSGA-II placement optimizer
  self_healing.py       Virtual-force recovery controller
  visualizer.py         Benchmark and recovery dashboards
main.py                 End-to-end pipeline
outputs/                Generated PNG dashboards and CSV metrics
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The default run uses 22 UAVs, 1,800 survivor locations, a population of 100, and 1,000 generations. It can take a while. For a quick smoke test, edit `SimConfig` in `src/config.py` to use a smaller `n_gen` and `pop_size`. The run writes `outputs/benchmark_metrics.csv` and two PNG dashboards, then displays the figures.

A random disaster layout is generated per run. Set `rng_seed` in `SimConfig` for repeatable results. These are simulation results, not operational flight guidance.
