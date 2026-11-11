# 3D UAV Mesh Benchmark & Self-Healing System

A complete Python project for modeling disaster-response UAV mesh deployment with:

- 3D Al-Hourani coverage modeling
- 22-UAV disaster zone simulation
- 1,800 survivor locations
- three benchmark baselines
- constrained NSGA-II optimizer
- autonomous self-healing controller
- visualization dashboards

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project structure

```text
src/
  __init__.py
  config.py
  physics.py
  graph_engine.py
  baselines.py
  nsga2_optimizer.py
  self_healing.py
  visualizer.py
main.py
requirements.txt
README.md
```

## Notes

The pipeline runs the three baseline algorithms and then executes the proposed optimizer and self-healing controller. It prints a comparative metrics table and displays the dashboards.
