import csv
import time
from pathlib import Path

import numpy as np

from src.baselines import run_baseline_1_pure_kmeans, run_baseline_2_kmeans_mst, run_baseline_3_connected_greedy
from src.config import SimConfig
from src.graph_engine import evaluate_mesh_topology
from src.nsga2_optimizer import ConstrainedNSGA2Optimizer
from src.physics import DisasterEnvironment
from src.self_healing import SelfHealingController
from src.visualizer import plot_baselines_dashboard, plot_complete_dashboard


def _save_metrics(path, rows):
    fields = ["algorithm", "raw_coverage_percent", "gbs_coverage_percent", "components", "cut_vertices", "fiedler_lambda2", "constraint_violation"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in rows:
            writer.writerow({
                "algorithm": name,
                "raw_coverage_percent": 100.0 * metrics["raw_coverage_ratio"],
                "gbs_coverage_percent": 100.0 * metrics["coverage_ratio"],
                "components": metrics["n_components"],
                "cut_vertices": metrics["n_articulation"],
                "fiedler_lambda2": metrics["algebraic_connectivity"],
                "constraint_violation": metrics["cv"],
            })


def main(config=None, show=True):
    start = time.time()
    config = config or SimConfig()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = DisasterEnvironment(config=config)
    print(f"[*] Generated {env.n_users} survivors in {env.n_clusters} hotspots; {env.n_uavs} UAVs.")
    print(f"[*] Optimal altitude: {env.h_opt:.1f} m; coverage radius: {env.ground_coverage_radius(env.h_opt):.1f} m\n")

    print("[*] Evaluating baseline algorithms...")
    coords_b1, m_b1 = run_baseline_1_pure_kmeans(env)
    coords_b2, m_b2 = run_baseline_2_kmeans_mst(env)
    coords_b3, m_b3 = run_baseline_3_connected_greedy(env)

    print(f"\n[*] Running constrained NSGA-II ({config.n_gen} generations)...")
    optimizer = ConstrainedNSGA2Optimizer(env, config.pop_size, config.n_gen, seed=config.rng_seed)
    population, objectives, violations, history = optimizer.run()
    feasible = np.flatnonzero(violations == 0.0)
    best_idx = feasible[np.argmin(objectives[feasible, 0])] if len(feasible) else int(np.argmin(violations))
    coords_p1 = population[best_idx].reshape(env.n_uavs, 3)
    m_p1 = evaluate_mesh_topology(coords_p1, env)

    center = coords_p1[:, :2].mean(axis=0)
    failed_idx = int(np.argmin(np.linalg.norm(coords_p1[:, :2] - center, axis=1)))
    coords_fail = np.delete(coords_p1, failed_idx, axis=0)
    m_fail = evaluate_mesh_topology(coords_fail, env)
    print(f"\n[*] Simulated crash: UAV #{failed_idx + 1}; running self-healing controller...")
    coords_heal, m_heal = SelfHealingController(env, seed=config.rng_seed).heal_mesh(coords_fail, episodes=95)

    rows = [
        ("Baseline 1: Pure K-Means", m_b1),
        ("Baseline 2: K-Means + MST", m_b2),
        ("Baseline 3: Connected Greedy", m_b3),
        ("Proposed: NSGA-II Phase 1", m_p1),
        (f"Mid-mission crash (UAV #{failed_idx + 1})", m_fail),
        ("Proposed: Self-Healed Phase 2", m_heal),
    ]
    print("\n" + "=" * 100)
    print(f"{'ALGORITHM / STAGE':<38} | {'RAW':>7} | {'GBS':>7} | {'COMP':>5} | {'CUT':>5} | {'LAMBDA2':>8}")
    print("-" * 100)
    for name, metrics in rows:
        print(f"{name:<38} | {100*metrics['raw_coverage_ratio']:6.1f}% | {100*metrics['coverage_ratio']:6.1f}% | {metrics['n_components']:5d} | {metrics['n_articulation']:5d} | {metrics['algebraic_connectivity']:8.3f}")
    print("=" * 100)

    _save_metrics(output_dir / "benchmark_metrics.csv", rows)
    fig1 = plot_baselines_dashboard(env, coords_b1, m_b1, coords_b2, m_b2, coords_b3, m_b3, coords_p1, m_p1, output_dir)
    fig2 = plot_complete_dashboard(env, history, coords_p1, m_p1, failed_idx, coords_fail, m_fail, coords_heal, m_heal, output_dir)
    print(f"[*] Runtime: {time.time() - start:.1f}s. Results saved in {output_dir}/")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig1)
        plt.close(fig2)
    return rows


if __name__ == "__main__":
    main()
