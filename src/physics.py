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


def main():
    Path("outputs").mkdir(exist_ok=True)
    start = time.time()
    config = SimConfig()
    env = DisasterEnvironment(config=config)

    print(f"[*] Generated {env.n_users} survivors across {env.n_clusters} random hotspots in a {int(env.area_size)}m x {int(env.area_size)}m zone.")
    print(f"[*] Fleet Size: {env.n_uavs} UAVs | Optimal Altitude: {env.h_opt:.1f}m | Max Coverage Radius: {env.ground_coverage_radius(env.h_opt):.1f}m\n")

    print("[*] Evaluating baseline algorithms...")
    coords_b1, m_b1 = run_baseline_1_pure_kmeans(env)
    coords_b2, m_b2 = run_baseline_2_kmeans_mst(env)
    coords_b3, m_b3 = run_baseline_3_connected_greedy(env)

    print("\n[*] Running Constrained NSGA-II (1000 generations)...")
    optimizer = ConstrainedNSGA2Optimizer(env, pop_size=config.pop_size, n_gen=config.n_gen, seed=42)
    pop_final, F_final, CV_final, hist_cov = optimizer.run()

    feasible_idx = np.where(CV_final == 0.0)[0]
    if len(feasible_idx) > 0:
        best_idx = feasible_idx[np.argmin(F_final[feasible_idx, 0])]
    else:
        best_idx = np.argmin(CV_final)

    coords_p1 = pop_final[best_idx].reshape((env.n_uavs, 3))
    m_p1 = evaluate_mesh_topology(coords_p1, env)

    swarm_center = np.mean(coords_p1[:, :2], axis=0)
    failed_idx = int(np.argmin(np.linalg.norm(coords_p1[:, :2] - swarm_center, axis=1)))
    coords_fail = np.delete(coords_p1, failed_idx, axis=0)
    m_fail = evaluate_mesh_topology(coords_fail, env)

    print(f"\n[*] Mid-mission crash scenario: UAV #{failed_idx + 1} lost. Running self-healing controller...")
    healer = SelfHealingController(env, step_size=18.0, seed=42)
    coords_heal, m_heal = healer.heal_mesh(coords_fail, episodes=95)

    print("\n" + "=" * 92)
    print(f"{'ALGORITHM / STAGE':<35} | {'RAW COV':<9} | {'GBS COV':<9} | {'COMP':<6} | {'CUT NODES':<9} | {'FIEDLER λ2'}")
    print("-" * 92)
    rows = [
        ("Baseline 1: Pure K-Means", m_b1),
        ("Baseline 2: K-Means + MST", m_b2),
        ("Baseline 3: Connected Greedy", m_b3),
        ("Proposed: NSGA-II (Phase 1)", m_p1),
        (f"Mid-Mission Crash (UAV #{failed_idx + 1} Lost)", m_fail),
        ("Proposed: Self-Healed (Phase 2)", m_heal),
    ]
    for name, metrics in rows:
        print(f"{name:<35} | {metrics['raw_coverage_ratio'] * 100:>7.1f}% | {metrics['coverage_ratio'] * 100:>7.1f}% | {metrics['n_components']:>5d} | {metrics['n_articulation']:>9d} | {metrics['algebraic_connectivity']:>8.3f}")
    print("=" * 92)

    plot_baselines_dashboard(env, coords_b1, m_b1, coords_b2, m_b2, coords_b3, m_b3, coords_p1, m_p1)
    plot_complete_dashboard(env, hist_cov, coords_p1, m_p1, failed_idx, coords_fail, m_fail, coords_heal, m_heal)

    print(f"\n[*] Total runtime: {time.time() - start:.1f}s")
    print("[*] Dashboard images saved to outputs/")

    return coords_b1, m_b1, coords_b2, m_b2, coords_b3, m_b3, coords_p1, m_p1, coords_heal, m_heal


if __name__ == "__main__":
    main()
