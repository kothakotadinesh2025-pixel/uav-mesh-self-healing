from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def _draw_network(ax, env, coords, metrics, title):
    covered = metrics["covered_mask"]
    ax.scatter(env.users[covered, 0], env.users[covered, 1], c="#27ae60", s=8, alpha=0.65, label="GBS-connected covered")
    ax.scatter(env.users[~covered, 0], env.users[~covered, 1], c="#e74c3c", marker="x", s=12, alpha=0.75, label="Uncovered")
    nodes = np.vstack((env.gbs_pos, coords))
    for u, v in metrics["edges"]:
        ax.plot(nodes[[u, v], 0], nodes[[u, v], 1], color="#1f618d", lw=1.1, alpha=0.65)
    for i, (x, y, _) in enumerate(coords):
        radius = float(env.ground_coverage_radius(coords[i, 2]))
        cut = i + 1 in metrics["articulation_nodes"]
        color = "#e67e22" if cut else "#3498db"
        ax.add_patch(Circle((x, y), radius, color=color, alpha=0.08, lw=1.0, ls="--" if cut else "-"))
        ax.scatter(x, y, c="#d35400" if cut else "#1b4f72", s=55, edgecolors="white", zorder=5)
    ax.scatter(env.gbs_pos[0], env.gbs_pos[1], c="#5b2c6f", marker="s", s=120, zorder=6, label="GBS")
    ax.set(title=title, xlim=(0, env.area_size), ylim=(0, env.area_size), xlabel="X (m)", ylabel="Y (m)")
    ax.grid(True, linestyle=":", alpha=0.4)


def plot_baselines_dashboard(env, coords_b1, m_b1, coords_b2, m_b2, coords_b3, m_b3, coords_p1, m_p1, output_dir="outputs"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=115)
    states = [(coords_b1, m_b1, "Baseline 1: Pure K-Means"), (coords_b2, m_b2, "Baseline 2: K-Means + MST"), (coords_b3, m_b3, "Baseline 3: Connected Greedy")]
    for ax, (coords, metrics, name) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), states):
        title = f"{name}\nGBS coverage {metrics['coverage_ratio']*100:.1f}% | components {metrics['n_components']} | cuts {metrics['n_articulation']}"
        _draw_network(ax, env, coords, metrics, title)
    ax = axes[1, 1]
    names = ["K-Means", "K-Means+MST", "Greedy", "NSGA-II"]
    metrics_list = [m_b1, m_b2, m_b3, m_p1]
    x = np.arange(4)
    ax.bar(x - 0.18, [m["raw_coverage_ratio"] * 100 for m in metrics_list], 0.36, label="Raw coverage (%)")
    ax.bar(x + 0.18, [m["coverage_ratio"] * 100 for m in metrics_list], 0.36, label="GBS-connected coverage (%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_title("Baseline and optimizer coverage comparison")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "baseline_dashboard.png", dpi=180)
    return fig


def plot_complete_dashboard(env, history, coords_p1, m_p1, failed_idx, coords_fail, m_fail, coords_heal, m_heal, output_dir="outputs"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=115)
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(history) + 1), history, color="#1f618d", lw=2)
    ax.set(title="NSGA-II best population coverage", xlabel="Generation", ylabel="Coverage (%)")
    ax.grid(True, alpha=0.35)
    states = [
        (coords_p1, m_p1, "Phase 1: Optimized mesh"),
        (coords_fail, m_fail, f"Crash: UAV #{failed_idx + 1} lost"),
        (coords_heal, m_heal, "Phase 2: Self-healed mesh"),
    ]
    for ax, (coords, metrics, name) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), states):
        title = f"{name}\nCoverage {metrics['coverage_ratio']*100:.1f}% | components {metrics['n_components']} | cut vertices {metrics['n_articulation']}"
        _draw_network(ax, env, coords, metrics, title)
        if name.startswith("Crash"):
            failed_xy = coords_p1[failed_idx, :2]
            ax.scatter(*failed_xy, c="#c0392b", marker="X", s=160, label="Failed UAV")
    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "nsga2_self_healing_dashboard.png", dpi=180)
    return fig
