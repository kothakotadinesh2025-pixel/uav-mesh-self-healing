from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_baselines_dashboard(env, coords_b1, m_b1, coords_b2, m_b2, coords_b3, m_b3, coords_p1, m_p1):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=115)

    panels = [
        (axes[0, 0], coords_b1, m_b1, "1. Baseline 1: Pure K-Means"),
        (axes[0, 1], coords_b2, m_b2, "2. Baseline 2: K-Means + MST"),
        (axes[1, 0], coords_b3, m_b3, "3. Baseline 3: Connected Greedy"),
    ]

    for ax, uavs, metrics, title in panels:
        covered = metrics["covered_mask"]
        ax.scatter(env.users[covered, 0], env.users[covered, 1], c="#27ae60", s=8, alpha=0.65)
        ax.scatter(env.users[~covered, 0], env.users[~covered, 1], c="#e74c3c", marker="x", s=14, alpha=0.80)

        all_nodes = np.vstack([env.gbs_pos, uavs])
        for u, v in metrics["edges"]:
            ax.plot([all_nodes[u, 0], all_nodes[v, 0]], [all_nodes[u, 1], all_nodes[v, 1]], color="#1f618d", linewidth=1.2, alpha=0.75)

        radii = env.ground_coverage_radius(uavs[:, 2])
        for i, ((x, y, _), r) in enumerate(zip(uavs, radii)):
            node_id = i + 1
            is_active = i in metrics["active_uav_indices"]
            is_cut = node_id in metrics["articulation_nodes"]
            if not is_active:
                color_fill, color_node, linestyle = "#95a5a6", "#7f8c8d", ":"
            elif is_cut:
                color_fill, color_node, linestyle = "#e67e22", "#d35400", "--"
            else:
                color_fill, color_node, linestyle = "#3498db", "#1b4f72", "-"

            ax.add_patch(Circle((x, y), r, color=color_fill, alpha=0.10, lw=1.1, ls=linestyle))
            ax.scatter(x, y, c=color_node, s=68, zorder=5, edgecolors="white", linewidth=1.0)

        ax.scatter(env.gbs_pos[0], env.gbs_pos[1], c="#5b2c6f", marker="s", s=130, zorder=6)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, env.area_size)
        ax.set_ylim(0, env.area_size)
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.grid(True, linestyle=":", alpha=0.4)

    ax_bar = axes[1, 1]
    categories = [
        "Raw Local\nCoverage (%)",
        "True GBS-Connected\nCoverage (%)",
        "Disconnected\nComponents",
        "Cut Vertices\n(Single Pt Failures)",
    ]
    vals_b1 = [m_b1["raw_coverage_ratio"] * 100, m_b1["coverage_ratio"] * 100, m_b1["n_components"], m_b1["n_articulation"]]
    vals_b2 = [m_b2["raw_coverage_ratio"] * 100, m_b2["coverage_ratio"] * 100, m_b2["n_components"], m_b2["n_articulation"]]
    vals_b3 = [m_b3["raw_coverage_ratio"] * 100, m_b3["coverage_ratio"] * 100, m_b3["n_components"], m_b3["n_articulation"]]
    vals_p1 = [m_p1["raw_coverage_ratio"] * 100, m_p1["coverage_ratio"] * 100, m_p1["n_components"], m_p1["n_articulation"]]

    x_pos = np.arange(len(categories))
    w = 0.20

    ax_bar.bar(x_pos - 1.5 * w, vals_b1, w, label="Baseline 1: Pure K-Means", color="#7f8c8d")
    ax_bar.bar(x_pos - 0.5 * w, vals_b2, w, label="Baseline 2: K-Means + MST", color="#e67e22")
    ax_bar.bar(x_pos + 0.5 * w, vals_b3, w, label="Baseline 3: Connected Greedy", color="#8e44ad")
    ax_bar.bar(x_pos + 1.5 * w, vals_p1, w, label="Proposed: Constrained NSGA-II", color="#27ae60")

    ax_bar.set_title("4. Benchmark Comparison: Baselines vs Proposed NSGA-II", fontsize=11.0, fontweight="bold")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(categories, fontsize=9.0, fontweight="bold")
    ax_bar.set_ylabel("Metric Value")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / "baseline_dashboard.png", dpi=180)


def plot_complete_dashboard(env, hist_cov, coords_p1, metrics_p1, failed_idx, coords_fail, metrics_fail, coords_heal, metrics_heal):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=115)

    ax_conv = axes[0, 0]
    gens = range(1, len(hist_cov) + 1)
    ax_conv.plot(gens, hist_cov, color="#1f618d", linewidth=2.5, label="Best Feasible Coverage (%)")
    ax_conv.fill_between(gens, hist_cov, min(hist_cov) - 1.0, color="#3498db", alpha=0.15)
    ax_conv.set_title(f"1. Large-Scale NSGA-II Convergence ({env.n_uavs} UAVs, {env.n_users} Survivors)", fontsize=11.5, fontweight="bold")
    ax_conv.set_xlabel("Evolutionary Generation")
    ax_conv.set_ylabel("Coverage (%)")
    ax_conv.set_ylim(max(0.0, min(hist_cov) - 3.0), min(102.0, max(hist_cov) + 3.0))
    ax_conv.grid(True, linestyle="--", alpha=0.4)
    ax_conv.legend(loc="lower right")

    states = [
        (axes[0, 1], coords_p1, metrics_p1, f"2. Phase 1: NSGA-II Optimal Biconnected Mesh\nCoverage: {metrics_p1['coverage_ratio']*100:.1f}% | Cut Vertices: {metrics_p1['n_articulation']}"),
        (axes[1, 0], coords_fail, metrics_fail, f"3. Mid-Mission Failure (UAV #{failed_idx + 1} Crashed)\nCoverage: {metrics_fail['coverage_ratio']*100:.1f}% | Cut Vertices: {metrics_fail['n_articulation']}"),
        (axes[1, 1], coords_heal, metrics_heal, f"4. Phase 2: Self-Healed Biconnected Mesh\nCoverage: {metrics_heal['coverage_ratio']*100:.1f}% | Cut Vertices: {metrics_heal['n_articulation']}"),
    ]

    for ax, uavs, metrics, title in states:
        covered = metrics["covered_mask"]
        ax.scatter(env.users[covered, 0], env.users[covered, 1], c="#27ae60", s=8, alpha=0.65)
        ax.scatter(env.users[~covered, 0], env.users[~covered, 1], c="#e74c3c", marker="x", s=14, alpha=0.80)

        all_nodes = np.vstack([env.gbs_pos, uavs])
        for u, v in metrics["edges"]:
            ax.plot([all_nodes[u, 0], all_nodes[v, 0]], [all_nodes[u, 1], all_nodes[v, 1]], color="#1f618d", linewidth=1.1, alpha=0.65)

        for i, ((x, y, z), r) in enumerate(zip(uavs, env.ground_coverage_radius(uavs[:, 2]))):
            is_cut = (i + 1) in metrics["articulation_nodes"]
            ax.add_patch(Circle((x, y), r, color="#e67e22" if is_cut else "#3498db", alpha=0.09, lw=1.1, ls="--" if is_cut else "-"))
            ax.scatter(x, y, c="#d35400" if is_cut else "#1b4f72", s=65, zorder=5, edgecolors="white", linewidth=1.0)

        ax.scatter(env.gbs_pos[0], env.gbs_pos[1], c="#5b2c6f", marker="s", s=130, zorder=6)
        ax.set_xlim(0, env.area_size)
        ax.set_ylim(0, env.area_size)
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title(title, fontsize=10.8, fontweight="bold")

    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / "nsga2_self_healing_dashboard.png", dpi=180)
