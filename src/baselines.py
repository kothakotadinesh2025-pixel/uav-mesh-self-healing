import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from src.graph_engine import evaluate_mesh_topology


def run_baseline_1_pure_kmeans(env):
    model = KMeans(n_clusters=env.n_uavs, n_init=10, random_state=42).fit(env.users)
    xy = np.clip(model.cluster_centers_, 70.0, env.area_size - 70.0)
    coords = np.column_stack((xy, np.full(env.n_uavs, env.h_opt)))
    return coords, evaluate_mesh_topology(coords, env)


def _prim_mst(nodes):
    in_tree, remaining, edges = [0], set(range(1, len(nodes))), []
    while remaining:
        u, v = min(((u, v) for u in in_tree for v in remaining),
                   key=lambda pair: np.linalg.norm(nodes[pair[0]] - nodes[pair[1]]))
        distance = float(np.linalg.norm(nodes[u] - nodes[v]))
        in_tree.append(v)
        remaining.remove(v)
        edges.append((u, v, distance))
    return edges


def run_baseline_2_kmeans_mst(env):
    n_k = min(10, env.n_uavs)
    model = KMeans(n_clusters=n_k, n_init=10, random_state=42).fit(env.users)
    centers = model.cluster_centers_
    nodes = np.vstack((env.gbs_pos[:2], centers))
    edges = _prim_mst(nodes)
    relay_count = env.n_uavs - n_k
    allocations = [max(0, int(np.ceil(d / (0.88 * env.r_comm_a2a))) - 1) for _, _, d in edges]
    allocation = [0] * len(edges)
    for idx in sorted(range(len(edges)), key=lambda i: -edges[i][2]):
        amount = min(allocations[idx], relay_count - sum(allocation))
        allocation[idx] = amount
    idx = 0
    while sum(allocation) < relay_count:
        allocation[idx % len(edges)] += 1
        idx += 1

    placed = [center for center in centers]
    for (u, v, _), n_relays in zip(edges, allocation):
        for s in range(1, n_relays + 1):
            alpha = s / (n_relays + 1.0)
            placed.append((1.0 - alpha) * nodes[u] + alpha * nodes[v])
    xy = np.clip(np.asarray(placed[:env.n_uavs]), 70.0, env.area_size - 70.0)
    coords = np.column_stack((xy, np.full(env.n_uavs, env.h_opt)))
    return coords, evaluate_mesh_topology(coords, env)


def run_baseline_3_connected_greedy(env):
    """Greedily add a UAV inside the communication range of the existing mesh."""
    coverage_radius = float(env.ground_coverage_radius(env.h_opt))
    hop = np.sqrt(max(100.0, (0.90 * env.r_comm_a2a) ** 2 - env.h_opt ** 2))
    angles = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    radial_steps = np.linspace(0.45 * hop, 0.95 * hop, 5)
    placed = []
    uncovered = np.ones(len(env.users), dtype=bool)

    for _ in range(env.n_uavs):
        parents = [env.gbs_pos[:2]] + placed
        candidates = np.vstack([
            np.column_stack((p[0] + r * np.cos(angles), p[1] + r * np.sin(angles)))
            for p in parents for r in radial_steps
        ])
        bounds = ((candidates >= 70.0) & (candidates <= env.area_size - 70.0)).all(axis=1)
        candidates = candidates[bounds]
        if placed and len(candidates):
            candidates = candidates[np.all(cdist(candidates, np.asarray(placed)) >= env.d_safe * 1.05, axis=1)]

        if len(candidates) == 0:
            # Keep the fallback inside bounds and avoid placing coincident UAVs.
            candidates = np.array([[env.gbs_pos[0] + min(env.d_safe * (len(placed) + 1), hop), env.gbs_pos[1]]])
            candidates = np.clip(candidates, 70.0, env.area_size - 70.0)

        remaining_users = env.users[uncovered]
        if len(remaining_users):
            gains = np.sum(cdist(candidates, remaining_users) <= coverage_radius, axis=1)
            if gains.max() > 0:
                chosen = candidates[int(np.argmax(gains))]
            else:
                target = remaining_users[np.argmin(np.linalg.norm(remaining_users - remaining_users.mean(axis=0), axis=1))]
                chosen = candidates[np.argmin(np.linalg.norm(candidates - target, axis=1))]
        else:
            chosen = candidates[0]
        placed.append(chosen)
        uncovered[np.linalg.norm(env.users - chosen, axis=1) <= coverage_radius] = False

    coords = np.column_stack((np.asarray(placed), np.full(env.n_uavs, env.h_opt)))
    return coords, evaluate_mesh_topology(coords, env)
