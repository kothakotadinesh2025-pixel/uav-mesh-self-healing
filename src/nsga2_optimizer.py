import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from src.graph_engine import evaluate_mesh_topology


def run_baseline_1_pure_kmeans(env):
    kmeans = KMeans(n_clusters=env.n_uavs, n_init=10, random_state=42)
    kmeans.fit(env.users)
    xy = np.clip(kmeans.cluster_centers_, 70.0, env.area_size - 70.0)
    z = np.full((env.n_uavs, 1), env.h_opt)
    uav_coords = np.hstack([xy, z])
    return uav_coords, evaluate_mesh_topology(uav_coords, env)


def run_baseline_2_kmeans_mst(env):
    n_k = 10
    kmeans = KMeans(n_clusters=n_k, n_init=10, random_state=42)
    kmeans.fit(env.users)
    centroids = kmeans.cluster_centers_

    nodes = np.vstack([env.gbs_pos[:2], centroids])
    in_tree = [0]
    rem = list(range(1, len(nodes)))
    mst_edges = []

    while rem:
        best_d = np.inf
        best_pair = (0, rem[0])
        for u in in_tree:
            for v in rem:
                d = np.linalg.norm(nodes[u] - nodes[v])
                if d < best_d:
                    best_d = d
                    best_pair = (u, v)
        in_tree.append(best_pair[1])
        rem.remove(best_pair[1])
        mst_edges.append((best_pair[0], best_pair[1], best_d))

    n_relays_available = env.n_uavs - n_k
    step_max = env.r_comm_a2a * 0.88
    edge_relays = []
    for u, v, d in mst_edges:
        needed = max(0, int(np.ceil(d / step_max)) - 1)
        edge_relays.append([u, v, d, needed])
    edge_relays.sort(key=lambda x: -x[2])

    used_relays = 0
    for item in edge_relays:
        k_alloc = min(item[3], n_relays_available - used_relays)
        item[3] = k_alloc
        used_relays += k_alloc

    idx = 0
    while used_relays < n_relays_available:
        edge_relays[idx % len(edge_relays)][3] += 1
        used_relays += 1
        idx += 1

    placed_xy = [c for c in centroids]
    for u, v, d, k_alloc in edge_relays:
        for s in range(1, k_alloc + 1):
            alpha = s / (k_alloc + 1.0)
            placed_xy.append((1.0 - alpha) * nodes[u] + alpha * nodes[v])

    xy = np.clip(np.array(placed_xy[:env.n_uavs]), 70.0, env.area_size - 70.0)
    z = np.full((env.n_uavs, 1), env.h_opt)
    uav_coords = np.hstack([xy, z])
    return uav_coords, evaluate_mesh_topology(uav_coords, env)


def run_baseline_3_connected_greedy(env):
    r_cov_opt = env.ground_coverage_radius(env.h_opt)
    max_hop_2d = np.sqrt(max(100.0, (0.90 * env.r_comm_a2a) ** 2 - env.h_opt ** 2))
    angles = np.linspace(0, 2.0 * np.pi, 36, endpoint=False)
    radii_steps = np.linspace(0.45 * max_hop_2d, 0.95 * max_hop_2d, 5)

    placed_xy = []
    uncovered_mask = np.ones(len(env.users), dtype=bool)

    for _ in range(env.n_uavs):
        active_parents = [env.gbs_pos[:2]] + placed_xy
        candidates = []
        for parent in active_parents:
            for r_step in radii_steps:
                cx = parent[0] + r_step * np.cos(angles)
                cy = parent[1] + r_step * np.sin(angles)
                pts = np.column_stack([cx, cy])
                candidates.append(pts)

        candidates = np.vstack(candidates)
        valid_bounds = (
            (candidates[:, 0] >= 70.0) & (candidates[:, 0] <= env.area_size - 70.0)
            & (candidates[:, 1] >= 70.0) & (candidates[:, 1] <= env.area_size - 70.0)
        )
        candidates = candidates[valid_bounds]

        if len(placed_xy) > 0:
            d_existing = cdist(candidates, np.array(placed_xy))
            safe_mask = np.all(d_existing >= env.d_safe * 1.05, axis=1)
            candidates = candidates[safe_mask]

        uncov_users = env.users[uncovered_mask]
        if len(uncov_users) > 0 and len(candidates) > 0:
            d_users = cdist(candidates, uncov_users)
            gains = np.sum(d_users <= r_cov_opt, axis=1)
            best_gain = np.max(gains)
            if best_gain > 0:
                best_cand = candidates[int(np.argmax(gains))]
            else:
                target_user = uncov_users[int(np.argmin(np.linalg.norm(uncov_users - np.mean(uncov_users, axis=0), axis=1)))]
                d_to_target = np.linalg.norm(candidates - target_user, axis=1)
                best_cand = candidates[int(np.argmin(d_to_target))]
        else:
            if len(candidates) == 0:
                best_cand = np.array([env.gbs_pos[0], env.gbs_pos[1]], dtype=float)
            else:
                best_cand = candidates[0]

        placed_xy.append(best_cand)
        newly_cov = np.linalg.norm(env.users - best_cand, axis=1) <= r_cov_opt
        uncovered_mask[newly_cov] = False

    xy = np.array(placed_xy)
    z = np.full((env.n_uavs, 1), env.h_opt)
    uav_coords = np.hstack([xy, z])
    return uav_coords, evaluate_mesh_topology(uav_coords, env)
