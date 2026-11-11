import numpy as np
from scipy.spatial.distance import cdist


def hopcroft_tarjan_biconnectivity(adj_matrix):
    n_nodes = adj_matrix.shape[0]
    tin = np.full(n_nodes, -1, dtype=int)
    low = np.full(n_nodes, -1, dtype=int)
    visited = np.zeros(n_nodes, dtype=bool)
    art_points = set()
    timer = 0
    adj_list = [np.flatnonzero(adj_matrix[u]) for u in range(n_nodes)]

    def dfs(u, parent, comp_accumulator):
        nonlocal timer
        visited[u] = True
        comp_accumulator.append(u)
        tin[u] = low[u] = timer
        timer += 1
        children_count = 0

        for v in adj_list[u]:
            if v == parent:
                continue
            if visited[v]:
                if tin[v] < low[u]:
                    low[u] = tin[v]
            else:
                children_count += 1
                dfs(v, u, comp_accumulator)
                if low[v] < low[u]:
                    low[u] = low[v]
                if parent != -1 and low[v] >= tin[u]:
                    art_points.add(u)

        if parent == -1 and children_count > 1:
            art_points.add(u)

    components = []
    for start_node in range(n_nodes):
        if not visited[start_node]:
            comp = []
            dfs(start_node, -1, comp)
            components.append(comp)

    gbs_component = components[0] if len(components) > 0 else []
    return len(components), gbs_component, sorted(list(art_points)), tin, low


def evaluate_mesh_topology(uav_coords, env):
    n_uavs = len(uav_coords)
    all_nodes = np.vstack([env.gbs_pos, uav_coords])
    dist_matrix = cdist(all_nodes, all_nodes)
    adj_matrix = (dist_matrix <= env.r_comm_a2a) & (dist_matrix > 0.0)

    n_components, gbs_comp, art_points, tin, low = hopcroft_tarjan_biconnectivity(adj_matrix)
    active_uav_indices = [node - 1 for node in gbs_comp if node > 0]

    if n_components == 1 and len(all_nodes) >= 3:
        degrees = np.sum(adj_matrix, axis=1).astype(float)
        laplacian = np.diag(degrees) - adj_matrix.astype(float)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        alg_conn = float(max(0.0, eigenvalues[1]))
        n_articulation = len(art_points)
    else:
        alg_conn = 0.0
        n_articulation = len(art_points) + (n_components - 1) * 3

    all_radii = env.ground_coverage_radius(uav_coords[:, 2])
    all_horiz_dists = cdist(env.users, uav_coords[:, :2])
    raw_covered_mask = np.any(all_horiz_dists <= all_radii[np.newaxis, :], axis=1)
    raw_coverage_ratio = float(np.mean(raw_covered_mask))

    if len(active_uav_indices) > 0:
        conn_uavs = uav_coords[active_uav_indices]
        radii = env.ground_coverage_radius(conn_uavs[:, 2])
        horiz_dists = cdist(env.users, conn_uavs[:, :2])
        covered_mask = np.any(horiz_dists <= radii[np.newaxis, :], axis=1)
        coverage_ratio = float(np.mean(covered_mask))
    else:
        covered_mask = np.zeros(len(env.users), dtype=bool)
        coverage_ratio = 0.0

    uav_dists = dist_matrix[1:, 1:].copy()
    np.fill_diagonal(uav_dists, np.inf)
    sep_violation = float(np.sum(np.maximum(0.0, env.d_safe - uav_dists)) / 2.0)

    cv = 100.0 * (n_components - 1) + 25.0 * n_articulation + sep_violation
    edges = [(i, j) for i in range(n_uavs + 1) for j in range(i + 1, n_uavs + 1) if adj_matrix[i, j]]

    return {
        "edges": edges,
        "raw_coverage_ratio": raw_coverage_ratio,
        "coverage_ratio": coverage_ratio,
        "covered_mask": covered_mask,
        "active_uav_indices": active_uav_indices,
        "n_components": n_components,
        "n_articulation": len(art_points),
        "penalty_articulation": n_articulation,
        "articulation_nodes": set(art_points),
        "algebraic_connectivity": alg_conn,
        "sep_violation": sep_violation,
        "cv": cv,
        "is_biconnected": (n_components == 1 and len(art_points) == 0),
    }
