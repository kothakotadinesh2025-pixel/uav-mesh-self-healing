import numpy as np
from scipy.spatial.distance import cdist


def hopcroft_tarjan_biconnectivity(adj_matrix):
    """Return component count, GBS component, articulation vertices, tin and low."""
    n = len(adj_matrix)
    tin = np.full(n, -1, dtype=int)
    low = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    articulation = set()
    adjacency = [np.flatnonzero(adj_matrix[i]) for i in range(n)]
    timer = 0

    def dfs(u, parent, component):
        nonlocal timer
        visited[u] = True
        component.append(u)
        tin[u] = low[u] = timer
        timer += 1
        children = 0
        for v in adjacency[u]:
            if v == parent:
                continue
            if visited[v]:
                low[u] = min(low[u], tin[v])
            else:
                children += 1
                dfs(v, u, component)
                low[u] = min(low[u], low[v])
                if parent != -1 and low[v] >= tin[u]:
                    articulation.add(u)
        if parent == -1 and children > 1:
            articulation.add(u)

    components = []
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(node, -1, component)
            components.append(component)
    gbs_component = next((c for c in components if 0 in c), [])
    return len(components), gbs_component, sorted(articulation), tin, low


def evaluate_mesh_topology(uav_coords, env):
    uav_coords = np.asarray(uav_coords, dtype=float)
    all_nodes = np.vstack((env.gbs_pos, uav_coords))
    distances = cdist(all_nodes, all_nodes)
    adjacency = (distances <= env.r_comm_a2a) & (distances > 0.0)
    n_components, gbs_component, articulation, tin, low = hopcroft_tarjan_biconnectivity(adjacency)
    active = [node - 1 for node in gbs_component if node > 0]

    if n_components == 1 and len(all_nodes) >= 3:
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency.astype(float)
        algebraic_connectivity = float(max(0.0, np.linalg.eigvalsh(laplacian)[1]))
    else:
        algebraic_connectivity = 0.0

    radii = env.ground_coverage_radius(uav_coords[:, 2])
    local_distances = cdist(env.users, uav_coords[:, :2])
    raw_mask = np.any(local_distances <= radii[None, :], axis=1)
    if active:
        connected_distances = local_distances[:, active]
        connected_radii = radii[active]
        covered_mask = np.any(connected_distances <= connected_radii[None, :], axis=1)
    else:
        covered_mask = np.zeros(len(env.users), dtype=bool)

    uav_distances = distances[1:, 1:].copy()
    np.fill_diagonal(uav_distances, np.inf)
    separation_violation = float(np.maximum(0.0, env.d_safe - uav_distances).sum() / 2.0)
    effective_cuts = len(articulation) + max(0, n_components - 1) * 3
    cv = 100.0 * max(0, n_components - 1) + 25.0 * effective_cuts + separation_violation
    edges = [(i, j) for i in range(len(all_nodes)) for j in range(i + 1, len(all_nodes)) if adjacency[i, j]]

    return {
        "edges": edges,
        "raw_coverage_ratio": float(raw_mask.mean()) if len(raw_mask) else 0.0,
        "coverage_ratio": float(covered_mask.mean()) if len(covered_mask) else 0.0,
        "covered_mask": covered_mask,
        "active_uav_indices": active,
        "n_components": n_components,
        "n_articulation": len(articulation),
        "penalty_articulation": effective_cuts,
        "articulation_nodes": set(articulation),
        "algebraic_connectivity": algebraic_connectivity,
        "sep_violation": separation_violation,
        "cv": cv,
        "is_biconnected": n_components == 1 and not articulation,
    }
