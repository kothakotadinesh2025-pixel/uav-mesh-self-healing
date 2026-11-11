import numpy as np

from src.graph_engine import evaluate_mesh_topology


class SelfHealingController:
    def __init__(self, env, step_size=18.0, seed=None):
        self.env = env
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

    def _reward(self, metrics):
        return (
            160.0 * metrics["coverage_ratio"]
            + 14.0 * metrics["algebraic_connectivity"]
            - 110.0 * (metrics["n_components"] - 1)
            - 38.0 * metrics["penalty_articulation"]
            - 2.0 * metrics["sep_violation"]
        )

    def _policy_gradient(self, coords):
        metrics = evaluate_mesh_topology(coords, self.env)
        uncovered = self.env.users[~metrics["covered_mask"]]
        forces = np.zeros_like(coords)

        all_nodes = np.vstack([self.env.gbs_pos, coords])
        adj = np.zeros((len(coords) + 1, len(coords) + 1), dtype=bool)
        for u, v in metrics["edges"]:
            adj[u, v] = True
            adj[v, u] = True

        for i in range(len(coords)):
            pos_i = coords[i, :2]

            if len(uncovered) > 0:
                d_u = np.linalg.norm(uncovered - pos_i, axis=1)
                near_u = uncovered[d_u < 680.0]
                if len(near_u) > 0:
                    vec = np.mean(near_u, axis=0) - pos_i
                    norm = np.linalg.norm(vec) + 1e-6
                    forces[i, :2] += 0.55 * (vec / norm) * 18.0

            node_id = i + 1
            if np.sum(adj[node_id]) < 2 or metrics["n_components"] > 1 or metrics["n_articulation"] > 0:
                dists = np.linalg.norm(all_nodes[:, :2] - pos_i, axis=1)
                dists[node_id] = np.inf
                nearest = np.argmin(dists)
                if np.isfinite(dists[nearest]):
                    vec = all_nodes[nearest, :2] - pos_i
                    norm = np.linalg.norm(vec) + 1e-6
                    forces[i, :2] += 0.95 * (vec / norm) * 22.0

            for j in range(len(coords)):
                if i == j:
                    continue
                diff = pos_i - coords[j, :2]
                dist = np.linalg.norm(diff)
                if dist < self.env.d_safe * 1.35:
                    forces[i, :2] += (diff / (dist + 1e-6)) * 25.0

            forces[i, 2] = 0.30 * (self.env.h_opt - coords[i, 2])

        return forces

    def heal_mesh(self, surviving_coords, episodes=95):
        best_coords = surviving_coords.copy()
        best_metrics = evaluate_mesh_topology(best_coords, self.env)
        best_reward = self._reward(best_metrics)

        curr = surviving_coords.copy()
        momentum = np.zeros_like(curr)

        for ep in range(episodes):
            grad = self._policy_gradient(curr)
            sigma = max(2.0, 9.0 * (1.0 - ep / episodes))
            noise = self.rng.normal(0.0, sigma, size=curr.shape)
            noise[:, 2] *= 0.25

            momentum = 0.65 * momentum + 0.35 * (grad + noise)
            norms = np.linalg.norm(momentum[:, :2], axis=1, keepdims=True) + 1e-6
            step_xy = (momentum[:, :2] / norms) * np.minimum(self.step_size, norms)
            step_z = np.clip(momentum[:, 2:3], -3.0, 3.0)

            cand = curr.copy()
            cand[:, :2] = np.clip(cand[:, :2] + step_xy, 70.0, self.env.area_size - 70.0)
            cand[:, 2:3] = np.clip(cand[:, 2:3] + step_z, self.env.h_min, self.env.h_max)

            cand_metrics = evaluate_mesh_topology(cand, self.env)
            cand_reward = self._reward(cand_metrics)

            if cand_reward >= best_reward:
                best_reward = cand_reward
                best_coords = cand.copy()
                best_metrics = cand_metrics
                curr = cand.copy()
            else:
                curr = 0.72 * best_coords + 0.28 * cand

        return best_coords, best_metrics
