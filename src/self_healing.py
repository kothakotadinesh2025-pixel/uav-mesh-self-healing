import numpy as np

from src.graph_engine import evaluate_mesh_topology


class SelfHealingController:
    def __init__(self, env, step_size=18.0, seed=None):
        self.env, self.step_size = env, step_size
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _reward(metrics):
        return (160.0 * metrics["coverage_ratio"] + 14.0 * metrics["algebraic_connectivity"]
                - 110.0 * (metrics["n_components"] - 1)
                - 38.0 * metrics["penalty_articulation"] - 2.0 * metrics["sep_violation"])

    def _policy_gradient(self, coords):
        metrics = evaluate_mesh_topology(coords, self.env)
        uncovered = self.env.users[~metrics["covered_mask"]]
        forces = np.zeros_like(coords)
        nodes = np.vstack((self.env.gbs_pos, coords))
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=bool)
        for u, v in metrics["edges"]:
            adjacency[u, v] = adjacency[v, u] = True

        for i, position in enumerate(coords):
            if len(uncovered):
                nearby = uncovered[np.linalg.norm(uncovered - position[:2], axis=1) < 680.0]
                if len(nearby):
                    vector = nearby.mean(axis=0) - position[:2]
                    forces[i, :2] += 0.55 * vector / (np.linalg.norm(vector) + 1e-6) * 18.0
            node_id = i + 1
            if adjacency[node_id].sum() < 2 or metrics["n_components"] > 1 or metrics["n_articulation"]:
                distances = np.linalg.norm(nodes[:, :2] - position[:2], axis=1)
                distances[node_id] = np.inf
                other = int(np.argmin(distances))
                vector = nodes[other, :2] - position[:2]
                forces[i, :2] += 0.95 * vector / (np.linalg.norm(vector) + 1e-6) * 22.0
            for j in range(len(coords)):
                if i == j:
                    continue
                vector = position[:2] - coords[j, :2]
                distance = np.linalg.norm(vector)
                if distance < self.env.d_safe * 1.35:
                    forces[i, :2] += vector / (distance + 1e-6) * 25.0
            forces[i, 2] = 0.30 * (self.env.h_opt - position[2])
        return forces

    def heal_mesh(self, surviving_coords, episodes=95):
        best = np.asarray(surviving_coords, dtype=float).copy()
        best_metrics = evaluate_mesh_topology(best, self.env)
        best_reward = self._reward(best_metrics)
        current, momentum = best.copy(), np.zeros_like(best)
        for episode in range(episodes):
            force = self._policy_gradient(current)
            noise = self.rng.normal(0.0, max(2.0, 9.0 * (1.0 - episode / max(episodes, 1))), current.shape)
            noise[:, 2] *= 0.25
            momentum = 0.65 * momentum + 0.35 * (force + noise)
            norms = np.linalg.norm(momentum[:, :2], axis=1, keepdims=True) + 1e-6
            step_xy = momentum[:, :2] / norms * np.minimum(self.step_size, norms)
            candidate = current.copy()
            candidate[:, :2] = np.clip(candidate[:, :2] + step_xy, 70.0, self.env.area_size - 70.0)
            candidate[:, 2] = np.clip(candidate[:, 2] + np.clip(momentum[:, 2], -3.0, 3.0), self.env.h_min, self.env.h_max)
            candidate_metrics = evaluate_mesh_topology(candidate, self.env)
            reward = self._reward(candidate_metrics)
            if reward >= best_reward:
                best, best_metrics, best_reward = candidate.copy(), candidate_metrics, reward
                current = candidate.copy()
            else:
                current = 0.72 * best + 0.28 * candidate
        return best, best_metrics
