import numpy as np
from sklearn.cluster import KMeans

from src.graph_engine import evaluate_mesh_topology


class ConstrainedNSGA2Optimizer:
    """NSGA-II with feasibility-first constraint dominance and vectorized sorting."""

    def __init__(self, env, pop_size=100, n_gen=1000, seed=None):
        self.env, self.pop_size, self.n_gen = env, int(pop_size), int(n_gen)
        self.rng = np.random.default_rng(seed)
        self.n_vars = env.n_uavs * 3
        self.xl = np.tile([70.0, 70.0, env.h_min], env.n_uavs)
        self.xu = np.tile([env.area_size - 70.0, env.area_size - 70.0, env.h_max], env.n_uavs)

    def seed_population_kmeans_mst(self):
        n_k = min(self.env.n_clusters, self.env.n_uavs - 6)
        centers = KMeans(n_clusters=n_k, n_init=5, random_state=42).fit(self.env.users).cluster_centers_
        nodes = np.vstack((self.env.gbs_pos[:2], centers))
        in_tree, remaining, edges = [0], set(range(1, len(nodes))), []
        while remaining:
            u, v = min(((u, v) for u in in_tree for v in remaining),
                       key=lambda pair: np.linalg.norm(nodes[pair[0]] - nodes[pair[1]]))
            in_tree.append(v)
            remaining.remove(v)
            edges.append((u, v))
        anchors = [c.copy() for c in centers]
        edges.sort(key=lambda e: -np.linalg.norm(nodes[e[0]] - nodes[e[1]]))
        n_bridges = self.env.n_uavs - n_k
        for idx in range(n_bridges):
            u, v = edges[idx % len(edges)]
            alpha = 0.38 + 0.24 * ((idx // len(edges)) % 2)
            direction = nodes[v] - nodes[u]
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular /= np.linalg.norm(perpendicular) + 1e-9
            anchors.append((1.0 - alpha) * nodes[u] + alpha * nodes[v] + (85.0 if idx % 2 == 0 else -85.0) * perpendicular)
        anchors = np.asarray(anchors[:self.env.n_uavs])

        population = np.empty((self.pop_size, self.n_vars))
        for i in range(self.pop_size):
            noise = 140.0 if i < int(0.75 * self.pop_size) else 260.0
            xy = np.clip(anchors + self.rng.normal(0.0, noise, (self.env.n_uavs, 2)), 80.0, self.env.area_size - 80.0)
            z = self.rng.uniform(self.env.h_min, self.env.h_max, (self.env.n_uavs, 1))
            population[i] = np.column_stack((xy, z)).ravel()
        return population

    def evaluate_population(self, population):
        objectives = np.zeros((len(population), 2))
        violations = np.zeros(len(population))
        for i, individual in enumerate(population):
            metrics = evaluate_mesh_topology(individual.reshape(self.env.n_uavs, 3), self.env)
            objectives[i] = (1.0 - metrics["coverage_ratio"], -metrics["algebraic_connectivity"])
            violations[i] = metrics["cv"]
        return objectives, violations

    @staticmethod
    def fast_non_dominated_sort(F, CV):
        n = len(F)
        objective_dominates = ((F[:, None, :] <= F[None, :, :]).all(axis=2)
                               & (F[:, None, :] < F[None, :, :]).any(axis=2))
        dominates = ((CV[:, None] < CV[None, :])
                     | ((CV[:, None] == CV[None, :]) & objective_dominates))
        domination_count = dominates.sum(axis=0).astype(int)
        ranks = np.full(n, -1, dtype=int)
        fronts = []
        current = np.flatnonzero(domination_count == 0)
        rank = 0
        while len(current):
            fronts.append(current.tolist())
            ranks[current] = rank
            domination_count -= dominates[current].sum(axis=0)
            domination_count[current] = -1
            current = np.flatnonzero(domination_count == 0)
            rank += 1
        return fronts, ranks

    @staticmethod
    def compute_crowding_distance(F, front_indices):
        k = len(front_indices)
        if k <= 2:
            return np.full(k, np.inf)
        distances = np.zeros(k)
        values = F[front_indices]
        for objective in range(F.shape[1]):
            order = np.argsort(values[:, objective])
            distances[order[[0, -1]]] = np.inf
            span = values[order[-1], objective] - values[order[0], objective]
            if span > 1e-12:
                distances[order[1:-1]] += (values[order[2:], objective] - values[order[:-2], objective]) / span
        return distances

    def _rank_and_crowding(self, F, CV):
        fronts, ranks = self.fast_non_dominated_sort(F, CV)
        crowding = np.zeros(len(F))
        for front in fronts:
            crowding[front] = self.compute_crowding_distance(F, front)
        return fronts, ranks, crowding

    def tournament_selection(self, population, ranks, crowding):
        a = self.rng.integers(0, len(population), len(population))
        b = self.rng.integers(0, len(population), len(population))
        choose_a = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowding[a] >= crowding[b]))
        return population[np.where(choose_a, a, b)].copy()

    def sbx_crossover_and_polynomial_mutation(self, parents, eta_c=15.0, eta_m=20.0):
        children = parents.copy()
        for i in range(0, len(parents) - 1, 2):
            if self.rng.random() < 0.9:
                u = self.rng.random(self.n_vars)
                beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta_c + 1.0)),
                                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))
                p1, p2 = parents[i].copy(), parents[i + 1].copy()
                children[i] = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                children[i + 1] = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        mask = self.rng.random(children.shape) < (1.5 / self.n_vars)
        r = self.rng.random(children.shape)
        delta = np.where(r < 0.5, (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0,
                         1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0)))
        children = np.where(mask, children + delta * (self.xu - self.xl)[None, :] * 0.20, children)
        return np.clip(children, self.xl, self.xu)

    def run(self):
        population = self.seed_population_kmeans_mst()
        F, CV = self.evaluate_population(population)
        _, ranks, crowding = self._rank_and_crowding(F, CV)
        history = []
        for generation in range(self.n_gen):
            feasible = CV == 0.0
            best = np.min(F[feasible, 0]) if feasible.any() else F[np.argmin(CV), 0]
            history.append((1.0 - best) * 100.0)
            offspring = self.sbx_crossover_and_polynomial_mutation(self.tournament_selection(population, ranks, crowding))
            off_F, off_CV = self.evaluate_population(offspring)
            combined = np.vstack((population, offspring))
            combined_F = np.vstack((F, off_F))
            combined_CV = np.concatenate((CV, off_CV))
            fronts, combined_ranks = self.fast_non_dominated_sort(combined_F, combined_CV)
            selected = []
            next_crowding = []
            for front in fronts:
                cd = self.compute_crowding_distance(combined_F, front)
                if len(selected) + len(front) <= self.pop_size:
                    selected.extend(front)
                    next_crowding.extend(cd)
                else:
                    remaining = self.pop_size - len(selected)
                    order = np.argsort(-cd)[:remaining]
                    selected.extend(np.asarray(front)[order].tolist())
                    next_crowding.extend(cd[order])
                    break
            selected = np.asarray(selected, dtype=int)
            population, F, CV = combined[selected], combined_F[selected], combined_CV[selected]
            ranks = combined_ranks[selected]
            crowding = np.asarray(next_crowding)
            if generation == 0 or (generation + 1) % 100 == 0:
                print(f"    [Gen {generation + 1:4d}/{self.n_gen}] Best coverage: {history[-1]:.1f}% | Feasible: {(CV == 0).sum()}/{self.pop_size}")
        return population, F, CV, history
