import numpy as np

from src.graph_engine import evaluate_mesh_topology


class ConstrainedNSGA2Optimizer:
    def __init__(self, env, pop_size=100, n_gen=1000, seed=None):
        self.env = env
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.rng = np.random.default_rng(seed)
        self.n_vars = env.n_uavs * 3
        self.xl = np.tile([70.0, 70.0, env.h_min], env.n_uavs)
        self.xu = np.tile([env.area_size - 70.0, env.area_size - 70.0, env.h_max], env.n_uavs)

    def seed_population_kmeans_mst(self):
        from sklearn.cluster import KMeans

        n_k = min(self.env.n_clusters, self.env.n_uavs - 6)
        kmeans = KMeans(n_clusters=n_k, n_init=5, random_state=42)
        kmeans.fit(self.env.users)
        centroids = kmeans.cluster_centers_

        nodes = np.vstack([self.env.gbs_pos[:2], centroids])
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

        anchors = [c for c in centroids]
        mst_edges.sort(key=lambda x: -x[2])
        n_bridges = self.env.n_uavs - len(anchors)

        for idx in range(n_bridges):
            u, v, _ = mst_edges[idx % len(mst_edges)]
            alpha = 0.38 + 0.24 * ((idx // len(mst_edges)) % 2)
            perp = np.array([-(nodes[v, 1] - nodes[u, 1]), nodes[v, 0] - nodes[u, 0]])
            perp = perp / (np.linalg.norm(perp) + 1e-6)
            offset = (85.0 if idx % 2 == 0 else -85.0) * perp
            bridge_pt = (1.0 - alpha) * nodes[u] + alpha * nodes[v] + offset
            anchors.append(bridge_pt)

        anchors = np.array(anchors[:self.env.n_uavs])

        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.pop_size):
            noise_xy = 140.0 if i < int(0.75 * self.pop_size) else 260.0
            xy = anchors + self.rng.normal(0.0, noise_xy, size=(self.env.n_uavs, 2))
            xy = np.clip(xy, 80.0, self.env.area_size - 80.0)
            z = self.rng.uniform(self.env.h_min, self.env.h_max, size=(self.env.n_uavs, 1))
            pop[i] = np.hstack([xy, z]).flatten()
        return pop

    def evaluate_population(self, pop):
        n = len(pop)
        F = np.zeros((n, 2))
        CV = np.zeros(n)
        for i in range(n):
            coords = pop[i].reshape((self.env.n_uavs, 3))
            m = evaluate_mesh_topology(coords, self.env)
            F[i, 0] = 1.0 - m["coverage_ratio"]
            F[i, 1] = -m["algebraic_connectivity"]
            CV[i] = m["cv"]
        return F, CV

    @staticmethod
    def fast_non_dominated_sort(F, CV):
        n = len(F)
        cv_p = CV[:, np.newaxis]
        cv_q = CV[np.newaxis, :]

        f0_p = F[:, 0:1]
        f1_p = F[:, 1:2]
        f0_q = F[:, 0:1].T
        f1_q = F[:, 1:2].T

        dominates = (
            (f0_p <= f0_q) & (f1_p <= f1_q) & ((f0_p < f0_q) | (f1_p < f1_q))
        )
        obj_dom = dominates
        dom_matrix = (cv_p < cv_q) | ((cv_p == cv_q) & obj_dom)

        n_dom = np.sum(dom_matrix, axis=0).astype(int)
        fronts = []
        current_front = np.flatnonzero(n_dom == 0)
        rank = 0

        while len(current_front) > 0:
            fronts.append(current_front.tolist())
            n_dom[np.asarray(current_front)] = -1
            for j in current_front:
                n_dom = n_dom - dom_matrix[j, :]
            current_front = np.flatnonzero(n_dom == 0)
            rank += 1

        return fronts, np.array([0] * n, dtype=int)

    @staticmethod
    def compute_crowding_distance(F, front_indices):
        k = len(front_indices)
        if k <= 2:
            return np.full(k, np.inf)
        distances = np.zeros(k)
        sub_F = F[front_indices]

        for m in range(F.shape[1]):
            order = np.argsort(sub_F[:, m])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            f_range = sub_F[order[-1], m] - sub_F[order[0], m]
            if f_range > 1e-9:
                distances[order[1:-1]] += (sub_F[order[2:], m] - sub_F[order[:-2], m]) / f_range
        return distances

    def tournament_selection(self, pop, ranks, crowding):
        n = len(pop)
        a = self.rng.integers(0, n, size=n)
        b = self.rng.integers(0, n, size=n)
        choose_a = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowding[a] >= crowding[b]))
        winners = np.where(choose_a, a, b)
        return pop[winners].copy()

    def sbx_crossover_and_polynomial_mutation(self, parents, eta_c=15.0, eta_m=20.0):
        offspring = parents.copy()
        n = len(parents)

        for i in range(0, n - 1, 2):
            if self.rng.random() < 0.9:
                u = self.rng.random(self.n_vars)
                beta = np.where(
                    u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta_c + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
                )
                p1, p2 = parents[i], parents[i + 1]
                offspring[i] = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                offspring[i + 1] = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

        prob_m = 1.5 / self.n_vars
        mask = self.rng.random(offspring.shape) < prob_m
        r = self.rng.random(offspring.shape)
        delta = np.where(
            r < 0.5,
            (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0)),
        )
        span = (self.xu - self.xl)[np.newaxis, :]
        offspring = np.where(mask, offspring + delta * span * 0.20, offspring)
        return np.clip(offspring, self.xl, self.xu)

    def run(self):
        pop = self.seed_population_kmeans_mst()
        F, CV = self.evaluate_population(pop)
        fronts, ranks = self.fast_non_dominated_sort(F, CV)
        crowding = np.zeros(self.pop_size)
        for front in fronts:
            crowding[front] = self.compute_crowding_distance(F, front)

        history_best_cov = []
        for gen in range(self.n_gen):
            feasible_mask = CV == 0.0
            if np.any(feasible_mask):
                history_best_cov.append((1.0 - np.min(F[feasible_mask, 0])) * 100.0)
            else:
                best_idx = np.argmin(CV)
                history_best_cov.append((1.0 - F[best_idx, 0]) * 100.0)

            parents = self.tournament_selection(pop, ranks, crowding)
            offspring = self.sbx_crossover_and_polynomial_mutation(parents)
            off_F, off_CV = self.evaluate_population(offspring)

            comb_pop = np.vstack([pop, offspring])
            comb_F = np.vstack([F, off_F])
            comb_CV = np.concatenate([CV, off_CV])

            # simplified nsga-ii iteration
            valid_fronts = []
            remaining = np.arange(len(comb_pop))
            while len(remaining) > 0:
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if (comb_F[j, 0] <= comb_F[i, 0] and comb_F[j, 1] <= comb_F[i, 1]
                                and (comb_F[j, 0] < comb_F[i, 0] or comb_F[j, 1] < comb_F[i, 1])):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                if not front:
                    break
                valid_fronts.append(np.array(front, dtype=int))
                remaining = np.array([idx for idx in remaining if idx not in front], dtype=int)

            next_indices = []
            for front in valid_fronts:
                if len(next_indices) + len(front) <= self.pop_size:
                    next_indices.extend(front.tolist())
                else:
                    rem = self.pop_size - len(next_indices)
                    cd = self.compute_crowding_distance(comb_F, front)
                    order = np.argsort(-cd)[:rem]
                    next_indices.extend(front[order].tolist())
                    break

            pop = comb_pop[next_indices]
            F = comb_F[next_indices]
            CV = comb_CV[next_indices]
            ranks = np.zeros(len(pop), dtype=int)
            crowding = np.zeros(len(pop), dtype=float)

            if (gen + 1) % 100 == 0 or gen == 0:
                n_feas = int(np.sum(CV == 0.0))
                print(f"    [Gen {gen + 1:4d}/{self.n_gen}] Best Coverage: {history_best_cov[-1]:.1f}% | Feasible Pop: {n_feas}/{self.pop_size}")

        return pop, F, CV, history_best_cov
