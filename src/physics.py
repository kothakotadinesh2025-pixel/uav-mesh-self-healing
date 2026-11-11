import numpy as np

from src.config import SimConfig


class DisasterEnvironment:
    """Fixed disaster layout and precomputed Al-Hourani A2G coverage model."""

    def __init__(self, config: SimConfig | None = None, seed: int | None = None):
        self.config = config or SimConfig()
        effective_seed = seed if seed is not None else self.config.rng_seed
        self.rng = np.random.default_rng(effective_seed)
        self.area_size = self.config.area_size
        self.h_min, self.h_max = self.config.h_min, self.config.h_max
        self.n_uavs, self.n_users = self.config.n_uavs, self.config.n_users
        self.r_comm_a2a, self.d_safe = self.config.r_comm_a2a, self.config.d_safe
        self.l_th_db = self.config.l_th_db
        self.n_clusters = int(self.rng.integers(self.config.cluster_min, self.config.cluster_max + 1))

        gbs_xy = self.rng.uniform(180.0, 420.0, size=2)
        self.gbs_pos = np.array([*gbs_xy, 0.0])
        self._precompute_al_hourani_radius_curve()
        self.users = self._generate_random_survivors()

    def _precompute_al_hourani_radius_curve(self):
        a_los, b_los = 9.61, 0.16
        eta_los, eta_nlos = 1.0, 20.0
        fc_hz, c_light = 2.0e9, 3.0e8
        self.h_grid = np.linspace(self.h_min, self.h_max, 321)
        r_grid = np.linspace(1.0, 600.0, 1000)
        h = self.h_grid[:, None]
        r = r_grid[None, :]
        distance = np.hypot(h, r)
        elevation = np.degrees(np.arctan2(h, r))
        p_los = 1.0 / (1.0 + a_los * np.exp(-b_los * (elevation - a_los)))
        fspl = 20.0 * np.log10(4.0 * np.pi * fc_hz * distance / c_light)
        path_loss = fspl + p_los * eta_los + (1.0 - p_los) * eta_nlos
        valid = path_loss <= self.l_th_db
        last_valid_index = np.where(valid.any(axis=1), valid.sum(axis=1) - 1, -1)
        self.r_cov_lookup = np.where(last_valid_index >= 0, r_grid[np.maximum(last_valid_index, 0)], 0.0)
        self.h_opt = float(self.h_grid[np.argmax(self.r_cov_lookup)])

    def ground_coverage_radius(self, altitudes):
        return np.interp(altitudes, self.h_grid, self.r_cov_lookup)

    def _generate_random_survivors(self):
        n_scattered = int(0.25 * self.n_users)
        n_clustered = self.n_users - n_scattered
        centers = self.rng.uniform(450.0, self.area_size - 220.0, size=(self.n_clusters, 2))
        counts = self.rng.multinomial(n_clustered, self.rng.dirichlet(np.ones(self.n_clusters)))
        groups = []
        for center, count in zip(centers, counts):
            spread = self.rng.uniform(95.0, 185.0, size=2)
            groups.append(self.rng.normal(center, spread, size=(count, 2)))
        groups.append(self.rng.uniform(100.0, self.area_size - 100.0, size=(n_scattered, 2)))
        return np.clip(np.vstack(groups), 60.0, self.area_size - 60.0)
