import numpy as np

from src.config import SimConfig


class DisasterEnvironment:
    def __init__(self, config: SimConfig | None = None, seed: int | None = None):
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(seed if seed is not None else self.config.rng_seed)

        self.area_size = self.config.area_size
        self.h_min = self.config.h_min
        self.h_max = self.config.h_max
        self.n_uavs = self.config.n_uavs
        self.n_users = self.config.n_users
        self.r_comm_a2a = self.config.r_comm_a2a
        self.d_safe = self.config.d_safe
        self.n_clusters = int(self.rng.integers(self.config.cluster_min, self.config.cluster_max + 1))

        self.a_los = 9.61
        self.b_los = 0.16
        self.eta_los = 1.0
        self.eta_nlos = 20.0
        self.fc_hz = 2.0e9
        self.c_light = 3.0e8
        self.l_th_db = 107.5

        gbs_xy = self.rng.uniform(180.0, 420.0, size=2)
        self.gbs_pos = np.array([gbs_xy[0], gbs_xy[1], 0.0], dtype=float)

        self._precompute_al_hourani_radius_curve()
        self.users = self._generate_random_survivors()

    def _precompute_al_hourani_radius_curve(self):
        self.h_grid = np.linspace(self.h_min, self.h_max, 321)
        r_test = np.linspace(1.0, 600.0, 1000)[np.newaxis, :]
        h_col = self.h_grid[:, np.newaxis]

        d_3d = np.sqrt(h_col**2 + r_test**2)
        theta_deg = np.degrees(np.arctan2(h_col, r_test))

        p_los = 1.0 / (1.0 + self.a_los * np.exp(-self.b_los * (theta_deg - self.a_los)))
        fspl = 20.0 * np.log10(4.0 * np.pi * self.fc_hz * d_3d / self.c_light)
        l_a2g = fspl + p_los * self.eta_los + (1.0 - p_los) * self.eta_nlos

        valid_mask = l_a2g <= self.l_th_db
        self.r_cov_lookup = np.where(
            np.any(valid_mask, axis=1),
            r_test[0, np.argmax(np.where(valid_mask, r_test, 0.0), axis=1)],
            90.0,
        )
        self.h_opt = float(self.h_grid[np.argmax(self.r_cov_lookup)])

    def ground_coverage_radius(self, altitudes):
        return np.interp(altitudes, self.h_grid, self.r_cov_lookup)

    def _generate_random_survivors(self):
        n_scattered = int(0.25 * self.n_users)
        n_clustered = self.n_users - n_scattered
        centers = self.rng.uniform(300.0, self.area_size - 300.0, size=(self.n_clusters, 2))
        weights = self.rng.dirichlet(np.ones(self.n_clusters))
        counts = self.rng.multinomial(n_clustered, weights)

        users_list = []
        for (cx, cy), count in zip(centers, counts):
            spread_x = self.rng.uniform(95.0, 185.0)
            spread_y = self.rng.uniform(95.0, 185.0)
            pts = self.rng.normal(loc=[cx, cy], scale=[spread_x, spread_y], size=(count, 2))
            users_list.append(pts)

        outliers = self.rng.uniform(100.0, self.area_size - 100.0, size=(n_scattered, 2))
        users_list.append(outliers)

        all_users = np.vstack(users_list)
        return np.clip(all_users, 60.0, self.area_size - 60.0)
