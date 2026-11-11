from dataclasses import dataclass


@dataclass
class SimConfig:
    area_size: float = 3000.0
    h_min: float = 60.0
    h_max: float = 220.0
    n_uavs: int = 22
    n_users: int = 1800
    r_comm_a2a: float = 720.0
    d_safe: float = 65.0
    pop_size: int = 100
    n_gen: int = 1000
    cluster_min: int = 12
    cluster_max: int = 16
    rng_seed: int | None = None


def make_default_config() -> SimConfig:
    return SimConfig()
