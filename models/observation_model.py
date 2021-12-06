import numpy as np


class ObservationModel:
    def __init__(self, state_dimension: int, env_size: int):
        self.state_dimension = state_dimension
        self.env_size = env_size

    def measure(self, particle_positions: np.ndarray, beacon_positions: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=len(beacon_positions) * len(particle_positions))

        r_a = np.repeat(particle_positions, repeats=len(beacon_positions), axis=0)
        r_b = np.tile(beacon_positions, (len(particle_positions), 1))

        measured_distances = np.reshape(np.linalg.norm(r_a - r_b, axis=1) + noise, (len(particle_positions), len(beacon_positions)))

        return measured_distances
