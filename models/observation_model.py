import numpy as np


class ObservationModel:
    def __init__(self, state_dimension: int, env_size: int):
        self.state_dimension = state_dimension
        self.env_size = env_size

    def measure(self, particle_positions: np.ndarray, beacon_positions: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
        """
        Measures distance from each particle to each beacon

        :param particle_positions: Positions of all particles
        :param beacon_positions: Positions of all beacons
        :param noise: Special type of noise; if not given: use default zero-mean Gaussian noise with std=0.1
        :return: Matrix of distances -> rows represent particles, columns - Euclidean distance to each beacon
        """
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=len(beacon_positions) * len(particle_positions))
        else:
            assert noise.shape == (len(beacon_positions) * len(particle_positions),)

        # make both arrays of the same size:
        particles_repeated = np.repeat(
            particle_positions, repeats=len(beacon_positions), axis=0
        )  # each particle pos. should be repeated #beacons times
        beacon_repeated = np.tile(
            beacon_positions, (len(particle_positions), 1)
        )  # all beacon positions are available for all particles

        measured_distances = np.reshape(
            np.linalg.norm(particles_repeated - beacon_repeated, axis=1) + noise,
            (len(particle_positions), len(beacon_positions)),
        )

        return measured_distances
