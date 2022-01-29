import numpy as np
import torch.nn as nn
import torch


class Lambda(nn.Module):
    def __init__(self, func, dfunc):
        super().__init__()
        self.func = func
        self.dfunc = dfunc

    def forward(self, x):
        return self.func(x)

    # not needed
    # def backward(self, x):
    #     return self.dfunc(x)


class ObservationModel(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        env_size: int,
        num_particles: int = 100,
        num_beacons: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dimension = state_dimension
        self.env_size = env_size
        self.num_particles_ = num_particles
        self.num_beacons = num_beacons
        # inspired by the paper
        self.min_obs_likelihood = 0.004

        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_dimension + (num_beacons * 3), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
            Lambda(
                lambda x: x * (1 - self.min_obs_likelihood) + self.min_obs_likelihood,
                lambda: 1 - self.min_obs_likelihood,
            ),
        )

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        # to pass asserts
        return torch.squeeze(out, -1)

    def prepare_input(
        self, particle_states, beacon_positions, measurement
    ) -> torch.Tensor:
        N = particle_states.shape[0]  # batch size
        M = particle_states.shape[1]  # number of particles
        beacon_repeated = torch.tile(
            beacon_positions.clone().reshape(N, -1)[:, None], (1, M, 1)
        )
        measurement_repeated = torch.tile(measurement.clone()[:, None], (1, M, 1))
        concat = torch.cat(
            (particle_states, beacon_repeated, measurement_repeated),
            axis=2,
        )
        return concat

    def measure(
        self,
        particle_positions: np.ndarray,
        beacon_positions: np.ndarray,
        noise: np.ndarray = None,
    ) -> np.ndarray:
        """
        Measures distance from each particle to each beacon

        :param particle_positions: Positions of all particles
        :param beacon_positions: Positions of all beacons
        :param noise: Special type of noise; if not given: use default zero-mean Gaussian noise with std=0.1
        :return: Matrix of distances -> rows represent particles, columns - Euclidean distance to each beacon
        """
        if not noise:
            noise = np.random.normal(
                loc=0.0, scale=0.1, size=len(beacon_positions) * len(particle_positions)
            )
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
