import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


class MotionModel(nn.Module):
    def __init__(self, state_dimension: int, env_size: int, mode: str):
        super().__init__()

        self.state_dimension = state_dimension
        self.env_size = env_size
        self.mode = mode

        # Apply motion model per batch element + per each particle
        # -> shared weights for each particle
        self.model = nn.Sequential(
            nn.Linear(state_dimension, 2 * state_dimension),
            nn.BatchNorm1d(100),  # TODO: BatchNorm is applied on the second input dimension for (N,C,L);
                                  # we need it on the L here (?) -> some permutation of the input has to be done
            nn.ReLU(),
            nn.Linear(2 * state_dimension, state_dimension * state_dimension),
            nn.BatchNorm1d(100),  # TODO: same here
            nn.ReLU(),
            # mean + lower triangular L for covariance matrix
            # S = L * L.T, with L having positive-valued diagonal entries
            nn.Linear(state_dimension * state_dimension, state_dimension * (state_dimension + 3) // 2),
            nn.BatchNorm1d(100),  # TODO: same for BatchNorm here
        )

    def standard_forward(
        self,
        particle_states: np.ndarray,
        controls: np.ndarray = None,
        noise: np.ndarray = None,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Propagates particle/disc states according to the motion model chosen for one time step dt

        :param particle_states: Particle states of dimension = self.state_dimension
        :param controls: Control values (if present)
        :param noise: Special type of noise; if not given: use default zero-mean Gaussian noise with std=0.1
        :param dt: Time step
        :return: New predicted particle states (propagated dt time steps)
        """
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=particle_states.shape)
        else:
            assert noise.shape == particle_states.shape

        predicted_particle_states = particle_states + noise

        predicted_particle_states[:, 0] += dt * particle_states[:, 2]
        predicted_particle_states[:, 1] += dt * particle_states[:, 3]

        out_left = predicted_particle_states[:, 0] < 0
        out_right = predicted_particle_states[:, 0] > self.env_size
        out_bottom = predicted_particle_states[:, 1] < 0
        out_top = predicted_particle_states[:, 1] > self.env_size

        if self.mode == "collide":
            predicted_particle_states[out_left, 0] = 0
            predicted_particle_states[out_right, 0] = self.env_size
            predicted_particle_states[out_bottom, 1] = 0
            predicted_particle_states[out_top, 1] = self.env_size
            predicted_particle_states[out_left | out_right, 2] *= -1
            predicted_particle_states[out_top | out_bottom, 3] *= -1

        elif self.mode == "wrap":
            predicted_particle_states[out_top, 0] = predicted_particle_states[:, 0] - self.env_size
            predicted_particle_states[out_bottom, 0] = self.env_size - predicted_particle_states[:, 0]
            predicted_particle_states[out_left, 1] = self.env_size - predicted_particle_states[:, 1]
            predicted_particle_states[out_right, 1] = predicted_particle_states[:, 1] - self.env_size

        return predicted_particle_states

    def forward(self, particle_states: torch.Tensor):
        N = particle_states.shape[0]  # batch size
        M = particle_states.shape[1]  # number of particles
        # particle_states = particle_states.to(self.device)  # TODO: uncomment for training

        x = self.model(particle_states)

        # Split to get the mean and covariance matrix separately
        predicted_mean, predicted_lower_diag = torch.split(
            x,
            [self.state_dimension, self.state_dimension * (self.state_dimension + 1) // 2],
            dim=2,
        )
        assert predicted_mean.shape == (N, M, self.state_dimension)
        assert predicted_lower_diag.shape == (N, M, self.state_dimension * (self.state_dimension + 1) // 2)

        # Reform to get the tensor of covariance matrices
        predicted_tril = torch.zeros(N, M, self.state_dimension, self.state_dimension)
        tril_indices = torch.tril_indices(row=self.state_dimension, col=self.state_dimension, offset=0)
        predicted_tril[:, :, tril_indices[0], tril_indices[1]] = predicted_lower_diag

        # Apply threshold to get only positive values on the diagonal -> to satisfy the constraint LowerCholesky()
        F.threshold(torch.diagonal(predicted_tril, offset=0, dim1=2, dim2=3), threshold=0, value=1.e-5, inplace=True)
        assert predicted_tril.shape == (N, M, self.state_dimension, self.state_dimension)

        # Sample (with gradient) from the resulting distribution -> "reparameterization trick"
        predicted_particle_states = D.MultivariateNormal(
            loc=predicted_mean,
            scale_tril=predicted_tril,
        ).rsample()

        return predicted_particle_states
