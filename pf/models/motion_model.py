import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


class BN(nn.Module):
    """
    BatchNorm1d is applied on the second input dimension for (N,C,L)
    -> we need it on the L here (?) -> some permutation of the input has to be done
    """

    def __init__(self, state_dimension: int):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm1d(state_dimension))

    def forward(self, input: torch.Tensor):
        prem = torch.permute(input, (0, 2, 1))
        output = self.model(prem)
        return torch.permute(output, (0, 2, 1))


class PositiveTanh(nn.Module):
    """
    The characteristic of Tanh() is that its output is in interval [-1, 1] / [0, 2] for PositiveTanh(),
    which might be an issue for our setup.
    However, it can be helpful in temporal (i.e., recurrent and non-iid) settings.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Tanh()

    def forward(self, input: torch.Tensor):
        x = self.model(input)
        ones = torch.ones(size=x.shape)
        return x + ones


class MotionModel(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        env_size: int,
        mode: str,
        non_linearity: str = "tanh",
    ):
        super().__init__()

        self.state_dimension = state_dimension
        self.env_size = env_size
        self.mode = mode
        if non_linearity.lower() == "tanh":
            self.non_linearity = nn.Tanh()
        else:
            self.non_linearity = nn.ReLU()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Apply motion model per batch element + per each particle
        # -> shared weights for each particle
        self.model = nn.Sequential(
            nn.Linear(state_dimension, 2 * state_dimension),
            self.non_linearity,
            nn.Linear(2 * state_dimension, state_dimension * state_dimension),
            self.non_linearity,
            # mean + lower triangular L for covariance matrix
            # S = L * L.T, with L having positive-valued diagonal entries
            nn.Linear(
                state_dimension * state_dimension,
                state_dimension * (state_dimension + 3) // 2,
            ),
            # nn.Linear(
            #   state_dimension * (state_dimension + 3) // 2,
            #   state_dimension * (state_dimension + 1),
            # ),
            # PositiveTanh(),  # TODO: think of possible use
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
            predicted_particle_states[out_top, 0] = (
                predicted_particle_states[:, 0] - self.env_size
            )
            predicted_particle_states[out_bottom, 0] = (
                self.env_size - predicted_particle_states[:, 0]
            )
            predicted_particle_states[out_left, 1] = (
                self.env_size - predicted_particle_states[:, 1]
            )
            predicted_particle_states[out_right, 1] = (
                predicted_particle_states[:, 1] - self.env_size
            )

        return predicted_particle_states

    def forward(self, particle_states: torch.Tensor):
        N = particle_states.shape[0]  # batch size
        M = particle_states.shape[1]  # number of particles
        particle_states = particle_states.to(self.device)

        x = self.model(particle_states)

        if x.shape == (N, M, self.state_dimension * (self.state_dimension + 3) // 2):
            # Split to get the mean and covariance matrix separately
            (
                predicted_mean,
                log_predicted_scale_diag,
                predicted_scale_lower,
            ) = torch.split(
                x,
                [
                    self.state_dimension,
                    self.state_dimension,
                    self.state_dimension * (self.state_dimension - 1) // 2,
                ],
                dim=2,
            )
            log_predicted_scale_diag = torch.clamp(
                log_predicted_scale_diag, min=-2, max=1
            )
            assert predicted_mean.shape == (N, M, self.state_dimension)
            assert log_predicted_scale_diag.shape == (N, M, self.state_dimension)
            assert predicted_scale_lower.shape == (
                N,
                M,
                self.state_dimension * (self.state_dimension - 1) // 2,
            )

            # Assume our network outputs log(value) -> this way we always get positive values on the diagonal
            predicted_scale_tril = torch.diag_embed(
                torch.exp(log_predicted_scale_diag)
            ).to(self.device)

            # Reform to get the lower part (under main diagonal) of the scale
            lower_tril_indices = torch.tril_indices(
                row=self.state_dimension,
                col=self.state_dimension,
                offset=-1,
                device=self.device,
            )
            predicted_scale_tril[
                :, :, lower_tril_indices[0], lower_tril_indices[1]
            ] = predicted_scale_lower
        elif x.shape == (N, M, self.state_dimension * (self.state_dimension + 1)):
            # Split to get the mean and covariance matrix separately
            predicted_mean, predicted_scale = torch.split(
                x,
                [self.state_dimension, self.state_dimension * self.state_dimension],
                dim=2,
            )
            assert predicted_mean.shape == (N, M, self.state_dimension)
            assert predicted_scale.shape == (
                N,
                M,
                self.state_dimension * self.state_dimension,
            )

            predicted_scale_tril = (
                predicted_scale.view(N, M, self.state_dimension, self.state_dimension)
                .tril()
                .to(self.device)
            )

        # Apply threshold to get only positive values on the diagonal -> to satisfy the constraint LowerCholesky()
        ## F.threshold(torch.diagonal(predicted_scale_tril, offset=0, dim1=2, dim2=3), threshold=0, value=1.e-5, inplace=True)
        # ^was resulting in RuntimeError

        # Or just make values on the diagonal absolute
        ## predicted_tril_cholesky = predicted_scale_tril
        ## predicted_tril_cholesky.diagonal(dim1=2, dim2=3).abs_()
        # ^apparently, this also did not work out well
        assert predicted_scale_tril.shape == (
            N,
            M,
            self.state_dimension,
            self.state_dimension,
        )

        # Sample (with gradient) from the resulting distribution -> "reparameterization trick"
        reparam = D.MultivariateNormal(
            loc=predicted_mean,
            scale_tril=predicted_scale_tril,
        )
        predicted_particle_states = reparam.rsample()

        # predicted_particle_states[:, :, 0:2] = predicted_particle_states[:, :, 0:2] * 100  # TODO: also a possibility

        # torch.clamp_(predicted_particle_states[:, :, :2], min=0, max=self.env_size)
        # torch.clamp_(predicted_particle_states[:, :, 2:], min=-10, max=10)

        return predicted_particle_states, reparam.precision_matrix


def cholesky_check(value):
    """
    Original LowerCholesky() constraint check from PyTorch
    """
    value_tril = value.tril()
    lower_triangular = (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
    positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
    return lower_triangular & positive_diagonal
