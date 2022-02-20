import torch
import torch.distributions as D
import torch.nn as nn
import matplotlib.pyplot as plt


def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


class DiffParticleFilter(nn.Module):
    def __init__(
        self,
        hparams,
        motion_model,
        observation_model,
        resample=True,
        estimation_method="weighted_average",
    ):
        super().__init__()

        self.hparams = hparams
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.resample = resample

        self.motion_model.apply(initialize_weight)
        self.observation_model.apply(initialize_weight)

        self.particle_states = torch.zeros(size=[])
        self.weights = torch.zeros(size=[])

        self.estimation_method = estimation_method

    def init_beliefs(self):
        """
        Sample particle states from a GMM, and assign a new set of uniform weights.
        Note: This should be done at each start of every new sequence.
        """
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]
        env_size = self.hparams["env_size"]

        # Sample particles as GMM
        mix = D.Categorical(
            torch.ones(
                M,
            )
        )
        comp = D.Independent(
            D.Normal(
                torch.hstack(
                    (torch.rand(M, 2) * env_size, torch.randn(M, state_dim - 2) * 3)
                ),
                torch.rand(M, state_dim),
            ),
            1,
        )
        self.gmm = D.MixtureSameFamily(mix, comp)

        self.particle_states = self.gmm.sample((N, M))
        assert self.particle_states.shape == (N, M, state_dim)

        # Visualize for debugging purposes:
        """plt.scatter(x=self.particle_states[0, :, 0], y=self.particle_states[0, :, 1])
        plt.quiver(
            self.particle_states[0, :, 0],
            self.particle_states[0, :, 1],
            self.particle_states[0, :, 2] / torch.max(self.particle_states[0, :, 2]),
            self.particle_states[0, :, 3] / torch.max(self.particle_states[0, :, 3]),
            color="g",
            units="xy",
            scale=0.1,
        )
        plt.show()"""

        # Normalize weights
        self.weights = self.particle_states.new_full((N, M), 1.0 / M)
        assert self.weights.shape == (N, M)

    def forward(self, measurement, beacon_positions):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]
        soft_resample_alpha = self.hparams["soft_resample_alpha"]

        # Apply motion model to predict next particle states
        self.particle_states, precision_matrix = self.motion_model.forward(self.particle_states)
        assert self.particle_states.shape == (N, M, state_dim)

        # Apply observation model to get the likelihood for each particle
        input_obs = self.observation_model.prepare_input(
            self.particle_states,
            beacon_positions,
            measurement,
        )
        observation_lik = self.observation_model(input_obs)
        assert observation_lik.shape == (N, M)

        # Update particle weights
        self.weights *= observation_lik
        self.weights /= torch.sum(self.weights, dim=1, keepdim=True)

        # TODO: change to `logsumexp` if log weights used
        assert self.weights.shape == (N, M)
        assert torch.allclose(
            torch.sum(self.weights, dim=1, keepdim=True), torch.ones(N)
        )

        if self.resample:
            self._soft_resample(soft_resample_alpha)

        # compute estimate
        if self.estimation_method == "weighted_average":
            self.estimates = torch.sum(
                self.weights.clone().unsqueeze(-1) * self.particle_states, dim=1
            ) / M
        elif self.estimation_method == "max":
            max_weight = torch.argmax(self.weights, dim=1)
            b = torch.arange(self.particle_states.shape[0]).type_as(max_weight)
            self.estimates = self.particle_states[b, max_weight]

        assert self.estimates.shape == (N, state_dim)

        return self.estimates, self.weights, self.particle_states, precision_matrix

    def _soft_resample(self, alpha):
        """
        According to https://arxiv.org/pdf/1805.08975.pdf
        :param alpha: soft-resampling alpha
        """
        N, M, state_dim = self.particle_states.shape

        uniform_weights = self.weights.new_full((N, M), 1.0 / M, requires_grad=True)
        probs = torch.sum(
            torch.stack([alpha * self.weights, (1 - alpha) * uniform_weights], dim=0),
            dim=0,
        )  # q(k)
        self.weights = self.weights / probs  # w' -> stays un-normalized

        # TODO: i dont think this is true, but the only way to get normalized weights
        # self.weights /= torch.sum(self.weights, dim=1, keepdim=True)

        assert probs.shape == (N, M)

        # Sampling from q(k) is left
        distribution = D.Categorical(probs=probs)
        indices = distribution.sample((M,)).T
        assert indices.shape == (N, M)

        self.particle_states = torch.gather(
            self.particle_states,
            dim=1,
            index=indices[:, :, None].expand(
                (-1, -1, state_dim)
            ),  # only expand the size of 3rd dimension
        )
        self.weights = torch.gather(
            self.weights,
            dim=1,
            index=indices[:, :]
        )
        assert self.particle_states.shape == (N, M, state_dim)
