import torch
import torch.distributions as D
import torch.nn as nn

from models.observation_model import ObservationModel


class DiffParticleFilter(nn.Module):
    def __init__(self, hparams, motion_model, observation_model):
        super().__init__()

        self.hparams = hparams
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.observation_model.apply(self.initialize_weight)

        self.particle_states = torch.Tensor
        self.weights = torch.Tensor

        self._init_beliefs()

    def _init_beliefs(self):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]

        # Sample particles as GMM
        mix = D.Categorical(
            torch.ones(
                M,
            )
        )
        comp = D.Independent(
            D.Normal(torch.randn(M, state_dim), torch.rand(M, state_dim)), 1
        )
        gmm = D.MixtureSameFamily(mix, comp)

        self.particle_states = gmm.sample((N, M))
        assert self.particle_states.shape == (N, M, state_dim)

        # Normalize weights
        self.weights = self.particle_states.new_full((N, M), 1.0 / M)
        assert self.weights.shape == (N, M)

    def initialize_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, measurement):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]

        # Apply motion model to predict next particle states
        self.particle_states = self.motion_model.forward(self.particle_states)
        assert self.particle_states.shape == (N, M, state_dim)

        # Apply observation model to get the likelihood for each particle
        input_obs = self.observation_model.prepare_input(
            self.particle_states[:, :, 0:2],
            self.hparams["beacon_positions"],
            measurement,
        )
        observation_lik = self.observation_model(input_obs)
        assert observation_lik.shape == (N, M)

        # Update particle weights
        self.weights *= observation_lik
        self.weights /= torch.sum(
            self.weights, dim=1, keepdim=True
        )  # TODO: change to `logsumexp` if log weights used
        assert self.weights.shape == (N, M)
        assert torch.allclose(
            torch.sum(self.weights, dim=1, keepdim=True), torch.ones(N)
        )
