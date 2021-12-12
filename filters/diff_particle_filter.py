import torch
import torch.distributions as D
import torch.nn as nn


class DiffParticleFilter(nn.Module):
    def __init__(self, hparams, motion_model, observation_model):
        super().__init__()

        self.hparams = hparams
        self.motion_model = motion_model
        self.observation_model = observation_model

        self.particle_states = torch.Tensor
        self.weights = torch.Tensor

        self._init_beliefs()

    def _init_beliefs(self):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]

        # Sample particles as GMM
        mix = D.Categorical(torch.ones(M,))
        comp = D.Independent(
            D.Normal(torch.randn(M, state_dim), torch.rand(M, state_dim)), 1
        )
        gmm = D.MixtureSameFamily(mix, comp)

        self.particle_states = gmm.sample((N, M))
        assert self.particle_states.shape == (N, M, state_dim)

        # Normalize weights
        self.weights = self.particle_states.new_full((N, M), 1.0 / M)
        assert self.weights.shape == (N, M)

    def forward(self):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]

        self.particle_states = self.motion_model.forward(self.particle_states)
        assert self.particle_states.shape == (N, M, state_dim)
