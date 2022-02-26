import torch
import torch.distributions as D
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


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
        log_prob=False,
    ):
        super().__init__()

        self.hparams = hparams
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.resample = resample

        # self.motion_model.apply(initialize_weight)
        self.observation_model.apply(initialize_weight)

        self.particle_states = torch.zeros(size=[])
        self.weights = torch.zeros(size=[])

        self.particle_log_weights = torch.zeros(size=[])
        self.estimation_method = estimation_method

        self.log_prob = log_prob

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
                    ((torch.rand(M, 2) * 2) - 1, torch.randn(M, state_dim - 2))
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

        if not self.log_prob:
            # Normalize weights
            self.weights = self.particle_states.new_full((N, M), 1.0 / M)
            assert self.weights.shape == (N, M)
        else:
            # Normalize weights
            self.particle_log_weights = self.particle_states.new_full(
                (N, M), float(-np.log(M, dtype=np.float32))
            )
            assert self.particle_log_weights.shape == (N, M)

    def forward(self, measurement, beacon_positions):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]
        soft_resample_alpha = self.hparams["soft_resample_alpha"]

        # Apply motion model to predict next particle states
        particle_states_diff, precision_matrix = self.motion_model.forward(
            self.particle_states.clone()
        )
        self.particle_states = self.particle_states + particle_states_diff
        # print(measurement)
        torch.clamp_(self.particle_states[:, :, :2], min=-1, max=1)
        torch.clamp_(self.particle_states[:, :, 2:], min=-2, max=2)
        assert self.particle_states.shape == (N, M, state_dim)

        if self.resample:
            if not self.log_prob:
                self._soft_resample(soft_resample_alpha)
            else:
                self._resample(soft_resample_alpha)

        # Apply observation model to get the likelihood for each particle
        input_obs = self.observation_model.prepare_input(
            self.particle_states,
            beacon_positions,
            measurement,
        )
        observation_lik = self.observation_model(input_obs)
        assert observation_lik.shape == (N, M)

        if not self.log_prob:
            # Update particle weights
            self.weights *= observation_lik
            self.weights /= torch.sum(self.weights, dim=1, keepdim=True)

        else:
            self.particle_log_weights = self.particle_log_weights + observation_lik
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )
            self.weights = torch.exp(self.particle_log_weights)

        assert self.weights.shape == (N, M)
        assert torch.allclose(
            torch.sum(self.weights, dim=1, keepdim=True), torch.ones(N)
        )

        # compute estimate
        if self.estimation_method == "weighted_average":
            self.estimates = torch.sum(
                self.weights.clone().unsqueeze(-1) * self.particle_states, dim=1
            )
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
        self.weights = torch.gather(self.weights, dim=1, index=indices[:, :])
        assert self.particle_states.shape == (N, M, state_dim)

    def _resample(self, soft_resample_alpha) -> None:
        """Resample particles."""
        # Note the distinction between `M`, the current number of particles, and
        # `self.num_particles`, the desired number of particles
        N, M, state_dim = self.particle_states.shape

        sample_logits: torch.Tensor
        uniform_log_weights = self.particle_log_weights.new_full(
            (N, self.hparams["num_particles"]), float(-np.log(M, dtype=np.float32))
        )
        if soft_resample_alpha < 1.0:
            # Soft resampling
            assert self.particle_log_weights.shape == (N, M)
            sample_logits = torch.logsumexp(
                torch.stack(
                    [
                        self.particle_log_weights + np.log(soft_resample_alpha),
                        uniform_log_weights + np.log(1.0 - soft_resample_alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            self.particle_log_weights = self.particle_log_weights - sample_logits
        else:
            # Standard particle filter re-sampling -- this stops gradients
            # This is the most naive flavor of resampling, and not the low
            # variance approach
            #
            # Note the distinction between M, the current # of particles,
            # and self.num_particles, the desired # of particles
            sample_logits = self.particle_log_weights
            self.particle_log_weights = uniform_log_weights

        assert sample_logits.shape == (N, M)
        distribution = torch.distributions.Categorical(logits=sample_logits)
        state_indices = distribution.sample((self.hparams["num_particles"],)).T
        assert state_indices.shape == (N, self.hparams["num_particles"])

        self.particle_states = torch.gather(
            self.particle_states,
            dim=1,
            index=state_indices[:, :, None].expand(
                (N, self.hparams["num_particles"], state_dim)
            ),
        )
