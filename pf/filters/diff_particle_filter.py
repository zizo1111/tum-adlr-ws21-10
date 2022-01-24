import torch
import torch.distributions as D
import torch.nn as nn

from pf.models.observation_model import ObservationModel


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
        self.observation_model.apply(self.initialize_weight)

        self.particle_states: torch.Tensor
        self.weights: torch.Tensor

        self.estimation_method = estimation_method
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

    def initialize_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, measurement, beacon_positions):
        N = self.hparams["batch_size"]
        M = self.hparams["num_particles"]
        state_dim = self.hparams["state_dimension"]
        soft_resample_alpha = self.hparams["soft_resample_alpha"]

        # Apply motion model to predict next particle states
        # TODO FIX:RuntimeError: one of the variables needed for gradient computation has been
        # modified by an inplace operation: [torch.FloatTensor [100]] is at version 2; expected
        #  version 1 instead. Hint: the backtrace further above shows the operation that failed
        #  to compute its gradient. The variable in question was changed in there or anywhere later.
        #  Good luck! -> coming from the motion_model forward

        if self.resample:
            self.particle_states = self.motion_model.forward(self.particle_states)
        else:
            self.particle_states = self.motion_model.forward(
                self.particle_states.clone().detach()
            )
        assert self.particle_states.shape == (N, M, state_dim)

        # Apply observation model to get the likelihood for each particle
        input_obs = self.observation_model.prepare_input(
            self.particle_states, beacon_positions, measurement,
        )
        observation_lik = self.observation_model(input_obs)
        assert observation_lik.shape == (N, M)

        # Update particle weights
        # self.weights = torch.mul(self.weights.clone(), observation_lik)
        # self.weights = torch.div(
        #     self.weights.clone().detach(),
        #     torch.sum(self.weights.clone().detach(), dim=1, keepdim=True),
        # )
        ## ^ seems like not needed anymore
        ## | as we have already fixed the following problem:

        ## changed this because it was causing
        # RuntimeError: one of the variables needed for gradient computation has been modified
        # by an inplace operation: [torch.FloatTensor [16, 1]], which is output 0 of AsStridedBackward0,
        # is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation
        # that failed to compute its gradient. The variable in question was changed in there
        # or anywhere later. Good luck!

        self.weights *= observation_lik
        self.weights /= torch.sum(
            self.weights, dim=1, keepdim=True
        )  # TODO: change to `logsumexp` if log weights used
        assert self.weights.shape == (N, M)
        assert torch.allclose(
            torch.sum(self.weights, dim=1, keepdim=True), torch.ones(N)
        )

        if self.resample:
            self._soft_resample(soft_resample_alpha)

        # compute output
        if self.estimation_method == "weighted_average":
            self.estimates = torch.sum(
                self.weights[:, :, None] * self.particle_states, dim=1
            )
        # TODO add other estimation methods e.g. max etc.

        assert self.estimates.shape == (N, state_dim)

        return self.estimates, self.weights, self.particle_states

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
        self.weights = self.weights / probs  # w'
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
        assert self.particle_states.shape == (N, M, state_dim)
