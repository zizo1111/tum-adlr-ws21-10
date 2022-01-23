import torch
import math


def RMSE(estimate, gt):
    return torch.sqrt(torch.mean((estimate - gt) ** 2))


# TODO nll
# https://github.com/akloss/differentiable_filters/blob/821889dec411927658c6ef7dd01c9028d2f28efd/differentiable_filters/contexts/base_context.py#L341
def NLL(estimate, gt, weights, particle_states, reduce_mean=True):
    N = particle_states.shape[0]  # batch size
    M = particle_states.shape[1]  # number of particles
    state_dim = particle_states.shape[-1]

    # difference between output and the gt states of the particles
    diff = particle_states - torch.tile(gt, (1, M, 1))
    assert diff.shape == (N, M, state_dim)

    # remove nans and infs and replace them with high values/zeros
    diff = torch.where(torch.isfinite(diff), diff, torch.ones_like(diff) * 1e5)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    weights /= torch.sum(weights, dim=-1, keepdims=True)
    assert weights.shape == (N, M)

    # #Not sure about this please re-check
    mixture_std = 1

    covar = torch.ones(state_dim)
    for k in range(state_dim):
        covar[k] *= mixture_std

    covar = torch.diag(torch.square(covar))

    if diff.ndim > 3:
        sl = diff.shape[1]
        diff = torch.reshape(diff, [N, -1, M, state_dim, 1])
        covar = torch.tile(covar[None, None, None, :, :], [N, sl, M, 1, 1])
    else:
        sl = 1
        diff = torch.reshape(diff, [N, M, state_dim, 1])
        covar = torch.tile(covar[None, None, :, :], [N, M, 1, 1])

    exponent = torch.matmul(
        torch.matmul(torch.transpose(diff, -2, -1), torch.inverse(covar)), diff
    )
    exponent = torch.reshape(exponent, [N, sl, M])

    normalizer = torch.log(torch.det(covar)) + (
        state_dim * torch.log(2 * torch.tensor(math.pi))
    )
    normalizer = torch.reshape(normalizer, [N, sl, M])

    log_like = -0.5 * (exponent + normalizer)
    log_like = torch.reshape(log_like, [N, sl, M])

    log_like = torch.where(
        torch.greater_equal(log_like, -500), log_like, torch.ones_like(log_like) * -500
    )

    exp = torch.exp(log_like)
    # the per particle likelihoods are weighted and summed in the particle
    # dimension
    weighted = weights * exp.reshape(weights.shape)

    weighted = torch.sum(weighted, dim=-1)

    # compute the negative logarithm
    likelihood = -(torch.log(torch.maximum(weighted, torch.tensor(1e-30))))

    if reduce_mean:
        likelihood = torch.mean(likelihood)

    return likelihood
