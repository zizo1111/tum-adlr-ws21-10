import numpy as np
from itertools import chain, zip_longest
import torch


def alt_chain(*iters, fillvalue=None):
    """returns a list of alternating values between the input lists/arrays"""
    return chain.from_iterable(zip_longest(*iters, fillvalue=fillvalue))


def split_given_size(a, size):
    """Split an array given the size and then appending the remainder"""
    return np.split(a, np.arange(size, len(a), size))


def normalize(state, measurement, setting):
    """Normalize weights and measurements according to envireoment to be between -1,1"""
    env_size = setting["env_size"][0]

    norm_measurement = measurement.clone().detach()
    norm_beacon_pos = setting["beacons_pos"].clone().detach()

    if state is not None:
        # normalize state
        norm_state = state.clone().detach()
        norm_state[:, :, :2] = (state[:, :, :2] - (0.5 * env_size)) / (0.5 * env_size)
        norm_state[:, :, 2:] = state[:, :, 2:] / 3.0
    else:
        norm_state = None

    # normalize measurement (measurement wont be between -1, 1, but -sqrt(2), sqrt(2))????
    norm_measurement = measurement / env_size

    # normalize beacon pos
    norm_beacon_pos = (setting["beacons_pos"] - (0.5 * env_size)) / (0.5 * env_size)

    return norm_state, norm_measurement, norm_beacon_pos


def unnormalize(state, measurement, setting):
    """Unnormalize weights and measurements according to envireoment to real values"""
    env_size = setting["env_size"][0]
    # unnormalize state
    unnorm_state = state.clone().detach()
    unnorm_measurement = measurement.clone().detach()
    unnorm_beacon_pos = setting["beacons_pos"].clone().detach()

    unnorm_state[:, :, :2] = (state[:, :, :2] * (0.5 * env_size)) + (0.5 * env_size)
    unnorm_state[:, :, 2:] = state[:, :, 2:] * 3.0

    # unnormalize measurement (measurement wont be between -1, 1, but -sqrt(2), sqrt(2))????
    unnorm_measurement = measurement * env_size

    # unnormalize beacon pos
    unnorm_beacon_pos = (setting["beacons_pos"] * (0.5 * env_size)) + (0.5 * env_size)

    return unnorm_state, unnorm_measurement, unnorm_beacon_pos


def unnormalize_estimate(estimate, particles, env_size):
    """Unnormalize weights and measurements according to envireoment to real values"""

    unnorm_estimate = estimate.clone()

    if particles is not None:
        unnorm_particles = particles.clone()

        # unnormalize particles
        unnorm_particles[:, :, :2] = (particles[:, :, :2] * (0.5 * env_size)) + (
            0.5 * env_size
        )
        unnorm_particles[:, :, 2:] = particles[:, :, 2:] * 3.0
    else:
        unnorm_particles = None

    # unnormalize estimate
    unnorm_estimate[:, :2] = (estimate[:, :2] * (0.5 * env_size)) + (0.5 * env_size)
    unnorm_estimate[:, 2:] = estimate[:, 2:] * 3.0

    return unnorm_estimate, unnorm_particles
