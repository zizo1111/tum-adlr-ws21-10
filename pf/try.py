import numpy as np
import torch.nn.functional as F
import torch
from pf.simulation.simulation_env import SimulationEnv
from pf.filters.particle_filter import ParticleFilter
from pf.filters.diff_particle_filter import DiffParticleFilter
from pf.models.motion_model import MotionModel
from pf.models.observation_model import ObservationModel
from pf.utils.dataset import DatasetSeq


def run_filter():
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"

    # PF config #
    num_particles = 1000

    # Environment config #
    num_discs = 1
    num_beacons = 2

    dynamics_model = MotionModel(
        state_dimension=state_dim, env_size=env_size, mode=mode
    )
    observation_model = ObservationModel(
        state_dimension=2, env_size=env_size
    )  # we can only measure distances to beacons -> only position is needed
    particle_filter = ParticleFilter(
        num_particles=num_particles,
        state_dimension=state_dim,
        env_size=env_size,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )
    test_env = SimulationEnv(
        size=env_size,
        num_discs=num_discs,
        num_beacons=num_beacons,
        mode="collide",
        auto=True,
        animate=True,
        p_filter=particle_filter,
    )

    print(test_env.get_error())


def run_diff_filter(test_path=None):
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"
    num_discs = 1
    # PF config #
    num_particles = 500

    # beacons config #
    num_beacons = 1

    dynamics_model = MotionModel(
        state_dimension=state_dim,
        env_size=env_size,
        mode=mode,
    )
    observation_model = ObservationModel(
        state_dimension=state_dim,
        env_size=env_size,
        num_particles=num_particles,
        num_beacons=num_beacons,
    )

    # Hyperparameters for differential filter #
    hparams = {
        "num_particles": num_particles,
        "state_dimension": state_dim,
        "env_size": env_size,
        "soft_resample_alpha": 0.7,
        "batch_size": 1,
    }

    pf_model = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )
    path = "saved_models/saved_model_final_presentation_mse_loss.pth"
    pf_model.load_state_dict(torch.load(path))
    pf_model.init_beliefs()
    pf_model.eval()

    if not test_path:
        test_env = SimulationEnv(
            size=env_size,
            num_discs=num_discs,
            num_beacons=num_beacons,
            mode=mode,
            auto=True,
            animate=True,
            dp_filter=pf_model,
        )

        print(test_env.get_error())
    else:
        set = DatasetSeq()
        set.load_dataset(test_path)


def create_dataset(path=None):
    set = DatasetSeq(create=True)
    set.save_dataset(path)


def load_dataset(path=None):
    set = DatasetSeq()
    set.load_dataset(path)
    seq = set.get_sequence(2)
    seq.play()


if __name__ == "__main__":
    # run_filter()
    run_diff_filter()
    # create_dataset()
