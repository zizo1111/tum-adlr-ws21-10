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


def run_diff_filter():
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"
    num_discs = 1
    # PF config #
    num_particles = 100

    # beacons config -> TODO #
    num_beacons = 2

    test_env = SimulationEnv(
        size=env_size,
        num_discs=num_discs,
        num_beacons=num_beacons,
        mode=mode,
        auto=False,
    )

    dynamics_model = MotionModel(
        state_dimension=state_dim,
        env_size=env_size,
        mode=mode,
    )
    observation_model = ObservationModel(
        state_dimension=2,
        env_size=env_size,
        num_particles=num_particles,
        num_beacons=num_beacons,
    )

    beacon_positions = np.tile(test_env.get_beacons_pos().astype(np.float32), (8, 1, 1))

    # Hyperparameters for differential filter #
    hparams = {
        "num_particles": num_particles,
        "state_dimension": state_dim,
        "env_size": env_size,
        "soft_resample_alpha": 0.7,
        "batch_size": 8,
    }

    diff_particle_filter = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )
    measurement, state = test_env.run_batch(hparams["batch_size"])
    # TODO: where are we supposed to get measurement from?
    # Num of measurements = Batch size
    # in test does it make sense to have batch number > 1
    es = diff_particle_filter(measurement, beacon_positions)

    loss = F.mse_loss(es, torch.from_numpy(state.reshape(8, -1)))
    loss.backward()


def create_dataset():
    set = DatasetSeq(create=True)
    set.save_dataset()


def load_dataset(path=None):
    set = DatasetSeq()
    set.load_dataset(path)
    seq = set.get_sequence(2)
    seq.play()


if __name__ == "__main__":
    # run_filter()
    run_diff_filter()
    # create_dataset()
    # import numpy as np

    # pos = np.array([[[0, 1], [2, 3]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    # beacon_t = pos  # np.tile(pos, (8, 1, 1))
    # test = beacon_t.reshape(3, -1)
    # t1 = np.tile(test[:, np.newaxis], (1, 5, 1))
    # print(beacon_t.shape, test.shape, t1.shape, t1)
