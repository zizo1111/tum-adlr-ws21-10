from simulation.simulation_env import SimulationEnv
from filters.particle_filter import ParticleFilter
from filters.diff_particle_filter import DiffParticleFilter
from models.motion_model import MotionModel
from models.observation_model import ObservationModel
from simulation.dataset import Dataset, Sequence


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

    # PF config #
    num_particles = 100

    # beacons config -> TODO #
    num_beacons = 2
    beacon_positions = [[0, env_size], [env_size, env_size]]

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

    # Hyperparameters for differential filter #
    hparams = {
        "num_particles": num_particles,
        "state_dimension": state_dim,
        "env_size": env_size,
        "soft_resample_alpha": 0.7,
        "batch_size": 8,
        "beacon_positions": beacon_positions,
    }

    diff_particle_filter = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )

    measurement = [[0.1, 50]]  # TODO: where are we supposed to get measurement from?
    diff_particle_filter.forward(measurement)


def create_dataset():
    set = Dataset(create=True)
    set.save_dataset()


def load_dataset(path=None):
    set = Dataset()
    set.load_dataset(path)
    seq = set.get_sequence(2)
    seq.play()


if __name__ == "__main__":
    run_filter()
    run_diff_filter()
