from simulation.simulation_env import SimulationEnv
from filters.particle_filter import ParticleFilter
from models.motion_model import MotionModel
from models.observation_model import ObservationModel
from simulation.dataset import Dataset, Sequence


def run_filter():
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"

    # PF config #
    num_particles = 5000

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
