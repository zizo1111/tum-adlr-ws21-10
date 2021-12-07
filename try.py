from simulation.simulation_env import SimulationEnv
from filters.particle_filter import ParticleFilter
from models.motion_model import MotionModel
from models.observation_model import ObservationModel

if __name__ == "__main__":
    dynamics_model = MotionModel(4, 200)
    observation_model = ObservationModel(2, 200)
    particle_filter = ParticleFilter(10000, 4, 200, dynamics_model, observation_model)
    test_env = SimulationEnv(200, 1, 2, mode="collide", auto=True, animate=True, p_filter=particle_filter)
