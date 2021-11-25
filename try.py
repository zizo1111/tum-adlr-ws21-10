from simulation_env import SimulationEnv
from filters.particle_filter import ParticleFilter
from models.motion_model import MotionModel

if __name__ == "__main__":
    dynamics_model = MotionModel(4, 200)
    particle_filter = ParticleFilter(5000, 4, 200, dynamics_model)
    test_env = SimulationEnv(200, 1, 3, wrap=True, auto=True, animate=True, p_filter=particle_filter)

    #print(particle_filter.particles)
    particle_filter.predict()
    #print(particle_filter.particles)

    # print(test_env.discs)
    print(test_env.discs_[0])
    print(test_env.beacons_[0])
    print(test_env.beacons_[1])
    print(f"Reading ---> {test_env.get_distance(0)}")
    test_env.update_step(5)
    # print(test_env.discs)
    print(test_env.discs_[0])
    print(test_env.beacons_[0])
    print(f"Reading ---> {test_env.get_distance(0)}")
