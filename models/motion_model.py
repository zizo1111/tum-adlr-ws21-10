import numpy as np


class MotionModel:
    def __init__(self, state_dimension: int, env_size: int):
        self.state_dimension = state_dimension
        self.env_size = env_size

    def forward(self, particle_states: np.ndarray, controls: np.ndarray = None, noise: np.ndarray = None, dt: float = 1.0) -> np.ndarray:
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=self.state_dimension)

        predicted_particle_states = particle_states + noise

        vel_x = particle_states[:, 2] * np.cos(particle_states[:, 3])
        vel_y = particle_states[:, 2] * np.sin(particle_states[:, 3])
        predicted_particle_states[:, 0] += vel_x * dt
        predicted_particle_states[:, 1] += vel_y * dt


        out_left = predicted_particle_states[:, 0] < 0
        out_right = predicted_particle_states[:, 0] > self.env_size
        out_bottom = predicted_particle_states[:, 1] < 0
        out_top = predicted_particle_states[:, 1] > self.env_size

        if True:
            predicted_particle_states[out_left, 0] = 0
            predicted_particle_states[out_right, 0] = self.env_size
            predicted_particle_states[out_bottom, 1] = 0
            predicted_particle_states[out_top, 1] = self.env_size
            predicted_particle_states[out_left | out_right, 3] = np.arctan(vel_y[out_left | out_right] / (-vel_x[out_left | out_right]))
            predicted_particle_states[out_top | out_bottom, 3] = np.arctan(-vel_y[out_top | out_bottom] / vel_x[out_top | out_bottom])

        return predicted_particle_states
