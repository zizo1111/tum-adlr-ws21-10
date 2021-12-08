import numpy as np


class MotionModel:
    def __init__(self, state_dimension: int, env_size: int, mode: str):
        self.state_dimension = state_dimension
        self.env_size = env_size
        self.mode = mode

    def forward(self, particle_states: np.ndarray, controls: np.ndarray = None, noise: np.ndarray = None, dt: float = 1.0) -> np.ndarray:
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=particle_states.shape)

        predicted_particle_states = particle_states + noise

        predicted_particle_states[:, 0] += dt * particle_states[:, 2]
        predicted_particle_states[:, 1] += dt * particle_states[:, 3]

        out_left = predicted_particle_states[:, 0] < 0
        out_right = predicted_particle_states[:, 0] > self.env_size
        out_bottom = predicted_particle_states[:, 1] < 0
        out_top = predicted_particle_states[:, 1] > self.env_size

        if self.mode == "collide":
            predicted_particle_states[out_left, 0] = 0
            predicted_particle_states[out_right, 0] = self.env_size
            predicted_particle_states[out_bottom, 1] = 0
            predicted_particle_states[out_top, 1] = self.env_size
            predicted_particle_states[out_left | out_right, 2] *= -1
            predicted_particle_states[out_top | out_bottom, 3] *= -1

        elif self.mode == "wrap":
            predicted_particle_states[out_top, 0] = predicted_particle_states[:, 0] - self.env_size
            predicted_particle_states[out_bottom, 0] = self.env_size - predicted_particle_states[:, 0]
            predicted_particle_states[out_left, 1] = self.env_size - predicted_particle_states[:, 1]
            predicted_particle_states[out_right, 1] = predicted_particle_states[:, 1] - self.env_size

        return predicted_particle_states
