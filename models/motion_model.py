import numpy as np


class MotionModel:
    def __init__(self, state_dimension: int, env_size: int):
        self.state_dimension = state_dimension
        self.env_size = env_size

    def forward(self, particle_states: np.ndarray, controls: np.ndarray = None, noise: np.ndarray = None) -> np.ndarray:
        if not noise:
            noise = np.random.normal(loc=0.0, scale=0.1, size=self.state_dimension)

        predicted_particle_states = particle_states + noise

        #print(particle_states)
        #print(noise)
        #print(predicted_particle_states)
        #print("--------------------------------")

        vel_x = particle_states[:, 2] * np.cos(particle_states[:, 3])
        vel_y = particle_states[:, 2] * np.sin(particle_states[:, 3])
        predicted_particle_states[:, 0] = particle_states[:, 0] + vel_x
        predicted_particle_states[:, 1] = particle_states[:, 1] + vel_y

        #predicted_particle_states[predicted_particle_states[:, 0] > 2 * self.env_size] = predicted_particle_states[predicted_particle_states[:, 0] > 2 * self.env_size] - 2 * self.env_size
        #predicted_particle_states[predicted_particle_states[:, 0] < 0] = 2 * self.env_size - predicted_particle_states[predicted_particle_states[:, 0] < 0]

        #predicted_particle_states[predicted_particle_states[:, 1] > 2 * self.env_size] = predicted_particle_states[predicted_particle_states[:, 1] > 2 * self.env_size] - 2 * self.env_size
        #predicted_particle_states[predicted_particle_states[:, 1] < 0] = 2 * self.env_size - predicted_particle_states[predicted_particle_states[:, 1] < 0]

        return predicted_particle_states
