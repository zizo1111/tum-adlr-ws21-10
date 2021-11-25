import numpy
import numpy as np


class MotionModel:
    def __init__(self, state_dimension: int, env_size: int):
        self.state_dimension = state_dimension
        self.env_size = env_size

    def forward(self, states: np.ndarray, controls: np.ndarray = None) -> np.ndarray:
        predicted_states = states

        vel_x = states[:, 2] * np.cos(states[:, 3])
        vel_y = states[:, 2] * np.sin(states[:, 3])
        predicted_states[:, 0] = states[:, 0] + vel_x
        predicted_states[:, 1] = states[:, 1] + vel_y

        predicted_states[predicted_states[:, 0] > 2 * self.env_size] = predicted_states[predicted_states[:, 0] > 2 * self.env_size] - 2 * self.env_size
        predicted_states[predicted_states[:, 0] < 0] = 2 * self.env_size - predicted_states[predicted_states[:, 0] < 0]

        predicted_states[predicted_states[:, 1] > 2 * self.env_size] = predicted_states[predicted_states[:, 1] > 2 * self.env_size] - 2 * self.env_size
        predicted_states[predicted_states[:, 1] < 0] = 2 * self.env_size - predicted_states[predicted_states[:, 1] < 0]

        return predicted_states
