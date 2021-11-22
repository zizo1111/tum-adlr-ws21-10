#!/usr/bin/env python3

import numpy as np

class SimulationEnv():
    def __init__(self, size, num_discs, num_beacons):
        """
        Initializes the simulation enviroment
        Parameters
        ----------
        size : int
            simulation enviroment size
        num_discs : int
            number of discs in the enviroment
        num_beacons : int
            number of beacons in the enviroment
        """
        self.env_size_ = size
        self.num_discs_ = num_discs
        self.num_beacons_ = num_beacons

        # parameters of the process model
        self.spring_force_ = 0.05
        self.drag_force_ = 0.0075

        self.discs_ = []
        # init discs
        for disc_num in range(self.num_discs_):
            # draw a random position
            pos = np.random.uniform(-self.env_size_//2, self.env_size_//2, size=(2))
            # draw a random velocity
            vel = np.random.normal(loc=0., scale=1., size=(2)) * 3
            self.discs_.append(np.array([pos[0], pos[1], vel[0], vel[1]]))
        
        # init beacons
        self.beacons_ = []
        for beacon_num in range(num_beacons):
            axis = np.random.choice([0,1])
            if axis == 0:
                y_pos = (np.random.uniform(-self.env_size_//2, self.env_size_//2, size=(1))).item()
                x_pos = np.random.choice([-self.env_size_//2, self.env_size_//2])
                self.beacons_.append(np.array([x_pos, y_pos]))
            else:
                x_pos = (np.random.uniform(-self.env_size_//2, self.env_size_//2, size=(1))).item()
                y_pos = np.random.choice([-self.env_size_//2, self.env_size_//2])
                self.beacons_.append(np.array([x_pos, y_pos]))
    
    def _process_model(self, state):
        """
        Calculates the next state of the disc.
        Parameters
        ----------
        state : np.array
            The state (position and velocity) of the disc
        Returns
        -------
        new_state : np.array
            The next state (position and velocity) of the disc
        """
        new_state = np.copy(state)
        pull_force = - self.spring_force_ * state[:2]
        drag_force = - self.drag_force_ * state[2:]**2 * np.sign(state[2:])
        new_state[0] += state[2]
        new_state[1] += state[3]
        new_state[2] += pull_force[0] + drag_force[0]
        new_state[3] += pull_force[1] + drag_force[1]

        return new_state
    
    def update_step(self):
        """
        Calculates the next state of the discs.
        """
        for disc_num in range(self.num_discs_):
            self.discs_[disc_num] = self._process_model(self.discs_[disc_num])
    
    def get_reading(self, beacon_num):
        """
        Calculates the current state of the discs wrt to beacon.
        Parameters
        ----------
        beacon_num : int
            beacon index
        Returns
        -------
        reading : np.array
            The current state (position and velocity) of the discs wrt to
            given beacon
        """
        reading = []
        for disc_num in range(self.num_discs_):
            reading.append(self.discs_[disc_num][:2] - self.beacons_[beacon_num])
        return np.asarray(reading)
    
    def get_beacons_pos(self):
        """
        returns the beacons postions in the simulation enviroment
        """
        return self.beacons_
    
# if __name__ == "__main__":
#     test_env = SimulationEnv(200, 1, 2)
#     # print(test_env.discs)
#     print(test_env.discs_[0][0])
#     print(test_env.get_reading(0))
#     test_env.update_step()
#     # print(test_env.discs)
#     # print(test_env.discs[0][0])
#     print(test_env.get_reading(0))