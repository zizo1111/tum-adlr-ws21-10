import numpy as np
import matplotlib.pyplot as plt
from simulation.animation import Animator
import math
import threading


class SimulationEnv:
    def __init__(
        self, size, num_discs, num_beacons, mode="def", auto=False, animate=False, p_filter=None
    ):
        """
        Initializes the simulation environment
        Parameters
        ----------
        size : int
            simulation environment size
        num_discs : int
            number of discs in the environment
        num_beacons : int
            number of beacons in the environment
        """
        self.env_size_ = size
        self.num_discs_ = num_discs
        self.num_beacons_ = num_beacons

        # parameters of the process model
        self.spring_force_ = 0.5
        self.drag_force_ = 0.0075

        if p_filter:
            self.particle_filter = p_filter
            self.particles = p_filter.get_particles()

        # auto process steps
        self.auto_ = auto

        # wrap the environment
        self.mode_ = mode

        discs_ = []
        # init discs
        for disc_num in range(self.num_discs_):
            # draw a random position
            pos = np.random.uniform(0, self.env_size_, size=(2))
            # draw a random velocity
            vel = np.random.normal(loc=0.0, scale=1.0, size=(2)) * 3
            discs_.append(np.array([pos[0], pos[1], vel[0], vel[1]]))
        self.discs_ = np.array(discs_)

        # init beacons
        beacons_ = []
        for beacon_num in range(num_beacons):
            axis = np.random.choice([0, 1])
            if axis == 0:
                y_pos = (np.random.uniform(0, self.env_size_, size=(1))).item()
                x_pos = np.random.choice([0, self.env_size_])
                beacons_.append(np.array([x_pos, y_pos]))
            else:
                x_pos = (np.random.uniform(0, self.env_size_, size=(1))).item()
                y_pos = np.random.choice([0, self.env_size_])
                beacons_.append(np.array([x_pos, y_pos]))
        self.beacons_ = np.array(beacons_)

        self.animate_ = animate
        # start auto run thread
        self.anim_start_ = False
        if self.auto_:
            self.lock_ = threading.Lock()
            self.auto_run(1)

        # initialize animation
        if self.animate_:
            self.animator_ = Animator(self.env_size_, self.beacons_, True, True)
            self.animator_.set_data(self.discs_, self.particle_filter.get_particles(), self.particle_filter.get_estimate())
            self.anim_start_ = True
            plt.show()

    def get_discs(self):
        return np.array(self.discs_)

    def _process_model(self, states, dt):
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
        noise = np.random.normal(loc=0.0, scale=0.1, size=states.shape)
        states += noise

        new_state = np.copy(states)
        new_state[:, 0] += dt * states[:, 2]
        new_state[:, 1] += dt * states[:, 3]

        out_left = new_state[:, 0] < 0
        out_right = new_state[:, 0] > self.env_size_
        out_bottom = new_state[:, 1] < 0
        out_top = new_state[:, 1] > self.env_size_

        if self.mode_ == "collide":
            new_state[out_left, 0] = 0
            new_state[out_right, 0] = self.env_size_
            new_state[out_bottom, 1] = 0
            new_state[out_top, 1] = self.env_size_
            new_state[out_left | out_right, 2] *= -1
            new_state[out_top | out_bottom, 3] *= -1

        elif self.mode_ == "wrap":
            new_state[out_top, 0] = new_state[:, 0] - self.env_size_
            new_state[out_bottom, 0] = self.env_size_ - new_state[:, 0]
            new_state[out_left, 1] = self.env_size_ - new_state[:, 1]
            new_state[out_right, 1] = new_state[:, 1] - self.env_size_

        if self.animate_ and not self.auto_:
            self.animator_.set_data(new_state, self.particle_filter.get_particles(), self.particle_filter.get_estimate())
        return new_state

    def update_step(self, dt):
        """
        Calculates the next state of the discs.
        """
        for disc_num in range(self.num_discs_):
            self.discs_ = self._process_model(self.discs_, dt)
        self.particle_filter.run(self.beacons_, self.get_distance(-1), dt)

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
            The current state (position) of the discs wrt to
            given beacon
        """
        reading = []
        for disc_num in range(self.num_discs_):
            reading.append(self.discs_[disc_num][:2] - self.beacons_[beacon_num])
        return np.asarray(reading)

    def get_distance(self, beacon_num):
        """
        Calculates the absolute distance between the discs and the beacon.
        Parameters
        ----------
        beacon_num : int
            beacon index
        Returns
        -------
        reading : np.array
            The he absolute distance between the discs and the
            given beacon
        """
        noise = np.random.normal(loc=0.0, scale=0.1, size=self.num_beacons_)

        dists = []
        if beacon_num != -1:
            for disc_num in range(self.num_discs_):
                pos = self.discs_[disc_num][:2] - self.beacons_[beacon_num]
                dists.append(math.sqrt(pos[0] ** 2 + pos[1] ** 2))
        else:
            for beacon in self.beacons_:
                pos = self.discs_[0][:2] - beacon
                dists.append(math.sqrt(pos[0] ** 2 + pos[1] ** 2))
        return np.asarray(dists + noise)

    def get_beacons_pos(self):
        """
        returns the beacons positions in the simulation environment
        """
        return self.beacons_

    def get_setup(self):
        """
        returns the simulation eviroment settings
        """
        return np.asarray(
            [
                self.env_size_,
                self.num_discs_,
                self.num_beacons_,
                self.beacons_,
                self.mode_,
            ],
            dtype=object,
        )

    def auto_run(self, dt):
        """
        auto runs the simulation enviroment, and updates the step regualry
        """
        self.auto_thread_ = threading.Timer(dt / 10, self.auto_run, args=[dt])
        self.auto_thread_.start()
        with self.lock_:
            self.update_step(dt)
            if self.anim_start_ and self.animate_:
                self.animator_.set_data(self.get_discs(), self.particle_filter.get_particles(), self.particle_filter.get_estimate())


if __name__ == "__main__":
    box = SimulationEnv(200, 1, 2, mode="def", auto=True, animate=True)
