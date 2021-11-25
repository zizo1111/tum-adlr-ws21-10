import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

class SimulationEnv():
    def __init__(self, size, num_discs, num_beacons, wrap=False, auto=False, animate=False):
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
        self.spring_force_ = 0.5
        self.drag_force_ = 0.0075
        
        #auto process steps
        self.auto = auto

        #wrap the enviroment
        self.wrap = wrap

        self.discs_ = []
        # init discs
        for disc_num in range(self.num_discs_):
            # draw a random position
            pos = np.random.uniform(0, 2*self.env_size_, size=(2))
            # draw a random velocity
            vel = np.random.normal(loc=0., scale=10., size=(2)) * 3
            self.discs_.append(np.array([pos[0], pos[1], vel[0], vel[1]]))
        
        # init beacons
        beacons_ = []
        for beacon_num in range(num_beacons):
            axis = np.random.choice([0,1])
            if axis == 0:
                y_pos = (np.random.uniform(0, 2*self.env_size_, size=(1))).item()
                x_pos = np.random.choice([0, 2*self.env_size_])
                beacons_.append(np.array([x_pos, y_pos]))
            else:
                x_pos = (np.random.uniform(0, 2*self.env_size_, size=(1))).item()
                y_pos = np.random.choice([0, 2*self.env_size_])
                beacons_.append(np.array([x_pos, y_pos]))
        self.beacons_ = np.array(beacons_)
        
        # initialize animation
        if(animate):
            self.fig = plt.figure()

            self.ani = animation.FuncAnimation(self.fig, self.animate, frames=600,
                                interval=100, blit=True, init_func=self.init_anim)
            plt.show()  
    
    def get_discs(self):
        return np.array(self.discs_)
    
    def _process_model(self, state, dt):
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
        # pull_force = - self.spring_force_ * state[:2]
        # drag_force = - self.drag_force_ * state[2:]**2 * np.sign(state[2:])

        # new_state = np.copy(state)
        new_state[0] += dt*state[2]
        new_state[1] += dt*state[3]
        # new_state[2] += pull_force[0] + drag_force[0]
        # new_state[3] += pull_force[1] + drag_force[1]
        if self.wrap:
            if new_state[0] > 2*self.env_size_:
                new_state[0] = new_state[0] - 2*self.env_size_
            elif new_state[0] < 0:
                new_state[0] = 2*self.env_size_ - new_state[0]
            
            if new_state[1] > 2*self.env_size_:
                new_state[1] = new_state[1] - 2*self.env_size_
            elif new_state[1] < 0:
                new_state[1] = 2*self.env_size_ - new_state[1]
        return new_state
    
    def update_step(self, dt):
        """
        Calculates the next state of the discs.
        """
        for disc_num in range(self.num_discs_):
            self.discs_[disc_num] = self._process_model(self.discs_[disc_num], dt)
    
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
        dists = []
        for disc_num in range(self.num_discs_):
            pos = self.discs_[disc_num][:2] - self.beacons_[beacon_num]
            dists.append(math.sqrt(pos[1]**2 + pos[2]**2))
        return np.asarray(dists)
    
    def get_beacons_pos(self):
        """
        returns the beacons postions in the simulation enviroment
        """
        return self.beacons_

    def init_anim(self):
        """initialize animation"""
        # set up figure and animation
        
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_size = self.env_size_ + 100
        self.ax = self.fig.add_subplot(5, 1, (1,4), aspect='equal', autoscale_on=True,
                            xlim=(0, 2*self.env_size_), ylim=(0, 2*self.env_size_))

        # particles holds the locations of the particles
        self.particles, = self.ax.plot([], [], 'bo', ms=6)

        # beacons drawn as red dots
        self.beacon_particles, = self.ax.plot([], [], 'ro', ms=6)
        self.beacon_particles.set_data(self.beacons_[:,0], self.beacons_[:,1])

        # rect is the box edge
        self.rect = plt.Rectangle([0, 0],
                            2*self.env_size_,
                            2*self.env_size_,
                            ec='none', lw=2, fc='none')

        self.ax.add_patch(self.rect)
        self.particles.set_data([], [])
        self.rect.set_edgecolor('none')

        return self.particles, self.rect

    def animate(self, i):
        """perform animation step"""
        if self.auto:
            # update step
            self.update_step(1/5)
        
        # update particles the animation
        self.rect.set_edgecolor('k')
        self.particles.set_data(self.get_discs()[:,0], self.get_discs()[:,1])

        return self.particles, self.rect


if __name__ == "__main__":
    box = SimulationEnv(200, 1, 2, wrap=True, auto=True, animate=True)