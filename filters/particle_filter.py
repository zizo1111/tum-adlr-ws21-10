import numpy as np
from scipy import stats


class ParticleFilter:
    def __init__(self, num_particles, state_dimension, env_size, motion_model):
        self.num_particles = num_particles
        self.state_dimension = state_dimension
        self.env_size = env_size
        self.motion_model = motion_model

        self.particles = np.empty((num_particles, state_dimension))
        # distribute discs_vis randomly with uniform weights
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self.particles[:, 0] = np.random.uniform(0, 2 * env_size, size=num_particles)
        self.particles[:, 1] = np.random.uniform(0, 2 * env_size, size=num_particles)
        self.particles[:, 2] = np.random.uniform(0, 3, size=num_particles)
        self.particles[:, 3] = np.random.uniform(0, 2 * np.pi, size=num_particles)

        self.estimate = self._estimate()

    def run(self, beacons, distances):
        self._predict()
        self._update(beacons, distances)
        #self._resample()
        return self._estimate()

    def _predict(self):
        """
        Prediction step of Particle filter

        :return: New set of particles after applying motion model
        """
        self.particles = self.motion_model.forward(self.particles)

    def _update(self, beacons, disc_distances):
        """
        Update step of Particle filter
        :param beacons:
        :param disc_distances:
        :return:
        """
        ##distance = np.linalg.norm(self.particles[:, 0:2] - beacons[0], axis=1)
        for i, beacon in enumerate(beacons):
            d = np.linalg.norm(self.particles[:, 0:2] - beacon, axis=1)
            ##distance = np.vstack((distance, d))
        ##distance = np.linalg.norm(distance.T - disc_distances, axis=1)
            self.weights *= stats.norm(d, 0.1).pdf(disc_distances[i])

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def _resample(self):
        # TODO re-think implementation
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / self.num_particles)

    def _estimate(self):
        """
        Compute state estimate (given particles and their weights)

        :return: Mean and variance of the state estimate
        """
        pos = self.particles[:, 0:self.state_dimension]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean)**2, weights=self.weights, axis=0)
        #print(mean)
        #print(var)
        return mean, var

    def get_particles(self):
        return self.particles
