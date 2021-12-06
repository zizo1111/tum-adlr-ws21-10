import numpy as np
from scipy import stats


class ParticleFilter:
    def __init__(self, num_particles, state_dimension, env_size, motion_model, observation_model):
        self.num_particles = num_particles
        self.state_dimension = state_dimension
        self.env_size = env_size
        self.motion_model = motion_model
        self.observation_model = observation_model

        self.particles = np.empty((num_particles, state_dimension))
        # distribute particles randomly with uniform weights
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self.particles[:, 0] = np.random.uniform(0, env_size, size=num_particles)
        self.particles[:, 1] = np.random.uniform(0, env_size, size=num_particles)
        self.particles[:, 2] = np.random.normal(loc=0.0, scale=1.0, size=num_particles) * 3  #np.random.uniform(0, 50, size=num_particles)
        self.particles[:, 3] = np.random.uniform(0, 2 * np.pi, size=num_particles)

        self._estimate()

    def run(self, beacons, distances, dt):
        self._predict(dt)
        self._update(beacons, distances)
        self._resample_residual()
        self._estimate()

    def _predict(self, dt):
        """
        Prediction step of Particle filter

        :return: New set of particles after applying motion model
        """
        self.particles = self.motion_model.forward(self.particles, dt)

    def _update(self, beacons, disc_distances):
        """
        Update step of Particle filter
        :param beacons:
        :param disc_distances:
        :return:
        """
        ##distance = np.linalg.norm(self.particles[:, 0:2] - beacons[0], axis=1)
        distance = self.observation_model.measure(self.particles[:, 0:2], beacons)
        for i, beacon in enumerate(beacons):
            d = np.linalg.norm(self.particles[:, 0:2] - beacon, axis=1)
            ##distance = np.vstack((distance, d))
        ##distance = np.linalg.norm(distance.T - disc_distances, axis=1)
            #self.weights *= stats.norm(d, 0.1).pdf(disc_distances[i])
        #print(disc_distances)
        #print(distance - disc_distances)
        error = np.linalg.norm(distance - disc_distances, axis=1)
        #print(error)
        #print(error.shape)
        """
        pdf = stats.norm(distance - disc_distances, 0.1)
        print(pdf)"""

        for i, b in enumerate(beacons):
            self.weights = self.weights * stats.norm(0.0, 10.0).pdf(np.abs(distance[:, i] - disc_distances[i]))

        ###self.weights = self.weights * stats.norm(0.0, 10.0).pdf(error)  # TODO: + or *?

        self.weights += 1.e-8  # avoid round-off to zero  # TODO: 1e-300 or -12 or -8?
        self.weights /= sum(self.weights)  # normalize

    def _resample_low_variance(self):
        new_particles = []

        r = np.random.uniform(low=0.0, high=1.0 / self.num_particles)
        c = self.weights[0]
        i = 0
        for m in range(self.num_particles):
            u = r + (m - 1) / self.num_particles
            while u > c:
                i += 1
                c += self.weights[i]
            new_particles.append(list(self.particles[i]))

        self.particles = np.array(new_particles)
        self.weights.fill(1.0 / self.num_particles)

    def _resample_residual(self):
        indices = np.zeros(self.num_particles).astype(int)

        # allocate ⌊N*w⌋ copies of each particle
        num_copies = np.floor(self.num_particles * self.weights).astype(int)
        k = 0
        for i in range(self.num_particles):
            for _ in range(num_copies[i]):  # make the copies
                indices[k] = i
                k += 1
        assert k == np.sum(num_copies)

        # compute the residual (to be resampled)
        residual = self.num_particles * self.weights - num_copies  # get fractional part
        residual /= sum(residual)  # normalize

        # either use multinomial resampling to allocate the rest
        # -> maximizes the variance of the samples
        indices = self.__multinomial(indices, residual, k)

        # or use stratified resampling
        #indices = self.__stratified(indices, residual, k)

        self.particles[:] = self.particles[indices]
        self.weights.fill(1.0 / len(self.weights))

        assert len(self.particles) == self.num_particles

    def __multinomial(self, indices, residual, k):
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0  # ensure sum is exactly one
        indices[k:self.num_particles] = np.searchsorted(cumulative_sum, np.random.random(self.num_particles - k))

        return indices

    def __stratified(self, indices, residual, k):
        # TODO: check for correctness
        # make N subdivisions (of the residual interval!),
        # and chose a random position within each one
        positions = (np.random.random(self.num_particles - k) + range(self.num_particles - k)) / (self.num_particles - k)

        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0  # ensure sum is exactly one
        i, j = 0, 0
        while i < self.num_particles - k:  # allocate rest
            if positions[i] < cumulative_sum[j]:
                indices[k + i] = j
                i += 1
            else:
                j += 1

        return indices

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
        self.estimate = mean, var

    def get_particles(self):
        return self.particles

    def get_estimate(self):
        return self.estimate
