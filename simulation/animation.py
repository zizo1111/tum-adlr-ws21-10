import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animator:
    def __init__(self, size, beacons, plot_particles=False, plot_estimates=False):
        self.env_size_ = size
        # plt.ion()
        self.fig = plt.figure()

        self.beacons_ = beacons
        self.discs_ = None

        self.plot_particles = plot_particles
        self.plot_estimates = plot_estimates

        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            frames=600,
            interval=100,
            blit=True,
            init_func=self.init_anim,
        )
        plt.show(block=False)

    def init_anim(self):
        """initialize animation"""
        # set up figure and animation

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_size = self.env_size_ + 100
        self.ax = self.fig.add_subplot(
            5,
            1,
            (1, 4),
            aspect="equal",
            autoscale_on=True,
            xlim=(0, self.env_size_),
            ylim=(0, self.env_size_),
        )

        # discs_vis holds the locations of the discs
        (self.discs_vis,) = self.ax.plot([], [], "bo", ms=6)

        # beacons drawn as red dots
        (self.beacons_vis,) = self.ax.plot([], [], "ro", ms=6)
        self.beacons_vis.set_data(self.beacons_[:, 0], self.beacons_[:, 1])

        # init particles visualization
        (self.particles_vis,) = self.ax.plot([], [], "b.", ms=1)
        (self.estimate_vis,) = self.ax.plot([], [], "go", ms=5)

        # rect is the box edge
        self.rect = plt.Rectangle(
            (0, 0), self.env_size_, self.env_size_, ec="none", lw=2, fc="none"
        )

        self.ax.add_patch(self.rect)
        self.discs_vis.set_data([], [])
        self.particles_vis.set_data([], [])
        self.estimate_vis.set_data([], [])
        self.rect.set_edgecolor("none")

        return self.discs_vis, self.rect

    def animate(self, i):
        """perform animation step"""

        # update discs_vis the animation
        self.rect.set_edgecolor("k")

        if self.discs_ is not None:
            self.discs_vis.set_data(self.discs_[:, 0], self.discs_[:, 1])

        # update particles visualization
        if self.plot_particles:
            self.particles_vis.set_data(self.particles_[:, 0], self.particles_[:, 1])

        if self.plot_estimates:
            self.estimate_vis.set_data(self.estimates_[0][0], self.estimates_[0][1])

        return self.discs_vis, self.particles_vis, self.estimate_vis, self.rect

    def set_data(self, discs, particles=None, estimates=None):

        self.discs_ = discs
        if self.plot_particles and particles is not None:
            self.particles_ = particles

        if self.plot_estimates and estimates is not None:
            self.estimates_ = estimates
