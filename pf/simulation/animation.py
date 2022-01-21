import numpy as np
import cv2
import matplotlib.pyplot as plt


class Animator:
    def __init__(self, size, beacons, show=True):
        self.env_size_ = size
        self.fig = plt.figure()
        self.show = show
        self.beacons_ = beacons
        self.discs_ = None

        self.particles_ = None
        self.estimate_ = None

        # set up the figure
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
        (self.discs_vis,) = self.ax.plot([], [], "ko", ms=6)

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

    def plot_fig(self):
        """
        Plots the plt figure

        :return ploted figure
        """

        # update discs_vis the animation
        self.rect.set_edgecolor("k")

        if self.discs_ is not None:
            self.discs_vis.set_data(self.discs_[:, 0], self.discs_[:, 1])

        # update particles visualization
        if self.particles_ is not None:
            self.particles_vis.set_data(self.particles_[:, 0], self.particles_[:, 1])

        if self.estimate_ is not None:
            self.estimate_vis.set_data(
                self.estimate_[0][0], self.estimate_[0][1]
            )  # only `mean` (i.e. [0]) is shown
        return self.fig

    def set_data(self, discs, particles=None, estimate=None):
        """
        Updates the data, plots the new figure and visualizes it in cv2 window

        :param discs: updated disc states
        :param particles: updated particles states
        :param estimate: updated estimate state
        """
        self.discs_ = discs
        if particles is not None:
            self.particles_ = particles

        if estimate is not None:
            self.estimate_ = estimate
        self.plot_fig()

        if self.show:
            cv2.imshow("Simulation", self.convert(self.fig))

            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
                return False

        return True
    
    def get_figure(self):
        return self.fig

    def convert(self, fig):
        """
        Converts plt figure to cv2 image

        :param fig: plt figure
        :return: cv2 image
        """
        fig.canvas.draw()
        # convert canvas to image
        graph_image = np.array(fig.canvas.get_renderer()._renderer)
        # it still is rgb, convert to opencv's default bgr
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
        return graph_image
