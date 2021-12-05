import numpy as np
import math
from simulation_env import SimulationEnv


class Sequence:
    def __init__(self, settings):

        self.env_size_ = settings[0]
        self.num_discs_ = settings[1]
        self.num_beacons_ = settings[2]
        self.beacons_ = settings[3]
        self.mode_ = settings[4]
        self.readings_ = []
        self.current_idx = 0

    def add_reading(self, reading):
        self.readings_.append(reading)

    def get_setup(self):
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

    def get_length(self):
        return len(self.readings_)

    def get_readings(self):
        return self.readings_

    def get_beacons_pos(self):
        """
        returns the beacons positions in the simulation environment
        """
        return self.beacons_

    def get_reading(self, beacon_num, frame_idx=None):
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
        idx = self.current_idx if frame_idx is None else frame_idx
        reading = []
        for disc_num in range(self.num_discs_):
            reading.append(
                self.readings_[idx][disc_num][:2] - self.beacons_[beacon_num]
            )
        self.current_idx += 1
        return np.asarray(reading)

    def get_distance(self, beacon_num, frame_idx=None):
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
        idx = self.current_idx if frame_idx is None else frame_idx

        # TODO should also noise be added here?
        noise = np.random.normal(loc=0.0, scale=0.1, size=self.num_beacons_)

        dists = []
        for disc_num in range(self.num_discs_):
            pos = self.readings_[idx][disc_num][:2] - self.beacons_[beacon_num]
            dists.append(math.sqrt(pos[0] ** 2 + pos[1] ** 2))
        self.current_idx += 1
        return np.asarray(dists + noise)


class Dataset:
    def __init__(
        self,
        create=False,
        num_sequences=10,
        sequence_length=100,
        env_size=200,
        num_discs=1,
        num_beacons=2,
        mode="random",
    ):
        self.sequences_ = []
        self.current_idx = 0
        if create:
            self.num_sequences_ = num_sequences
            self.sequence_length_ = sequence_length
            self.env_size_ = env_size
            self.num_discs_ = num_discs
            self.num_beacons_ = num_beacons
            self.mode_ = mode
            self.create_dataset()

    def get_sequence_settings(self, sequence_index):
        return self.dataset_[sequence_index][0]

    def create_dataset(self):
        # dataset = []
        for sequence in range(self.num_sequences_):

            if self.mode_ == "random":
                mode_ = np.random.choice(["def", "wrap", "collide"])
            else:
                mode_ = self.mode_

            sim = SimulationEnv(
                self.env_size_, self.num_discs_, self.num_beacons_, mode_
            )
            seq = Sequence(sim.get_setup())

            for frame in range(self.sequence_length_):
                sim.update_step(1)
                seq.add_reading(sim.get_discs())

            self.sequences_.append(seq)

    def save_dataset(self, path=None):
        settings = np.asarray(
            [
                self.env_size_,
                self.num_discs_,
                self.num_beacons_,
                self.mode_,
                self.num_sequences_,
                self.sequence_length_,
            ],
            dtype=object,
        )
        path_ = path
        if path is None:
            path_ = "dataset.npy"

        with open(path_, "wb") as f:
            np.save(f, settings)
            np.save(f, np.asarray(self.sequences_, dtype=object), allow_pickle=True)

    def load_dataset(self, path="dataset.npy"):

        with open(path, "rb") as f:
            settings = np.load(f, allow_pickle=True)
            self.env_size_ = settings[0]
            self.num_discs_ = settings[1]
            self.num_beacons_ = settings[2]
            self.mode_ = settings[3]
            self.num_sequences_ = settings[4]
            self.sequence_length_ = settings[5]

            self.sequences_ = np.load(f, allow_pickle=True)

    def get_sequence(self, seq_idx=None):
        idx = self.current_idx if seq_idx is None else seq_idx
        return self.sequences_[idx]


if __name__ == "__main__":
    # set = Dataset(create=True)
    # set.save_dataset()

    set = Dataset()
    set.load_dataset()
