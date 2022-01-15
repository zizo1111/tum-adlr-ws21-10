import numpy as np
import math
from torch.utils.data import Dataset
import torch
from pf.simulation.simulation_env import SimulationEnv
from pf.simulation.animation import Animator


class Sequence:
    def __init__(self, settings):

        self.env_size_ = settings[0]
        self.num_discs_ = settings[1]
        self.num_beacons_ = settings[2]
        self.beacons_ = settings[3]
        self.mode_ = settings[4]
        self.dt_ = settings[5]
        self.states_ = []

    def add_state(self, state):
        self.states_.append(state)

    def get_settings(self):
        settings = {
            "env_size": self.env_size_,
            "num_discs": self.num_discs_,
            "num_beacons": self.num_beacons_,
            "beacons_pos": self.beacons_.astype(np.float32),
            "mode": self.mode_,
            "dt": self.dt_,
        }
        return settings

    def get_length(self):
        return len(self.states_)

    def get_state(self, idx):
        if idx != -1:
            return np.array(self.states_[idx])
        else:
            return np.array(self.states_)

    def get_measurements(self):
        measurements = []
        for idx in range(len(self.states_)):
            measurements.append(self.get_distance(-1, idx))
        return np.array(measurements)

    def get_beacons_pos(self):
        """
        returns the beacons positions in the simulation environment
        """
        return self.beacons_

    def get_reading(self, beacon_num, idx):
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
        ## i dont think we need that ?
        reading = []
        for disc_num in range(self.num_discs_):
            reading.append(self.states_[idx][disc_num][:2] - self.beacons_[beacon_num])
        return np.asarray(reading)

    def get_distance(self, beacon_num, idx):
        """
        Calculates the absolute distance between the discs and the beacon.
        Parameters
        ----------
        beacon_num : int
            beacon index
        Returns
        -------
        distance : np.array
            The he absolute distance between the discs and the
            given beacon
        """
        # TODO should also noise be added here?
        noise = np.random.normal(loc=0.0, scale=0.1, size=self.num_beacons_)

        dists = []
        dists = []
        if beacon_num != -1:
            for disc_num in range(self.num_discs_):
                pos = self.discs_[disc_num][:2] - self.beacons_[beacon_num]
                dists.append(math.sqrt(pos[0] ** 2 + pos[1] ** 2))
        else:
            for disc_num in range(self.num_discs_):
                pos = self.states_[idx][disc_num][:2] - self.beacons_[beacon_num]
                dists.append(math.sqrt(pos[0] ** 2 + pos[1] ** 2))
        return np.asarray(dists + noise)

    def play(self):
        animator = Animator(self.env_size_, self.beacons_)
        for state in self.states_:
            animator.set_data(state)


class DatasetSeq:
    def __init__(
        self,
        create=False,
        num_sequences=100,
        sequence_length=100,
        env_size=200,
        num_discs=1,
        num_beacons=2,
        mode="collide",
        dt=1,
    ):
        self.sequences_ = []
        if create:
            self.num_sequences_ = num_sequences
            self.sequence_length_ = sequence_length
            self.length_ = self.sequence_length_ * self.num_sequences_
            self.env_size_ = env_size
            self.num_discs_ = num_discs
            self.num_beacons_ = num_beacons
            self.mode_ = mode
            self.dt_ = dt
            self.create_dataset()

    def get_sequence_settings(self, idx):
        return self.dataset_[idx][0]

    def create_dataset(self):
        # dataset = []
        for sequence in range(self.num_sequences_):

            if self.mode_ == "random":
                mode_ = np.random.choice(["def", "wrap", "collide"])
            else:
                mode_ = self.mode_

            sim = SimulationEnv(
                self.env_size_, self.num_discs_, self.num_beacons_, mode_, dt=self.dt_
            )
            seq = Sequence(sim.get_setup())

            for frame in range(self.sequence_length_):
                sim.update_step(1)
                seq.add_state(sim.get_discs())

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
            path_ = "datasets/dataset.npy"

        with open(path_, "wb") as f:
            np.save(f, settings)
            np.save(f, np.asarray(self.sequences_, dtype=object), allow_pickle=True)

    def load_dataset(self, path=None):

        path_ = "datasets/dataset.npy"
        if path:
            path_ = path

        with open(path_, "rb") as f:
            settings = np.load(f, allow_pickle=True)
            self.env_size_ = settings[0]
            self.num_discs_ = settings[1]
            self.num_beacons_ = settings[2]
            self.mode_ = settings[3]
            self.num_sequences_ = settings[4]
            self.sequence_length_ = settings[5]
            self.length_ = self.sequence_length_ * self.num_sequences_
            self.sequences_ = np.load(f, allow_pickle=True)

    def get_sequence(self, idx):
        return self.sequences_[idx]

    def get_length(self, getSequence=False):
        if getSequence:
            return self.num_sequences_
        else:
            return self.length_

    def get_item(self, idx, getSequence=False):
        if getSequence:
            seq = self.get_sequence(idx)
            sample = {
                "state": torch.from_numpy(seq.get_state(-1).astype(np.float32)),
                "measurement": torch.from_numpy(
                    seq.get_measurements().astype(np.float32)
                ),
                "setting": seq.get_settings(),
            }
        else:
            seq_idx = math.floor((idx / self.length_) * self.num_sequences_)
            seq = self.get_sequence(seq_idx)
            frame_idx = idx - (seq_idx * seq.get_length())
            measurement = seq.get_distance(-1, frame_idx)
            state = seq.get_state(frame_idx)

            sample = {
                "state": torch.from_numpy(state.astype(np.float32)),
                "measurement": torch.from_numpy(measurement.astype(np.float32)),
                "setting": seq.get_settings(),
            }
        return sample


class PFDataset(Dataset):
    """PF torch dataset."""

    def __init__(self, path, getSequence=False):
        self.datasetSeq_ = DatasetSeq()
        self.datasetSeq_.load_dataset(path)
        self.getSequence_ = getSequence

    def __len__(self):
        return self.datasetSeq_.get_length(self.getSequence_)

    def __getitem__(self, idx):
        return self.datasetSeq_.get_item(idx, self.getSequence_)


# TODO
# incorporate mode in model?
# fix constant sequence length
#
# if __name__ == "__main__":
# set = DatasetSeq(create=True)
# set.save_dataset()

# set = DatasetSeq()
# set.load_dataset()
