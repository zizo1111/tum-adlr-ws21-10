import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pf.filters.diff_particle_filter import DiffParticleFilter
from pf.utils.dataset import PFDataset, DatasetSeq, Sequence
from pf.utils.loss import RMSE
from pf.models.motion_model import MotionModel
from pf.models.observation_model import ObservationModel


def train_epoch(train_loader, model, loss_fn, optimizer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        state = data["state"]
        measurement = data["measurement"]
        setting = data["setting"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(measurement, setting["beacons_pos"])

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)
        loss = F.mse_loss(outputs, state_reshaped)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

    return last_loss


def train(train_set, val_set=None, test_set=None):
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"
    num_discs = 1
    # PF config #
    num_particles = 100

    # beacons config -> TODO #
    num_beacons = 2

    dynamics_model = MotionModel(
        state_dimension=state_dim,
        env_size=env_size,
        mode=mode,
    )
    observation_model = ObservationModel(
        state_dimension=2,
        env_size=env_size,
        num_particles=num_particles,
        num_beacons=num_beacons,
    )

    # Hyperparameters for differential filter #
    hparams = {
        "num_particles": num_particles,
        "state_dimension": state_dim,
        "env_size": env_size,
        "soft_resample_alpha": 0.7,
        "batch_size": 8,
    }

    model = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )

    train_dataloader = DataLoader(
        train_set, batch_size=hparams["batch_size"], shuffle=False, num_workers=0
    )

    # TODO loss
    loss_fn = RMSE
    # TODO change optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 1
    model.train(True)

    for i in range(EPOCHS):
        epoch_loss = train_epoch(train_dataloader, model, loss_fn, optimizer)
        print(epoch_loss)


if __name__ == "__main__":
    train_set = PFDataset("datasets/dataset.npy")
    train(train_set)
