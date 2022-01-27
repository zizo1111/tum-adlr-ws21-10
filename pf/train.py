from operator import imod
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pf.filters.diff_particle_filter import DiffParticleFilter
from pf.utils.dataset import PFDataset, DatasetSeq, Sequence
from pf.utils.loss import MSE, RMSE, NLL
from pf.models.motion_model import MotionModel
from pf.models.observation_model import ObservationModel
from pf.simulation.animation import Animator
from torch.utils.tensorboard import SummaryWriter


def train_epoch(train_loader, model, loss_fn, optimizer, writer, epoch):
    running_loss = 0.0
    last_loss = 0.0

    # with torch.autograd.set_detect_anomaly(True):
    for i, data in enumerate(train_loader):
        # print(i)
        state = data["state"]
        measurement = data["measurement"]
        setting = data["setting"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        estimate, weights, particles_states = model(measurement, setting["beacons_pos"])

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)

        # MSE loss
        # loss = F.mse_loss(outputs, state_reshaped)

        # RMSE loss
        loss = loss_fn(estimate, state_reshaped)
        # loss = NLL(estimate, state, weights, particles_states)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

            env = setting["env_size"].clone().detach().numpy()[0]
            beac_pos = setting["beacons_pos"].clone().detach().numpy()[0]
            st = state_reshaped.clone().detach().numpy()[0].reshape(1, 4)
            est = estimate.clone().detach().numpy()[0].reshape(1, 4)
            animator = Animator(env, beac_pos, show=False)
            _ = animator.set_data(
                st,
                estimate=est,
            )

            writer.add_scalar(
                "training loss", last_loss / 100, epoch * len(train_loader) + i
            )
            writer.add_figure(
                "estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(train_loader) + i,
            )
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
        state_dimension=state_dim,
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

    pf_model = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )

    train_dataloader = DataLoader(
        train_set, batch_size=hparams["batch_size"], shuffle=False, num_workers=0
    )

    losses = [MSE, RMSE, NLL]
    loss_fn = MSE
    assert loss_fn in losses

    # TODO change optimizer
    optimizer = torch.optim.Adam(pf_model.parameters(), lr=1e-3)

    EPOCHS = 100
    print(pf_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pf_model.to(device)
    pf_model.train(True)

    writer = SummaryWriter("runs/exp1")
    for i in range(EPOCHS):
        epoch_loss = train_epoch(
            train_dataloader, pf_model, loss_fn, optimizer, writer, i
        )
        print("Epoch {}, loss: {}".format(i + 1, epoch_loss))


if __name__ == "__main__":
    train_set = PFDataset("datasets/dataset.npy")
    train(train_set)
