from operator import imod
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pf.filters.diff_particle_filter import DiffParticleFilter
from pf.utils.dataset import PFDataset, DatasetSeq, Sequence, PFSampler
from pf.utils.loss import MSE, RMSE, NLL
from pf.models.motion_model import MotionModel
from pf.models.observation_model import ObservationModel
from pf.simulation.animation import Animator
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def train_epoch(
    train_loader, model, loss_fn, optimizer, train_set_settings, writer, epoch
):
    running_loss = 0.0
    last_loss = 0.0
    loss = torch.zeros(size=[])
    seq_len = train_set_settings["sequence_length"]

    for i, data in enumerate(train_loader):
        if i % seq_len == 0:
            # New sequences -> zero everything, and re-initialize beliefs of the PF
            optimizer.zero_grad()
            loss = torch.zeros(size=[])
            model.init_beliefs()

        state = data["state"]
        measurement = data["measurement"]
        setting = data["setting"]

        # Make predictions for this batch
        estimate, weights, particles_states = model(measurement, setting["beacons_pos"])

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)

        if loss_fn in [MSE, RMSE]:
            loss += loss_fn(state_reshaped, estimate)
        elif loss_fn in [NLL]:
            loss += loss_fn(state, weights, particles_states, covariance=model.gmm.component_distribution.variance)

        running_loss += loss.item()
        if i % seq_len == seq_len - 1:
            loss /= seq_len
            loss.backward()
            # Adjust learning weights
            optimizer.step()

            last_loss = running_loss / (seq_len * seq_len)
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
                "training loss", last_loss / seq_len, epoch * len(train_loader) + i
            )
            writer.add_figure(
                "estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(train_loader) + i,
            )
    return last_loss


def val_epoch(val_loader, model, loss_fn, val_set_settings, writer, epoch):
    running_loss = 0.0
    last_loss = 0.0
    loss = torch.zeros(size=[])
    seq_len = val_set_settings["sequence_length"]

    for i, data in enumerate(val_loader):
        if i % seq_len == 0:
            # New sequences -> zero everything, and re-initialize beliefs of the PF
            loss = torch.zeros(size=[])
            model.init_beliefs()

        state = data["state"]
        measurement = data["measurement"]
        setting = data["setting"]

        # Make predictions for this batch
        estimate, weights, particles_states = model(measurement, setting["beacons_pos"])

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)

        if loss_fn in [MSE, RMSE]:
            loss += loss_fn(state_reshaped, estimate)
        elif loss_fn in [NLL]:
            loss += loss_fn(state, weights, particles_states, covariance=model.gmm.component_distribution.variance)

        running_loss += loss.item()
        if i % seq_len == seq_len - 1:
            loss /= seq_len

            last_loss = running_loss / (seq_len * seq_len)
            print("Val  batch {} loss: {}".format(i + 1, last_loss))
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
                "val loss", last_loss / seq_len, epoch * len(val_loader) + i
            )
            writer.add_figure(
                "Val estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(val_loader) + i,
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
    sampler = PFSampler(train_set, hparams["batch_size"])
    train_dataloader = DataLoader(
        train_set,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=0,
        sampler=sampler,
    )

    if val_set:
        val_sampler = PFSampler(val_set, hparams["batch_size"])
        val_dataloader = DataLoader(
            val_set,
            batch_size=hparams["batch_size"],
            shuffle=False,
            num_workers=0,
            sampler=val_sampler,
        )
    losses = [MSE, RMSE, NLL]
    loss_fn = MSE
    assert loss_fn in losses

    # TODO change optimizer
    optimizer = torch.optim.Adam(pf_model.parameters(), lr=5.0e-3, weight_decay=1.0e-2)

    EPOCHS = 100
    print(pf_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pf_model.to(device)
    pf_model.train(True)
    train_set_settings = train_set.get_settings()

    patience = 5
    curr_patience = 0
    last_loss = 1e8
    writer = SummaryWriter("runs/exp1")
    state_dict = None
    saved = False
    for i in range(EPOCHS):
        epoch_loss = train_epoch(
            train_dataloader,
            pf_model,
            loss_fn,
            optimizer,
            train_set_settings,
            writer,
            i,
        )
        print("Epoch {}, loss: {}".format(i + 1, epoch_loss))

        if val_set:
            epoch_loss_val = val_epoch(
                val_dataloader,
                pf_model,
                loss_fn,
                train_set_settings,
                writer,
                i,
            )
            print("Validation Epoch {}, loss: {}".format(i + 1, epoch_loss_val))

            if epoch_loss_val > last_loss:
                curr_patience += 1

            else:
                state_dict = pf_model.state_dict().copy()
                curr_patience = 0
                last_loss = epoch_loss_val

            if curr_patience > patience:
                print(
                    "Epoch {}: Validation Loss {}, best_loss: {}... Saving model".format(
                        i + 1, epoch_loss_val, last_loss
                    )
                )
                # Saving State Dict
                torch.save(
                    state_dict,
                    "saved_models/saved_model{}.pth".format(
                        datetime.timestamp(datetime.now())
                    ),
                )
                saved = True
                break
    if not saved:
        if state_dict is None:
            state_dict = pf_model.state_dict().copy()
        torch.save(
            state_dict,
            "saved_models/saved_model{}.pth".format(datetime.timestamp(datetime.now())),
        )


if __name__ == "__main__":
    train_set = PFDataset("datasets/dataset.npy")
    val_set = PFDataset("datasets/val_dataset.npy")
    train(train_set, val_set)
