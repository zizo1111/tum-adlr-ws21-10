import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pf.filters.diff_particle_filter import DiffParticleFilter
from pf.utils.dataset import PFDataset, DatasetSeq, Sequence, PFSampler
from pf.utils.loss import MSE, RMSE, NLL
from pf.utils.utils import normalize, unnormalize_estimate
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

        norm_state, norm_measurement, norm_beacon_pos = normalize(
            state, measurement, setting
        )

        # Make predictions for this batch
        estimate, weights, particles_states = model(norm_measurement, norm_beacon_pos)

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)

        if loss_fn in [MSE, RMSE]:
            loss += loss_fn(norm_state.reshape(N, -1), estimate)
        elif loss_fn in [NLL]:
            loss += loss_fn(
                norm_state,
                weights,
                particles_states,
                covariance=model.component_covariances,
            )

        running_loss += loss.item()
        if i % seq_len == seq_len - 1:
            loss /= seq_len
            loss.backward()
            # Adjust learning weights
            optimizer.step()

            last_loss = running_loss / (seq_len * seq_len)
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

            env_size = setting["env_size"].clone().detach().numpy()[0]
            beac_pos = setting["beacons_pos"].clone().detach().numpy()[0]
            st = state_reshaped.clone().detach().numpy()[0].reshape(1, 4)
            unnorm_est, unorm_particles = unnormalize_estimate(
                estimate.clone().detach(), particles_states.clone().detach(), env_size
            )
            particles_st = unorm_particles[0]
            est = unnorm_est[0].reshape(1, 4)
            print("train:", est, st)
            animator = Animator(env_size, beac_pos, show=False)
            _ = animator.set_data(st, estimate=est, particles=particles_st)

            writer.add_scalar(
                "training loss", last_loss / seq_len, epoch * len(train_loader) + i
            )
            writer.add_figure(
                "estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(train_loader) + i,
            )
    return last_loss, model


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

        norm_state, norm_measurement, norm_beacon_pos = normalize(
            state, measurement, setting
        )

        # Make predictions for this batch
        estimate, weights, particles_states = model(norm_measurement, norm_beacon_pos)

        # Compute the loss and its gradients
        N = state.shape[0]  # batch_size
        state_reshaped = state.reshape(N, -1)

        if loss_fn in [MSE, RMSE]:
            loss += loss_fn(norm_state.reshape(N, -1), estimate)
        elif loss_fn in [NLL]:
            loss += loss_fn(
                norm_state,
                weights,
                particles_states,
                covariance=model.component_covariances,
            )

        running_loss += loss.item()
        if i % seq_len == seq_len - 1:
            loss /= seq_len

            last_loss = running_loss / (seq_len * seq_len)
            print("Val  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

            env_size = setting["env_size"].clone().detach().numpy()[0]
            beac_pos = setting["beacons_pos"].clone().detach().numpy()[0]
            st = state_reshaped.clone().detach().numpy()[0].reshape(1, 4)
            unnorm_est, unorm_particles = unnormalize_estimate(
                estimate.clone().detach(), particles_states.clone().detach(), env_size
            )
            particles_st = unorm_particles[0]
            est = unnorm_est[0].reshape(1, 4)
            print("val:", est, st)
            animator = Animator(env_size, beac_pos, show=True)
            _ = animator.set_data(st, estimate=est, particles=particles_st)

            writer.add_scalar(
                "val loss", last_loss / seq_len, epoch * len(val_loader) + i
            )
            writer.add_figure(
                "Val estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(val_loader) + i,
            )
    return last_loss


def pretrain_motion_epoch(
    train_loader, model, loss_fn, optimizer, train_set_settings, writer, epoch
):
    running_loss = 0.0
    last_loss = 0.0
    loss = torch.zeros(size=[])
    seq_len = train_set_settings["sequence_length"]

    for i, data in enumerate(train_loader):
        if i % seq_len == 0:
            # New sequences -> zero everything, and re-initialize beliefs of the PF
            last_state = None
        optimizer.zero_grad()
        loss = torch.zeros(size=[])

        state = data["state"]
        measurement = data["measurement"]
        setting = data["setting"]

        norm_state, _, _ = normalize(state, measurement, setting)

        if last_state is not None:
            # Make predictions for this batch
            particle_states_diff = model(last_state)
            estimate = last_state + particle_states_diff
            N = state.shape[0]  # batch_size
            estimate = estimate.reshape(N, -1)
            # Compute the loss and its gradients
            state_reshaped = state.reshape(N, -1)

            if loss_fn in [MSE, RMSE]:
                loss += loss_fn(norm_state.reshape(N, -1), estimate)

            running_loss += loss.item()
        last_state = norm_state
        if i % seq_len == seq_len - 1:
            loss /= seq_len
            loss.backward()
            # Adjust learning weights
            optimizer.step()

            last_loss = running_loss / (seq_len * seq_len)
            print("Motion  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

            env_size = setting["env_size"].clone().detach().numpy()[0]
            beac_pos = setting["beacons_pos"].clone().detach().numpy()[0]
            st = state_reshaped.clone().detach().numpy()[0].reshape(1, 4)
            unnorm_est, _ = unnormalize_estimate(
                estimate.clone().detach(), None, env_size
            )
            est = unnorm_est[0].reshape(1, 4)
            print("Motion model train:", est, st)
            print(estimate[0], est)
            animator = Animator(env_size, beac_pos, show=True)
            _ = animator.set_data(st, estimate=est)

            writer.add_scalar(
                "Motion training loss",
                last_loss / seq_len,
                epoch * len(train_loader) + i,
            )
            writer.add_figure(
                "Motion estimation vs. actual",
                animator.get_figure(),
                global_step=epoch * len(train_loader) + i,
            )
    return last_loss, model


def train(
    train_set, val_set=None, test_set=None, pretrain_motion=True, fix_weights=False
):
    # General configuration #
    state_dim = 4
    env_size = 200
    mode = "collide"
    num_discs = 1
    # PF config #
    num_particles = 500

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
        num_discs=num_discs,
    )

    # Hyperparameters for differential filter #
    hparams = {
        "num_particles": num_particles,
        "state_dimension": state_dim,
        "env_size": env_size,
        "soft_resample_alpha": 0.7,
        "batch_size": 8,
        "lr_decay": False,
    }

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
    losses_pretrain_motion = [MSE, RMSE]

    train_set_settings = train_set.get_settings()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/exp1")

    if pretrain_motion:
        loss_fn = MSE
        assert loss_fn in losses_pretrain_motion

        optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1.0e-3)
        dynamics_model.to(device)
        dynamics_model.train(True)
        PRE_EPOCHS = 5
        for i in range(PRE_EPOCHS):
            epoch_loss, dynamics_model = pretrain_motion_epoch(
                train_dataloader,
                dynamics_model,
                loss_fn,
                optimizer,
                train_set_settings,
                writer,
                i,
            )
            print("Epoch {}, loss: {}".format(i + 1, epoch_loss))

    EPOCHS = 100
    loss_fn = MSE
    assert loss_fn in losses

    pf_model = DiffParticleFilter(
        hparams=hparams,
        motion_model=dynamics_model,
        observation_model=observation_model,
    )

    print(pf_model)
    pf_model.to(device)

    if pretrain_motion and fix_weights:
        # fix weights of dynamics model
        dynamics_model.train(False)
        observation_model.train(True)
        optimizer = torch.optim.Adam(observation_model.parameters(), lr=1.0e-4)
    else:
        pf_model.train(True)
        # `weight_decay=1.0e-2` also showed good results
        optimizer = torch.optim.Adam(pf_model.parameters(), lr=5.0e-4)
        if hparams["lr_decay"]:
            decayRate = 0.98
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=decayRate
            )
    patience = 3
    curr_patience = 0
    last_loss = 1e8
    state_dict = None
    saved = False
    for i in range(EPOCHS):
        epoch_loss, pf_model = train_epoch(
            train_dataloader,
            pf_model,
            loss_fn,
            optimizer,
            train_set_settings,
            writer,
            i,
        )
        print("Epoch {}, loss: {}".format(i + 1, epoch_loss))

        if hparams["lr_decay"]:
            if i % 2 == 0 and i > 0:
                my_lr_scheduler.step()

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
    train_set = PFDataset("datasets/dataset_50.npy")
    val_set = PFDataset("datasets/dataset_50_val.npy")
    train(train_set, val_set)
