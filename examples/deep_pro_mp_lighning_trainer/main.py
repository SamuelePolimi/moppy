import os
import time
import lightning as L
import torch
import torch.nn as nn
import torch.utils.data as data

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning import Trainer, LightningModule

from moppy.deep_promp import DecoderDeepProMP, EncoderDeepProMP, DeepProMo_Lighning
from moppy.trajectory import TrajectoryDataset
from moppy.trajectory.state import JointConfiguration, SinusState

L.seed_everything(42)

DATASET_PATH = os.environ.get("PATH_DATASETS", "data")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "models")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

trajectory_state_class = SinusState
latent_dim = 2
epochs = 100
lr = 0.005
hidden_neurons = [10, 20, 20, 10]
activation_function = nn.ReLU
folder_path = '../sinus/trajectories'


def define_for_rlbench():
    global latent_dim, epochs, lr, hidden_neurons, activation_function, trajectory_state_class, folder_path
    latent_dim = 10
    epochs = 100
    lr = 0.005
    hidden_neurons = [128, 128]
    activation_function = nn.ReLU
    trajectory_state_class = JointConfiguration
    folder_path = '../rlbench/trajectories/ReachTarget'


define_for_rlbench()


train_dataset = TrajectoryDataset(folder_path=folder_path, state_class=trajectory_state_class)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45, 5])
train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)
val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, num_workers=16)


class PrintEpochSummary(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: LightningModule):
        self.start_time = time.time()  # Record start time at epoch beginning

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        current_epoch = trainer.current_epoch
        time_taken = time.time() - self.start_time if self.start_time is not None else 0
        print(f"Epoch end: [{current_epoch+1}/{trainer.max_epochs}] | Time: {time_taken:.2f}s")


def train(latent_dim=latent_dim,
          epochs=epochs,
          lr=lr,
          hidden_neurons=hidden_neurons,
          activation_function=activation_function,
          trajectory_state_class=trajectory_state_class):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH,
                                      "latent_dim_%i" % latent_dim),
        accelerator="auto",  # Use as many GPUS as are available
        devices="auto",  # Use as many as are available
        max_epochs=epochs,
        enable_progress_bar=False,  # Do not show the progressbar, because of the outputfile will fill up
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
            PrintEpochSummary(),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    encoder = EncoderDeepProMP(latent_variable_dimension=latent_dim,
                               hidden_neurons=hidden_neurons,
                               activation_function=activation_function,
                               trajectory_state_class=trajectory_state_class)

    decoder = DecoderDeepProMP(latent_variable_dimension=latent_dim,
                               hidden_neurons=hidden_neurons,
                               activation_function=activation_function,
                               trajectory_state_class=trajectory_state_class)

    model = DeepProMo_Lighning(name="7_joint_reach_target",
                               encoder=encoder,
                               decoder=decoder,
                               learning_rate=lr,
                               epochs=epochs,
                               save_path="./output/")

    trainer.fit(model, train_loader, val_loader)
    model.save_models()
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result


model_ld, result_ld = train()
