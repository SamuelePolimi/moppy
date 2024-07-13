from typing import List
import lightning as L
import torch
import torch.optim as optim

from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.trajectory.trajectory import Trajectory

from . import DeepProMP


class DeepProMo_Lighning(L.LightningModule, DeepProMP):
    def __init__(self,
                 name: str,
                 encoder: EncoderDeepProMP,
                 decoder: DecoderDeepProMP,
                 save_path: str = './deep_promp/output/',
                 learning_rate: float = 0.005,
                 epochs: int = 100,
                 beta: float = 0.01):
        L.LightningModule.__init__(self)
        DeepProMP.__init__(self, name, encoder, decoder, save_path, learning_rate, epochs, beta)

    def forward(self, x: Trajectory):
        mu, sigma = self.encoder(x)
        latent_var_z = self.encoder.sample_latent_variables(mu, sigma, len(x))
        times = torch.tensor(x.get_times()).reshape(-1, 1)
        decoded = self.decoder(latent_var_z, times)
        return mu, sigma, decoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        use_traj: List[Trajectory] = []
        trajectories, lengths = batch
        for i, tra in enumerate(trajectories):
            nt = Trajectory()
            for j in range(lengths[i]):
                nt.add_point(self.encoder.trajectory_state_class.from_vector(tra[j]))
            use_traj.append(nt)
        beta = self.kl_annealing_scheduler(i+1, n_cycles=4, max_epoch=self.epochs, saturation_point=0.5)

        loss_total = 0

        for data in use_traj:
            mu, sigma, decoded = self.forward(data)
            loss, mse, kl = DeepProMP.calculate_elbo(decoded.reshape(-1, 1), data.to_vector().reshape(-1, 1), mu, sigma, beta)
            loss_total += loss
        loss = loss_total / len(use_traj)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
