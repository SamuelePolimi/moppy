from typing import List, Tuple

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

from matplotlib import pyplot as plt

from moppy.deep_promp import EncoderDeepProMP
from moppy.interfaces import MovementPrimitive, LatentEncoder, LatentDecoder
from moppy.kid_promp.forward_kinematics import quat_mul_batch
from moppy.kid_promp.kid_decoder import DecoderKIDProMP
from moppy.trajectory import Trajectory




class KIDPMP(MovementPrimitive):

    def __init__(self,
                 name: str,
                 encoder: EncoderDeepProMP,
                 decoder: DecoderKIDProMP,
                 save_path: str = './deep_promp/output/',
                 learning_rate: float = 0.005,
                 epochs: int = 100,
                 beta: float = 0.01):
        super().__init__(name, encoder, decoder)

        # Check if the encoder and decoder are compatible
        if encoder.latent_variable_dimension != decoder.latent_variable_dimension:
            raise ValueError("The encoder and decoder must have the same latent variable dimension. "
                             f"Got encoder latent variable dimension '{encoder.latent_variable_dimension}'"
                             f"and decoder latent variable dimension '{decoder.latent_variable_dimension}'")
        self.latent_variable_dimension = encoder.latent_variable_dimension

        self.encoder = encoder
        self.decoder = decoder
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta

        self.transpose = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        self.rotate = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.scale = torch.tensor([1.0], requires_grad=True)

        # Initialize the losses lists
        self.train_loss = []  # Training loss => ELBO
        self.kl_train_loss = []  # KL divergence => Part of the ELBO aka training loss
        self.mse_train_loss = []  # Mean Squared Error => Part of the ELBO aka training loss
        self.loss_validation = []  # Validation loss => MEAN Squared Error

    @staticmethod
    def kl_annealing_scheduler(current_epoch, n_cycles=4, max_epoch=1000, saturation_point=0.5):
        """KL annealing scheduler"""
        tau = ((current_epoch - 1) % (math.ceil(max_epoch / n_cycles))) / (math.ceil(max_epoch / n_cycles))
        return tau / saturation_point if tau < saturation_point else 1

    @staticmethod
    def gauss_kl(mu_q, std_q):
        """Calculate the Kullback-Leibler (KL) divergence between a Gaussian distribution and a standard Gaussian distribution."""

        return torch.mean(-torch.log(std_q) + (std_q ** 2 + mu_q ** 2) / 2 - 0.5)

    def elbo_batch(self, y_pred, y_star, mu, sigma, beta=1.0, gamma=0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse = self.mse_pose_batch(y_pred, y_star)

        # KL divergence between approximate posterior (q) and prior (p)
        kl = KIDPMP.gauss_kl(mu_q=mu, std_q=sigma)

        penalty = torch.norm(self.rotate - torch.tensor([0.0, 0.0, 0.0, 1.0])) + torch.square(self.scale - 1.0)

        # Combine terms with beta weighting
        elbo = mse + kl * beta + penalty * gamma
        return elbo, mse, kl

    def mse_pose_batch(self, y_pred, y_star):
        """
        For multiple poses, the poses are expected to be in the shape (n, 7) where n is the number of poses.
        """

        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_star = y_star.unsqueeze(0)

        # transpose the pose
        y_pred_pos = y_pred[:, :3] + self.transpose

        # rotate the pose
        y_pred_quat = y_pred[:, 3:]  # TODO quat_mul_batch(torch.unsqueeze(self.rotate, 0), y_pred[:, 3:])

        # scale the pose
        y_pred_pos *= self.scale

        y_star_pos = y_star[:, :3]
        y_star_quat = y_star[:, 3:]

        mse_pos = nn.MSELoss()(y_pred_pos, y_star_pos)

        # Siciliano 3.91
        # pred.eta * desired.epsilon - desired.eta * pred.epsilon - cross(desired.epsilon, pred.epsilon)
        e_o = ((y_pred_quat[:, 3] * y_star_quat[:, :3].T).T - (y_star_quat[:, 3] * y_pred_quat[:, :3].T).T
               - torch.cross(y_star_quat[:, :3], y_pred_quat[:, :3], dim=-1))
        mse_quat = nn.MSELoss()(e_o, torch.zeros_like(e_o))

        return mse_pos + mse_quat

    def train(self,
              trajectories: List[Trajectory],
              kl_annealing=True,
              beta: float = None,
              learning_rate: float = None,
              epochs: int = None) -> None:
        """Train the DeepProMP using the given trajectories. The training is done using the Evidence Lower Bound (ELBO).
        The ELBO is the loss function used to train the DeepProMP. The training is done using the Adam optimizer."""
        if beta is not None:
            self.beta = beta
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs

        training_start_time = time.time()
        print("Start Kinematic Informed ProMP training ...")

        training_set = trajectories[:(len(trajectories) * 9) // 10]
        validation_set = trajectories[-len(trajectories) // 10:]
        print("Total set: ", len(trajectories))
        print(f"Training set: {len(training_set)}")
        print(f"Validation set: {len(validation_set)}")

        optimizer = optim.Adam(params=list(self.encoder.net.parameters()) + list(self.decoder.net.parameters())
                                      + [self.transpose, self.rotate, self.scale],
                               lr=self.learning_rate)
        kl_traj = []
        mse_traj = []
        losses_validation = []
        elbo_loss_traj = []

        for i in range(self.epochs):
            start_time = time.time()
            mse_tot = 0
            kl_tot = 0
            loss_tot = 0
            for tr_i, data in enumerate(training_set):
                optimizer.zero_grad()  # Zero the gradients of the optimizer to avoid accumulation
                mu, sigma = self.encoder(data)
                # latent_var_z = self.encoder.sample_latent_variable(mu, sigma)

                latent_var_z = self.encoder.sample_latent_variables(mu, sigma, len(data))

                times = torch.tensor(data.get_times()).reshape(-1, 1)
                decoded = self.decoder(latent_var_z, times)

                if kl_annealing:
                    # note that I added a * self.beta here so the maximum can be lowered.
                    beta = KIDPMP.kl_annealing_scheduler(i + 1, n_cycles=4, max_epoch=self.epochs,
                                                         saturation_point=0.5) * self.beta
                else:
                    beta = self.beta

                loss, mse, kl = self.elbo_batch(decoded, data.to_vector_2d(), mu, sigma, beta)

                loss.backward()
                optimizer.step()
                mse_tot += mse.detach().numpy()
                kl_tot += kl.detach().numpy()
                loss_tot += loss.detach().numpy()

            kl_traj.append(kl_tot / len(training_set))
            mse_traj.append(mse_tot / len(training_set))
            elbo_loss_traj.append(loss_tot / len(training_set))
            # validation
            validation_loss = self.validate(validation_set)
            losses_validation.append(validation_loss)
            duration = time.time() - start_time
            num_digits_epochs = len(str(abs(self.epochs)))  # Number of digits of the epochs to format the output
            print(f"Epoch {i + 1:{num_digits_epochs}}/{self.epochs} "
                  f"({duration:.2f}s): "
                  f"validation loss = {validation_loss.item():12.10f}, "
                  f"train_loss = {elbo_loss_traj[-1].item():12.10f}, "
                  f"mse = {mse_traj[-1].item():12.10f}, "
                  f"kl = {kl_traj[-1].item():12.10f}")
            print(f"trans: {self.transpose}, rot: {self.rotate}, scale: {self.scale}")

        print(f"Training finished (Time = {(time.time() - training_start_time):.2f}s).")

        self.loss_validation = losses_validation
        self.mse_train_loss = mse_traj
        self.kl_train_loss = kl_traj
        self.train_loss = elbo_loss_traj

        print("Saving losses ...", end='', flush=True)
        self.save_losses()
        print("finished.\nPlotting...", end='', flush=True)
        self.save_losses_plots()
        print("finished.\nSaving models...", end='', flush=True)
        self.save_models()
        print("finished.")

    def test(self):
        raise NotImplementedError()

    def validate(self, trajectories: List[Trajectory]):
        loss = 0
        for traj in trajectories:
            mu, sigma = self.encoder(traj)
            latent_var_z = self.encoder.sample_latent_variable(mu, sigma)
            decoded = []
            for j in traj.get_points():
                decoded.append(self.decoder(latent_var_z, j.get_time()))
            decoded = torch.cat(decoded)

            loss += self.mse_pose_batch(decoded, traj.to_vector_2d()).detach().numpy()
        return loss / len(trajectories)  # Average loss

    def save_models(self, save_path: str = None):
        """Save the encoder and decoder models to the given path. If no path is given, the default save_path is used."""
        use_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(use_path):
            os.makedirs(use_path)
        self.decoder.save_model(use_path)
        self.encoder.save_model(use_path)

    def save_losses(self, save_path: str = None):
        """Save the losses to the given path. If no path is given, the default save_path is used."""
        save_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        values_to_plot_with_filenames = [
            (self.loss_validation, 'validation_loss.pth'),
            (self.kl_train_loss, 'kl_loss.pth'),
            (self.mse_train_loss, 'mse_loss.pth'),
            (self.train_loss, 'train_loss.pth')
        ]

        for values, file_name in values_to_plot_with_filenames:
            file_path = os.path.join(save_path, file_name)
            torch.save(values, file_path)

    def save_losses_plots(self, save_path: str = None):
        """Save the plots of the losses to the given path. If no path is given, the default save_path is used."""
        save_path = save_path if save_path is not None else self.save_path  # Use the given path or the default one

        self.plot_values(values=[self.loss_validation], path=save_path, file_name='validation_loss.png',
                         plot_title='validation loss')
        self.plot_values(values=[self.kl_train_loss], path=save_path, file_name='kl_loss.png', plot_title="kl loss")
        self.plot_values(values=[self.mse_train_loss], path=save_path, file_name='ms_loss.png', plot_title="ms loss")
        self.plot_values(values=[self.train_loss], path=save_path, file_name='train_loss.png', plot_title="Traing Loss")

    def plot_values(self,
                    values: List[List],
                    file_name: str,
                    plot_title: str = "Plot",
                    path: str = None, ):
        """
        Plot the given values and save the plot to the given path. If no path is given, the default save_path is used.

        values: List[List]: The values to plot. Each list in the list is a line in the plot. (Cannot be None or empty)
        """
        if values is None or len(values) == 0:
            raise ValueError(
                f"Cannot plot '{plot_title}' at '{path}' without values. Please provide  at least one value list.")

        if not path:
            path = self.save_path
        if not os.path.exists(path):
            os.makedirs(path)

        file_path = os.path.join(path, file_name)
        plt.close()
        for i in values:
            plt.plot(i)
            plt.title(plot_title)
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.grid(True)

        # Save the plot as an image file
        plt.savefig(file_path)

    def __str__(self):
        return f"DeepProMP({self.name})" '{' + \
            '\n\t' + f"learning_rate: {self.learning_rate}" + \
            '\n\t' + f"epochs: {self.epochs}" + \
            '\n\t' + f"beta: {self.beta}" + \
            '\n\t' + f"save_path: {self.save_path}" + \
            '\n' + f"encoder: {self.encoder}" + \
            '\n' + f"decoder: {self.decoder}" + \
            '\n' + '}'
