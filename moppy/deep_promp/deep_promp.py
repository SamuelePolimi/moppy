from typing import List, Tuple

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

from matplotlib import pyplot as plt

from . import DecoderDeepProMP, EncoderDeepProMP
from moppy.interfaces import MovementPrimitive
from moppy.trajectory import Trajectory
from moppy.logger import Logger


class DeepProMP(MovementPrimitive, Logger):
    """A DeepProMP is a probabilistic movement primitive that uses deep neural networks to encode and decode trajectories."""

    def __init__(self,
                 name: str,
                 encoder: EncoderDeepProMP,
                 decoder: DecoderDeepProMP,
                 save_path: str = './deep_promp/output/',
                 log_to_tensorboard: bool = False,
                 learning_rate: float = 0.005,
                 epochs: int = 100,
                 beta: float = 0.01,
                 n_cycles: int = 8) -> None:
        MovementPrimitive.__init__(self, name, encoder, decoder)
        Logger.__init__(self, log_dir=save_path, logging_enabled=log_to_tensorboard)

        # Check if the encoder and decoder are instances/subclasses of EncoderDeepProMP and DecoderDeepProMP
        if not issubclass(type(encoder), EncoderDeepProMP):
            raise TypeError(f"The encoder must be an instance of '{EncoderDeepProMP.__name__}' or a subclass. Got '{type(encoder)}'."
                            f"\nThe usable classes are {[EncoderDeepProMP] + EncoderDeepProMP.__subclasses__()}")
        if not issubclass(type(decoder), DecoderDeepProMP):
            raise TypeError(f"The decoder must be an instance of '{DecoderDeepProMP.__name__}' or a subclass. Got '{type(decoder)}'."
                            f"\nThe usable classes are {[DecoderDeepProMP] + DecoderDeepProMP.__subclasses__()}")

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
        self.n_cycles = n_cycles

        # Initialize the losses lists
        self.train_loss = []  # Training loss => ELBO
        self.kl_train_loss = []  # KL divergence => Part of the ELBO aka training loss
        self.mse_train_loss = []  # Mean Squared Error => Part of the ELBO aka training loss
        self.loss_validation = []  # Validation loss => MEAN Squared Error

    @staticmethod
    def kl_annealing_scheduler(current_epoch, n_cycles=4, max_epoch=1000, saturation_point=0.5):
        """KL annealing scheduler"""
        tau = ((current_epoch - 1) % (math.ceil(max_epoch / n_cycles))) / (math.ceil(max_epoch / n_cycles))
        return tau/saturation_point if tau < saturation_point else 1

    @staticmethod
    def gauss_kl(mu_q, std_q):
        """Calculate the Kullback-Leibler (KL) divergence between a Gaussian distribution and a standard Gaussian distribution."""

        return torch.mean(-torch.log(std_q) + (std_q ** 2 + mu_q ** 2) / 2 - 0.5)

    @staticmethod
    def calculate_elbo(y_pred, y_star, mu, sigma, beta=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the Evidence Lower Bound (ELBO) using the reconstruction loss and the KL divergence.
        The ELBO is the loss function used to train the DeepProMP."""

        # Reconstruction loss (assuming Mean Squared Error)
        mse = nn.MSELoss()(y_pred, y_star)
        # log_prob = torch.distributions.Normal(loc=mu, scale=sigma).log_prob(torch.tensor(y_star, requires_grad=True)).sum()
        # losses.append(log_prob)
        # KL divergence between approximate posterior (q) and prior (p)
        kl = DeepProMP.gauss_kl(mu_q=mu, std_q=sigma)

        # print("mse %s, kl %s" % (mse, kl))
        # Combine terms with beta weighting
        elbo = mse + kl * beta
        return elbo, mse, kl

    def train(self,
              trajectories: List[Trajectory],
              kl_annealing=True,
              beta: float = None,
              learning_rate: float = None,
              epochs: int = None,
              n_cycles: int = None) -> None:
        """Train the DeepProMP using the given trajectories. The training is done using the Evidence Lower Bound (ELBO).
        The ELBO is the loss function used to train the DeepProMP. The training is done using the Adam optimizer."""
        if beta is not None:
            self.beta = beta
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs
        if n_cycles is not None:
            self.n_cycles = n_cycles

        training_start_time = time.time()
        print("Start DeepProMP training ...")

        training_set = trajectories[:(len(trajectories) * 9) // 10]
        validation_set = trajectories[-len(trajectories) // 10:]
        print("Total set: ", len(trajectories))
        print(f"Training set: {len(training_set)}")
        print(f"Validation set: {len(validation_set)}")

        optimizer = optim.Adam(params=list(self.encoder.net.parameters()) + list(self.decoder.net.parameters()),
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
                    beta = DeepProMP.kl_annealing_scheduler(i+1, n_cycles=self.n_cycles, max_epoch=self.epochs, saturation_point=0.5) * self.beta
                else:
                    beta = self.beta
                loss, mse, kl = DeepProMP.calculate_elbo(decoded.reshape(-1, 1), data.to_vector().reshape(-1, 1), mu, sigma, beta)
                # print(f"{i + 1}/{episodes} - {tr_i + 1}/{len(trajectories)} = {loss.item()}")
                loss.backward()
                optimizer.step()
                mse_tot += mse.detach().numpy()
                kl_tot += kl.detach().numpy()
                loss_tot += loss.detach().numpy()
            self.log_metrics(i, {'mse loss': mse_tot / len(training_set), 'kl loss': kl_tot / len(training_set), 'elbo loss': loss_tot / len(training_set)})

            kl_traj.append(kl_tot / len(training_set))
            mse_traj.append(mse_tot / len(training_set))
            elbo_loss_traj.append(loss_tot / len(training_set))
            # validation
            validation_loss = self.validate(validation_set)
            self.log_metrics(i, {'validation loss': validation_loss})
            losses_validation.append(validation_loss)
            duration = time.time() - start_time
            num_digits_epochs = len(str(abs(self.epochs)))  # Number of digits of the epochs to format the output
            print(f"Epoch {i+1:{num_digits_epochs}}/{self.epochs} "
                  f"({duration:.2f}s): "
                  f"validation loss = {validation_loss.item():12.10f}, "
                  f"train_loss = {elbo_loss_traj[-1].item():12.10f}, "
                  f"mse = {mse_traj[-1].item():12.10f}, "
                  f"kl = {kl_traj[-1].item():12.10f}")

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

        return self.loss_validation

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

            loss += nn.MSELoss()(decoded, traj.to_vector()).detach().numpy()
        return loss / len(trajectories)  # Average loss

    def save_models(self, save_path: str = None):
        """Save the encoder and decoder models to the given path. If no path is given, the default save_path is used."""
        use_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(use_path):
            os.makedirs(use_path)
        self.decoder.save_model(use_path)
        self.decoder.save_decoder(use_path)
        self.encoder.save_model(use_path)
        self.encoder.save_encoder(use_path)

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

        self.plot_values(values=[self.loss_validation], path=save_path, file_name='validation_loss.png', plot_title='validation loss')
        self.plot_values(values=[self.kl_train_loss], path=save_path, file_name='kl_loss.png', plot_title="kl loss")
        self.plot_values(values=[self.mse_train_loss], path=save_path, file_name='ms_loss.png', plot_title="ms loss")
        self.plot_values(values=[self.train_loss], path=save_path, file_name='train_loss.png', plot_title="Traing Loss")

        # Plot all the losses in one plot, for better comparison and overview
        plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Losses')
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        axs[0, 0].plot(self.loss_validation)
        axs[0, 0].set_title('Validation Loss')
        axs[0, 1].plot(self.kl_train_loss)
        axs[0, 1].set_title('KL Loss')
        axs[1, 0].plot(self.mse_train_loss)
        axs[1, 0].set_title('MSE Loss')
        axs[1, 1].plot(self.train_loss)
        axs[1, 1].set_title('Training Loss')
        plt.savefig(os.path.join(save_path, 'all_losses.png'))

    def plot_values(self,
                    values: List[List],
                    file_name: str,
                    plot_title: str = "Plot",
                    path: str = None,):
        """
        Plot the given values and save the plot to the given path. If no path is given, the default save_path is used.

        values: List[List]: The values to plot. Each list in the list is a line in the plot. (Cannot be None or empty)
        """
        if values is None or len(values) == 0:
            raise ValueError(f"Cannot plot '{plot_title}' at '{path}' without values. Please provide  at least one value list.")

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
