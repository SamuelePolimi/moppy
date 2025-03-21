from typing import List, Tuple

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

from matplotlib import pyplot as plt

from .decoder_deep_pro_mp import DecoderDeepProMP
from .deep_promp import DeepProMP, EncoderDeepProMP

from .encoder_as_actor import EncoderAsActor, RobosuiteDemoStartingPosition
from moppy.trajectory import Trajectory
from moppy.logger import Logger


class TrainEncoderAsActor(Logger):

    def __init__(self,
                 name: str,
                 encoder: EncoderAsActor,
                 decoder: DecoderDeepProMP,
                 save_path: str = './deep_promp/output/',
                 log_to_tensorboard: bool = False,
                 learning_rate: float = 0.005,
                 epochs: int = 100) -> None:
        Logger.__init__(self, log_dir=save_path, logging_enabled=log_to_tensorboard)

        # Check if the encoder and decoder are instances/subclasses of EncoderDeepProMP and DecoderDeepProMP
        if not issubclass(type(encoder), EncoderAsActor):
            raise TypeError(f"The encoder must be an instance of '{EncoderAsActor.__name__}' or a subclass. Got '{type(encoder)}'."
                            f"\nThe usable classes are {[EncoderAsActor] + EncoderAsActor.__subclasses__()}")
        if not issubclass(type(decoder), DecoderDeepProMP):
            raise TypeError(f"The decoder must be an instance of '{DecoderDeepProMP.__name__}' or a subclass. Got '{type(decoder)}'."
                            f"\nThe usable classes are {[DecoderDeepProMP] + DecoderDeepProMP.__subclasses__()}")
        self.encoder_for_bayesian = EncoderDeepProMP(10, [10])
        # Check if the encoder and decoder are compatible
        if encoder.latent_variable_dimension != decoder.latent_variable_dimension:
            raise ValueError("The encoder and decoder must have the same latent variable dimension. "
                             f"Got encoder latent variable dimension '{encoder.latent_variable_dimension}'"
                             f"and decoder latent variable dimension '{decoder.latent_variable_dimension}'")
        self.latent_variable_dimension = encoder.latent_variable_dimension

        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize the losses lists
        self.mse_train_loss = []  # Mean Squared Error
        self.loss_validation = []  # Validation loss

    def train(self,
              trajectories: List[Tuple[RobosuiteDemoStartingPosition, Trajectory]],
              learning_rate: float = None,
              epochs: int = None) -> None:

        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs

        training_start_time = time.time()
        print("Start DeepProMP training ...")

        training_set = trajectories[:(len(trajectories) * 9) // 10]
        validation_set = trajectories[-len(trajectories) // 10:]
        print("Total set: ", len(trajectories))
        print(f"Training set: {len(training_set)}")
        print(f"Validation set: {len(validation_set)}")

        optimizer = optim.Adam(params=self.encoder.net.parameters(), lr=self.learning_rate)
        mse_traj = []
        losses_validation = []
        validation_loss = self.validate(validation_set)
        self.log_metrics(-1, {'validation loss': validation_loss})
        losses_validation.append(validation_loss)
        for i in range(self.epochs):
            start_time = time.time()
            mse_tot = 0
            for tr_i, data in enumerate(training_set):
                optimizer.zero_grad()  # Zero the gradients of the optimizer to avoid accumulation
                obs, z = data
                z = torch.tensor(z, dtype=torch.float32)
                latent_var_z = self.encoder(obs)
                mu = latent_var_z[:self.latent_variable_dimension].float()
                sigma = latent_var_z[self.latent_variable_dimension:].float()

                sampled_latent_var_z = EncoderAsActor.sample_latent_variable(mu=mu, sigma=sigma)
                mu_points = torch.zeros((1, self.latent_variable_dimension), dtype=torch.float64)
                sigma_points = torch.zeros((1, self.latent_variable_dimension), dtype=torch.float64)
                mu_points[0] = mu
                sigma_points[0] = sigma
                mu, sigma = self.encoder_for_bayesian.bayesian_aggregation(mu_points=mu_points, sigma_points=sigma_points)
                loss, mse, kl = DeepProMP.calculate_elbo(y_pred=sampled_latent_var_z, y_star=z, mu=mu, sigma=sigma, beta=0.01)
                loss.backward()
                optimizer.step()
                mse_tot += loss.detach().numpy()

            self.log_metrics(i, {'mse loss': mse_tot / len(training_set)})

            mse_traj.append(mse_tot / len(training_set))
            # validation
            validation_loss = self.validate(validation_set)
            self.log_metrics(i, {'validation loss': validation_loss})
            losses_validation.append(validation_loss)
            duration = time.time() - start_time
            num_digits_epochs = len(str(abs(self.epochs)))  # Number of digits of the epochs to format the output
            print(f"Epoch {i+1:{num_digits_epochs}}/{self.epochs} "
                  f"({duration:.2f}s): "
                  f"validation loss = {validation_loss.item():12.10f}, "
                  f"mse = {mse_traj[-1].item():12.10f}")

        print(f"Training finished (Time = {(time.time() - training_start_time):.2f}s).")

        self.loss_validation = losses_validation
        self.mse_train_loss = mse_traj

        print("Saving losses ...", end='', flush=True)
        self.save_losses()
        print("finished.\nPlotting...", end='', flush=True)
        self.save_losses_plots()
        print("finished.\nSaving models...", end='', flush=True)
        self.save_models()
        print("finished.")

    def test(self):
        raise NotImplementedError()

    def validate(self, trajectories: List[Tuple[RobosuiteDemoStartingPosition, Trajectory]]):
        mse_tot = 0
        for data in trajectories:
            obs, z = data
            latent_var_z = self.encoder(obs).double()
            mu = latent_var_z[:self.latent_variable_dimension]
            sigma = latent_var_z[self.latent_variable_dimension:]
            mu_points = torch.zeros((1, self.latent_variable_dimension), dtype=torch.float64)
            sigma_points = torch.zeros((1, self.latent_variable_dimension), dtype=torch.float64)
            mu_points[0] = mu
            sigma_points[0] = sigma
            mu, sigma = self.encoder_for_bayesian.bayesian_aggregation(mu_points=mu_points, sigma_points=sigma_points)
            sampled_latent_var_z = EncoderAsActor.sample_latent_variable(mu, sigma)
            loss, mse, kl = DeepProMP.calculate_elbo(y_pred=sampled_latent_var_z, y_star=z, mu=mu, sigma=sigma, beta=0.00)
            mse_tot += loss.detach().numpy()
        return mse_tot / len(trajectories)  # Average loss

    def save_models(self, save_path: str = None):
        """Save the encoder and decoder models to the given path. If no path is given, the default save_path is used."""
        use_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(use_path):
            os.makedirs(use_path)
        self.encoder.save_model(use_path)
        self.encoder.save_encoder(use_path)

    def save_losses(self, save_path: str = None):
        """Save the losses to the given path. If no path is given, the default save_path is used."""
        save_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        values_to_plot_with_filenames = [(self.mse_train_loss, 'mse_loss.pth'),
                                         (self.loss_validation, 'validation_loss.pth')]

        for values, file_name in values_to_plot_with_filenames:
            file_path = os.path.join(save_path, file_name)
            torch.save(values, file_path)

    def save_losses_plots(self, save_path: str = None):
        """Save the plots of the losses to the given path. If no path is given, the default save_path is used."""
        save_path = save_path if save_path is not None else self.save_path  # Use the given path or the default one
        self.plot_values(values=[self.mse_train_loss], path=save_path, file_name='ms_loss.png', plot_title="ms loss")
        self.plot_values(values=[self.loss_validation], path=save_path, file_name='loss_validation.png', plot_title="loss validation")

        # Plot the validation loss and mse loss in one plot, for better comparison and overview
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle('Losses')
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        axs[0].plot(self.loss_validation)
        axs[0].set_title('Validation Loss')
        axs[1].plot(self.mse_train_loss)
        axs[1].set_title('MSE Loss')
        plt.savefig(os.path.join(self.save_path, 'validation_and_mse_losses.png'))

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
            '\n\t' + f"save_path: {self.save_path}" + \
            '\n' + f"encoder: {self.encoder}" + \
            '\n' + f"decoder: {self.decoder}" + \
            '\n' + '}'
