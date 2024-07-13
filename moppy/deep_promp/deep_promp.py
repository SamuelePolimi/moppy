from typing import List

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

from matplotlib import pyplot as plt

from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.interfaces.movement_primitive import MovementPrimitive
from moppy.trajectory.trajectory import Trajectory


def plot_values(values: List[List], path: str, file_name: str, title: str):
    file_path = os.path.join(path, file_name)
    plt.close()
    for i in values:
        plt.plot(i)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

    # Save the plot as an image file
    plt.savefig(file_path)


def gauss_kl(mu_q, std_q):
    """Calculate the Kullback-Leibler (KL) divergence between a Gaussian distribution and a standard Gaussian distribution."""

    return torch.mean(-torch.log(std_q) + (std_q ** 2 + mu_q ** 2) / 2 - 0.5)


def calculate_elbo(y_pred, y_star, mu, sigma, beta=1.0):
    """Calculate the Evidence Lower Bound (ELBO) using the reconstruction loss and the KL divergence.
    The ELBO is the loss function used to train the DeepProMP."""

    # Reconstruction loss (assuming Mean Squared Error)
    mse = nn.MSELoss()(y_pred, y_star)
    # log_prob = torch.distributions.Normal(loc=mu, scale=sigma).log_prob(torch.tensor(y_star, requires_grad=True)).sum()
    # losses.append(log_prob)
    # KL divergence between approximate posterior (q) and prior (p)
    kl = gauss_kl(mu_q=mu, std_q=sigma)

    # print("mse %s, kl %s" % (mse, kl))
    # Combine terms with beta weighting
    elbo = mse + kl * beta
    return elbo, mse, kl


class DeepProMP(MovementPrimitive):
    """A DeepProMP is a probabilistic movement primitive that uses deep neural networks to encode and decode trajectories."""

    def __init__(self,
                 name: str,
                 encoder: EncoderDeepProMP,
                 decoder: DecoderDeepProMP,
                 save_path: str = './deep_promp/output/',
                 learning_rate: float = 0.005,
                 epochs: int = 100,
                 beta: float = 0.01):
        super().__init__(name, encoder, decoder)

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

    @staticmethod
    def kl_annealing_scheduler(current_epoch, n_cycles=4, max_epoch=1000, saturation_point=0.5):
        """KL annealing scheduler"""
        tau = ((current_epoch - 1) % (math.ceil(max_epoch / n_cycles))) / (math.ceil(max_epoch / n_cycles))
        return tau/saturation_point if tau < saturation_point else 1

    def train(self, trajectories: List[Trajectory], kl_annealing=True) -> None:
        """Train the DeepProMP using the given trajectories. The training is done using the Evidence Lower Bound (ELBO).
        The ELBO is the loss function used to train the DeepProMP. The training is done using the Adam optimizer."""
        # Optimizers
        training_set = trajectories[:(len(trajectories) * 9) // 10]
        validation_set = trajectories[-len(trajectories) // 10:]
        print("Total set: ", len(trajectories))
        print(f"Training set: {len(training_set)}")
        print(f"Validation set: {len(validation_set)}")

        optimizer = optim.Adam(params=list(self.encoder.net.parameters()) + list(self.decoder.net.parameters()),
                               lr=self.learning_rate)
        losses_traj = []
        kl_traj = []
        mse_traj = []
        losses_validation = []
        elbo_loss_traj = []
        epochs = self.epochs

        for i in range(epochs):
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
                beta = DeepProMP.kl_annealing_scheduler(i+1, n_cycles=4, max_epoch=epochs, saturation_point=0.5)
                loss, mse, kl = calculate_elbo(decoded.reshape(-1, 1), data.to_vector().reshape(-1, 1), mu, sigma, beta)
                # print(f"{i + 1}/{episodes} - {tr_i + 1}/{len(trajectories)} = {loss.item()}")
                loss.backward()
                optimizer.step()
                mse_tot += mse.detach().numpy()
                kl_tot += kl.detach().numpy()
                loss_tot += loss.detach().numpy()
            losses_traj.append(mse_tot / len(training_set))
            kl_traj.append(kl_tot / len(training_set))
            mse_traj.append(mse_tot / len(training_set))
            elbo_loss_traj.append(loss_tot / len(training_set))
            # validation
            validation_loss = self.validate(validation_set)
            losses_validation.append(validation_loss)
            duration = time.time() - start_time
            print(f"Epoch {i+1:3}/{epochs} ({duration}s): validation loss = {validation_loss.item()}, train_loss = "
                f"{losses_traj[-1].item()}"
                f", mse = {mse_traj[-1].item()},"
                f" kl = {kl_traj[-1].item()}")

        file_path = os.path.join(self.save_path, 'validation_loss.pth')
        torch.save(losses_validation, file_path)

        file_path = os.path.join(self.save_path, 'train_loss.pth')
        torch.save(losses_traj, file_path)

        file_path = os.path.join(self.save_path, 'mse.pth')
        torch.save(mse_traj, file_path)

        file_path = os.path.join(self.save_path, 'kl.pth')
        torch.save(kl_traj, file_path)

        file_path = os.path.join(self.save_path, 'elbo_loss.pth')
        torch.save(elbo_loss_traj, file_path)

        print("Training finished")
        print("Plotting...", end='', flush=True)

        values_losses_traj = [t.item() for t in losses_traj]
        values_losses_validation = [t.item() for t in losses_validation]
        values_mse_traj = [t.item() for t in mse_traj]
        values_kl_traj = [t.item() for t in kl_traj]
        values_elbo_loss_traj = [t.item() for t in elbo_loss_traj]

        plot_values(values=[values_losses_traj], path=self.save_path, file_name='train_loss.png', title="train_loss")
        plot_values(values=[values_losses_validation], path=self.save_path, file_name='validation_loss.png', title='validation_loss')
        plot_values(values=[values_kl_traj], path=self.save_path, file_name='kl.png', title="kl")
        plot_values(values=[values_mse_traj], path=self.save_path, file_name='msloss.png', title="msloss")
        plot_values(values=[values_elbo_loss_traj], path=self.save_path, file_name='elbo_loss.png', title="elbo_loss")

        print("finished")
        print("Saving models...", end='', flush=True)
        self.decoder.save_model(self.save_path)
        self.encoder.save_model(self.save_path)
        print("finished")

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

    def __str__(self):
        return f"DeepProMP({self.name})" '{' + \
            '\n\t' + f"learning_rate: {self.learning_rate}" + \
            '\n\t' + f"epochs: {self.epochs}" + \
            '\n\t' + f"beta: {self.beta}" + \
            '\n\t' + f"save_path: {self.save_path}" + \
            '\n' + f"encoder: {self.encoder}" + \
            '\n' + f"decoder: {self.decoder}" + \
            '\n' + '}'
