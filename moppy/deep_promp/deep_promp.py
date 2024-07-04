from typing import List, Type, Union

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from matplotlib import pyplot as plt

from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.interfaces.movement_primitive import MovementPrimitive
from moppy.trajectory.state.joint_configuration import JointConfiguration
from moppy.trajectory.trajectory import Trajectory


losses = []


def gauss_kl(mu_q, std_q, mu_p=None, std_p=None, scale=1.0):
    mu_p = torch.zeros_like(mu_q) if mu_p is None else mu_p
    std_p = torch.ones_like(std_q) * scale if std_p is None else std_p

    # independent treats them as multivariate Normals
    indep = distributions.Independent
    q_dist = indep(distributions.Normal(mu_q, std_q), 1)
    p_dist = indep(distributions.Normal(mu_p, std_p), 1)

    return distributions.kl_divergence(q_dist, p_dist)


def calculate_elbo(y_pred, y_star, mu, sigma, beta=0.5):
    """Calculate the Evidence Lower Bound (ELBO) using the reconstruction loss and the KL divergence.
    The ELBO is the loss function used to train the DeepProMP."""

    # Reconstruction loss (assuming Mean Squared Error)
    log_prob = nn.MSELoss()(y_pred, y_star)
    # losses.append(log_prob)
    # KL divergence between approximate posterior (q) and prior (p)
    kl = gauss_kl(mu_q=mu, std_q=sigma, scale=1.)

    # Combine terms with beta weighting
    elbo = log_prob + kl * beta
    return elbo


class DeepProMP(MovementPrimitive):
    """A DeepProMP is a probabilistic movement primitive that uses deep neural networks to encode and decode trajectories."""

    def __init__(self, name: str, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP, save_path: str = './deep_promp/output/'):
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

        self.decoder = decoder
        self.encoder = encoder
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.latent_variable_dimension = encoder.latent_variable_dimension

    @classmethod
    def from_latent_variable_dimension(
            cls,
            name: str,
            latent_variable_dimension: int,
            hidden_neurons_encoder: List[int],
            hidden_neurons_decoder: List[int],
            trajectory_state_class: Type[JointConfiguration] = JointConfiguration,
            activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU
    ):
        """Create a DeepProMP from the latent variable dimension and the hidden neurons of the encoder and decoder.
        decoder and encoder are instances of DecoderDeepProMP and EncoderDeepProMP."""

        encoder = EncoderDeepProMP(
            latent_variable_dimension=latent_variable_dimension,
            hidden_neurons=hidden_neurons_encoder,
            trajectory_state_class=trajectory_state_class,
            activation_function=activation_function,)

        decoder = DecoderDeepProMP(
            latent_variable_dimension=latent_variable_dimension,
            hidden_neurons=hidden_neurons_decoder,
            trajectory_state_class=trajectory_state_class,
            activation_function=activation_function,)
        return cls(name=name, encoder=encoder, decoder=decoder)

    def train(self, trajectories: List[Trajectory]) -> None:
        """Train the DeepProMP using the given trajectories. The training is done using the Evidence Lower Bound (ELBO).
        The ELBO is the loss function used to train the DeepProMP. The training is done using the Adam optimizer."""
        # Optimizers
        training_set = trajectories[:(len(trajectories) * 9) // 10]
        validation_set = trajectories[-len(trajectories) // 10:]
        print("Total set: ", len(trajectories))
        print(f"Training set: {len(training_set)}")
        print(f"Validation set: {len(validation_set)}")

        optimizer = optim.Adam(list(self.encoder.net.parameters()) + list(self.decoder.net.parameters()), lr=0.001)
        losses_traj = []
        losses_validation = []
        episodes = 100
        for i in range(episodes):
            for tr_i, data in enumerate(training_set):
                optimizer.zero_grad()  # Zero the gradients of the optimizer to avoid accumulation
                mu, sigma = self.encoder(data)
                latent_var_z = self.encoder.sample_latent_variable(mu, sigma)

                decoded = []
                for j in data.get_points():
                    decoded.append(self.decoder(latent_var_z, j.get_time()))
                decoded = torch.cat(decoded)

                loss = calculate_elbo(decoded, data.to_vector(), mu, sigma)
                print(f"{i + 1}/{episodes} - {tr_i + 1}/{len(trajectories)} = {loss.item()}")
                loss.backward()
                losses_traj.append(loss.detach().numpy())
                optimizer.step()
            # validation
            validation_loss = self.validate(validation_set)
            losses_validation.append(validation_loss)
            print(f"Episode {i+1} validation loss = " + validation_loss)

        print("Training finished")
        print("Plotting...", end='', flush=True)
        # Extract values from tensors
        values = [t.item() for t in losses_traj]
        # Plotting
        plt.plot(values)
        plt.title('Tensor Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot as an image file
        values = [t.item() for t in losses]
        # Plotting
        plt.plot(values)
        plt.title('Tensor Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig(self.save_path + 'msloss.png')

        plt.close()
        plt.plot(losses_validation)
        plt.title('Validation Loss')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig(self.save_path + 'validation_loss.png')

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
            '\n' + f"encoder: {self.encoder}" + \
            '\n' + f"decoder: {self.decoder}" + \
            '\n' + '}'
