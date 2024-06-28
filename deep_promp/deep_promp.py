from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from matplotlib import pyplot as plt
from torch.distributions import Normal

from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from interfaces.movement_primitive import MovementPrimitive
from trajectory.state.joint_configuration import JointConfiguration
from trajectory.trajectory import Trajectory


class DeepProMP(MovementPrimitive):
    """A DeepProMP is a probabilistic movement primitive that uses deep neural networks to encode and decode trajectories."""

    def __init__(self, name: str, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP):
        super().__init__(name, encoder, decoder)
        print("DeepProMP init")

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
        self.latent_variable_dimension = encoder.latent_variable_dimension
        self.prior = Normal(torch.zeros(self.latent_variable_dimension), torch.ones(self.latent_variable_dimension))

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

    def train(self, trajectories: List[Trajectory]):

        # Loss function
        criterion = nn.MSELoss()

        # Optimizers
        optimizer = optim.Adam(list(self.encoder.net.parameters()) + list(self.decoder.net.parameters()), lr=0.001)
        losses_traj = []
        for i in range(100):
            for data in trajectories:
                data: Trajectory = data
                mu, sigma = self.encoder(data)
                latent_var_z = torch.cat((mu, sigma), dim=0)
                losses = []
                for j in data.get_points():
                    decoded = self.decoder(latent_var_z, j.get_time())
                    # loss = criterion(decoded, j.to_vector())
                    loss = self.calculate_elbo(decoded, j.to_vector(), mu, sigma, beta=1.0)
                    losses.append(loss)
                loss = torch.mean(torch.stack(losses))
                print(loss)
                losses_traj.append(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Extract values from tensors
        values = [t.item() for t in losses_traj]

        # Plotting
        plt.plot(values, marker='o')
        plt.title('Tensor Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig('tensor_values_plot.png')

    # https://github.com/tonyduan/variational-autoencoders/blob/master/src/blocks.py
    def calculate_elbo(self, y_pred, y_star, mu, sigma, beta=1.0):
        # Reconstruction loss (assuming Mean Squared Error)
        log_prob = nn.MSELoss()(y_pred, y_star)

        # KL divergence between approximate posterior (q) and prior (p)
        kl = gauss_kl(mu_q=mu, std_q=sigma, scale=1.)

        # Combine terms with beta weighting
        elbo = log_prob + kl * beta
        return elbo

    def test(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()


def gauss_kl(mu_q, std_q, mu_p=None, std_p=None, scale=1.0):
    mu_p = torch.zeros_like(mu_q) if mu_p is None else mu_p
    std_p = torch.ones_like(std_q) * scale if std_p is None else std_p

    # independent treats them as multivariate Normals
    indep = distributions.Independent
    q_dist = indep(distributions.Normal(mu_q, std_q), 1)
    p_dist = indep(distributions.Normal(mu_p, std_p), 1)

    return distributions.kl_divergence(q_dist, p_dist)
