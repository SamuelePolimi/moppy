from typing import List, Type

import os
import torch
import torch.nn as nn
from torch import Tensor

from moppy.interfaces import LatentEncoder
from moppy.trajectory.state import JointConfiguration, TrajectoryState
from moppy.trajectory import Trajectory, T


class EncoderDeepProMP(LatentEncoder, nn.Module):

    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[TrajectoryState] = JointConfiguration,
                 activation_function: Type[nn.Module] = nn.ReLU,
                 activation_function_params: dict = {}):
        nn.Module.__init__(self)

        # Check if the trajectory state class is a subclass of TrajectoryState
        if trajectory_state_class not in TrajectoryState.__subclasses__():
            raise TypeError(f"The trajectory state class must be a subclass of '{TrajectoryState.__name__}'. "
                            f"Got '{trajectory_state_class}'"
                            f"\nThe usable subclasses are {TrajectoryState.__subclasses__()}")

        self.input_dimension = trajectory_state_class.get_dimensions()
        self.activation_function = activation_function
        self.activation_function_params = activation_function_params
        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension
        self.trajectory_state_class = trajectory_state_class

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [self.input_dimension] + hidden_neurons + [latent_variable_dimension * 2]

        # Check if the neurons list is empty or has fewer than 2 elements
        if not self.neurons or len(self.neurons) < 2:
            raise ValueError("The number of neurons must be at least 2. Got '%s'" % self.neurons)
        if not all(isinstance(neuron, int) for neuron in self.neurons):
            raise ValueError("All elements of neurons must be of type int. Got '%s'" % self.neurons)
        if not all(neuron > 0 for neuron in self.neurons):
            raise ValueError("All elements of neurons must be greater than 0. Got '%s'" % self.neurons)

        layers = self.create_layers()
        self.net = nn.Sequential(*layers).float()

        # Initialize the weights and biases of the network
        self.net.apply(self.__init_weights)

    @classmethod
    def load_from_save_file(cls, path: str = '', file: str = "encoder_deep_pro_mp.pth") -> 'EncoderDeepProMP':
        """Load a model from a file and return a EncoderDeepProMP instance."""
        file_path = os.path.join(path, file)
        # Load the model data
        model_data = torch.load(file_path)


        # Reconstruct the model using the saved configuration
        model = cls(
            latent_variable_dimension=model_data['latent_variable_dimension'],
            hidden_neurons=model_data['hidden_neurons'],
            trajectory_state_class=model_data['trajectory_state_class'],
            activation_function=model_data['activation_function'],
            activation_function_params=model_data['activation_function_params']
        )

        # Load the model weights
        model.net.load_state_dict(model_data['state_dict'])

        return model

    def create_layers(self):
        layers = []
        for i in range(len(self.neurons) - 2):
            layers += [nn.Linear(self.neurons[i], self.neurons[i + 1]),
                       self.activation_function(**self.activation_function_params)]
        layers += [nn.Linear(self.neurons[-2], self.neurons[-1])]
        return layers

    def __init_weights(self, m):
        """Initialize the weights and biases of the network using Xavier initialization and a bias"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                nn.init.constant_(m.bias, 0.05)

    def encode_to_latent_variable(
            self,
            trajectory: Trajectory
    ) -> tuple[Tensor, Tensor]:
        """
        Encodes a Trajectory into a mu and sigma (both have the same size of self.latent_variable_dimension).
        1. Each point of the trajectory gets run throw the NN and the resulting nu and sigma gets saved.
        2. bayesian_aggregation is then used on the mu's and sigma's and the resulting mu and sigma is returned.

        Args:
            trajectory (Trajectory): The Trajectory to encode. The used TrajectoryState class of the Points has to be the same as used in the Encoder.

        Returns:
            tuple[Tensor, Tensor]: The resulting mu and sigma. Both Tensor of size self.latent_variable_dimension.
        """

        # trajectory cannot be None or empty
        if trajectory is None or len(trajectory.get_points()) == 0:
            raise ValueError("Given Trajectory is ether None or does not contain any Points.")
        # used TrajectoryState class for trajectory has to be the same as used in the Encoder
        if not isinstance(trajectory.get_points()[0], self.trajectory_state_class):
            raise ValueError("Cannot encode Trajectory  with a different TrajectoryState Type then the Encoder:\n"
                             f"{type(trajectory.get_points()[0]).__name__}(trajectory) != {self.trajectory_state_class.__name__}(encoder)")

        traj_points: List[T] = trajectory.get_points()

        mu_points = torch.zeros((len(traj_points), self.latent_variable_dimension), dtype=torch.float64)
        sigma_points = torch.zeros((len(traj_points), self.latent_variable_dimension), dtype=torch.float64)

        for i, x in enumerate(traj_points):
            x: T = x
            x_tensor = x.to_vector_time()

            if x_tensor.shape[0] != self.input_dimension:
                raise ValueError("The input shape of the encoder network is incorrect. Got %s, expected %s" % (
                    x_tensor.shape[0], self.input_dimension))
            output = self.net(x_tensor)

            if not output.shape[0] == 2 * self.latent_variable_dimension:
                raise ValueError("The output shape of the encoder network should have a mu and sigma"
                                 " for each dimension of the latent variable.")

            mu_point = output[:self.latent_variable_dimension]
            sigma_point = output[self.latent_variable_dimension:]

            # TODO there was detach() here, but I removed it. Check if it is necessary
            mu_points[i] = mu_point
            sigma_points[i] = sigma_point

        # Calculate the vectors mu_z and sigma_z using the formulas on top right of page 3.
        # Assuming self.bayesian_aggregation is modified to accept torch.Tensor inputs
        mu_z, sigma_z_sq = self.bayesian_aggregation(mu_points, sigma_points)

        return mu_z, sigma_z_sq

    def sample_latent_variable(self,
                               mu: torch.Tensor,
                               sigma: torch.Tensor,
                               percentage_of_standard_deviation=None) -> torch.Tensor:
        # This is the complete procedure of sampling a latent variable z from a normal distribution
        # specified by mu and sigma.

        # 1. Calculate the standard deviation of the normal distribution.
        # if percentage_of_standard_deviation is not None:
        #    sigma = sigma * percentage_of_standard_deviation

        # 2. Sample each element of z from a normal distribution specified by mu and sigma.
        # TODO: matyas: implement the "percentage_of_standard_deviation" feature
        z_sampled = torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
        z_sampled = z_sampled * sigma + mu
        return z_sampled

    def sample_latent_variables(self,
                                mu: torch.Tensor,
                                sigma: torch.Tensor,
                                size: int = 1) -> torch.Tensor:
        """Sample n=size latent variables from the given mu and sigma tensors."""
        dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        samples = dist.sample((size,))
        z_sampled = samples * sigma + mu
        return z_sampled

    def bayesian_aggregation(self,
                             mu_points: torch.Tensor,
                             sigma_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = 10e-10
        mu_points += epsilon
        sigma_points += epsilon
        # TODO verify if the sum is calculated correctly, dim = 0 should be correct
        sum_mu_a_over_sigma_a_squared = torch.sum(mu_points / (sigma_points ** 2), dim=0)
        sum_sigma_a_inverse = torch.sum(1 / (sigma_points ** 2), dim=0)

        # Calculate sigma_z^2(A) without context variables
        sigma_z_sq = 1 / (1 + sum_sigma_a_inverse)

        # Adjust for context variables if needed
        # This part is omitted since the assumption is about having only via-points (A)

        # Calculate mu_z(A) using the formula without context variables
        mu_z = sigma_z_sq * sum_mu_a_over_sigma_a_squared

        if mu_z.shape != (self.latent_variable_dimension,):
            raise ValueError("The shape of mu_z should be equal to the latent variable dimension.")
        if sigma_z_sq.shape != mu_z.shape:
            raise ValueError("The shape of sigma_z_sq should be equal to the latent variable dimension.")

        return mu_z, sigma_z_sq

    def save_encoder(self, path: str = '', filename: str = "encoder_deep_pro_mp.pth"):
        """Save the encoder to a file, including the state_dict of the network and the configuration of the model.
        The configuration includes the latent_variable_dimension, hidden_neurons, trajectory_state_class, activation function,
        and activation function parameters.
        Can be loaded using the load_from_save_file method."""

        file_path = os.path.join(path, filename)
        model_data = {
            'state_dict': self.net.state_dict(),
            'latent_variable_dimension': self.latent_variable_dimension,
            'hidden_neurons': self.hidden_neurons,
            'trajectory_state_class': self.trajectory_state_class,
            'activation_function': self.activation_function,
            'activation_function_params': self.activation_function_params
        }
        torch.save(model_data, file_path)

    def save_model(self, path: str = '', filename: str = "encoder_model_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, path: str = '', filename: str = "encoder_model_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        self.net.load_state_dict(torch.load(file_path))

    def forward(self, trajectory: Trajectory) -> tuple[Tensor, Tensor]:
        return self.encode_to_latent_variable(trajectory)

    def __str__(self):
        ret: str = 'EncoderDeepProMP {'
        ret += '\n\t' + f'neurons: {self.neurons}'
        ret += '\n\t' + f'input_dimension: {self.input_dimension}'
        ret += '\n\t' + f'hidden_neurons: {self.hidden_neurons}'
        ret += '\n\t' + f'latent_variable_dimension: {self.latent_variable_dimension}'
        ret += '\n\t' + f'activation_function: {self.activation_function}'
        ret += '\n\t' + f'trajectory_state_class: {self.trajectory_state_class}'
        ret += '\n\t' + f'net: {str(self.net)}'
        ret += '\n' + '}'
        return ret

    def __repr__(self):
        return f'EncoderDeepProMP(neurons={self.neurons})'
