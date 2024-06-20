from typing import List, Tuple, Union, Type

import torch
import torch.nn as nn
from torch import Tensor

from interfaces.latent_encoder import LatentEncoder
from trajectory.state.joint_configuration import JointConfiguration
from trajectory.trajectory import Trajectory, T


class EncoderDeepProMP(LatentEncoder):

    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[JointConfiguration] = JointConfiguration,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()
        print("EncoderDeepProMP init")
        self.input_dimension = trajectory_state_class.get_dimensions()
        self.activation_function = activation_function
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

        # create the network
        linear_layer = nn.Linear
        layers = []
        for i in range(len(self.neurons) - 1):
            if i == len(self.neurons) - 2:
                layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), nn.Sigmoid()]
            else:
                layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), self.activation_function()]
        self.net = nn.Sequential(*layers).float()

    def encode_to_latent_variable(
            self,
            trajectory: Trajectory
    ) -> tuple[Tensor, Tensor]:
        traj_points: List[T] = trajectory.get_points()

        # This is the complete procedure of encoding a trajectory to a latent variable z
        # (including bayesian aggregation).

        # 1. For each traj_point tuple (t_i, x_i) in traj_points,
        # pass it through the encoder network and get the vectors mu_i and sigma_i.

        mu_points = torch.empty((len(traj_points), self.latent_variable_dimension))
        sigma_points = torch.empty((len(traj_points), self.latent_variable_dimension))

        for i, x in enumerate(traj_points):
            x: T = x
            x_tensor = torch.from_numpy(x.to_vector())

            if x_tensor.shape[0] != self.input_dimension:
                raise ValueError(
                    "The input shape of the encoder network should have the same dimension as the TrajectoryState.")
            output = self.net(x_tensor)

            if not output.shape[0] == 2 * self.latent_variable_dimension:
                raise ValueError("The output shape of the encoder network should have a mu and sigma"
                                 " for each dimension of the latent variable.")

            mu_point = output[:self.latent_variable_dimension]
            sigma_point = output[self.latent_variable_dimension:]

            mu_points[i] = mu_point.detach()
            sigma_points[i] = sigma_point.detach()

        # Calculate the vectors mu_z and sigma_z using the formulas on top right of page 3.
        # Assuming self.bayesian_aggregation is modified to accept torch.Tensor inputs
        mu_z, sigma_z_sq = self.bayesian_aggregation(mu_points, sigma_points)

        return mu_z, sigma_z_sq

    def sample_latent_variable(self, mu: torch.Tensor, sigma: torch.Tensor, percentage_of_standard_deviation=None) -> torch.Tensor:
        # This is the complete procedure of sampling a latent variable z from a normal distribution
        # specified by mu and sigma.

        # 1. Calculate the standard deviation of the normal distribution.
        #if percentage_of_standard_deviation is not None:
        #    sigma = sigma * percentage_of_standard_deviation

        # 2. Sample each element of z from a normal distribution specified by mu and sigma.
        z_sampled = torch.normal(mu, sigma)
        return z_sampled

    def bayesian_aggregation(self, mu_points: torch.Tensor, sigma_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __str__(self):
        ret: str = 'EncoderDeepProMP {'
        ret += '\n\t' + f'neurons: {self.neurons}'
        ret += '\n\t' + f'input_dimension: {self.input_dimension}'
        ret += '\n\t' + f'latent_variable_dimension: {self.latent_variable_dimension}'
        ret += '\n\t' + f'hidden_neurons: {self.hidden_neurons}'
        ret += '\n\t' + f'activation_function: {self.activation_function}'
        ret += '\n\t' + f'trajectory_state_class: {self.trajectory_state_class}'
        ret += '\n\t' + f'net: {str(self.net)}'
        ret += '\n' + '}'
        return ret

    def __repr__(self):
        return f'EncoderDeepProMP(neurons={self.neurons})'

    def __call__(self, *args, **kwargs):
        return self.encode_to_latent_variable(*args, **kwargs)