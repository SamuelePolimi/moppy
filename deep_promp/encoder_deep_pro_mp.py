from typing import List, Tuple, Union, Type

import numpy as np
import torch
import torch.nn as nn
from interfaces.latent_encoder import LatentEncoder
from trajectory.joint_configuration_trajectory_state import JointConfigurationTrajectoryState
from trajectory.trajectory import Trajectory, T
from trajectory.trajectory_state import TrajectoryState


class EncoderDeepProMP(LatentEncoder):

    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[JointConfigurationTrajectoryState] = JointConfigurationTrajectoryState,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()

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
            layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), self.activation_function()]
        self.net = nn.Sequential(*layers).float()

    def encode_to_latent_variable(
            self,
            trajectory: Trajectory
    ) -> np.array:
        traj_points: List[T] = trajectory.get_points()

        # This is the complete procedure of encoding a trajectory to a latent variable z
        # (including bayesian aggregation).

        # 1. For each traj_point tuple (t_i, x_i) in traj_points,
        # pass it through the encoder network and get the vectors mu_i and sigma_i.

        points_mu_sigma = []

        for x in traj_points:
            x: T = x
            # mu_i and sigma_i are vectors with the same dimension as the TrajectoryState that is being used.
            x_tensor = torch.from_numpy(x.to_vector())
            if x_tensor.shape[0] != self.input_dimension:
                raise ValueError(
                    "The input shape of the encoder network should have the same dimension as the TrajectoryState.")
            output = self.net(x_tensor)
            # Check if the output shape is 2 * trajectory_state_dimensions
            if not output.shape[0] == 2 * self.latent_variable_dimension:
                raise ValueError(
                    "The output shape of the encoder network should have a mu and sigma for each dimension of the "
                    "TrajectoryState.")

            mu_point = output[:self.latent_variable_dimension]
            sigma_point = output[self.latent_variable_dimension:]
            points_mu_sigma.append((mu_point, sigma_point))

        # TODO get hyperparamenter for dimension of latent variable (unsure if needed)?

        # 2. Calculate the vectors mu_z and sigma_z using the formulas on top right of page 3.
        mu_z, sigma_z_sq = self.bayesian_aggregation(points_mu_sigma)

        # 3. Sample the latent variable z vector.
        # This is done by sampling each element of z from a normal distribution specified by mu_z and sigma_z.
        z_sampled = np.random.normal(mu_z, sigma_z_sq)  # TODO use sqrt of sigma_z_sq or not?

        return z_sampled

    def bayesian_aggregation(self, mu_sigma_points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        mu_as, sigma_as = zip(*mu_sigma_points)

        # TODO verify if the sum is calculated correctly, not sure about axis
        sum_mu_a_over_sigma_a_squared = np.sum(mu_as / (sigma_as ** 2), axis=0)
        sum_sigma_a_inverse = np.sum(1 / (sigma_as ** 2), axis=0)

        # Calculate sigma_z^2(A) without context variables
        sigma_z_sq = 1 / (1 + sum_sigma_a_inverse)

        # Adjust for context variables if needed
        # This part is omitted since the assumption is about having only via-points (A)

        # Calculate mu_z(A) using the formula without context variables
        mu_z = sigma_z_sq * sum_mu_a_over_sigma_a_squared

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