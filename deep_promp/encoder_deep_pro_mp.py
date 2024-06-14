from typing import List, Tuple, Union

import numpy as np
import torch.nn as nn

from interfaces.latent_encoder import LatentEncoder
from trajectory.trajectory import Trajectory


class EncoderDeepProMP(LatentEncoder):

    def __init__(self,
                 neurons: List[int],
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()

        if not neurons or len(neurons) < 2:
            raise ValueError("The number of neurons must be at least 2. Got '%s'" % neurons)
        if not all(isinstance(neuron, int) for neuron in neurons):
            raise ValueError("All elements of neurons must be of type int. Got '%s'" % neurons)
        if not all(neuron > 0 for neuron in neurons):
            raise ValueError("All elements of neurons must be greater than 0. Got '%s'" % neurons)
        if not all(neurons[i] >= neurons[i + 1] for i in range(len(neurons) - 1)):
            raise ValueError("The number of neurons must be decrease monotonically. Got '%s'" % neurons)

        self.neurons = neurons
        self.activation_function = activation_function

        linear_layer = nn.Linear
        layers = []

        # create the network
        for i in range(len(self.neurons) - 1):
            layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), self.activation_function()]

        self.net = nn.Sequential(*layers).float()

    def encode_to_latent_variable(
            self,
            trajectory: Trajectory
    ) -> np.array:
        traj_points = trajectory.get_points()

        # This is the complete procedure of encoding a trajectory to a latent variable z
        # (including bayesian aggregation).

        # 1. For each traj_point tuple (t_i, x_i) in traj_points,
        # pass it through the encoder network and get the vectors mu_i and sigma_i.

        points_mu_sigma = []

        for t, x in traj_points:
            # mu_i and sigma_i are vectors with the same dimension as the TrajectoryState that is being used.
            output = self.net(x.to_vector())
            # Check if the output shape is 2 * trajectory_state_dimensions
            if not output.shape[0] == 2 * x.get_dimensions():
                raise ValueError(
                    "The output shape of the encoder network should have a mu and sigma for each dimension of the "
                    "TrajectoryState.")

            mu_point, sigma_point = output[:x.get_dimensions()], output[x.get_dimensions():]
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
        ret: str = "EncoderDeepProMP(neurons=%s)" % self.neurons
        ret += "\n" + str(self.net)
        return ret

    def __call__(self, x):
        return self.net(x)
