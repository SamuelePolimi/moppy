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
        # pass it through the encoder network and get the vectors mu_i and rho_i.

        points_mu_rho = []

        for t, x in traj_points:
            # mu_i and rho_i are vectors with the same dimension as the TrajectoryState that is being used.
            output = self.net(x.to_vector())
            # Check if the output shape is 2 * trajectory_state_dimensions
            if not output.shape[0] == 2 * x.get_dimensions():
                raise ValueError(
                    "The output shape of the encoder network should have a mu and rho for each dimension of the "
                    "TrajectoryState.")

            mu_point, rho_point = output[:x.get_dimensions()], output[x.get_dimensions():]
            points_mu_rho.append((mu_point, rho_point))

        # TODO get hyperparamenter for dimension of latent variable

        # 2. Calculate the vectors mu_z and rho_z using the formulas on top right of page 3.
        mu_z, rho_z = self.bayesian_aggregation(points_mu_rho)

        # 3. Sample the latent variable z vector.
        # This is done by sampling each element of z from a normal distribution specified by mu_z and rho_z.
        z_sampled = np.random.normal(mu_z, rho_z)

        return z_sampled

    def bayesian_aggregation(self, mu_rho_points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Formula with both via-points(A) and context variables(C)
        \mu_z(A,C)=\sigma_z^2(A,C)\left(\sum_{i=1}^{n}\frac{\mu_a(a_i)}{\sigma_a^2(a_i)}+\sum_{i=1}^{m}\frac{\mu_c(c_i)}{\sigma_c^2(c_i)}\right)
        \sigma_z^2(A,C)=\left(1+\sum_{i=1}^{n}\sigma_a^2(a_i)^{-1}+\sum_{i=1}^{m}\sigma_c^2(c_i)^{-1}\right)^{-1}
        
        Assumtion: We only have via-points(A) and no context variables(C) | mu_i == \mu_a(a_i) and rho_i == \sigma_a^2(a_i)
        Formula without context variables(C):
        \mu_z(A)=\sigma_z^2(A)\left(\sum_{i=1}^{n}\frac{\mu_a(a_i)}{\sigma_a^2(a_i)}\right)
        \sigma_z^2(A)=\left(1+\sum_{i=1}^{n}\sigma_a^2(a_i)^{-1}\right)^{-1}
        """

        # Calculate the summations in the formulas.
        rho_sum = 0
        mu_div_roh_sum = 0
        for mu, rho in mu_rho_points:
            roh_sum += 1 / rho
            mu_div_roh_sum += mu / rho

        # Calculate the final mu_z and sigma_z according to the assumptions.
        sigma_z = 1 / (1 + roh_sum)
        mu_z = sigma_z * mu_div_roh_sum

        return (mu_z, sigma_z)

    def __str__(self):
        ret: str = "EncoderDeepProMP(neurons=%s)" % self.neurons
        ret += "\n" + str(self.net)
        return ret

    def __call__(self, x):
        return self.net(x)
