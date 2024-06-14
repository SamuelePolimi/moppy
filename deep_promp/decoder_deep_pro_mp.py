from typing import List, Union

import numpy as np
import torch.nn as nn

from interfaces.latent_decoder import LatentDecoder


class DecoderDeepProMP(LatentDecoder):
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
        if not all(neurons[i] <= neurons[i + 1] for i in range(len(neurons) - 1)):
            raise ValueError("The number of neurons must be decrease monotonically. Got '%s'" % neurons)

        self.neurons = neurons
        self.activation_function = activation_function

        linear_layer = nn.Linear
        layers = []

        # create the network
        for i in range(len(self.neurons) - 1):
            layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), self.activation_function()]

        self.net = nn.Sequential(*layers).float()

    def decode_from_latent_variable(self, latent_variable: np.array, time: float):
        # This is the complete procedure of decoding a latent variable z to a trajectory point x

        # 2. Pass the already sampled z (latent_variable) and the time through the decoder network to get the
        # trajectory state x
        trajectory_state_mu_sigma = self.net(latent_variable, time)

        # TODO find better way get the size of the output layer of the decoder network and split it into mu and sigma
        output_size = len(trajectory_state_mu_sigma)
        # TODO the dimensions on both mu and sigma should match the dimensions of the TrajectoryState, check for it

        return trajectory_state_mu_sigma[:output_size // 2], trajectory_state_mu_sigma[output_size // 2:]

    def evidence_lowerbound(self, sampled_latent_variable: np.array, trajectory_state_distribution: np.array):
        evidence_lowerbound = 0

        # Maximise the probability for each dimension of the trajectory_state_distribution
        # TODO


        # Ensure that the latent_variable stays close to a normal distribution
        # norm (0, I) . logpdf (z)
        # TODO

        return evidence_lowerbound



    def __str__(self):
        ret: str = "DecoderDeepProMP(neurons=%s)" % self.neurons
        ret += "\n" + str(self.net)
        return ret

    def __call__(self, x):
        return self.net(x)
