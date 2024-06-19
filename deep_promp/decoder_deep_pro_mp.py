from typing import List, Union, Type

import numpy as np
import torch
import torch.nn as nn

from interfaces.latent_decoder import LatentDecoder
from trajectory.state.joint_configuration import JointConfiguration
from trajectory.state.trajectory_state import TrajectoryState


class DecoderDeepProMP(LatentDecoder):
    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[TrajectoryState] = JointConfiguration,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()
        print("DecoderDeepProMP init")

        # The output dimension is the total dimension of the trajectory state minus the time dimension
        self.output_dimension = trajectory_state_class.get_dimensions() - trajectory_state_class.get_time_dimension()
        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension
        self.activation_function = activation_function
        self.trajectory_state_class = trajectory_state_class

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [latent_variable_dimension * 2 + trajectory_state_class.get_time_dimension()] + hidden_neurons + [self.output_dimension]

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

    def decode_from_latent_variable(self, latent_variable: np.array, time: float):
        # This is the complete procedure of decoding a latent variable z to a trajectory point x

        # 2. Pass the already sampled z (latent_variable) and the time through the decoder network to get the
        # trajectory state x
        nn_input = np.concatenate((latent_variable, [time]), axis=0)
        nn_input = torch.from_numpy(nn_input).float()
        trajectory_state_mu_sigma = self.net(nn_input)

        # TODO REMOVE THIS RETURN; THIS IS JUST FOR TESTING
        return trajectory_state_mu_sigma

        # Output should be a vector with the same dimension as the TrajectoryState that is being used.
        return self.trajectory_state_class.from_vector_without_time(trajectory_state_mu_sigma)

    def evidence_lowerbound(self, sampled_latent_variable: np.array, trajectory_state_distribution: np.array):
        evidence_lowerbound = 0

        # Maximise the probability for each dimension of the trajectory_state_distribution
        # TODO


        # Ensure that the latent_variable stays close to a normal distribution
        # norm (0, I) . logpdf (z)
        # TODO

        return evidence_lowerbound

    def __str__(self):
        ret: str = "DecoderDeepProMP {"
        ret += "\n\t" + f'neurons: {self.neurons}'
        ret += '\n\t' + f'latent_variable_dimension: {self.latent_variable_dimension}'
        ret += "\n\t" + f'hidden_neurons: {self.hidden_neurons}'
        ret += "\n\t" + f'output_dimension: {self.output_dimension}'
        ret += "\n\t" + f'activation_function: {self.activation_function}'
        ret += "\n\t" + f'trajectory_state_class: {self.trajectory_state_class}'
        ret += "\n\t" + f'net: {str(self.net)}'
        ret += "\n" + "}"
        return ret

    def __repr__(self):
        return f'DecoderDeepProMP(neurons={self.neurons})'

    def __call__(self, *args, **kwargs):
        return self.decode_from_latent_variable(*args, **kwargs)
