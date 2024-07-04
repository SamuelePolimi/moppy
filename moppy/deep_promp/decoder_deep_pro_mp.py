from typing import List, Union, Type

import os
import torch
import torch.nn as nn

from moppy.interfaces.latent_decoder import LatentDecoder
from moppy.trajectory.state.joint_configuration import JointConfiguration
from moppy.trajectory.state.trajectory_state import TrajectoryState


class DecoderDeepProMP(LatentDecoder):
    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[TrajectoryState] = JointConfiguration,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()

        # Check if the trajectory state class is a subclass of TrajectoryState
        if trajectory_state_class not in TrajectoryState.__subclasses__():
            raise TypeError(f"The trajectory state class must be a subclass of '{TrajectoryState.__name__}'. "
                            f"Got '{trajectory_state_class}'"
                            f"\nThe usable subclasses are {TrajectoryState.__subclasses__()}")

        # The output dimension is the total dimension of the trajectory state minus the time dimension
        self.output_dimension = trajectory_state_class.get_dimensions() - trajectory_state_class.get_time_dimension()
        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension
        self.activation_function = activation_function
        self.trajectory_state_class = trajectory_state_class

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [latent_variable_dimension + trajectory_state_class.get_time_dimension()] + \
            hidden_neurons + [self.output_dimension]

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
                layers += [linear_layer(self.neurons[i], self.neurons[i + 1])]
            else:
                layers += [linear_layer(self.neurons[i], self.neurons[i + 1]), self.activation_function()]
        self.net = nn.Sequential(*layers).float()

    def decode_from_latent_variable(self, latent_variable: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        """This is the complete procedure of decoding a latent variable z to a Tensor representing the trajectoryState
        using the decoder network. The latent variable z is concatenated with the time t"""

        if isinstance(time, float):
            time = torch.tensor([time])
        nn_input = torch.cat((latent_variable, time), dim=0).float()
        return self.net(nn_input)

    def save_model(self, path: str = '', filename: str = "decoder_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, path: str = '', filename: str = "decoder_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        self.net.load_state_dict(torch.load(file_path))

    def __str__(self):
        ret: str = "DecoderDeepProMP() {"
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
