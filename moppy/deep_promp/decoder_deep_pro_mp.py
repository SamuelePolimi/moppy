from typing import List, Union, Type

import os
import torch
import torch.nn as nn

from moppy.interfaces import LatentDecoder
from moppy.trajectory.state import JointConfiguration, TrajectoryState


class DecoderDeepProMP(LatentDecoder, nn.Module):
    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[TrajectoryState] = JointConfiguration,
                 activation_function: Type[nn.Module] = nn.ReLU,
                 activation_function_params: dict = {}):
        nn.Module.__init__(self)

        # Check if the trajectory state class is a subclass of TrajectoryState
        if TrajectoryState not in trajectory_state_class.__mro__:
            raise TypeError(f"The trajectory state class must be a subclass of '{TrajectoryState.__name__}'. "
                            f"Got '{trajectory_state_class}'"
                            f"\nThe parent classes are: {trajectory_state_class.__mro__}")

        # The output dimension is the total dimension of the trajectory state minus the time dimension
        self.output_dimension = trajectory_state_class.get_dimensions() - trajectory_state_class.get_time_dimension()
        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension
        self.activation_function = activation_function
        self.activation_function_params = activation_function_params
        self.trajectory_state_class = trajectory_state_class

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [latent_variable_dimension + trajectory_state_class.get_time_dimension()] + \
            hidden_neurons + [self.output_dimension]
        if latent_variable_dimension <= 0:
            raise ValueError("The latent_variable_dimension must be greater than 0. Got '%s'" % latent_variable_dimension)
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
    def load_from_save_file(cls, path: str = '', file: str = "decoder_deep_pro_mp.pth") -> 'DecoderDeepProMP':
        """Load a model from a file and return a DecoderDeepProMP instance."""
        # Load the model data
        file_path = os.path.join(path, file)
        model_data = torch.load(file_path)

        # Reconstruct the model using the saved configuration
        model = cls(
            latent_variable_dimension=model_data['latent_variable_dimension'],
            hidden_neurons=model_data['hidden_neurons'],
            trajectory_state_class=model_data['trajectory_state_class'],
            activation_function=model_data['activation_function'],
            activation_function_params=model_data['activation_function_params']
        )

        # Load the saved state_dict into the model's neural network
        model.net.load_state_dict(model_data['state_dict'])
        model.eval()  # Set to evaluation mode if needed

        return model

    def create_layers(self):
        layers = []
        for i in range(len(self.neurons) - 2):
            layers += [nn.Linear(self.neurons[i], self.neurons[i + 1]),
                       self.activation_function(**self.activation_function_params)]
        layers += [nn.Linear(self.neurons[-2], self.neurons[-1])]
        return layers

    def __init_weights(self, m):
        """Initialize the weights and biases of the network using Xavier initialization and a bias of 0.01"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def decode_from_latent_variable(self, latent_variable: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        """This is the complete procedure of decoding a latent variable z to a Tensor representing the trajectoryState
        using the decoder network. The latent variable z is concatenated with the time t"""

        if isinstance(time, float):
            time = torch.tensor([time])
        nn_input = torch.cat((latent_variable, time), dim=-1).float()
        return self.net(nn_input)

    def save_decoder(self, path: str = '', filename: str = "decoder_deep_pro_mp.pth"):
        """Save the decoder to a file, including the state_dict of the network and the configuration of the model.
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

    def save_model(self, path: str = '', filename: str = "decoder_model_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, path: str = '', filename: str = "decoder_model_deep_pro_mp.pth"):
        file_path = os.path.join(path, filename)
        self.net.load_state_dict(torch.load(file_path))

    def forward(self, latent_variable: torch.Tensor, time: torch.Tensor | float):
        return self.decode_from_latent_variable(latent_variable, time)

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
