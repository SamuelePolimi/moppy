import torch
import torch.nn as nn

from typing import List, Union
from interfaces.encoder_deeppromp import EncoderProMP


class VariationalGaussianEncoder(EncoderProMP):

    def __init__(self,
                 input_neurons: int,
                 z_dimension: int,
                 hidden_layers_neurons: List[int] = [256],
                 activation_function="relu"):

        self.encoder = VariationalGaussian(input_neurons=input_neurons,
                                           z_dimension=z_dimension,
                                           hidden_layers_neurons=hidden_layers_neurons,
                                           activation_function=activation_function)

    def generate_latent_variable(self, trajectory, context_trajectory):
        raise NotImplementedError

    def sample(self, mean, standart_deviation, percentage_of_standard_deviation=None):
        raise NotImplementedError


class VariationalGaussian(nn.Module):

    def __init__(self,
                 input_neurons: int,
                 z_dimension: int,
                 hidden_layers_neurons: List[int] = [256],
                 activation_function="relu",
                 linear=False):
        super().__init__()
        if linear:
            self.transform = nn.Linear(in_features=input_neurons, out_features=z_dimension * 2)
        else:
            self.transform = MultiLayerPerceptron(input_neurons=input_neurons,
                                                  hidden_layers_neurons=hidden_layers_neurons,
                                                  output_neurons=z_dimension * 2,
                                                  activation_function=activation_function)
        self.standart_activation_function = nn.Softplus()
        self.z_dimension = z_dimension

    def forward(self, inputs, sample=True):
        params = self.transform(inputs)
        mu, log_std = torch.split(params, [self.z_dimension, self.z_dimension], dim=-1)
        log_std = torch.clamp(log_std, -10, 10)
        return mu, self.standart_activation_function(log_std)

    def forward_features(self, phi):
        params = self.transform.net[-1](phi)
        mu, log_std = torch.split(params, [self.z_dimension, self.z_dimension], dim=-1)

        log_std = torch.clamp(log_std, -10, 10)
        return mu, self.standart_activation_function(log_std)


class MultiLayerPerceptron(nn.Module):
    """ This class is a simple MultiLayerPerceptron implementation, it is a simple wrapper around the
    torch.nn.Sequential class. It is useful to create a simple feedforward neural network with
    a variable number of hidden layers. The activation function can be chosen from the torch.nn module.
    The input and output dimensions are given by the input_neurons and output_neurons parameters.
    The hidden_layers_neurons parameter is a list of integers that specifies the number of neurons in each hidden.
    """

    def __init__(self,
                 input_neurons: int,
                 hidden_layers_neurons: List[int],
                 output_neurons: int,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(activation_function(), Union[nn.Tanh, nn.ReLU, nn.Sigmoid]):
            raise ValueError("The activation function must be one of [nn.Tanh, nn.ReLU, nn.Sigmoid]. Not {}".format(type(activation_function())))
        linear_layer = nn.Linear

        # Add input(first) layer
        layers = [linear_layer(input_neurons, hidden_layers_neurons[0]), activation_function()]

        # Add hidden layers: hidden_layers_neurons[i] -> hidden_layers_neurons[i+1]
        for (in_d, out_d) in zip(hidden_layers_neurons[:-1], hidden_layers_neurons[1:]):
            layers = layers + [linear_layer(in_d, out_d)]
            layers = layers + [activation_function()]

        # Add ouput(last) layer
        layers = layers + [linear_layer(hidden_layers_neurons[-1], output_neurons)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    """
    def _select_lin(self, lin):
        if lin == 'regular':
            return nn.Linear
        elif lin == 'reg-no-bias':
            return lambda input, output: nn.Linear(input, output, bias=False)
    """
