import torch.nn as nn

from typing import List, Tuple, Union

from interfaces.encoder_pro_mp import EncoderProMP
from mp_types.types import LatentVariableZ
from trajectory.trajectory import Trajectory


class EncoderDeepProMP(EncoderProMP):

    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: List[int],
                 output_neurons: int,
                 activation_function: Union[nn.Tanh, nn.ReLU, nn.Sigmoid] = nn.ReLU):
        super().__init__()
        linear_layer = nn.Linear
        layers = []

        # Add input(first) layer
        layers = [linear_layer(input_neurons, hidden_neurons[0]), activation_function()]

        # Add hidden layers
        for i in range(len(hidden_neurons) - 1):
            layers += [linear_layer(hidden_neurons[i], hidden_neurons[i + 1]), activation_function()]

        # Add output(last) layer
        layers += [linear_layer(hidden_neurons[-1], output_neurons)]
        self.net = nn.Sequential(*layers)

    def generate_latent_variable(
            self,
            trajectory: Trajectory,
            context_trajectory: List[Trajectory]
            ) -> List[LatentVariableZ]:
        raise NotImplementedError()

    def sample(self,
               mean: float,
               standard_deviation: float,
               percentage_of_standard_deviation: float = None
               ) -> Tuple[LatentVariableZ, float]:
        raise NotImplementedError()


a = EncoderDeepProMP(1, [2, 3], 4)