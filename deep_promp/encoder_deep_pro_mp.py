import torch.nn as nn

from typing import List, Tuple, Union

from interfaces.encoder_pro_mp import EncoderProMP
from mp_types.types import LatentVariableZ
from trajectory.trajectory import Trajectory


class EncoderDeepProMP(EncoderProMP):

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

    def __str__(self):
        ret: str = "EncoderDeepProMP(neurons=%s)" % self.neurons
        ret += "\n" + str(self.net)
        return ret

    def __call__(self, x):
        return self.net(x)
