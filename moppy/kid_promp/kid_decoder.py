import os
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from moppy.interfaces import LatentDecoder
from moppy.kid_promp.forward_kinematics import forward_kinematics_batch
from moppy.trajectory.state import EndEffectorPose, TrajectoryState


class DecoderKIDProMP(LatentDecoder, nn.Module):
    """
    A normal ProMP decoder that instead of outputting an EndEffectorPose,
    it outputs a list of joint configurations which are then fed
    into differentiable kinematics to retrieve a reachable EndEffectorPose.
    """

    def __init__(self, latent_variable_dimension: int, hidden_neurons: List[int],
                 activation_function: Type[nn.Module] = nn.Softmax, activation_function_params: dict = {},
                 dh_parameters_craig: List[dict] = None, degrees_of_freedom=7,
                 min_joints: List[float] = None, max_joints: List[float] = None):
        nn.Module.__init__(self)
        self.dh_parameters = dh_parameters_craig
        self.output_dimension = degrees_of_freedom  # Joint configuration size of the robot.

        if min_joints is None or max_joints is None or len(min_joints) != len(max_joints) or len(min_joints) != degrees_of_freedom:
            raise ValueError("The minimum and maximum joint values must be provided.")

        self.min_joints = torch.tensor(min_joints)
        self.max_joints = torch.tensor(max_joints)

        # Check if any minimum joint value is greater than the maximum joint value
        if torch.any(self.min_joints > self.max_joints):
            raise ValueError("The minimum joint values must be less than the maximum joint values")

        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension

        self.activation_function = activation_function
        self.activation_function_params = activation_function_params

        self.trajectory_state_class = EndEffectorPose

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [latent_variable_dimension + EndEffectorPose.get_time_dimension()] + \
                       hidden_neurons + [self.output_dimension]
        if latent_variable_dimension <= 0:
            raise ValueError(
                "The latent_variable_dimension must be greater than 0. Got '%s'" % latent_variable_dimension)
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

        nn_output = self.decode_to_joints(latent_variable, time)
        return forward_kinematics_batch(self.dh_parameters, nn_output)

    def decode_to_joints(self, latent_variable: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        if isinstance(time, float):
            time = torch.tensor([time])
        nn_input = torch.cat((latent_variable, time), dim=-1).float()
        nn_output = self.net(nn_input)

        joint_pct = (0.5 + torch.tanh(0.15 * nn_output) * 0.4)
        tanh_joint_range = self.min_joints + (self.max_joints - self.min_joints) * joint_pct

        return tanh_joint_range

        # smooth clamp
        #min_jnt = F.softplus(nn_output - self.min_joints, 0.5, 20.0) + self.min_joints
        #clamped_jnt = self.max_joints - F.softplus(self.max_joints - min_jnt, 0.5, 20.0)
        #return clamped_jnt

    def sigmoid_clamp(self, x, mi, mx):
        # Normalize x between mi and mx
        normalized_x = (x - mi) / (mx - mi)
        # sigmoid will smoothly map normalized_x between 0 and 1
        sigmoid = torch.sigmoid(4 * (normalized_x - 0.5))
        # Scale and shift to match the original function's output range
        result = mi + (mx - mi) * sigmoid
        # The result will always be between mi and mx
        return result

    def save_model(self, path: str = '', filename: str = "decoder_kid.pth"):
        file_path = os.path.join(path, filename)
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, path: str = '', filename: str = "decoder_kid.pth"):
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
