import os
from typing import List, Type

import torch
import torch.nn as nn

from moppy.interfaces import LatentDecoder
from moppy.trajectory.state import EndEffectorPose, TrajectoryState


class DecoderKIDProMP(LatentDecoder, nn.Module):
    """
    A normal ProMP decoder that instead of outputting an EndEffectorPose,
    it outputs a list of joint configurations which are then fed
    into differentiable kinematics to retrieve a reachable EndEffectorPose.
    """

    def __init__(self, latent_variable_dimension: int, hidden_neurons: List[int],
                 activation_function: Type[nn.Module] = nn.Softmax, activation_function_params: dict = {},
                 dh_parameters: List[dict] = None):
        nn.Module.__init__(self)
        self.dh_parameters = dh_parameters
        self.output_dimension = len(dh_parameters)  # Joint configuration size of the robot.

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

        if isinstance(time, float):
            time = torch.tensor([time])
        nn_input = torch.cat((latent_variable, time), dim=-1).float()
        nn_output = self.net(nn_input)

        if len(nn_output.shape) == 1:
            return self.forward_kinematics(nn_output)
        elif len(nn_output.shape) == 2:
            return torch.stack([self.forward_kinematics(joint_configuration)
                                for joint_configuration in nn_output], dim=0)
        else:
            raise ValueError("Too many dimensions")

    def forward_kinematics(self, joint_configuration: torch.Tensor) -> torch.Tensor:
        """
        This function calculates the forward kinematics of the robot.
        :param joint_configuration: joint configuration of the robot.
        :return: end effector pose of the robot.
        """

        cum_mat = torch.eye(4)
        for i in range(1, self.output_dimension + 1):
            cum_mat = torch.matmul(cum_mat, self.homog_trans_mat(i, joint_configuration))

        # extract position and quaternion orientation from the matrix.
        # TODO fix sqrt NaN
        eta = 0.5 * torch.sqrt(1 + torch.trace(cum_mat[:3, :3]))
        epsilon = (1 / 2) * torch.tensor(
            [torch.sign(cum_mat[2, 1] - cum_mat[1, 2]) * torch.sqrt(cum_mat[0, 0] - cum_mat[1, 1] - cum_mat[2, 2] + 1),
             torch.sign(cum_mat[0, 2] - cum_mat[2, 0]) * torch.sqrt(cum_mat[1, 1] - cum_mat[2, 2] - cum_mat[0, 0] + 1),
             torch.sign(cum_mat[1, 0] - cum_mat[0, 1]) * torch.sqrt(cum_mat[2, 2] - cum_mat[0, 0] - cum_mat[1, 1] + 1)])

        return torch.tensor([cum_mat[0, 3], cum_mat[1, 3], cum_mat[2, 3], epsilon[0], epsilon[1], epsilon[2], eta])

    def homog_trans_mat(self, n, joint_configuration: torch.Tensor) -> torch.Tensor:
        """
        This function returns T_{n}^{n-1}, expect when n=0, then returns T_0^0, which is I.
        :param n: number of the transformation (goal index).
        :param joint_configuration: joint configuration of the robot.
        :return: 4x4 homogeneous transformation matrix
        """
        if n == 0:
            return torch.eye(4)

        a = torch.tensor([self.dh_parameters[n - 1]['a']])
        alpha = torch.tensor([self.dh_parameters[n - 1]['alpha']])
        d = torch.tensor([self.dh_parameters[n - 1]['d']])

        theta = torch.tensor([self.dh_parameters[n - 1]['theta']]) + joint_configuration[n - 1]

        return torch.tensor([
            [torch.cos(theta), -torch.sin(theta) * torch.cos(alpha), torch.sin(theta) * torch.sin(alpha),
             a * torch.cos(theta)],
            [torch.sin(theta), torch.cos(theta) * torch.cos(alpha), -torch.cos(theta) * torch.sin(alpha),
             a * torch.sin(theta)],
            [0, torch.sin(alpha), torch.cos(alpha), d],
            [0, 0, 0, 1]
        ])


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
