from typing import List, Type

import torch
import torch.nn as nn

from moppy.interfaces import LatentDecoder
from moppy.deep_promp import DecoderDeepProMP
from moppy.trajectory.state import EndEffectorPose, TrajectoryState, JointConfiguration


class DecoderKIDProMP(DecoderDeepProMP, nn.Module):
    """
    A normal ProMP decoder that instead of outputting an EndEffectorPose,
    it outputs a list of joint configurations which are then fed
    into differentiable kinematics to retrieve a reachable EndEffectorPose.
    """
    # TODO https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 trajectory_state_class: Type[TrajectoryState] = EndEffectorPose,
                 activation_function: Type[nn.Module] = nn.ReLU,
                 activation_function_params: dict = {},
                 dh_parameters: List[dict] = None):
        super().__init__(latent_variable_dimension, hidden_neurons, trajectory_state_class, activation_function,
                         activation_function_params)

        self.dh_parameters = dh_parameters
        self.output_dimension = JointConfiguration.get_dimensions() - JointConfiguration.get_time_dimension()

    def decode_from_latent_variable(self, latent_variable: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        """This is the complete procedure of decoding a latent variable z to a Tensor representing the trajectoryState
        using the decoder network. The latent variable z is concatenated with the time t"""

        if isinstance(time, float):
            time = torch.tensor([time])
        nn_input = torch.cat((latent_variable, time), dim=-1).float()
        nn_output = self.net(nn_input)
        return self.forward_kinematics(nn_output)

    def forward_kinematics(self, joint_configuration: torch.Tensor) -> torch.Tensor:
        """
        This function calculates the forward kinematics of the robot.
        :param joint_configuration: joint configuration of the robot.
        :return: end effector pose of the robot.
        """

        cum_mat = torch.eye(4)
        for i in range(1, self.output_dimension + 2):
            cum_mat = torch.matmul(cum_mat, self.homog_trans_mat(i, joint_configuration))

        # extract position and quaternion orientation from the matrix.
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

        a = self.dh_parameters[n - 1]['a']
        alpha = self.dh_parameters[n - 1]['alpha']
        d = self.dh_parameters[n - 1]['d']
        theta = self.dh_parameters[n - 1]['theta'] + joint_configuration[n - 1]

        return torch.array([
            [torch.cos(theta), -torch.sin(theta) * torch.cos(alpha), torch.sin(theta) * torch.sin(alpha),
             a * torch.cos(theta)],
            [torch.sin(theta), torch.cos(theta) * torch.cos(alpha), -torch.cos(theta) * torch.sin(alpha),
             a * torch.sin(theta)],
            [0, torch.sin(alpha), torch.cos(alpha), d],
            [0, 0, 0, 1]
        ])
