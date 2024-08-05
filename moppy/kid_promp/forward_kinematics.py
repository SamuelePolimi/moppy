import torch


def forward_kinematics(dh_parameters, joint_configuration: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the forward kinematics of the robot.
    :param dh_parameters: dh parameters of the robot.
    :param joint_configuration: joint configuration of the robot.
    :return: end effector pose of the robot.
    """

    cum_mat = torch.eye(4)
    for i in range(1, len(dh_parameters) + 1):
        cum_mat = torch.matmul(cum_mat, homog_trans_mat(dh_parameters, i, joint_configuration))

    # extract position and quaternion orientation from the matrix.
    # TODO fix sqrt NaN
    eta = 0.5 * torch.sqrt(1 + torch.trace(cum_mat[:3, :3]))
    epsilon = (1 / 2) * torch.tensor(
        [torch.sign(cum_mat[2, 1] - cum_mat[1, 2]) * torch.sqrt(cum_mat[0, 0] - cum_mat[1, 1] - cum_mat[2, 2] + 1),
         torch.sign(cum_mat[0, 2] - cum_mat[2, 0]) * torch.sqrt(cum_mat[1, 1] - cum_mat[2, 2] - cum_mat[0, 0] + 1),
         torch.sign(cum_mat[1, 0] - cum_mat[0, 1]) * torch.sqrt(cum_mat[2, 2] - cum_mat[0, 0] - cum_mat[1, 1] + 1)])

    return torch.tensor([cum_mat[0, 3], cum_mat[1, 3], cum_mat[2, 3], epsilon[0], epsilon[1], epsilon[2], eta])


def homog_trans_mat(dh_parameters, n, joint_configuration: torch.Tensor) -> torch.Tensor:
    """
    This function returns T_{n}^{n-1}, expect when n=0, then returns T_0^0, which is I.
    :param dh_parameters: dh parameters of the robot.
    :param n: number of the transformation (goal index).
    :param joint_configuration: joint configuration of the robot.
    :return: 4x4 homogeneous transformation matrix
    """
    if n == 0:
        return torch.eye(4)

    a = torch.tensor([dh_parameters[n - 1]['a']])
    alpha = torch.tensor([dh_parameters[n - 1]['alpha']])
    d = torch.tensor([dh_parameters[n - 1]['d']])

    theta = torch.tensor([dh_parameters[n - 1]['theta']]) + joint_configuration[n - 1]

    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta) * torch.cos(alpha), torch.sin(theta) * torch.sin(alpha),
         a * torch.cos(theta)],
        [torch.sin(theta), torch.cos(theta) * torch.cos(alpha), -torch.cos(theta) * torch.sin(alpha),
         a * torch.sin(theta)],
        [0, torch.sin(alpha), torch.cos(alpha), d],
        [0, 0, 0, 1]
    ])
