import torch


def forward_kinematics(dh_parameters_craig, joint_configuration: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the forward kinematics of the robot.
    :param dh_parameters_craig: modified dh parameters of the robot.
    :param joint_configuration: joint configuration of the robot.
    :return: end effector pose of the robot.
    """

    cum_mat = torch.eye(4)
    for i in range(1, len(dh_parameters_craig) + 1):
        cum_mat = torch.matmul(cum_mat, homog_trans_mat_craig(dh_parameters_craig, i, joint_configuration))

    return mat_to_pose(cum_mat)


def mat_to_pose(mat: torch.Tensor) -> torch.Tensor:
    # extract position and quaternion orientation from the matrix.

    # mathematically, the sqrt contents can reach exactly zero.
    # due to rounding errors they can be slightly negative and cause NaNs.
    # to avoid this, we use torch.relu to ensure the contents are always positive.
    eta = 0.5 * torch.sqrt(torch.relu(1 + torch.trace(mat[:3, :3])))
    epsilon = 0.5 * torch.Tensor(
        [torch.sign(mat[2, 1] - mat[1, 2]) * torch.sqrt(torch.relu(mat[0, 0] - mat[1, 1] - mat[2, 2] + 1)),
         torch.sign(mat[0, 2] - mat[2, 0]) * torch.sqrt(torch.relu(mat[1, 1] - mat[2, 2] - mat[0, 0] + 1)),
         torch.sign(mat[1, 0] - mat[0, 1]) * torch.sqrt(torch.relu(mat[2, 2] - mat[0, 0] - mat[1, 1] + 1))])

    return torch.Tensor([mat[0, 3], mat[1, 3], mat[2, 3], epsilon[0], epsilon[1], epsilon[2], eta])


def homog_trans_mat_craig(dh_parameters_craig, i, joint_configuration: torch.Tensor) -> torch.Tensor:
    """
    This function returns T_{n}^{n-1}, expect when n=0, then returns T_0^0, which is I.
    :param dh_parameters_craig: modified dh parameters of the robot (craigs convention)
    :param i: number of the transformation (goal index).
    :param joint_configuration: joint configuration of the robot.
    :return: 4x4 homogeneous transformation matrix
    """

    if i == 0:
        return torch.eye(4)

    a_nm1 = torch.Tensor([dh_parameters_craig[i - 1]['a']])
    alpha_nm1 = torch.Tensor([dh_parameters_craig[i - 1]['alpha']])
    d_n = torch.Tensor([dh_parameters_craig[i - 1]['d']])
    theta_n = (torch.Tensor([dh_parameters_craig[i - 1]['theta']]) +
               (joint_configuration[i - 1] if i - 1 < len(joint_configuration) else 0))

    return torch.Tensor([
        [torch.cos(theta_n), -torch.sin(theta_n), 0, a_nm1],
        [torch.sin(theta_n) * torch.cos(alpha_nm1), torch.cos(theta_n) * torch.cos(alpha_nm1), -torch.sin(alpha_nm1),
         -d_n * torch.sin(alpha_nm1)],
        [torch.sin(theta_n) * torch.sin(alpha_nm1), torch.cos(theta_n) * torch.sin(alpha_nm1), torch.cos(alpha_nm1),
         d_n * torch.cos(alpha_nm1)],
        [0, 0, 0, 1]])
