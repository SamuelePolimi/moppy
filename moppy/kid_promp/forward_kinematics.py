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
    eta = 0.5 * torch.sqrt(1 + torch.trace(mat[:3, :3]))  # TODO fix sqrt NaN
    epsilon = 0.5 * torch.tensor(
        [torch.sign(mat[2, 1] - mat[1, 2]) * torch.sqrt(mat[0, 0] - mat[1, 1] - mat[2, 2] + 1),
         torch.sign(mat[0, 2] - mat[2, 0]) * torch.sqrt(mat[1, 1] - mat[2, 2] - mat[0, 0] + 1),
         torch.sign(mat[1, 0] - mat[0, 1]) * torch.sqrt(mat[2, 2] - mat[0, 0] - mat[1, 1] + 1)])

    return torch.tensor([mat[0, 3], mat[1, 3], mat[2, 3], epsilon[0], epsilon[1], epsilon[2], eta])


def homog_trans_mat(dh_parameters_standard, n, joint_configuration: torch.Tensor) -> torch.Tensor:
    """
    This function returns T_{n}^{n-1}, expect when n=0, then returns T_0^0, which is I.
    :param dh_parameters_standard: dh parameters of the robot.
    :param n: number of the transformation (goal index).
    :param joint_configuration: joint configuration of the robot.
    :return: 4x4 homogeneous transformation matrix
    """
    if n == 0:
        return torch.eye(4)

    a = torch.tensor([dh_parameters_standard[n - 1]['a']])
    alpha = torch.tensor([dh_parameters_standard[n - 1]['alpha']])
    d = torch.tensor([dh_parameters_standard[n - 1]['d']])

    theta = torch.tensor([dh_parameters_standard[n - 1]['theta']]) + joint_configuration[n - 1]

    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta) * torch.cos(alpha), torch.sin(theta) * torch.sin(alpha),
         a * torch.cos(theta)],
        [torch.sin(theta), torch.cos(theta) * torch.cos(alpha), -torch.cos(theta) * torch.sin(alpha),
         a * torch.sin(theta)],
        [0, torch.sin(alpha), torch.cos(alpha), d],
        [0, 0, 0, 1]
    ])


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

    a_nm1 = torch.tensor([dh_parameters_craig[i - 1]['a']])
    alpha_nm1 = torch.tensor([dh_parameters_craig[i - 1]['alpha']])
    d_n = torch.tensor([dh_parameters_craig[i - 1]['d']])
    theta_n = (torch.tensor([dh_parameters_craig[i - 1]['theta']]) +
               (joint_configuration[i - 1] if i - 1 < len(joint_configuration) else 0))

    return torch.tensor([
        [torch.cos(theta_n), -torch.sin(theta_n), 0, a_nm1],
        [torch.sin(theta_n) * torch.cos(alpha_nm1), torch.cos(theta_n) * torch.cos(alpha_nm1), -torch.sin(alpha_nm1),
         -d_n * torch.sin(alpha_nm1)],
        [torch.sin(theta_n) * torch.sin(alpha_nm1), torch.cos(theta_n) * torch.sin(alpha_nm1), torch.cos(alpha_nm1),
         d_n * torch.cos(alpha_nm1)],
        [0, 0, 0, 1]])
