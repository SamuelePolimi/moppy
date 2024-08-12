import torch


def forward_kinematics_batch(dh_parameters_craig, joint_configurations: torch.Tensor) -> torch.Tensor:
    """
    Returns the end effector pose of the robot for a batch of joint configurations.
    :param dh_parameters_craig: Modified DH parameters of the robot (Craig's convention).
    :param joint_configurations: Joint configurations of the robot, shape (n, x).
    :return: A tensor of shape (n, 7) containing the end effector pose for each joint configuration.
    """
    if len(joint_configurations.shape) != 2:
        joint_configurations = joint_configurations.unsqueeze(0)

    n, _ = joint_configurations.shape

    cum_mat = torch.eye(4, device=joint_configurations.device).repeat(n, 1, 1)
    for i in range(1, len(dh_parameters_craig) + 1):
        cum_mat = torch.bmm(cum_mat, homog_trans_mat_craig_batch(dh_parameters_craig, i, joint_configurations))

    return mat_to_pose_batch(cum_mat)


def mat_to_pose_batch(mat: torch.Tensor) -> torch.Tensor:
    # Assuming mat has shape (n, 4, 4)
    if mat.shape[-2:] != (4, 4):
        raise ValueError("Input tensor must have shape (n, 4, 4)")

    # Extract position and quaternion orientation from the matrix for each batch element
    eta = 0.5 * torch.sqrt(torch.relu(1 + torch.vmap(torch.trace)(mat[:, :3, :3])))  # shape (n,)
    epsilon = 0.5 * torch.stack([
        torch.sign(mat[:, 2, 1] - mat[:, 1, 2]) * torch.sqrt(
            torch.relu(mat[:, 0, 0] - mat[:, 1, 1] - mat[:, 2, 2] + 1)),
        torch.sign(mat[:, 0, 2] - mat[:, 2, 0]) * torch.sqrt(
            torch.relu(mat[:, 1, 1] - mat[:, 2, 2] - mat[:, 0, 0] + 1)),
        torch.sign(mat[:, 1, 0] - mat[:, 0, 1]) * torch.sqrt(torch.relu(mat[:, 2, 2] - mat[:, 0, 0] - mat[:, 1, 1] + 1))
    ], dim=0)  # shape (3, n)

    position_xyz = mat[:, :3, 3]  # shape (n, 3)
    # concat correctly position_xy, epsilon, eta
    return torch.cat([position_xyz, epsilon.T, eta.unsqueeze(1)], dim=1)


def homog_trans_mat_craig_batch(dh_parameters_craig, i, joint_configurations: torch.Tensor) -> torch.Tensor:
    """
    Returns a batch of 4x4 homogeneous transformation matrices based on Craig's DH parameters and joint configurations.
    :param dh_parameters_craig: Modified DH parameters of the robot (Craig's convention).
    :param i: Number of the transformation (goal index).
    :param joint_configurations: Joint configurations of the robot, shape (n, x).
    :return: A tensor of shape (n, 4, 4) containing the homogeneous transformation matrices.
    """
    n, _ = joint_configurations.shape

    if i == 0:
        # Return identity matrix for each joint configuration
        return torch.eye(4, device=joint_configurations.device).repeat(n, 1, 1)

    a_nm1 = torch.tensor([dh_parameters_craig[i - 1]['a']], device=joint_configurations.device).repeat(n)
    alpha_nm1 = torch.tensor([dh_parameters_craig[i - 1]['alpha']], device=joint_configurations.device).repeat(n)
    d_n = torch.tensor([dh_parameters_craig[i - 1]['d']], device=joint_configurations.device).repeat(n)

    # Compute theta_n for each joint configuration
    theta_n = torch.tensor([dh_parameters_craig[i - 1]['theta']], device=joint_configurations.device).repeat(n)
    theta_n += joint_configurations[:, i - 1] if i - 1 < joint_configurations.shape[1] else torch.zeros(n)

    # Construct transformation matrices for each joint configuration
    trans_matrices = torch.zeros((n, 4, 4), device=joint_configurations.device)
    trans_matrices[:, 0, 0] = torch.cos(theta_n)
    trans_matrices[:, 0, 1] = -torch.sin(theta_n)
    trans_matrices[:, 0, 3] = a_nm1

    trans_matrices[:, 1, 0] = torch.sin(theta_n) * torch.cos(alpha_nm1)
    trans_matrices[:, 1, 1] = torch.cos(theta_n) * torch.cos(alpha_nm1)
    trans_matrices[:, 1, 2] = -torch.sin(alpha_nm1)
    trans_matrices[:, 1, 3] = -d_n * torch.sin(alpha_nm1)

    trans_matrices[:, 2, 0] = torch.sin(theta_n) * torch.sin(alpha_nm1)
    trans_matrices[:, 2, 1] = torch.cos(theta_n) * torch.sin(alpha_nm1)
    trans_matrices[:, 2, 2] = torch.cos(alpha_nm1)
    trans_matrices[:, 2, 3] = d_n * torch.cos(alpha_nm1)

    trans_matrices[:, 3, 3] = 1

    return trans_matrices


def quat_mul_batch(quaternion0, quaternion1):
    x0 = quaternion0[:, 0]
    y0 = quaternion0[:, 1]
    z0 = quaternion0[:, 2]
    w0 = quaternion0[:, 3]

    x1 = quaternion1[:, 0]
    y1 = quaternion1[:, 1]
    z1 = quaternion1[:, 2]
    w1 = quaternion1[:, 3]

    return torch.stack([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    ], dim=-1)
