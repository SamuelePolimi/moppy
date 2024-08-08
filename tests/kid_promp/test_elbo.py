import torch

from moppy.kid_promp.kid_promp import KIDPMP


#@staticmethod
#def mse_pose(y_pred, y_star):
#    y_pred_pos = y_pred[:3]
#    y_pred_quat = y_pred[3:]
#    y_star_pos = y_star[:3]
#    y_star_quat = y_star[3:]

    # TODO use self.transpose and self.rotate to transform the pos and rot
    # TODO scale the pos with self.scale

#    mse_pos = nn.MSELoss()(y_pred_pos, y_star_pos)

    # Siciliano 3.91
    # self.eta * desired_uq.epsilon - desired_uq.eta * self.epsilon - np.cross(desired_uq.epsilon, self.epsilon)
#    e_o = y_star_quat[3] * y_pred_quat[:3] - y_pred_quat[3] * y_star_quat[:3] - torch.cross(y_star_quat[:3],
#                                                                                            y_pred_quat[:3], dim=-1)
#    mse_quat = nn.MSELoss()(e_o, torch.zeros_like(e_o))

#    return mse_pos + mse_quat

def test_elbo():
    pose = torch.randn(7)
    print(KIDPMP.elbo_batch(pose, pose, torch.tensor(0), torch.tensor(1), 0.0))
    pose_zoff = pose.clone()
    pose_zoff[2] += 0.1
    print(KIDPMP.elbo_batch(pose, pose_zoff, torch.tensor(0), torch.tensor(1), 0.0))

    pose_quat_inverted = pose.clone()
    pose_quat_inverted[3:] *= -1
    print(KIDPMP.elbo_batch(pose, pose_quat_inverted, torch.tensor(0), torch.tensor(1), 0.0))

    pose_quat_wrong = pose.clone()
    pose_quat_wrong[6] += 0.1
    print(KIDPMP.elbo_batch(pose, pose_quat_wrong, torch.tensor(0), torch.tensor(1), 0.0))

    multiple_poses = torch.randn(10, 7)
    print(KIDPMP.elbo_batch(multiple_poses, multiple_poses, torch.zeros(10), torch.ones(10), 0.0))

    multiple_poses_one_diff = multiple_poses.clone()
    multiple_poses_one_diff[0, 2] += 0.1
    print(KIDPMP.elbo_batch(multiple_poses, multiple_poses_one_diff, torch.zeros(10), torch.ones(10), 0.0))

test_elbo()