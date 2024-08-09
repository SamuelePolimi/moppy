import torch

from moppy.kid_promp.kid_promp import KIDPMP


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