import numpy as np

from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from trajectory.trajectory import Trajectory
from trajectory.joint_configuration_trajectory_state import JointConfigurationTrajectoryState


def hand_written_trajectory():
    ret = Trajectory[JointConfigurationTrajectoryState]()
    pt1 = JointConfigurationTrajectoryState(joint_configuration=np.array([1, 2, 3, 4, 5, 6, 7]),
                                            gripper_open=1,
                                            time=0.1)
    pt2 = JointConfigurationTrajectoryState(joint_configuration=np.array([10, 11, 12, 13, 14, 15, 16]),
                                            gripper_open=0,
                                            time=0.1)

    ret.add_point(pt1)
    ret.add_point(pt2)

    return ret


def load_from_file_trajectory():
    return Trajectory.load_points_from_file("./ReachTarget_trajectory_0.pth",
                                            JointConfigurationTrajectoryState)


if __name__ == '__main__':
    a = EncoderDeepProMP(3, [8, 7])
    b = DecoderDeepProMP(3, [7, 8])
    print(a)
    print(b)
    # tr1 = hand_written_trajectory()
    tr1 = load_from_file_trajectory()

    mu, sigma = a.encode_to_latent_variable(tr1)
    print("encoded mu", mu)
    print("encoded sigma", sigma)
    z = np.concatenate((np.array(mu), np.array(sigma)))
    tr2 = b.decode_from_latent_variable(z, tr1[0].time)
    print("decoded", tr2)
    print("original", tr1)