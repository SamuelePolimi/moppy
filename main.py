import numpy as np
import torch

from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.deep_promp import DeepProMP
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
    """Load the trajectories from the files"""
    tr = []
    for i in range(50):
        tr.append(Trajectory.load_points_from_file("deep_promp/ReachTarget/ReachTarget_trajectory_%s.pth" % i,
                                            JointConfigurationTrajectoryState))

    return tr


def test_decoder_and_encoder():
    a = EncoderDeepProMP(3, [8, 7])
    b = DecoderDeepProMP(3, [7, 8])
    print(a)
    print(b)
    # tr1 = hand_written_trajectory()
    tr1 = load_from_file_trajectory()[0]
    print(tr1)

    mu, sigma = a.encode_to_latent_variable(tr1)
    print("encoded mu", mu)
    print("encoded sigma", sigma)
    z = np.concatenate((np.array(mu), np.array(sigma)))
    tr2 = b.decode_from_latent_variable(z, tr1[0].time)
    print("decoded", tr2)
    print("original", tr1)


def test_deep_pro_mp():
    encoder = EncoderDeepProMP(3, [8, 7])
    decoder = DecoderDeepProMP(3, [7, 8])
    deep_pro_mp = DeepProMP("deep_pro_mp", encoder, decoder)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    test_deep_pro_mp()
