from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.deep_promp import DeepProMP
from trajectory.state.trajectory_state import TrajectoryState
from trajectory.trajectory import Trajectory
from trajectory.state.joint_configuration import JointConfiguration


def load_from_file_trajectory():
    """Load the trajectories from the files"""
    tr = []
    for i in range(50):
        tr.append(Trajectory.load_points_from_file("deep_promp/ReachTarget/ReachTarget_trajectory_%s.pth" % i,
                                                   JointConfiguration))

    return tr


def test_deep_pro_mp():
    encoder = EncoderDeepProMP(3, [8, 7])
    decoder = DecoderDeepProMP(3, [7, 8])
    deep_pro_mp = DeepProMP("deep_pro_mp", encoder, decoder)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    test_deep_pro_mp()
