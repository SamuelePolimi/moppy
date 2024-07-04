import numpy as np

from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.deep_promp.deep_promp import DeepProMP
from moppy.trajectory.trajectory import Trajectory
from moppy.trajectory.state.sinus_state import SinusState


def load_from_file_trajectory():
    """Load the trajectories from the files"""
    tr = []
    for i in range(50):
        tr.append(Trajectory.load_points_from_file("sinus_test/trajectories/sin_%s.pth" % i, SinusState))

    return tr


def test_deep_pro_mp():
    encoder = EncoderDeepProMP(3, [8, 7], SinusState)
    decoder = DecoderDeepProMP(3, [7, 8], SinusState)
    deep_pro_mp = DeepProMP("deep_pro_mp", encoder, decoder)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    test_deep_pro_mp()
