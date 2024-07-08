import torch.nn as nn

from moppy.deep_promp.utils import set_seed
from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.deep_promp.deep_promp import DeepProMP
from moppy.trajectory.trajectory import Trajectory
from moppy.trajectory.state.sinus_state import SinusState




def load_from_file_trajectory():
    """Load the trajectories from the files"""
    tr = []
    for i in range(50):
        tr.append(Trajectory.load_points_from_file("trajectories/sin_%s.pth" % i, SinusState))

    return tr


def test_deep_pro_mp():
    encoder = EncoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    decoder = DecoderDeepProMP(2,  [10, 20, 20, 10], SinusState, nn.Tanh)
    deep_pro_mp = DeepProMP("deep_pro_mp", encoder, decoder)
    print(deep_pro_mp)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    set_seed(0)
    test_deep_pro_mp()
