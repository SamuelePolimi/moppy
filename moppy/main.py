import argparse
from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.deep_promp import DeepProMP
from moppy.deep_promp.utils import set_seed
from trajectory.trajectory import Trajectory
from trajectory.state.joint_configuration import JointConfiguration


def load_from_file_trajectory():
    """Load the trajectories from the files"""
    tr = []
    for i in range(50):
        tr.append(Trajectory.load_points_from_file("deep_promp/ReachTarget/ReachTarget_trajectory_%s.pth" % i,
                                                   JointConfiguration))

    return tr


def test_deep_pro_mp(args):
    encoder = EncoderDeepProMP(3, [128, 128])
    decoder = DecoderDeepProMP(3, [128, 128])
    deep_pro_mp = DeepProMP(name="7_joint_reach_target",
                            encoder=encoder,
                            decoder=decoder,
                            learning_rate=args.learning_rate,
                            epochs=args.epochs,
                            beta=args.beta,
                            save_path=args.save_path)
    print(deep_pro_mp)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    test_deep_pro_mp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--rnd_seed", type=int, help="random seed for experiment.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="lerning_rate used by the adam optimizer.")
    parser.add_argument("--epochs", default=1000, type=int, help="The amout of epochs used in the training.")
    parser.add_argument("--beta", default=1, type=float, help="The kl-divergence ratio.")
    parser.add_argument("--save_path", default='./deep_promp/output/', type=str, help="The folder moppy will save your files.")

    args = parser.parse_args()

    if args.rnd_seed is not None:
        set_seed(args.rnd_seed)
    else:
        set_seed(0)

    test_deep_pro_mp(args)
