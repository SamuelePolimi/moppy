import argparse
from typing import Union
import torch.nn as nn
import os

from moppy.deep_promp import DecoderDeepProMP, EncoderDeepProMP, DeepProMP
from moppy.trajectory import Trajectory
from moppy.trajectory.state import EndEffectorPose
from moppy.deep_promp.utils import set_seed


def load_from_file_trajectory():
    """Load the trajectories from the files"""
    tr = []
    for filename in os.listdir("trajectories/"):
        if not filename.endswith(".pth"):
            continue
        tr.append(Trajectory.load_points_from_file("trajectories/" + filename, EndEffectorPose))
    return tr


def get_activation_function(ac_str: str) -> Union[nn.ReLU, nn.Sigmoid, nn.Tanh]:
    if ac_str == "relu":
        return nn.ReLU
    elif ac_str == "sigmoid":
        return nn.Sigmoid
    elif ac_str == "tanh":
        return nn.Tanh
    else:
        raise ValueError("Activation function not implemented.")


def test_deep_pro_mp(args):
    encoder = EncoderDeepProMP(latent_variable_dimension=args.latent_var,
                               hidden_neurons=[128, 128],
                               trajectory_state_class=EndEffectorPose,
                               activation_function=get_activation_function(args.activation_func), )

    decoder = DecoderDeepProMP(latent_variable_dimension=args.latent_var,
                               hidden_neurons=[128, 128],
                               trajectory_state_class=EndEffectorPose,
                               activation_function=get_activation_function(args.activation_func), )

    deep_pro_mp = DeepProMP(name="table_tennis",
                            encoder=encoder,
                            decoder=decoder,
                            learning_rate=args.learning_rate,
                            epochs=args.epochs,
                            beta=args.beta,
                            save_path=args.save_path)
    deep_pro_mp.train(load_from_file_trajectory())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--rnd_seed", type=int, help="random seed for experiment.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning_rate used by the adam optimizer.")
    parser.add_argument("--epochs", default=1000, type=int, help="The amount of epochs used in the training.")
    parser.add_argument("--beta", default=1, type=float, help="The kl-divergence ratio.")
    parser.add_argument("--save_path", default='./deep_promp/output/', type=str,
                        help="The folder moppy will save your files.")
    parser.add_argument("--latent_var", default='3', type=int, help="The size of the latent var.")
    parser.add_argument("--activation_func", default='relu', type=str,
                        help="The activation function used in the network.")
    args = parser.parse_args()

    if args.rnd_seed is not None:
        set_seed(args.rnd_seed)
    else:
        set_seed(0)

    test_deep_pro_mp(args)
