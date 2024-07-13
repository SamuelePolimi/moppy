import argparse
from typing import Union
import torch.nn as nn
import os

from moppy.deep_promp import DecoderDeepProMP, EncoderDeepProMP, DeepProMP
from moppy.trajectory import Trajectory
from moppy.trajectory.state import EndEffectorPose
from moppy.deep_promp.utils import set_seed

import matplotlib.pyplot as plt
import random

def load_trajectories():
    """Load the trajectories from the files"""
    tr = []
    for filename in os.listdir("trajectories/"):
        if not filename.endswith(".pth"):
            continue
        tr.append(Trajectory.load_points_from_file("trajectories/" + filename, EndEffectorPose))

    # shuffle the trajectories

    random.shuffle(tr)

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


def init_mp(args):
    encoder = EncoderDeepProMP(latent_variable_dimension=args.latent_var,
                               hidden_neurons=[128, 128],
                               trajectory_state_class=EndEffectorPose,
                               activation_function=get_activation_function(args.activation_func), )

    decoder = DecoderDeepProMP(latent_variable_dimension=args.latent_var,
                               hidden_neurons=[128, 128],
                               trajectory_state_class=EndEffectorPose,
                               activation_function=get_activation_function(args.activation_func), )

    if args.test_model:
        encoder.load_model('./deep_promp/output/')
        decoder.load_model('./deep_promp/output/')

    deep_pro_mp = DeepProMP(name="table_tennis",
                            encoder=encoder,
                            decoder=decoder,
                            learning_rate=args.learning_rate,
                            epochs=args.epochs,
                            beta=args.beta,
                            save_path=args.save_path)
    return deep_pro_mp


def test_model(mp):
    trajectories = load_trajectories()
    # get random trajectory
    trajectory = trajectories[-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point in trajectory.get_points():
        pos = point.position
        ax.scatter(pos[0], pos[1], pos[2], color='b')

    mu, sigma = mp.encoder.encode_to_latent_variable(trajectory)
    z = mp.encoder.sample_latent_variable(mu, sigma)

    step = 1.0 / len(trajectory.get_points())

    time = 0.0

    while time < 1.0:
        value = mp.decoder.decode_from_latent_variable(z, time).detach().numpy()
        ax.scatter(value[0], value[1], value[2], color='r')
        time += step

    plt.savefig("output.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--rnd_seed", type=int, help="random seed for experiment.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning_rate used by the adam optimizer.")
    parser.add_argument("--epochs", default=200, type=int, help="The amount of epochs used in the training.")
    parser.add_argument("--beta", default=1, type=float, help="The kl-divergence ratio.")
    parser.add_argument("--save_path", default='./deep_promp/output/', type=str,
                        help="The folder moppy will save your files.")
    parser.add_argument("--latent_var", default='4', type=int, help="The size of the latent var.")
    parser.add_argument("--activation_func", default='relu', type=str,
                        help="The activation function used in the network.")
    parser.add_argument("--test_model", default=False, type=bool, help="Test the model instead of training.")

    args = parser.parse_args()

    if args.rnd_seed is not None:
        set_seed(args.rnd_seed)
    else:
        set_seed(0)

    mp = init_mp(args)

    if args.test_model:
        test_model(mp)
    else:
        mp.train(load_trajectories())
