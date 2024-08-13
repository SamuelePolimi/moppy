import argparse

import torch
import torch.nn as nn
import os

from moppy.deep_promp import DecoderDeepProMP, EncoderDeepProMP, DeepProMP
from moppy.kid_promp.kid_decoder import DecoderKIDProMP
from moppy.kid_promp.kid_promp import KIDPMP
from moppy.trajectory import Trajectory
from moppy.trajectory.state import EndEffectorPose
from moppy.deep_promp.utils import set_seed

import matplotlib.pyplot as plt
import random
import json


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


def init_mp(args):
    encoder = EncoderDeepProMP(latent_variable_dimension=args.latent_var,
                               hidden_neurons=[128, 128],
                               trajectory_state_class=EndEffectorPose,
                               activation_function=nn.Softplus,
                               activation_function_params={"beta": 2.0})

    decoder = DecoderKIDProMP(latent_variable_dimension=args.latent_var,
                              hidden_neurons=[128, 128],
                              activation_function=nn.Softplus,
                              activation_function_params={"beta": 2.0},
                              dh_parameters_craig=json.load(open("dh_params.json")),
                              min_joints=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                              max_joints=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


    if args.test_model or args.interactive:
        encoder.load_model('./output/')
        decoder.load_model('./output/')

    deep_pro_mp = KIDPMP(name="table_tennis",
                            encoder=encoder,
                            decoder=decoder,
                            learning_rate=args.learning_rate,
                            epochs=args.epochs,
                            beta=args.beta,
                            save_path=args.save_path)
    return deep_pro_mp


def test_model(mp):
    trajectories = load_trajectories()
    index = 0
    for trajectory in trajectories:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point in trajectory.get_points():
            pos = point.position
            ax.scatter(pos[0], pos[1], pos[2], color='b')

        mu, sigma = mp.encoder.encode_to_latent_variable(trajectory)
        print(f"{mu} +- {sigma}")
        z = mp.encoder.sample_latent_variable(mu, sigma)

        step = 1.0 / len(trajectory.get_points())

        time = 0.0

        while time < 1.0:
            value = mp.decoder.decode_from_latent_variable(z, time).detach().numpy()
            ax.scatter(value[0], value[1], value[2], color='r')
            time += step

        plt.savefig("comparisons/output" + str(index) + ".png")
        plt.close()
        index += 1


def interactive(mp):
    #  3d plot with sliders for each latent variable component
    from matplotlib.widgets import Slider

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # init z randomly
    trajectory = load_trajectories()[0]
    mu, sigma = mp.encoder.encode_to_latent_variable(trajectory)
    z = mp.encoder.sample_latent_variable(mu, sigma)

    sliders = []

    def update(val):
        z = torch.tensor([slider.val for slider in sliders]).float()
        print(z)
        ax.clear()
        for point in trajectory.get_points():
            pos = point.position
            ax.scatter(pos[0], pos[1], pos[2], color='b')
        step = 0.025
        time = 0.0
        while time < 1.0:
            value = mp.decoder.decode_from_latent_variable(z, time).detach().numpy()
            ax.scatter(value[0], value[1], value[2], color='r')
            time += step

    for i in range(len(z)):
        axslider = fig.add_axes([0.1, 0.1 + i * 0.05, 0.8, 0.03])
        slider = Slider(axslider, f'z{i}', -1, 1, valinit=z[i].item())
        slider.on_changed(update)
        sliders.append(slider)

    update(-1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--rnd_seed", type=int, help="random seed for experiment.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning_rate used by the adam optimizer.")
    parser.add_argument("--epochs", default=900, type=int, help="The amount of epochs used in the training.")
    parser.add_argument("--beta", default=0.003, type=float, help="The kl-divergence ratio.") # 0.0025 for vanilla
    parser.add_argument("--save_path", default='./output/', type=str,
                        help="The folder moppy will save your files.")
    parser.add_argument("--latent_var", default='3', type=int, help="The size of the latent var.")
    parser.add_argument("--test_model", default=False, type=bool, help="Test the model instead of training.")
    parser.add_argument("--interactive", default=False, type=bool, help="Test the model instead of training.")

    args = parser.parse_args()

    if args.rnd_seed is not None:
        set_seed(args.rnd_seed)
    else:
        set_seed(3)

    mp = init_mp(args)

    if args.interactive:
        interactive(mp)
    elif args.test_model:
        test_model(mp)
    else:
        mp.train(load_trajectories(), kl_annealing=False)
