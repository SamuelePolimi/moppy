import random
from typing import List
from matplotlib import pyplot as plt
import numpy
import torch
from moppy import SinusState, Trajectory, EncoderDeepProMP, DecoderDeepProMP
import torch.nn as nn
import torch.optim as optim


num_steps = 100


def plot_trajs(trajs: List[Trajectory], file_name: str):
    plt.close()
    for i, traj in enumerate(trajs):
        plt.plot(traj.to_vector().detach().numpy())
        plt.title(file_name)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

    plt.savefig(f'{file_name}.png')


def get_sin_trajectory(amplitude, frequency):
    """Generate a sinusoidal trajectory, given the amplitude and frequency, and return it as a Trajectory object."""
    traj = Trajectory()
    time = 0.0
    for _ in range(num_steps + 1):
        sin_val = amplitude * numpy.sin(frequency * time * 2 * numpy.pi)
        traj.add_point(SinusState(value=sin_val, time=time))
        time += 1/num_steps
    return traj


def test_train_sinus_encoder():

    encoder = EncoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    optimizer = optim.Adam(list(encoder.net.parameters()), lr=0.001)
    losses = []
    losses_temp = []
    for i in range(10000):
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f"Step {i}")
        amplitude = random.randint(1, 10)
        frequency = random.randint(1, 1)

        mu, sigma = encoder.encode_to_latent_variable(get_sin_trajectory(amplitude, frequency))
        Z = encoder.sample_latent_variable(mu, sigma).float()

        gen_amplitude = Z[0]
        gen_frequency = Z[1]
        loss = nn.MSELoss()(torch.tensor([amplitude + 0.0, frequency + 0.0], requires_grad=True).float(),
                            torch.tensor([gen_amplitude, gen_frequency], requires_grad=True).float()
                            )
        loss.backward()
        optimizer.step()
        losses_temp.append(loss.item())
        if i % 100 == 0 and i != 0:
            # calculate the avarage loss of the last 100 steps
            losses.append(sum(losses_temp) / len(losses_temp))
            losses_temp = []

    plt.plot(losses)
    plt.title('Loss during training of the encoder')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('train_encoder_msloss.png')


def test_train_sinus_decoder():

    decoder = DecoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    optimizer = optim.Adam(list(decoder.net.parameters()), lr=0.001)
    losses = []
    for i in range(10000 + 1):
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f"Step {i}")
        amplitude = random.randint(1, 10)
        frequency = random.randint(1, 3)
        traj = get_sin_trajectory(amplitude, frequency)
        coked_traj = Trajectory()
        for point in traj.get_points():
            new_point = decoder.decode_from_latent_variable(torch.tensor([amplitude, frequency]), point.get_time()).float()
            coked_traj.add_point(SinusState.from_vector_without_time(new_point))

        loss = nn.MSELoss()(traj.to_vector(), coked_traj.to_vector())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # same as detach().numpy()
        if i % 100 == 0:
            plot_trajs([traj, coked_traj], f'{i}')

    plt.close()
    plt.plot(losses)
    plt.title('Loss during training of the encoder')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('train_decoder_msloss.png')


if __name__ == '__main__':
    test_train_sinus_encoder()
    # test_train_sinus_decoder()
