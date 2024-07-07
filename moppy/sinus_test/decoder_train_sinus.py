
import numpy
import random
import torch
import torch.optim as optim
import torch.nn as nn

from typing import List
from matplotlib import pyplot as plt

from moppy import SinusState, Trajectory, DecoderDeepProMP


num_steps = 100
img_folder = 'img/'


def generate_trajectory_set(n: int) -> List[dict]:
    ret = []
    for i in range(n):
        amplitude = random.uniform(1, 10)
        frequency = random.uniform(1, 1)
        traj = get_sin_trajectory(amplitude, frequency)
        ret.append({'traj': traj, 'amplitude': amplitude, 'frequency': frequency})
    return ret


def plot_trajs(trajs: List[Trajectory], file_name: str, title: str):
    plt.close()
    for i, traj in enumerate(trajs):
        plt.plot(traj.to_vector().detach().numpy())
        plt.title(title)
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


def __train_decoder(decoder: DecoderDeepProMP,
                    optimizer, iterations: int,
                    amplitude_range: List[int],
                    frequency_range: List[int]):
    losses = []
    for i in range(iterations + 1):
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f"{i}, ", end="", flush=True)
        amplitude = random.uniform(*amplitude_range)
        frequency = random.uniform(*frequency_range)
        traj = get_sin_trajectory(amplitude, frequency)
        coked_traj = Trajectory()
        for point in traj.get_points():
            new_point = decoder.decode_from_latent_variable(latent_variable=torch.tensor([amplitude, frequency]),
                                                            time=point.get_time()
                                                            ).float()
            coked_traj.add_point(SinusState.from_vector_without_time(new_point))

        loss = nn.MSELoss()(traj.to_vector(), coked_traj.to_vector())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # same as detach().numpy()
        if i % 100 == 0:
            plot_trajs(trajs=[traj, coked_traj],
                       file_name=f'img/decoder/{i}',
                       title=f" iter {i} - loss: {loss.item():.2f} - amp: {amplitude:.2f} - freq: {frequency:.2f}")

    plt.close()
    plt.plot(losses)
    plt.title(f'LOSS decoder, i: {iterations}, nn: {decoder.neurons}, activ func: {decoder.activation_function.__name__}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'{img_folder}/decoder_msloss.png')
    return losses


def __generate_amplitude_test(decoder: DecoderDeepProMP, iterations: int):
    plt.close()
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        amplitude = i + 1
        frequency = 1
        traj = get_sin_trajectory(amplitude, frequency)
        coked_traj = Trajectory()
        for point in traj.get_points():
            new_point = decoder.decode_from_latent_variable(torch.tensor([amplitude, frequency]), point.get_time()).float()
            coked_traj.add_point(SinusState.from_vector_without_time(new_point))

        ax.plot(traj.to_vector().detach().numpy())
        ax.plot(coked_traj.to_vector().detach().numpy())
        ax.set_title(f"Amp: {amplitude} - Freq: {frequency}")
        ax.grid(True)

    fig.suptitle(f'Decoder, i: {iterations}, nn: {decoder.neurons}, activ func: {decoder.activation_function.__name__} ')
    plt.tight_layout()
    plt.savefig(f'{img_folder}/decoder_amp_test.png')


def test_train_sinus_decoder():

    decoder = DecoderDeepProMP(2, [10, 20, 20, 10], SinusState, nn.Tanh)
    optimizer = optim.Adam(list(decoder.net.parameters()), lr=0.001)
    iterations = 2_000
    print(f"Training decoder(nn = {decoder.neurons}, iterations = {iterations},"
          f" activation function = {decoder.activation_function.__name__}) ...")
    __train_decoder(decoder=decoder,
                    optimizer=optimizer,
                    iterations=iterations,
                    amplitude_range=[1, 10],
                    frequency_range=[1, 1])
    print("Done")
    print("Generating amplitude test ...", end="", flush=True)
    __generate_amplitude_test(decoder=decoder,
                              iterations=iterations)
    print(" Done")


if __name__ == '__main__':
    print(f"Image folder: '{img_folder}'")
    test_train_sinus_decoder()
