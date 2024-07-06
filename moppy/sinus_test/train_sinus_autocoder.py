import random
from typing import List
from matplotlib import pyplot as plt
import numpy
import torch
from moppy import SinusState, Trajectory, EncoderDeepProMP, DecoderDeepProMP
import torch.nn as nn
import torch.optim as optim


num_steps = 100


def generate_trajectory_set(n: int):
    ret = []
    for i in range(n):
        amplitude = random.randint(1, 10)
        frequency = random.randint(1, 5)
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


def test_train_sinus_encoder():

    encoder = EncoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    optimizer = optim.Adam(list(encoder.net.parameters()), lr=0.001)
    losses = []
    trajectory_set = generate_trajectory_set(50)
    validation_set = random.sample(trajectory_set, 5)
    training_set = [item for item in trajectory_set if item not in validation_set]

    for i in range(100):
        print(f"Step {i} - Training ... ", end="", flush=True)
        for traj in training_set:
            optimizer.zero_grad()

            amplitude = traj['amplitude']
            frequency = traj['frequency']
            traj = traj['traj']

            mu, sigma = encoder.encode_to_latent_variable(traj)
            Z = encoder.sample_latent_variable(mu, sigma).float()

            gen_amplitude = Z[0]
            gen_frequency = Z[1]
            loss = nn.MSELoss()(torch.tensor([amplitude + 0.0, frequency + 0.0], requires_grad=True).float(),
                                torch.tensor([gen_amplitude, gen_frequency], requires_grad=True).float()
                                )
            loss.backward()
            optimizer.step()

        print("Done - Validating ... ", end="", flush=True)

        validation_losses = []
        for traj in validation_set:
            amplitude = traj['amplitude']
            frequency = traj['frequency']
            traj = traj['traj']

            mu, sigma = encoder.encode_to_latent_variable(traj)
            Z = encoder.sample_latent_variable(mu, sigma).float()

            gen_amplitude = Z[0]
            gen_frequency = Z[1]
            loss = nn.MSELoss()(torch.tensor([amplitude + 0.0, frequency + 0.0], requires_grad=True).float(),
                                torch.tensor([gen_amplitude, gen_frequency], requires_grad=True).float()
                                )
            validation_losses.append(loss.item())
        losses.append(sum(validation_losses) / len(validation_losses))
        print("Done - Validation loss: ", losses[-1])
    print(losses)

    plt.plot(losses)
    plt.title('Loss during training of the encoder')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('train_encoder_msloss.png')


def train_decoder(decoder: DecoderDeepProMP, optimizer, iterations: int, amplitude_range: List[int], frequency_range: List[int]):
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
            new_point = decoder.decode_from_latent_variable(torch.tensor([amplitude, frequency]), point.get_time()).float()
            coked_traj.add_point(SinusState.from_vector_without_time(new_point))

        loss = nn.MSELoss()(traj.to_vector(), coked_traj.to_vector())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # same as detach().numpy()
        if i % 100 == 0:
            plot_trajs([traj, coked_traj], f'decoder/{i}', f" iter {i} - loss: {loss.item():.2f} - amp: {amplitude:.2f} - freq: {frequency:.2f}")

    plt.close()
    plt.plot(losses)
    plt.title('Loss during training of the encoder')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('train_decoder_msloss.png')
    return losses


def generate_amplitude_test(decoder: DecoderDeepProMP, iterations: int):
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
    plt.savefig('decoder_amp_test.png')


def test_train_sinus_decoder():

    decoder = DecoderDeepProMP(2, [10, 20, 20, 10], SinusState, nn.Tanh)
    optimizer = optim.Adam(list(decoder.net.parameters()), lr=0.001)
    iterations = 2000
    print(f"Training decoder(nn = {decoder.neurons}, iterations = {iterations}, activation function = {decoder.activation_function.__name__}) ...")
    train_decoder(decoder=decoder,
                  optimizer=optimizer,
                  iterations=iterations,
                  amplitude_range=[1, 10],
                  frequency_range=[1, 1])
    print("Done")
    print("Generating amplitude test ...", end="", flush=True)
    generate_amplitude_test(decoder=decoder,
                            iterations=iterations)
    print(" Done")


if __name__ == '__main__':
    test_train_sinus_encoder()
    # test_train_sinus_decoder()
