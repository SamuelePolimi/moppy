
import numpy
import random
import torch
import torch.optim as optim
import torch.nn as nn

from typing import List
from matplotlib import pyplot as plt

from moppy.trajectory.state import SinusState
from moppy.deep_promp import EncoderDeepProMP
from moppy.trajectory import Trajectory


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(0)

points_per_trajectory = 100
img_folder = 'img/'


def generate_trajectory_set(n: int) -> List[dict]:
    ret = []
    for _ in range(n):
        amplitude = random.uniform(1, 10)
        frequency = random.uniform(1, 1)
        traj = get_sin_trajectory(amplitude, frequency)
        ret.append({'traj': traj, 'amplitude': amplitude, 'frequency': frequency})
    return ret


def get_sin_trajectory(amplitude, frequency):
    """Generate a sinusoidal trajectory, given the amplitude and frequency, and return it as a Trajectory object."""
    traj = Trajectory()
    time = 0.0
    for _ in range(points_per_trajectory + 1):
        sin_val = amplitude * numpy.sin(frequency * time * 2 * numpy.pi)
        traj.add_point(SinusState(value=sin_val, time=time))
        time += 1/points_per_trajectory
    return traj


def test_train_sinus_encoder():

    encoder = EncoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    optimizer = optim.Adam(list(encoder.net.parameters()), lr=0.005)
    losses = []
    trajectory_set = generate_trajectory_set(50)
    validation_set = random.sample(trajectory_set, 5)
    training_set = [item for item in trajectory_set if item not in validation_set]
    iterations = 10
    for i in range(iterations):
        print(f"Step {i:02}/{iterations} - Training ... ", end="", flush=True)
        for traj in training_set:
            optimizer.zero_grad()

            amplitude = traj['amplitude']
            frequency = traj['frequency']
            traj = traj['traj']
            mu, sigma = encoder.encode_to_latent_variable(traj)

            log_likelihood = torch.distributions.Normal(loc=mu, scale=torch.ones_like(sigma))\
                .log_prob(torch.tensor([amplitude, frequency])).sum()
            loss = -log_likelihood
            # This would be equivalent. I have removed the contribution of sigma, since it collapses to a deterministic
            # function in absence of a proper regularization
            # loss = nn.MSELoss()(torch.tensor([amplitude + 0.0, frequency + 0.0]).double(), mu)
            loss.backward()
            optimizer.step()

        print("Done - Validating ... ", end="", flush=True)

        validation_losses = []
        for traj in validation_set:
            amplitude = traj['amplitude']
            frequency = traj['frequency']
            traj = traj['traj']

            mu, sigma = encoder.encode_to_latent_variable(traj)
            # print(f"Mu: {mu} - Sigma: {sigma}")
            Z = encoder.sample_latent_variable(mu, sigma).float()

            loss = nn.MSELoss()(torch.tensor([amplitude + 0.0, frequency + 0.0]).float(), mu)
            validation_losses.append(loss.item())
        losses.append(sum(validation_losses) / len(validation_losses))
        print("Done - Validation loss: ", losses[-1])

    print(losses)

    plt.plot(losses)
    plt.title('validation loss of the encoder')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'{img_folder}/encoder_msloss.png')


if __name__ == '__main__':
    print(f"Image folder: '{img_folder}'")
    test_train_sinus_encoder()
