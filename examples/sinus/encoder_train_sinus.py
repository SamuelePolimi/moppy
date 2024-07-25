
import random
import torch
import torch.optim as optim
import torch.nn as nn

from matplotlib import pyplot as plt

from moppy.trajectory.state import SinusState
from moppy.deep_promp import (
    EncoderDeepProMP,
    generate_sin_trajectory_set_labeled, set_seed)
from moppy.trajectory import Trajectory


set_seed(0)

img_folder = 'img/'


def test_train_sinus_encoder():

    encoder = EncoderDeepProMP(2, [10, 20, 20, 10], SinusState)
    optimizer = optim.Adam(list(encoder.net.parameters()), lr=0.005)
    losses = []
    trajectory_set = generate_sin_trajectory_set_labeled(50)
    validation_set = random.sample(trajectory_set, 5)
    training_set = [item for item in trajectory_set if item not in validation_set]
    iterations = 10
    for i in range(iterations):
        print(f"Step {i:02}/{iterations} - Training ... ", end="", flush=True)
        for traj in training_set:
            optimizer.zero_grad()

            amplitude: float = traj['amplitude']
            frequency: float = traj['frequency']
            traj: Trajectory[SinusState] = traj['traj']
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
