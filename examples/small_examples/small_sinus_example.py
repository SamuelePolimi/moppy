from typing import List, Tuple

import random
import os
import numpy as np
import torch

from matplotlib import pyplot as plt


# Moppy imports
from moppy.deep_promp import DeepProMP, EncoderDeepProMP, DecoderDeepProMP
from moppy.trajectory.state import SinusState
from moppy.trajectory.trajectory import Trajectory

num_steps = 100
save_path = './results_small_sinus_example/'


def plot_trajectories(labeled_trajectories: List[Tuple[Trajectory, str]], file_name: str, plot_title: str):
    plt.close()
    for i, (traj, label) in enumerate(labeled_trajectories):
        plt.plot(traj.to_vector().detach().numpy(), label=label)
        plt.title(plot_title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend(loc='upper center')

    plt.savefig(file_name)


def generate_sin_trajectory_set(n: int) -> List[dict]:
    ret = []
    for i in range(n):
        # Generate random amplitude and frequency for the sinusoidal trajectory
        amplitude = random.uniform(1, 10)
        frequency = random.uniform(1, 1)  # To limit the training time, we only use frequency of 1
        traj = generate_sin_trajectory(amplitude, frequency)
        ret.append(traj)
    return ret


def generate_sin_trajectory(amplitude, frequency):
    """Generate a sinusoidal trajectory, given the amplitude and frequency, and return it as a Trajectory object."""
    traj = Trajectory()
    time = 0.0
    for _ in range(num_steps + 1):
        sin_val = amplitude * np.sin(frequency * time * 2 * np.pi)
        traj.add_point(SinusState(value=sin_val, time=time))
        time += 1/num_steps
    return traj


def main():
    print("Starting the DeepProMP example...\n")
    # Define the DecoderDeepProMP
    decoder = DecoderDeepProMP(latent_variable_dimension=2,
                               hidden_neurons=[10, 20, 30, 20, 10],
                               trajectory_state_class=SinusState)

    # Define the EncoderDeepProMP
    encoder = EncoderDeepProMP(latent_variable_dimension=2,
                               hidden_neurons=[10, 20, 30, 20, 10],
                               trajectory_state_class=SinusState)

    # Define the DeepProMP
    deep_pro_mp = DeepProMP(name="sinus_main",
                            encoder=encoder,
                            decoder=decoder,
                            learning_rate=0.005,  # Learning rate of the optimizer (Adam)
                            epochs=50,  # Number of epochs to train the model (each epoch is a full pass through the dataset)
                            save_path=save_path)  # Save path is where the model will be saved

    # Generate a set of sinusoidal trajectories
    traj_set = generate_sin_trajectory_set(50)

    # Train the DeepProMP
    deep_pro_mp.train(traj_set)

    ######################################################################################################################
    # Lets now test the trained model and try to autoencode a sinusoidal trajectory and compare it with the original one #
    ######################################################################################################################

    amplitude = 5
    frequency = 1  # We only trained the model with frequency of 1, change the generate_sin_trajectory_set function to train with different frequencies as well (will take more time to train)
    # Generate a sinusoidal trajectory
    traj = generate_sin_trajectory(amplitude, frequency)
    # Encode the trajectory to mu and sigma representation
    mu, sigma = encoder.encode_to_latent_variable(traj)
    # Sample from mu and sigma to get the latent variable
    latent_variable_z = encoder.sample_latent_variable(mu, sigma)

    autoencoded_trajectory = Trajectory()  # This is a empty trajectory that will contain the autoencoded trajectory

    time = 0.0  # The time of a trajectory has to be between 0 and 1.
    # Decode the latent variable to get the reconstructed trajectory
    for _ in range(num_steps + 1):
        state: torch.Tensor = decoder.decode_from_latent_variable(latent_variable=latent_variable_z, time=time)  # This return a tensor that contains the values of a sinus state (without time)
        sinus_state: SinusState = SinusState.from_vector_without_time(vector=state, time=time)  # Convert the tensor to a SinusState object, we dont generate the time because we input it into the decoder
        autoencoded_trajectory.add_point(sinus_state)  # Add the state to the trajectory object
        time += 1/num_steps  # Increase the time for the next point

    ###########################################################################
    # Now we can plot the original and autoencoded trajectory to compare them #
    ###########################################################################

    file_path = os.path.join(save_path, "Trajectories_Comparison.png")

    plot_trajectories(
        labeled_trajectories=[(traj, "generated Trajectory"), (autoencoded_trajectory, "autoencoded trajectory")],
        file_name=file_path,
        plot_title="Original vs Autoencoded Trajectory")

    print(f"Trajectories saved at {file_path}")
    print("Done!")


if __name__ == '__main__':
    main()
