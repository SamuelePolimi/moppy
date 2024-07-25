import os
from typing import List
import torch

# Moppy imports
from moppy.deep_promp import (
    DeepProMP, EncoderDeepProMP, DecoderDeepProMP,
    plot_trajectories, generate_sin_trajectory, generate_sin_trajectory_set)
from moppy.trajectory.state import SinusState
from moppy.trajectory.trajectory import Trajectory

save_path = './results_small_sinus_example/'


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

    print(120 * "#")
    print("This example will train a DeepProMP with sinusoidal trajectories and then autoencode a sinusoidal trajectory:")
    print(f"\t - For this example we use the TrajectoryState class {decoder.trajectory_state_class.__name__}")
    print(f"\t - Encoder with {encoder.latent_variable_dimension} latent variables and {encoder.neurons} neurons")
    print(f"\t - Decoder with {decoder.latent_variable_dimension} latent variables and {decoder.neurons} neurons")
    print(f"\t - DeepProMP with {deep_pro_mp.epochs} epochs and learning rate of {deep_pro_mp.learning_rate}")
    print(120 * "#")

    # Generate a set of sinusoidal trajectories
    traj_set: List[Trajectory[SinusState]] = generate_sin_trajectory_set(50)

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

    autoencoded_trajectory = Trajectory[SinusState]()  # This is a empty trajectory that will contain the autoencoded trajectory

    # Decode the latent variable to get the reconstructed trajectory
    for point in traj.get_points():
        t = point.get_time()
        state: torch.Tensor = decoder.decode_from_latent_variable(latent_variable=latent_variable_z, time=t)  # This return a tensor that contains the values of a sinus state (without time)
        sinus_state: SinusState = SinusState.from_vector_without_time(vector=state, time=t)  # Convert the tensor to a SinusState object, we dont generate the time because we input it into the decoder
        autoencoded_trajectory.add_point(sinus_state)  # Add the state to the trajectory object

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
