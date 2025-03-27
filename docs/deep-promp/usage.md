---
title: Usage Example
created: 2024-11-02
last_updated: 2024-11-02
---

## Training DeepProMP on Sinusoidal Trajectories

This example demonstrates how to set up, train, and use the DeepProMP model on a set of sinusoidal trajectories using `moppy`.

## Overview

In this tutorial, we will:

1. Set up an encoder and decoder using DeepProMP.
2. Train DeepProMP on a sinusoidal trajectory set.
3. Autoencode a sinusoidal trajectory and compare the results with the original.

## Code Example

You can view the full code for this example in the [GitHub repository](https://github.com/SamuelePolimi/moppy/blob/main/examples/small_examples/small_sinus_example.py).

Below is the core code used for generating and training the model on sinusoidal trajectories.

```python
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
                            learning_rate=0.005,
                            epochs=50,
                            save_path=save_path)

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

    # Autoencode a sinusoidal trajectory and compare with the original
    amplitude, frequency = 5, 1
    traj = generate_sin_trajectory(amplitude, frequency)
    mu, sigma = encoder.encode_to_latent_variable(traj)
    latent_variable_z = encoder.sample_latent_variable(mu, sigma)
    autoencoded_trajectory = Trajectory[SinusState]()

    for point in traj.get_points():
        t = point.get_time()
        state = decoder.decode_from_latent_variable(latent_variable=latent_variable_z, time=t)
        sinus_state = SinusState.from_vector_without_time(vector=state, time=t)
        autoencoded_trajectory.add_point(sinus_state)

    # Plot the original and autoencoded trajectory
    file_path = os.path.join(save_path, "Trajectories_Comparison.png")
    plot_trajectories(
        labeled_trajectories=[(traj, "Generated Trajectory"), (autoencoded_trajectory, "Autoencoded Trajectory")],
        file_name=file_path,
        plot_title="Original vs Autoencoded Trajectory")

    print(f"Trajectories saved at {file_path}")
    print("Done!")

if __name__ == '__main__':
    main()
```

## Console Output

The output below is what you should expect in the terminal during execution. The training progress will show validation and training loss for each epoch.

```bash
Starting the DeepProMP example...

########################################################################################################################
This example will train a DeepProMP with sinusoidal trajectories and then autoencode a sinusoidal trajectory:
         - For this example we use the TrajectoryState class SinusState
         - Encoder with 2 latent variables and [2, 10, 20, 30, 20, 10, 4] neurons
         - Decoder with 2 latent variables and [3, 10, 20, 30, 20, 10, 1] neurons
         - DeepProMP with 50 epochs and learning rate of 0.005
########################################################################################################################
Start DeepProMP training ...
Total set:  50
Training set: 45
Validation set: 5
Epoch  1/50 (2.35s): validation loss = 14.3365, train_loss = 14.9515, mse = 14.9515, kl = 5.8765
...
Epoch 50/50 (2.23s): validation loss = 0.3477, train_loss = 0.5100, mse = 0.5100, kl = 5.7649
Training finished (Time = 134.91s).
Saving losses ...finished.
Plotting...finished.
Saving models...finished.
Trajectories saved at ./results_small_sinus_example/Trajectories_Comparison.png
Done!
```

## Resulting Files

After running the example, you will find the following files in the output directory (`./results_small_sinus_example/`):

```bash
.
├── all_losses.png
├── decoder_deep_pro_mp.pth
├── decoder_model_deep_pro_mp.pth
├── encoder_deep_pro_mp.pth
├── encoder_model_deep_pro_mp.pth
├── kl_loss.png
├── kl_loss.pth
├── mse_loss.pth
├── ms_loss.png
├── train_loss.png
├── train_loss.pth
├── Trajectories_Comparison.png
├── validation_loss.png
└── validation_loss.pth

```

Each file serves a purpose:

- **Model Files** (`encoder_deep_pro_mp.pth`, `decoder_deep_pro_mp.pth`): Contains trained model parameters for reuse.
- **Loss Files** (`train_loss.png`, `validation_loss.png`, `kl_loss.png`, etc.): Visualize training progress for key metrics, including MSE and KL divergence.
- **Combined Loss Plot** (`all_losses.png`): Consolidates all key loss metrics—training, validation, MSE, and KL divergence—into a single plot for easier monitoring.
- **Trajectory Comparison** (`Trajectories_Comparison.png`): Plot comparing the original and autoencoded trajectories to assess model performance.

## Visualizations

- **Trajectory Comparison**: The final plot (`Trajectories_Comparison.png`) displays the **original trajectory** and the **autoencoded trajectory** side by side. This comparison shows how accurately DeepProMP reconstructs sinusoidal patterns from latent representations.

  ![Trajectory Comparison](/assets/img/deep-promp/Trajectories_Comparison.png)

- **Loss Metrics**: The combined loss plot (`all_losses.png`) shows the training and validation loss progress over each epoch, along with MSE and KL divergence values. This comprehensive view helps assess the model's convergence and stability.

  ![All Losses](/assets/img/deep-promp/all_losses.png)

## Additional Notes

- **Custom Trajectories**: This example uses a fixed-frequency sinusoidal wave. To experiment with varying frequencies, you can modify the `generate_sin_trajectory_set` function.
- **Parameter Adjustments**: Adjusting the latent variable dimensions, learning rate, and network architecture (e.g., number of neurons per layer) can enhance model performance based on the trajectory patterns you wish to model.

## Example Code

You can view this example and access the full code in the [GitHub repository](https://github.com/SamuelePolimi/moppy/blob/main/examples/small_examples/small_sinus_example.py).

---

This guide provides an overview of DeepProMP’s capabilities using sinusoidal trajectories. Experiment with different configurations and trajectory shapes to fully leverage the model’s potential for trajectory-based tasks.
