from typing import List, Tuple

import random
import numpy as np
import torch

from matplotlib import pyplot as plt

from moppy.trajectory import Trajectory
from moppy.trajectory.state import SinusState
from moppy.deep_promp import DeepProMP


def set_seed(seed) -> None:
    """Set the seed for reproducibility of the experiments.

    Args:
        seed (int): The seed to set.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_trajectories(labeled_trajectories: List[Tuple[Trajectory[SinusState], str]], file_name: str, plot_title: str) -> None:
    """
    Plot the trajectories and save the plot to a file. The trajectories are labeled with the given labels.
    This works for SinusState trajectories.

    Args:
        labeled_trajectories (List[Tuple[Trajectory, str]]):  List of tuples containing the trajectory and the label. Example: [(traj1, 'label1'), (traj2, 'label2')]
        file_name (str): The name of the file where the plot will be saved. Example: 'plot.png'
        plot_title (str): The title of the plot. Example: 'Trajectories plot'

    Returns:
        None
    """
    plt.close()
    for i, (traj, label) in enumerate(labeled_trajectories):
        plt.plot(traj.to_vector().detach().numpy(), label=label)
        plt.title(plot_title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend(loc='upper center')

    plt.savefig(file_name)


def generate_sin_trajectory(amplitude: float | int, frequency: float | int, num_steps: int = 100) -> Trajectory:
    """Generate a sinusoidal trajectory, given the amplitude and frequency, and return it as a Trajectory object. The trajectory has num_steps points.
    The time is normalized to be in the range [0, 1].

    Args:
        amplitude (float | int): The amplitude of the sinusoidal trajectory.
        frequency (float | int): The frequency of the sinusoidal trajectory.
        num_steps (int, optional): The number of points in the trajectory. Defaults to 100.

    Returns:
        Trajectory: The generated sinusoidal trajectory, with num_steps points.
    """
    traj: Trajectory = Trajectory()
    t = 0.0
    for _ in range(num_steps):
        sin_val = amplitude * np.sin(frequency * t * 2 * np.pi)
        traj.add_point(SinusState(value=sin_val, time=t))
        t += 1/(num_steps - 1)
    return traj


def generate_sin_trajectory_set(n: int,
                                amplitude_range: Tuple[int, int] = (1, 10),
                                frequency_range: Tuple[int, int] = (1, 1)) -> List[Trajectory]:
    """Generate a set of n sinusoidal trajectories with random amplitude and frequency.
    The default amplitude range is [1, 10] and the default frequency range is [1, 1].
    The random numbers are not integers but floats in the given range.

    Args:
        n (int): The number of trajectories to generate.
        amplitude_range (Tuple[int, int], optional): The range of the amplitude of the sinusoidal trajectory. Defaults to (1, 10).
        frequency_range (Tuple[int, int], optional): The range of the frequency of the sinusoidal trajectory. Defaults to (1, 1).

    Returns:
        List[Trajectory]: The list of generated sinusoidal trajectories.
    """
    ret = []
    for i in range(n):
        # Generate random amplitude and frequency for the sinusoidal trajectory
        amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
        frequency = random.uniform(frequency_range[0], frequency_range[1])  # To limit the training time, we only use frequency of 1
        traj = generate_sin_trajectory(amplitude, frequency)
        ret.append(traj)
    return ret


def generate_sin_trajectory_set_labeled(n: int,
                                        amplitude_range: Tuple[int, int] = (1, 10),
                                        frequency_range: Tuple[int, int] = (1, 1)) -> List[dict]:
    """Generate a set of n sinusoidal trajectories with random amplitude and frequency.
    The default amplitude range is [1, 10] and the default frequency range is [1, 1].
    The random numbers are not integers but floats in the given range.

    Args:
        n (int): The number of trajectories to generate.
        amplitude_range (Tuple[int, int], optional): The range of the amplitude of the sinusoidal trajectory. Defaults to (1, 10).
        frequency_range (Tuple[int, int], optional): The range of the frequency of the sinusoidal trajectory. Defaults to (1, 1).

    Returns:
        List[dict]: The list of generated sinusoidal trajectories with their amplitude and frequency. Example: [{'traj': traj1, 'amplitude': 5, 'frequency': 1},]
    """

    ret: List[dict] = []
    for i in range(n):
        # Generate random amplitude and frequency for the sinusoidal trajectory
        amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
        frequency = random.uniform(frequency_range[0], frequency_range[1])  # To limit the training time, we only use frequency of 1
        traj = generate_sin_trajectory(amplitude, frequency)
        ret.append({'traj': traj, 'amplitude': amplitude, 'frequency': frequency})
    return ret


def plot_kl_annealing(
        steps: int = 1_000,
        n_cycles_values: List[int] = [4],
        save_path: str = './kl_annealing_plot.png',
        show_plot: bool = True):
    """
        Plots the KL annealing schedule for given number of steps and cycles.

        Args:
            steps (int): The total number of steps for the annealing schedule. Default is 1,000.
            n_cycles_values (List[int]): A list of integers representing the number of cycles for the annealing schedule. Default is [4].
            save_path (str): The file path where the plot will be saved. Default is './kl_annealing_plot.png'.
            show_plot (bool): A flag to indicate whether to display the plot. Default is True.

        Returns:
            None

        Raises:
            ValueError: If n_cycles_values is an empty list or if steps is less than 1.

        Example:
            Plot (save and show) the KL annealing schedule for 1,000 steps with 4, 16, and 64 cycles:
            >>> plot_kl_annealing(steps=1000, n_cycles_values=[4, 16, 64])

            Plot and save the KL annealing schedule for 1,000 steps with 4 cycles, without showing the plot:
            >>> plot_kl_annealing(steps=1000, n_cycles_values=4, save_path='./kl_annealing_plot.png', show_plot=False)
    """
    if isinstance(n_cycles_values, int):
        n_cycles_values = [n_cycles_values]

    if len(n_cycles_values) == 0:
        raise ValueError('At least one value for n_cycles_values is required.')
    if steps < 1:
        raise ValueError('steps must be greater than 0.')

    fig, axs = plt.subplots(len(n_cycles_values), 1, figsize=(10, int(2 * len(n_cycles_values))))

    if len(n_cycles_values) == 1:
        # If only one value is given, the axs is not a list but a single axis
        axs = [axs]

    for ax, n_cycles in zip(axs, n_cycles_values):
        x = np.linspace(1, steps, steps)
        y = np.array([DeepProMP.kl_annealing_scheduler(t, n_cycles=n_cycles, max_epoch=steps) for t in x])
        ax.plot(x, y, label=f'n_cycles={n_cycles}')
        ax.legend(loc='lower right')  # Set the legend location to the lower right corner so its uniform for all plots

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot with high resolution
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
