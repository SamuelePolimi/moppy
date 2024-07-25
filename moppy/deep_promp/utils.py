from typing import List, Tuple

import random
import numpy as np
import torch

from matplotlib import pyplot as plt

from moppy.trajectory import Trajectory
from moppy.trajectory.state import SinusState


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


def plot_trajectories(labeled_trajectories: List[Tuple[Trajectory, str]], file_name: str, plot_title: str) -> None:
    """Plot the trajectories and save the plot to a file. The trajectories are labeled with the given labels.

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
