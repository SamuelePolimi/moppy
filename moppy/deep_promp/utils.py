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

    Example:
        >>> traj = generate_sin_trajectory(amplitude=1,frequency=1,num_steps=100)
        >>> len(traj)
        100
        >>> traj = generate_sin_trajectory(amplitude=1,frequency=1)
        >>> len(traj)
        100
        >>> traj[0].get_time()
        tensor([0.])
        >>> traj[-1].get_time()
        tensor([1.])
    """
    traj: Trajectory = Trajectory()
    t = 0.0
    for _ in range(num_steps):
        sin_val = amplitude * np.sin(frequency * t * 2 * np.pi)
        traj.add_point(SinusState(value=sin_val, time=t))
        t += 1/(num_steps - 1)
    return traj


def generate_sin_trajectory_set(n: int) -> List[Trajectory]:
    ret = []
    for i in range(n):
        # Generate random amplitude and frequency for the sinusoidal trajectory
        amplitude = random.uniform(1, 10)
        frequency = random.uniform(1, 1)  # To limit the training time, we only use frequency of 1
        traj = generate_sin_trajectory(amplitude, frequency)
        ret.append(traj)
    return ret


if __name__ == '__main__':
    import doctest
    doctest.testmod()
