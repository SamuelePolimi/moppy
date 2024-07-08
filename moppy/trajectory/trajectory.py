import os
import torch

from typing import Generic, List, TypeVar, Type

from moppy.trajectory.state.trajectory_state import TrajectoryState

T = TypeVar('T', bound=TrajectoryState)


class Trajectory(Generic[T]):
    """
    A trajectory is a set of T, T has a normalized timestep and the information of a Trajectory Point
    """

    def __init__(self, trajectory: List[T] = None):
        self.trajectory = []
        if trajectory is None:
            return
        for state in trajectory:
            self.add_point(state)

    def __str__(self) -> str:
        return f"TJ: {self.trajectory}"

    def add_point(self, state: T):
        """Add a point to the trajectory."""
        # if the trajectory has a last point, check if the trajectory state dimensions are the same
        if len(self.trajectory) > 0:
            if not state.get_dimensions() == self.trajectory[-1].get_dimensions():
                raise ValueError("The trajectory state dimensions must be the same for all points in the trajectory.")

        self.trajectory.append(state)

    def get_points(self) -> List[T]:
        """Return the trajectory as a list of tuples."""
        return self.trajectory

    def get_times(self) -> List[float]:
        """Return the times of the trajectory."""
        return [state.get_time() for state in self.trajectory]

    def to_vector(self) -> torch.Tensor:
        """Return the trajectory as a tensor. The tensor is a concatenation of the trajectory states."""
        return torch.cat([state.to_vector() for state in self.trajectory])

    @classmethod
    def load_points_from_file(cls, file_path: str, trajectory_state_class: Type[T]):
        """Load the trajectory points from a file. The file should be a torch file.
        The file should contain the trajectory points as a list of dictionaries."""

        if not os.path.isfile(file_path):
            raise ValueError(f"File '{file_path}' does not exist.")
        content = torch.load(file_path)

        # Normalize the time values to be between 0 and 1 for the trajectory
        time = normalize([v['time'] for v in content])
        for i, state in enumerate(content):
            state['time'] = time[i]

        # Create the trajectory from the content of the file
        ret = cls[trajectory_state_class]()
        for state in content:
            item = trajectory_state_class.from_dict(state)
            ret.add_point(item)
        return ret

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, i) -> T:
        return self.trajectory[i]


def normalize(data: List[float]) -> List[float]:
    min_value = min(data)
    max_value = max(data)
    return [(x - min_value) / (max_value - min_value) for x in data]
