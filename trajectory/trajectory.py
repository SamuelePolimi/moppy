import os

from typing import Generic, List, TypeVar, Type

import numpy as np
import torch

from trajectory.trajectory_state import TrajectoryState

T = TypeVar('T', bound=TrajectoryState)


class Trajectory(Generic[T]):
    """
    A trajectory is a set of T, T has a nomralized timestep and the information of a Trajectory Point
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

    @classmethod
    def load_points_from_file(cls, file_path: str, trajectory_state_class: Type[T]):
        """Load the trajectory points from a file."""
        # TODO: This is still just a test, this should be implemented in the future, DOES NOT WORK CORRECTLY
        if not os.path.isfile(file_path):
            raise ValueError(f"File '{file_path}' does not exist.")
        content = torch.load(file_path)
        ret = cls[trajectory_state_class]()
        for i, vec in enumerate(content):
            vec = np.concatenate((vec, [1, i * 0.01]))
            item = trajectory_state_class.from_vector(vec)
            cls.add_point(ret, item)
        return ret

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, i) -> T:
        return self.trajectory[i]
