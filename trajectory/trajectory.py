from typing import Generic, List, TypeVar

from trajectory.trajectory_state import TrajectoryState

T = TypeVar('T', bound=TrajectoryState)


class Trajectory(Generic[T]):
    """
    A trajectory is a set of T, T has a nomralized timestep and the information of a Trajectory Point
    """

    def __init__(self):
        self.trajectory: List[T] = []

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

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, i) -> T:
        return self.trajectory[i]
