from typing import Generic, List, Tuple, TypeVar

from trajectory.trajectory_state import TrajectoryState

T = TypeVar('T', bound=TrajectoryState)


class Trajectory(Generic[T]):
    """
    A trajectory is a set of tuples (t_i, x_i) where t_i is a normalized time step (between 0.0 and 1.0)
    and x_i is a trajectory state at time t_i.
    """

    def __init__(self):
        self.trajectory: List[Tuple[float, T]] = []

    def __str__(self) -> str:
        return f"TJ: {self.trajectory}"

    def add_point(self, time_step: float, state: T):
        """Add a point to the trajectory."""
        if not 0.0 <= time_step <= 1.0:
            raise ValueError(
                "The time step should be between 0.0 and 1.0, but got {}.".
                format(time_step))

        # if the trajectory has a last point, check if the trajectory state dimensions are the same
        if len(self.trajectory) > 0:
            if not state.get_dimensions() == self.trajectory[-1][1].get_dimensions():
                raise ValueError("The trajectory state dimensions must be the same for all points in the trajectory.")

        self.trajectory.append((time_step, state))

    def get_points(self) -> List[Tuple[float, T]]:
        """Return the trajectory as a list of tuples."""
        return self.trajectory

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, i) -> Tuple[float, T]:
        return self.trajectory[i]
