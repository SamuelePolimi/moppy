from typing import Generic, List, Tuple, TypeVar

T = TypeVar('T', bound='TrajectoryState')


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

        self.trajectory.append((time_step, state))

    def get_trajectory(self) -> List[Tuple[float, T]]:
        """Return the trajectory as a list of tuples."""
        return self.trajectory

    def __len__(self):
        return len(self.trajectory)
