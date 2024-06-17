import numpy as np
from trajectory.trajectory_state import TrajectoryState


class JointConfigurationTrajectoryState(TrajectoryState):
    """A trajectory state that represents a joint configuration trajectory."""

    total_dimension: int = 9
    time_dimension: int = 1

    def __init__(self, joint_configuration: np.ndarray, gripper_open: float, time: float) -> None:

        if gripper_open not in [0, 1]:
            raise ValueError("gripper_open should be either 0 or 1.")

        super().__init__(time)
        self.joint_configuration = joint_configuration
        self.gripper_open = gripper_open

    @classmethod
    def from_vector_without_time(cls, vector: np.ndarray[float], time: float = 0) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a vector without the time dimension."""
        return cls(vector[:-1], vector[-1], time)

    def to_vector(self) -> np.ndarray:
        """Convert the state into a numpy array. So it can be used in a neural network."""
        ret = np.array(self.joint_configuration + [self.gripper_open] + [self.time], dtype=np.float32)
        if len(ret) != self.get_dimensions():
            raise ValueError("The length of the vector should be equal to the number of dimensions.")
        return ret

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"JointConfigurationTrajectoryState(joint_configuration={self.joint_configuration}, " \
               f"gripper_open={self.gripper_open}, time={self.time})"
