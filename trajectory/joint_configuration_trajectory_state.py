import numpy as np
from trajectory.trajectory_state import TrajectoryState


class JointConfigurationTrajectoryState(TrajectoryState):
    """A trajectory state that represents a joint configuration trajectory."""

    total_dimension: int = 9
    time_dimension: int = 1

    def __init__(self, joint_configuration: np.array, gripper_open: float, time: float) -> None:

        if gripper_open > 1 or gripper_open < 0:
            raise ValueError("gripper_open should be either 0 or 1.")

        super().__init__(time)
        self.joint_configuration = joint_configuration
        self.gripper_open = gripper_open

    @classmethod
    def from_vector_without_time(cls, vector: np.array, time: float = 0) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a vector without the time dimension."""
        if len(vector) != cls.total_dimension - cls.time_dimension:
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(vector)} != {cls.total_dimension - cls.time_dimension})")
        return cls(vector[:-1], vector[-1], time)

    @classmethod
    def from_vector(cls, vector: np.array) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a vector."""
        if len(vector) != cls.total_dimension:
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(vector)} != {cls.total_dimension})")
        return cls(vector[:-2], vector[-2], vector[-1])

    def to_vector(self) -> np.array:
        """Convert the state into a numpy array. So it can be used in a neural network."""
        ret = np.concatenate((self.joint_configuration, [self.gripper_open], [self.time]), dtype=np.float32)
        if len(ret) != self.get_dimensions():
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(ret)} != {self.get_dimensions()})")
        return ret

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"JointConfigurationTrajectoryState(joint_configuration={self.joint_configuration}, " \
               f"gripper_open={self.gripper_open}, time={self.time})"
