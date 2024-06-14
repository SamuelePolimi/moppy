import numpy as np
from trajectory.trajectory_state import TrajectoryState


class JointConfigurationTrajectoryState(TrajectoryState):
    """A trajectory state that represents a joint configuration trajectory."""

    def __init__(self, joint_configuration: np.ndarray, gripper_open: float, time: float) -> None:

        if gripper_open not in [0, 1]:
            raise ValueError("gripper_open should be either 0 or 1.")

        super().__init__(time)
        self.joint_configuration = joint_configuration
        self.gripper_open = gripper_open

    def to_vector(self) -> np.ndarray:
        """Convert the state into a numpy array. So it can be used in a neural network."""
        ret = np.array(self.joint_configuration + [self.gripper_open] + [self.time], dtype=np.float32)
        if len(ret) != self.get_dimensions():
            raise ValueError("The length of the vector should be equal to the number of dimensions.")
        return ret

    def get_dimensions(self) -> int:
        """7 joint configuration dimensions + 1 gripper open dimension + 1 time dimension."""
        joint_configuration_dimensions = 7
        gripper_open_dimensions = 1
        time_dimensions = 1
        return joint_configuration_dimensions + gripper_open_dimensions + time_dimensions

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"JointConfigurationTrajectoryState(joint_configuration={self.joint_configuration}, " \
               f"gripper_open={self.gripper_open}, time={self.time})"
