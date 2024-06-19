import numpy as np
import torch

from trajectory.trajectory_state import TrajectoryState


class JointConfigurationTrajectoryState(TrajectoryState):
    """A trajectory state that represents a joint configuration trajectory."""

    total_dimension: int = 9
    time_dimension: int = 1

    def __init__(self, joint_positions: np.array, gripper_open: float, time: float) -> None:

        if gripper_open > 1 or gripper_open < 0:
            raise ValueError("gripper_open should be either 0 or 1.")

        if isinstance(time, torch.Tensor):
            self.time = time.detach().numpy()
        else:
            self.time = time
        if isinstance(joint_positions, torch.Tensor):
            self.joint_positions = joint_positions.detach().numpy()
        else:
            self.joint_positions = np.array(joint_positions)
        if isinstance(gripper_open, torch.Tensor):
            self.gripper_open = gripper_open.detach().numpy()
        else:
            self.gripper_open = gripper_open

    @classmethod
    def from_vector_without_time(cls, vector: np.array, time: float = 0.) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a vector without the time dimension."""
        if len(vector) != cls.total_dimension - cls.time_dimension:
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(vector)} != {cls.total_dimension - cls.time_dimension})")
        return cls(vector[:-1], vector[-1], time)

    @classmethod
    def from_vector(cls, vector: np.array) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a vector.
        The vector should be of shape (n,) where n is the total number of dimensions of the state.
        vector[:-2] should be the joint configuration, vector[-2] should be the gripper open value and vector[-1] should be the time."""
        if len(vector) != cls.total_dimension:
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(vector)} != {cls.total_dimension})")
        return cls(vector[:-2], vector[-2], vector[-1])

    def to_vector(self) -> np.ndarray:
        """Convert the state into a numpy array. So it can be used in a neural network."""
        ret = np.concatenate((self.joint_positions, [self.gripper_open], [self.time]), dtype=np.float32)
        if len(ret) != self.get_dimensions():
            raise ValueError(f"The length of the vector should be equal to the number of dimensions.({len(ret)} != {self.get_dimensions()})")
        return ret

    @classmethod
    def from_dict(cls, data: dict) -> 'JointConfigurationTrajectoryState':
        """Create a JointConfigurationTrajectoryState from a dictionary."""
        return cls(data["joint_positions"], data["gripper_open"], data["time"])

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"JointConfigurationTrajectoryState(joint_configuration={self.joint_positions}, " \
               f"gripper_open={self.gripper_open}, time={self.time})"
