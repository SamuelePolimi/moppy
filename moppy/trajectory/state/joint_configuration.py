import numpy as np
import torch

from . import TrajectoryState


class JointConfiguration(TrajectoryState):
    """A trajectory state that represents a joint configuration trajectory."""

    def __init__(self,
                 joint_positions: torch.Tensor | np.ndarray,
                 gripper_open: torch.Tensor | float,
                 time: torch.Tensor | float) -> None:
        super().__init__()
        if gripper_open > 1 or gripper_open < -1:
            raise ValueError("gripper_open should be either -1 or 1.")

        # make sure that all vars are torch tensors
        # because we want to use them in a neural network

        if isinstance(time, float | int):
            self.time = torch.tensor([time]).float()
        elif isinstance(time, np.ndarray):
            self.time = torch.from_numpy(time).float()
        elif isinstance(time, torch.Tensor):
            if time.dim() == 0:
                self.time = time.unsqueeze(0)
            else:
                self.time = time
        else:
            self.time = torch.tensor(time, dtype=torch.float)

        if isinstance(joint_positions, np.ndarray):
            self.joint_positions = torch.from_numpy(joint_positions).float()
        elif not isinstance(joint_positions, torch.Tensor):
            self.joint_positions = torch.tensor(joint_positions, dtype=torch.float)
        else:
            self.joint_positions = joint_positions

        if isinstance(gripper_open, np.ndarray):
            self.gripper_open = torch.from_numpy(gripper_open).float()
        elif not isinstance(gripper_open, torch.Tensor):
            self.gripper_open = torch.tensor(gripper_open, dtype=torch.float)
        else:
            self.gripper_open = gripper_open

    @classmethod
    def from_vector_without_time(cls, vector: np.ndarray | torch.Tensor, time: float = 0.) -> 'JointConfiguration':
        """Create a JointConfigurationTrajectoryState from a vector without the time dimension."""
        if len(vector) != cls.get_dimensions() - cls.get_time_dimension():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions() - cls.get_time_dimension()})")
        return cls(vector[:-1], vector[-1], time)

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> 'JointConfiguration':
        """Create a JointConfigurationTrajectoryState from a vector.
        The vector should be of shape (n,) where n is the total number of dimensions of the state.
        vector[:-2] should be the joint configuration,
        vector[-2] should be the gripper open value,
        vector[-1] should be the time."""
        if len(vector) != cls.get_dimensions():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions()})")
        return cls(vector[:-2], vector[-2], vector[-1])

    def to_vector_without_time(self) -> torch.Tensor:
        """Convert the state into a numpy array. So it can be used in a neural network."""
        # ret = np.concatenate((self.joint_positions, [self.gripper_open], [self.time]), dtype=np.float32)
        ret = torch.cat(
                        tensors=(
                            self.joint_positions,
                            torch.tensor([self.gripper_open], dtype=torch.float)),
        ).float()

        if len(ret) != self.get_dimensions() - self.get_time_dimension():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(ret)} != {self.get_dimensions() - self.get_time_dimension()})")
        return ret

    def get_time(self) -> torch.Tensor:
        return self.time

    @classmethod
    def from_dict(cls, data: dict) -> 'JointConfiguration':
        """Create a JointConfigurationTrajectoryState from a dictionary."""
        return cls(data["joint_positions"], data["gripper_open"], data["time"])

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"JointConfigurationTrajectoryState(joint_configuration={self.joint_positions}, " \
               f"gripper_open={self.gripper_open}, time={self.time})"

    @classmethod
    def get_dimensions(cls) -> int:
        """Get the total number of dimensions of the state."""
        return 9

    @classmethod
    def get_time_dimension(cls) -> int:
        """Get the number of dimensions that represent the time."""
        return 1


class JointConfiguration2(JointConfiguration):
    """A trajectory state that represents a joint configuration trajectory."""

    @classmethod
    def get_dimensions(cls) -> int:
        """Get the total number of dimensions of the state."""
        return 8
