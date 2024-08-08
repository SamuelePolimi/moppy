import numpy as np
import torch

from . import TrajectoryState


class EndEffectorPose(TrajectoryState):
    """
    A class that represents the pose of the end effector.
    The pose consists of a 3D position (XYZ), a quaternion orientation (XYZW) and a time.
    """
    def __init__(self,
                 position: torch.Tensor | np.ndarray,
                 orientation: torch.Tensor | np.ndarray,
                 time: torch.Tensor | float) -> None:
        super().__init__()
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("Position should be a 3D vector and orientation should be a quaternion. Given: "
                             f"position={position}, orientation={orientation}")

        if not torch.is_tensor(position):
            self.position = torch.tensor(position, dtype=torch.float)
        else:
            self.position = position

        if not torch.is_tensor(orientation):
            self.orientation = torch.tensor(orientation, dtype=torch.float)
        else:
            self.orientation = orientation

        # ensure that the quaternion is normalized
        if self.orientation[3] < 0:
            self.orientation = -self.orientation

        if not torch.is_tensor(time):
            self.time = torch.tensor([time], dtype=torch.float)
        else:
            self.time = time

    @classmethod
    def from_vector_without_time(cls, vector: np.ndarray | torch.Tensor, time: float = 0.) -> 'EndEffectorPose':
        if len(vector) != cls.get_dimensions() - cls.get_time_dimension():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions() - cls.get_time_dimension()})")
        return cls(vector[0:3], vector[3:], time)

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> 'EndEffectorPose':
        if len(vector) != cls.get_dimensions():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions()})")
        return cls(vector[0:3], vector[3:7], vector[7])

    @classmethod
    def from_dict(cls, data: dict) -> 'EndEffectorPose':
        return cls(data['position'], data['orientation'], data['time'])

    def to_vector_without_time(self) -> torch.Tensor:
        return torch.cat((self.position, self.orientation)).float()

    def get_time(self) -> torch.Tensor:
        return self.time

    @classmethod
    def get_dimensions(cls) -> int:
        return 3 + 4 + 1

    @classmethod
    def get_time_dimension(cls) -> int:
        return 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Pose(position={self.position}, orientation={self.orientation}, time={self.time})"
