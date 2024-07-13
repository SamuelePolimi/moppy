import numpy as np
import torch

from . import TrajectoryState


class Pose(TrajectoryState):
    def __init__(self,
                 position: torch.Tensor | np.ndarray,
                 orientation: torch.Tensor | np.ndarray,
                 time: torch.Tensor | float) -> None:
        super().__init__()

        if not torch.is_tensor(position):
            self.position = torch.tensor(position, dtype=torch.float)
        else:
            self.position = position

        if not torch.is_tensor(orientation):
            self.orientation = torch.tensor(orientation, dtype=torch.float)
        else:
            self.orientation = orientation

        if not torch.is_tensor(time):
            self.time = torch.tensor(time, dtype=torch.float)
        else:
            self.time = time


    @classmethod
    def from_vector_without_time(cls, vector: np.ndarray | torch.Tensor, time: float = 0.) -> 'Pose':
        if len(vector) != cls.get_dimensions() - cls.get_time_dimension():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions() - cls.get_time_dimension()})")
        return cls(vector[:-1], vector[-1], time)

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> 'Pose':
        if len(vector) != cls.get_dimensions():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(vector)} != {cls.get_dimensions()})")
        return cls(vector[:-1], vector[-1], vector[-2])

    @classmethod
    def from_dict(cls, data: dict) -> 'Pose':
        return cls(data['position'], data['orientation'], data['time'])

    def to_vector_without_time(self) -> torch.Tensor:
        return torch.cat((self.position, self.orientation)).float()