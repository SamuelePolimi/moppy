

import torch
from . import TrajectoryState


class SinusState(TrajectoryState):

    def __init__(self, value: float, time: float):
        super().__init__()
        if isinstance(value, float):
            self.value = torch.tensor([value]).float()
        elif isinstance(value, torch.Tensor):
            self.value = value
        else:
            self.value = torch.tensor(value, dtype=torch.float)

        if isinstance(time, float):
            self.time = torch.tensor([time]).float()
        elif isinstance(time, torch.Tensor):
            self.time = time
        else:
            self.time = torch.tensor(time, dtype=torch.float)

        # print(f"value={self.value}, time={self.time}")

    def to_vector_without_time(self) -> torch.Tensor:
        ret = self.value
        if len(ret) != self.get_dimensions() - self.get_time_dimension():
            raise ValueError(
                f"The length of the vector should be equal to the number of dimensions."
                f"({len(ret)} != {self.get_dimensions() - self.get_time_dimension()})")
        return ret

    @classmethod
    def from_vector_without_time(cls, vector: torch.Tensor | float, time: float = 0.) -> 'SinusState':
        """Create a SinusTrajectoryState from a vector without the time dimension."""
        return cls(vector, time)

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> 'SinusState':
        """Create a SinusTrajectoryState from a vector."""
        return cls(vector[0], vector[-1])

    @classmethod
    def from_dict(cls, data: dict) -> 'SinusState':
        """Create a SinusTrajectoryState from a dictionary."""
        return cls(value=data['value'], time=data['time'])

    @classmethod
    def get_dimensions(cls) -> int:
        return 2

    @classmethod
    def get_time_dimension(cls) -> int:
        return 1

    def get_time(self) -> torch.Tensor:
        return self.time

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Sinus(value={self.value}, time={self.time})"
