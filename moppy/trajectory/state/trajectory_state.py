from abc import ABC
from abc import abstractmethod

import numpy as np
import torch


class TrajectoryState(ABC):
    """
    A trajectory state can describe anything that can be converted into a vector.
    It could be e.g. a vector of joint configurations or a vector of end effector positions and orientations.
    """

    def __init__(self):
        if self.get_dimensions() < 0 or self.get_time_dimension() < 0 or self.get_time_dimension() >= self.get_dimensions():
            raise ValueError("Implement the get_dimensions and get_time_dimension methods of the subclass correctly.")

    @abstractmethod
    def to_vector_without_time(self) -> torch.Tensor:
        """Convert the state into a numpy array without time.
        The array should be of shape (n,) where n is the dimension of the state."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @abstractmethod
    def get_time(self) -> torch.Tensor:
        """Get the time of the state."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def to_vector_time(self) -> torch.Tensor:
        return torch.cat((self.to_vector_without_time(), self.get_time()), dim=0).float()

    @classmethod
    @abstractmethod
    def from_vector_without_time(cls, vector: np.ndarray, time) -> 'TrajectoryState':
        """Create a TrajectoryState from a vector without the time dimension."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @classmethod
    @abstractmethod
    def from_vector(cls, vector: np.ndarray) -> 'TrajectoryState':
        """Create a TrajectoryState from a vector."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> 'TrajectoryState':
        """Create a TrajectoryState from a dictionary."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @classmethod
    def get_dimensions(cls) -> int:
        """Get the total number of dimensions of the state."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    @classmethod
    def get_time_dimension(cls) -> int:
        """Get the number of dimensions that represent the time."""
        raise NotImplementedError("This method must be implemented by the subclass.")