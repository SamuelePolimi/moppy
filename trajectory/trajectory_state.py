from abc import ABC
from abc import abstractmethod

import numpy as np


class TrajectoryState(ABC):
    """
    A trajectory state can describe anything that can be converted into a vector.
    It could be e.g. a vector of joint configurations or a vector of end effector positions and orientations.
    """
    total_dimension: int = -1
    time_dimension: int = -1

    def __init__(self, time):
        if not 0.0 <= time <= 1.0:
            raise ValueError("The time should be between 0.0 and 1.0, but got {}.".format(time))
        self.time = time

    @classmethod
    @abstractmethod
    def from_vector_without_time(cls, vector: np.ndarray, time: float) -> 'TrajectoryState':
        """Create a TrajectoryState from a vector without the time dimension."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @classmethod
    @abstractmethod
    def from_vector(cls, vector: np.ndarray) -> 'TrajectoryState':
        """Create a TrajectoryState from a vector."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """Convert the state into a numpy array. The array should be of shape (n,) where n is the dimension of the state."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @classmethod
    def get_dimensions(cls) -> int:
        """Get the total number of dimensions of the state."""
        return cls.total_dimension

    @classmethod
    def get_time_dimension(cls) -> int:
        """Get the number of dimensions that represent the time."""
        return cls.time_dimension
