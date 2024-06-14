from abc import ABC
from abc import abstractmethod

import numpy as np


class TrajectoryState(ABC):
    """
    A trajectory state can describe anything that can be converted into a vector.
    It could be e.g. a vector of joint configurations or a vector of end effector positions and orientations.
    """

    def __init__(self, time):
        if not 0.0 <= time <= 1.0:
            raise ValueError("The time should be between 0.0 and 1.0, but got {}.".format(time))
        self.time = time

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """Convert the state into a numpy array. The array should be of shape (n,) where n is the dimension of the state."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def get_dimensions(self) -> int:
        """Return the dimension of the state. This should strictly be the length of the numpy array."""
        raise NotImplementedError("This method should be implemented by the subclass.")
