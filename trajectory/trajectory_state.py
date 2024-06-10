from abc import ABC
from abc import abstractmethod
import numpy as np


class TrajectoryState(ABC):
    """
    A trajectory state can describe anything that can be converted into a vector.
    It could be e.g. a vector of joint configurations or a vector of end effector positions and orientations.
    """

    @abstractmethod
    def to_np_array(self) -> np.ndarray:
        """Convert the state into a numpy array."""
        pass
