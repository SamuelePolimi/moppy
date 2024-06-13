from abc import ABC, abstractmethod

import numpy as np


class LatentDecoder(ABC):

    @abstractmethod
    def decode_from_latent_variable(self, latent_variable: np.array, time: float):
        """

        :param latent_variable: A sampled latent variable z vector.
        :param time: A normalized time step between 0.0 and 1.0.
        :return: A normal distribution for each dimension of a TrajectoryState.
        """
        pass
