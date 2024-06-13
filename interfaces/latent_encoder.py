from abc import ABC, abstractmethod

import numpy as np

from trajectory.trajectory import Trajectory


class LatentEncoder(ABC):

    @abstractmethod
    def encode_to_latent_variable(self, trajectory: Trajectory) -> np.array:
        """
        Encode a trajectory to a latent variable z, which is a vector of a fixed dimension, which is a hyperparameter.
        :param trajectory: a trajectory to encode
        :return: the sampled latent variable z
        """
        pass
