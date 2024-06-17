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

    @abstractmethod
    def sample_latent_variable(self, mu: np.ndarray, sigma: np.ndarray, percentage_of_standard_deviation=None) -> np.array:
        """
        Sample a latent variable z from a normal distribution specified by mu and sigma.
        :param mu: the mean of the normal distribution
        :param sigma: the standard deviation of the normal distribution
        :param percentage_of_standard_deviation: the percentage of the standard deviation to sample from
        :return: the sampled latent variable z
        """
        pass
