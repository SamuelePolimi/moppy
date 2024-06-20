from abc import ABC, abstractmethod


class LatentDecoder(ABC):

    @abstractmethod
    def decode_from_latent_variable(self, latent_variable, time):
        """

        :param latent_variable: A sampled latent variable z vector.
        :param time: The time step.
        :return: A normal distribution for each dimension of a TrajectoryState.
        """
        pass
