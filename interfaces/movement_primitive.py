from abc import ABC, abstractmethod
from typing import List

from trajectory.trajectory import Trajectory
from interfaces.latent_decoder import LatentDecoder
from interfaces.latent_encoder import LatentEncoder


class MovementPrimitive(ABC):
    """
    Movement primitives are a combination of an encoder and a decoder architecture.
    A movement primitive provides a distribution of trajectory states given a context trajectory and a normalized time step.
    """

    def __init__(self, name, encoder: LatentEncoder, decoder: LatentDecoder):
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

    def get_trajectory_distribution(self, context_trajectory: Trajectory, time: float):
        """
        :param context_trajectory: a context trajectory
        :param time: a normalized time step between 0.0 and 1.0
        :return: a normal distribution for each dimension of a TrajectoryState.
        """
        if context_trajectory is None:
            # note that the context trajectory can be empty!
            raise ValueError("The context trajectory should not be None")
        if not 0.0 <= time <= 1.0:
            raise ValueError("The time step should be between 0.0 and 1.0, but got {}.".format(time))

        z = self.encoder.encode_to_latent_variable(context_trajectory)
        return self.decoder.decode_from_latent_variable(z, time)

    @abstractmethod
    def train(self, trajectories: List[Trajectory]):
        pass
