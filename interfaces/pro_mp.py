from abc import ABC, abstractmethod
from typing import List

from trajectory.trajectory import Trajectory
from interfaces.decoder_pro_mp import DecoderProMP
from interfaces.encoder_pro_mp import EncoderProMP


class ProMPInterface(ABC):
    """ This class is an interface for the ProMP class. It defines
    the methods that a ProMP class should implement."""

    def __init__(self, name, encoder: EncoderProMP, decoder: DecoderProMP):
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def train(self, trajectories: List[Trajectory]):
        pass
