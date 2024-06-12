import torch
from abc import ABC, abstractmethod


class DecoderProMP(ABC):
    def __init__(self, stroke):
        self.stroke = stroke

    @abstractmethod
    def generate_configuration(self, z: torch.Tensor, time: float):
        pass
