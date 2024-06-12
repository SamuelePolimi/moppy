import torch
from abc import ABC, abstractmethod


class DecoderProMP(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_configuration(self, z: torch.Tensor, time: float):
        pass
