from typing import List

from abc import ABC, abstractmethod

from mp_types.types import LatentVariableZ


class LatentDecoder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def decode_from_latent_variable(self, z: List[LatentVariableZ], time: float):
        # In neural network environments, z is a torch.Tensor, but in other environments it could be a numpy array.
        pass
