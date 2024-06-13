from abc import ABC, abstractmethod
from typing import List, Tuple

from trajectory.trajectory import Trajectory
from mp_types.types import LatentVariableZ


class LatentEncoder(ABC):

    @abstractmethod
    def encode_to_latent_variable(self, trajectory: Trajectory) -> List[LatentVariableZ]:
        pass
