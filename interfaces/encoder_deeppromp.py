from abc import ABC, abstractmethod
from typing import List, Tuple

from classes.trajectory import Trajectory
from movement_primitives_types.types import LatentVariableZ


class EncoderProMP(ABC):

    @abstractmethod
    def generate_latent_variable(
            self,
            trajectory: Trajectory,
            context_trajectory: List[Trajectory]
            ) -> List[LatentVariableZ]:
        pass

    @abstractmethod
    def sample(self,
               mean: float,
               standard_deviation: float,
               percentage_of_standard_deviation: float = None
               ) -> Tuple[LatentVariableZ, float]:
        pass
