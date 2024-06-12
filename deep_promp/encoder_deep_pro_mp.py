from typing import List, Tuple

from interfaces.encoder_pro_mp import EncoderProMP
from mp_types.types import LatentVariableZ
from trajectory.trajectory import Trajectory


class EncoderDeepProMP(EncoderProMP):

    def generate_latent_variable(
            self,
            trajectory: Trajectory,
            context_trajectory: List[Trajectory]
            ) -> List[LatentVariableZ]:
        raise NotImplementedError()

    def sample(self,
               mean: float,
               standard_deviation: float,
               percentage_of_standard_deviation: float = None
               ) -> Tuple[LatentVariableZ, float]:
        raise NotImplementedError()
