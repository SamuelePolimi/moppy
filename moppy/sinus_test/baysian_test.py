from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.trajectory.state.sinus_state import SinusState
import torch

encoder = EncoderDeepProMP(2, [10, 10], SinusState)

means = torch.FloatTensor([1.0, 100.0, 1.0, 100.0])
valiances = torch.FloatTensor([1.0, 100.0, 1.0, 100.0])

print(encoder.bayesian_aggregation(means, valiances))
