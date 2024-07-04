from matplotlib import pyplot as plt
from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.trajectory.state.sinus_state import SinusState
from moppy.trajectory.trajectory import Trajectory

import torch

encoder = EncoderDeepProMP(3, [8, 7], SinusState)
encoder.load_model('deep_promp/output')

decoder = DecoderDeepProMP(3, [7, 8], SinusState)
decoder.load_model('deep_promp/output')

traj = Trajectory.load_points_from_file('./trajectories/sin_0.pth', SinusState)

mu, sigma = encoder.encode_to_latent_variable(traj)
Z = encoder.sample_latent_variable(mu, sigma)

steps = len(traj.get_points())
time = 0.0
out_traj = Trajectory()
for _ in steps:
    value = decoder.decode_from_latent_variable(Z, time)
    out_traj.add_point(SinusState.from_vector(value))
    time += 1/steps

plt.plot([p.value[0] for p in traj.get_points()])
plt.title('Input sinus')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

plt.plot([p.value[0] for p in out_traj.get_points()])
plt.title('Output sinus')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Save the plot as an image file
plt.savefig('./sinus_trajectories.png')