from matplotlib import pyplot as plt
from moppy.deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from moppy.deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from moppy.trajectory.state.sinus_state import SinusState
from moppy.trajectory.trajectory import Trajectory

import torch

encoder = EncoderDeepProMP(3, [10, 10, 10, 10], SinusState)
encoder.load_model('../deep_promp/output')

decoder = DecoderDeepProMP(3, [10, 10, 10, 10], SinusState)
decoder.load_model('../deep_promp/output')
name = 'sin_25'
traj = Trajectory.load_points_from_file(f'./trajectories/{name}.pth', SinusState)

mu, sigma = encoder.encode_to_latent_variable(traj)
print(f"mu: {mu}")
print(f"sigma: {sigma}")
Z = encoder.sample_latent_variable(mu, sigma)
print(Z)
# Z = torch.normal(torch.zeros(3), torch.ones(3))
# print(Z)
steps = len(traj.get_points())
time = 0.0
out_traj = Trajectory()
for i in range(steps):
    value = decoder.decode_from_latent_variable(Z, time)
    out_traj.add_point(SinusState(value, traj.get_points()[i].get_time()))
    time += 1/steps

time_vector = torch.tensor([p.get_time()[0] for p in traj.get_points()])
value_vector = torch.tensor([p.value.item() for p in traj.get_points()])
plt.plot(time_vector, value_vector)

plt.title('Input sinus')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

time_vector = torch.tensor([p.get_time()[0] for p in out_traj.get_points()])
value_vector = torch.tensor([p.value.item() for p in out_traj.get_points()])

plt.plot(time_vector, value_vector)
plt.title('Output sinus')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Save the plot as an image file
plt.savefig(f'./sinus_trajectories_{name}.png')
