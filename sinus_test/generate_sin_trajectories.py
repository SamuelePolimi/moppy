
import torch
import random
import numpy

traj = []
num_steps = 100
time = 0.0

amplitude = random.randint(1, 10)
frequency = random.randint(1, 5)

for i in range(50):
    for _ in range(num_steps + 1):
        vec = {
                'value': amplitude * numpy.sin(frequency * time),
                'time': time
            }
        traj.append(vec)
        time += 1/num_steps

    traj[-1]["time"] = 1.0

    torch.save(traj, f"trajectories/sin_{i}.pth")