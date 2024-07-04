
import torch
import random
import numpy

traj = []
num_steps = 50

amplitude = random.randint(1, 10)
frequency = random.randint(1, 5)

for i in range(50):
    time = 0.0
    for _ in range(num_steps + 1):
        sin_val = amplitude * numpy.sin(frequency * time * 2 * numpy.pi)
        vec = {
                'value': sin_val.item(),
                'time': time
            }
        traj.append(vec)
        time += 1/num_steps

    torch.save(traj, f"trajectories/sin_{i}.pth")