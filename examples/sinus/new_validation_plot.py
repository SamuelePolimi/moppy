import glob
import torch
from matplotlib import pyplot as plt
import numpy as np
from pylab import *

from datapoint import DataPoint

lrs = ['1E-2','5E-3','2E-3','1E-3','5E-4','1E-4','2E-4']
betas = ['2E-2','5E-2','1E-0','1E-2','25E-2','5E-1']
lds = ['4','5','2','3']
afs = ['tanh','sigmoid','relu']

def load_values(folder: str):
    points: list[DataPoint] = []
    for lr in lrs:
        for beta in betas:
            for ld in lds:
                for af in afs:
                    files = glob.glob(f"{folder}/lr_{lr}/beta_{beta}/ld_{ld}/af_{af}/*/validation_loss.pth")
                    validation_values = []
                    for f in files:
                        validation_values.append(torch.load(f)[-1])
                        # validation_values.append(min(torch.load(f)))

                    new_point = DataPoint(lr, beta, ld, af, validation_values)
                    points.append(new_point)

    return points

# points = load_values("../../moppy/sinus_test/output")
# torch.save(points, "data_points.pt")

points: list[DataPoint] = torch.load('data_points.pt')

filtered = [p for p in points if p.af == 'relu' and p.beta == '2E-2']

x = np.array([p.ld for p in filtered])
y = np.array([p.lr for p in filtered])
z = np.array([p.get_mean() for p in filtered])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Heatmap")
ax.set_xlabel('Latent dimention')
ax.set_ylabel('Learning rate')
ax.set_zlabel('validation')
surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=10)
fig.colorbar(surf)

# plt.show()
fig.savefig(f'./new_validation.png')