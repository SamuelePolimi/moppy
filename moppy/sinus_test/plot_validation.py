import glob
import torch
from matplotlib import pyplot as plt
import numpy as np
from pylab import *

class DataPoint():
    lr: float
    beta: float
    validation: list[float]

    def __init__(self, lr, beta, validation_val) -> None:
        self.lr = lr
        self.beta = beta
        self.validation = [validation_val]

    def get_mean(self):
        return np.mean(self.validation)


def extract_values_from_path(filepath):
    values = {}
    for part in filepath.split("/")[2:-1]:  # Skip leading "./output"
        key, value = part.split("_")
        values[key] = float(value) if 'E' in value else int(value)
    return values

def load_values(folder: str, points: list[DataPoint]):
    files = glob.glob(folder)
    if not points:
        # Create new point list and populate
        points: list[DataPoint] = []
        for f in files:
            # Output: {'seed': 329, 'lr': 0.002, 'beta': 0.0002}
            extracted_values = extract_values_from_path(f) 
            validation_value = torch.load(f)[-1]
            # validation_value = min(torch.load(f))

            new_point = DataPoint(extracted_values['lr'], extracted_values['beta'], validation_value)
            points.append(new_point)
    else:
        # Add validation value to objects
        for i, f in enumerate(files):
            validation_value = torch.load(f)[-1]
            # validation_value = min(torch.load(f))
            points[i].validation.append(validation_value)

    return points


seeds = [12, 220, 329, 2304, 6064]
points: list[DataPoint] = None
for s in seeds:
    points = load_values("./output/seed_" + str(s) + "/*/*/validation_loss.pth", points)


# Extract data for plotting
x = np.array([p.lr for p in points])
y = np.array([p.beta for p in points])
z = np.array([p.get_mean() for p in points])

# creating figures 
fig = plt.figure(figsize=(10, 10)) 
ax = fig.add_subplot(111, projection='3d') 
  
# creating the heatmap 
#img = ax.scatter(x, y, z, marker='s', s=200, color='green') 

# adding title and labels 
ax.set_title("3D Heatmap")
ax.set_xlabel('lr')
ax.set_ylabel('beta')
ax.set_zlabel('validation')
surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=10)
fig.colorbar(surf)

# displaying plot 
plt.show()
fig.savefig(f'./validation.png')