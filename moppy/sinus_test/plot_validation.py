import glob
from typing import List
import torch
from matplotlib import pyplot as plt
import numpy as np
from pylab import *

class DataPoint():
    lr: float
    beta: float
    validation: List[float]

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

# Example usage
files = glob.glob("./output/seed_12/*/*/validation_loss.pth")
points: List[DataPoint] = []
for f in files:
    extracted_values = extract_values_from_path(f) # Output: {'seed': 329, 'lr': 0.002, 'beta': 0.0002}
    validation_value = torch.load(f)[-1]

    new_point = DataPoint(extracted_values['lr'], extracted_values['beta'], validation_value)
    points.append(new_point)

files = glob.glob("./output/seed_220/*/*/validation_loss.pth")
for i, f in enumerate(files):
    validation_value = torch.load(f)[-1]
    points[i].validation.append(validation_value)

files = glob.glob("./output/seed_329/*/*/validation_loss.pth")
for i, f in enumerate(files):
    validation_value = torch.load(f)[-1]
    points[i].validation.append(validation_value)

files = glob.glob("./output/seed_2304/*/*/validation_loss.pth")
for i, f in enumerate(files):
    validation_value = torch.load(f)[-1]
    points[i].validation.append(validation_value)

files = glob.glob("./output/seed_6064/*/*/validation_loss.pth")
for i, f in enumerate(files):
    validation_value = torch.load(f)[-1]
    points[i].validation.append(validation_value)

# Extract data for plotting
x = [p.lr for p in points]
y = [p.beta for p in points]
z = [p.get_mean() for p in points]

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
ax.plot_trisurf(x, y, z, color="red", alpha=0.9)

# displaying plot 
plt.show() 
fig.savefig(f'./validation.png')