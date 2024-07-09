import glob
import torch
from matplotlib import pyplot as plt
import numpy as np
from pylab import *

files = glob.glob("./output/*/*/*/validation_loss.pth")
print(len(files))

lowest=[]
for f in files:
    arr = torch.load(f)
    lowest.append((arr[-1], f))

minl = min([v[0] for v in lowest])
for i in lowest:
    if i[0] == minl:
        print(i)
exit()
np_lowest = np.array(lowest)
data = np_lowest.reshape(5, 7, 6)

# print(data)

data = np.mean(data, axis=2)

# Extract data for plotting
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# creating figures 
fig = plt.figure(figsize=(10, 10)) 
ax = fig.add_subplot(111, projection='3d') 
  
# setting color bar 
color_map = cm.ScalarMappable(cmap=cm.Greens_r) 
color_map.set_array(np_lowest) 
  
# creating the heatmap 
img = ax.scatter(x, y, z, marker='s', 
                 s=200, color='green') 
plt.colorbar(color_map) 
  
# adding title and labels 
ax.set_title("3D Heatmap") 
ax.set_xlabel('X-axis') 
ax.set_ylabel('Y-axis') 
ax.set_zlabel('Z-axis') 
  
# displaying plot 
plt.show() 
fig.savefig(f'./validation.png')