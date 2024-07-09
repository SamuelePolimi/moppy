import glob
import torch
from matplotlib import pyplot as plt
import numpy as np

files = glob.glob("./output/*/*/*/validation_loss.pth")
print(len(files))

lowest=[]
for f in files:
    arr = torch.load(f)
    lowest.append(arr[-1])

np_lowest = np.array(lowest)
data = np_lowest.reshape(5, 7, 6)

print(data)
exit()

data = np.mean(data, axis=2)

# Extract data for plotting
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a figure and 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter3D(x, y, z)  # blue colored circles

# Customize the plot (optional)
ax.set_xlabel('lr')
ax.set_ylabel('beta')
ax.set_zlabel('avg. validation')
ax.set_title('Hyperparameter optimization')

fig.savefig(f'./validation.png')