from matplotlib import pyplot as plt
import torch
from datapoint import DataPoint

points: list[DataPoint] = torch.load('data_points.pt')
print('Loaded')

filtered = [p for p in points if p.lr == '2E-4' and p.beta == '2E-2']
print('Filtered')

def sort_func(p: DataPoint):
    return p.ld

filtered.sort(key=sort_func)

plt.plot(
    [float(p.ld) for p in filtered if p.af == 'relu'],
    [p.get_mean() for p in filtered if p.af == 'relu'],
    label="Relu"
)

plt.plot(
    [float(p.ld) for p in filtered if p.af == 'sigmoid'],
    [p.get_mean() for p in filtered if p.af == 'sigmoid'],
    label="Sigmoid"
)

plt.plot(
    [float(p.ld) for p in filtered if p.af == 'tanh'],
    [p.get_mean() for p in filtered if p.af == 'tanh'],
    label="Tanh"
)

plt.title('Activation functions')
plt.xlabel('Latent dimention')
plt.ylabel('Validation error')
plt.grid(True)
plt.legend(loc='upper center')

# plt.show()
plt.savefig('./activation_functions.png')
