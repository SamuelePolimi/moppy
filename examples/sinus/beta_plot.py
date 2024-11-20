from matplotlib import pyplot as plt
import torch
from datapoint import DataPoint

points: list[DataPoint] = torch.load('data_points.pt')
print('Loaded')


filtered = [p for p in points if p.lr == '2E-4' and p.af == 'relu' and p.ld == '5']
print('Filtered')
print(len(filtered))
print([float(p.get_mean()) for p in filtered])

filtered.sort(key=lambda p: float(p.beta))

plt.plot(
    [p.beta for p in filtered],
    [p.get_mean() for p in filtered],
    label="Beta",
    marker="H"
)

plt.title('Activation functions')
plt.xlabel('Latent dimention')
plt.ylabel('Validation error')
plt.grid(True)
plt.legend(loc='upper center')

# plt.show()
plt.savefig('./beta.png')
