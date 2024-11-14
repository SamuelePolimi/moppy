from moppy.deep_promp.deep_promp import DeepProMP
import matplotlib.pyplot as plt
import numpy as np

n_cycles_values = [4, 50, 100, 500, 1000]  # Array of different n_cycles values
i = 1000  # Fixed i value

fig, axs = plt.subplots(len(n_cycles_values), 1, figsize=(10, 15))

for ax, n_cycles in zip(axs, n_cycles_values):
    x = np.linspace(1, i, i)
    y = np.array([DeepProMP.kl_annealing_scheduler(t, n_cycles=n_cycles, max_epoch=i) for t in x])
    ax.plot(x, y, label=f'n_cycles={n_cycles}')
    ax.legend()

fig.suptitle(f'KL Annealing Schedules (max_epochs={i})', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
