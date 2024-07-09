from moppy.deep_promp.deep_promp import DeepProMP
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 1000, 1000)

y = np.array([DeepProMP.kl_annealing_scheduler(t) for t in x])
plt.plot(x, y)
plt.show()
