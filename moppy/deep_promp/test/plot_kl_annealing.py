from moppy.deep_promp import plot_kl_annealing


if __name__ == '__main__':
    """
    Plot the KL annealing for different number of cycles.
    The function plot_kl_annealing can be found in moppy/deep_promp/utils.py.
    """
    plot_kl_annealing(steps=1000, n_cycles_values=[4, 16, 64])
