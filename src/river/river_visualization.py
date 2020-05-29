#!/usr/bin/env python3
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def load_data():
    heads = pd.read_csv('output/river_49.000000_1', sep=',').to_numpy()
    heads = np.delete(heads, -1, axis=1).astype(float)
    return np.delete(heads, -1, axis=1)

def plot_heads(heads):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(1, 100, 1)
    Y = np.arange(1, 100, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, heads, cmap=cm.coolwarm_r,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    heads = load_data()
    plot_heads(heads)