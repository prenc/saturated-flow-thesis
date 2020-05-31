#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
from json import JSONDecodeError

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

PARAMS_PATH = "../params.h"
TEST_CONFIG_FILE = "./conf.json"


def load_data():
    heads = pd.read_csv('output/river/49.000000_1', sep=',').to_numpy()
    heads = np.delete(heads, -1, axis=1).astype(float)
    return np.delete(heads, -1, axis=1)


def plot_heads(heads, config):
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
    plt.title(config['river_position'])
    plt.show()


def read_config():
    try:
        with open(TEST_CONFIG_FILE, "r") as config_json:
            return json.load(config_json)['river']['test_specs']
    except FileNotFoundError:
        self._log.error(
            f"Could not find the config file. Make sure the file "
            f"'{TEST_CONFIG_FILE}' is in the script directory."
        )

if __name__ == '__main__':
    config = read_config()
    heads = load_data()
    plot_heads(heads,config)
