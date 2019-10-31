from typing import List

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

SIMULATION_ITERATIONS = 1000

ROWS = 100
COLS = 100

LAYERS = 1
CELL_SIZE_X = 10
CELL_SIZE_Y = 10
THICKNESS = 50

Syinitial = 0.1
Kinitial = 0.0000125

headFixed = 50
headCalculated = 50

delta_t = 4000

qw = 0.001

convergence = ((CELL_SIZE_X * CELL_SIZE_Y) * Syinitial) / (Kinitial * 4)

posSx = int(COLS / 2)
posSy = int(ROWS / 2)


class Cell:
    def __init__(self, head=headFixed, K=Kinitial, Sy=Syinitial, source=0):
        self.head = head
        self.K = K
        self.Sy = Sy
        self.source = source


def main():
    read_ca = init_ca()

    for _ in range(SIMULATION_ITERATIONS):
        read_ca = simulation_step(read_ca)

    save_ca_to_file(read_ca)
    # visualize_ca(read_ca)


def simulation_step(read_ca):
    write_ca = read_ca.copy()
    for y in range(ROWS):
        if y == 0 or y == ROWS - 1:
            continue
        for x in range(COLS):
            transition_function(read_ca, write_ca, x, y)
    return write_ca


def visualize_ca(ca):
    v = np.array([[cell.head for cell in row] for row in ca])

    (x, y) = np.meshgrid(np.arange(v.shape[0]), np.arange(v.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x, y, v, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )

    ax.view_init(-10, 35)
    plt.show()


def save_ca_to_file(ca):
    with open("heads_with_wr_cas.txt", "w") as f:
        for row in ca:
            f.write(str([cell.head for cell in row]) + "\n")


def transition_function(read_ca, write_ca, x, y):
    neighbors = get_neighbor_cells(read_ca, x, y)

    cell = read_ca[y][x]
    Q = 0
    for neighbor in neighbors:
        diff_head = neighbor.head - cell.head
        tmp_t = neighbor.K * THICKNESS

        Q += diff_head * tmp_t

    Q -= cell.source
    area = CELL_SIZE_X * CELL_SIZE_Y
    ht1 = (Q * delta_t) / (area * cell.Sy)

    write_ca[y][x].head += ht1


def get_neighbor_cells(ca, x, y) -> List[Cell]:
    neighbors = []
    if y - 1 >= 0:
        neighbors.append(ca[y - 1][x])
    if y + 1 < ROWS:
        neighbors.append(ca[y + 1][x])
    if x - 1 >= 0:
        neighbors.append(ca[y][x - 1])
    if x + 1 < COLS:
        neighbors.append(ca[y][x + 1])
    return neighbors


def init_ca() -> List[List[Cell]]:
    ca = [[Cell() for _ in range(COLS)] for _ in range(ROWS)]
    ca[posSy][posSx].source = qw

    for y in range(ROWS):
        x = COLS - 1
        ca[y][x].head = headCalculated

    return ca


if __name__ == "__main__":
    main()
