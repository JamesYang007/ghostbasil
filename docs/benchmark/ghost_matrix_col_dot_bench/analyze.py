import numpy as np
import matplotlib.pyplot as plt
import os

filepath = os.path.dirname(os.path.abspath(__file__))


def plot_test(ax, L, n_groups, p, ghost_times, naive_times, label1, label2):
    ax.plot(p, ghost_times, marker='.', ls='-', label=label1)
    ax.plot(p, naive_times, marker='.', ls='-', label=label2)
    ax.set_title(f"L={L}, Groups={n_groups}")
    ax.set_xlabel("Block rows/columns")
    ax.set_ylabel('Time (ns)')
    ax.legend()
    return fig, ax

fig, ax = plt.subplots(2, 2, figsize=(6, 6))

# Test 1
L = 10
p = np.array([10, 20, 30, 50, 80, 100, 200, 300])
n_groups = 2
ghost_times = np.array([
    45, 50, 51, 62, 71, 79, 120, 141
])
naive_times = np.array([
    7, 15, 21, 42, 91, 126, 236, 376
])
block_times = np.array([
    10, 10, 12, 15, 24, 24, 47, 69
])
plot_test(ax[0,0], L, n_groups, p, ghost_times, naive_times, "ghost_matrix", "dense_matrix")
plot_test(ax[0,1], L, n_groups, p, ghost_times, block_times, "ghost_matrix", "block_matrix")

# Test 2
L = 100
p = np.array([10, 20, 30, 50])
n_groups = 2
ghost_times = np.array([
    58, 69, 75, 89
])
naive_times = np.array([
    133, 248, 346, 617
])
plot_test(ax[1,0], L, n_groups, p, ghost_times, naive_times, "ghost_matrix", "dense_matrix")

p = np.concatenate([p, np.array([100, 500, 1000])])
ghost_times = np.concatenate([ghost_times,
                         np.array([127, 281, 463])])
block_times = np.array([
    15, 20, 29, 33, 43, 160, 394
])
plot_test(ax[1,1], L, n_groups, p, ghost_times, block_times, "ghost_matrix", "block_matrix")

fig.tight_layout()
plt.savefig(os.path.join(filepath, "fig.png"))
