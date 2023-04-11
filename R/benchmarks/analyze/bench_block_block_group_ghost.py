import numpy as np
import matplotlib.pyplot as plt
import os


def analyze(baseline_data_path,
            new_version_data_path,
            fig_path):
        # check that n, p were the same in both versions
        n_baseline = np.genfromtxt(
            os.path.join(baseline_data_path, 'n.csv'),
            delimiter=',')
        n_new = np.genfromtxt(
            os.path.join(new_version_data_path, 'n.csv'),
            delimiter=',')
        assert(np.array_equal(n_baseline, n_new))
        n = n_baseline.astype(int)

        p_baseline = np.genfromtxt(
            os.path.join(baseline_data_path, 'p.csv'),
            delimiter=',')
        p_new = np.genfromtxt(
            os.path.join(new_version_data_path, 'p.csv'),
            delimiter=',')
        assert(np.array_equal(p_baseline, p_new))
        p = p_baseline.astype(int)

        L_baseline = np.genfromtxt(
            os.path.join(baseline_data_path, 'l.csv'),
            delimiter=',')
        L_new = np.genfromtxt(
            os.path.join(new_version_data_path, 'l.csv'),
            delimiter=',')
        assert(np.array_equal(L_baseline, L_new))
        L = L_baseline.astype(int)

        # average out the baseline and new_version since both do the same thing.
        bench_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'block_basil_times.csv'),
            delimiter=',') * 1e-9
        block = np.reshape(bench_times, newshape=(-1, n.size))

        bench_times = np.genfromtxt(
            os.path.join(new_version_data_path, 'block_basil_times.csv'),
            delimiter=',') * 1e-9
        block += np.reshape(bench_times, newshape=(-1, n.size))
        block *= 0.5

        bench_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'dense_basil_times.csv'),
            delimiter=',') * 1e-9
        dense = np.reshape(bench_times, newshape=(-1, n.size))

        bench_times = np.genfromtxt(
            os.path.join(new_version_data_path, 'dense_basil_times.csv'),
            delimiter=',') * 1e-9
        dense += np.reshape(bench_times, newshape=(-1, n.size))
        dense *= 0.5

        plt.plot(L, block, ls='-', label='BlockBlockGroupGhostMatrix')
        plt.plot(L, dense, ls='--', label='Dense')
        plt.title(f"BlockBlockGroupGhostMatrix vs Dense Basil (p={p})")
        plt.ylabel("Time (s)")
        plt.xlabel("Number of Blocks")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'block_block_group_ghost.png'))
