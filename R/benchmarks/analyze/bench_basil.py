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

        # average out the baseline and new_version since both do the same thing.
        bench_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'basil_times.csv'),
            delimiter=',') * 1e-9
        baseline = np.reshape(bench_times, newshape=(-1, n.size))

        bench_times = np.genfromtxt(
            os.path.join(new_version_data_path, 'basil_times.csv'),
            delimiter=',') * 1e-9
        new_version = np.reshape(bench_times, newshape=(-1, n.size))

        plt.plot(p, baseline, ls='-', label='baseline')
        plt.plot(p, new_version, ls='--', label='new_version')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'basil.png'))
