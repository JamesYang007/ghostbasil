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
            os.path.join(baseline_data_path, 'group_lasso_low_groups_times.csv'),
            delimiter=',') * 1e-9
        bench_times = np.reshape(bench_times, newshape=(-1, n.size))
        bench2_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'group_lasso_low_groups_times.csv'),
            delimiter=',') * 1e-9
        bench2_times = np.reshape(bench_times, newshape=(-1, n.size))
        low_bench_times = 0.5 * (bench_times + bench2_times)

        bench_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'group_lasso_high_groups_times.csv'),
            delimiter=',') * 1e-6
        bench_times = np.reshape(bench_times, newshape=(-1, n.size))
        bench2_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'group_lasso_high_groups_times.csv'),
            delimiter=',') * 1e-6
        bench2_times = np.reshape(bench_times, newshape=(-1, n.size))
        high_bench_times = 0.5 * (bench_times + bench2_times)

        _, axes = plt.subplots(1, 2)
        axes[0].plot(p, low_bench_times, ls='-', marker='.')
        axes[0].set_title('$0.2p$ number of groups')
        axes[0].set_xlabel('p')
        axes[0].set_ylabel('Time (s)')
        axes[1].plot(p, high_bench_times, ls='-', marker='.')
        axes[1].set_title('$0.8p$ number of groups')
        axes[1].set_xlabel('p')
        axes[1].set_ylabel('Time (s)')
        plt.suptitle('Group Lasso Fit Time (100 $\lambda$ values until $0.5 \lambda_{\max}$)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, 'group_lasso.png'))
