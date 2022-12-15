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
            os.path.join(baseline_data_path, 'group_lasso_times.csv'),
            delimiter=',') * 1e-9
        bench_times = np.reshape(bench_times, newshape=(-1, n.size))
        bench2_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'group_lasso_times.csv'),
            delimiter=',') * 1e-9
        bench2_times = np.reshape(bench_times, newshape=(-1, n.size))
        gl_bench_times = 0.5 * (bench_times + bench2_times)

        bench_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'lasso_times.csv'),
            delimiter=',') * 1e-9
        bench_times = np.reshape(bench_times, newshape=(-1, n.size))
        bench2_times = np.genfromtxt(
            os.path.join(baseline_data_path, 'lasso_times.csv'),
            delimiter=',') * 1e-9
        bench2_times = np.reshape(bench_times, newshape=(-1, n.size))
        l_bench_times = 0.5 * (bench_times + bench2_times)

        plt.plot(p, gl_bench_times, ls='-', marker='.', label='group-lasso')
        plt.plot(p, l_bench_times, ls='--', marker='.', label='lasso')
        plt.xlabel('p')
        plt.ylabel('Time (s)')
        plt.suptitle('Group Lasso vs. Lasso (100 $\lambda$ values until $0.5 \lambda_{\max}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, 'group_lasso_vs_lasso.png'))
