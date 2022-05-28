import cvxpy as cp
import numpy as np
import argparse
import os

prefix = os.path.dirname(os.path.abspath(__file__))
prefix = os.path.join(prefix, 'data')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--suffix',
    type=str,
    default='',
    help='Suffix to filenames of saved quantities.'
)

args = parser.parse_args()
suffix = args.suffix


def get_input(suffix):
    n = np.loadtxt(os.path.join(prefix, f"n_{suffix}.csv"), delimiter=',', dtype=int)
    p = np.loadtxt(os.path.join(prefix, f"p_{suffix}.csv"), delimiter=',', dtype=int)
    s = np.loadtxt(os.path.join(prefix, f"s_{suffix}.csv"), delimiter=',')
    lmdas = np.loadtxt(os.path.join(prefix, f"lmda_{suffix}.csv"), delimiter=',')
    lmdas.resize(lmdas.size)
    strong_set = np.loadtxt(os.path.join(prefix, f"strong_set_{suffix}.csv"), delimiter=',', dtype=int)
    if strong_set.size == 0:
        strong_set = np.arange(0, p, 1)

    return n, p, s, lmdas, strong_set


def generate_data(n, p, seed=0):
    '''
    Generate A, r in the objective
        (1-s)/2 \beta^\top A \beta - \beta^\top r + s/2 \beta^\top \beta
    Returns A, r.
    '''
    np.random.seed(seed)
    X = np.random.normal(size=(n, p))
    beta = np.random.normal(size=p) * \
        np.random.binomial(1, 0.5, size=p)
    y = X @ beta + np.random.normal(size=n)
    return X.T @ X / n, X.T @ y / n


def generate_solution(A, r, s, lmda, strong_set):
    p = len(r)
    beta = cp.Variable(p)
    objective = cp.Minimize(
        ((1-s)/2) * cp.quad_form(beta, cp.Parameter(shape=A.shape, value=A, PSD=True)) -
        beta @ r +
        s/2 * cp.sum(cp.square(beta)) +
        lmda * cp.norm1(beta)
    )
    constraints = [
        beta[j] == 0
        for j in range(p)
        if j not in strong_set
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.OSQP, max_iter=100000,
                        eps_abs=1e-8, eps_rel=1e-8)
    return result, beta.value


def save_solution(A, r, strong_set, objs, betas, suffix=''):
    np.savetxt(os.path.join(prefix, f'A_{suffix}.csv'), A, delimiter=',')
    np.savetxt(os.path.join(prefix, f'r_{suffix}.csv'), r, delimiter=',')
    np.savetxt(os.path.join(prefix, f'strong_set_{suffix}.csv'), strong_set, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(prefix, f'obj_{suffix}.csv'), objs, delimiter=',')
    np.savetxt(os.path.join(prefix, f'beta_{suffix}.csv'), betas, delimiter=',')


if __name__ == '__main__':
    n, p, s, lmdas, strong_set = get_input(suffix)
    print(f'n: {n}')
    print(f'p: {p}')
    print(f's: {s}')
    print(f'lmdas: {lmdas}')
    print(f'strong_set: {strong_set}')
    print(f'suffix: {suffix}')
    A, r = generate_data(n, p)
    objs = np.zeros(lmdas.size)
    betas = np.zeros((p, lmdas.size))
    for i in range(lmdas.size):
        obj, beta = generate_solution(A, r, s, lmdas[i], strong_set)
        objs[i] = obj
        betas[:, i] = beta
    save_solution(A, r, strong_set, objs, betas, suffix)
