import cvxpy as cp
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n',
    type=int,
    nargs='?',
    default=10,
    help='Number of datapoints (default: 10).'
)
parser.add_argument(
    '-p',
    type=int,
    nargs='?',
    default=3,
    help='Number of features (default: 3).'
)
parser.add_argument(
    '-s',
    type=float,
    nargs='?',
    default=0.5,
    help='Regularization strength in objective (default: 0.5).'
)
parser.add_argument(
    '-l',
    type=float,
    nargs='*',
    default=[1.0],
    help='Lambda (default: [1.0]).'
)
parser.add_argument(
    '--strong',
    nargs='*',
    type=int,
    default=None,
    help='Strong set (default: None).'
)
parser.add_argument(
    '--suffix',
    type=str,
    default='',
    help='Suffix to filenames of saved quantities.'
)

args = parser.parse_args()
n = args.n
p = args.p
s = args.s
lmdas = np.array(args.l)
suffix = args.suffix
strong_set = args.strong
if strong_set is None:
    strong_set = np.arange(0, p, 1)

print(f'n: {n}')
print(f'p: {p}')
print(f's: {s}')
print(f'lmdas: {lmdas}')
print(f'strong_set: {strong_set}')
print(f'suffix: {suffix}')


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
        (1-s)/2 * cp.quad_form(beta, A) -
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


def save_solution(A, r, s, strong_set, lmdas, objs, betas, suffix=''):
    prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(prefix, 'data')
    np.savetxt(os.path.join(prefix, f'A_{suffix}.csv'), A, delimiter=',')
    np.savetxt(os.path.join(prefix, f'r_{suffix}.csv'), r, delimiter=',')
    np.savetxt(os.path.join(prefix, f's_{suffix}.csv'), np.array([s]), delimiter=',')
    np.savetxt(os.path.join(prefix, f'lmda_{suffix}.csv'), lmdas, delimiter=',')
    np.savetxt(os.path.join(prefix, f'obj_{suffix}.csv'), objs, delimiter=',')
    np.savetxt(os.path.join(prefix, f'beta_{suffix}.csv'), betas, delimiter=',')
    np.savetxt(os.path.join(prefix, f'strong_set_{suffix}.csv'), strong_set, delimiter=',',
               fmt='%d')


if __name__ == '__main__':
    A, r = generate_data(n, p)
    objs = np.zeros(lmdas.size)
    betas = np.zeros((p, lmdas.size))
    for i in range(lmdas.size):
        obj, beta = generate_solution(A, r, s, lmdas[i], strong_set)
        objs[i] = obj
        betas[:, i] = beta
    save_solution(A, r, s, strong_set, lmdas, objs, betas, suffix)
