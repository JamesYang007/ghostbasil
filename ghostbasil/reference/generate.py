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
    help='Suffix to filenames of saved quantities.',
)
parser.add_argument(
    '--group',
    action='store_true',
    help='Looks for group version.',
)

args = parser.parse_args()
suffix = args.suffix
is_group = args.group


def get_lasso_input(suffix):
    n = np.loadtxt(os.path.join(prefix, f"n_{suffix}.csv"), delimiter=',', dtype=int)
    p = np.loadtxt(os.path.join(prefix, f"p_{suffix}.csv"), delimiter=',', dtype=int)
    alpha = np.loadtxt(os.path.join(prefix, f"alpha_{suffix}.csv"), delimiter=',')
    penalty = np.loadtxt(os.path.join(prefix, f"penalty_{suffix}.csv"), delimiter=',')
    if penalty.size == 0:
        penalty = np.ones(p)
    penalty = penalty * (p / np.sum(penalty))
    lmdas = np.loadtxt(os.path.join(prefix, f"lmda_{suffix}.csv"), delimiter=',')
    lmdas.resize(lmdas.size)
    strong_set = np.loadtxt(os.path.join(prefix, f"strong_set_{suffix}.csv"), delimiter=',', dtype=int)
    if strong_set.size == 0:
        strong_set = np.arange(0, p, 1)

    return n, p, alpha, lmdas, penalty, strong_set


def get_group_lasso_input(suffix):
    n = np.loadtxt(os.path.join(prefix, f"n_{suffix}.csv"), delimiter=',', dtype=int)
    p = np.loadtxt(os.path.join(prefix, f"p_{suffix}.csv"), delimiter=',', dtype=int)
    alpha = np.loadtxt(os.path.join(prefix, f"alpha_{suffix}.csv"), delimiter=',')
    penalty = np.loadtxt(os.path.join(prefix, f"penalty_{suffix}.csv"), delimiter=',')
    groups = np.loadtxt(os.path.join(prefix, f"groups_{suffix}.csv"), delimiter=',', dtype=int)
    n_groups = len(groups) - 1
    if penalty.size == 0:
        penalty = np.ones(n_groups)
    penalty = penalty * (n_groups / np.sum(penalty))
    lmdas = np.loadtxt(os.path.join(prefix, f"lmda_{suffix}.csv"), delimiter=',')
    lmdas.resize(lmdas.size)
    strong_set = np.loadtxt(os.path.join(prefix, f"strong_set_{suffix}.csv"), delimiter=',', dtype=int)
    if strong_set.size == 0:
        strong_set = np.arange(0, n_groups, 1)

    return n, p, alpha, lmdas, penalty, strong_set, groups


def generate_lasso_data(n, p, seed=0):
    '''
    Generate A, r in the objective
        1/2 \beta^\top A \beta - \beta^\top r + lmda \sum_i p_i ((1-\alpha)/2 \beta_i^2 + \alpha |\beta_i|)
    Returns A, r.
    '''
    np.random.seed(seed)
    X = np.random.normal(size=(n, p))
    beta = np.random.normal(size=p) * \
        np.random.binomial(1, 0.5, size=p)
    y = X @ beta + np.random.normal(size=n)
    return X.T @ X / n + np.diag(np.full(p, 1e-1)), X.T @ y / n


def generate_group_lasso_data(n, p, groups, seed=0):
    '''
    Generate A, r in the group-lasso objective.
    Note that A needs to obey a certain structure.
    '''
    A, r = generate_lasso_data(n, p, seed)
    
    n_groups = len(groups) - 1
    eigs = [
        np.linalg.eigh(A[groups[i] : groups[i+1], groups[i] : groups[i+1]])
        for i in range(n_groups)
    ] 
    eigvals = [e[0] for e in eigs]
    eigvecs = [e[1] for e in eigs]
    for i in range(n_groups):
        gi_b = groups[i]
        gi_e = groups[i+1]
        for j in range(n_groups):
            gj_b = groups[j]
            gj_e = groups[j+1]
            view = A[gi_b : gi_e, gj_b : gj_e]
            if i == j:
                view[...] = np.zeros(view.shape)
                np.fill_diagonal(view, np.maximum(eigvals[i], 0))
            else:
                view[...] = eigvecs[i].T @ view @ eigvecs[j]
        r[gi_b : gi_e] = eigvecs[i].T @ r[gi_b : gi_e]
                
    return A, r


def generate_lasso_solution(A, r, alpha, lmda, penalty, strong_set):
    '''
    Generates solution for the lasso problem.
    '''
    p = len(r)
    beta = cp.Variable(p)
    objective = cp.Minimize(
        0.5 * cp.quad_form(beta, cp.Parameter(shape=A.shape, value=A, PSD=True)) -
        beta @ r +
        lmda * (1-alpha) * 0.5 * cp.sum(cp.multiply(cp.square(beta), penalty)) +
        lmda * alpha * cp.sum(cp.multiply(cp.abs(beta), penalty))
    )
    constraints = [
        beta[j] == 0
        for j in range(p)
        if j not in strong_set
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(
        #solver=cp.OSQP, max_iter=100000,
        #eps_abs=1e-9, eps_rel=1e-9,
        solver=cp.SCS, 
        max_iters=10000,
        eps=1e-24,
    )
    return result, beta.value


def generate_group_lasso_solution(A, r, groups, alpha, penalty, lmda, strong_set):
    '''
    Generates solution for the group lasso problem.
    '''
    p = len(r)
    beta = cp.Variable(p)
    expr = (
        0.5 * cp.quad_form(beta, cp.Parameter(shape=A.shape, value=A, PSD=True))
        - beta @ r
    )
    for j in range(len(groups) - 1):
        expr += lmda * (1-alpha) / 2 * penalty[j] * cp.sum(cp.square(beta[groups[j]:groups[j+1]]))
        expr += lmda * alpha * penalty[j] * cp.norm2(beta[groups[j]:groups[j+1]])
    objective = cp.Minimize(expr)
    constraints = [
        beta[groups[j] : groups[j+1]] == 0
        for j in range(len(groups)-1)
        if not (j in strong_set)
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(
        solver=cp.SCS, 
        max_iters=10000,
        eps=1e-24,
    )
    return result, beta.value


def save_solution(A, r, penalty, strong_set, objs, betas, suffix=''):
    np.savetxt(os.path.join(prefix, f'A_{suffix}.csv'), A, delimiter=',')
    np.savetxt(os.path.join(prefix, f'r_{suffix}.csv'), r, delimiter=',')
    np.savetxt(os.path.join(prefix, f'penalty_{suffix}.csv'), penalty, delimiter=',')
    np.savetxt(os.path.join(prefix, f'strong_set_{suffix}.csv'), strong_set, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(prefix, f'obj_{suffix}.csv'), objs, delimiter=',')
    np.savetxt(os.path.join(prefix, f'beta_{suffix}.csv'), betas, delimiter=',')


if __name__ == '__main__':
    if is_group:
        n, p, alpha, lmdas, penalty, strong_set, groups = get_group_lasso_input(suffix)
        print(f'n: {n}')
        print(f'p: {p}')
        print(f'alpha: {alpha}')
        print(f'lmdas: {lmdas}')
        print(f'penalty: {penalty}')
        print(f'strong_set: {strong_set}')
        print(f'groups: {groups}')
        print(f'suffix: {suffix}')
        A, r = generate_group_lasso_data(n, p, groups)
        print(A)
        print(r)
        objs = np.zeros(lmdas.size)
        betas = np.zeros((p, lmdas.size))
        for i in range(lmdas.size):
            obj, beta = generate_group_lasso_solution(A, r, groups, alpha, penalty, lmdas[i], strong_set)
            objs[i] = obj
            betas[:, i] = beta
        save_solution(A, r, penalty, strong_set, objs, betas, suffix)

    else:
        n, p, alpha, lmdas, penalty, strong_set = get_lasso_input(suffix)
        print(f'n: {n}')
        print(f'p: {p}')
        print(f'alpha: {alpha}')
        print(f'lmdas: {lmdas}')
        print(f'penalty: {penalty}')
        print(f'strong_set: {strong_set}')
        print(f'suffix: {suffix}')
        A, r = generate_lasso_data(n, p)
        print(A)
        print(r)
        objs = np.zeros(lmdas.size)
        betas = np.zeros((p, lmdas.size))
        for i in range(lmdas.size):
            obj, beta = generate_lasso_solution(A, r, alpha, lmdas[i], penalty, strong_set)
            objs[i] = obj
            betas[:, i] = beta
        save_solution(A, r, penalty, strong_set, objs, betas, suffix)