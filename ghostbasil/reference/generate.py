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


def get_lasso_input(suffix):
    n = np.loadtxt(os.path.join(prefix, f"n_{suffix}.csv"), delimiter=',', dtype=int)
    p = np.loadtxt(os.path.join(prefix, f"p_{suffix}.csv"), delimiter=',', dtype=int)
    s = np.loadtxt(os.path.join(prefix, f"s_{suffix}.csv"), delimiter=',')
    lmdas = np.loadtxt(os.path.join(prefix, f"lmda_{suffix}.csv"), delimiter=',')
    lmdas.resize(lmdas.size)
    strong_set = np.loadtxt(os.path.join(prefix, f"strong_set_{suffix}.csv"), delimiter=',', dtype=int)
    if strong_set.size == 0:
        strong_set = np.arange(0, p, 1)

    return n, p, s, lmdas, strong_set


def get_group_lasso_input(suffix):
    out = get_lasso_input(suffix) 
    groups = np.loadtxt(os.path.join(prefix, f"groups_{suffix}.csv"), delimiter=',', dtype=int)
    return out + (groups,)


def generate_lasso_data(n, p, seed=0):
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


def generate_lasso_solution(A, r, s, lmda, strong_set):
    '''
    Generates solution for the lasso problem.
    '''
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


def generate_group_lasso_solution(A, r, groups, s, lmda, strong_set):
    '''
    Generates solution for the group lasso problem.
    '''
    p = len(r)
    beta = cp.Variable(p)
    expr = (
        ((1-s)/2) * cp.quad_form(beta, cp.Parameter(shape=A.shape, value=A, PSD=True))
        - beta @ r
        + s/2 * cp.sum(cp.square(beta))
    )
    for j in range(len(groups) - 1):
        expr += lmda * cp.norm2(beta[groups[j]:groups[j+1]])
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
        eps=1e-16,
    )
    return result, beta.value


def save_solution(A, r, strong_set, objs, betas, suffix=''):
    np.savetxt(os.path.join(prefix, f'A_{suffix}.csv'), A, delimiter=',')
    np.savetxt(os.path.join(prefix, f'r_{suffix}.csv'), r, delimiter=',')
    np.savetxt(os.path.join(prefix, f'strong_set_{suffix}.csv'), strong_set, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(prefix, f'obj_{suffix}.csv'), objs, delimiter=',')
    np.savetxt(os.path.join(prefix, f'beta_{suffix}.csv'), betas, delimiter=',')


if __name__ == '__main__':
    n, p, s, lmdas, strong_set, groups = get_group_lasso_input(suffix)
    print(f'n: {n}')
    print(f'p: {p}')
    print(f's: {s}')
    print(f'lmdas: {lmdas}')
    print(f'strong_set: {strong_set}')
    print(f'groups: {groups}')
    print(f'suffix: {suffix}')
    A, r = generate_group_lasso_data(n, p, groups)
    print(A)
    print(r)
    objs = np.zeros(lmdas.size)
    betas = np.zeros((p, lmdas.size))
    for i in range(lmdas.size):
        obj, beta = generate_group_lasso_solution(A, r, groups, s, lmdas[i], strong_set)
        objs[i] = obj
        betas[:, i] = beta
    save_solution(A, r, strong_set, objs, betas, suffix)
