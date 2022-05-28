# Reference Data and Solution Generation

This directory contains a helper module `generate.py`
that generates random input data and solutions for solving the lasso problem of interest
using `CVXPY` to test our implementation against the "truth".

Run the following to see all the options:
```
python generate.py -h
```

Before running the script, a set of files must be placed in `data/`.
Populate the following files (where `suffix` is any suffix you choose):

- `n_{suffix}.csv`: a single integer for the number of samples.
- `p_{suffix}.csv`: a single integer for the number of features.
- `s_{suffix}.csv`: a single value for the regularization of covariance matrix towards identity.
- `lmda_{suffix}.csv`: a column vector of the L1 regularization hyperparameter. 
- `strong_set_{suffix}.csv`: a column vector of features to be included in the strong set. 
    If it is empty, the full feature set will be used.

As an example, one of the already saved set of files was created by running:
```
python generate.py --suffix lasso_1
```
