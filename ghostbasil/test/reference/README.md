# Reference Data and Solution Generation

This directory contains a helper module `generate.py`
that generates random input data and solutions for solving the lasso problem of interest
using `CVXPY` to test our implementation against the "truth".

Run the following to see all the options:
```
python generate.py -h
```

As an example, one of the already saved set of files was created by running:
```
python generate.py -n 10 -p 3 --suffix lasso_1
```
