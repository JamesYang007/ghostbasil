source('matrix_util.R')

test.compare(generate.data.ghost, n=10, p=5, M=2, seed=132)
test.compare(generate.data.ghost, n=10, p=100, M=2, seed=132)
test.compare(generate.data.ghost, n=10, p=500, M=7, seed=132)
test.compare(generate.data.ghost, n=100, p=2, M=2, seed=582)
test.compare(generate.data.ghost, n=100, p=10, M=7, seed=9283)