library(devtools)
load_all()

p <- 10
X <- matrix(rnorm(p * p), p)
Sigma <- (t(X) %*% X) / p
D <- BlockMatrix(lapply(1:p, function(i) matrix(0.1,1,1)))
S <- Sigma - diag(0.1, p, p)
n.groups <- 2
gmat <- BlockGroupGhostMatrix(S, D, n.groups)