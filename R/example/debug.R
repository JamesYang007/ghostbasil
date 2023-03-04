library(devtools)
load_all()

n <- 10
p <- 100
X <- matrix(rnorm(n*p), n)
y <- X %*% rnorm(p) + rnorm(n)
A <- (t(X) %*% X) / p
r <- t(X) %*% y / p
out.1 <- ghostbasil(A, r, alpha=1, do.early.exit=T, min.ratio=1e-6)
out.2 <- ghostbasil(A, r, alpha=1, do.early.exit=F, min.ratio=1e-6)
print(length(out.1$lmdas) <= length(out.2$lmdas))