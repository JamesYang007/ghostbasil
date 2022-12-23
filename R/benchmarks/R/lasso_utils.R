generate.data <- function(n, p, rho=0.3, sparsity=0.3, seed=0)
{
    # Generate A, r
    set.seed(seed)
    X <- matrix(rnorm(n * p), n, p)
    Sigma <- matrix(rho, p, p) 
    diag(Sigma) <- 1
    R <- chol(Sigma)
    X <- X %*% R
    beta.idx <- rbinom(p, 1, 1-sparsity)
    beta <- rnorm(p, 0.5) * beta.idx
    y <- X %*% beta + rnorm(n)
    A <- (t(X) %*% X) / n
    r <- (t(X) %*% y) / n
    list(A=A, r=r)
}