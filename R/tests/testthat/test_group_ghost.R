generate.data <- function(n, p, M, seed=0)
{
    X <- matrix(rnorm(n * p), n, p)
    y <- X %*% rnorm(p) + rnorm(n)
    Sigma <- t(X) %*% X / n
    D <- matrix(0, p, p) 
    diag(D) <- min(eigen(Sigma, symmetric=T, only.values=T)$values)
    S <- Sigma - D

    r <- t(X) %*% y / n

    A.dense <- matrix(0, p * M, p * M)
    for (i in 1:M) {
        ii.b <- p * (i-1) + 1
        ii.e <- ii.b + p - 1
        for (j in 1:M) {
            jj.b <- p * (j-1) + 1
            jj.e <- jj.b + p - 1
            if (i == j) {
                A.dense[ii.b:ii.e, jj.b:jj.e] <- Sigma
            } else {
                A.dense[ii.b:ii.e, jj.b:jj.e] <- S
            }
        }
    }

    list(
        A=GroupGhostMatrix(S, D, M),
        A.dense=A.dense,
        r=r
    )
}

tester <- function(...)
{
    data <- generate.data(...)
    A <- data$A
    A.dense <- data$A.dense
    r <- data$r

    out.dense <- ghostbasil(A.dense, r) 
    lmdas <- out.dense$lmdas
    out.gg <- ghostbasil(A, r, user.lambdas=lmdas)

    expect_equal(out.dense$lmdas, out.gg$lmdas)
    expect_equal(out.dense$betas, out.gg$betas)
    expect_equal(out.dense$rsqs, out.gg$rsqs)
}

tester(n=10, p=5, M=2, seed=132)
tester(n=10, p=100, M=2, seed=132)
tester(n=10, p=500, M=7, seed=132)
tester(n=100, p=2, M=2, seed=582)
tester(n=100, p=10, M=7, seed=9283)