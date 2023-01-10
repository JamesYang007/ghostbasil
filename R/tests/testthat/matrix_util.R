generate.data <- function(n, p, seed)
{
    set.seed(seed)
    X <- matrix(rnorm(n * p), n, p)
    y <- X %*% rnorm(p) + rnorm(n)
    Sigma <- t(X) %*% X / n
    r <- t(X) %*% y / n
    list(Sigma=Sigma, r=r)
}

generate.data.ghost <- function(n, p, M, seed)
{
    data <- generate.data(n, p, seed)
    Sigma <- data$Sigma
    r <- data$r
    
    eps <- min(eigen(Sigma, symmetric=T, only.values=T)$values)
    D <- rep(eps, p) 
    S <- Sigma - diag(D)

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
        A=GhostMatrix(S, D, M),
        A.dense=A.dense,
        r=r
    )
}

generate.data.group.ghost <- function(n, p, M, seed=0)
{
    data <- generate.data(n, p, seed)
    Sigma <- data$Sigma
    r <- data$r

    D <- matrix(0, p, p) 
    diag(D) <- min(eigen(Sigma, symmetric=T, only.values=T)$values)
    S <- Sigma - D

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

generate.data.block <- function(n, p, L, seed=0)
{
    data <- lapply(1:L, function(l) generate.data(n, p, seed + l - 1))
    mat.list <- lapply(1:L, function(l) data[[l]]$Sigma)
    r <- as.numeric(sapply(1:L, function(l) data[[l]]$r))

    A.dense <- matrix(0, p * L, p * L)
    for (i in 1:L) {
        ii.b <- p * (i-1) + 1
        ii.e <- ii.b + p - 1
        A.dense[ii.b:ii.e, ii.b:ii.e] <- mat.list[[i]]
    }

    list(
        A=BlockMatrix(mat.list),
        A.dense=A.dense,
        r=r
    )
}

test.compare <- function(generate.data.fn, ...)
{
    data <- generate.data.fn(...)
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