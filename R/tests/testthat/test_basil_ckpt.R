test_ckpt <- function(A, r, max.lambdas=100, n.threads=-1, ...)
{
    # initial fit
    out.expected <- ghostbasil(A, r, max.lambdas=max.lambdas, n.threads=n.threads, ...)

    expect_equal(out.expected$error, "")

    lmdas <- out.expected$lmdas
    sub.idx <- as.integer(length(lmdas) / 2)
    expect_gte(sub.idx, 1)
    lmdas.sub <- lmdas[1:(sub.idx-1)]
    lmdas.rest <- lmdas[sub.idx:length(lmdas)]

    # fit on subset and checkpoint
    out.ckpt <- ghostbasil(A, r, max.lambdas=length(lmdas.sub), user.lambdas=lmdas.sub, n.threads=n.threads, ...)

    expect_equal(out.ckpt$lmdas, lmdas.sub)

    # fit on the rest with checkpoint
    out.actual <- ghostbasil(A, r, max.lambdas=length(lmdas.rest), user.lambdas=lmdas.rest, 
                            n.threads=n.threads, checkpoint=out.ckpt$checkpoint, ...)

    expect_equal(out.actual$lmdas, lmdas.rest)

    # fit on the whole
    expect_equal(
        out.actual$lmdas,
        out.expected$lmdas[sub.idx:length(lmdas)]
    )
    expect_equal(
        out.actual$rsqs,
        out.expected$rsqs[sub.idx:length(lmdas)]
    )
    expect_equal(
        as.matrix(out.actual$betas), 
        as.matrix(out.expected$betas[, sub.idx:length(lmdas)])
    )
}

# ==============================================================
# TEST Dense Checkpoint
# ==============================================================

test_dense_ckpt <- function(n=100, p=50, seed=0, ...)
{
    set.seed(seed)

    X <- matrix(rnorm(n * p), n, p)
    y <- X %*% rnorm(p) + rnorm(n)
    A <- t(X) %*% X / n 
    r <- t(X) %*% y / n
    
    test_ckpt(A, r, ...)
}

# Note: not all examples will work because 
# it's not guaranteed that the feeding the checkpoint will result in the same strong sets.
# It is not an issue from a correctness point of view.
# The results will just be off by a small margin due to different convergence.
# The following are invariants, i.e. further changes should still keep these working.

test_dense_ckpt(n=100, p=50, seed=0, min.ratio=1e-6)
test_dense_ckpt(n=100, p=100, seed=0, min.ratio=1e-6)
test_dense_ckpt(n=100, p=100, alpha=0.8, seed=843, min.ratio=1e-6)
test_dense_ckpt(n=2, p=100, alpha=0.5, seed=941, min.ratio=1e-6)

# ==============================================================
# TEST Ghost Checkpoint
# ==============================================================
test_ghost_ckpt <- function(n=100, p=50, M=2, seed=0, ...)
{
    set.seed(seed)

    X <- matrix(rnorm(n * p), n, p)
    y <- X %*% rnorm(p) + rnorm(n)
    S <- t(X) %*% X / n
    D <- rep(min(eigen(S, symmetric=T, only.values=T)$values), p)
    S <- S - diag(D)
    A <- GhostMatrix(S, D, M)
    r <- t(X) %*% y / n
    r <- rep(r, times=M)
    
    test_ckpt(A, r, ...)
}

test_ghost_ckpt(n=100, p=2, M=2, seed=0, min.ratio=1e-6)
test_ghost_ckpt(n=100, p=50, M=5, seed=0, min.ratio=1e-6)
test_ghost_ckpt(n=100, p=100, M=2, alpha=0.2, seed=0, min.ratio=1e-6)

# ==============================================================
# TEST Block<Dense> Checkpoint
# ==============================================================

test_block_dense_ckpt <- function(n=100, p=50, L=10, seed=0, ...)
{
    set.seed(seed)
    mat.list <- list()
    vec.list <- c()
    for (i in 1:L) {
        X <- matrix(rnorm(n * p), n, p)
        y <- X %*% rnorm(p) + rnorm(n)
        A <- t(X) %*% X / n
        r <- t(X) %*% y / n
        mat.list[[i]] <- A / L
        vec.list <- c(vec.list, r / L)
    }
    A <- BlockMatrix(mat.list)
    r <- vec.list
    test_ckpt(A, r, ...)
}

test_block_dense_ckpt(n=100, p=50, L=10, alpha=0.5, seed=123, min.ratio=1e-6)
test_block_dense_ckpt(n=100, p=50, L=1, alpha=0.1, seed=123, min.ratio=1e-6)
test_block_dense_ckpt(n=10, p=100, L=2, alpha=0.0, seed=8421, min.ratio=1e-6)

# ==============================================================
# TEST Block<Ghost> Checkpoint
# ==============================================================

test_block_ghost_ckpt <- function(n=100, p=50, L=10, M=2, seed=0, ...)
{
    set.seed(seed)
    mat.list <- list()
    vec.list <- c()
    for (i in 1:L) {
        X <- matrix(rnorm(n * p), n, p)
        y <- X %*% rnorm(p) + rnorm(n)
        S <- t(X) %*% X / (n * L)
        D <- rep(min(eigen(S, symmetric=T, only.values=T)$values), p)
        S <- S - diag(D)
        A <- GhostMatrix(S, D, M)
        r <- t(X) %*% y / (n * L)
        r <- rep(r, times=M)
        mat.list[[i]] <- A
        vec.list <- c(vec.list, r)
    }
    A <- BlockGhostMatrix(blocks=mat.list)
    r <- vec.list
    test_ckpt(A, r, ...)
}

test_block_ghost_ckpt(n=100, p=2, L=2, M=2, seed=0, min.ratio=1e-6)
test_block_ghost_ckpt(n=100, p=5, L=2, M=5, seed=0, min.ratio=1e-6)
test_block_ghost_ckpt(n=100, p=10, L=10, M=2, alpha=0.5, seed=0, min.ratio=1e-6)
