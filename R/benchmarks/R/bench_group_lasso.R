library(devtools)
load_all()
library(microbenchmark)

n <- 100
ps <- c(100, 200, 400, 800, 1600, 3200)
times <- 1

write.csv.default(n, 'n.csv')
write.csv.default(ps, 'p.csv')

generate.data <- function(n, p, n.groups, rho=0.3, sparsity=0.3, seed=0)
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

    # Generate random groups
    groups <- sort(sample(2:p, n.groups-1, replace=F))
    groups <- as.integer(c(1, groups, p+1))

    # construct strong set quantities
    strong.set <- as.integer(1:n.groups)
    strong.begins <- groups[strong.set]
    strong.beta <- as.double(rep(0, times=p))
    is.active <- as.integer(rep(0, times=p))
    
    # construct active set quantities
    active.set <- integer(0)
    active.begins <- integer(0)
    active.order <- integer(0)
    
    list(A=A, r=r, groups=groups, 
        strong.set=strong.set, 
        strong.begins=strong.begins,
        strong.beta=strong.beta,
        is.active=is.active,
        active.set=active.set,
        active.begins=active.begins,
        active.order=active.order,
        rsq=as.double(0.0)
    )
}

group.lasso.process.data <- function(data, n.lmdas)
{
    A <- data$A
    r <- data$r
    groups <- data$groups
    n.groups <- length(groups) - 1
    
    A.eigs <- lapply(
        1:n.groups, 
        function(i) { 
            ib = groups[i]
            ie = groups[i+1]-1
            eigen(A[ib:ie, ib:ie], T)
        })
    
    for (i in 1:n.groups) {
        ib <- groups[i]
        ie <- groups[i+1]-1
        for (j in 1:n.groups) {
            jb <- groups[j]
            je <- groups[j+1]-1
            if (i == j) {
                if (jb != je) {
                    A[ib:ie, jb:je] <- 0
                    diag(A[ib:ie, jb:je]) <- A.eigs[[i]]$values
                }
            } else {
                A[ib:ie, jb:je] <- (
                    t(A.eigs[[i]]$vectors) %*% A[ib:ie, jb:je] %*% A.eigs[[j]]$vectors
                )
            }
        }
        r[ib:ie] <- t(A.eigs[[i]]$vectors) %*% r[ib:ie]
    }

    # lambda sequence
    r.norms <- sapply(1:n.groups, function(i) {
        ib <- groups[i]
        ie <- groups[i+1]-1
        sqrt(sum(r[ib:ie]**2))
    })
    lmda.max <- max(r.norms)
    lmdas <- lmda.max * (0.5 ** ((1:n.lmdas) / n.lmdas))
    
    data$A <- A
    data$r <- as.double(r)
    data$strong.A.diag <- as.double(diag(A))
    data$lmdas <- as.double(lmdas)
    data$strong.grad <- data$r

    data
}

bench <- function(
    n, ps, n.groups.prop, n.lmdas, seed=0, times=times, 
    s=0.0, max.cds=10000, thr=1e-20, newton.tol=1e-12, newton.max.iters=1000,
    debug=F, ...
)
{
    n.ps <- length(ps)
    out <- rep(0, times=n.ps)
    for (i in 1:n.ps) {
        p <- ps[i]
        n.groups <- as.integer(p * n.groups.prop)

        print(paste("Iteration ", i, ": ", p, sep=''))

        start <- proc.time()
        data <- generate.data(n=n, p=p, n.groups=n.groups, seed=seed, ...)
        gen.data.time <- proc.time() - start
        print("Generate data time:")
        print(gen.data.time)

        start <- proc.time()
        data <- group.lasso.process.data(data, n.lmdas=n.lmdas)
        glp.data.time <- proc.time() - start
        print('Process data time:')
        print(glp.data.time)

        if (debug) {
            print(head(data$A))
            print(as.integer(data$groups-1))
            print(s)
            print(as.integer(data$strong.set-1))
            print(as.integer(data$strong.begins-1))
            print(data$strong.A.diag)
            print(data$lmdas)
            print(max.cds)
            print(thr)
            print(newton.tol)
            print(newton.max.iters)
            print(data$rsq)
            print(data$strong.beta)
            print(data$strong.grad)
            print(data$active.set)
            print(data$active.begins)
            print(data$active.order)
            print(data$is.active)
        }

        bench.out <- microbenchmark(
            {
                group_lasso__(
                    data$A,
                    as.integer(data$groups-1),
                    s,
                    as.integer(data$strong.set-1),
                    as.integer(data$strong.begins-1),
                    data$strong.A.diag,
                    data$lmdas,
                    max.cds,
                    thr,
                    newton.tol,
                    newton.max.iters,
                    data$rsq,
                    data$strong.beta,
                    data$strong.grad,
                    data$active.set,
                    data$active.begins,
                    data$active.order,
                    data$is.active
                )
            },
            times=ifelse(length(times) == 1, times, times[i]),
            unit='us'
        )
        out[i] <- summary(bench.out)$mean
        print(paste("Fitting Time: ", out[i], sep=''))
    }
    
    out
}

bench.times <- bench(n, ps, n.groups.prop=0.2, n.lmdas=100, times=times, seed=9183)
write.csv.default(bench.times, 'group_lasso_low_groups_times.csv')

bench.times <- bench(n, ps, n.groups.prop=0.8, n.lmdas=100, times=times, seed=9183)
write.csv.default(bench.times, 'group_lasso_high_groups_times.csv')
