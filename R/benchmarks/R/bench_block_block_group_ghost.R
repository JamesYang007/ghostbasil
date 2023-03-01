source('lasso_utils.R')

library(microbenchmark)

generate.data.block.group.ghost <- function(n, p, M, seed=0)
{
    data <- generate.data(n, p, seed=seed)
    Sigma <- data$A
    r <- data$r

    min.eval <- min(eigen(Sigma, symmetric=T, only.values=T)$values)
    D <- lapply(1:p, function(i) matrix(min.eval, 1, 1))
    D <- BlockMatrix(D)
    S <- Sigma - diag(min.eval, p, p)

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

    r <- rep(r, times=M)

    list(
        A=BlockGroupGhostMatrix(S, D, M),
        A.dense=A.dense,
        r=r
    )
}

generate.data.block.block.group.ghost <- function(n, p, L, M, seed=0)
{
    data <- lapply(1:L, function(l) generate.data.block.group.ghost(n, p, M, seed+l-1))    
    mat.list <- lapply(1:L, function(l) data[[l]]$A)
    r <- as.numeric(sapply(1:L, function(l) data[[l]]$r))
    A <- BlockBlockGroupGhostMatrix(mat.list)
    A.dense <- A$to_dense()
    list(
        A=A,
        A.dense=A.dense,
        r=r
    )
}

bench <- function(
    n, p, Ls, M=2, seed=0, times=times,
    ...
)
{
    n.Ls <- length(Ls)
    block.out <- rep(0, times=n.Ls)
    dense.out <- rep(0, times=n.Ls)
    models <- list()
    for (i in 1:n.Ls) {
        L <- Ls[i]

        print(paste("Iteration ", i, ": ", L, sep=''))

        start <- proc.time()
        data <- generate.data.block.block.group.ghost(n=n, p=p, L=L, M=M, seed=seed)
        gen.data.time <- proc.time() - start
        print("Generate data time:")
        print(gen.data.time)
        
        bench.out <- microbenchmark(
            {
                models[[i]] <- ghostbasil(
                    A=data$A,
                    r=data$r,
                    ...
                )
            },
            times=ifelse(length(times) == 1, times, times[i]),
            unit='ns'
        )
        block.out[i] <- summary(bench.out)$mean
        print(paste("Fitting Time: ", block.out[i], sep=''))
        print(paste("Error msg: ", models[[i]]$error, sep=''))

        # NOTE: models gets overwritten, but unused outside.
        bench.out <- microbenchmark(
            {
                models[[i]] <- ghostbasil(
                    A=data$A.dense,
                    r=data$r,
                    ...
                )
            },
            times=ifelse(length(times) == 1, times, times[i]),
            unit='ns'
        )
        dense.out[i] <- summary(bench.out)$mean
        print(paste("Fitting Time: ", dense.out[i], sep=''))
        print(paste("Error msg: ", models[[i]]$error, sep=''))
    }
    
    list(block.times=block.out, dense.times=dense.out, models=models)
}

n <- 100
p <- 50
Ls <- c(1, 2, 5, 10, 20, 50)
times <- 1
seed <- 9183

write.csv.default(n, 'n.csv')
write.csv.default(p, 'p.csv')
write.csv.default(Ls, 'l.csv')

bench.out <- bench(n, p, Ls, times=times, seed=seed)
bench.times <- bench.out$block.times
write.csv.default(bench.times, 'block_basil_times.csv')
bench.times <- bench.out$dense.times
write.csv.default(bench.times, 'dense_basil_times.csv')