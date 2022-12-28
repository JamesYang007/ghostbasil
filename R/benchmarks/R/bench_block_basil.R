source('lasso_utils.R')

library(microbenchmark)

generate.data.block <- function(n, p, L, seed=0, ...)
{
    outs <- lapply(1:L, function(i) {
        generate.data(n, p, seed=seed+i, ...)
    })
    mats <- lapply(1:L, function(i) {outs[[i]]$A})
    A <- BlockMatrix(mats)
    r <- as.numeric(sapply(1:L, function(i) {outs[[i]]$r}))
    list(A=A, r=r)
}

bench <- function(
    n, p, Ls, seed=0, times=times, 
    ...
)
{
    n.Ls <- length(Ls)
    out <- rep(0, times=n.Ls)
    models <- list()
    for (i in 1:n.Ls) {
        L <- Ls[i]

        print(paste("Iteration ", i, ": ", L, sep=''))

        start <- proc.time()
        data <- generate.data.block(n=n, p=p, L=L, seed=seed)
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
        out[i] <- summary(bench.out)$mean
        print(paste("Fitting Time: ", out[i], sep=''))
        print(paste("Error msg: ", models[[i]]$error, sep=''))
    }
    
    list(times=out, models=models)
}

n <- 100
p <- 50
Ls <- c(1, 2, 5, 10, 50, 100)
times <- 1
seed <- 9183

write.csv.default(n, 'n.csv')
write.csv.default(p, 'p.csv')
write.csv.default(Ls, 'l.csv')

bench.out <- bench(n, p, Ls, times=times, seed=seed)
bench.times <- bench.out$times
write.csv.default(bench.times, 'block_basil_times.csv')