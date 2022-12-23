source('lasso_utils.R')

library(microbenchmark)

bench <- function(
    n, ps, seed=0, times=times, 
    ...
)
{
    n.ps <- length(ps)
    out <- rep(0, times=n.ps)
    models <- list()
    for (i in 1:n.ps) {
        p <- ps[i]

        print(paste("Iteration ", i, ": ", p, sep=''))

        start <- proc.time()
        data <- generate.data(n=n, p=p, seed=seed)
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
ps <- c(100, 200, 400, 800, 1600, 3200)
times <- 1

write.csv.default(n, 'n.csv')
write.csv.default(ps, 'p.csv')

bench.out <- bench(n, ps, times=times, seed=9183,
                     delta.strong.size=100,
                     lambdas.iter=10,
                     thr=1e-7,
                     n.threads=-1)
bench.times <- bench.out$times
write.csv.default(bench.times, 'basil_times.csv')