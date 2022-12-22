source('group_lasso_utils.R')

lasso.process.data <- function(data)
{
    data$strong.order <- order(data$strong.set)
    data$active.set.order <- order(data$active.set) 

    data
}

lasso.bench <- function(n, ps, n.lmdas, times, seed, alpha=1.0, max.cds=100000, thr=1e-20, debug=F, ...)
{
    n.ps <- length(ps)
    out <- rep(0, times=n.ps)
    models <- list()
    for (i in 1:n.ps) {
        p <- ps[i]
        n.groups <- as.integer(p)
        penalty <- rep(1, n.groups)

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
        
        start <- proc.time()
        data <- lasso.process.data(data)
        lp.data.time <- proc.time() - start
        print('Process data time:')
        print(lp.data.time)

        if (debug) {
            print(head(data$A))
            print(as.integer(data$groups-1))
            print(alpha)
            print(as.integer(data$strong.set-1))
            print(as.integer(data$strong.order-1))
            print(data$strong.A.diag)
            print(data$lmdas)
            print(data$rsq)
            print(data$strong.beta)
            print(data$strong.grad)
            print(data$active.set)
            print(data$active.order)
            print(data$active.set.order)
            print(data$is.active)
        }

        bench.out <- microbenchmark(
            {
                models[[i]] <- lasso__(
                    data$A,
                    alpha,
                    penalty,
                    as.integer(data$strong.set-1),
                    as.integer(data$strong.order-1),
                    data$strong.A.diag,
                    data$lmdas,
                    max.cds,
                    thr,
                    data$rsq,
                    data$strong.beta,
                    data$strong.grad,
                    data$active.set,
                    data$active.order,
                    data$active.set.order,
                    data$is.active
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
ps <- c(100, 200, 400, 800, 1600)
#ps <- 3200
times <- 1
seed <- 41923
n.lmdas <- 100

write.csv.default(n, 'n.csv')
write.csv.default(ps, 'p.csv')

gl.out <- bench(n, ps, n.groups.prop=1, n.lmdas=n.lmdas, times=times, seed=seed, debug=F)
gl.bench.times <- gl.out$times
write.csv.default(gl.bench.times, 'group_lasso_times.csv')

l.out <- lasso.bench(n, ps, n.lmdas=n.lmdas, times=times, seed=seed, debug=F)
l.bench.times <- l.out$times
write.csv.default(l.bench.times, 'lasso_times.csv')

for (i in 1:length(ps)) {
    tmp <- max(abs(gl.out$models[[i]]$beta - l.out$models[[i]]$beta))
    print(tmp)
    tmp <- (gl.out$models[[i]]$n_cds - l.out$models[[i]]$n_cds) 
    print(tmp)
}
gl.bench.times / l.bench.times
