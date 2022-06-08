library(microbenchmark)
library(devtools)
source('utils.R')

# Load GhostBASIL
GhostBASIL.path <- paste(thisPath(), '/../R', sep='')
load_all(GhostBASIL.path)

p <- 500
s <- 0.5
n_groups <- 2
dataset <- generate.data(p, p)
Sigma <- dataset$A
D <- rep(1, dim(Sigma)[1])
r <- dataset$r

bm <- GhostMatrix(Sigma, D, n_groups)
br <- rep(r, times=n_groups)

bmd <- matrix(0, n_groups*p, n_groups*p)
for (k in 1:n_groups) {
    for (j in 1:n_groups) {
        if (j != k) {
            bmd[(j-1)*p + 1:p, (k-1)*p + 1:p] <- Sigma 
        } else {
            bmd[(j-1)*p + 1:p, (k-1)*p + 1:p] <- Sigma + diag(D)
        }
    }
}

bench <- function(A, n.threads=0) {
    out.time <- microbenchmark(out <- ghostbasil(A,br,s,n.threads=n.threads), times=10L, unit='s')
    list(out, out.time)
}
