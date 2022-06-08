library(microbenchmark)
library(devtools)
source('utils.R')

# Load GhostBASIL
GhostBASIL.path <- paste(thisPath(), '/../R', sep='')
load_all(GhostBASIL.path)

p <- 100
s <- 0.5
n_blocks <- 2
dataset <- generate.data(p, p)
A <- dataset$A
r <- dataset$r

mat_list <- list()
for (k in 1:n_blocks) {
    mat_list <- append(mat_list, list(A))
}

bm <- BlockMatrix(mat_list)
br <- rep(r, times=n_blocks)

bmd <- matrix(0, n_blocks*p, n_blocks*p)
for (k in 1:n_blocks) {
    bmd[(k-1)*p + 1:p, (k-1)*p + 1:p] <- A
}

bench <- function(A, n.threads=0) {
    out.time <- microbenchmark(out <- ghostbasil(A,br,s,n.threads=n.threads), times=10L, unit='s')
    list(out, out.time)
}
