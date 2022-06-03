library(microbenchmark)
library(devtools)
source('utils.R')

# Load GhostBASIL
GhostBASIL.path <- paste(thisPath(), '/../R', sep='')
load_all(GhostBASIL.path)

p <- 100
s <- 0.5
n_groups <- 2
dataset <- generate.data(p, p)
A <- dataset$A
r <- dataset$r

mat_list <- list()
for (k in 1:n_groups) {
    mat_list <- append(mat_list, list(A))
}

bm <- BlockMatrix(mat_list)
br <- rep(r, times=n_groups)

bmd <- matrix(0, n_groups*p, n_groups*p)
for (k in 1:n_groups) {
    bmd[(k-1)*p + 1:p, (k-1)*p + 1:p] <- A
}

bench <- function(A) {
    out.time <- microbenchmark(out <- ghostbasil(A,br,s), times=10L, unit='s')
    list(out, out.time)
}
