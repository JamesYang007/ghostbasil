library(microbenchmark)
library(devtools)
source('utils.R')

# Load GhostBASIL
load_all()

L <- 3
p <- 100
s <- 0.5
n.groups <- 2
eps <- 0.1

dataset <- generate.data(p, p)
Sigma <- dataset$A
D <- rep(eps, dim(Sigma)[1])
r <- dataset$r

matrices <- list()
vectors <- list()
for (l in 1:L) {
    matrices <- append(matrices, list(Sigma))
    vectors <- append(vectors, list(D))
}

bm <- BlockGhostMatrix(matrices=matrices,
                       vectors=vectors,
                       n.groups=n.groups)
br <- rep(r, times=(n.groups * L))

denses <- list()
for (l in 1:L) {
    dense <- matrix(0, n.groups * p, n.groups * p)
    for (k in 1:n.groups) {
        for (j in 1:n.groups) {
            if (j != k) {
                dense[(j-1)*p + 1:p, (k-1)*p + 1:p] <- Sigma 
            } else {
                dense[(j-1)*p + 1:p, (k-1)*p + 1:p] <- Sigma + diag(D)
            }
        }
    }
    denses <- append(denses, list(dense))
}
bmd <- BlockMatrix(denses)

bench <- function(A, n.threads=0) {
    out.time <- microbenchmark(out <- ghostbasil(A,br,s,n.threads=n.threads), times=10L, unit='s')
    list(out, out.time)
}
