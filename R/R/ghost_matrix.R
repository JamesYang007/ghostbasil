#' Overload for GhostMatrix dimension.
#' @export
dim.Rcpp_GhostMatrix__ <- function(x) x$dim

#' Creates an instance of a GhostMatrix.
#' @param   Sigma   covariance matrix of original features.
#' @param   D       diagonal to subtract from Sigma in the off-diagonal blocks.
#' @param   n.groups    number of groups (e.g. 1 knockoff means 2 groups).
#' @export
GhostMatrix <- function(Sigma, D, n.groups)
{
    Sigma <- as.matrix(Sigma)
    D <- as.numeric(D)
    n.groups <- as.integer(n.groups)

    if (n.groups < 2) {
        stop(paste("Number of groups must be at least 2.",
             "If number of groups <= 1, use the usual matrix instead,",
             "since GhostMatrix degenerates to the top-left corner matrix.",
             sep=' '))
    }
    Sigma.dim <- dim(Sigma)
    if (Sigma.dim[1] != Sigma.dim[2]) {
        stop("Sigma is not square.")
    }
    if (Sigma.dim[1] == 0) {
        stop("Sigma is empty.")
    }
    if (Sigma.dim[1] != length(D)) {
        stop("Sigma must have both dimensions as the length of D.")
    }

    out <- new(GhostMatrix__, Sigma, D, n.groups) 

    out
}
