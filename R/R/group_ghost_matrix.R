#' Overload for GroupGhostMatrix dimension.
#'
#' @param   x   GroupGhostMatrix object.
#' @export
dim.Rcpp_GroupGhostMatrix__ <- function(x) x$dim

#' Creates an instance of a GroupGhostMatrix.
#'
#' This matrix has the block-form of Sigma on the off-diagonal
#' and Sigma + D on the diagonal.
#' For example, if n.groups is 2, then GhostMatrix represents a matrix of the form
#' \deqn{
#'  \begin{bmatrix}
#'      \Sigma+D & \Sigma \\
#'      \Sigma & \Sigma+D
#'  \end{bmatrix}
#' }
#'
#' @param   Sigma   covariance matrix of original features (dense matrix) minus D.
#' @param   D       matrix to add to Sigma on the diagonal blocks (dense matrix).
#' @param   n.groups    number of groups (e.g. 1 knockoff means 2 groups).
#' @export
GroupGhostMatrix <- function(Sigma, D, n.groups)
{
    Sigma <- as.matrix(Sigma)
    D <- as.matrix(D)
    n.groups <- as.integer(n.groups)

    if (n.groups < 2) {
        stop(paste("Number of groups must be at least 2.",
             "If number of groups <= 1, use the usual matrix instead,",
             "since GhostMatrix degenerates to the top-left corner matrix.",
             sep=' '))
    }
    Sigma.dim <- dim(Sigma)
    D.dim <- dim(D)
    if (Sigma.dim[1] != Sigma.dim[2]) {
        stop("Sigma is not square.")
    }
    if (Sigma.dim[1] == 0) {
        stop("Sigma is empty.")
    }
    if (sum(Sigma.dim != D.dim) > 0) {
        stop("Sigma must have both dimensions as the length of D.")
    }

    out <- methods::new(GroupGhostMatrix__, Sigma, D, n.groups) 

    out
}
