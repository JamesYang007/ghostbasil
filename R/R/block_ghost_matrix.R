#' Overload for BlockGhostMatrix dimension.
#'
#' @param   x   BlockGhostMatrix object.
#' @export
dim.Rcpp_BlockGhostMatrix__ <- function(x) x$dim

#' Creates an instance of a BlockGhostMatrix.
#'
#' This matrix represents a block diagonal matrix where each
#' block matrix is a GhostMatrix.
#'
#' @param   blocks  list of matrices of the same type.
#'                  If specified, matrices, vectors, n.groups will be ignored.
#'                  Otherwise, the three will be used instead.
#' @param   matrices    list of dense matrices to construct each GhostMatrix.
#' @param   vectors     list of dense vectors to construct each GhostMatrix.
#' @param   n.groups    vector of the number of groups for each GhostMatrix.
#'                      If only one value is specified, it is assumed to be
#'                      the same value for each GhostMatrix.
#' @export
BlockGhostMatrix <- function(blocks=list(),
                             matrices=list(),
                             vectors=list(),
                             n.groups=c())
{
    if (length(blocks) <= 0) {
        # replicate value if only one provided.
        if (length(n.groups) == 1) {
            n.groups <- rep(n.groups, length(matrices))
        }
        if ((length(matrices) != length(vectors)) ||
            (length(matrices) != length(n.groups))) {
            stop("Length of matrices, vectors, and n.groups must be the same.")
        }
        if (length(matrices) <= 0) {
            stop("Length of matrices, vectors, and n.groups must be positive.")
        }
        for (i in 1:length(matrices)) {
            gmat <- GhostMatrix(matrices[[i]], vectors[[i]], n.groups[i])
            blocks <- append(blocks, list(gmat))
        }
    }

    block.type <- class(blocks[[1]])
    if (all(block.type != 'Rcpp_GhostMatrix__')) {
        stop("Every block must be of GhostMatrix type.")
    }
    for (i in 1:length(blocks)) {
        block.i <- blocks[[i]]
        block.dim <- dim(block.i)
        if (any(class(block.i) != block.type)) {
            stop(paste("Block at index", i, 
                       "has a different type from all previous blocks.",
                       sep=' '))
        }
        if (block.dim[1] != block.dim[2]) {
            stop(paste("Block at index", i, "is not square.", 
                       sep=' '))
        }
        if ((block.dim[1] == 0) | 
            (block.dim[2] == 0)) {
            stop(paste("Block at index", i, "is empty.", 
                       sep=' '))
        }
    }

    out <- methods::new(BlockGhostMatrix__, blocks) 
    
    out
}
