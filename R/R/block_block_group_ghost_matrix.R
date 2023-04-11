#' Overload for BlockBlockGroupGhostMatrix dimension.
#'
#' @param   x   BlockBlockGroupGhostMatrix object.
#' @export
dim.Rcpp_BlockBlockGroupGhostMatrix__ <- function(x) x$dim

#' Creates an instance of a BlockBlockGroupGhostMatrix.
#'
#' This matrix represents a block diagonal matrix where each
#' block matrix is a BlockGroupGhostMatrix.
#'
#' @param   blocks  list of matrices of the same type.
#'                  If specified, the other arguments will be ignored.
#'                  Otherwise, the other arguments will be used. 
#' @param   off.diagonals    list of dense off-diagonal blocks to construct each BlockGroupGhostMatrix.
#' @param   diagonals     list of non-off-diagonal D BlockMatrix to construct each BlockGroupGhostMatrix.
#' @param   n.groups    vector of the number of groups for each BlockGroupGhostMatrix.
#'                      If only one value is specified, it is assumed to be
#'                      the same value for each BlockGroupGhostMatrix.
#' @export
BlockBlockGroupGhostMatrix <- function(blocks=list(),
                                       off.diagonals=list(),
                                       diagonals=list(),
                                       n.groups=c())
{
    if (length(blocks) <= 0) {
        # replicate value if only one provided.
        if (length(n.groups) == 1) {
            n.groups <- rep(n.groups, length(off.diagonals))
        }
        if ((length(off.diagonals) != length(diagonals)) ||
            (length(diagonals) != length(n.groups))) {
            stop("Length of off.diagonals, diagonals, and n.groups must be the same.")
        }
        if (length(off.diagonals) <= 0) {
            stop("Length of off.diagonals, diagonals, and n.groups must be positive.")
        }
        for (i in 1:length(off.diagonals)) {
            gmat <- BlockGroupGhostMatrix(off.diagonals[[i]], diagonals[[i]], n.groups[i])
            blocks <- append(blocks, list(gmat))
        }
    }

    block.type <- class(blocks[[1]])
    if (all(block.type != 'Rcpp_BlockGroupGhostMatrix__')) {
        stop("Every block must be of BlockGroupGhostMatrix type.")
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

    out <- methods::new(BlockBlockGroupGhostMatrix__, blocks) 
    
    out
}
