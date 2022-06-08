#' Overload for GhostMatrix dimension.
#' @export
dim.Rcpp_BlockMatrix__ <- function(x) x$dim

#' Creates an instance of a BlockMatrix.
#' @param   blocks  list of dense matrices.
#' @export
BlockMatrix <- function(blocks)
{
    if (length(blocks) <= 0) {
        stop("Length of blocks must be positive.")
    }
    block.type <- class(blocks[[1]])
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

    out <- new(BlockMatrix__, blocks) 
    
    out
}
