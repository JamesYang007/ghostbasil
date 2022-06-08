#' Fits PGR objective with BASIL framework.
#'
#' @param   A   data covariance matrix.
#'              It must be one of four types: matrix, GhostMatrix__, BlockMatrix__, BlockGhostMatrix__.
#'              For types of the form xxx__, see the function xxx.
#' @param   r   correlation vector.
#' @param   s   regularization to shrink A towards identity ((1-s) * A + s * I).
#' @param   user.lambdas    user-specified sequence of lambdas. Will be sorted in decreasing order if not sorted already.
#' @param   max.lambdas     maximum number of lambdas to compute solutions for.
#'                          This defines the granularity of the lambda sequence 
#'                          starting from the maximum lambda
#'                          to the smallest lambda, which is the maximum lambda * min.ratio.
#'                          Note that the call may terminate early and output solutions for fewer lambdas than max.lambdas.
#'                          This can occur, for example, if the algorithm detects that the R^2 isn't changing enough.
#'                          The maximum lambda is the largest lambda where solution isn't 0.
#' @param   lambdas.iter    number of lambdas to compute strong set solutions for at each iteration of the BASIL algorithm.
#'                          Internally, capped at max.lambdas.
#'                          Must be positive.
#' @param   strong.size     initial number of strong set variables to include.
#'                          Internally, capped at max.strong.size.
#' @param   delta.strong.size   number of strong set variables to include at every iteration of the BASIL algorithm.
#'                              Internally, capped at number of non-strong variables at every iteration of BASIL.
#'                              Must be positive.
#' @param   max.strong.size     maximum number of strong set size. 
#'                              Internally, capped at number of features.
#'                              Must be positive.
#' @param   max.cds         maximum number of coordinate descents per BASIL iteration.
#'                          Must be positive.
#' @param   thr             coordinate descent convergence threshold.
#' @param   min.ratio       a factor times the maximum lambda value defines the smallest lambda value.
#'                          This is only used if user.lambdas is empty.
#' @param   n.threads       number of OpenMP threads for KKT check. 
#'                          Set it to 0 (default) to disable OpenMP usage.
#'                          Set it to -1 to use all available logical cores.
#'                          Note that if this value is too high, performance may worsen.
#'                          A general rule of thumb is to use the number of physical (not logical) cores.
#' @export
ghostbasil <- function(A, r, s, 
                      user.lambdas=c(), 
                      max.lambdas=100,
                      lambdas.iter=10, 
                      strong.size=100, 
                      delta.strong.size=500, 
                      max.strong.size=10000, 
                      max.cds=100000, 
                      thr=1e-7,
                      min.ratio=1e-6,
                      n.threads=0)
{
    # proper casting of inputs
    user.lambdas <- as.numeric(user.lambdas)
    max.lambas <- as.integer(max.lambdas)
    lambdas.iter <- as.integer(lambdas.iter)
    strong.size <- as.integer(strong.size)
    delta.strong.size <- as.integer(delta.strong.size)
    max.strong.size <- as.integer(max.strong.size)
    max.cds <- as.integer(max.cds)
    thr <- as.numeric(thr)
    min.ratio <- as.numeric(min.ratio)
    n.threads <- as.integer(n.threads)

    # input checking
    if (length(user.lambdas) != 0) {
        user.lambdas <- sort(user.lambdas, decreasing=T)
    }
    if ((length(user.lambdas) <= 0) & 
        (max.lambdas <= 0)) {
        stop("Maximum number of lambdas must be greater than 0 if user.lambdas is empty.")
    }
    if (lambdas.iter <= 0) {
        stop("Number of lambdas per BASIL iteration must be greater than 0.")
    }
    if (strong.size <= 0) {
        stop("Initial size of strong set must be greater than 0.")
    }
    if (delta.strong.size <= 0) {
        stop("Number of lambdas to add per BASIL iteration must be greater than 0.")
    }
    if (max.strong.size <= 0) {
        stop("Maximum strong set size must be greater than 0.")
    }
    if (max.cds <= 0) {
        stop("Maximum number of coordinate descents per BASIL iteration must be greater than 0.")
    }
    if (n.threads < -1) {
        stop("Number of threads must be at least -1.")
    }

    # choose the C++ routine
    basil_cpp <- NA
    if (any(class(A) == 'matrix')) {
        basil_cpp <- basil_dense__
    }
    else if (any(class(A) == 'Rcpp_BlockMatrix__')) {
        basil_cpp <- basil_block_dense__
    }
    else if (any(class(A) == 'Rcpp_GhostMatrix__')) {
        basil_cpp <- basil_ghost__
    }
    else if (any(class(A) == 'Rcpp_BlockGhostMatrix__')) {
        basil_cpp <- basil_block_ghost__
    }
    else {
        stop("Unrecognized type of A.")
    }
    out <- basil_cpp(
                    A=A,
                    r=r,
                    s=s,
                    user_lmdas=user.lambdas,
                    max_n_lambdas=max.lambdas,
                    n_lambdas_iter=lambdas.iter,
                    strong_size=strong.size, 
                    delta_strong_size=delta.strong.size, 
                    max_strong_size=max.strong.size,
                    max_n_cds=max.cds, 
                    thr=thr,
                    min_ratio=min.ratio,
                    n_threads=n.threads
                    )

    # raise any warnings
    if (out$error != "") warning(out$error)
    
    out
}
