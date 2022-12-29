#' Fits PGR objective with BASIL framework.
#'
#' @param   A   data covariance matrix.
#'              It must be one of four types: matrix, GhostMatrix__, BlockMatrix__, BlockGhostMatrix__.
#'              For types of the form xxx__, see the function xxx.
#' @param   r   correlation vector.
#' @param   alpha   elastic net proportion.
#' @param   penalty         penalty factor for each coefficient.
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
#' @param   use.strong.rule     uses the strong rule with minimal dependence on fixed incremental rule if TRUE. 
#'                              Note that delta.strong.size is still used in tandem with strong rule.
#'                              It is recommended to keep delta.strong.size small if strong rule is used.
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
#' @param   checkpoint      A checkpoint object returned by a previous call to ghostbasil.
#'                          It is useful to pass the checkpoint when one would like to continue
#'                          fitting down a path of lambdas from the last lambda.
#' @export
ghostbasil <- function(A, r, alpha=1,
                      penalty=rep(1, times=length(r)),
                      user.lambdas=c(), 
                      max.lambdas=100,
                      lambdas.iter=5, 
                      use.strong.rule=T,
                      delta.strong.size=10, 
                      max.strong.size=10000, 
                      max.cds=100000, 
                      thr=1e-7,
                      min.ratio=1e-2,
                      n.threads=-1,
                      checkpoint=list())
{
    # proper casting of inputs
    penalty <- as.numeric(penalty)
    user.lambdas <- as.numeric(user.lambdas)
    max.lambdas <- as.integer(max.lambdas)
    lambdas.iter <- as.integer(lambdas.iter)
    use.strong.rule <- as.logical(use.strong.rule)
    delta.strong.size <- as.integer(delta.strong.size)
    max.strong.size <- as.integer(max.strong.size)
    max.cds <- as.integer(max.cds)
    thr <- as.numeric(thr)
    min.ratio <- as.numeric(min.ratio)
    n.threads <- as.integer(n.threads)

    # input checking
    if ((alpha < 0) | (alpha > 1)) {
        stop("alpha must be in [0,1].")
    }
    if (length(penalty) != length(r)) {
        stop("Penalty length must be same as that of r.")
    }
    if (sum(penalty < 0) > 0) {
        stop("Penalty must all be non-negative.") 
    }
    if (sum(penalty == 0) == length(penalty)) {
        stop("Penalty cannot be all 0.")
    }
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

    # rescale penalty to sum to nvars
    penalty <- penalty * (length(penalty) / sum(penalty))

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
                    alpha=alpha,
                    penalty=penalty,
                    user_lmdas=user.lambdas,
                    max_n_lambdas=max.lambdas,
                    n_lambdas_iter=lambdas.iter,
                    use_strong_rule=use.strong.rule,
                    delta_strong_size=delta.strong.size, 
                    max_strong_size=max.strong.size,
                    max_n_cds=max.cds, 
                    thr=thr,
                    min_ratio=min.ratio,
                    n_threads=n.threads,
                    checkpoint=checkpoint
                    )

    # raise any warnings
    if (out$error != "") warning(out$error)
    
    out
}
