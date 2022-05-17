#' Fits PGR objective with BASIL framework.
#' @param   A   data covariance matrix (currently must be a dense matrix).
#' @param   y   response vector.
#' @param   s   regularization to shrink A towards identity ((1-s) * A + s * I).
#' @param   n_knockoffs     number of knockoffs used to construct A and y (default: 0).
#' @param   n_lambdas       number of lambdas to compute solutions for. Currently there is no early-stopping rule.
#' @param   n_lambdas_iter  maximum number of lambdas to compute strong set solutions for at each iteration of the BASIL algorithm.
#' @param   strong_size     maximum number of strong set variables to append at each iteration of the BASIL algorithm.
#' @param   delta_strong_size   number of strong set variables to increase strong_size at KKT failure of first lambda.
#' @param   n_iters         maximum number of iterations of BASIL. 
#' @param   max_cds         maximum number of total coordinate descents.
#' @param   thr             convergence threshold.
#' @export
fit_basil <- function(A, y, s, n_knockoffs=0, n_lambdas=100, 
                      n_lambdas_iter=10, strong_size=1000, 
                      delta_strong_size=100, n_iters=100, max_cds=1000, thr=1e-14) 
{
    out <- fit_basil__(
                A,y,s,n_knockoffs,n_lambdas,n_lambdas_iter,
                strong_size, delta_strong_size, n_iters,
                max_cds, thr)
    if (length(out$betas) == 0) return(out)

    betas <- Matrix(nrow=nrow(out$betas[[1]]), ncol=0, sparse=T)
    for (b in out$betas) { betas <- cbind(betas, b) }
    lmdas <- c()
    for (l in out$lmdas) { lmdas <- c(lmdas, l) }
    
    list(betas=betas, lmdas=lmdas)
}
