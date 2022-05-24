#' Fits PGR objective with BASIL framework.
#' @param   A   data covariance matrix (currently must be a dense matrix).
#' @param   y   response vector.
#' @param   s   regularization to shrink A towards identity ((1-s) * A + s * I).
#' @param   user.lambdas    user-specified sequence of lambdas. Will be sorted in decreasing order if not sorted already.
#' @param   max.lambdas     maximum number of lambdas to compute solutions for. 
#' @param   lambdas.iter    number of lambdas to compute strong set solutions for at each iteration of the BASIL algorithm.
#'                          Internally, capped at max.lambdas.
#' @param   strong.size     initial number of strong set variables to include.
#'                          Internally, capped at max.strong.size.
#' @param   delta.strong.size   number of strong set variables to include at every iteration of the BASIL algorithm.
#'                              Internally, capped at number of non-strong variables at every iteration of BASIL.
#' @param   max.strong.size     maximum number of strong set size. 
#'                              Internally, capped at number of features.
#' @param   max.cds         maximum number of coordinate descents per BASIL iteration.
#' @param   thr             coordinate descent convergence threshold.
#' @export
ghostbasil <- function(A, y, s, 
                      user.lambdas=c(), 
                      max.lambdas=100,
                      lambdas.iter=10, 
                      strong.size=1000, 
                      delta.strong.size=100, 
                      max.strong.size=10000, 
                      max.cds=100000, 
                      thr=1e-7,
                      n.threads=-1) 
{
    user.lambdas <- as.numeric(user.lambdas)
    if (length(user.lambdas) != 0) {
        user.lambdas <- sort(user.lambdas, decreasing=T)
    }

    out <- fit_basil__(
                A=A,
                y=y,
                s=s,
                user_lmdas=user.lambdas,
                max_n_lambdas=max.lambdas,
                n_lambdas_iter=lambdas.iter,
                strong_size=strong.size, 
                delta_strong_size=delta.strong.size, 
                max_strong_size=max.strong.size,
                max_n_cds=max.cds, 
                thr=thr,
                n_threads=n.threads)

    if (length(out$betas) == 0) return(out)

    betas <- Matrix(nrow=nrow(out$betas[[1]]), ncol=0, sparse=T)
    for (b in out$betas) { betas <- cbind(betas, b) }
    lmdas <- c()
    for (l in out$lmdas) { lmdas <- c(lmdas, l) }

    if (out$error != "") {
        warning(out$error)
    }
    
    list(betas=betas, lmdas=lmdas, error=out$error)
}
