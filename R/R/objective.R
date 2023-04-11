#' Computes the PGR objective that is minimized by ghostbasil.
#'
#' The objective is given by:
#' \deqn{
#'  \frac{1}{2} \beta^\top A \beta - \beta^\top r 
#'      + \lambda \sum\limits_i p_i \left(\alpha |\beta_i| + \frac{1-\alpha}{2} \beta_i^2 \right)
#' }
#'
#' @param   A   covariance matrix. Currently, only supports dense and sparse matrices.
#' @param   r   correlation vector.
#' @param   penalty penalty vector p.
#' @param   alpha   elastic net proportion.
#' @param   lmda    lambda regularization value.
#' @param   beta    vector of coefficients
#' @export
objective <- function(A, r, penalty, alpha, lmda, beta)
{
    out <- NA
    if (any(class(beta) == 'dgCMatrix')) {
        out <- objective_sparse__(A, r, penalty, alpha, lmda, beta)
    } else {
        beta <- as.numeric(beta)
        out <- objective_dense__(A, r, penalty, alpha, lmda, beta)
    }
    if (out$error != "") warning(out$error)
    out$objective
}
