#' Computes the objective that is minimized by ghostbasil
#' \deqn{
#'  \frac{1-s}{2} \beta^\top A \beta - \beta^\top r + \frac{s}{2} ||\beta||_2 + \lambda ||\beta||_1
#' }
#' @param   A   covariance matrix. Currently, only supports dense matrix.
#' @param   r   covariance between covariates and response.
#' @param   s   regularization strength of A towards identity.
#' @param   lmda    lambda regularization value
#' @param   beta    vector of values.
#' @export
objective <- function(A, r, s, lmda, beta)
{
    out <- NA
    if (any(class(beta) == 'dgCMatrix')) {
        out <- objective_sparse__(A, r, s, lmda, beta)
    } else {
        beta <- as.numeric(beta)
        out <- objective_dense__(A, r, s, lmda, beta)
    }
    if (out$error != "") warning(out$error)
    out$objective
}
