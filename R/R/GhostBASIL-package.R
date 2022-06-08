#' @useDynLib ghostbasil, .registration = TRUE
#' @import Rcpp 
#' @import Matrix
#' @importFrom "methods" new
NULL
Rcpp::loadModule("core_module", T)
