#' @useDynLib ghostbasil, .registration = TRUE
#' @import Rcpp 
#' @import Matrix
NULL
Rcpp::loadModule("core_module", T)
