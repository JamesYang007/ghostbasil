#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/group_lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
List update_group_coeffs__(
    const Eigen::Map<Eigen::VectorXd> L,
    const Eigen::Map<Eigen::VectorXd> v,
    double lmda,
    double s,
    double tol=1e-8,
    size_t max_iters=1000
)
{
    Eigen::VectorXd x_sol;
    size_t iters = 0;
    ghostbasil::GroupLasso<double> gl;
    gl.update_group_coefficients__(
        L, v, lmda, s, tol, max_iters, x_sol, iters
    );
    return List::create(
        Named("beta", x_sol),
        Named("iters", iters)
    );
}
