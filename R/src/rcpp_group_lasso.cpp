#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/group_lasso.hpp>

using namespace Rcpp;

List solve_sub_coeffs_exp(
    const Eigen::Map<Eigen::MatrixXd> C,
    const Eigen::Map<Eigen::VectorXd> y,
    double lmda,
    double step_size,
    const Eigen::Map<Eigen::VectorXd> x,
    size_t max_iters=1000,
    double tol=1e-8
)
{
    Eigen::VectorXd x_sol = x;
    ghostbasil::solve_sub_coeffs(
        C, y, lmda, step_size, x_sol, max_iters, tol
    );
    return List::create(
        Named("beta", x_sol),
        Named("iters", iters)
    );
}