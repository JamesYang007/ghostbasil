#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/group_lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
std::vector<double> solve_quartic__(
    double a,
    double b,
    double c,
    double d,
    double e
)
{
    std::vector<double> x(4);
    ghostbasil::solve_quartic(a, b, c, d, e, x.data());    
    return x;
}

// [[Rcpp::export]]
double solve_sub_coord_desc__(
    double a,
    double b,
    double c,
    double d
)
{
    return ghostbasil::solve_sub_coord_desc(a, b, c, d);    
}

// [[Rcpp::export]]
List solve_sub_coeffs__(
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
    size_t iters = 0;
    ghostbasil::solve_sub_coeffs(
        C, y, lmda, step_size, x_sol, iters, max_iters, tol
    );
    return List::create(
        Named("beta", x_sol),
        Named("iters", iters)
    );
}

// [[Rcpp::export]]
List solve_sub_coeffs_mix__(
    const Eigen::Map<Eigen::MatrixXd> C,
    const Eigen::Map<Eigen::VectorXd> y,
    double lmda,
    double step_size,
    const Eigen::Map<Eigen::VectorXd> x,
    size_t max_cd_iters=100,
    double cd_tol=1e-6,
    size_t max_iters=1000,
    double tol=1e-8
)
{
    Eigen::VectorXd x_sol = x;
    size_t iters = 0;
    ghostbasil::solve_sub_coeffs_mix(
        C, y, x_sol, lmda, max_cd_iters, cd_tol,
        step_size, iters, max_iters, tol
    );
    return List::create(
        Named("beta", x_sol),
        Named("iters", iters)
    );
}